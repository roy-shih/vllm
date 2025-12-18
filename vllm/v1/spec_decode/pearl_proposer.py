# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
PEARL (Parallel Speculative Decoding with Adaptive Draft Length) Proposer.

Multi-Process Implementation for Physical Isolation:
- Main Process (Target Worker): Handles `propose` requests, writes to SHM.
- Draft Process (Draft Worker): Runs on separate GPU, reads SHM, infers, writes back.
"""

import torch
import torch.nn as nn
import multiprocessing
import atexit
import time
import numpy as np

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.pearl_ipc import PearlIPC, PearlIPCConfig
from vllm.v1.spec_decode.pearl_worker import run_draft_worker_process

logger = init_logger(__name__)


class PEARLProposer:
    """
    PEARL Proposer (Client-Side).
    Manages the Draft Worker process and handles IPC.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        assert self.speculative_config.method == "pearl"

        self.device = device # Target device
        self.runner = runner

        # PEARL configuration
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens
        self.gamma = self.speculative_config.pearl_gamma
        if self.gamma == -1:
            self.gamma = max(self.num_speculative_tokens, 5)

        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.batch_size_limit = vllm_config.scheduler_config.max_num_seqs

        # Draft Worker Configuration
        self.draft_gpu_id = self.speculative_config.pearl_draft_gpu_id
        self.disabled = False
        
        # IPC Setup
        self.ipc_config = PearlIPCConfig(
            batch_size=self.batch_size_limit,
            max_model_len=self.max_model_len,
            gamma=self.gamma,
            vocab_size=vllm_config.model_config.get_vocab_size(), # Approximate, worker loads real model
            shm_name_prefix=f"pearl_ipc_{int(time.time())}" # Unique prefix
        )
        
        # Create IPC (Owner)
        self.ipc = PearlIPC(self.ipc_config, create=True)
        
        # Events
        self.start_event = multiprocessing.Event()
        self.done_event = multiprocessing.Event()
        self.exit_event = multiprocessing.Event()

        # Process management
        self.worker_process = None

        # Statistics
        self.total_drafts = 0
        self.total_accepted = 0
        
        # Adaptive tracking
        self.adaptive_enabled = True # Always enable tracking for now
        from collections import deque
        self.acceptance_history = deque(maxlen=self.gamma)
        self.current_draft_tokens = self.num_speculative_tokens

        # Req ID tracking for Delta Updates
        # Map batch_index -> req_id_hash (or str)
        # We use a list to track the previous req_id at each batch slot.
        # State tracking
        self.last_req_ids = [None] * self.batch_size_limit
        self.last_expected_pos = np.zeros(self.batch_size_limit, dtype=np.int32)
        
        # Ouroboros Feature Flag
        self.read_ahead_enabled = True

        logger.info(f"[PEARL] Initializing Multi-Process Proposer. Target Device: {device}, Draft GPU: {self.draft_gpu_id}")
        
        # Register cleanup
        atexit.register(self.shutdown)



    def load_model(self, target_model: nn.Module) -> None:
        """
        Start the Draft Worker process instead of loading model locally.
        """
        try:
            logger.info("[PEARL] Spawning Draft Worker process...")
            
            ctx = multiprocessing.get_context('spawn')
            self.worker_process = ctx.Process(
                target=run_draft_worker_process,
                args=(
                    0, # Rank
                    self.draft_gpu_id, 
                    self.ipc_config,
                    self.vllm_config, 
                    self.start_event, 
                    self.done_event, 
                    self.exit_event
                )
            )
            self.worker_process.start()
            logger.info(f"[PEARL] Draft Worker started (PID: {self.worker_process.pid})")
            
            # Register for stats
            try:
                from vllm.v1.spec_decode.pearl_stats_hook import register_pearl_proposer
                register_pearl_proposer(self)
            except Exception:
                pass
                
        except Exception as e:
            logger.error(f"[PEARL] Failed to start worker: {e}. Disabling PEARL.")
            self.disabled = True

    @torch.inference_mode()
    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple | None = None,
    ) -> torch.Tensor:
        """
        Delegate proposal generation to Draft Worker via IPC.
        Implements Delta Update Logic using Request ID tracking.
        """
        batch_size = next_token_ids.shape[0]
        
        # Fast Fallback checks
        if self.disabled:
             return torch.zeros((batch_size, self.num_speculative_tokens), device=self.device, dtype=torch.int32)
             
        if self.worker_process is None or not self.worker_process.is_alive():
            logger.error("[PEARL] Worker process is dead! Disabling speculation.")
            self.disabled = True
            return torch.zeros((batch_size, self.num_speculative_tokens), device=self.device, dtype=torch.int32)

        # 1. Prepare Inputs for IPC (Delta Update Protocol)
        
        # Access Numpy Views
        ipc_inputs = self.ipc.np_inputs # [max_batch, max_len]
        ipc_seqlens = self.ipc.np_seqlens # [max_batch]
        ipc_flags = self.ipc.np_is_new_req # [max_batch]
        
        query_start_loc = common_attn_metadata.query_start_loc
        
        # Efficient access to CPU arrays
        all_tokens_cpu = target_token_ids.cpu().numpy()
        next_tokens_cpu = next_token_ids.cpu().numpy()
        starts = query_start_loc.cpu().numpy()
        
        model_max = self.max_model_len
        
        # Get Current Request IDs
        # We rely on self.runner being available and populated
        current_req_ids = []
        if self.runner and hasattr(self.runner, "input_batch"):
             # Assuming input_batch.req_ids matches the batch order
             current_req_ids = self.runner.input_batch.req_ids
        else:
            # Fallback (Danger): Treat all as new if we can't track
            logger.warning("[PEARL] No runner/req_ids found. Forcing full update.")
            current_req_ids = [f"unknown_{time.time()}_{i}" for i in range(batch_size)]

        # Default: optimistic HIT assumption if enabled
        all_match = self.read_ahead_enabled

        for i in range(batch_size):
            req_id = current_req_ids[i]
            is_new = (req_id != self.last_req_ids[i])
            self.last_req_ids[i] = req_id # Update history
            
            ipc_flags[i] = 1 if is_new else 0
            
            next_tok = next_tokens_cpu[i]
            write_pos = 0 # Computed below
            
            if is_new:
                all_match = False # New request breaks pipeline assumption
                
                s, e = starts[i], starts[i+1]
                ctx_len = e - s
                total_len = ctx_len + 1
                
                if total_len > model_max:
                    keep_len = model_max - 1
                    ipc_inputs[i, :keep_len] = all_tokens_cpu[e-keep_len:e]
                    ipc_inputs[i, keep_len] = next_tok
                    ipc_seqlens[i] = model_max
                    write_pos = model_max
                else:
                    ipc_inputs[i, :ctx_len] = all_tokens_cpu[s:e]
                    ipc_inputs[i, ctx_len] = next_tok
                    ipc_seqlens[i] = total_len
                    write_pos = total_len
            else:
                # Delta Update / Feedback
                s, e = starts[i], starts[i+1]
                ctx_len = e - s
                
                write_pos = ctx_len
                if write_pos >= model_max:
                     write_pos = model_max - 1
                
                ipc_inputs[i, 0] = next_tok
                ipc_seqlens[i] = write_pos # Overloaded: Write Position
                
                # Check prediction
                if write_pos != self.last_expected_pos[i]:
                    all_match = False

            # Update expected pos for *next* turn (assuming this turn runs)
            self.last_expected_pos[i] = write_pos + self.num_speculative_tokens

        
        # Zero out inactive slots
        if batch_size < self.batch_size_limit:
             ipc_seqlens[batch_size:] = 0

        # 3. Ouroboros Execution flow
        output_tensor = self.ipc.get_output_tensor() # CPU Tensor View
        result = None

        if all_match and self.done_event.is_set():
            # HIT! Zero Latency.
            # 1. Read the Speculative Output immediately.
            # 2. Signal Worker (Ack) to confirm our prediction. 
            
            # Read Output (from SHM to GPU)
            result = output_tensor[:batch_size, :self.current_draft_tokens].to(self.device, non_blocking=True)
            
            # Signal Worker to proceed (Ack)
            self.start_event.set()
        
        else:
            # MISS (Rewrite/Wait).
            
            # If done_event was set, it contains garbage (wrong speculation).
            # We must clear it because we are sending new instructions.
            if self.done_event.is_set():
                self.done_event.clear()
            
            # Signal Worker (New/Rewind)
            self.start_event.set()
            
            # Wait for Worker to finish CORRECT task
            if not self.done_event.wait(timeout=2.0): # 2s timeout
                 logger.error("[PEARL] Draft Worker Timed out! Disabled.")
                 self.disabled = True
                 return torch.zeros((batch_size, self.num_speculative_tokens), device=self.device, dtype=torch.int32)
            
            self.done_event.clear()
            
            # Read Output
            result = output_tensor[:batch_size, :self.current_draft_tokens].to(self.device, non_blocking=True)

        # Pad if needed implementation
        if self.current_draft_tokens < self.num_speculative_tokens:
            padding = torch.zeros((batch_size, self.num_speculative_tokens - self.current_draft_tokens), 
                                dtype=torch.int32, device=self.device)
            result = torch.cat([result, padding], dim=1)
            
        return result

    def update_acceptance_stats(self, num_accepted_tokens: int, num_draft_tokens: int):
        """
        Update local stats loop. 
        In future Phase 2, we send this to Worker for its internal adaptive logic.
        """
        self.total_accepted += num_accepted_tokens
        self.total_drafts += num_draft_tokens
        # Implement adaptive logic here if controlling K from main process
        # ... (logic same as before)

    def shutdown(self):
        logger.info("[PEARL] Shutting down Proposer...")
        self.exit_event.set()
        self.start_event.set() # Wake up worker if waiting
        
        if self.worker_process:
            self.worker_process.join(timeout=1)
            if self.worker_process.is_alive():
                self.worker_process.terminate()
        
        self.ipc.destroy()

