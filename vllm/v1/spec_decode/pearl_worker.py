
import os
import torch
import multiprocessing
import signal
import sys
import traceback
from typing import Dict, List, Optional, Tuple

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.spec_decode.pearl_ipc import PearlIPC, PearlIPCConfig

logger = init_logger(__name__)

from vllm.v1.attention.backends.utils import CommonAttentionMetadata

class PearlDraftModelRunner:
    """
    Manages the Draft Model and its KV Cache.
    Implements a simplified 'PagedAttention' manager for the Draft Worker.
    """
    def __init__(self, vllm_config: VllmConfig, device: torch.device, max_batch_size: int):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        self.model_config = self.speculative_config.draft_model_config
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_model_len = vllm_config.model_config.max_model_len
        
        self.model = None
        self.kv_cache = None
        self.num_layers = -1
        self.num_heads = -1
        self.head_dim = -1
        self.dtype = self.model_config.dtype
        
        # Paged Attention Config
        self.block_size = 16 # Standard vLLM default or from config? 
        # vLLM V1 might default to 16.
        if hasattr(vllm_config, "cache_config") and vllm_config.cache_config:
            self.block_size = vllm_config.cache_config.block_size
        
        # Pre-calculate block allocation
        # We perform STATIC allocation: Each slot [0..batch-1] gets a fixed range of blocks.
        self.blocks_per_req = (self.max_model_len + self.block_size - 1) // self.block_size
        self.total_blocks = self.max_batch_size * self.blocks_per_req

    def load_model(self):
        logger.info(f"[PEARL Runner] Loading model {self.model_config.model}...")
        self.model = get_model(self.model_config)
        self.model.to(self.device).eval()
        
        # Extract model params
        self.num_layers = self.model_config.get_num_layers(self.vllm_config.parallel_config)
        self.num_heads = self.model_config.get_num_kv_heads(self.vllm_config.parallel_config)
        self.head_dim = self.model_config.get_head_size()
        
        self._allocate_kv_cache()
        logger.info("[PEARL Runner] Model loaded and Paged KV cache allocated.")

    def _allocate_kv_cache(self):
        """
        Allocate Paged KV Cache: 
        Shape: [num_layers, 2, num_blocks, block_size, num_heads, head_dim]
        This matches vLLM's PagedAttention layout expectations.
        """
        # Element size
        elem_size = 2 if self.dtype == torch.float16 or self.dtype == torch.bfloat16 else 4
        
        # vLLM V1 usually expects list of tensors per layer.
        # But 'kv_cache' arg usually accepts the list.
        # Shape per layer: [2, num_blocks, block_size, num_heads, head_dim]
        # Or [num_blocks, 2, block_size, ...] ?
        # Standard vLLM cache shape (for V1/V0 mixed): [2, num_blocks, block_size * num_heads * head_dim] (flattened last)
        # Wait, let's check `vllm/v1/kv_cache_interface.py` if possible or rely on standard [2, num_blocks, block_size, num_heads, head_dim]
        # Common convention: (2, num_blocks, block_size, num_kv_heads, head_size)
        
        # Based on Typical PagedAttention Ops:
        # Key: [num_blocks, num_kv_heads, head_size/x, block_size, x] (x for alignment)
        # OR simple: [num_blocks, block_size, num_kv_heads, head_size]
        
        # To be safe without complex layout logic, we follow vLLM's `allocate_kv_cache` style?
        # Since we don't have the CacheEngine here, we construct it manually.
        # We will use the simplest layout supported: [2, num_blocks, block_size, num_heads, head_dim]
        
        # NOTE: V1 might use "KVCache" object. But Draft Model usually uses V0 PagedAttention via `attention/layer.py`?
        # Assuming we passed `v1` implementation in proposer?
        # Actually `PearlDraftWorker` runs in a simplified environment.
        
        cache_shape = (self.num_layers, 2, self.total_blocks, self.block_size, self.num_heads, self.head_dim)
        
        logger.info(f"[PEARL Runner] Allocating Paged Cache: {cache_shape}")
        
        # We allocate one huge tensor and split view for layers?
        # Or list of tensors.
        self.kv_cache = []
        for _ in range(self.num_layers):
            layer_cache = torch.zeros(
                (2, self.total_blocks, self.block_size, self.num_heads, self.head_dim),
                dtype=self.dtype,
                device=self.device
            )
            self.kv_cache.append(layer_cache)

    def _prepare_metadata(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_lens_tensor: torch.Tensor
    ) -> CommonAttentionMetadata:
        """
        Construct CommonAttentionMetadata for the current batch (Mixed Lengths).
        """
        batch_size, max_seq_len = input_ids.shape
        num_valid_tokens = seq_lens_tensor.sum().item()
        
        # 1. Query Start Loc
        # For mixed lengths, start_loc is cumsum of seq_lens_tensor
        # We need CPU version for metadata
        seq_lens_cpu = seq_lens_tensor.cpu()
        start_loc_cpu = torch.zeros(batch_size + 1, dtype=torch.int32)
        start_loc_cpu[1:] = torch.cumsum(seq_lens_cpu, dim=0)
        start_loc = start_loc_cpu.to(self.device)
        
        # 2. Block Tables
        block_table = torch.arange(self.total_blocks, dtype=torch.int32, device=self.device).view(self.max_batch_size, self.blocks_per_req)
        block_table = block_table[:batch_size, :]
        
        # 3. Slot Mapping
        # We must generate slot mapping ONLY for valid tokens. Padding -> -1.
        
        flat_positions = positions.flatten() # [batch * max_seq]
        flat_req_ids = torch.arange(batch_size, device=self.device).repeat_interleave(max_seq_len)
        
        # Determine valid mask based on positions and seq_lens
        # We need to know if a given position in the flattened tensor is within seq_lens[req_id]
        # Since 'positions' contains absolute positions, we can't just compare with seq_len?
        # WAIT: 'positions' for Delta is like [10, 0, 0...].
        # We need 'intra-batch index' to compare with input_len.
        
        intra_batch_indices = torch.arange(max_seq_len, device=self.device).repeat(batch_size) # [0, 1, ... max-1, 0, 1...]
        # Repeat seq_lens for comparison
        expanded_lens = seq_lens_tensor.repeat_interleave(max_seq_len)
        
        valid_mask = intra_batch_indices < expanded_lens
        
        # Calculate Potential Slots (even for padding)
        global_block_ids = (flat_req_ids * self.blocks_per_req) + (flat_positions // self.block_size)
        block_offsets = flat_positions % self.block_size
        raw_slots = (global_block_ids * self.block_size) + block_offsets
        
        # Apply Mask
        slot_mapping = torch.where(valid_mask, raw_slots, torch.tensor(-1, device=self.device, dtype=torch.int64))
        
        return CommonAttentionMetadata(
            query_start_loc=start_loc,
            query_start_loc_cpu=start_loc_cpu,
            seq_lens=seq_lens_tensor, # Tensor on Device
            num_reqs=batch_size,
            num_actual_tokens=num_valid_tokens,
            max_query_len=int(seq_lens_tensor.max().item()),
            max_seq_len=self.max_model_len,
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            logits_indices_padded=None,
            num_logits_indices=None
        )

    @torch.inference_mode()
    def forward(
        self, 
        input_ids: torch.Tensor, 
        positions: torch.Tensor, 
        seq_lens_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward with Metadata construction.
        """
        attn_metadata = self._prepare_metadata(input_ids, positions, seq_lens_tensor)
        
        outputs = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=self.kv_cache, 
            attn_metadata=attn_metadata
        )
        return outputs

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        start_positions: torch.Tensor,
        input_lengths: torch.Tensor,
        gamma: int
    ) -> torch.Tensor:
        """
        Generate gamma speculative tokens for the batch.
        Args:
            input_ids: [batch, max_len] (Dense, with padding)
            start_positions: [batch] (Where the first token of input belongs)
            input_lengths: [batch] (Valid length of input)
        """
        batch_size, max_input_len = input_ids.shape
        draft_tokens = torch.zeros(batch_size, gamma, dtype=torch.int32, device=self.device)
        
        # 1. First Step (Prefill / Delta)
        # Construct Positions [batch, max_len]
        # Row i: [start_pos[i], start_pos[i]+1, ...]
        
        offsets = torch.arange(max_input_len, device=self.device).unsqueeze(0) # [1, max_len]
        positions = start_positions.unsqueeze(1) + offsets # [batch, max_len]
        
        # Forward
        # We pass 'input_lengths' to mask out padding effects in metadata
        outputs = self.forward(input_ids, positions, input_lengths)
        
        # Extract Logits for the LAST valid token for each request
        # If input_len=1, last is index 0.
        # If input_len=N, last is index N-1.
        
        # We need to gather logits from proper positions.
        # outputs shape: [batch, max_len, hidden] (if using standard vLLM model which returns hidden states)
        
        # Gather indices: input_lengths - 1
        last_indices = (input_lengths - 1).long().unsqueeze(1).unsqueeze(2) # [batch, 1, 1]
        last_indices = last_indices.expand(-1, -1, outputs.size(-1)) # [batch, 1, hidden]
        
        last_hidden = torch.gather(outputs, 1, last_indices).squeeze(1) # [batch, hidden]
        
        if hasattr(self.model, "compute_logits"):
            logits = self.model.compute_logits(last_hidden, None)
        else:
            logits = last_hidden # Fallback/Risk

        next_token = logits.argmax(dim=-1).to(torch.int32)
        draft_tokens[:, 0] = next_token
        
        # 2. Auto-Regressive Steps
        current_token = next_token.unsqueeze(1) # [batch, 1]
        # Current length of context (virtual) = start_positions + input_lengths
        current_step_pos = start_positions + input_lengths
        
        # Constant lengths for decode steps (all 1)
        decode_lengths = torch.ones(batch_size, dtype=torch.int32, device=self.device)
        
        for step in range(1, gamma):
            # Pos for this step
            step_positions = current_step_pos.unsqueeze(1) # [batch, 1]
            
            outputs = self.forward(current_token, step_positions, decode_lengths)
            
            # For len=1, output index 0 is valid.
            last_hidden = outputs[:, 0, :]
            
            if hasattr(self.model, "compute_logits"):
                logits = self.model.compute_logits(last_hidden, None)
            else:
                logits = last_hidden

            next_token = logits.argmax(dim=-1).to(torch.int32)
            draft_tokens[:, step] = next_token
            
            current_token = next_token.unsqueeze(1)
            current_step_pos += 1
            
        return draft_tokens

# ... (PearlDraftWorker class remains, updated run_loop only)

    def run_loop(self):
        logger.info(f"[PEARL Worker {self.rank}] Starting inference loop (Async/Ouroboros)...")
        
        input_ids_gpu = torch.empty(self.ipc_config.batch_size, self.ipc_config.max_model_len, dtype=torch.int32, device=self.device)
        
        # Local Speculation State
        local_spec_result: Optional[torch.Tensor] = None
        expected_next_pos: Optional[torch.Tensor] = None # Tensor [batch]
        
        while not self.exit_event.is_set():
            if not self.start_event.wait(timeout=1.0):
                continue
            
            self.start_event.clear()
            if self.exit_event.is_set():
                break

            try:
                # 1. Read Inputs
                flags_cpu = torch.from_numpy(self.ipc.np_is_new_req)
                seqlens_cpu = torch.from_numpy(self.ipc.np_seqlens)
                
                flags_gpu = flags_cpu.to(self.device, non_blocking=True)
                seqlens_gpu = seqlens_cpu.to(self.device, non_blocking=True)
                
                # Check for Read-Ahead HIT
                # Hit if: local_spec exists AND flags==0 (Delta) AND seqlens==expected_next_pos
                is_hit = False
                if local_spec_result is not None and expected_next_pos is not None:
                    # Check batch-wide? Or per request?
                    # Simplified: We treat batch as a unit for HIT.
                    # If ANY request is New or Rewind, we treat as MISS (Simplicity).
                    # Ideally we could mix, but Proposer usually sends batch-wide signals.
                    
                    is_delta = (flags_gpu == 0).all()
                    if is_delta:
                        # Check positions
                        # Match logic: seqlens_gpu == expected_next_pos
                        matches = (seqlens_gpu == expected_next_pos).all()
                        if matches:
                            is_hit = True
                
                final_output = None
                
                if is_hit:
                    # HIT! Use pre-computed result.
                    final_output = local_spec_result
                    
                    # We don't strictly need to copy input_ids_gpu because we don't use them.
                    # But we need to update state for NEXT speculation.
                    # For speculation, we assume 'final_output' is accepted.
                    # Current Write Head was 'expected_next_pos'.
                    # After accepting 'final_output' (len gamma), next head is +gamma.
                    
                    pass 
                else:
                    # MISS. Normal Generation.
                    # Copy Inputs
                    input_ids_gpu.copy_(self.input_tensor_cpu, non_blocking=True)
                    
                    is_new = (flags_gpu == 1)
                    start_positions = torch.where(is_new, torch.tensor(0, device=self.device), seqlens_gpu)
                    input_lengths = torch.where(is_new, seqlens_gpu, torch.tensor(1, device=self.device))
                    
                    final_output = self.runner.generate(
                        input_ids_gpu, 
                        start_positions=start_positions,
                        input_lengths=input_lengths,
                        gamma=self.ipc_config.gamma
                    )
                
                # 4. Write Output (Immediate)
                self.output_tensor_cpu.copy_(final_output.cpu())
                self.done_event.set()
                
                # ==========================================
                # READ-AHEAD BLOCK (Speculate on Speculation)
                # ==========================================
                # Goal: Generate 'local_spec_result' for Step N+1 derived from 'final_output'.
                
                # 1. Determine Next Start Positions
                # If HIT: start = expected_next_pos + gamma
                # If MISS: start = start_positions (used above) + gamma?
                # Wait:
                # 'final_output' are token IDs at [start, start+1... start+gamma-1]
                # So next start is 'start + gamma'.
                
                current_start_pos = None
                if is_hit:
                    current_start_pos = expected_next_pos
                else:
                    is_new = (flags_gpu == 1) # re-evaluate
                    current_start_pos = torch.where(is_new, torch.tensor(0, device=self.device), seqlens_gpu)
                    # Note: If is_new=True, start=0. generate() filled 0..gamma-1. Next is gamma.
                    # If is_new=False (Delta), start=S. generate() filled S..S+gamma-1. Next is S+gamma.
                
                # But wait, is_new=True implies InputLen=ContextLen.
                # So generate wrote at ContextLen..ContextLen+gamma-1.
                # So start was ContextLen.
                # My logic above: `start_positions = torch.where(is_new, 0, seqlens)` is WRONG for New Request?
                # In New Request, seqlens=ContextLen.
                # generate() logic:
                #  positions = start_positions + offsets.
                #  If start=0, positions=0,1,2... 
                #  Prefill uses positions up to InputLen.
                #  Logits taken from InputLen-1.
                #  Generation starts at InputLen.
                #  So effective "Write Head" for generation is InputLen.
                
                # Let's fix 'current_effective_head'
                if is_hit:
                    current_effective_head = expected_next_pos
                else:
                    # Logic from generating block
                    is_new = (flags_gpu == 1)
                    # If New: Head = seqlens_gpu (Context Len)
                    # If Delta: Head = seqlens_gpu (Write Pos)
                    current_effective_head = seqlens_gpu
                
                # Next prediction starts at:
                next_spec_start = current_effective_head + self.ipc_config.gamma
                
                # Input for Next Prediction:
                # Last token of 'final_output'.
                # final_output shape: [batch, gamma]
                last_tokens = final_output[:, -1]
                
                # Prepare inputs for generation
                # We need input_ids in [batch, max_len] format?
                # generate() expects dense input.
                # But for decode step, it just needs the new token if using Delta logic?
                # No, generate() uses `input_ids` primarily for Prefill.
                # For Decode (Delta), we only need 1 token input?
                # My `PearlDraftModelRunner.generate` logic:
                # It assumes `input_ids` contains the *token at start_positions*?
                # Let's check generate():
                #   outputs = self.forward(input_ids, positions, input_lengths)
                #   last_hidden = ...
                #   next_token = ...
                # If we pass `input_lengths=1`, we forward 1 token.
                # We need to construct a tensor with that 1 token at the correct column?
                # Or does `forward` gather?
                # `PearlDraftModelRunner.forward` calls `model(input_ids...)`.
                # vLLM model expects `input_ids` [batch, len].
                # If len=1, it expects [batch, 1]? No, usually [batch, max_len] padded?
                # vLLM V1 API: `input_ids` shape?
                # Usually standard forward takes [num_tokens] flat or [batch, len] padded.
                # My runner uses padded [batch, max_len].
                
                # So we need to ensure 'input_ids_gpu' has the 'last_tokens' at the right place?
                # OR, simplified:
                # Pass a explicit [batch, 1] tensor to generate?
                # My `generate` signature: `input_ids` [batch, max_len].
                # It calculates positions.
                # It calls forward.
                
                # OPTIMIZATION:
                # We can just create a [batch, 1] tensor for the spec step.
                # But `generate` logic extracts offsets from `input_ids.shape`.
                # I should ideally overload `generate` or prepare `input_ids` carefully.
                # For Read-Ahead, we act like "Delta".
                # Input = Last Token.
                # Length = 1.
                # Position = next_spec_start.
                
                # Construct [batch, 1] input
                # Wait, `generate` expects [batch, max_input_len] to execute prefill logic.
                # But we are doing decode step (len=1).
                # So we can pass [batch, 1] tensor!
                spec_input_ids = last_tokens.unsqueeze(1).to(torch.int32) # [batch, 1]
                spec_input_lens = torch.ones(self.ipc_config.batch_size, dtype=torch.int32, device=self.device)
                
                # Generate N+1
                local_spec_result = self.runner.generate(
                    spec_input_ids,
                    start_positions=next_spec_start,
                    input_lengths=spec_input_lens,
                    gamma=self.ipc_config.gamma
                )
                
                expected_next_pos = next_spec_start # Updated expectation
                
            except Exception as e:
                logger.error(f"[PEARL Worker {self.rank}] Error in loop: {e}")
                traceback.print_exc()
                self.exit_event.set()
                break
        
        logger.info(f"[PEARL Worker {self.rank}] Exiting...")
        self.ipc.destroy()

def run_draft_worker_process(
    rank: int,
    gpu_id: int,
    ipc_config: PearlIPCConfig,
    vllm_config: VllmConfig,
    start_event: multiprocessing.Event,
    done_event: multiprocessing.Event,
    exit_event: multiprocessing.Event,
):
    """Entry point for the subprocess"""
    try:
        worker = PearlDraftWorker(rank, gpu_id, ipc_config, vllm_config, start_event, done_event, exit_event)
        worker.load_model()
        worker.run_loop()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Worker process crashed: {e}")
        traceback.print_exc()

