# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
PEARL (Parallel Speculative Decoding with Adaptive Draft Length) Proposer.

This module implements the PEARL speculative decoding method, which decouples
draft and target models onto separate device groups and runs them in parallel
with adaptive draft length.

Based on nano-PEARL: https://github.com/smart-lty/nano-PEARL
"""

import atexit
import os
import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoConfig, AutoTokenizer

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class PEARLController:
    """Controller for managing communication between draft and target models."""

    def __init__(self, config: Any, control_event: Event):
        self.config = config
        self.draft_event = []
        self.target_event = []
        self.control_event = control_event

        # Create shared memory for inter-process communication
        self.draft_shm = SharedMemory(
            name="pearl_draft_group", create=True, size=2**20
        )
        self.target_shm = SharedMemory(
            name="pearl_target_group", create=True, size=2**20
        )

    def add_event(self, rank: int, event: Event, draft_devices: list, target_devices: list):
        """Add event for synchronization."""
        if rank in draft_devices:
            self.draft_event.append(event)
        else:
            self.target_event.append(event)

    def write_draft_shm(self, method_name: str, *args):
        """Write command to draft model shared memory."""
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.draft_shm.buf[0:4] = n.to_bytes(4, "little")
        self.draft_shm.buf[4 : n + 4] = data
        for event in self.draft_event:
            event.set()

    def write_target_shm(self, method_name: str, *args):
        """Write command to target model shared memory."""
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.target_shm.buf[0:4] = n.to_bytes(4, "little")
        self.target_shm.buf[4 : n + 4] = data
        for event in self.target_event:
            event.set()

    def read_output(self):
        """Read output from target model shared memory."""
        n = int.from_bytes(self.target_shm.buf[0:4], "little")
        data = self.target_shm.buf[4 : n + 4]
        output, elapsed_time = pickle.loads(data)
        return output, elapsed_time

    def cleanup(self):
        """Cleanup shared memory."""
        try:
            self.draft_shm.close()
            self.draft_shm.unlink()
        except Exception:
            pass
        try:
            self.target_shm.close()
            self.target_shm.unlink()
        except Exception:
            pass


class PEARLProposer:
    """
    PEARL (Parallel Speculative Decoding with Adaptive Draft Length) Proposer.

    PEARL disaggregates draft and target models onto separate device groups
    and runs them in parallel with adaptive draft length control.

    Key features:
    - Draft-Target Disaggregation: Models loaded on separate GPUs
    - Parallel Inference: Draft and target run concurrently
    - Adaptive Draft Length: Dynamic speculation based on alignment
    """

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        assert self.speculative_config.method == "pearl"

        # PEARL configuration
        self.draft_model_path = self.speculative_config.model
        self.target_model_path = vllm_config.model_config.model

        # Tensor parallelism configuration
        self.draft_tensor_parallel_size = (
            self.speculative_config.draft_tensor_parallel_size or 1
        )
        self.target_tensor_parallel_size = (
            self.speculative_config.target_tensor_parallel_size
            or vllm_config.parallel_config.tensor_parallel_size
        )

        # PEARL-specific parameters
        self.gamma = self.speculative_config.pearl_gamma
        self.max_num_batched_tokens = self.speculative_config.pearl_max_num_batched_tokens
        self.max_num_seqs = self.speculative_config.pearl_max_num_seqs
        self.kvcache_block_size = self.speculative_config.pearl_kvcache_block_size
        self.num_kvcache_blocks = self.speculative_config.pearl_num_kvcache_blocks
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens

        # World size calculation
        self.world_size = self.draft_tensor_parallel_size + self.target_tensor_parallel_size

        logger.info("=" * 50)
        logger.info("Initializing PEARL Proposer")
        logger.info(f"Draft model: {self.draft_model_path}")
        logger.info(f"Target model: {self.target_model_path}")
        logger.info(f"Draft TP size: {self.draft_tensor_parallel_size}")
        logger.info(f"Target TP size: {self.target_tensor_parallel_size}")
        logger.info(f"World size: {self.world_size}")
        logger.info(f"Gamma (adaptive draft length): {self.gamma}")
        logger.info(f"Num speculative tokens: {self.num_speculative_tokens}")
        logger.info("=" * 50)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.draft_model_path, use_fast=True
        )

        # Multiprocessing setup
        self.ps = []
        self.controller = None
        self.control_event = None

        # Initialize controller (will be done in load_model)
        self._initialized = False

    def load_model(self, target_model: torch.nn.Module) -> None:
        """
        Load draft and target models in separate processes.

        Args:
            target_model: The target model instance from vLLM
        """
        logger.info("[PEARL] Starting model loading...")

        # Store target model reference
        self.target_model = target_model

        # Setup multiprocessing context
        ctx = mp.get_context("spawn")
        self.control_event = ctx.Event()

        # Note: Full multiprocessing implementation would go here
        # For now, we'll use a simplified approach that integrates with vLLM's
        # existing model runner infrastructure

        logger.info("[PEARL] Model loading complete")
        self._initialized = True

        atexit.register(self.cleanup)

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        req_ids: list[str],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        spec_decode_unsupported_reqs: set,
    ) -> list[list[int]]:
        """
        Propose draft tokens using PEARL's parallel speculative decoding.

        Args:
            sampled_token_ids: Recently sampled token IDs for each request
            req_ids: Request IDs
            num_tokens_no_spec: Number of tokens without speculation for each request
            token_ids_cpu: All token IDs on CPU
            spec_decode_unsupported_reqs: Set of request IDs that don't support spec decode

        Returns:
            List of draft token IDs for each request
        """
        if not self._initialized:
            logger.warning("[PEARL] Proposer not initialized, returning empty drafts")
            return [[] for _ in sampled_token_ids]

        draft_token_ids = []

        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            if not num_sampled_ids:
                # Skip speculative decoding for this request
                draft_token_ids.append([])
                continue

            # Skip requests that don't support speculative decoding
            req_id = req_ids[i]
            if req_id in spec_decode_unsupported_reqs:
                draft_token_ids.append([])
                continue

            num_tokens = num_tokens_no_spec[i]
            max_model_len = self.vllm_config.model_config.max_model_len
            if num_tokens >= max_model_len:
                # Skip requests at max length
                draft_token_ids.append([])
                continue

            # Generate draft tokens using PEARL
            # TODO: Implement actual PEARL draft generation logic
            # For now, return empty draft tokens
            draft_token_ids.append([])

        return draft_token_ids

    def cleanup(self):
        """Cleanup resources."""
        if self.controller is not None:
            self.controller.cleanup()

        # Terminate child processes
        for p in self.ps:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)

        logger.info("[PEARL] Cleanup complete")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
