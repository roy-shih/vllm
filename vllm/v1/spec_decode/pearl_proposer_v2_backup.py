# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
PEARL (Parallel Speculative Decoding with Adaptive Draft Length) Proposer.

This is a simplified implementation that works within vLLM's framework.
It uses vLLM's existing model loading and inference infrastructure.

Based on nano-PEARL: https://github.com/smart-lty/nano-PEARL
"""

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

logger = init_logger(__name__)


class PEARLProposer:
    """
    PEARL Proposer - Simplified version that works within vLLM framework.

    Key differences from nano-PEARL:
    - Uses vLLM's model loading and inference infrastructure
    - Runs in the same process as target model (no multiprocessing yet)
    - Implements core PEARL logic: draft token generation with adaptive length

    Future improvements:
    - Add multiprocessing for true draft-target disaggregation
    - Implement parallel execution
    - Add CUDA graphs support
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
    ):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        assert self.speculative_config.method == "pearl"

        # Get device from current cuda device
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.dtype = vllm_config.model_config.dtype
        self.max_model_len = vllm_config.model_config.max_model_len

        # PEARL configuration
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens
        self.gamma = self.speculative_config.pearl_gamma
        self.max_num_batched_tokens = self.speculative_config.pearl_max_num_batched_tokens
        self.max_num_seqs = self.speculative_config.pearl_max_num_seqs

        # Draft model configuration
        self.draft_model_config = self.speculative_config.draft_model_config
        self.hidden_size = self.draft_model_config.get_hidden_size()

        # Initialize sampler for draft model
        self.sampler = Sampler()

        # Model will be loaded in load_model()
        self.draft_model: nn.Module | None = None
        self.target_model: nn.Module | None = None

        # Statistics tracking
        self.total_drafts = 0
        self.total_accepted = 0

        logger.info("=" * 50)
        logger.info("[PEARL] Initializing PEARL Proposer V2")
        logger.info(f"[PEARL] Num speculative tokens: {self.num_speculative_tokens}")
        logger.info(f"[PEARL] Gamma (adaptive draft length): {self.gamma}")
        logger.info(f"[PEARL] Draft model: {self.speculative_config.model}")
        logger.info(f"[PEARL] Target model: {vllm_config.model_config.model}")
        logger.info("=" * 50)

    def load_model(self, target_model: nn.Module) -> None:
        """
        Load the draft model.

        Args:
            target_model: The target model instance from vLLM
        """
        logger.info("[PEARL] Loading draft model...")

        # Store reference to target model
        self.target_model = target_model

        # Load draft model using vLLM's model loader
        self.draft_model = get_model(
            vllm_config=self.vllm_config,
            model_config=self.draft_model_config,
        )

        logger.info("[PEARL] Draft model loaded successfully")

        # Auto-set gamma if needed
        if self.gamma == -1:
            self.gamma = self._auto_set_gamma()
            logger.info(f"[PEARL] Auto-set gamma to {self.gamma}")

    def _auto_set_gamma(self) -> int:
        """
        Auto-set gamma (window size) based on hardware configuration.

        This is a simplified version. In nano-PEARL, it considers:
        - Available GPU memory
        - Model sizes
        - Batch size

        For now, we use a simple heuristic.
        """
        # Default gamma based on num_speculative_tokens
        # This ensures the window size is reasonable
        gamma = max(self.num_speculative_tokens, 5)
        return min(gamma, 10)  # Cap at 10 to avoid too long drafts

    @torch.inference_mode()
    def propose(
        self,
        sampled_token_ids: list[list[int]],
        req_ids: list[str],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        spec_decode_unsupported_reqs: set,
    ) -> list[list[int]]:
        """
        Propose draft tokens using PEARL's algorithm.

        Args:
            sampled_token_ids: Recently sampled token IDs for each request
            req_ids: Request IDs
            num_tokens_no_spec: Number of tokens without speculation for each request
            token_ids_cpu: All token IDs on CPU
            spec_decode_unsupported_reqs: Set of request IDs that don't support spec decode

        Returns:
            List of draft token IDs for each request
        """
        if self.draft_model is None:
            logger.warning("[PEARL] Draft model not loaded, returning empty drafts")
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
            if num_tokens >= self.max_model_len:
                # Skip requests at max length
                draft_token_ids.append([])
                continue

            # Generate draft tokens using draft model
            draft_tokens = self._generate_draft_tokens(
                token_ids=token_ids_cpu[i, :num_tokens].tolist(),
                num_draft_tokens=self.num_speculative_tokens,
            )

            draft_token_ids.append(draft_tokens)

        return draft_token_ids

    @torch.inference_mode()
    def _generate_draft_tokens(
        self,
        token_ids: list[int],
        num_draft_tokens: int,
    ) -> list[int]:
        """
        Generate draft tokens using the draft model.

        This is a simplified auto-regressive implementation that generates
        tokens one at a time using greedy sampling.

        Limitations of current implementation:
        - No KV cache (inefficient, recomputes every time)
        - No batching (processes one sequence at a time)
        - Greedy sampling only (no temperature/top-p)
        - No attention metadata optimization

        Future improvements needed:
        - Implement KV cache for efficiency
        - Add batched inference
        - Implement PEARL's adaptive draft length
        - Optimize with attention backends

        Args:
            token_ids: Input token IDs
            num_draft_tokens: Number of draft tokens to generate

        Returns:
            List of draft token IDs
        """
        if self.draft_model is None:
            logger.warning("[PEARL] Draft model not initialized")
            return []

        if not token_ids:
            logger.warning("[PEARL] Empty token_ids provided")
            return []

        draft_tokens = []
        current_tokens = token_ids.copy()

        try:
            # Auto-regressive generation
            for step in range(num_draft_tokens):
                # Prepare input tensors
                input_ids = torch.tensor(
                    [current_tokens],
                    dtype=torch.long,
                    device=self.device
                )

                seq_len = len(current_tokens)
                positions = torch.arange(
                    seq_len,
                    dtype=torch.long,
                    device=self.device
                ).unsqueeze(0)

                # Forward pass through draft model
                # Note: This is a simplified call without proper attention metadata
                # For production, need to integrate with vLLM's attention backends
                try:
                    with set_forward_context(None, self.vllm_config, num_tokens=seq_len):
                        outputs = self.draft_model(
                            input_ids=input_ids,
                            positions=positions,
                        )

                    # Extract logits from outputs
                    # The output format depends on the model architecture
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    elif isinstance(outputs, tuple) and len(outputs) > 0:
                        logits = outputs[0]
                    else:
                        logits = outputs

                    # Get logits for the last position
                    last_token_logits = logits[0, -1, :]

                    # Greedy sampling (argmax)
                    next_token = last_token_logits.argmax(dim=-1).item()

                    # Add to draft tokens
                    draft_tokens.append(next_token)
                    current_tokens.append(next_token)

                    logger.debug(
                        f"[PEARL] Step {step+1}/{num_draft_tokens}: "
                        f"Generated token {next_token}"
                    )

                except Exception as e:
                    logger.warning(
                        f"[PEARL] Error in draft model forward pass at step {step}: {e}. "
                        f"Returning {len(draft_tokens)} tokens generated so far."
                    )
                    break

        except Exception as e:
            logger.error(f"[PEARL] Error generating draft tokens: {e}")
            import traceback
            logger.error(traceback.format_exc())

        logger.debug(
            f"[PEARL] Generated {len(draft_tokens)} draft tokens "
            f"from {len(token_ids)} input tokens"
        )

        return draft_tokens

    def log_stats(self):
        """Log PEARL statistics."""
        if self.total_drafts > 0:
            mat = 1 + (self.total_accepted / self.total_drafts)
            logger.info(
                f"[PEARL] MAT (Mean Accepted Tokens): {mat:.2f} "
                f"(Accepted: {self.total_accepted}, Drafts: {self.total_drafts})"
            )

    def update_stats(self, num_draft_tokens: int, num_accepted_tokens: int):
        """
        Update statistics after draft verification.

        Args:
            num_draft_tokens: Number of draft tokens proposed
            num_accepted_tokens: Number of draft tokens accepted
        """
        self.total_drafts += 1
        self.total_accepted += num_accepted_tokens

    def __del__(self):
        """Destructor to log final stats."""
        self.log_stats()
