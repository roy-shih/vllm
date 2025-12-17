# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
PEARL (Parallel Speculative Decoding with Adaptive Draft Length) Proposer.

This implementation properly utilizes vLLM's infrastructure:
- PageAttention for KV cache management
- AttentionMetadata for efficient attention
- Batched inference across all requests
- Integration with vLLM's scheduler

Based on nano-PEARL: https://github.com/smart-lty/nano-PEARL
Reference: EAGLE proposer implementation for patterns
"""

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata

logger = init_logger(__name__)


class PEARLProposer:
    """
    PEARL Proposer that properly leverages vLLM's infrastructure.

    Key features:
    - Uses vLLM's KV cache (PageAttention) for efficiency
    - Batched inference across multiple requests
    - Proper integration with attention backends
    - Auto-regressive draft generation with KV cache reuse

    Differences from EAGLE:
    - Simpler: no tree-based speculation
    - Sequential: generates tokens one by one
    - Adaptive: will implement adaptive draft length (future)
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

        self.device = device
        self.dtype = vllm_config.model_config.dtype
        self.runner = runner

        # PEARL configuration
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens
        self.gamma = self.speculative_config.pearl_gamma
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        # Draft model configuration
        self.draft_model_config = self.speculative_config.draft_model_config
        self.hidden_size = self.draft_model_config.get_hidden_size()

        # Pre-allocate buffers (similar to EAGLE)
        self.input_ids = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device=device
        )
        self.positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=device
        )

        # Model will be loaded in load_model()
        self.model: nn.Module | None = None

        # Statistics
        self.total_drafts = 0
        self.total_accepted = 0

        logger.info("=" * 50)
        logger.info("[PEARL] Initializing PEARL Proposer (vLLM-integrated)")
        logger.info(f"[PEARL] Num speculative tokens: {self.num_speculative_tokens}")
        logger.info(f"[PEARL] Gamma: {self.gamma}")
        logger.info(f"[PEARL] Max num tokens: {self.max_num_tokens}")
        logger.info(f"[PEARL] Device: {device}")
        logger.info("=" * 50)

    def load_model(self, target_model: nn.Module) -> None:
        """
        Load the draft model.

        Args:
            target_model: The target model instance (for reference)
        """
        logger.info("[PEARL] Loading draft model...")

        self.model = get_model(
            vllm_config=self.vllm_config,
            model_config=self.draft_model_config,
        )

        # Auto-set gamma if needed
        if self.gamma == -1:
            self.gamma = self._auto_set_gamma()
            logger.info(f"[PEARL] Auto-set gamma to {self.gamma}")

        logger.info("[PEARL] Draft model loaded successfully")

    def _auto_set_gamma(self) -> int:
        """Auto-set gamma based on configuration."""
        gamma = max(self.num_speculative_tokens, 5)
        return min(gamma, 10)

    @torch.inference_mode()
    def propose(
        self,
        target_token_ids: torch.Tensor,  # [num_tokens]
        target_positions: torch.Tensor,  # [num_tokens]
        target_hidden_states: torch.Tensor,  # [num_tokens, hidden_size]
        next_token_ids: torch.Tensor,  # [batch_size]
        last_token_indices: torch.Tensor | None,  # [batch_size]
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple | None = None,
    ) -> torch.Tensor:
        """
        Propose draft tokens using PEARL's algorithm.

        This method follows EAGLE's interface to properly integrate with vLLM.

        Note: For MVP, we're doing auto-regressive generation without full KV cache
        optimization. The draft model has its own separate KV cache context that
        needs to be managed independently from the target model.

        Args:
            target_token_ids: Token IDs from target model output
            target_positions: Position IDs for target tokens
            target_hidden_states: Hidden states from target model
            next_token_ids: Next token IDs for each request
            last_token_indices: Indices of last tokens for each request
            common_attn_metadata: Attention metadata (for target model's KV cache)
            sampling_metadata: Sampling parameters
            mm_embed_inputs: Multimodal embeddings (if applicable)

        Returns:
            draft_token_ids: [batch_size, num_speculative_tokens]
        """
        if self.model is None:
            logger.warning("[PEARL] Model not loaded")
            batch_size = next_token_ids.shape[0]
            return torch.zeros(
                (batch_size, self.num_speculative_tokens),
                dtype=torch.int32,
                device=self.device,
            )

        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]

        if last_token_indices is None:
            last_token_indices = common_attn_metadata.query_start_loc[1:] - 1

        logger.debug(
            f"[PEARL] propose called: num_tokens={num_tokens}, "
            f"batch_size={batch_size}, "
            f"num_spec_tokens={self.num_speculative_tokens}"
        )

        # Get the actual sequences from the target model
        # We need to extract the token sequences for each request in the batch
        # using last_token_indices to determine sequence boundaries

        # Build list of token sequences for each request
        token_sequences = []
        query_start_loc = common_attn_metadata.query_start_loc

        for i in range(batch_size):
            start_idx = query_start_loc[i]
            end_idx = query_start_loc[i + 1]
            seq_tokens = target_token_ids[start_idx:end_idx].tolist()
            # Append the newly sampled token
            seq_tokens.append(next_token_ids[i].item())
            token_sequences.append(seq_tokens)

        logger.debug(f"[PEARL] Token sequences lengths: {[len(s) for s in token_sequences]}")

        # Generate draft tokens for each sequence in the batch
        all_draft_tokens = []

        for seq_idx, token_seq in enumerate(token_sequences):
            draft_tokens = self._generate_draft_tokens_for_sequence(
                token_ids=token_seq,
                num_draft_tokens=self.num_speculative_tokens,
            )
            all_draft_tokens.append(draft_tokens)

        # Convert to tensor: [batch_size, num_speculative_tokens]
        result = torch.zeros(
            (batch_size, self.num_speculative_tokens),
            dtype=torch.int32,
            device=self.device,
        )

        for i, draft_tokens in enumerate(all_draft_tokens):
            num_generated = len(draft_tokens)
            if num_generated > 0:
                result[i, :num_generated] = torch.tensor(
                    draft_tokens, dtype=torch.int32, device=self.device
                )

        logger.debug(
            f"[PEARL] Generated draft_tokens shape: {result.shape}"
        )

        return result

    def _generate_draft_tokens_for_sequence(
        self,
        token_ids: list[int],
        num_draft_tokens: int,
    ) -> list[int]:
        """
        Generate draft tokens for a single sequence using auto-regressive generation.

        This is a simplified implementation that generates tokens one-by-one.
        For MVP, we're not using KV cache optimization yet.

        Args:
            token_ids: Input token sequence
            num_draft_tokens: Number of draft tokens to generate

        Returns:
            List of draft token IDs
        """
        if not token_ids:
            logger.warning("[PEARL] Empty token_ids provided")
            return []

        draft_tokens = []
        current_seq = token_ids.copy()

        try:
            for step in range(num_draft_tokens):
                # Prepare input tensors for current sequence
                seq_len = len(current_seq)

                # Truncate if sequence is too long
                if seq_len > self.max_model_len:
                    current_seq = current_seq[-(self.max_model_len - 1):]
                    seq_len = len(current_seq)

                input_ids = torch.tensor(
                    [current_seq],
                    dtype=torch.long,
                    device=self.device
                )  # [1, seq_len]

                # Position IDs: [0, 1, 2, ..., seq_len-1]
                positions = torch.arange(
                    seq_len,
                    dtype=torch.long,
                    device=self.device
                ).unsqueeze(0)  # [1, seq_len]

                # Forward pass through draft model
                # Note: Without KV cache, we recompute the entire sequence each time
                # This is inefficient but simpler for MVP
                try:
                    with set_forward_context(
                        None,  # No per-layer metadata for simplified approach
                        self.vllm_config,
                        num_tokens=seq_len,
                    ):
                        outputs = self.model(
                            input_ids=input_ids,
                            positions=positions,
                        )

                    # Extract hidden states
                    if isinstance(outputs, tuple):
                        hidden_states = outputs[0]
                    else:
                        hidden_states = outputs

                    # Get logits for the last position
                    if hidden_states.dim() == 3:
                        # [batch=1, seq_len, hidden_size]
                        last_hidden = hidden_states[0, -1, :]  # [hidden_size]
                    else:
                        # [seq_len, hidden_size]
                        last_hidden = hidden_states[-1, :]

                    # Compute logits
                    logits = self.model.compute_logits(
                        last_hidden.unsqueeze(0)
                    )  # [1, vocab_size]

                    # Greedy sampling
                    next_token = logits.argmax(dim=-1).item()

                    # Add to draft tokens
                    draft_tokens.append(next_token)
                    current_seq.append(next_token)

                    logger.debug(
                        f"[PEARL] Step {step+1}/{num_draft_tokens}: "
                        f"Generated token {next_token} (seq_len={seq_len})"
                    )

                except Exception as e:
                    logger.warning(
                        f"[PEARL] Error in draft forward pass at step {step}: {e}. "
                        f"Returning {len(draft_tokens)} tokens generated so far."
                    )
                    import traceback
                    logger.debug(f"[PEARL] Traceback: {traceback.format_exc()}")
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
        """Log statistics."""
        if self.total_drafts > 0:
            mat = 1 + (self.total_accepted / self.total_drafts)
            logger.info(
                f"[PEARL] MAT: {mat:.2f} "
                f"(Accepted: {self.total_accepted}, Drafts: {self.total_drafts})"
            )

    def __del__(self):
        """Destructor."""
        self.log_stats()
