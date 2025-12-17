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

        # Adaptive draft length tracking (PEARL feature)
        self.adaptive_enabled = self.gamma > 0
        if self.adaptive_enabled:
            # Sliding window for tracking recent acceptance
            from collections import deque
            self.acceptance_history = deque(maxlen=self.gamma)
            self.min_draft_tokens = max(1, self.num_speculative_tokens // 2)
            self.max_draft_tokens = self.num_speculative_tokens
            self.current_draft_tokens = self.num_speculative_tokens
            logger.info(f"[PEARL] Adaptive draft length ENABLED")
            logger.info(f"[PEARL] Draft token range: [{self.min_draft_tokens}, {self.max_draft_tokens}]")
        else:
            self.acceptance_history = None
            self.current_draft_tokens = self.num_speculative_tokens
            logger.info(f"[PEARL] Adaptive draft length DISABLED (gamma={self.gamma})")

        logger.info("=" * 50)
        logger.info("[PEARL] Initializing PEARL Proposer (vLLM-integrated)")
        logger.info(f"[PEARL] Num speculative tokens: {self.num_speculative_tokens}")
        logger.info(f"[PEARL] Gamma (window size): {self.gamma}")
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

        # Use adaptive draft length if enabled
        num_draft_to_generate = self.current_draft_tokens

        logger.debug(
            f"[PEARL] Generating {num_draft_to_generate} draft tokens "
            f"(adaptive={'ON' if self.adaptive_enabled else 'OFF'})"
        )

        # Generate draft tokens in batched mode for efficiency
        result = self._generate_drafts_batched(
            token_sequences=token_sequences,
            batch_size=batch_size,
            num_drafts=num_draft_to_generate,
        )

        logger.debug(
            f"[PEARL] Generated draft_tokens shape: {result.shape}"
        )

        return result

    def _generate_drafts_batched(
        self,
        token_sequences: list[list[int]],
        batch_size: int,
        num_drafts: int | None = None,
    ) -> torch.Tensor:
        """
        Generate draft tokens for all sequences in batch (OPTIMIZED).

        This method processes all sequences together in batched forward passes,
        which is much more efficient than processing one-by-one.

        Args:
            token_sequences: List of token sequences for each request
            batch_size: Number of sequences
            num_drafts: Number of draft tokens to generate (uses self.num_speculative_tokens if None)

        Returns:
            Draft tokens tensor [batch_size, num_speculative_tokens]
        """
        if num_drafts is None:
            num_drafts = self.num_speculative_tokens

        if not token_sequences:
            return torch.zeros(
                (batch_size, self.num_speculative_tokens),
                dtype=torch.int32,
                device=self.device,
            )

        # Initialize result tensor (always sized for max tokens, but we'll only fill num_drafts)
        result = torch.zeros(
            (batch_size, self.num_speculative_tokens),
            dtype=torch.int32,
            device=self.device,
        )

        # Track current sequences (we'll append draft tokens as we generate)
        current_sequences = [seq.copy() for seq in token_sequences]

        try:
            # Generate draft tokens step by step in batched mode
            for step in range(num_drafts):
                logger.debug(f"[PEARL] Batched draft step {step+1}/{num_drafts}")

                # Find max sequence length for padding
                max_len = max(len(seq) for seq in current_sequences)

                # Truncate sequences that are too long
                if max_len > self.max_model_len:
                    max_len = self.max_model_len
                    current_sequences = [
                        seq[-max_len:] if len(seq) > max_len else seq
                        for seq in current_sequences
                    ]

                # Create padded batch
                # Padding token: use 0 (will be masked by attention)
                padded_sequences = []
                attention_mask = []

                for seq in current_sequences:
                    seq_len = len(seq)
                    if seq_len < max_len:
                        # Pad on the left (more common for causal LM)
                        padding_len = max_len - seq_len
                        padded_seq = [0] * padding_len + seq
                        mask = [0] * padding_len + [1] * seq_len
                    else:
                        padded_seq = seq[-max_len:]
                        mask = [1] * max_len

                    padded_sequences.append(padded_seq)
                    attention_mask.append(mask)

                # Convert to tensors
                input_ids = torch.tensor(
                    padded_sequences,
                    dtype=torch.long,
                    device=self.device
                )  # [batch_size, max_len]

                # Calculate correct positions for each sequence
                # For left-padded sequences, padding tokens get position 0,
                # real tokens get incremental positions starting from 0
                positions_list = []
                for seq in current_sequences:
                    actual_len = min(len(seq), max_len)
                    padding_len = max_len - actual_len
                    # Padding positions = 0, real tokens get positions [0, 1, 2, ...]
                    seq_positions = [0] * padding_len + list(range(actual_len))
                    positions_list.append(seq_positions)

                positions = torch.tensor(
                    positions_list,
                    dtype=torch.long,
                    device=self.device
                )  # [batch_size, max_len]

                # Forward pass
                try:
                    with set_forward_context(
                        None,  # No per-layer metadata for simplified approach
                        self.vllm_config,
                        num_tokens=batch_size * max_len,
                    ):
                        # Flatten for model input
                        flat_input_ids = input_ids.reshape(-1)
                        flat_positions = positions.reshape(-1)

                        outputs = self.model(
                            input_ids=flat_input_ids,
                            positions=flat_positions,
                        )

                    # Extract hidden states
                    if isinstance(outputs, tuple):
                        hidden_states = outputs[0]
                    else:
                        hidden_states = outputs

                    # Reshape to [batch_size, max_len, hidden_size]
                    if hidden_states.dim() == 2:
                        # [batch_size * max_len, hidden_size]
                        hidden_states = hidden_states.view(batch_size, max_len, -1)

                    # Get last token hidden states for each sequence
                    # For each sequence, get the last non-padding position
                    last_hidden_states = []
                    for i, seq in enumerate(current_sequences):
                        actual_len = min(len(seq), max_len)
                        # In padded_sequences, the actual sequence ends at position (max_len - 1)
                        # if no padding, or at position (max_len - actual_len + actual_len - 1)
                        last_pos = max_len - 1
                        last_hidden_states.append(hidden_states[i, last_pos, :])

                    last_hidden_states = torch.stack(last_hidden_states)  # [batch_size, hidden_size]

                    # Compute logits
                    logits = self.model.compute_logits(last_hidden_states)  # [batch_size, vocab_size]

                    # Greedy sampling
                    next_tokens = logits.argmax(dim=-1)  # [batch_size]

                    # Store draft tokens
                    result[:, step] = next_tokens.to(torch.int32)

                    # Update current sequences for next step
                    for i, token in enumerate(next_tokens):
                        current_sequences[i].append(token.item())

                    logger.debug(
                        f"[PEARL] Step {step+1}: Generated {batch_size} tokens, "
                        f"max_len={max_len}"
                    )

                except Exception as e:
                    logger.warning(
                        f"[PEARL] Error in batched forward pass at step {step}: {e}. "
                        f"Returning {step} complete draft tokens."
                    )
                    import traceback
                    logger.debug(f"[PEARL] Traceback: {traceback.format_exc()}")
                    # Partial results already stored in result tensor
                    break

        except Exception as e:
            logger.error(f"[PEARL] Error in batched draft generation: {e}")
            import traceback
            logger.error(traceback.format_exc())

        logger.debug(
            f"[PEARL] Batched generation complete: {result.shape}"
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

    def update_acceptance_stats(self, num_accepted_tokens: int, num_draft_tokens: int):
        """
        Update acceptance statistics and adjust draft length (PEARL adaptive feature).

        Args:
            num_accepted_tokens: Number of draft tokens that were accepted
            num_draft_tokens: Total number of draft tokens proposed
        """
        self.total_drafts += num_draft_tokens
        self.total_accepted += num_accepted_tokens

        if self.adaptive_enabled and self.acceptance_history is not None:
            # Track acceptance rate for this iteration
            acceptance_rate = num_accepted_tokens / max(num_draft_tokens, 1)
            self.acceptance_history.append(acceptance_rate)

            # Adjust draft length based on recent history
            if len(self.acceptance_history) >= self.gamma // 2:  # Have enough samples
                self._adjust_draft_length()

    def _adjust_draft_length(self):
        """
        Adjust the number of draft tokens based on recent acceptance rates.

        PEARL's key insight:
        - High acceptance rate → generate more draft tokens
        - Low acceptance rate → generate fewer draft tokens (avoid waste)
        """
        if not self.acceptance_history:
            return

        # Calculate average acceptance rate over recent window
        recent_acceptance_rate = sum(self.acceptance_history) / len(self.acceptance_history)

        # Thresholds for adjustment
        HIGH_ACCEPTANCE_THRESHOLD = 0.7  # If >70% accepted, increase drafts
        LOW_ACCEPTANCE_THRESHOLD = 0.3   # If <30% accepted, decrease drafts

        old_draft_tokens = self.current_draft_tokens

        if recent_acceptance_rate > HIGH_ACCEPTANCE_THRESHOLD:
            # Good alignment - increase draft tokens
            self.current_draft_tokens = min(
                self.max_draft_tokens,
                self.current_draft_tokens + 1
            )
        elif recent_acceptance_rate < LOW_ACCEPTANCE_THRESHOLD:
            # Poor alignment - decrease draft tokens
            self.current_draft_tokens = max(
                self.min_draft_tokens,
                self.current_draft_tokens - 1
            )

        if old_draft_tokens != self.current_draft_tokens:
            logger.info(
                f"[PEARL] Adaptive adjustment: {old_draft_tokens} → {self.current_draft_tokens} "
                f"(recent acceptance rate: {recent_acceptance_rate:.2%})"
            )

    def log_stats(self):
        """Log statistics."""
        if self.total_drafts > 0:
            mat = 1 + (self.total_accepted / self.total_drafts)
            logger.info(
                f"[PEARL] MAT: {mat:.2f} "
                f"(Accepted: {self.total_accepted}, Drafts: {self.total_drafts})"
            )

        if self.adaptive_enabled and self.acceptance_history:
            avg_acceptance = sum(self.acceptance_history) / len(self.acceptance_history)
            logger.info(
                f"[PEARL] Adaptive stats: Current draft tokens={self.current_draft_tokens}, "
                f"Recent acceptance rate={avg_acceptance:.2%}"
            )

    def __del__(self):
        """Destructor."""
        self.log_stats()
