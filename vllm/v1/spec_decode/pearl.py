# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
PEARL speculative decoding helpers (draft side).

This module is currently self-contained to ease incremental integration. It
implements the draft-side gamma-window generation and packaging of tokens for
verification. Distributed wiring (subgroups/broadcast) is expected to be
handled by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Sequence

import torch


@dataclass
class PearlSequenceState:
    """Minimal per-sequence state for PEARL draft generation."""

    seq_id: int
    token_ids: List[int] = field(default_factory=list)
    pre_verify: bool = True
    ignore_eos: bool = False
    num_prompt_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.num_prompt_tokens is None:
            self.num_prompt_tokens = len(self.token_ids)

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    @property
    def last_token(self) -> int:
        return self.token_ids[-1]


@dataclass
class DraftVerifyPayload:
    """Payload the draft side prepares for target-side verification."""

    to_verify_tokens: List[int]
    verify_counts: List[int]
    next_round_tokens_per_seq: List[List[int]]

    @property
    def flat_next_round_tokens(self) -> List[int]:
        return [t for seq_tokens in self.next_round_tokens_per_seq for t in seq_tokens]


class PearlDraftRunner:
    """Draft-side PEARL runner: gamma greedy decode and package verification."""

    def __init__(self, gamma: int, device: str | None = None) -> None:
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        self.gamma = gamma
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def _prepare_decode_inputs(
        self, seqs: Sequence[PearlSequenceState]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare decode inputs (last token + position) for each sequence."""
        input_ids = torch.tensor(
            [seq.last_token for seq in seqs],
            dtype=torch.long,
            device=self.device,
        )
        positions = torch.tensor(
            [len(seq.token_ids) - 1 for seq in seqs],
            dtype=torch.long,
            device=input_ids.device,
        )
        return input_ids, positions

    def _package_verification(
        self,
        seqs: Sequence[PearlSequenceState],
        pre_window_tokens: Sequence[int],
    ) -> DraftVerifyPayload:
        """
        Build the verification payload:
        - pre_verify sequences send 1 token (the token before the draft window).
        - post_verify sequences send gamma tokens (the window to be judged).
        Next-round input is always the latest gamma tokens per sequence.
        """
        to_verify_tokens: List[int] = []
        verify_counts: List[int] = []
        next_round_tokens_per_seq: List[List[int]] = []

        for idx, seq in enumerate(seqs):
            window = seq.token_ids[-self.gamma :]
            next_round_tokens_per_seq.append(window)
            if seq.pre_verify:
                # Verify the token immediately before the newly drafted window.
                to_verify_tokens.append(pre_window_tokens[idx])
                verify_counts.append(1)
            else:
                to_verify_tokens.extend(seq.token_ids[-2 * self.gamma + 1 : -self.gamma + 1])
                verify_counts.append(self.gamma)

        return DraftVerifyPayload(
            to_verify_tokens=to_verify_tokens,
            verify_counts=verify_counts,
            next_round_tokens_per_seq=next_round_tokens_per_seq,
        )

    def pearl_step(
        self,
        seqs: Sequence[PearlSequenceState],
        run_model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> DraftVerifyPayload:
        """
        Run gamma decode steps greedily (temperature-free), append tokens
        without EOS checks, and return the verification payload.

        Args:
            seqs: sequences currently running on the draft side.
            run_model_fn: callable accepting (input_ids, positions) and
                returning logits (batch, vocab).

        Returns:
            DraftVerifyPayload for target-side verification.
        """
        pre_window_tokens = [seq.last_token for seq in seqs]
        for _ in range(self.gamma):
            input_ids, positions = self._prepare_decode_inputs(seqs)
            logits = run_model_fn(input_ids, positions)
            sample_tokens = logits.argmax(dim=-1)
            token_ids = sample_tokens.tolist()
            for seq, token_id in zip(seqs, token_ids):
                seq.append_token(token_id)

        return self._package_verification(seqs, pre_window_tokens)
