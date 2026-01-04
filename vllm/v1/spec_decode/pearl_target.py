# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PEARL target-side verification helpers (CPU-friendly).

This module mirrors the target verifier logic in a simplified, deterministic
form to enable unit testing without distributed setup or GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch

from .pearl import DraftVerifyPayload, PearlSequenceState


@dataclass
class TargetVerifyResult:
    acc: List[bool]
    rollout: List[int]
    revise_token: List[int]
    finish: List[bool]


class TargetVerifier:
    """Target-side verifier with Bernoulli acceptance and rollback semantics."""

    def __init__(self, gamma: int, eos_token_id: int | Sequence[int], max_tokens: int):
        self.gamma = gamma
        self.eos_set = (
            set(eos_token_id)
            if isinstance(eos_token_id, (list, tuple, set))
            else {eos_token_id}
        )
        self.max_tokens = max_tokens

    def _accept_reject(
        self,
        probs: torch.Tensor,
        token: int,
    ) -> tuple[bool, int]:
        """Bernoulli accept; on reject sample a revised token."""
        u = torch.rand((), device=probs.device)
        accept = u <= probs[token]
        if accept:
            return True, -1
        # avoid sampling the same token if its prob dominates; mask it out
        masked = probs.clone()
        masked[token] = 0.0
        if masked.sum() == 0:
            return False, token
        dist = torch.distributions.Categorical(masked / masked.sum())
        revise = int(dist.sample().item())
        return False, revise

    def _finish_check(self, seq: PearlSequenceState, token: int, accepted: bool) -> bool:
        if accepted and (not seq.ignore_eos) and (token in self.eos_set):
            return True
        completion_tokens = len(seq.token_ids) - (seq.num_prompt_tokens or 0)
        return completion_tokens >= self.max_tokens

    def verify_and_update(
        self,
        seqs: Sequence[PearlSequenceState],
        payload: DraftVerifyPayload,
        logits: torch.Tensor,
        temperatures: torch.Tensor | None = None,
    ) -> TargetVerifyResult:
        """Verify tokens and mutate seqs according to accept/reject outcomes."""
        if logits.shape[0] != len(payload.to_verify_tokens):
            raise ValueError("Mismatch between logits rows and to-verify tokens.")
        if temperatures is not None:
            # For now we ignore temperatures to avoid extra branching; callers
            # can pre-scale logits if needed.
            _ = temperatures
        probs = torch.softmax(logits, dim=-1)

        acc: List[bool] = []
        rollout: List[int] = []
        revise_token: List[int] = []
        finish: List[bool] = []

        cursor = 0
        for seq_idx, seq in enumerate(seqs):
            verify_count = payload.verify_counts[seq_idx]
            if verify_count == 1:
                token = payload.to_verify_tokens[cursor]
                accepted, revised = self._accept_reject(probs[cursor], token)
                if accepted:
                    # Draft already appended tokens; no-op on success.
                    pass
                else:
                    # rollback drafted gamma window and insert revised token
                    seq.token_ids = seq.token_ids[: -(self.gamma)] + [revised]
                finish_flag = self._finish_check(seq, token if accepted else revised, accepted)
                acc.append(accepted)
                rollout.append(self.gamma if not accepted else 0)
                revise_token.append(revised)
                finish.append(finish_flag)
                cursor += 1
                seq.pre_verify = not accepted
            else:
                # post-verify window of length gamma
                window_tokens = payload.to_verify_tokens[cursor : cursor + verify_count]
                window_probs = probs[cursor : cursor + verify_count]
                first_reject = -1
                revised_token = -1
                for j, (tok, p) in enumerate(zip(window_tokens, window_probs)):
                    accepted, revised = self._accept_reject(p, tok)
                    if not accepted:
                        first_reject = j
                        revised_token = revised
                        break
                if first_reject == -1:
                    # all accepted
                    acc.append(True)
                    rollout.append(0)
                    revise_token.append(-1)
                    finish_flag = any(
                        (t in self.eos_set) and (not seq.ignore_eos) for t in window_tokens
                    ) or self._finish_check(seq, window_tokens[-1], True)
                    finish.append(finish_flag)
                    seq.pre_verify = False
                else:
                    # rollback rejected suffix and insert revised token
                    rollback_n = verify_count - first_reject
                    seq.token_ids = seq.token_ids[: -self.gamma] + seq.token_ids[
                        -self.gamma : -self.gamma + first_reject
                    ]
                    seq.token_ids.append(revised_token)
                    acc.append(False)
                    rollout.append(rollback_n)
                    revise_token.append(revised_token)
                    finish_flag = self._finish_check(seq, revised_token, False)
                    finish.append(finish_flag)
                    seq.pre_verify = True
                cursor += verify_count

        return TargetVerifyResult(
            acc=acc, rollout=rollout, revise_token=revise_token, finish=finish
        )


def _gather_probs(logits: torch.Tensor, tokens: Sequence[int]) -> torch.Tensor:
    """Select probabilities for provided tokens; logits assumed batch-first."""
    probs = torch.softmax(logits, dim=-1)
    idx = torch.tensor(tokens, device=logits.device)
    return probs.gather(dim=1, index=idx.view(-1, 1)).squeeze(1)


def verify_on_target(
    seqs: Sequence[PearlSequenceState],
    payload: DraftVerifyPayload,
    logits: torch.Tensor,
    gamma: int,
    eos_token_id: int | Sequence[int],
    max_tokens: int,
    temperatures: torch.Tensor | None = None,
) -> TargetVerifyResult:
    """
    Deterministic verification: accept if target argmax matches draft token,
    otherwise reject and pick target argmax as revised token. Rollout is
    computed as remaining tokens in gamma window when rejection occurs.
    """
    if temperatures is not None:
        # For now we keep deterministic behavior; temperature unused.
        _ = temperatures

    num_to_verify = len(payload.to_verify_tokens)
    if logits.shape[0] != num_to_verify:
        raise ValueError("Mismatch between logits rows and to-verify tokens.")

    # Compute acceptance by comparing argmax with provided token.
    argmax_tokens = logits.argmax(dim=-1).tolist()
    acc_flags = [
        argmax_tokens[i] == payload.to_verify_tokens[i] for i in range(num_to_verify)
    ]

    acc: List[bool] = []
    rollout: List[int] = []
    revise_token: List[int] = []
    finish: List[bool] = []

    cursor = 0
    eos_set = set(eos_token_id) if isinstance(eos_token_id, (list, tuple, set)) else {eos_token_id}

    for seq_idx, seq in enumerate(seqs):
        verify_count = payload.verify_counts[seq_idx]
        seq_acc = True
        seq_rollout = 0
        seq_revise = -1
        seq_finish = False

        if verify_count == 1:
            seq_acc = acc_flags[cursor]
            if not seq_acc:
                seq_revise = argmax_tokens[cursor]
                seq_rollout = gamma
            seq_finish = seq_acc and (not seq.ignore_eos) and (
                payload.to_verify_tokens[cursor] in eos_set
            )
            cursor += 1
        else:
            # post-verify window of length gamma
            window_flags = acc_flags[cursor : cursor + verify_count]
            window_tokens = payload.to_verify_tokens[cursor : cursor + verify_count]
            first_reject = next((i for i, ok in enumerate(window_flags) if not ok), -1)
            if first_reject == -1:
                seq_acc = True
                seq_rollout = 0
                seq_revise = -1
                seq_finish = any((t in eos_set) for t, ok in zip(window_tokens, window_flags) if ok and not seq.ignore_eos)
            else:
                seq_acc = False
                seq_rollout = verify_count - first_reject
                seq_revise = argmax_tokens[cursor + first_reject]
                seq_finish = (window_tokens[first_reject] in eos_set) and (not seq.ignore_eos)
            cursor += verify_count

        acc.append(seq_acc)
        rollout.append(seq_rollout)
        revise_token.append(seq_revise)
        # Also consider max_tokens cap.
        hit_cap = (len(seq.token_ids) - seq.num_prompt_tokens) >= max_tokens
        finish.append(seq_finish or hit_cap)

    return TargetVerifyResult(
        acc=acc, rollout=rollout, revise_token=revise_token, finish=finish
    )
