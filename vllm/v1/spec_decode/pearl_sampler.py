# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PEARL-specific acceptance sampler for speculative verification."""

from __future__ import annotations

import torch

from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import (
    PLACEHOLDER_TOKEN_ID,
    RejectionSampler,
    apply_sampling_constraints,
)
from vllm.v1.sample.sampler import Sampler, SamplerOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


class PearlRejectionSampler(RejectionSampler):
    """Use PEARL-style accept/reject with a fallback bonus token."""

    def __init__(self, sampler: Sampler):
        super().__init__(sampler)

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        draft_probs: torch.Tensor | None,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        del draft_probs
        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices
        batch_size = len(metadata.num_draft_tokens)
        device = logits.device
        pre_verify = metadata.pre_verify
        use_pre_verify = pre_verify is not None
        if pre_verify is None:
            pre_verify = [False] * batch_size

        # Sample bonus tokens from target logits (used when no draft tokens).
        bonus_logits = logits[bonus_logits_indices]
        bonus_sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=sampling_metadata,
            predict_bonus_token=True,
            logprobs_mode_override="processed_logits"
            if self.is_processed_logprobs_mode
            else "raw_logits",
        )
        bonus_token_ids = bonus_sampler_output.sampled_token_ids

        # Process target logits for the draft tokens.
        raw_target_logits = logits[target_logits_indices].to(torch.float32)
        target_logits = self.apply_logits_processors(
            raw_target_logits, sampling_metadata, metadata
        )
        target_logits = apply_sampling_constraints(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
        )
        target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)

        output_token_ids = torch.full(
            (batch_size, metadata.max_spec_len + 1),
            PLACEHOLDER_TOKEN_ID,
            dtype=torch.int32,
            device=device,
        )

        target_argmax = None
        if sampling_metadata.all_greedy:
            target_argmax = target_probs.argmax(dim=-1)

        cursor = 0
        for req_idx, num_draft in enumerate(metadata.num_draft_tokens):
            if num_draft == 0:
                output_token_ids[req_idx, 0] = bonus_token_ids[req_idx]
                continue
            gen = sampling_metadata.generators.get(req_idx)
            if use_pre_verify and pre_verify[req_idx]:
                tok = int(metadata.draft_token_ids[cursor].item())
                if target_argmax is not None:
                    chosen = int(target_argmax[cursor].item())
                    if tok == chosen:
                        output_token_ids[req_idx, :num_draft] = (
                            metadata.draft_token_ids[cursor : cursor + num_draft]
                        )
                    else:
                        output_token_ids[req_idx, 0] = chosen
                else:
                    prob = target_probs[cursor, tok]
                    u = torch.rand((), device=device, generator=gen)
                    if u <= prob:
                        output_token_ids[req_idx, :num_draft] = (
                            metadata.draft_token_ids[cursor : cursor + num_draft]
                        )
                    else:
                        masked = target_probs[cursor].clone()
                        masked[tok] = 0.0
                        if masked.sum() == 0:
                            revised = tok
                        else:
                            revised = int(
                                torch.multinomial(masked, 1, generator=gen).item()
                            )
                        output_token_ids[req_idx, 0] = revised
                cursor += num_draft
                continue

            accepted_all = True
            for j in range(num_draft):
                tok = int(metadata.draft_token_ids[cursor + j].item())
                if target_argmax is not None:
                    chosen = int(target_argmax[cursor + j].item())
                    if tok == chosen:
                        output_token_ids[req_idx, j] = tok
                        continue
                    output_token_ids[req_idx, j] = chosen
                    accepted_all = False
                    break

                prob = target_probs[cursor + j, tok]
                u = torch.rand((), device=device, generator=gen)
                if u <= prob:
                    output_token_ids[req_idx, j] = tok
                    continue

                masked = target_probs[cursor + j].clone()
                masked[tok] = 0.0
                if masked.sum() == 0:
                    revised = tok
                else:
                    revised = int(
                        torch.multinomial(masked, 1, generator=gen).item()
                    )
                output_token_ids[req_idx, j] = revised
                accepted_all = False
                break

            if accepted_all:
                output_token_ids[req_idx, :num_draft] = (
                    metadata.draft_token_ids[cursor : cursor + num_draft]
                )
                if not use_pre_verify:
                    output_token_ids[req_idx, num_draft] = bonus_token_ids[req_idx]
            cursor += num_draft

        logprobs_tensors = None
        if sampling_metadata.max_num_logprobs is not None:
            bonus_logprobs = None
            if bonus_sampler_output.logprobs_tensors is not None:
                bonus_logprobs = bonus_sampler_output.logprobs_tensors.logprobs
            if bonus_logprobs is not None:
                logprobs_tensors = self._get_logprobs_tensors(
                    sampling_metadata.max_num_logprobs,
                    metadata,
                    logits,
                    target_logits
                    if self.is_processed_logprobs_mode
                    else raw_target_logits,
                    bonus_logprobs,
                    output_token_ids,
                )

        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=logprobs_tensors,
        )
