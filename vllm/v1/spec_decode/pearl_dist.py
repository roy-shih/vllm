# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PEARL distributed helpers: rank partitioning and NCCL subgroup creation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.distributed as dist

from vllm.config.speculative import SpeculativeConfig
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID


@dataclass
class PearlRole:
    draft_ranks: List[int]
    target_ranks: List[int]
    is_draft: bool
    local_rank: int
    global_rank: int


@dataclass
class PearlGroups:
    role: PearlRole
    draft_group: dist.ProcessGroup
    target_group: dist.ProcessGroup
    verify_group: dist.ProcessGroup


def compute_pearl_role(
    global_rank: int, draft_tp: int, target_tp: int
) -> PearlRole:
    draft_ranks = list(range(draft_tp))
    target_ranks = list(range(draft_tp, draft_tp + target_tp))
    return _compute_role_from_ranks(global_rank, draft_ranks, target_ranks)


def _compute_role_from_ranks(
    global_rank: int,
    draft_ranks: list[int],
    target_ranks: list[int],
) -> PearlRole:
    if global_rank in draft_ranks:
        is_draft = True
        local_rank = draft_ranks.index(global_rank)
    elif global_rank in target_ranks:
        is_draft = False
        local_rank = target_ranks.index(global_rank)
    else:
        raise ValueError(
            f"Rank {global_rank} not in draft ({draft_ranks}) or target "
            f"({target_ranks}) ranges."
        )
    return PearlRole(
        draft_ranks=draft_ranks,
        target_ranks=target_ranks,
        is_draft=is_draft,
        local_rank=local_rank,
        global_rank=global_rank,
    )


def _reshape_world_ranks(
    world_size: int,
    dp: int,
    pp: int,
    pcp: int,
    tp: int,
) -> torch.Tensor:
    per_replica = dp * pp * pcp * tp
    if per_replica <= 0 or world_size % per_replica != 0:
        raise ValueError(
            "PEARL rank layout mismatch: "
            f"world_size({world_size}) is not divisible by "
            f"dp({dp})*pp({pp})*pcp({pcp})*tp({tp})."
        )
    outer = world_size // per_replica
    return torch.arange(world_size).reshape(outer, dp, pp, pcp, tp)


def _flatten_stage_tp(
    stage_ranks: torch.Tensor, tp_start: int, tp_end: int
) -> list[int]:
    # Order by pp descending so the last stage is the leader.
    pp_size = stage_ranks.shape[0]
    ordered: list[int] = []
    for pp_idx in range(pp_size - 1, -1, -1):
        for tp_idx in range(tp_start, tp_end):
            ordered.append(int(stage_ranks[pp_idx, tp_idx].item()))
    return ordered


def build_pearl_group_ranks(
    world_size: int,
    dp: int,
    pp: int,
    pcp: int,
    tp: int,
    draft_tp: int,
    target_tp: int,
) -> list[tuple[list[int], list[int]]]:
    all_ranks = _reshape_world_ranks(world_size, dp, pp, pcp, tp)
    groups: list[tuple[list[int], list[int]]] = []
    for outer_idx in range(all_ranks.shape[0]):
        for dp_idx in range(dp):
            for pcp_idx in range(pcp):
                stage_ranks = all_ranks[outer_idx, dp_idx, :, pcp_idx, :]
                draft_ranks = _flatten_stage_tp(stage_ranks, 0, draft_tp)
                target_ranks = _flatten_stage_tp(
                    stage_ranks, draft_tp, draft_tp + target_tp
                )
                groups.append((draft_ranks, target_ranks))
    return groups


def init_pearl_groups(spec_cfg: SpeculativeConfig) -> PearlGroups:
    """Create PEARL NCCL subgroups based on draft/target TP sizes."""
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before PEARL groups.")
    world_size = dist.get_world_size()
    draft_tp = spec_cfg.pearl_draft_tensor_parallel_size or 1
    target_tp = spec_cfg.pearl_target_tensor_parallel_size or 0
    parallel_cfg = spec_cfg.target_parallel_config
    if parallel_cfg is None:
        raise ValueError("target_parallel_config must be provided for PEARL.")
    tp = parallel_cfg.tensor_parallel_size
    if draft_tp + target_tp != tp:
        raise ValueError(
            f"PEARL TP mismatch: draft_tp({draft_tp}) + "
            f"target_tp({target_tp}) != tp({tp})."
        )
    dp = parallel_cfg.data_parallel_size
    pp = parallel_cfg.pipeline_parallel_size
    pcp = parallel_cfg.prefill_context_parallel_size

    all_ranks = _reshape_world_ranks(world_size, dp, pp, pcp, tp)
    rank = dist.get_rank()
    indices = (all_ranks == rank).nonzero(as_tuple=False)
    if indices.numel() == 0:
        raise RuntimeError(f"Rank {rank} not found in PEARL rank layout.")
    outer_idx, dp_idx, pp_idx, pcp_idx, tp_idx = indices[0].tolist()
    stage_ranks = all_ranks[outer_idx, dp_idx, :, pcp_idx, :]
    draft_ranks = _flatten_stage_tp(stage_ranks, 0, draft_tp)
    target_ranks = _flatten_stage_tp(stage_ranks, draft_tp, draft_tp + target_tp)
    role = _compute_role_from_ranks(rank, draft_ranks, target_ranks)

    group_list = build_pearl_group_ranks(
        world_size, dp, pp, pcp, tp, draft_tp, target_tp
    )
    draft_group = None
    target_group = None
    verify_group = None
    group_idx = 0
    for outer in range(all_ranks.shape[0]):
        for dp_idx_iter in range(dp):
            for pcp_idx_iter in range(pcp):
                draft_ranks_iter, target_ranks_iter = group_list[group_idx]
                group_idx += 1
                # These calls must be executed by all ranks in the default process group.
                draft_pg = dist.new_group(draft_ranks_iter)
                target_pg = dist.new_group(target_ranks_iter)
                verify_pg = dist.new_group(
                    [draft_ranks_iter[0]] + target_ranks_iter
                )
                if (outer, dp_idx_iter, pcp_idx_iter) == (
                    outer_idx,
                    dp_idx,
                    pcp_idx,
                ):
                    draft_group = draft_pg
                    target_group = target_pg
                    verify_group = verify_pg

    assert draft_group is not None
    assert target_group is not None
    assert verify_group is not None
    return PearlGroups(
        role=role,
        draft_group=draft_group,
        target_group=target_group,
        verify_group=verify_group,
    )


def pack_pearl_proposals(
    req_ids: list[str],
    proposals: dict[str, list[int]],
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Pack proposal tokens into a dense tensor for broadcast."""
    max_len = max(max_len, 1)
    packed = torch.full(
        (len(req_ids), max_len),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )
    for row_idx, req_id in enumerate(req_ids):
        tokens = proposals.get(req_id)
        if not tokens:
            continue
        length = min(len(tokens), max_len)
        packed[row_idx, :length] = torch.tensor(
            tokens[:length], dtype=torch.int32, device=device
        )
    return packed


def unpack_pearl_proposals(
    req_ids: list[str], packed: torch.Tensor
) -> dict[str, list[int]]:
    """Unpack proposal tokens from a dense tensor."""
    proposals: dict[str, list[int]] = {}
    if packed.numel() == 0:
        return proposals
    rows = packed.detach().cpu().tolist()
    for req_id, row in zip(req_ids, rows):
        tokens = [int(tok) for tok in row if tok != PLACEHOLDER_TOKEN_ID]
        if tokens:
            proposals[req_id] = tokens
    return proposals


def pack_pearl_proposal_logprobs(
    req_ids: list[str],
    proposals: dict[str, list[int]],
    proposal_logprobs: dict[str, list[tuple[np.ndarray, np.ndarray, int | None]]],
    max_len: int,
    max_num_logprobs: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack proposal tokens + logprobs into dense tensors for broadcast."""
    max_len = max(max_len, 1)
    logprob_width = max(max_num_logprobs, 0) + 1
    packed_tokens = torch.full(
        (len(req_ids), max_len),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )
    packed_logprob_token_ids = torch.full(
        (len(req_ids), max_len, logprob_width),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )
    packed_logprobs = torch.zeros(
        (len(req_ids), max_len, logprob_width),
        dtype=torch.float32,
        device=device,
    )
    packed_ranks = torch.full(
        (len(req_ids), max_len),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for row_idx, req_id in enumerate(req_ids):
        tokens = proposals.get(req_id) or []
        rows = proposal_logprobs.get(req_id) or []
        limit = min(len(tokens), max_len)
        for tok_idx in range(limit):
            packed_tokens[row_idx, tok_idx] = int(tokens[tok_idx])
            if tok_idx >= len(rows):
                continue
            token_ids, logprobs, rank = rows[tok_idx]
            token_ids_tensor = torch.as_tensor(
                token_ids[:logprob_width], dtype=torch.int32, device=device
            )
            logprobs_tensor = torch.as_tensor(
                logprobs[:logprob_width], dtype=torch.float32, device=device
            )
            packed_logprob_token_ids[
                row_idx, tok_idx, : token_ids_tensor.numel()
            ] = token_ids_tensor
            packed_logprobs[row_idx, tok_idx, : logprobs_tensor.numel()] = (
                logprobs_tensor
            )
            if rank is not None:
                packed_ranks[row_idx, tok_idx] = int(rank)
    return packed_tokens, packed_logprob_token_ids, packed_logprobs, packed_ranks


def unpack_pearl_proposal_logprobs(
    req_ids: list[str],
    packed_tokens: torch.Tensor,
    packed_logprob_token_ids: torch.Tensor,
    packed_logprobs: torch.Tensor,
    packed_ranks: torch.Tensor,
) -> tuple[
    dict[str, list[int]],
    dict[str, list[tuple[np.ndarray, np.ndarray, int | None]]],
]:
    """Unpack proposal tokens + logprobs from dense tensors."""
    proposals: dict[str, list[int]] = {}
    proposal_logprobs: dict[str, list[tuple[np.ndarray, np.ndarray, int | None]]] = {}
    if packed_tokens.numel() == 0:
        return proposals, proposal_logprobs
    tokens_rows = packed_tokens.detach().cpu().tolist()
    logprob_token_ids = packed_logprob_token_ids.detach().cpu().numpy()
    logprobs = packed_logprobs.detach().cpu().numpy()
    ranks = packed_ranks.detach().cpu().tolist()
    for row_idx, req_id in enumerate(req_ids):
        tokens: list[int] = []
        rows: list[tuple[np.ndarray, np.ndarray, int | None]] = []
        for tok_idx, token in enumerate(tokens_rows[row_idx]):
            if token == PLACEHOLDER_TOKEN_ID:
                continue
            tokens.append(int(token))
            rank_value = int(ranks[row_idx][tok_idx])
            rows.append(
                (
                    logprob_token_ids[row_idx, tok_idx].astype(np.int32),
                    logprobs[row_idx, tok_idx].astype(np.float32),
                    None if rank_value < 0 else rank_value,
                )
            )
        if tokens:
            proposals[req_id] = tokens
            proposal_logprobs[req_id] = rows
    return proposals, proposal_logprobs
