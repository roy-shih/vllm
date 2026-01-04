# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm.v1.spec_decode.pearl_dist import (
    build_pearl_group_ranks,
    compute_pearl_role,
    pack_pearl_proposal_logprobs,
    pack_pearl_proposals,
    unpack_pearl_proposals,
    unpack_pearl_proposal_logprobs,
)


def test_compute_pearl_role_partition():
    role0 = compute_pearl_role(global_rank=0, draft_tp=2, target_tp=2)
    role2 = compute_pearl_role(global_rank=2, draft_tp=2, target_tp=2)

    assert role0.is_draft is True
    assert role0.local_rank == 0
    assert role0.draft_ranks == [0, 1]
    assert role0.target_ranks == [2, 3]

    assert role2.is_draft is False
    assert role2.local_rank == 0
    assert role2.draft_ranks == [0, 1]
    assert role2.target_ranks == [2, 3]


def test_compute_pearl_role_raises_on_mismatch():
    try:
        compute_pearl_role(global_rank=4, draft_tp=2, target_tp=2)
    except ValueError as e:
        assert "Rank 4 not in draft" in str(e)
    else:
        assert False, "Expected ValueError for out-of-range rank"


def test_pack_unpack_pearl_proposals():
    req_ids = ["req0", "req1", "req2"]
    proposals = {"req0": [10, 11], "req2": [12]}

    packed = pack_pearl_proposals(
        req_ids,
        proposals,
        max_len=2,
        device=torch.device("cpu"),
    )
    assert packed.shape == (3, 2)
    assert packed.tolist() == [
        [10, 11],
        [PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID],
        [12, PLACEHOLDER_TOKEN_ID],
    ]

    unpacked = unpack_pearl_proposals(req_ids, packed)
    assert unpacked == {"req0": [10, 11], "req2": [12]}


def test_pack_unpack_pearl_proposal_logprobs():
    req_ids = ["req0", "req1"]
    proposals = {"req0": [3, 4]}
    proposal_logprobs = {
        "req0": [
            (
                np.array([3, 1], dtype=np.int32),
                np.array([-0.1, -0.2], dtype=np.float32),
                0,
            ),
            (
                np.array([4, 2], dtype=np.int32),
                np.array([-0.3, -0.4], dtype=np.float32),
                None,
            ),
        ]
    }
    packed = pack_pearl_proposal_logprobs(
        req_ids,
        proposals,
        proposal_logprobs,
        max_len=2,
        max_num_logprobs=1,
        device=torch.device("cpu"),
    )
    packed_tokens, packed_token_ids, packed_logprobs, packed_ranks = packed
    assert packed_tokens.tolist() == [
        [3, 4],
        [PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID],
    ]
    assert packed_token_ids.shape == (2, 2, 2)
    assert packed_logprobs.shape == (2, 2, 2)
    assert packed_ranks.tolist() == [[0, -1], [-1, -1]]

    unpacked_tokens, unpacked_logprobs = unpack_pearl_proposal_logprobs(
        req_ids,
        packed_tokens,
        packed_token_ids,
        packed_logprobs,
        packed_ranks,
    )
    assert unpacked_tokens == {"req0": [3, 4]}
    assert "req1" not in unpacked_logprobs
    rows = unpacked_logprobs["req0"]
    np.testing.assert_array_equal(rows[0][0], np.array([3, 1], dtype=np.int32))
    np.testing.assert_allclose(
        rows[0][1], np.array([-0.1, -0.2], dtype=np.float32), rtol=0, atol=1e-6
    )
    assert rows[0][2] == 0
    np.testing.assert_array_equal(rows[1][0], np.array([4, 2], dtype=np.int32))
    np.testing.assert_allclose(
        rows[1][1], np.array([-0.3, -0.4], dtype=np.float32), rtol=0, atol=1e-6
    )
    assert rows[1][2] is None


def test_pearl_group_ranks_with_pp_dp():
    dp = 2
    pp = 2
    pcp = 1
    tp = 4
    draft_tp = 1
    target_tp = 3
    world_size = dp * pp * pcp * tp

    all_ranks = torch.arange(world_size).reshape(dp, pp, pcp, tp)
    groups = build_pearl_group_ranks(
        world_size, dp, pp, pcp, tp, draft_tp, target_tp
    )
    assert len(groups) == dp * pcp

    for dp_idx in range(dp):
        stage_ranks = all_ranks[dp_idx, :, 0, :]
        expected_draft_leader = int(stage_ranks[pp - 1, 0].item())
        expected_target_leader = int(stage_ranks[pp - 1, draft_tp].item())
        draft_ranks, target_ranks = groups[dp_idx]

        assert len(draft_ranks) == pp * draft_tp
        assert len(target_ranks) == pp * target_tp
        assert draft_ranks[0] == expected_draft_leader
        assert target_ranks[0] == expected_target_leader

        combined = set(draft_ranks + target_ranks)
        assert combined == set(stage_ranks.flatten().tolist())
