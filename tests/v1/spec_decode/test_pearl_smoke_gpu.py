# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.worker.block_table import BlockTable


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for PEARL GPU smoke"
)


def test_pearl_context_lens_smoke_gpu():
    """Ensure context_lens is carried through CommonAttentionMetadata on GPU."""
    bt = BlockTable(
        block_size=2,
        max_num_reqs=1,
        max_num_blocks_per_req=4,
        max_num_batched_tokens=8,
        pin_memory=False,
        device=torch.device("cuda"),
        kernel_block_size=2,
        cp_kv_cache_interleave_size=1,
    )
    bt.add_row([1, 2], row_idx=0)
    positions = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int64)
    req_indices = torch.zeros_like(positions)
    slot_mapping = bt.build_slot_mapping(req_indices, positions).to("cuda")
    bt.slot_mapping.gpu[: positions.numel()] = slot_mapping
    query_start_loc = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([3], device="cuda", dtype=torch.int32)
    ctx = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32)

    cm = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=seq_lens,
        _seq_lens_cpu=seq_lens.cpu(),
        _num_computed_tokens_cpu=torch.tensor([0], device="cuda", dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=3,
        max_query_len=3,
        max_seq_len=3,
        block_table_tensor=bt.block_table.gpu[:1],
        slot_mapping=slot_mapping,
        causal=True,
        context_lens=ctx,
    )
    assert torch.equal(cm.context_lens[:3], ctx)
