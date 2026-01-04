# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.v1.spec_decode.pearl_kv import (
    PearlKVAdapter,
    context_lens_from_windows,
    slot_mapping_from_logical_blocks,
)
from vllm.v1.worker.block_table import BlockTable


def test_slot_mapping_from_logical_blocks():
    block_table = [5, 6]
    positions = [0, 1, 2, 3]
    slots = slot_mapping_from_logical_blocks(block_table, positions, block_size=2)
    assert slots == [10, 11, 12, 13]


def test_context_lens_from_windows():
    seq_lens = [5, 7]
    window_sizes = [2, 3]
    ctx = context_lens_from_windows(seq_lens, window_sizes)
    assert ctx == [4, 5, 5, 6, 7]


def test_block_table_rollback_row_to_num_blocks():
    bt = BlockTable(
        block_size=2,
        max_num_reqs=2,
        max_num_blocks_per_req=4,
        max_num_batched_tokens=16,
        pin_memory=False,
        device=torch.device("cpu"),
        kernel_block_size=2,
        cp_kv_cache_interleave_size=1,
    )
    bt.add_row([1, 2, 3], row_idx=0)
    assert bt.num_blocks_per_row[0] == 3
    bt.rollback_row_to_num_blocks(0, 1)
    assert bt.num_blocks_per_row[0] == 1
    assert bt.block_table.cpu[0, 1] == 0


def test_pearl_kv_adapter_slot_mapping():
    bt = BlockTable(
        block_size=2,
        max_num_reqs=2,
        max_num_blocks_per_req=4,
        max_num_batched_tokens=16,
        pin_memory=False,
        device=torch.device("cpu"),
        kernel_block_size=2,
        cp_kv_cache_interleave_size=1,
    )
    adapter = PearlKVAdapter(bt)
    adapter.set_logical_blocks(0, [5, 6])
    slots = adapter.build_slot_mapping(0, positions=[0, 1, 2, 3])
    assert slots == [10, 11, 12, 13]
    adapter.rollback_to_tokens(0, num_tokens=2)
    slots2 = adapter.build_slot_mapping(0, positions=[0, 1])
    assert slots2 == [10, 11]


def test_pearl_kv_adapter_context_lens_masking():
    bt = BlockTable(
        block_size=2,
        max_num_reqs=2,
        max_num_blocks_per_req=4,
        max_num_batched_tokens=16,
        pin_memory=False,
        device=torch.device("cpu"),
        kernel_block_size=2,
        cp_kv_cache_interleave_size=1,
    )
    adapter = PearlKVAdapter(bt)
    seq_lens = [5, 7]
    window_sizes = [2, 3]
    slot_mapping = torch.tensor([10, -1, 12, 13, -1], dtype=torch.int64)
    context_lens = adapter.build_context_lens_for_batch(
        seq_lens=seq_lens, window_sizes=window_sizes, slot_mapping=slot_mapping
    )
    assert context_lens.tolist() == [4, -1, 5, 6, -1]


def test_pearl_kv_adapter_slot_mapping_batch():
    bt = BlockTable(
        block_size=2,
        max_num_reqs=2,
        max_num_blocks_per_req=4,
        max_num_batched_tokens=16,
        pin_memory=False,
        device=torch.device("cpu"),
        kernel_block_size=2,
        cp_kv_cache_interleave_size=1,
    )
    adapter = PearlKVAdapter(bt)
    adapter.set_logical_blocks(0, [5, 6])
    adapter.set_logical_blocks(1, [8])
    req_indices = [0, 0, 0, 1, 1]
    positions = [0, 1, 2, 0, 1]
    slots = adapter.build_slot_mapping_for_batch(req_indices, positions)
    assert slots.tolist() == [10, 11, 12, 16, 17]


def test_block_table_build_slot_mapping_multi_req():
    bt = BlockTable(
        block_size=2,
        max_num_reqs=2,
        max_num_blocks_per_req=4,
        max_num_batched_tokens=16,
        pin_memory=False,
        device=torch.device("cpu"),
        kernel_block_size=2,
        cp_kv_cache_interleave_size=1,
    )
    bt.add_row([1, 2], row_idx=0)
    bt.add_row([3, 4], row_idx=1)
    req_indices = [0, 0, 1, 1]
    positions = [0, 1, 0, 1]
    slots = bt.build_slot_mapping(req_indices=req_indices, positions=positions)
    # row 0 uses blocks [1,2]; row 1 uses [3,4]; block_size=2
    assert slots.tolist() == [2, 3, 6, 7]
