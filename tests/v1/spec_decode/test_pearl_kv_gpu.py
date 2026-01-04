# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch

from vllm.v1.worker.block_table import BlockTable
from vllm.v1.spec_decode.pearl_kv import PearlKVAdapter


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for GPU smoke test"
)


def test_block_table_rollback_and_slot_mapping_gpu():
    bt = BlockTable(
        block_size=2,
        max_num_reqs=1,
        max_num_blocks_per_req=4,
        max_num_batched_tokens=16,
        pin_memory=False,
        device=torch.device("cuda"),
        kernel_block_size=2,
        cp_kv_cache_interleave_size=1,
    )
    # Populate a row with two blocks
    bt.add_row([10, 11], row_idx=0)
    positions = np.array([0, 1, 2, 3], dtype=np.int64)
    req_indices = np.zeros_like(positions)
    # Build slot mapping and ensure it uses both blocks
    slot_mapping = bt.build_slot_mapping(req_indices, positions)
    assert torch.equal(
        slot_mapping.cpu(), torch.tensor([20, 21, 22, 23], dtype=torch.int64)
    )
    # Roll back to 1 block (<=2 tokens)
    bt.rollback_row_to_num_tokens(0, num_tokens=2)
    assert bt.num_blocks_per_row[0] == 1
    positions2 = np.array([0, 1], dtype=np.int64)
    slot_mapping2 = bt.build_slot_mapping(req_indices[:2], positions2)
    assert torch.equal(slot_mapping2.cpu(), torch.tensor([20, 21], dtype=torch.int64))


def test_pearl_kv_adapter_multi_req_gpu():
    bt = BlockTable(
        block_size=2,
        max_num_reqs=2,
        max_num_blocks_per_req=4,
        max_num_batched_tokens=16,
        pin_memory=False,
        device=torch.device("cuda"),
        kernel_block_size=2,
        cp_kv_cache_interleave_size=1,
    )
    adapter = PearlKVAdapter(bt)
    adapter.set_logical_blocks(0, [1, 2])
    adapter.set_logical_blocks(1, [3, 4])
    req_indices = [0, 0, 1, 1]
    positions = [0, 1, 0, 1]
    slots = adapter.build_slot_mapping_for_batch(req_indices, positions)
    assert torch.equal(slots.cpu(), torch.tensor([2, 3, 6, 7], dtype=torch.int64))
    context_lens = adapter.build_context_lens_for_batch(
        seq_lens=[4, 4], window_sizes=[2, 2], slot_mapping=slots
    )
    assert context_lens.tolist() == [3, 4, 3, 4]
    masked_slots = slots.clone()
    masked_slots[1] = -1
    masked_context = adapter.build_context_lens_for_batch(
        seq_lens=[4, 4], window_sizes=[2, 2], slot_mapping=masked_slots
    )
    assert masked_context.tolist() == [3, -1, 3, 4]
