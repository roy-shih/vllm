# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PEARL KV helpers for slot mapping and context lens generation."""

from __future__ import annotations

from typing import Iterable, List, Sequence

from vllm.v1.worker.block_table import BlockTable, MultiGroupBlockTable


def slot_mapping_from_logical_blocks(
    block_table: Sequence[int],
    positions: Sequence[int],
    block_size: int,
) -> List[int]:
    """Compute slot mapping for flattened tokens given a logical block table."""
    slots: List[int] = []
    for pos in positions:
        block_idx = pos // block_size
        offset = pos % block_size
        block_id = block_table[block_idx]
        slots.append(block_id * block_size + offset)
    return slots


def context_lens_from_windows(
    seq_lens: Sequence[int],
    window_sizes: Sequence[int],
) -> List[int]:
    """Flattened context_lens for tokens in sliding windows.

    For each sequence with current length L and window size w, emit:
    [L - w + 1, L - w + 2, ..., L]
    """
    out: List[int] = []
    for L, w in zip(seq_lens, window_sizes):
        out.extend(range(L - w + 1, L + 1))
    return out


class PearlKVAdapter:
    """Adapter to map logical block tables into the paged BlockTable."""

    def __init__(self, block_table: BlockTable | MultiGroupBlockTable):
        if isinstance(block_table, MultiGroupBlockTable):
            if len(block_table.block_tables) != 1:
                raise ValueError(
                    "PEARL currently supports a single KV cache group."
                )
            block_table = block_table[0]
        self.block_table = block_table
        self.block_size = block_table.block_size

    def set_logical_blocks(self, row_idx: int, logical_block_ids: Sequence[int]) -> None:
        """Populate a BlockTable row from logical block ids."""
        self.block_table.add_row(list(logical_block_ids), row_idx=row_idx)

    def rollback_to_tokens(self, row_idx: int, num_tokens: int) -> None:
        """Rollback a row to the given number of tokens (ceil to blocks)."""
        self.block_table.rollback_row_to_num_tokens(row_idx, num_tokens)

    def build_slot_mapping(self, row_idx: int, positions: Sequence[int]) -> List[int]:
        """Compute slot mapping for given positions of a single row."""
        req_indices = [row_idx] * len(positions)
        slots = self.block_table.build_slot_mapping(
            req_indices=self._to_np(req_indices), positions=self._to_np(positions)
        )
        return slots.tolist()

    def build_slot_mapping_for_batch(
        self, req_indices: Sequence[int], positions: Sequence[int]
    ) -> "torch.Tensor":
        """Compute slot mapping for a batch of positions across rows."""
        import torch

        slots = self.block_table.build_slot_mapping(
            req_indices=req_indices, positions=positions
        )
        if isinstance(slots, torch.Tensor):
            return slots.to(dtype=torch.int64, copy=False)
        return torch.tensor(slots, dtype=torch.int64)

    def build_context_lens_for_batch(
        self,
        seq_lens: Sequence[int],
        window_sizes: Sequence[int],
        slot_mapping: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """Compute per-token context lengths for flattened PEARL windows."""
        import torch

        context_lens = context_lens_from_windows(seq_lens, window_sizes)
        lens_tensor = torch.tensor(context_lens, dtype=torch.int32)
        if slot_mapping is None:
            return lens_tensor
        if not isinstance(slot_mapping, torch.Tensor):
            slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64)
        if slot_mapping.numel() != lens_tensor.numel():
            raise ValueError("slot_mapping and context_lens size mismatch")
        mask = slot_mapping.cpu() == -1
        if mask.any():
            lens_tensor = lens_tensor.clone()
            lens_tensor[mask] = -1
        return lens_tensor

    @staticmethod
    def _to_np(seq: Sequence[int]):
        import numpy as np

        return np.array(seq, dtype=np.int64)
