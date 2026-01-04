# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PEARL scheduler and KV block manager (logical-level, CPU-friendly).

This module models the PEARL scheduling/rollback logic independently from the
GPU runner to allow unit testing without distributed setup. The logical block
manager mirrors the behavior needed to build block tables and perform KV
rollbacks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, List, Sequence, Tuple


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class PearlSequence:
    """Minimal per-sequence state for scheduler/KV bookkeeping."""

    seq_id: int
    token_ids: List[int]
    max_tokens: int
    eos: int | List[int]
    block_size: int = 256
    ignore_eos: bool = False
    pre_verify: bool = True
    num_cached_tokens: int = 0
    block_table: List[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    _prompt_len: int | None = None

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.token_ids)

    @property
    def num_blocks(self) -> int:
        return (len(self.token_ids) + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        if not self.block_table:
            return 0
        return len(self.token_ids) - (len(self.block_table) - 1) * self.block_size

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    def rollback_tokens(self, n: int) -> None:
        if n <= 0 or n > len(self.token_ids):
            raise ValueError("Invalid rollback length.")
        del self.token_ids[-n:]

    @property
    def block_size(self) -> int:
        return self._block_size

    @block_size.setter
    def block_size(self, value: int) -> None:
        self._block_size = value

    def is_eos(self, token_id: int) -> bool:
        if isinstance(self.eos, list):
            return token_id in self.eos
        return token_id == self.eos

    @property
    def num_completion_tokens(self) -> int:
        prompt_len = self._prompt_len if self._prompt_len is not None else 0
        return max(0, len(self.token_ids) - prompt_len)

    @property
    def num_prompt_tokens(self) -> int:
        return self._prompt_len if self._prompt_len is not None else len(self.token_ids)

    def __post_init__(self) -> None:
        if self._prompt_len is None:
            self._prompt_len = len(self.token_ids)


class LogicalBlock:
    __slots__ = ("block_id", "ref_count", "key", "token_ids")

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_count = 0
        self.key: Tuple[int, ...] | None = None
        self.token_ids: Tuple[int, ...] | None = None

    def reset(self) -> None:
        self.ref_count = 1
        self.key = None
        self.token_ids = None


class LogicalBlockManager:
    """Logical KV block manager with hash-based prefix reuse and rollback."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.block_size = block_size
        self.blocks: List[LogicalBlock] = [LogicalBlock(i) for i in range(num_blocks)]
        self.free_ids: Deque[int] = deque(range(num_blocks))
        self.used_ids: set[int] = set()
        self.key_to_block: Dict[Tuple[int, ...], int] = {}

    def _reserve_block(self, block_id: int) -> LogicalBlock:
        block = self.blocks[block_id]
        if block.ref_count != 0:
            raise RuntimeError("Block already in use.")
        block.reset()
        self.free_ids.remove(block_id)
        self.used_ids.add(block_id)
        return block

    def _release_block(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block.ref_count != 0:
            raise RuntimeError("Block still referenced.")
        self.used_ids.remove(block_id)
        self.free_ids.append(block_id)

    def can_allocate(self, seq: PearlSequence) -> bool:
        return len(self.free_ids) >= seq.num_blocks

    def _block_key(self, token_ids: Sequence[int], prefix_key: Tuple[int, ...] | None) -> Tuple[int, ...] | None:
        if len(token_ids) < self.block_size:
            return None
        return (0,) if prefix_key is None else prefix_key + (0,) + tuple(token_ids)

    def allocate(self, seq: PearlSequence) -> None:
        if seq.block_table:
            raise RuntimeError("Sequence already allocated.")
        prefix_key: Tuple[int, ...] | None = None
        for i in range(seq.num_blocks):
            start = i * self.block_size
            end = min(len(seq.token_ids), start + self.block_size)
            tokens = seq.token_ids[start:end]
            key = self._block_key(tokens, prefix_key)
            prefix_key = key
            block_id = self.key_to_block.get(key, -1) if key is not None else -1
            if block_id != -1 and self.blocks[block_id].token_ids == tuple(tokens):
                block = self.blocks[block_id]
                block.ref_count += 1
            else:
                if not self.free_ids:
                    raise RuntimeError("Out of blocks.")
                block_id = self.free_ids[0]
                block = self._reserve_block(block_id)
                block.token_ids = tuple(tokens)
                if key is not None:
                    block.key = key
                    self.key_to_block[key] = block_id
            seq.block_table.append(block_id)
            if key is not None:
                seq.num_cached_tokens += self.block_size

    def can_append(self, seq: PearlSequence) -> bool:
        # Need a free block when the next token will start a new block.
        need = 1 if len(seq.token_ids) % self.block_size == 0 else 0
        return len(self.free_ids) >= need

    def may_append(self, seq: PearlSequence) -> None:
        # Called before appending the next token; allocate new block when
        # current length is exactly at a block boundary (next token will start
        # a fresh block).
        if len(seq.token_ids) % self.block_size != 0:
            return
        if not self.free_ids:
            raise RuntimeError("Out of blocks for append.")
        block_id = self.free_ids[0]
        block = self._reserve_block(block_id)
        block.token_ids = None
        seq.block_table.append(block_id)

        # If previous block is now full, store its key for prefix reuse.
        if len(seq.block_table) >= 2:
            prev_block_id = seq.block_table[-2]
            prev_tokens = tuple(
                seq.token_ids[-self.block_size :]
            )  # previous full block tokens
            prefix_key = (
                self.blocks[seq.block_table[-3]].key if len(seq.block_table) >= 3 else None
            )
            key = self._block_key(prev_tokens, prefix_key)
            if key is not None:
                self.blocks[prev_block_id].key = key
                self.blocks[prev_block_id].token_ids = prev_tokens
                self.key_to_block[key] = prev_block_id
            seq.num_cached_tokens += self.block_size

    def rollback(self, seq: PearlSequence, n: int) -> None:
        if n <= 0:
            return
        before_blocks = len(seq.block_table)
        seq.rollback_tokens(n)
        after_blocks = (len(seq.token_ids) + self.block_size - 1) // self.block_size
        if after_blocks < before_blocks:
            # Release tail blocks.
            for block_id in seq.block_table[after_blocks:]:
                block = self.blocks[block_id]
                block.ref_count -= 1
                if block.ref_count == 0:
                    self._release_block(block_id)
            seq.block_table = seq.block_table[:after_blocks]
            released = before_blocks - after_blocks
            seq.num_cached_tokens = max(0, seq.num_cached_tokens - released * self.block_size)

    def deallocate(self, seq: PearlSequence) -> None:
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._release_block(block_id)
        seq.block_table.clear()
        seq.num_cached_tokens = 0


class PearlScheduler:
    """PEARL scheduler with prefill/decode, preemption, and rollback hook."""

    def __init__(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        block_manager: LogicalBlockManager,
    ) -> None:
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.block_manager = block_manager
        self.waiting: Deque[PearlSequence] = deque()
        self.running: Deque[PearlSequence] = deque()
        self.finished: List[PearlSequence] = []

    def add(self, seq: PearlSequence) -> None:
        self.waiting.append(seq)

    def is_finished(self) -> bool:  # pragma: no cover - trivial
        return not self.waiting and not self.running

    def _prefill(self) -> tuple[List[PearlSequence], bool]:
        scheduled: List[PearlSequence] = []
        num_seqs = 0
        num_tokens = 0
        while (
            self.waiting
            and num_seqs < self.max_num_seqs
            and num_tokens < self.max_num_batched_tokens
        ):
            seq = self.waiting[0]
            if (
                num_tokens + len(seq.token_ids) > self.max_num_batched_tokens
                or not self.block_manager.can_allocate(seq)
            ):
                break
            self.waiting.popleft()
            self.block_manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            scheduled.append(seq)
            num_seqs += 1
            num_tokens += len(seq.token_ids) - seq.num_cached_tokens
        if scheduled:
            return scheduled, True
        return [], False

    def _decode(self) -> tuple[List[PearlSequence], bool]:
        scheduled: List[PearlSequence] = []
        num_seqs = 0
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    victim = self.running.pop()
                    self.preempt(victim)
                else:
                    self.preempt(seq)
                    break
            else:
                self.block_manager.may_append(seq)
                scheduled.append(seq)
                num_seqs += 1
        if not scheduled and self.waiting:
            # Try to resume prefill if decode has no schedulable seqs.
            return self._prefill()
        self.running.extendleft(reversed(scheduled))
        return scheduled, False

    def schedule(self) -> tuple[List[PearlSequence], bool]:
        # Prefill takes precedence if waiting queue not empty and running empty
        # or unable to decode.
        if self.waiting:
            scheduled, is_prefill = self._prefill()
            if scheduled:
                return scheduled, is_prefill
        return self._decode()

    def preempt(self, seq: PearlSequence) -> None:
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self, seqs: Sequence[PearlSequence], token_ids: Sequence[int]
    ) -> List[bool]:
        finished = []
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            # EOS/max_tokens check
            should_finish = (not seq.ignore_eos and seq.is_eos(token_id)) or (
                seq.num_completion_tokens >= seq.max_tokens
            )
            finished.append(should_finish)
            if should_finish:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)
                self.finished.append(seq)
        return finished

    def rollback(self, seq: PearlSequence, n: int) -> None:
        self.block_manager.rollback(seq, n)

    def clear(self) -> None:  # pragma: no cover - defensive
        for q in (self.waiting, self.running, self.finished):
            while q:
                seq = q.pop()
                self.block_manager.deallocate(seq)
        self.block_manager.key_to_block.clear()
        for block in self.block_manager.blocks:
            block.ref_count = 0
            block.key = None
            block.token_ids = None
        self.block_manager.free_ids = deque(range(len(self.block_manager.blocks)))
        self.block_manager.used_ids.clear()
