# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import torch

from vllm.sequence import IntermediateTensors
from vllm.v1.spec_decode.pearl import PearlDraftRunner, PearlSequenceState
from vllm.v1.spec_decode.pearl_dist import PearlGroups, PearlRole
from vllm.v1.spec_decode.pearl_scheduler import (
    LogicalBlockManager,
    PearlScheduler,
    PearlSequence,
)
from vllm.v1.spec_decode.pearl_target import (
    TargetVerifier,
    TargetVerifyResult,
    verify_on_target,
)
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.pearl_sampler import PearlRejectionSampler
from vllm.v1.worker import gpu_model_runner as gpu_model_runner_module
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def test_draft_runner_packages_verification():
    seqs = [
        PearlSequenceState(seq_id=0, token_ids=[1, 2, 3]),
        PearlSequenceState(seq_id=1, token_ids=[5, 6, 7]),
    ]

    def run_model_fn(input_ids, positions):
        # Echo back incremented ids to keep deterministic.
        vocab = 10
        logits = torch.zeros(input_ids.size(0), vocab)
        for i, tok in enumerate(input_ids.tolist()):
            logits[i, (tok + 1) % vocab] = 1.0
        return logits

    runner = PearlDraftRunner(gamma=2, device="cpu")
    payload = runner.pearl_step(seqs, run_model_fn)

    # Each seq should append 2 tokens.
    assert seqs[0].token_ids[-2:] == [4, 5]
    assert seqs[1].token_ids[-2:] == [8, 9]

    # pre_verify => verify 1 token per seq.
    assert payload.verify_counts == [1, 1]
    assert payload.to_verify_tokens == [3, 7]
    # Next round tokens are the drafted gamma window.
    assert payload.next_round_tokens_per_seq == [[4, 5], [8, 9]]


def test_scheduler_prefill_decode_and_preempt():
    # Use enough blocks so two sequences can prefill.
    block_mgr = LogicalBlockManager(num_blocks=4, block_size=2)
    sched = PearlScheduler(
        max_num_seqs=2, max_num_batched_tokens=8, block_manager=block_mgr
    )
    seq_a = PearlSequence(
        seq_id=0, token_ids=[1, 2, 3], max_tokens=10, eos=0, block_size=2
    )
    seq_b = PearlSequence(
        seq_id=1, token_ids=[4, 5], max_tokens=10, eos=0, block_size=2
    )
    seq_c = PearlSequence(
        seq_id=2, token_ids=[6, 7, 8], max_tokens=10, eos=0, block_size=2
    )

    for seq in (seq_a, seq_b, seq_c):
        sched.add(seq)

    # Prefill should schedule two sequences (block capacity = 2).
    scheduled, is_prefill = sched.schedule()
    assert is_prefill
    assert {s.seq_id for s in scheduled} == {0, 1}

    # Decode phase: only running seqs remain; can_append passes for seq 0; seq 1 has full block; no preempt.
    scheduled, is_prefill = sched.schedule()
    assert not is_prefill
    assert {s.seq_id for s in scheduled} == {0, 1}

    # Simulate append at block boundary to trigger preempt when capacity tight.
    block_mgr.free_ids.clear()  # exhaust capacity
    scheduled, _ = sched.schedule()
    # seq 1 should be preempted back to waiting due to lack of capacity.
    assert seq_b in sched.waiting


def test_target_verify_accept_and_reject():
    seqs = [
        PearlSequenceState(seq_id=0, token_ids=[1, 2, 3]),
        PearlSequenceState(seq_id=1, token_ids=[4, 5, 6]),
    ]
    payload = PearlDraftRunner(gamma=2, device="cpu").pearl_step(
        seqs,
        lambda ids, pos: torch.eye(10)[(ids + 1) % 10],  # deterministic argmax = tok+1
    )

    # Build logits to accept first seq (matches) and reject first token of second seq.
    logits = torch.full((len(payload.to_verify_tokens), 10), -1.0)
    # seq 0: to_verify token is 3 -> set highest logit at 3 to accept
    logits[0, payload.to_verify_tokens[0]] = 2.0
    # seq 1: to_verify token is original 6+? actually to_verify tokens stored pre-step: 6+? we don't care, force reject
    logits[1, :] = -1.0
    logits[1, 0] = 3.0  # argmax != to_verify

    res: TargetVerifyResult = verify_on_target(
        seqs=seqs,
        payload=payload,
        logits=logits,
        gamma=2,
        eos_token_id=9,
        max_tokens=16,
    )

    assert res.acc[0] is True
    assert res.rollout[0] == 0
    assert res.acc[1] is False
    assert res.rollout[1] == 2  # full gamma window
    assert res.revise_token[1] == 0


def test_target_verifier_accept_and_reject_with_sampling():
    torch.manual_seed(0)
    seqs = [
        PearlSequenceState(seq_id=0, token_ids=[1, 2, 3]),
        PearlSequenceState(seq_id=1, token_ids=[4, 5, 6]),
    ]
    payload = PearlDraftRunner(gamma=2, device="cpu").pearl_step(
        seqs,
        lambda ids, pos: torch.eye(10)[(ids + 1) % 10],
    )
    # Build logits: accept seq0 token, reject seq1 token with high prob on alt token.
    logits = torch.full((len(payload.to_verify_tokens), 10), -1.0)
    logits[0, payload.to_verify_tokens[0]] = 5.0  # strong accept
    logits[1, 0] = 5.0  # force reject (to_verify is different)

    verifier = TargetVerifier(gamma=2, eos_token_id=9, max_tokens=16)
    res = verifier.verify_and_update(seqs, payload, logits)

    assert res.acc == [True, False]
    # After accept, seq0 keeps drafted tokens only (no double-append).
    assert len(seqs[0].token_ids) == 5
    assert seqs[0].token_ids == [1, 2, 3, 4, 5]
    # After reject, seq1 should have rolled back gamma and inserted revised token => len restored to prompt+1
    assert len(seqs[1].token_ids) == 4


def test_pearl_logprobs_alignment():
    vocab_size = 5
    req_ids = ["req0", "req1"]

    sampling_metadata = SamplingMetadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=2,
        no_penalties=True,
        prompt_token_ids=torch.empty(0, dtype=torch.int32),
        frequency_penalties=torch.tensor([], dtype=torch.float32),
        presence_penalties=torch.tensor([], dtype=torch.float32),
        repetition_penalties=torch.tensor([], dtype=torch.float32),
        output_token_ids=[[] for _ in req_ids],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        spec_token_ids=[[] for _ in req_ids],
    )

    input_batch = SimpleNamespace(
        req_ids=req_ids,
        sampling_metadata=sampling_metadata,
        vocab_size=vocab_size,
    )

    sampler = Sampler(logprobs_mode="raw_logprobs")
    rejection_sampler = PearlRejectionSampler(sampler)

    class FakeRunner:
        def __init__(self):
            self.input_batch = input_batch
            self.sampler = sampler
            self.rejection_sampler = rejection_sampler
            self.pearl_pending_proposals = {
                "req0": [3, 0],
                "req1": [2],
            }
            self.pearl_pending_proposal_logprobs = {
                "req0": [
                    (
                        np.array([3, 1, 2], dtype=np.int32),
                        np.array([-0.1, -0.2, -0.3], dtype=np.float32),
                        0,
                    ),
                    (
                        np.array([0, 2, 3], dtype=np.int32),
                        np.array([-0.05, -0.2, -0.4], dtype=np.float32),
                        0,
                    ),
                ]
            }

        def _update_states_after_model_execute(self, sampled_token_ids):
            _ = sampled_token_ids

    runner = FakeRunner()

    draft_token_ids = torch.tensor([2, 1, 4], dtype=torch.int32)
    spec_decode_metadata = SpecDecodeMetadata(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=[1, 2],
        cu_num_draft_tokens=torch.tensor([1, 3], dtype=torch.int32),
        cu_num_sampled_tokens=torch.tensor([2, 5], dtype=torch.int32),
        target_logits_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
        bonus_logits_indices=torch.tensor([3, 4], dtype=torch.int32),
        logits_indices=torch.arange(5, dtype=torch.int32),
        pre_verify=[True, False],
    )

    logits = torch.tensor(
        [
            [0.0, 0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 3.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    sampler_output, _, _, _, _ = GPUModelRunner._sample_pearl(
        runner, None, logits, spec_decode_metadata
    )
    sampled_ids, cu_num_tokens = RejectionSampler.parse_output(
        sampler_output.sampled_token_ids,
        vocab_size,
        return_cu_num_tokens=True,
    )

    assert sampled_ids == [[3, 0], [1]]


def test_pearl_pp_intermediate_sync(monkeypatch):
    class FakePPGroup:
        def __init__(self) -> None:
            self.is_first_rank = False
            self.is_last_rank = False
            self.recv_calls = 0
            self.sent = None

        def recv_tensor_dict(self, *args, **kwargs):
            self.recv_calls += 1
            return {"hidden": torch.zeros(1)}

        def send_tensor_dict(self, tensors, *args, **kwargs):
            self.sent = tensors

    pp_group = FakePPGroup()
    monkeypatch.setattr(
        gpu_model_runner_module, "get_pp_group", lambda: pp_group
    )
    monkeypatch.setattr(
        gpu_model_runner_module, "get_tp_group", lambda: SimpleNamespace()
    )

    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        broadcast_pp_output=False,
    )

    received = GPUModelRunner._pearl_pp_recv(runner, None, {})
    assert pp_group.recv_calls == 1
    assert isinstance(received, IntermediateTensors)

    output = IntermediateTensors({"hidden": torch.ones(1)})
    GPUModelRunner._pearl_pp_send(runner, output, {})
    assert pp_group.sent == output.tensors


def test_pearl_sync_proposals_broadcasts_to_draft_group(monkeypatch):
    broadcast_calls = []

    def fake_broadcast(tensor, src, group):
        broadcast_calls.append((src, group))

    monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)

    role = PearlRole(
        draft_ranks=[1, 0],
        target_ranks=[2],
        is_draft=True,
        local_rank=1,
        global_rank=0,
    )
    draft_group = object()
    verify_group = object()
    pearl_groups = PearlGroups(
        role=role,
        draft_group=draft_group,
        target_group=object(),
        verify_group=verify_group,
    )

    runner = SimpleNamespace(
        pearl_groups=pearl_groups,
        input_batch=SimpleNamespace(
            req_ids=["req0"],
            sampling_metadata=SimpleNamespace(max_num_logprobs=None),
        ),
        device=torch.device("cpu"),
        _pearl_get_gamma=lambda: 2,
        _pearl_barrier=lambda: None,
    )

    proposals, _ = GPUModelRunner._pearl_sync_proposals(
        runner,
        {"req0": [1, 2]},
        None,
    )

    assert proposals["req0"] == [1, 2]
    assert broadcast_calls == [(role.draft_ranks[0], draft_group)]


def test_pearl_bonus_logits_processors() -> None:
    vocab_size = 4
    req_ids = ["req0"]
    allowed_mask = torch.zeros((1, vocab_size), dtype=torch.bool)
    allowed_mask[0, 0] = True

    sampling_metadata = SamplingMetadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=torch.empty(0, dtype=torch.int32),
        frequency_penalties=torch.tensor([], dtype=torch.float32),
        presence_penalties=torch.tensor([], dtype=torch.float32),
        repetition_penalties=torch.tensor([], dtype=torch.float32),
        output_token_ids=[[]],
        allowed_token_ids_mask=allowed_mask,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        spec_token_ids=[[]],
    )

    input_batch = SimpleNamespace(
        req_ids=req_ids,
        sampling_metadata=sampling_metadata,
        vocab_size=vocab_size,
    )

    sampler = Sampler(logprobs_mode="raw_logprobs")
    rejection_sampler = PearlRejectionSampler(sampler)

    class FakeRunner:
        def __init__(self):
            self.input_batch = input_batch
            self.sampler = sampler
            self.rejection_sampler = rejection_sampler
            self.pearl_pending_proposals = {"req0": [0]}
            self.pearl_pending_proposal_logprobs = {}

        def _update_states_after_model_execute(self, sampled_token_ids):
            _ = sampled_token_ids

    runner = FakeRunner()

    spec_decode_metadata = SpecDecodeMetadata(
        draft_token_ids=torch.tensor([1], dtype=torch.int32),
        num_draft_tokens=[1],
        cu_num_draft_tokens=torch.tensor([1], dtype=torch.int32),
        cu_num_sampled_tokens=torch.tensor([2], dtype=torch.int32),
        target_logits_indices=torch.tensor([0], dtype=torch.int32),
        bonus_logits_indices=torch.tensor([1], dtype=torch.int32),
        logits_indices=torch.tensor([0, 1], dtype=torch.int32),
        pre_verify=[False],
    )

    logits = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [5.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    sampler_output, verify_result, _, _, _ = GPUModelRunner._sample_pearl(
        runner, None, logits, spec_decode_metadata
    )
    sampled_token = int(sampler_output.sampled_token_ids[0, 0].item())
    assert sampled_token == 1
    assert verify_result["req0"]["acc"] is False


def test_pearl_compute_auto_gamma() -> None:
    gamma = GPUModelRunner._pearl_compute_auto_gamma(
        draft_time=1.0,
        draft_tokens=8.0,
        target_time=2.0,
        target_tokens=4.0,
        max_gamma=8,
    )
    assert gamma == 4
    gamma = GPUModelRunner._pearl_compute_auto_gamma(
        draft_time=1.0,
        draft_tokens=8.0,
        target_time=2.0,
        target_tokens=4.0,
        max_gamma=2,
    )
    assert gamma == 2
    gamma = GPUModelRunner._pearl_compute_auto_gamma(
        draft_time=0.0,
        draft_tokens=8.0,
        target_time=1.0,
        target_tokens=4.0,
        max_gamma=8,
    )
    assert gamma is None
