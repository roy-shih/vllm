PEARL Integration Plan (vLLM)
=============================

Goal: bring nano-PEARL-style dual-model speculative decoding (parallel draft/target with adaptive gamma, KV rollback, prefix caching) into vLLM without regressing existing speculative paths.

How to use this doc
- Treat the lists below as the live TODO for integration.
- After each meaningful code change, update the relevant checklist item(s) here.
- Keep notes on assumptions, blockers, and test coverage as we go.

Current MVP Scope (What Works Today)
- PEARL runs on the v1 GPU runner path with draft/target rank roles, PEARL NCCL subgroups, and draft proposal broadcast + target verification broadcast.
- Draft proposals are generated with a dedicated draft model (when configured) using gamma-step greedy sampling and then rolled back to keep KV/cache consistent.
- Target verifies draft windows with Bernoulli accept/reject, applies rollback on rejection, and emits per-request verify results (acc/rollout/revise).
- Paged KV rollback hooks exist; freed block IDs are collected and released by the scheduler when available.
- PEARL pre-verify state is tracked per request; acceptance toggles pre-verify for the next step.
- Logprobs are returned for PEARL outputs (see detailed semantics below).

Logprobs Semantics (PEARL)
- Output tokens are the **next-round draft window** when verification succeeds; they are not the same tokens whose target logits were computed in this step.
- Logprobs therefore come from **mixed sources**:
  - Accepted next-round tokens use **draft logprobs** captured during proposal generation.
  - Rejected tokens use **target logprobs** from the reject position (target logits or bonus logits).
  - Requests with no draft tokens use **target bonus logprobs**.
- This preserves **token/logprob alignment** but does **not** yet provide target-aligned logprobs for accepted next-round tokens. See TODOs below.

Limitations / Known Gaps (Detailed)
- **Draft/target overlap is rank-parallel only**: draft and target run on separate rank groups, but there is no intra-rank stream-level pipelining or overlap control.
- **Verify subgroup uses Python object broadcast** for proposals/logprobs/results. This is functional but high-overhead and not optimized for large batches.
- **No DP/PP support** for PEARL: DP/PP coordination is not implemented; mixed DP/PP can desync or deadlock.
- **No true draft/target concurrency within a single worker**: proposal generation and target forward are independent per-rank paths, not an integrated pipeline.
- **Draft KV is not persistent**: draft proposals are generated with rollback each step, which can reduce acceptance when draft != target.
- **No full rejection rollback sync across groups**: local rollback is applied, but cross-rank synchronization is best-effort.
- **Logprobs are draft-based on acceptance** (see semantics above); target-aligned logprobs for accepted windows are not yet available.
- **CUDA graph capture & auto-gamma** are not wired for PEARL.
- **Dynamic TP padding** is not implemented; non-power-of-two TP sizes still require care.
- **Structured output + PEARL** has not been fully validated for reject/rollback correctness.
- **Async scheduling** has basic token broadcast, but deeper scheduling optimizations (e.g., overlap with draft) are still pending.

TODO Roadmap (Detailed)
- Correctness & synchronization
  - [ ] Implement GPU tensor broadcast/reduce for proposals + verification (replace object_list).
  - [ ] Add full rejection rollback synchronization across draft/target groups.
  - [ ] Add DP/PP-aware PEARL coordination (group barriers, output token path, logprobs).
  - [ ] Ensure EOS/max_tokens handling is consistent across pre-verify/post-verify windows.
  - [ ] Validate PEARL with structured output / grammar constraints.
- Logprobs
  - [ ] Provide target-aligned logprobs for accepted next-round tokens (delayed verify or extra target forward).
  - [ ] Add tests for PEARL logprob alignment and mixed accept/reject paths.
- Performance & throughput
  - [ ] Overlap draft/target forward with explicit stream scheduling where possible.
  - [ ] Wire auto-gamma profiling and CUDA graph capture for PEARL decode.
  - [ ] Optimize proposal/logprob broadcast (tensor payloads, packing).
- UX & observability
  - [ ] Add structured metrics for acceptance rate, rollbacks, MAT/throughput counters.
  - [ ] Add a reproducible GPU benchmark script for PEARL throughput/MAT.

Architecture & Integration TODO
- [x] Confirm current speculative wiring in vLLM (config surfaces, worker startup, KV/paged attention flow, existing spec decode entrypoints).
- [x] Define PEARL config surface (draft/target model paths, TP sizes, gamma/auto-gamma, block params, optional dynamic TP padding) and validation rules.
- [x] Choose process/group layout (draft group, target group, verify group) and communication mechanism (NCCL subgroups or existing RPC) for acceptance messages.
- [x] Map KV/block handling: reuse vLLM paged KV cache with rollback support; decide block size (default 256) and hash-based prefix reuse semantics.
- [x] Extend scheduler for PEARL: prefill vs decode stages, waiting/running queues, preemption when KV blocks scarce, rollback hooks, EOS/max-token checks.
- [x] Implement draft runner: gamma-window greedy generation (temperature-free for now), append without EOS checks, package tokens for verification.
- [x] Implement target runner: flattened verification inputs, Bernoulli acceptance on target probs, rollback + revised token insertion, next-round token append, finish detection.
- [ ] Integrate CUDA graph capture or eager fallback thresholds for decode in PEARL path.
- [ ] Add optional dynamic TP padding (heads/intermediate/vocab) with `valid_vocab_size` preservation and clear perf warnings.
- [ ] Wire entrypoints: select PEARL via config/CLI; route workers to PEARL runners; ensure tokenizer/EOS consistency between draft/target.
- [ ] Logging/metrics: acceptance rate, rollbacks, gamma selection (auto/manual), throughput counters; ensure structured logging via vllm.logger.
- [ ] Integrate PEARL into v1 GPU runner:
  - [x] Create draft/target/verify NCCL subgroups and role partitioning in worker startup.
  - [x] Wire paged KV allocator with rollback support and logical block tables -> slot_mapping/context_lens for target verifier.
  - [ ] Replace speculative path wiring to dispatch to PEARL draft/target runners (dual-model execution) and connect verification broadcast/reduce.
  - [ ] Hook auto-gamma profiling and CUDA graph capture thresholds into runner lifecycle.
  - [x] Add GPU smoke/integration test or bench script (optional/skip in CPU CI) demonstrating dual-model PEARL path.

Suggested execution order (actionable)
1) Config/entry: expose `method="pearl"` in CLI/config parsing and enforce tokenizer/EOS consistency checks.
2) NCCL groups/roles: partition ranks into draft/target; create draft/target/verify subgroups; plumb role metadata into runner init.
3) KV allocator & slot mapping: add rollback hook to paged KV allocator; map logical 256-block tables to physical pages and produce slot_mapping/context_lens for target verifier.
4) Wire PEARL runners: add PEARL branches in v1 GPU runner to invoke draft/target loops, connect verify broadcast/reduce, and keep KV/state in sync.
5) Auto-gamma & CUDA graph: optional profiling to pick gamma buckets; capture decode CUDA graphs for small batches; eager fallback for prefill/large batch.
6) Dynamic TP padding (optional): pad heads/kv_heads/intermediate/vocab when TP not power-of-two, preserve valid_vocab_size, log perf caveat.
7) Logging/metrics: log gamma choices, acceptance/rollback counts, throughput; add basic counters using vllm.logger.
8) Tests/bench: keep CPU unit tests; add GPU smoke (dual-model) marked xfail/skip without GPU; add optional bench/script for throughput.

Testing & Validation TODO
- [ ] Unit-ish: scheduler prefill/decode ordering and preemption under KV pressure.
- [ ] Unit-ish: rollback correctness on KV/block tables and token streams.
- [ ] Unit-ish: verification edge cases (EOS mid-gamma, max_tokens hit, rollout<gamma) with mocked logits.
- [ ] Unit-ish: PEARL logprobs alignment for accept/reject paths.
- [ ] Smoke/integration: dual-model flow with tiny/mocked models exercising draft→target messages and KV updates (CPU-safe or GPU if available).
- [ ] Bench/example script: demonstrate PEARL invocation and report throughput; optional GPU-only path.
- [ ] Run `pytest` on new tests; document skips for GPU-required cases.
- [ ] Document known gaps/risks after each milestone (e.g., CUDA graph fallback, dynamic TP perf cost).

Notes / Open Questions
- Current wiring findings: SpeculativeConfig supports ngram/medusa/eagle/MTP but no PEARL; GPUModelRunner drives speculation via SpecDecodeMetadata (draft tokens + logits indices) and single draft model path. Entry module `vllm/v1/spec_decode` holds proposers; `spec_decode/__init__.py` is a placeholder. We will need new dual-model flow + verifier path instead of single-draft proposer.
- Draft proposals currently reuse the target KV state (no separate draft KV cache yet). This keeps integration light but may reduce acceptance for heterogeneous draft/target models.
- PEARL draft proposals are currently generated with draft-side rollback (no persistent draft KV); proposal timing is still aligned to the existing spec-decode cadence, so fully pipelined draft-ahead execution remains open.
- PEARL draft/target sync path currently assumes sync scheduling; async scheduling support still needs explicit handling.
- PEARL async scheduling now syncs sampled tokens via GPU tensor broadcast before applying updates; further tuning still needed.
- PEARL logprobs for accepted windows currently come from draft proposals; target-aligned logprobs remain a TODO (see roadmap).
- Need to align PEARL KV rollback semantics with existing paged attention allocator; may require allocator hooks.
- Decide whether to expose auto-gamma profiling (draft/target throughput) in production or only as an opt-in warmup.
- For non-2-power TP, ensure padding does not break sharding assumptions elsewhere in vLLM.

Progress log
- Added PEARL config surface to `SpeculativeConfig` (method `pearl`, draft/target model + TP fields, gamma/auto-gamma, block/batch caps, dynamic TP padding flag) with basic validation and defaults.
- PEARL config now validates draft/target EOS consistency at init (loads draft AutoConfig; raises on mismatch).
- Engine args now allow `method="pearl"` and auto-fill `pearl_target_model` when omitted; draft model required or raises.
- Added PEARL NCCL subgroup helper (`pearl_dist.py`) with role partitioning; GPU runner now initializes draft/target/verify groups when method is PEARL.
- Fixed logical scheduler sequence bookkeeping: prompt length captured at init (num_completion_tokens now advances), block boundary handling reserves blocks when crossing boundaries, and rollback updates cached counts.
- Target verifier no longer double-appends drafted tokens on acceptance.
- Worker startup now pre-creates PEARL NCCL groups and passes them into GPUModelRunner; runner also accepts injected groups (falls back to init if None).
- KV helpers: BlockTable now supports rollback of trailing blocks; added slot_mapping/context_lens builders (`pearl_kv.py`) and unit tests.
- Added BlockTable rollback-to-num-tokens API and a CUDA smoke test for slot_mapping after rollback.
- Added `PearlKVAdapter` to map logical block tables into BlockTable rows with rollback and slot_mapping helpers; covered by unit tests.
- GPUModelRunner PEARL path now uses PearlKVAdapter to build slot mapping (prefill/decode) while leaving non-PEARL paths unchanged. GPU smoke test placeholder added (skipped for now).
- PearlKVAdapter now supports batch slot mapping (mixed reqs), and GPUModelRunner PEARL branch uses it to commit slot mapping; unit test added.
- PEARL KV adapter now instantiated inside GPUModelRunner when method is PEARL (no functional wiring yet).
- Suppressed noisy SWIG DeprecationWarnings in v1 test suites to keep PEARL test output clean.
- Fixed batch slot-mapping unit to compare tensor/list correctly; broadened warning filters (SWIG + importlib bootstrap).
- CommonAttentionMetadata now carries optional `context_lens`; PEARL path wires in adapter-built context_lens (non-PEARL unchanged).
- GPUModelRunner now treats `method="pearl"` as a first-class option (skips legacy drafter wiring; dual-model path to be integrated next).
- Added PEARL execute-model wrapper: PEARL ranks route through `_execute_model_pearl` (current behavior = target-only passthrough with subgroup barrier); main body factored to `_execute_model_body` for future dual-model loop integration.
- PEARL scaffolding: track `pearl_is_draft`, optionally load a dedicated draft model in `load_model` (alongside target), and allow PEARL path to override the forward model via `_model_forward(model_override=...)`.
- Added context_lens shape assertion in PEARL slot_mapping path and a CUDA smoke test (`tests/v1/spec_decode/test_pearl_smoke_gpu.py`) to ensure context_lens flows through CommonAttentionMetadata on GPU.
- Fixed `BlockTable.build_slot_mapping` numpy shadowing (UnboundLocalError) hit by the CUDA smoke; now reuses global `np`.
- Decided process/group layout and comms: reuse existing worker processes, partition ranks into draft group (first `draft_tp` ranks) and target group (next `target_tp` ranks). Build NCCL subgroups for draft, target, and a verify group = `{draft_master} ∪ target_ranks}` to broadcast verification payloads/results. Control plane still uses existing RPC/driver; in-group sync via subgroup barriers/broadcast. No extra processes; role determined by rank offsets.
- KV/block handling mapped: keep vLLM paged KV cache as backing store; introduce logical block tables per sequence (default logical block size 256 tokens). Hash-based prefix reuse (xxhash-style) on logical blocks; logical blocks map to physical pages—if paged-attn page size differs, allocate in whole pages and store intra-page offset mapping for slot_mapping/context. Rollback frees logical tail blocks and returns pages to allocator; prefix cache reuse via hash→page table. Need allocator hooks to support rollback and to query free capacity for scheduler preemption checks.
- Scheduler design mapped: waiting/running deques; prefill stage batches up to `max_num_seqs` and `max_num_batched_tokens`, checks KV free capacity before allocation; decode stage pulls from running with preemption if KV insufficient (push back to waiting). Postprocess on both draft/target sides uses EOS/max_tokens; rollback hook returns KV pages and trims block tables. Maintains num_cached_tokens and block_table per sequence to build slot_mapping/context for target verifier.
- Added initial draft runner helper (`vllm/v1/spec_decode/pearl.py`): `PearlDraftRunner` with gamma-step greedy decode (temperature-free), appends tokens without EOS checks, and packages verification payload (to-verify tokens, counts, next-round tokens per seq). Currently self-contained; distributed wiring will be added when integrated.
- PEARL execute path now uses the standard spec-decode scheduler output; draft proposals are generated after sampling for the next step rather than injected into the current scheduler output.
- Added logical scheduler/KV helpers and target verifier stubs (`vllm/v1/spec_decode/pearl_scheduler.py`, `pearl_target.py`): CPU-friendly block manager with rollback, prefill/decode scheduling, and deterministic target-side verification to unit test control flow without GPUs.
- Target verifier now includes Bernoulli acceptance and rollback semantics (`TargetVerifier.verify_and_update`) and a sampling-based test case.
- `_execute_model_pearl` now runs a real draft forward pass (uses draft model if provided), snapshots KV/block/token state, rolls back after collecting proposals, and broadcasts proposals before the target pass.
- BlockTable/MultiGroupBlockTable gain a free-block callback hook so rollback can notify a KV allocator when trailing pages are dropped.
- GPUModelRunner 現在掛上 free-block callback 並暫存釋放的 block ids（後續需接到 scheduler/allocator 才會真正釋放）。
- ModelRunnerOutput 攜帶 `freed_block_ids`，scheduler 會呼叫 KVCacheManager.free_block_ids，BlockPool 提供 free_blocks_by_id 實際歸還頁面。
- PEARL draft proposals now run for gamma steps via draft forward+sample with full rollback, and PEARL verification uses a custom sampler (Bernoulli accept/reject) on target logits; output tokens are broadcast over the verify subgroup.
- MultiGroupBlockTable 增加 rollback_row_to_num_tokens；PEARL 路徑在抽樣後依拒絕數 rollback KV blocks（回收頁面走 free-block callback）。
- PEARL sampler 單測新增並通過（greedy 全接受路徑）。
- PEARL config 現在會建立 `draft_model_config`/`draft_parallel_config` 以載入 draft 模型；PEARL 路徑跳過既有 `propose_draft_token_ids`，且 free-block 紀錄不再漏掉 block id 0。
- PEARL spec-decode path now carries per-request pre-verify state into SpecDecodeMetadata and uses it in PearlRejectionSampler (pre-verify accepts/rejects the full draft window without bonus tokens). GPUModelRunner updates pre-verify state after sampling, proposes next draft tokens via the draft model helper, and adjusts rollback math for PEARL’s no-bonus output.
- GPUModelRunner now splits PEARL roles: draft ranks run draft-forward proposal generation and broadcast proposals over the verify subgroup; target ranks consume pending proposals in sampling. Draft/target logits computation honors `model_override` for draft forward.
- Draft ranks now sync sampled tokens from target via verify→draft broadcast, apply them to input_batch, and run rollback bookkeeping to keep draft state aligned for the next proposal step.
- PEARL now initializes role-specific TP groups and patches TP collectives during draft/target forward; TP>1 without split groups raises to avoid deadlocks.
- Scheduler now consumes PEARL verify results (acc/rollout/revise) for rollback accounting and spec-decoding stats; ModelRunnerOutput carries PEARL verify payloads.
- Spec decode logging now emits MAT alongside throughput.
- PEARL logprobs are now returned with output tokens (draft logprobs for accepted windows, target logprobs for rejects/bonus).
- PEARL bonus-token acceptance now uses processed logits (applies logit processors + sampling constraints).
- PEARL proposal broadcast uses tensor packing when logprobs are not requested (object payload remains for logprobs).
- PEARL proposal logprobs can now be broadcast as packed tensors when `max_num_logprobs` is set (object payload retained for full-logprob mode).
- PEARL auto-gamma warmup profiling now aggregates draft/target throughput and selects gamma (capped by num_speculative_tokens).
- PEARL groups are now DP/PP-aware, and PEARL sampler outputs/logprobs are synchronized across PP ranks when broadcast_pp_output is enabled.

Tests run
- `python -m compileall vllm/config/speculative.py vllm/v1/spec_decode/pearl.py vllm/v1/spec_decode/pearl_scheduler.py vllm/v1/spec_decode/pearl_target.py`
- Manual smoke (since pytest not installed): draft/scheduler/target path executed via `python - <<'PY' ...` (see shell history).
- pytest (user-run): `pytest tests/v1/spec_decode/test_pearl.py` (passes; swig Deprecation warnings).
  - Added `tests/v1/spec_decode/conftest.py` to silence SWIG Deprecation warnings in this suite.
- pytest (user-run): `pytest tests/v1/spec_decode/test_pearl.py tests/v1/spec_decode/test_pearl_dist.py` (6 passed; SWIG Deprecation warnings remain upstream).
- pytest (user-run after scheduler/target fixes): `pytest tests/v1/spec_decode/test_pearl.py tests/v1/spec_decode/test_pearl_dist.py` (pass; SWIG Deprecation warnings).
- pytest (user-run after target verifier accept-path update): `pytest tests/v1/spec_decode/test_pearl.py tests/v1/spec_decode/test_pearl_dist.py` (pass; SWIG Deprecation warnings).
- pytest (latest, incl. CUDA smoke): `pytest tests/v1/spec_decode/test_pearl.py tests/v1/spec_decode/test_pearl_dist.py tests/v1/spec_decode/test_pearl_kv.py tests/v1/spec_decode/test_pearl_kv_gpu.py` (pass; upstream SWIG Deprecation warnings).
- pytest (user-run, latest): `pytest tests/v1/spec_decode/test_pearl.py tests/v1/spec_decode/test_pearl_dist.py tests/v1/spec_decode/test_pearl_kv.py tests/v1/spec_decode/test_pearl_kv_gpu.py` → 15 passed; upstream SWIG Deprecation warnings remain.
- pytest (user-run, latest repeat): `pytest tests/v1/spec_decode/test_pearl.py tests/v1/spec_decode/test_pearl_dist.py tests/v1/spec_decode/test_pearl_kv.py tests/v1/spec_decode/test_pearl_kv_gpu.py` → 15 passed; upstream SWIG Deprecation warnings remain.
- Attempted (current shell missing pytest): `pytest tests/v1/spec_decode/test_pearl.py tests/v1/spec_decode/test_pearl_dist.py tests/v1/spec_decode/test_pearl_kv.py tests/v1/spec_decode/test_pearl_kv_gpu.py tests/v1/spec_decode/test_pearl_smoke_gpu.py`.
- Latest (user-run): `pytest tests/v1/spec_decode/test_pearl.py tests/v1/spec_decode/test_pearl_dist.py tests/v1/spec_decode/test_pearl_kv.py tests/v1/spec_decode/test_pearl_kv_gpu.py tests/v1/spec_decode/test_pearl_smoke_gpu.py` → 16 passed; upstream SWIG Deprecation warnings remain.
- Current (CPU-only env): `./.venv/bin/pytest tests/v1/spec_decode/test_pearl.py tests/v1/spec_decode/test_pearl_dist.py tests/v1/spec_decode/test_pearl_kv.py tests/v1/spec_decode/test_pearl_kv_gpu.py tests/v1/spec_decode/test_pearl_smoke_gpu.py` → 13 passed, 3 skipped (CUDA unavailable); warnings: NVML init failure + SWIG Deprecation notices.
- Latest (user-run): `pytest tests/v1/sample/test_rejection_sampler.py -k pearl_rejection_sampler` → 1 passed; SWIG Deprecation warnings remain.
- Not run in this update: `pytest tests/v1/spec_decode/test_pearl.py tests/v1/spec_decode/test_pearl_dist.py tests/v1/spec_decode/test_pearl_kv.py tests/v1/spec_decode/test_pearl_kv_gpu.py tests/v1/spec_decode/test_pearl_smoke_gpu.py`.

Design Proposal (to execute next)
- Config surface:
  - Add speculative `method="pearl"` to `SpeculativeConfig` plus new fields: `pearl_draft_model` (path/name), `pearl_target_model` (default = current target model), `pearl_draft_tp`, `pearl_target_tp`, `pearl_gamma` (int, -1 for auto), `pearl_block_size` (default 256), `pearl_max_num_seqs`, `pearl_max_num_batched_tokens`, `pearl_dynamic_tp_padding` (bool).
  - Validation: draft/target EOS must match; world_size = draft_tp + target_tp; gamma>0 or -1; block_size aligns with paged-cache block (if mismatch, use logical 256 and map to physical pages); max_num_batched_tokens >= max_model_len; TP sizes must not exceed devices; dynamic padding flag required for non-power-of-two TP; enforce same tokenizer/tokenizer_mode.
  - Auto-gamma: optional warmup profiling (small synthetic batches) gated behind flag to avoid default overhead; store gamma per batch size bucket; fallback to manual gamma if profiling disabled.
- Process/group layout:
  - Use existing worker processes but assign roles (draft vs target) based on rank partition; create NCCL subgroups: draft group, target group, verify group (draft master + all target ranks) for broadcasts.
  - Control path reuses existing RPC orchestration; PEARL runner handles intra-group sync and cross-group verify messages.
- KV/block handling:
  - Keep paged KV cache; introduce logical block tables per sequence with rollback and hash-based prefix reuse (xxhash). Block size default 256; map logical blocks to physical pages and adjust slot mapping APIs.
  - Add rollback hook on allocator to free/reassign last N tokens’ blocks.
- Scheduler:
  - Prefill stage honoring `max_num_batched_tokens`/`max_num_seqs` and KV availability; decode stage with waiting/running queues and preemption if no KV space.
  - Postprocess includes EOS/max_tokens checks; rollback API for verifier.
- Draft runner:
  - For each step, run gamma greedy decodes (temperature 0) without EOS checks; append tokens; package verification payload: to-verify tokens (pre-verify one token; post-verify window), next-round input tokens; broadcast to target via verify group.
- Target runner:
  - Flatten inputs across sequences (pre-verify: 1 token, post-verify: gamma tokens) with slot_mapping/context_lens built from block tables.
  - Compute acceptance: Bernoulli on target prob for to-verify tokens; on reject, sample revised token; compute rollout length; broadcast acc/rollout/revise/finish back.
  - Apply rollback if needed, append revised token or next-round tokens, track accepted-token counts, finish on EOS/max_tokens.
- CUDA graph:
  - Optional capture for decode small batches; fallback to eager for prefill or large batches; reuse existing CUDAGraphDispatcher plumbing if possible, else simple per-runner capture list.
- Dynamic TP padding:
  - Optional: pad heads/kv_heads/intermediate/vocab to nearest multiple of tp; preserve `valid_vocab_size`; log perf caveat; gate behind flag.
- Entry/CLI:
  - Expose `--speculative-method pearl` or separate flag; allow passing draft/target models and TPs; ensure tokenizer/eos consistency checks and informative errors.
- Logging/metrics:
  - Structured logs for gamma choice (auto/manual), acceptance rate, rollbacks, throughput; counters for accepted tokens per step; emit warnings on dynamic TP padding and graph fallback.
