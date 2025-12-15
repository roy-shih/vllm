# PEARL Integration - Implementation Status

## Current Status: Framework Complete, Core Logic Pending

Date: 2025-12-15

## What's Been Done ✅

### 1. Integration Framework (COMPLETE)
- ✅ Added "pearl" to `SpeculativeMethod` enum
- ✅ Added PEARL configuration parameters to `SpeculativeConfig`
- ✅ Created `PEARLProposer` class with proper interface
- ✅ Integrated with `GPUModelRunner`
- ✅ Integrated with vLLM's scheduler and metrics system

### 2. Configuration System (COMPLETE)
- ✅ `pearl_gamma`: Adaptive draft length window
- ✅ `pearl_max_num_batched_tokens`: Batch token limit
- ✅ `pearl_max_num_seqs`: Sequence limit
- ✅ `pearl_kvcache_block_size`: KV cache configuration
- ✅ `pearl_num_kvcache_blocks`: KV cache blocks
- ✅ `target_tensor_parallel_size`: TP configuration

### 3. MAT Logging (READY)
- ✅ vLLM's existing `SpecDecodingStats` system will automatically log MAT
- ✅ Formula: MAT = 1 + (num_accepted_tokens / num_drafts)
- ✅ Logged as "Mean acceptance length" in vLLM metrics
- ✅ Available in Prometheus metrics

## What Needs To Be Done ⚠️

### Critical Path to Performance Parity

#### PRIORITY 1: Draft Token Generation (REQUIRED FOR ANY PERFORMANCE)
**Status**: Placeholder implementation only
**Current**: Returns empty list
**Needed**: Actual draft token generation

**Implementation Options**:

**Option A: Simplified Single-Process Version** (Recommended for MVP)
- Use vLLM's existing model loading (`get_model()`)
- Generate draft tokens auto-regressively using draft model
- No multiprocessing initially
- Simpler but still functional

**Option B: Full nano-PEARL Port** (For Full Performance)
- Port multiprocessing architecture from nano-PEARL
- Implement SharedMemory communication
- Separate draft and target processes
- Parallel execution

**Files to Modify**:
- `vllm/v1/spec_decode/pearl_proposer.py` - Implement `_generate_draft_tokens()`

**Key nano-PEARL Components to Port**:
```
nano_pearl/pearl_engine/pearl_model_runner.py
  - ModelRunnerBase
  - DraftModelRunner
  - TargetModelRunner
  - prepare_prefill/prepare_decode methods

nano_pearl/pearl_engine/scheduler.py
  - Scheduler
  - Sequence management
  - Block allocation

nano_pearl/pearl_engine/sequence.py
  - Sequence class
  - SequenceStatus

nano_pearl/pearl_engine/block_manager.py
  - BlockManager
  - KV cache management
```

#### PRIORITY 2: KV Cache Integration
**Status**: Not implemented
**Needed**: Proper KV cache management for draft model

**Implementation Steps**:
1. Integrate with vLLM's KV cache system (`vllm/v1/kv_cache_interface.py`)
2. Implement block allocation for draft tokens
3. Handle prefix caching (important for PEARL's performance)

#### PRIORITY 3: Attention Backend Integration
**Status**: Not implemented
**Needed**: Proper attention metadata for draft model inference

**Implementation Steps**:
1. Create `AttentionMetadata` for draft model
2. Handle both prefill and decode phases
3. Support Flash Attention / xFormers / other backends

#### PRIORITY 4: Adaptive Draft Length
**Status**: Placeholder auto-set gamma
**Needed**: Actual adaptive draft length logic

**nano-PEARL Algorithm**:
```python
# Simplified version of adaptive draft length:
if alignment_good:
    # Draft model generates without interruption
    draft_length = min(gamma, max_remaining)
else:
    # Target model prevents trash drafts
    draft_length = min(current_good_streak, gamma)
```

#### PRIORITY 5: Multiprocessing Architecture (Optional for MVP)
**Status**: Not implemented
**Needed for**: True draft-target disaggregation and parallel execution

**Components**:
1. Separate processes for draft and target
2. SharedMemory communication (see `PEARLController` stub)
3. Process synchronization via Events
4. Distributed groups for TP

### 性能优化 (After Core Works)

1. **CUDA Graphs**: Cache execution graphs for decode phase
2. **Memory Optimization**: Efficient buffer management
3. **Batch Processing**: Handle variable-length sequences efficiently

## Implementation Roadmap

### Phase 1: MVP - Basic Draft Token Generation (Est: 4-8 hours)
**Goal**: Generate draft tokens, see non-zero MAT

1. Implement simple auto-regressive draft generation in `_generate_draft_tokens()`
2. Use vLLM's existing model inference infrastructure
3. Handle basic KV cache (can be inefficient initially)
4. Test with small models (Llama-3-1B draft, Llama-3-8B target)

**Success Criteria**:
- Draft tokens are generated (non-empty)
- MAT > 1.0 (some drafts accepted)
- No crashes or errors

### Phase 2: Performance - Optimize Draft Generation (Est: 8-16 hours)
**Goal**: Improve draft quality and speed

1. Implement proper KV cache management
2. Integrate with vLLM's attention backends
3. Add batching support
4. Optimize memory usage

**Success Criteria**:
- MAT comparable to nano-PEARL's ngram baseline
- Reasonable throughput (>50% of target-only)

### Phase 3: PEARL Features - Adaptive Length (Est: 8-16 hours)
**Goal**: Implement PEARL-specific optimizations

1. Implement adaptive draft length algorithm
2. Add alignment tracking
3. Implement gamma tuning

**Success Criteria**:
- Adaptive behavior observable in logs
- MAT improves with good alignment

### Phase 4: Full Integration - Multiprocessing (Est: 16-32 hours)
**Goal**: Full nano-PEARL feature parity

1. Implement multiprocessing architecture
2. Add SharedMemory communication
3. Implement parallel draft-target execution
4. Port nano-PEARL's scheduler and sequence management

**Success Criteria**:
- Draft and target run in separate processes
- Parallel execution observable
- Performance parity with nano-PEARL

### Phase 5: Production - Optimization & Polish (Est: 8-16 hours)
**Goal**: Production-ready implementation

1. Add CUDA graphs support
2. Comprehensive error handling
3. Memory optimization
4. Performance tuning

**Success Criteria**:
- Stable under load
- Performance matches or exceeds nano-PEARL
- Clean error messages

## Quick Start Guide for Developers

### Understanding the Current Code

**Entry Point**: `vllm/v1/worker/gpu_model_runner.py:394-397`
```python
elif self.speculative_config.method == "pearl":
    self.drafter = PEARLProposer(self.vllm_config)
    self.drafter.load_model(self.model.model)
```

**Proposer**: `vllm/v1/spec_decode/pearl_proposer.py`
- `__init__()`: Initialize configuration
- `load_model()`: Load draft model (currently stub)
- `propose()`: Generate draft tokens (currently returns empty)
- `_generate_draft_tokens()`: **← THIS IS WHERE YOU NEED TO IMPLEMENT**

**Integration Point**: `vllm/v1/worker/gpu_model_runner.py:3413-3422`
```python
elif spec_config.method == "pearl":
    draft_token_ids = self.drafter.propose(...)
```

### Where to Start Implementing

1. **File**: `vllm/v1/spec_decode/pearl_proposer.py`
2. **Method**: `_generate_draft_tokens()`
3. **Goal**: Return non-empty list of draft token IDs

**Minimal Implementation**:
```python
def _generate_draft_tokens(self, token_ids, num_draft_tokens):
    # Option 1: Simple greedy generation
    current_tokens = torch.tensor([token_ids], device=self.device)

    drafts = []
    for _ in range(num_draft_tokens):
        # Forward pass through draft model
        with torch.no_grad():
            outputs = self.draft_model(current_tokens)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
            drafts.append(next_token.item())
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0)], dim=1)

    return drafts
```

**Note**: This is oversimplified and won't work directly. You need to:
- Handle attention metadata properly
- Manage KV cache
- Use proper position encodings
- Handle batching

### Reference Implementations

Look at these vLLM files for examples:
- `vllm/v1/spec_decode/eagle.py` - Complex but feature-complete
- `vllm/v1/spec_decode/medusa.py` - Simpler, good starting point
- `vllm/v1/spec_decode/ngram_proposer.py` - Simplest, no model inference

Look at nano-PEARL for PEARL-specific logic:
- `/home/user/vllm/unieai-dev/nano-PEARL/nano_pearl/pearl_engine/pearl_model_runner.py`
- `/home/user/vllm/unieai-dev/nano-PEARL/nano_pearl/pearl_engine/scheduler.py`

## Testing Strategy

### Unit Tests
```python
# Test draft token generation
def test_pearl_draft_generation():
    proposer = PEARLProposer(vllm_config)
    proposer.load_model(target_model)

    token_ids = [1, 2, 3, 4, 5]
    drafts = proposer._generate_draft_tokens(token_ids, num_draft_tokens=5)

    assert len(drafts) == 5
    assert all(isinstance(t, int) for t in drafts)
```

### Integration Tests
```python
# Test full pipeline
def test_pearl_integration():
    llm = LLM(
        model="meta-llama/Llama-3-8b",
        speculative_config={
            "method": "pearl",
            "model": "meta-llama/Llama-3-1b",
            "num_speculative_tokens": 5,
        }
    )

    outputs = llm.generate(["Hello world"], SamplingParams(max_tokens=20))
    assert len(outputs) > 0
```

### Performance Tests
```bash
# Benchmark MAT
python benchmarks/benchmark_serving.py \
    --model meta-llama/Llama-3-70b \
    --speculative-config '{"method": "pearl", "model": "meta-llama/Llama-3-8b", "num_speculative_tokens": 5}'
```

## Debugging Tips

### Enable Debug Logging
```python
import logging
logging.getLogger("vllm.v1.spec_decode.pearl_proposer").setLevel(logging.DEBUG)
```

### Check MAT in Logs
Look for:
```
SpecDecoding metrics: Mean acceptance length: X.XX
```

MAT values:
- `1.00`: No drafts accepted (all rejected)
- `1.50`: 50% of drafts accepted on average
- `2.00`: 100% of drafts accepted (perfect predictions)

### Common Issues

1. **"Draft model not loaded"**
   - Check `load_model()` is called
   - Check model path is correct

2. **Empty draft tokens**
   - Check `_generate_draft_tokens()` implementation
   - Add debug logging to see if it's being called

3. **Crashes during generation**
   - Check tensor shapes and devices
   - Verify attention metadata is correct
   - Check KV cache compatibility

## Performance Expectations

Based on nano-PEARL benchmarks (H200, HumanEval, BS=32):

| Metric | Target | MVP | Optimized | Full |
|--------|--------|-----|-----------|------|
| MAT | 2.5-3.0 | 1.1-1.3 | 1.5-2.0 | 2.5-3.0 |
| Throughput | 3546 tok/s | ~500 tok/s | ~1500 tok/s | ~3500 tok/s |
| Speedup | 3.06x | ~0.5x | ~1.5x | ~3x |

## Questions?

- **nano-PEARL source**: `/home/user/vllm/unieai-dev/nano-PEARL/`
- **vLLM spec decode**: `/home/user/vllm/vllm/v1/spec_decode/`
- **Integration docs**: `/home/user/vllm/unieai-dev/PEARL_INTEGRATION_README.md`

## Next Immediate Steps

1. ✅ Review this document
2. 🔄 Implement basic `_generate_draft_tokens()`
3. ⏭️ Test with small model
4. ⏭️ Verify MAT logging works
5. ⏭️ Iterate to improve performance

Good luck! 🚀
