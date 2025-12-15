# nano-PEARL Integration Plan for vLLM

## Overview
Integrate nano-PEARL's parallel speculative decoding framework into vLLM to enable:
1. Draft-Target model disaggregation across separate GPU groups
2. Parallel inference with adaptive draft length
3. API server support (which nano-PEARL currently lacks)

## Architecture Analysis

### nano-PEARL Architecture
- **Multi-process design**: Draft and Target models run in separate processes
- **Communication**: SharedMemory for inter-process communication
- **Parallelism**: Draft and Target inference run in parallel
- **Adaptive draft length**: Dynamic speculation based on alignment
- **Limitation**: No API server, offline inference only

### vLLM Speculative Decoding Architecture
- **Proposer pattern**: All methods implement `propose()` and `load_model()`
- **API integration**: Full API server support via engine args
- **Scheduler integration**: Draft tokens handled in scheduler
- **Multiple methods**: EAGLE, Medusa, N-gram, Suffix, etc.

## Integration Strategy

### Phase 1: Add PEARL as New Speculative Method
1. **Add "pearl" to SpeculativeMethod enum** (`vllm/config/speculative.py`)
2. **Create PEARLProposer class** (`vllm/v1/spec_decode/pearl_proposer.py`)
3. **Extend SpeculativeConfig** to support PEARL-specific parameters

### Phase 2: Implement PEARLProposer
Location: `vllm/v1/spec_decode/pearl_proposer.py`

Key components:
- Implement `propose()` method following vLLM's proposer interface
- Implement `load_model()` for draft/target model initialization
- Adapt nano-PEARL's multiprocessing architecture
- Integrate SharedMemory communication
- Support adaptive draft length via gamma parameter

### Phase 3: Configuration Integration
1. **Extend SpeculativeConfig**:
   - Add `draft_tensor_parallel_size` parameter
   - Add `target_tensor_parallel_size` parameter
   - Add `gamma` parameter for adaptive draft length
   - Add `max_num_batched_tokens` parameter
   - Add `kvcache_block_size` parameter

2. **Update CLI args** (`vllm/engine/arg_utils.py`):
   - Add PEARL-specific arguments
   - Update validation for PEARL method

### Phase 4: Worker Integration
1. **Create PEARLSpeculator** (`vllm/v1/worker/gpu/spec_decode/pearl.py`)
2. **Update speculator factory** (`vllm/v1/worker/gpu/spec_decode/__init__.py`)
3. **Integrate with GPU model runner** (`vllm/v1/worker/gpu_model_runner.py`)

### Phase 5: Copy nano-PEARL Core Components
Copy necessary components from nano-PEARL:
- `nano_pearl/pearl_engine/pearl_model_runner.py` → Adapt to vLLM's ModelRunner interface
- `nano_pearl/pearl_engine/scheduler.py` → Adapt to vLLM's scheduler
- `nano_pearl/pearl_config.py` → Merge into vLLM's SpeculativeConfig
- `nano_pearl/layers/` → Use vLLM's existing layers where possible

## Implementation Steps

### Step 1: Update SpeculativeMethod Enum
File: `vllm/config/speculative.py`
- Add "pearl" to the SpeculativeMethod Literal type

### Step 2: Create PEARLProposer
File: `vllm/v1/spec_decode/pearl_proposer.py`
- Create class inheriting from base proposer pattern
- Implement multiprocessing architecture
- Implement SharedMemory communication
- Implement draft/target parallel execution

### Step 3: Extend SpeculativeConfig
File: `vllm/config/speculative.py`
- Add PEARL-specific configuration parameters
- Add validation logic for PEARL method

### Step 4: Update CLI Arguments
File: `vllm/engine/arg_utils.py`
- Add `--pearl-*` arguments for PEARL-specific configs
- Update `create_speculative_config()` to handle PEARL

### Step 5: Create Worker-Level Integration
File: `vllm/v1/worker/gpu/spec_decode/pearl.py`
- Create `PEARLSpeculator` class
- Integrate with CUDA graphs if applicable
- Handle draft token generation

### Step 6: Update Factory Functions
File: `vllm/v1/worker/gpu/spec_decode/__init__.py`
- Add PEARL case to `init_speculator()` factory

### Step 7: Test Integration
- Create test scripts
- Test offline inference
- Test API server integration
- Benchmark performance

## Key Differences to Handle

### 1. Process vs Thread Model
- nano-PEARL: Multi-process with SharedMemory
- vLLM: Multi-GPU in single process
- **Solution**: Maintain nano-PEARL's multi-process architecture within PEARLProposer

### 2. API Server Support
- nano-PEARL: No API server
- vLLM: Full API server support
- **Solution**: PEARLProposer integrates with vLLM's existing API infrastructure

### 3. Configuration System
- nano-PEARL: Custom PEARLConfig dataclass
- vLLM: SpeculativeConfig with multiple method support
- **Solution**: Extend SpeculativeConfig with PEARL parameters

### 4. Device Management
- nano-PEARL: Explicit device group assignment
- vLLM: Managed by ParallelConfig
- **Solution**: Map PEARL's device groups to vLLM's parallel config

## Benefits of Integration

1. **API Server Support**: PEARL gains full API server capabilities
2. **Unified Interface**: Use vLLM's standard CLI and API
3. **Ecosystem Integration**: Works with vLLM's monitoring, logging, metrics
4. **Production Ready**: Leverages vLLM's production-tested infrastructure
5. **Comparison**: Easy benchmarking against other vLLM speculative methods

## Testing Strategy

1. **Unit Tests**: Test PEARLProposer in isolation
2. **Integration Tests**: Test with vLLM engine
3. **API Tests**: Test via API server
4. **Performance Tests**: Benchmark against EAGLE, Medusa, etc.
5. **Correctness Tests**: Verify output quality matches expectations

## Timeline

1. Step 1-3: Configuration and structure (~2 hours)
2. Step 4-5: Core implementation (~4 hours)
3. Step 6-7: Integration and testing (~2 hours)

Total estimated effort: ~8 hours of development time
