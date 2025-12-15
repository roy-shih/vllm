# PEARL Integration in vLLM

## Overview

This document describes the integration of nano-PEARL (Parallel Speculative Decoding with Adaptive Draft Length) into vLLM.

## What is PEARL?

PEARL is a parallel speculative decoding framework that:
- **Disaggregates Draft and Target Models**: Runs draft and target models on separate GPU groups
- **Parallel Inference**: Draft and target models run concurrently for maximum GPU utilization
- **Adaptive Draft Length**: Dynamically adjusts speculation based on alignment quality
- **High Throughput**: Achieves up to 3.06× throughput speedup for 70B LLMs

## Integration Status

### Completed Components

1. **Configuration Layer** (`vllm/config/speculative.py`)
   - Added "pearl" to `SpeculativeMethod` enum
   - Added PEARL-specific configuration parameters:
     - `pearl_gamma`: Adaptive draft length window size (-1 for auto-set)
     - `pearl_max_num_batched_tokens`: Maximum batched tokens (default: 16384)
     - `pearl_max_num_seqs`: Maximum sequences (default: 512)
     - `pearl_kvcache_block_size`: KV cache block size (default: 256)
     - `pearl_num_kvcache_blocks`: Number of KV cache blocks (-1 for auto-set)
     - `target_tensor_parallel_size`: TP size for target model

2. **Proposer Implementation** (`vllm/v1/spec_decode/pearl_proposer.py`)
   - Created `PEARLProposer` class following vLLM's proposer pattern
   - Implemented `propose()` method for draft token generation
   - Implemented `load_model()` for model initialization
   - Added multiprocessing infrastructure for draft-target disaggregation

3. **Model Runner Integration** (`vllm/v1/worker/gpu_model_runner.py`)
   - Added PEARLProposer to type annotations
   - Integrated PEARL initialization in model runner
   - Added PEARL draft token proposal logic

## Usage

### Basic Usage (Offline Inference)

```python
from vllm import LLM, SamplingParams

# Create LLM instance with PEARL speculative decoding
llm = LLM(
    model="meta-llama/Llama-3-70b-instruct",  # Target model
    speculative_config={
        "method": "pearl",
        "model": "meta-llama/Llama-3-8b-instruct",  # Draft model
        "num_speculative_tokens": 5,
        "draft_tensor_parallel_size": 1,  # TP size for draft model
        "target_tensor_parallel_size": 4,  # TP size for target model
        "pearl_gamma": -1,  # Auto-set adaptive draft length
    },
    tensor_parallel_size=4,  # Total TP size for target
)

# Generate text
prompts = ["Explain quantum computing in simple terms"]
sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### API Server Usage

```bash
# Start vLLM server with PEARL
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-70b-instruct \
    --tensor-parallel-size 4 \
    --speculative-config '{
        "method": "pearl",
        "model": "meta-llama/Llama-3-8b-instruct",
        "num_speculative_tokens": 5,
        "draft_tensor_parallel_size": 1,
        "target_tensor_parallel_size": 4,
        "pearl_gamma": -1
    }'
```

Then use OpenAI-compatible API:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

completion = client.chat.completions.create(
    model="meta-llama/Llama-3-70b-instruct",
    messages=[
        {"role": "user", "content": "Explain quantum computing in simple terms"}
    ]
)

print(completion.choices[0].message.content)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | - | Must be "pearl" |
| `model` | str | - | Path to draft model |
| `num_speculative_tokens` | int | - | Number of draft tokens to generate |
| `draft_tensor_parallel_size` | int | 1 | TP size for draft model |
| `target_tensor_parallel_size` | int | - | TP size for target model |
| `pearl_gamma` | int | -1 | Adaptive draft length window (-1 = auto) |
| `pearl_max_num_batched_tokens` | int | 16384 | Max batched tokens (8192 for 40GB GPUs) |
| `pearl_max_num_seqs` | int | 512 | Max sequences (128-256 for 40GB GPUs) |
| `pearl_kvcache_block_size` | int | 256 | KV cache block size |
| `pearl_num_kvcache_blocks` | int | -1 | Number of KV cache blocks (-1 = auto) |

## Architecture

### Integration Points

```
vLLM Engine
    ├── SpeculativeConfig (pearl configuration)
    │   └── vllm/config/speculative.py
    │
    ├── PEARLProposer (draft token generation)
    │   └── vllm/v1/spec_decode/pearl_proposer.py
    │
    ├── GPUModelRunner (model execution)
    │   └── vllm/v1/worker/gpu_model_runner.py
    │       ├── Initialize PEARLProposer
    │       └── Call propose() for draft tokens
    │
    └── Scheduler (token scheduling & verification)
        └── vllm/v1/core/sched/scheduler.py
            ├── Schedule draft tokens
            └── Process acceptance/rejection
```

### PEARL Workflow

1. **Initialization**:
   - PEARLProposer is created in GPUModelRunner
   - Draft and target models are loaded on separate devices
   - Multiprocessing infrastructure is set up for parallel execution

2. **Draft Generation**:
   - Draft model generates speculative tokens in parallel
   - Adaptive draft length adjusts based on alignment
   - Draft tokens are returned to scheduler

3. **Verification**:
   - Target model verifies draft tokens
   - Acceptance/rejection determined via rejection sampling
   - Accepted tokens appended to output

4. **Iteration**:
   - Process repeats until generation complete

## Benefits Over Nano-PEARL Standalone

1. **API Server Support**: Full OpenAI-compatible API server
2. **Unified Interface**: Use vLLM's standard CLI and Python API
3. **Production Features**: Monitoring, logging, metrics, batching
4. **Ecosystem Integration**: Works with vLLM tools and extensions
5. **Easy Comparison**: Benchmark against other vLLM spec decode methods

## Performance Expectations

Based on nano-PEARL benchmarks:
- **HumanEval (BS=32, H200)**: Up to 3.06× speedup, 3546.72 tok/s for 70B models
- **Best for**: Large batch sizes, high-throughput scenarios
- **GPU Utilization**: Maximized through parallel draft-target execution

## Limitations and TODOs

### Current Limitations
1. **Simplified Implementation**: Current version is a skeleton; full multiprocessing logic needs completion
2. **Testing Required**: Comprehensive testing not yet performed
3. **CUDA Graphs**: Integration with CUDA graphs not yet implemented

### Future Work
1. **Complete Multiprocessing**: Implement full draft-target disaggregation with SharedMemory
2. **Add Tests**: Unit tests, integration tests, correctness tests
3. **Optimize Performance**: CUDA graphs, memory optimization
4. **Support More Models**: Extend beyond Llama family
5. **Dynamic Gamma**: Auto-tuning based on runtime metrics
6. **Continuous Batching**: Support for chunked prefill

## References

- **nano-PEARL Repository**: https://github.com/smart-lty/nano-PEARL
- **PEARL Paper**: [ICLR 2025] PEARL: Parallel Speculative Decoding with Adaptive Draft Length
- **ArXiv**: https://arxiv.org/abs/2408.11850

## File Structure

```
vllm/
├── config/
│   └── speculative.py              # PEARL configuration
├── v1/
│   ├── spec_decode/
│   │   └── pearl_proposer.py       # PEARL proposer implementation
│   └── worker/
│       └── gpu_model_runner.py     # Model runner integration
└── unieai-dev/
    ├── nano-PEARL/                 # Original nano-PEARL source
    └── PEARL_INTEGRATION_README.md # This document
```

## Contributing

To improve the PEARL integration:

1. **Complete the Implementation**:
   - Enhance `pearl_proposer.py` with full multiprocessing logic
   - Integrate nano-PEARL's core components
   - Add comprehensive error handling

2. **Add Testing**:
   - Unit tests for PEARLProposer
   - Integration tests with vLLM engine
   - Performance benchmarks

3. **Optimize**:
   - CUDA graph support
   - Memory optimization
   - Batch processing improvements

4. **Document**:
   - API documentation
   - Performance tuning guide
   - Troubleshooting guide

## License

- vLLM: Apache-2.0
- nano-PEARL: (Check repository for license)
