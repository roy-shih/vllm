# PEARL Integration Testing Guide

## Quick Test Script

创建测试文件 `test_pearl_simple.py`:

```python
#!/usr/bin/env python3
"""Simple test for PEARL integration."""

import logging
from vllm import LLM, SamplingParams

# Enable debug logging to see PEARL's draft token generation
logging.basicConfig(level=logging.DEBUG)

# Configure PEARL with small models for quick testing
llm = LLM(
    model="facebook/opt-125m",  # Small target model
    speculative_config={
        "method": "pearl",
        "model": "facebook/opt-125m",  # Same model as draft for simplicity
        "num_speculative_tokens": 3,
        "draft_tensor_parallel_size": 1,
        "target_tensor_parallel_size": 1,
        "pearl_gamma": 5,
    },
    max_model_len=512,
    enforce_eager=True,  # Disable CUDA graphs for debugging
)

# Test generation
prompts = ["Hello, my name is"]
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=20,
)

print("\n" + "="*50)
print("Running PEARL test...")
print("="*50)

outputs = llm.generate(prompts, sampling_params)

print("\n" + "="*50)
print("Results:")
print("="*50)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print(f"Tokens: {len(output.outputs[0].token_ids)}")

print("\n" + "="*50)
print("Check logs above for:")
print("- '[PEARL] Loading draft model...'")
print("- '[PEARL] Draft model loaded successfully'")
print("- '[PEARL] Generated X draft tokens from Y input tokens'")
print("- 'SpecDecoding metrics: Mean acceptance length: X.XX'")
print("="*50)
```

## Running the Test

```bash
cd /home/user/vllm
python unieai-dev/test_pearl_simple.py
```

## Expected Behavior

### Success Indicators

1. **Initialization Logs**:
```
[PEARL] Initializing PEARL Proposer V2
[PEARL] Num speculative tokens: 3
[PEARL] Gamma (adaptive draft length): 5
[PEARL] Loading draft model...
[PEARL] Draft model loaded successfully
```

2. **Draft Generation Logs** (DEBUG level):
```
[PEARL] Draft token generation called for X input tokens, requesting 3 draft tokens
[PEARL] Step 1/3: Generated token 1234
[PEARL] Step 2/3: Generated token 5678
[PEARL] Step 3/3: Generated token 9012
[PEARL] Generated 3 draft tokens from X input tokens
```

3. **MAT Metrics** (INFO level):
```
SpecDecoding metrics: Mean acceptance length: 1.XX
```

- `MAT = 1.00`: All drafts rejected (bad, but means pipeline works)
- `MAT = 1.50`: 50% acceptance rate (decent for MVP)
- `MAT > 2.00`: Good acceptance rate

### Failure Indicators

1. **Model Loading Error**:
```
[PEARL] Draft model not loaded, returning empty drafts
```
**Fix**: Check model path is correct

2. **Forward Pass Error**:
```
[PEARL] Error in draft model forward pass at step X: ...
```
**Fix**: Model might need additional parameters. Check error details.

3. **No Draft Tokens Generated**:
```
[PEARL] Generated 0 draft tokens from X input tokens
```
**Fix**: Check error logs for details

## Debugging

### Enable Full Logging

```python
import logging

# Set all vLLM loggers to DEBUG
logging.getLogger("vllm").setLevel(logging.DEBUG)

# Or just PEARL
logging.getLogger("vllm.v1.spec_decode.pearl_proposer").setLevel(logging.DEBUG)
```

### Check What's Being Called

Add print statements to `pearl_proposer.py`:

```python
def propose(self, ...):
    print(f"[DEBUG] propose() called with {len(sampled_token_ids)} requests")
    ...

def _generate_draft_tokens(self, token_ids, num_draft_tokens):
    print(f"[DEBUG] _generate_draft_tokens() called: {len(token_ids)} tokens in, {num_draft_tokens} requested")
    ...
```

### Check Model Output Format

```python
def _generate_draft_tokens(self, token_ids, num_draft_tokens):
    ...
    outputs = self.draft_model(input_ids=input_ids, positions=positions)

    # Debug: Check what we got
    print(f"[DEBUG] Output type: {type(outputs)}")
    print(f"[DEBUG] Output dir: {dir(outputs)}")
    if hasattr(outputs, 'logits'):
        print(f"[DEBUG] Logits shape: {outputs.logits.shape}")
    ...
```

## Common Issues & Solutions

### Issue 1: Model Requires kv_caches Parameter

**Error**:
```
TypeError: forward() missing required argument: 'kv_caches'
```

**Solution**: Update `_generate_draft_tokens()` to pass empty KV cache:

```python
# Create empty KV cache
kv_caches = [None] * self.draft_model.config.num_hidden_layers

outputs = self.draft_model(
    input_ids=input_ids,
    positions=positions,
    kv_caches=kv_caches,
)
```

### Issue 2: Attention Metadata Required

**Error**:
```
RuntimeError: Expected attention_metadata but got None
```

**Solution**: This is complex. For MVP, you might need to:
1. Use a different model that doesn't require attention metadata
2. Or implement basic attention metadata (see EAGLE implementation)

### Issue 3: Model on Wrong Device

**Error**:
```
RuntimeError: Expected all tensors to be on the same device
```

**Solution**: Check device assignment:

```python
print(f"[DEBUG] Device: {self.device}")
print(f"[DEBUG] Model device: {next(self.draft_model.parameters()).device}")
print(f"[DEBUG] Input device: {input_ids.device}")
```

## Performance Monitoring

### Check MAT Over Time

```bash
# Run longer generation and monitor MAT
python -c "
from vllm import LLM, SamplingParams

llm = LLM(
    model='facebook/opt-125m',
    speculative_config={
        'method': 'pearl',
        'model': 'facebook/opt-125m',
        'num_speculative_tokens': 5,
    },
)

# Generate multiple times
for i in range(10):
    outputs = llm.generate(['Hello'], SamplingParams(max_tokens=50))
    print(f'Iteration {i+1} complete')
"
```

Watch for MAT in logs after each iteration.

### Benchmark Throughput

```python
import time
from vllm import LLM, SamplingParams

# Without PEARL
llm_baseline = LLM(model="facebook/opt-125m")

# With PEARL
llm_pearl = LLM(
    model="facebook/opt-125m",
    speculative_config={
        "method": "pearl",
        "model": "facebook/opt-125m",
        "num_speculative_tokens": 5,
    },
)

prompts = ["Hello"] * 10
params = SamplingParams(max_tokens=100)

# Baseline
start = time.time()
llm_baseline.generate(prompts, params)
baseline_time = time.time() - start

# PEARL
start = time.time()
llm_pearl.generate(prompts, params)
pearl_time = time.time() - start

print(f"Baseline: {baseline_time:.2f}s")
print(f"PEARL: {pearl_time:.2f}s")
print(f"Speedup: {baseline_time/pearl_time:.2f}x")
```

**Expected for MVP**:
- If MAT < 1.2: Likely slower (overhead > benefit)
- If MAT > 1.5: Should see some speedup
- If MAT > 2.0: Good speedup expected

## Next Steps After Basic Test Works

1. **Test with Different Models**:
   - Llama-3-1B (draft) → Llama-3-8B (target)
   - Different model families

2. **Test with Larger num_speculative_tokens**:
   - Start with 3
   - Increase to 5, 7, 10
   - Monitor MAT and performance

3. **Test with Batching**:
   - Multiple prompts
   - Different prompt lengths

4. **Optimize**:
   - Add KV cache
   - Implement batched inference
   - Add adaptive draft length

## Reference: Expected Log Flow

```
[INFO] [PEARL] Initializing PEARL Proposer V2
[INFO] [PEARL] Num speculative tokens: 3
[INFO] [PEARL] Gamma (adaptive draft length): 5
[INFO] [PEARL] Draft model: facebook/opt-125m
[INFO] [PEARL] Target model: facebook/opt-125m
[INFO] [PEARL] Loading draft model...
[INFO] [PEARL] Draft model loaded successfully

... (generation starts) ...

[DEBUG] [PEARL] Draft token generation called for 5 input tokens, requesting 3 draft tokens
[DEBUG] [PEARL] Step 1/3: Generated token 262
[DEBUG] [PEARL] Step 2/3: Generated token 11
[DEBUG] [PEARL] Step 3/3: Generated token 766
[DEBUG] [PEARL] Generated 3 draft tokens from 5 input tokens

... (verification by target model) ...

[INFO] SpecDecoding metrics: Mean acceptance length: 1.67, Accepted throughput: X.XX tokens/s, ...

... (more generations) ...

[INFO] [PEARL] MAT (Mean Accepted Tokens): 1.67 (Accepted: 10, Drafts: 15)
```

Good luck with testing! 🚀
