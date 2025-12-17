# PEARL 重新设计：利用 vLLM 的 PageAttention 和 KV Cache

## 问题分析

当前实现的问题：
1. ❌ 自己做 auto-regressive generation（完全重新计算）
2. ❌ 没有使用 KV cache（每次重新计算整个序列）
3. ❌ 没有使用 PageAttention（浪费显存）
4. ❌ 没有批处理（一次只处理一个序列）
5. ❌ 没有利用 vLLM 的 attention backends（FlashAttention, etc.）

**核心问题**：我们整合到 vLLM 就是为了它的 PageAttention、KV cache、scheduler 优势，但当前实现完全没有用到！

## EAGLE 的做法（正确的参考）

EAGLE 的关键设计：

```python
def propose(
    self,
    target_token_ids: torch.Tensor,  # 从 target model 得到
    target_positions: torch.Tensor,
    target_hidden_states: torch.Tensor,
    next_token_ids: torch.Tensor,
    common_attn_metadata: CommonAttentionMetadata,  # ← 关键！
    ...
) -> torch.Tensor:
    # 1. 使用 attention metadata builder
    attn_metadata = attn_metadata_builder.build_for_drafting(
        common_attn_metadata=common_attn_metadata,
        draft_index=0
    )

    # 2. 使用 set_forward_context 设置上下文
    with set_forward_context(per_layer_attn_metadata, ...):
        # 3. 调用 draft model（利用 KV cache）
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            hidden_states=...,
        )

    # 4. 批量采样
    logits = self.model.compute_logits(hidden_states)
    draft_tokens = logits.argmax(dim=-1)

    return draft_tokens
```

**关键点**：
- ✅ 使用 `CommonAttentionMetadata` 管理 KV cache
- ✅ 使用 `AttentionMetadataBuilder` 构建 attention metadata
- ✅ 通过 `set_forward_context` 设置上下文
- ✅ KV cache 自动管理（PageAttention）
- ✅ 批量处理所有序列

## PEARL 的新设计

### 核心思路

**不做 multi-process（暂时）**，先做一个在同一进程中但**正确使用 vLLM 基础设施**的版本。

### 方案：基于 vLLM Model Runner 的 PEARL

```python
class PEARLProposer:
    def __init__(self, vllm_config: VllmConfig):
        # 创建 attention metadata builder
        self.attn_metadata_builder = self._get_attention_metadata_builder()

        # 预分配缓冲区（类似 EAGLE）
        self.input_ids = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        self.positions = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        self.hidden_states = torch.zeros((max_num_tokens, hidden_size), dtype=dtype, device=device)

    def load_model(self, target_model):
        # 加载 draft model
        self.draft_model = get_model(vllm_config, draft_model_config)

        # 初始化 attention metadata builder（从 runner 获取）
        # 这样就能使用 vLLM 的 KV cache 系统

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        req_ids: list[str],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        spec_decode_unsupported_reqs: set,
        # ← 需要添加这些参数
        common_attn_metadata: CommonAttentionMetadata,
        positions: torch.Tensor,
        ...
    ) -> list[list[int]]:
        # 1. 构建 draft model 的 attention metadata
        draft_attn_metadata = self.attn_metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata,
            draft_index=0
        )

        # 2. 批量处理所有请求
        batch_size = len(sampled_token_ids)
        num_draft_per_req = self.num_speculative_tokens

        # 3. 为每个 draft step 生成 tokens
        all_draft_tokens = []
        for step in range(num_draft_per_req):
            # Prepare inputs（批量）
            # ...

            # Forward pass with KV cache
            with set_forward_context(per_layer_attn_metadata, self.vllm_config, ...):
                hidden_states = self.draft_model(
                    input_ids=batch_input_ids,
                    positions=batch_positions,
                    # KV cache 由 attention metadata 管理！
                )

            # Sample next tokens（批量）
            logits = self.draft_model.compute_logits(hidden_states)
            next_tokens = logits.argmax(dim=-1)  # greedy

            all_draft_tokens.append(next_tokens)

        # 4. 重组为 per-request 的 draft tokens
        return self._reorganize_drafts(all_draft_tokens, batch_size)
```

### 关键改动

#### 1. `propose()` 方法签名需要改变

当前：
```python
def propose(
    self,
    sampled_token_ids: list[list[int]],
    req_ids: list[str],
    ...
) -> list[list[int]]:
```

需要改为（类似 EAGLE）：
```python
def propose(
    self,
    sampled_token_ids: list[list[int]],
    req_ids: list[str],
    num_tokens_no_spec: np.ndarray,
    token_ids_cpu: np.ndarray,
    spec_decode_unsupported_reqs: set,
    # 新增：来自 target model 的信息
    common_attn_metadata: CommonAttentionMetadata,
    positions: torch.Tensor,
    ...
) -> list[list[int]]:
```

**问题**：这意味着需要修改 `gpu_model_runner.py` 中调用 `propose()` 的地方。

#### 2. 初始化 AttentionMetadataBuilder

```python
def load_model(self, target_model):
    self.draft_model = get_model(vllm_config, draft_model_config)

    # 获取 attention backend
    # 需要从 model runner 获取，或者自己创建
    from vllm.v1.attention.backends.utils import get_attention_metadata_builder

    self.attn_metadata_builder = get_attention_metadata_builder(
        attention_type=...,  # FlashAttention, Triton, etc.
        kv_cache_config=...,
        num_layers=...,
    )
```

#### 3. 批量生成 Draft Tokens

```python
def _generate_batch_drafts(
    self,
    batch_input_ids: torch.Tensor,  # [batch_size]
    batch_positions: torch.Tensor,  # [batch_size]
    attn_metadata: CommonAttentionMetadata,
) -> torch.Tensor:  # [batch_size, num_speculative_tokens]

    draft_tokens_list = []

    for step in range(self.num_speculative_tokens):
        # Build per-layer attention metadata
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata

        # Forward pass
        with set_forward_context(per_layer_attn_metadata, self.vllm_config, ...):
            hidden_states = self.draft_model(
                input_ids=batch_input_ids,
                positions=batch_positions,
            )

        # Sample
        logits = self.draft_model.compute_logits(hidden_states)
        next_tokens = logits.argmax(dim=-1)  # [batch_size]

        draft_tokens_list.append(next_tokens)

        # Update for next step
        batch_input_ids = next_tokens
        batch_positions = batch_positions + 1

    # Stack: [batch_size, num_speculative_tokens]
    return torch.stack(draft_tokens_list, dim=1)
```

## 实施计划

### Phase 1: 最小化改动（快速验证）

**目标**：验证能否正确使用 attention metadata

1. 修改 `propose()` 签名，接收 `common_attn_metadata`
2. 创建简单的 attention metadata builder
3. 使用 `set_forward_context` 调用 draft model
4. 验证 KV cache 是否工作

**预期**：应该比当前实现快很多（因为有 KV cache）

### Phase 2: 批量处理

**目标**：批量生成 draft tokens

1. 修改为批量处理所有请求
2. 使用批量 forward pass
3. 优化内存使用

**预期**：吞吐量显著提升

### Phase 3: PEARL 特性

**目标**：实现 adaptive draft length

1. 实现 gamma 自适应
2. 根据对齐质量调整 draft length
3. 实现 PEARL 的并行逻辑（可选）

**预期**：MAT 提升到 2.5+

## 挑战

### 1. 接口兼容性

当前 ngram/suffix proposer 的接口：
```python
def propose(
    sampled_token_ids,
    req_ids,
    num_tokens_no_spec,
    token_ids_cpu,
    spec_decode_unsupported_reqs,
) -> list[list[int]]:
```

EAGLE 的接口：
```python
def propose(
    target_token_ids,
    target_positions,
    target_hidden_states,
    next_token_ids,
    last_token_indices,
    common_attn_metadata,
    sampling_metadata,
    mm_embed_inputs,
) -> torch.Tensor:
```

**解决方案**：
- 查看 `gpu_model_runner.py` 如何调用不同的 proposer
- 可能需要为 PEARL 添加特殊处理（类似 EAGLE）

### 2. AttentionMetadata 创建

需要从哪里获取/创建？
- 选项A：从 model runner 传递
- 选项B：自己创建（需要访问 KV cache config）

### 3. KV Cache 管理

Draft model 的 KV cache 如何分配？
- 是否需要独立的 KV cache pool？
- 还是共享 target model 的 KV cache？

## 下一步

1. **研究 gpu_model_runner.py 如何调用 EAGLE**
2. **了解 AttentionMetadata 如何创建和传递**
3. **实现新的 propose() 方法**
4. **测试验证 KV cache 是否工作**

这样才能真正利用 vLLM 的优势！
