# DFlash 首请求复读问题分析与修复计划

## 问题描述

在 Ascend NPU 上使用 DFlash 推测解码（Qwen3.5-122B）时，首个请求会出现 token 复读现象，后续请求精度正常。GPU（H20）上无此问题。

## 根因分析

### 已排除的假设：RoPE 返回值被丢弃

**初始判断**：`patch_qwen3_dflash.py` 第 43 行 `rotary_emb()` 返回值被丢弃，可能导致 RoPE 未应用。

**排除原因**：经核实 Triton RoPE kernel (`vllm_ascend/ops/triton/rope.py`) 使用 `tl.store` 直接写入传入的 `q_ptr`/`k_ptr` 底层存储，是**原地修改**。`rope_forward_triton` 传入的 `q`/`k` 是 `all_k_flat` 的 `.view()`（不创建新存储），kernel 通过 `tl.store` 直接写回 `all_k_flat` 的内存。返回值 `q, k` 与输入共享底层存储，丢弃返回值不影响已完成的原地修改。

### 主要 Bug：`set_inputs_first_pass` 原地修改 `cad` 导致 CPU 侧元数据过期

**文件**: `vllm_ascend/spec_decode/dflash_proposer.py`，第 129-147 行

#### 问题详情

Ascend `set_inputs_first_pass` **原地修改** `cad` (CommonAttentionMetadata) 对象，更新了 GPU 侧字段（`seq_lens`, `query_start_loc`, `slot_mapping`），但**未更新** CPU 侧镜像字段：

- `cad.seq_lens_cpu` — 未更新
- `cad._seq_lens_cpu` — 未更新
- `cad.num_computed_tokens_cpu` — 未更新
- `cad._num_computed_tokens_cpu` — 未更新

**对比上游** `DFlashProposer.set_inputs_first_pass`（`vllm/v1/spec_decode/dflash.py` 第 160-178 行）：创建**新的** `CommonAttentionMetadata` 对象，显式设置 `_seq_lens_cpu=None`、`_num_computed_tokens_cpu=None`，确保下游代码不会使用过期值。

#### 影响分析

1. **`build()` 方法中的 `actual_seq_lengths_kv`**：当 `parallel_drafting=True` 时（DFlash 就是 parallel_drafting），`attention_v1.py:302-303` 会覆盖 `seq_lens` 为 GPU tensor，所以 `actual_seq_lengths_kv` 计算正确。**此路径不受影响。**

2. **`_propose` 中 FULL graph mode 的 padding 逻辑**（`llm_base_proposer.py:676-677`）：当 `method == "dflash"` 时，只更新了 `seq_lens`（GPU），没有像其他方法那样更新 `seq_lens_cpu` 和 `_seq_lens_cpu`。如果 graph replay 时有其他代码读取这些 CPU 字段，可能出问题。

3. **`prepare_inputs_padded` 创建的 `spec_common_attn_metadata`**（`llm_base_proposer.py:1844-1865`）：从原始 `common_attn_metadata` 复制 `_seq_lens_cpu` 和 `seq_lens_cpu`，这些值在 DFlash 场景下可能已过期。

4. **首请求特有**：首请求时 `spec_decode_metadata=None`，不调用 `prepare_inputs_padded`，直接使用 `_build_attention_metadata` 生成的 `common_attn_metadata`。此时 `_seq_lens_cpu` 包含的是 target model prefill 的序列长度，DFlash 修改了 `cad.seq_lens`（GPU）但 `_seq_lens_cpu` 仍是旧值。后续请求通过 `prepare_inputs_padded` 创建新的 metadata 对象，虽然 `_seq_lens_cpu` 也不是 DFlash 调整后的值，但此时序列长度变化较小（decode 每步只增加 1 token），影响较小。

### 次要 Bug：`context_slot_mapping` 使用 int32 而非 int64

**文件**: `vllm_ascend/spec_decode/dflash_proposer.py`，第 31-41 行

- Ascend 版本的 `_context_slot_mapping_buffer`、`_slot_mapping_buffer`、`_context_positions_buffer`、`positions` 均使用 `torch.int32`
- 上游版本使用 `torch.int64`
- 对于 `max_model_len=65536`、`block_size=128`，slot 值最大约 65536，在 int32 范围内不会溢出
- 但对于更大的模型配置可能存在溢出风险
- `do_kv_cache_update` 中的 `reshape_and_cache` 期望 `slot_mapping` 为 int64，传入 int32 可能导致类型不匹配

### 需要进一步排查的方向

由于 `_seq_lens_cpu` 在 `parallel_drafting` 路径下被覆盖，单纯过期可能不是首请求复读的直接原因。还需要排查：

1. **`precompute_and_store_context_kv` patch 的精度差异**：Ascend patch 使用 `self.hidden_norm()` 替代 `ops.rms_norm()`，使用 `rotary_emb()` 替代 `ops.rotary_embedding()`。虽然 RoPE 是原地修改的，但 `hidden_norm` 和 `k_norm_layer` 的数值精度可能与上游不同。

2. **首请求 KV cache 初始化**：首请求时 draft model 的 KV cache 可能未正确初始化，`do_kv_cache_update` 中的 `self.key_cache` 延迟绑定可能有问题。

3. **`_dflash_hidden_states` 缓冲区**：Ascend 版本复制到预分配的 `torch.zeros` 缓冲区，上游直接引用 target tensor。虽然只读 `[:num_context]`，但缓冲区分配方式不同可能影响内存布局。

## 修复计划

### Step 1：修复 `set_inputs_first_pass` — 创建新对象而非原地修改

**文件**: `vllm_ascend/spec_decode/dflash_proposer.py`

将第 129-148 行（原地修改 + return）替换为创建新的 `AscendCommonAttentionMetadata` 对象，对齐上游模式：

```python
new_cad = AscendCommonAttentionMetadata(
    query_start_loc=new_query_start_loc,
    query_start_loc_cpu=(
        torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * num_query_per_req
    ).to(torch.int32),
    seq_lens_cpu=None,              # 与上游一致，设为 None
    _seq_lens_cpu=None,             # 与上游一致，设为 None
    _num_computed_tokens_cpu=None,  # 与上游一致，设为 None
    seq_lens_cpu_upper_bound=(
        cad.seq_lens_cpu_upper_bound + num_query_per_req
        if cad.seq_lens_cpu_upper_bound is not None else None
    ),
    num_reqs=cad.num_reqs,
    num_actual_tokens=num_query_total,
    max_query_len=num_query_per_req,
    max_seq_len=cad.max_seq_len + num_query_per_req,
    block_table_tensor=cad.block_table_tensor,
    slot_mapping=query_slot_mapping,
    causal=False,
    attn_state=AscendAttentionState.ChunkedPrefill,
    decode_token_per_req=num_query_per_req,
    actual_seq_lengths_q=[num_query_per_req] * batch_size,
    positions=cad.positions,
    positions_cpu=cad.positions_cpu,
    kvcomp_metadata=cad.kvcomp_metadata,
)
return num_query_total, token_indices_to_sample, new_cad, None
```

**关键变更**：
- `_seq_lens_cpu=None`、`seq_lens_cpu=None`：强制下游使用 GPU `seq_lens`（与上游一致）
- `_num_computed_tokens_cpu=None`、`num_computed_tokens_cpu=None`：避免过期值
- `causal=False`、`attn_state=ChunkedPrefill`：DFlash 需要的设置
- 从原 `cad` 携带 `block_table_tensor`、`positions`、`kvcomp_metadata` 等

### Step 2：修复 `context_slot_mapping` 数据类型 int32 → int64

**文件**: `vllm_ascend/spec_decode/dflash_proposer.py`

```python
self._context_slot_mapping_buffer = torch.zeros(
    self.max_num_tokens,
    dtype=torch.int64,  # int32 → int64，与上游一致
    device=device,
)
self._slot_mapping_buffer = torch.zeros(
    self.max_query_tokens,
    dtype=torch.int64,  # int32 → int64
    device=device,
)
self._context_positions_buffer = torch.zeros(
    self.max_num_tokens,
    dtype=torch.int64,  # int32 → int64
    device=device,
)
self.positions = torch.zeros(
    self.max_query_tokens,
    dtype=torch.int64,  # int32 → int64
    device=device,
)
```

### Step 3：添加诊断日志（用于验证根因）

由于 `_seq_lens_cpu` 在 `parallel_drafting` 路径下被覆盖，单纯过期可能不是直接原因。需要添加日志进一步排查。

#### 3.1 在 `set_inputs_first_pass` 中添加日志

```python
logger.info(
    "[DFLASH_DIAG] set_inputs_first_pass: batch_size=%d, num_context=%d, "
    "num_query_total=%d, has_num_rejected=%s, "
    "cad._seq_lens_cpu=%s, cad.seq_lens_cpu=%s, "
    "cad.seq_lens(GPU)=%s, new_seq_lens=%s",
    batch_size, num_context, num_query_total,
    has_num_rejected,
    cad._seq_lens_cpu[:batch_size] if cad._seq_lens_cpu is not None else None,
    cad.seq_lens_cpu[:batch_size] if cad.seq_lens_cpu is not None else None,
    cad.seq_lens[:batch_size],
    (effective_seq_lens + num_query_per_req)[:batch_size],
)
```

#### 3.2 在 `precompute_and_store_context_kv` 中添加日志

在 `patch_qwen3_dflash.py` 的 RoPE 调用前后添加日志，验证 RoPE 是否正确应用：

```python
# 在 rotary_emb 调用前
k_norm_before = all_k_flat.norm().item()
self.layers[0].self_attn.rotary_emb(positions_repeated, all_k_flat, tmpv)
k_norm_after = all_k_flat.norm().item()
logger.info(
    "[DFLASH_DIAG] RoPE: k_norm_before=%.4f, k_norm_after=%.4f, "
    "changed=%s, num_ctx=%d, L=%d",
    k_norm_before, k_norm_after,
    abs(k_norm_before - k_norm_after) > 0.001,
    num_ctx, L,
)
```

#### 3.3 在 `build()` 中添加日志

在 `attention_v1.py` 的 `build()` 方法中，针对 DFlash 场景添加日志：

```python
if self.speculative_config and self.speculative_config.parallel_drafting:
    logger.info(
        "[DFLASH_DIAG] build: seq_lens(GPU)=%s, _seq_lens_cpu=%s, "
        "seq_lens_cpu=%s, actual_seq_lengths_kv=%s, "
        "actual_seq_lengths_q=%s",
        common_attn_metadata.seq_lens[:num_reqs],
        common_attn_metadata._seq_lens_cpu[:num_reqs] if common_attn_metadata._seq_lens_cpu is not None else None,
        common_attn_metadata.seq_lens_cpu[:num_reqs] if common_attn_metadata.seq_lens_cpu is not None else None,
        seq_lens.tolist(),
        query_start_loc_cpu[1:].tolist(),
    )
```

## 验证方法

1. 添加诊断日志后，在 NPU 上运行首请求测试，收集日志
2. 分析日志确认：
   - `_seq_lens_cpu` 是否过期
   - `actual_seq_lengths_kv` 是否正确
   - RoPE 是否正确应用（k_norm 是否变化）
3. 实施 Step 1 和 Step 2 的修复
4. 重新运行测试，验证首请求无复读
5. 验证后续请求仍然正常

## 涉及文件

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `vllm_ascend/spec_decode/dflash_proposer.py` | Bug 修复 | 创建新 cad 对象 + int32→int64 |
| `vllm_ascend/patch/worker/patch_qwen3_dflash.py` | 诊断日志 | 验证 RoPE 是否正确应用 |
| `vllm_ascend/attention/attention_v1.py` | 诊断日志 | 验证 attention metadata 是否正确 |
