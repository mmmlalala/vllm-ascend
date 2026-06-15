import torch
import torch.nn.functional as F
from vllm.logger import logger
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model


def precompute_and_store_context_kv(
    self,
    context_states: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor | None = None,
) -> None:
    if not hasattr(self, "_num_attn_layers"):
        self._build_fused_kv_buffers()

    num_ctx = context_states.shape[0]
    L = self._num_attn_layers
    kv = self._kv_size
    hd = self._head_dim
    nkv = self._num_kv_heads

    # --- Fused KV projection (one GEMM for all layers) ---
    normed_context_states = self.hidden_norm(context_states)
    all_kv_flat = F.linear(normed_context_states, self._fused_kv_weight, self._fused_kv_bias)
    # Single contiguous copy that separates K/V and transposes to
    # layer-major layout.  Result: [2, L, num_ctx, nkv, hd] contiguous.
    # Indexing dim-0 gives contiguous [L, num_ctx, nkv, hd] for K and V.
    all_kv = all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(2, 1, 0, 3, 4).contiguous()
    all_k = all_kv[0]  # [L, num_ctx, nkv, hd], contiguous
    all_v = all_kv[1]  # [L, num_ctx, nkv, hd], contiguous

    # --- Per-layer RMSNorm K (3D: [num_ctx, nkv, hd] per layer) ---
    all_k_normed = torch.empty_like(all_k)
    for i in range(L):
        k_norm_layer = self.layers[i].self_attn.k_norm
        all_k_normed[i] = k_norm_layer(all_k[i])

    # --- Fused RoPE across all layers ---
    # View as [L * num_ctx, kv] so RoPE sees one big batch (no copy).
    # In-place RoPE: pass K as the "query" arg with key=None.
    all_k_flat = all_k_normed.view(L * num_ctx, kv)
    positions_repeated = context_positions.repeat(L)
    tmpv = all_k_flat.clone()
    self.layers[0].self_attn.rotary_emb(positions_repeated, all_k_flat, tmpv)

    if context_slot_mapping is None:
        return

    # --- Per-layer cache insert ---
    all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
    for i in range(L):
        attn = self._attn_layers[i]
        kv_cache = attn.kv_cache
        attn.impl.do_kv_cache_update(
            attn,
            all_k_final[i],
            all_v[i],
            kv_cache,
            context_slot_mapping,
        )

    # [DFLASH_DIAG] Verify KV cache write: read back key_cache at the
    # written slots and compare with the K we just wrote.
    # Only check layer 0 to keep overhead low.
    key_cache = self._attn_layers[0].impl.key_cache
    if key_cache is not None:
        block_size = key_cache.shape[1]
        slots = context_slot_mapping[:min(num_ctx, 5)]
        block_ids = slots // block_size
        block_offsets = slots % block_size
        # Read back the first few slots from key_cache
        readback_vals = []
        for s_idx in range(slots.shape[0]):
            bid = block_ids[s_idx].item()
            boff = block_offsets[s_idx].item()
            # key_cache shape: [num_blocks, block_size, nkv, hd]
            readback_k = key_cache[bid, boff, :, :].detach().cpu()
            expected_k = all_k_final[0, s_idx, :, :].detach().cpu()
            diff_norm = (readback_k - expected_k).norm().item()
            readback_vals.append(
                "slot=%d:block=%d:off=%d:diff_norm=%.6f"
                % (slots[s_idx].item(), bid, boff, diff_norm)
            )
        logger.info(
            "[DFLASH_DIAG] precompute_kv cache_verify: num_ctx=%d, L=%d, "
            "slot_mapping[:5]=%s, slot_dtype=%s, cache_readback=[%s]",
            num_ctx, L,
            context_slot_mapping[:5].tolist(),
            context_slot_mapping.dtype,
            ", ".join(readback_vals),
        )


DFlashQwen3Model.precompute_and_store_context_kv = precompute_and_store_context_kv
