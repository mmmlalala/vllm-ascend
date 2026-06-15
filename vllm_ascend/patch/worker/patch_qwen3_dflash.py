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

    # [DFLASH_DIAG] Log input stats (context_states is small enough to CPU)
    try:
        cs_cpu = context_states[:64].float().cpu()
        logger.info(
            "[DFLASH_DIAG] precompute_kv input: num_ctx=%d, L=%d, "
            "context_states_norm=%.4f, context_states_has_nan=%s, "
            "context_positions[:5]=%s, slot_mapping_dtype=%s",
            num_ctx, L,
            cs_cpu.norm().item(),
            torch.isnan(cs_cpu).any().item(),
            context_positions[:5].tolist(),
            context_slot_mapping.dtype if context_slot_mapping is not None else None,
        )
    except Exception as e:
        logger.info("[DFLASH_DIAG] precompute_kv input stat failed: %s", e)

    # --- Fused KV projection (one GEMM for all layers) ---
    normed_context_states = self.hidden_norm(context_states)
    all_kv_flat = F.linear(normed_context_states, self._fused_kv_weight, self._fused_kv_bias)
    # Single contiguous copy that separates K/V and transposes to
    # layer-major layout.  Result: [2, L, num_ctx, nkv, hd] contiguous.
    # Indexing dim-0 gives contiguous [L, num_ctx, nkv, hd] for K and V.
    all_kv = all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(2, 1, 0, 3, 4).contiguous()
    all_k = all_kv[0]  # [L, num_ctx, nkv, hd], contiguous
    all_v = all_kv[1]  # [L, num_ctx, nkv, hd], contiguous

    # [DFLASH_DIAG] Log KV projection output (small slice to CPU)
    try:
        k_cpu = all_k[0, :64].reshape(64, -1).float().cpu()
        v_cpu = all_v[0, :64].reshape(64, -1).float().cpu()
        logger.info(
            "[DFLASH_DIAG] precompute_kv after KV proj layer0: "
            "k_norm=%.4f, k_has_nan=%s, v_norm=%.4f, v_has_nan=%s",
            k_cpu.norm().item(), torch.isnan(k_cpu).any().item(),
            v_cpu.norm().item(), torch.isnan(v_cpu).any().item(),
        )
    except Exception as e:
        logger.info("[DFLASH_DIAG] precompute_kv after KV proj stat failed: %s", e)

    # --- Per-layer RMSNorm K (3D: [num_ctx, nkv, hd] per layer) ---
    all_k_normed = torch.empty_like(all_k)
    for i in range(L):
        k_norm_layer = self.layers[i].self_attn.k_norm
        all_k_normed[i] = k_norm_layer(all_k[i])

    # [DFLASH_DIAG] Log k_norm output (small slice to CPU)
    try:
        kn_cpu = all_k_normed[0, :64].reshape(64, -1).float().cpu()
        logger.info(
            "[DFLASH_DIAG] precompute_kv after k_norm layer0: "
            "k_normed_norm=%.4f, k_normed_has_nan=%s",
            kn_cpu.norm().item(), torch.isnan(kn_cpu).any().item(),
        )
    except Exception as e:
        logger.info("[DFLASH_DIAG] precompute_kv after k_norm stat failed: %s", e)

    # --- Fused RoPE across all layers ---
    # View as [L * num_ctx, kv] so RoPE sees one big batch (no copy).
    # In-place RoPE: pass K as the "query" arg with key=None.
    all_k_flat = all_k_normed.view(L * num_ctx, kv)
    positions_repeated = context_positions.repeat(L)
    tmpv = all_k_flat.clone()
    self.layers[0].self_attn.rotary_emb(positions_repeated, all_k_flat, tmpv)
    # [DFLASH_DIAG] Log K after RoPE (small slice to CPU)
    try:
        kr_cpu = all_k_flat[:64].reshape(64, -1).float().cpu()
        logger.info(
            "[DFLASH_DIAG] precompute_kv after RoPE: "
            "k_rope_norm=%.4f, k_rope_has_nan=%s",
            kr_cpu.norm().item(), torch.isnan(kr_cpu).any().item(),
        )
    except Exception as e:
        logger.info("[DFLASH_DIAG] precompute_kv after RoPE stat failed: %s", e)

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


DFlashQwen3Model.precompute_and_store_context_kv = precompute_and_store_context_kv
