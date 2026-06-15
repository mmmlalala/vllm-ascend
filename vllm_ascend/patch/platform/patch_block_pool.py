import os

from vllm.logger import logger
from vllm.v1.core.block_pool import BlockPool


_original_init = BlockPool.__init__


def _patched_init(
    self,
    num_gpu_blocks: int,
    enable_caching: bool,
    hash_block_size: int,
    enable_kv_cache_events: bool = False,
    metrics_collector=None,
):
    _original_init(
        self,
        num_gpu_blocks,
        enable_caching,
        hash_block_size,
        enable_kv_cache_events,
        metrics_collector,
    )

    skip_low_blocks_env = os.environ.get("VLLM_ASCEND_SKIP_LOW_BLOCKS")
    if not skip_low_blocks_env:
        return

    try:
        skip_low_blocks = int(skip_low_blocks_env)
    except ValueError:
        logger.warning(
            "Invalid VLLM_ASCEND_SKIP_LOW_BLOCKS=%r, skip disabled.",
            skip_low_blocks_env,
        )
        return

    if skip_low_blocks <= 1:
        return

    # Reserve (skip_low_blocks - 1) low blocks so that the first real
    # allocation starts at block_id = skip_low_blocks - 1.  Low-numbered
    # blocks on Ascend NPU may contain stale device memory that corrupts
    # FIA attention output, causing token repetition on the first request.
    reserve_count = min(skip_low_blocks - 1, self.free_block_queue.num_free_blocks)
    self._reserved_low_blocks = self.free_block_queue.popleft_n(reserve_count)
    for block in self._reserved_low_blocks:
        assert block.ref_cnt == 0
        block.ref_cnt += 1

    if self._reserved_low_blocks:
        logger.warning(
            "VLLM_ASCEND_SKIP_LOW_BLOCKS=%d reserved low blocks "
            "[%d, %d], first alloc should be %d.",
            skip_low_blocks,
            self._reserved_low_blocks[0].block_id,
            self._reserved_low_blocks[-1].block_id,
            self._reserved_low_blocks[-1].block_id + 1,
        )

    if reserve_count != skip_low_blocks - 1:
        logger.warning(
            "VLLM_ASCEND_SKIP_LOW_BLOCKS=%d requested but only "
            "%d low blocks were reserved.",
            skip_low_blocks,
            reserve_count,
        )


BlockPool.__init__ = _patched_init
