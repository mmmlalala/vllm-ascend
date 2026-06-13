#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Patch Scheduler._update_waiting_for_remote_kv() to handle last_token_id
# from the prefill instance. This is critical for reasoning models (e.g.
# Qwen3.5 with enable_thinking) where the prefill instance generates a
# reasoning token that must be preserved in the decode instance's output.
#
# The prefill instance generates 1 token (max_tokens=1) and its KV cache
# is transferred to the decode instance. However, the decode instance
# only knows about the prompt tokens, not the prefill's output token.
# Without this patch, the decode instance recomputes the last prompt token
# and may generate a different first token (e.g., the thinking end token
# instead of a reasoning token), causing reasoning_content to be empty.
#
# With this patch, the decode instance appends the prefill's output token
# (last_token_id) to its prompt_token_ids and adjusts num_computed_tokens
# accordingly. The KV cache for this token was already transferred from
# the prefill instance. The decode instance then recomputes this token
# (as the last prompt token) to get logits for sampling the next token.
# The detokenizer will correctly decode the output because it uses the
# prompt_token_ids (which now include last_token_id) as context.
#
# NOTE: The text of last_token_id itself is not included in the output
# because it's treated as a prompt token. This means the first reasoning
# token's text may be missing from the reasoning_content. However, this
# is a minor issue compared to having completely empty reasoning_content.
# A future improvement could inject last_token_id into the output tokens
# and update the detokenizer accordingly.
#

from vllm.logger import logger
from vllm.v1.core.sched.scheduler import Scheduler

_original_update_waiting_for_remote_kv = Scheduler._update_waiting_for_remote_kv


def _patched_update_waiting_for_remote_kv(self, request):
    """
    Patched version of _update_waiting_for_remote_kv that handles
    last_token_id from the prefill instance.

    When the prefill instance generates a token (e.g., a reasoning token
    for Qwen3.5 with enable_thinking=True), this token must be preserved
    in the decode instance's output. The last_token_id is passed through
    kv_transfer_params from the prefill instance via the metaserver.

    We append last_token_id to the request's prompt_token_ids so that:
    1. The decode instance knows about this token
    2. The KV cache for this token (already transferred) is correctly
       accounted for
    3. The decode instance recomputes this token (as the last prompt
       token) to get logits for the next token
    4. The detokenizer uses this token as context for decoding
    """
    # Check for last_token_id BEFORE calling the original method,
    # because the original method adjusts num_computed_tokens.
    params = request.kv_transfer_params
    last_token_id = None
    if params and "last_token_id" in params:
        last_token_id = params["last_token_id"]
        # Remove last_token_id from params to avoid re-processing
        del params["last_token_id"]

    _original_update_waiting_for_remote_kv(self, request)

    if last_token_id is not None:
        # Append the prefill instance's output token to the request's
        # prompt_token_ids. This makes the decode instance aware of
        # the prefill's generated token, so it can start generating
        # from the correct position.
        #
        # The KV cache for this token has already been transferred
        # from the prefill instance (it's included in the KV transfer
        # because the prefill instance's computed_tokens includes
        # the output token).
        #
        # By adding it to prompt_token_ids, the decode instance will
        # recompute this token (as the last prompt token) to get logits
        # for sampling the next token. The detokenizer will also use
        # this token as context for decoding the subsequent output.
        request.prompt_token_ids.append(last_token_id)
        request._all_token_ids.append(last_token_id)
        request.num_prompt_tokens += 1

        # Since we added a new prompt token and its KV cache is already
        # available from the transfer, num_computed_tokens should be
        # adjusted. The original method set num_computed_tokens to
        # num_tokens - 1 (for recomputing the last token). Now that
        # we've added the last_token_id as a prompt token, we need
        # to re-adjust:
        # - num_tokens is now num_prompt_tokens (including last_token_id)
        # - num_computed_tokens should be num_tokens - 1 (recompute
        #   last_token_id to get logits for the next token)
        request.num_computed_tokens = request.num_tokens - 1

        logger.info(
            "Appended last_token_id=%s from prefill to request %s "
            "(num_prompt_tokens=%d, num_computed_tokens=%d)",
            last_token_id,
            request.request_id,
            request.num_prompt_tokens,
            request.num_computed_tokens,
        )


Scheduler._update_waiting_for_remote_kv = _patched_update_waiting_for_remote_kv
