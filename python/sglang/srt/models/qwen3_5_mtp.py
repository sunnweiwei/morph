# Copyright 2023-2024 SGLang Team
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
# ==============================================================================

"""Inference-only Qwen3_5 MTP model."""

import copy
import logging
from contextlib import ExitStack
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group
from sglang.srt.distributed.device_communicators import triton_symm_mem_ag
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
from sglang.srt.runtime_context import get_flags, get_parallel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_npu

logger = logging.getLogger(__name__)


class Qwen3_5ForCausalLMMTP(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)

        self.is_multimodal = hasattr(config, "text_config")
        if self.is_multimodal:
            config = config.text_config

        # Deep-copy so MTP mutations below don't leak into the target's config.
        config = copy.deepcopy(config)

        # The MTP model is unquantized in the nvfp4 checkpoint.
        if quant_config and quant_config.get_name() in (
            "modelopt_fp4",
            "modelopt_mixed",
        ):
            quant_config = None
        if (
            is_npu()
            and get_global_server_args().speculative_draft_model_quantization is None
        ):
            quant_config = None

        # Quark-quantized Qwen3.5 MXFP4 checkpoints ship the MTP module in
        # bf16; every `mtp.*` layer appears under the quantization exclude
        # list. Detect that and skip quantization here so linear/MoE weight
        # loaders allocate bf16 shapes (see sgl-project/sglang#23113).
        if quant_config and quant_config.get_name() == "quark":
            exclude_layers = getattr(quant_config, "exclude_layers", [])
            if any(
                isinstance(layer, str) and layer.startswith("mtp.")
                for layer in exclude_layers
            ):
                quant_config = None

        self.config = config
        self.tp_size = get_parallel().tp_size
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        RMSNorm_cls = GemmaRMSNorm
        self.pre_fc_norm_embedding = RMSNorm_cls(
            config.hidden_size, config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = RMSNorm_cls(config.hidden_size, config.rms_norm_eps)
        mtp_config = copy.deepcopy(config)
        mtp_config.num_hidden_layers = 1
        mtp_config.full_attention_interval = 1
        self.model = Qwen3_5ForCausalLM(
            mtp_config,
            quant_config,
            prefix=add_prefix("mtp", prefix),
            is_nextn=True,
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )

        self.logits_processor = LogitsProcessor(config)
        self._online_mtp_stats_gatherer = triton_symm_mem_ag.MultimemAllGatherer(
            max_tokens=triton_symm_mem_ag.recommended_max_tokens(
                include_prefill=False, floor=128
            ),
            enabled=self.tp_size > 1,
            # Every draft step contains intervening TP collectives in the MTP
            # transformer, satisfying the same contract as the logits gatherer.
            skip_entry_sync=True,
        )

    def prepare_online_local_vocab_ce(self) -> None:
        """Build the tiny symmetric statistics collective before graph capture."""

        indices = self.lm_head.shard_indices
        local_vocab = indices.org_vocab_end_index - indices.org_vocab_start_index
        if (
            indices.num_added_elements
            or indices.num_added_vocab_padding
            or local_vocab != self.lm_head.weight.shape[0]
        ):
            raise ValueError(
                "online local-vocabulary CE requires equal contiguous base-vocab shards"
            )
        if self.logits_processor.final_logit_softcapping:
            raise ValueError(
                "online local-vocabulary CE does not support logit softcapping"
            )
        packet = torch.zeros(
            (1, 4), dtype=torch.float32, device=self.lm_head.weight.device
        )
        self._online_mtp_stats_gatherer(packet.view(torch.bfloat16))

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        text_config = getattr(config, "text_config", config)
        return ModelConfigForExpertLocation(
            num_layers=text_config.num_hidden_layers,
            num_logical_experts=text_config.num_experts,
            num_groups=None,
        )

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        if not self.config.tie_word_embeddings:
            del self.lm_head.weight

        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def set_lm_head_from_target(self, target_lm_head):
        if self.config.tie_word_embeddings:
            return

        self.lm_head = target_lm_head

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        exit_stack = ExitStack()
        online_mtp_tap = getattr(forward_batch, "online_mtp_tap", None)
        if (
            is_npu()
            and self.quant_config is None
            and get_flags().quantization is not None
        ):
            # ascend mtp unquant
            exit_stack.enter_context(envs.SGLANG_DEEPEP_BF16_DISPATCH.override(True))
            exit_stack.enter_context(
                envs.DEEP_NORMAL_MODE_USE_INT8_QUANT.override(False)
            )

        try:
            assert input_embeds is None
            input_embeds = forward_batch.mm_input_embeds
            if (
                forward_batch.forward_mode.is_extend()
                and forward_batch.contains_mm_inputs()
                and not forward_batch.forward_mode.is_draft_extend_v2()
            ):
                assert input_embeds is not None
                last_indices = (
                    forward_batch.extend_start_loc + forward_batch.extend_seq_lens - 1
                ).long()
                input_embeds[last_indices] = self.model.embed_tokens(
                    input_ids[last_indices]
                )

            if input_embeds is None:
                input_embeds = self.model.embed_tokens(input_ids)

            hidden_states = forward_batch.spec_info.hidden_states

            if online_mtp_tap is not None:
                online_mtp_tap.record("input_embedding", input_embeds)
                online_mtp_tap.record("target_hidden", hidden_states)

            if not forward_batch.forward_mode.is_idle():
                input_embeds = self.pre_fc_norm_embedding(input_embeds)
                hidden_states = self.pre_fc_norm_hidden(hidden_states)
            hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)

            if online_mtp_tap is not None:
                online_mtp_tap.record("fc_input", hidden_states)

            hidden_states = self.fc(hidden_states)

            if online_mtp_tap is not None:
                online_mtp_tap.record("fc_output", hidden_states)

            with get_global_expert_distribution_recorder().disable_this_region():
                hidden_states = self.model(
                    input_ids,
                    positions,
                    forward_batch,
                    hidden_states,
                )
        finally:
            exit_stack.close()

        local_vocab_ce = bool(
            online_mtp_tap is not None and online_mtp_tap.local_vocab_ce
        )
        output = self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            local_vocab_only=local_vocab_ce,
        )
        if online_mtp_tap is not None:
            if online_mtp_tap.async_ce_producer or online_mtp_tap.defer_ce:
                # Preserve inference logits until labels arrive, then process
                # them on the dedicated online-learning stream. The async path
                # starts its label-independent producer immediately after the
                # graph-stable snapshot, overlapping target verification.
                online_mtp_tap.record("ce_logits", output.next_token_logits)
            else:
                if online_mtp_tap.group_ce_projection:
                    from sglang.srt.speculative.online_mtp_training import (
                        vocab_parallel_ce_forward_probability,
                    )

                    indices = self.lm_head.shard_indices
                    fp8_ce_projection = online_mtp_tap.fp8_ce_weight is not None
                    probability_dtype = (
                        torch.float8_e4m3fn
                        if fp8_ce_projection
                        else self.lm_head.weight.dtype
                    )
                    probability_out = online_mtp_tap.reserve_ce_probability(
                        output.next_token_logits.shape[0],
                        (indices.org_vocab_end_index - indices.org_vocab_start_index),
                        dtype=probability_dtype,
                        device=output.next_token_logits.device,
                    )
                    if local_vocab_ce:
                        from sglang.srt.speculative.triton_ops.online_mtp_lse import (
                            compact_global_vocab_stats,
                            compact_local_vocab_stats,
                            compact_vocab_probability,
                        )

                        local_logits = output.next_token_logits
                        if local_logits.dtype != self.lm_head.weight.dtype:
                            raise RuntimeError(
                                "local-vocabulary CE logits lost their inference dtype"
                            )
                        packed_stats = compact_local_vocab_stats(
                            local_logits,
                            global_vocab_start=indices.org_vocab_start_index,
                        )
                        gathered_stats = self._online_mtp_stats_gatherer(
                            packed_stats.view(torch.bfloat16)
                        ).view(torch.float32)
                        logsumexp, greedy_index = compact_global_vocab_stats(
                            gathered_stats, tp_size=self.tp_size
                        )
                        probability = compact_vocab_probability(
                            local_logits,
                            logsumexp,
                            local_start=0,
                            local_stop=local_logits.shape[1],
                            output_dtype=probability_dtype,
                            out=probability_out,
                            store_scale=(448.0 if fp8_ce_projection else 1.0),
                        )
                        online_mtp_tap.record("ce_logsumexp", logsumexp)
                        online_mtp_tap.record_ce_probability(probability)
                        online_mtp_tap.record_ce_argmax(greedy_index)
                    elif online_mtp_tap.group_ce_steps:
                        # Keep the three graph-local logits alive only until
                        # the D4 loop finishes. The grouped producer then
                        # launches one LSE pair and one probability kernel for
                        # all steps, writing directly into ``probability_out``.
                        online_mtp_tap.record_grouped_ce_logits(
                            output.next_token_logits
                        )
                    elif online_mtp_tap.fuse_ce_argmax:
                        from sglang.srt.speculative.triton_ops.online_mtp_lse import (
                            compact_vocab_logsumexp_argmax,
                            compact_vocab_probability,
                        )

                        logsumexp, greedy_index = compact_vocab_logsumexp_argmax(
                            output.next_token_logits
                        )
                        probability = compact_vocab_probability(
                            output.next_token_logits,
                            logsumexp,
                            local_start=indices.org_vocab_start_index,
                            local_stop=indices.org_vocab_end_index,
                            output_dtype=probability_dtype,
                            out=probability_out,
                            store_scale=(448.0 if fp8_ce_projection else 1.0),
                        )
                        online_mtp_tap.record("ce_logsumexp", logsumexp)
                        online_mtp_tap.record_ce_probability(probability)
                        online_mtp_tap.record_ce_argmax(greedy_index)
                    else:
                        logsumexp, probability = (
                            vocab_parallel_ce_forward_probability(
                                output.next_token_logits,
                                self.lm_head,
                                use_triton_lse=online_mtp_tap.use_triton_lse,
                                use_triton_expcast=online_mtp_tap.use_triton_expcast,
                                out=probability_out,
                                output_dtype=probability_dtype,
                                probability_store_scale=(
                                    448.0 if fp8_ce_projection else 1.0
                                ),
                            )
                        )
                        online_mtp_tap.record("ce_logsumexp", logsumexp)
                        online_mtp_tap.record_ce_probability(probability)
                else:
                    from sglang.srt.speculative.online_mtp_training import (
                        vocab_parallel_ce_forward_stats,
                    )

                    logsumexp, expected_weight = vocab_parallel_ce_forward_stats(
                        output.next_token_logits,
                        self.lm_head,
                        use_triton_lse=online_mtp_tap.use_triton_lse,
                        use_triton_expcast=online_mtp_tap.use_triton_expcast,
                    )
                    online_mtp_tap.record("ce_logsumexp", logsumexp)
                    online_mtp_tap.record("ce_expected_weight", expected_weight)
        return output

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for MoE experts (non-fused/fused)
        num_experts = getattr(self.config, "num_experts", None)
        if num_experts is not None:
            expert_params_mapping = FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=num_experts,
            )
        else:
            expert_params_mapping = []

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )

        # fused experts: experts.w13_weight / experts.w2_weight
        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]

        def load_fused_expert_weights(
            name: str,
            params_dict: dict,
            loaded_weight: torch.Tensor,
            shard_id: str,
            num_experts: int,
        ):
            param = params_dict[name]
            weight_loader = param.weight_loader
            # Let EP MoE layer handle expert_ids that do not belong to local moe rank
            for expert_id in range(num_experts):
                curr_expert_weight = loaded_weight[expert_id]
                weight_loader(
                    param,
                    curr_expert_weight,
                    name,
                    shard_id,
                    expert_id,
                )
            return True

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Only process MTP branch weights
            if "mtp" not in name:
                continue

            if name.startswith("mtp."):
                # Remove the mtp. prefix for processing
                name = name.replace("mtp.", "model.")

                name = name.replace("model.fc", "fc")
                name = name.replace("model.pre_fc", "pre_fc")

            if ".self_attn." in name:
                name = name.replace(".self_attn", "")

            # 1) Process stacked parameters (q_proj/k_proj/v_proj & gate_proj/up_proj)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Check if this is a fused expert weight
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                # Skip non-matching weights
                if weight_name not in name:
                    continue

                # Skip MoE experts.* here, handled separately below
                if "mlp.experts" in name:
                    continue

                name_mapped = name.replace(weight_name, param_name)

                # Skip loading extra parameters for GPTQ/modelopt models.
                if (
                    name_mapped.endswith(ignore_suffixes)
                    and name_mapped not in params_dict
                ):
                    continue

                if name_mapped not in params_dict:
                    continue

                param = params_dict[name_mapped]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                name = name_mapped
                break
            else:
                # 2) Process MoE expert weights (including fused experts)
                is_expert_weight = False

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)

                    # Fused experts: single checkpoint weight contains multiple experts
                    if is_fused_expert and num_experts is not None:
                        if "experts.gate_up_proj" in name:
                            # gate_up_proj fused: split into w1 / w3
                            loaded_w1, loaded_w3 = loaded_weight.chunk(2, dim=-2)
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_w1,
                                "w1",
                                num_experts,
                            )
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_w3,
                                "w3",
                                num_experts,
                            )
                        else:
                            # down_proj fused: distribute entire weight
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                    else:
                        # Non-fused expert, load by expert_id/shard
                        if (
                            name_mapped.endswith(ignore_suffixes)
                            and name_mapped not in params_dict
                        ):
                            continue
                        if name_mapped not in params_dict:
                            break
                        param = params_dict[name_mapped]
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    name = name_mapped
                    break
                else:
                    # Skip expert weight if not handled by current rank
                    if is_expert_weight:
                        continue

                    # 3) Regular non-stacked / non-expert parameters, use default loader
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue

                    if name in params_dict:
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning_once(
                            f"Parameter {name} not found in params_dict, skip loading"
                        )

            loaded_params.add(name)
        return loaded_params


EntryClass = [Qwen3_5ForCausalLMMTP]
