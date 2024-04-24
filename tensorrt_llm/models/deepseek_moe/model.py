# Author: Chengdong Liang
# Date: 2024-04-23

from ..._utils import pad_vocab_size
from ...functional import PositionEmbeddingType, Tensor, ACT2FN, is_gated_activation, non_gated_version
from ...layers import (MLP, Attention, AttentionMaskType, ColumnLinear,
                       Embedding, LayerNorm, GatedMLP, FusedGatedMLP, MOE, MoeConfig)
from ...module import Module
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              PretrainedConfig)
from ...top_model_mixin import TopModelMixin
from ...functional import recv, send
from ...plugin import init_all_reduce_helper


class DeepseekMoeDecoderLayer(Module):
    def __init__(self, config: PretrainedConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        self.input_layernorm = LayerNorm(normalized_shape=config.hidden_size,
                                         eps=config.norm_epsilon,
                                         dtype=config.dtype)
        layers_range = config.mapping.pp_layers(config.num_hidden_layers)


        # hidden_act = non_gated_version(config.hidden_act)

        local_layer_idx = layer_idx - layers_range[0]

        self.attention = Attention(
            local_layer_idx=local_layer_idx,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            dtype=config.dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=config.bias,
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            rotary_embedding_base=config.rotary_base,
            rotary_embedding_scaling=config.rotary_scaling,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            quant_mode=config.quant_mode,
        )

        ClsMLP = GatedMLP
        self.use_shared = False
        self.ffn_dim = config.intermediate_size
        self.hidden_act = "silu"
        mlp_kwargs = {}
        # moe模型
        if config.moe_config['num_experts'] > 1 and layer_idx >= config.moe_first_k_dense_replace:
            ClsMLP = MOE
            mlp_kwargs = {
                "moe_config":
                MoeConfig(
                    config.moe_config['num_experts'],
                    config.moe_config['top_k'],
                    config.moe_config['tp_mode'],
                    config.moe_config['normalization_mode'],
                ),
                "tp_rank": config.mapping.tp_rank,
            }
            self.ffn_dim = config.moe_intermediate_size
            self.hidden_act = config.hidden_act
            if config.moe_shared_num_experts > 0:
                self.use_shared = True

        print("ffn_dim: ", self.ffn_dim)
        self.mlp = ClsMLP(hidden_size=config.hidden_size,
                          ffn_hidden_size=self.ffn_dim,
                          hidden_act=self.hidden_act,
                          dtype=config.dtype,
                          bias=config.bias,
                          tp_group=config.mapping.tp_group,
                          tp_size=config.mapping.tp_size,
                          quant_mode=config.quant_mode,
                          **mlp_kwargs)

        if self.use_shared:
            self.shared_experts = GatedMLP(
                hidden_size=config.hidden_size,
                ffn_hidden_size=self.ffn_dim * config.moe_shared_num_experts,
                hidden_act="silu",
                dtype=config.dtype,
                bias=config.bias,
                tp_group=config.mapping.tp_group,
                tp_size=config.mapping.tp_size,
                quant_mode=config.quant_mode
            )

        self.post_layernorm = LayerNorm(normalized_shape=config.hidden_size,
                                        eps=config.norm_epsilon,
                                        dtype=config.dtype)

    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None):

        assert isinstance(hidden_states, Tensor)

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        attention_output = self.attention(hidden_states,
                                          attention_mask=attention_mask,
                                          use_cache=use_cache,
                                          kv_cache_params=kv_cache_params,
                                          attention_params=attention_params)

        if use_cache:
            attention_output, presents = attention_output

        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        tmp_hidden_states = self.mlp(hidden_states)
        if self.use_shared:
            out_hidden_states = tmp_hidden_states + self.shared_experts(hidden_states)
        else:
            out_hidden_states = tmp_hidden_states

        out_hidden_states = residual + out_hidden_states

        if use_cache:
            return (out_hidden_states, presents)
        return out_hidden_states


class DeepseekMoeModel(Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config

        if config.mapping.is_first_pp_rank():
            self.vocab_embedding = Embedding(config.vocab_size,
                                             config.hidden_size,
                                             dtype=config.dtype)

        self.layers = DecoderLayerList(DeepseekMoeDecoderLayer, config)
        if config.mapping.is_last_pp_rank():
            self.ln_f = LayerNorm(normalized_shape=config.hidden_size,
                                  eps=config.norm_epsilon,
                                  dtype=config.dtype)

    def forward(self,
                input_ids,
                position_ids,
                use_cache=False,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                hidden_states=None):

        if self.config.mapping.is_first_pp_rank():
            hidden_states = self.vocab_embedding(input_ids)
        else:
            hidden_states = recv(hidden_states, self.config.mapping.prev_pp_rank())


        hidden_states = self.layers(hidden_states,
                                    use_cache=use_cache,
                                    attention_mask=attention_mask,
                                    kv_cache_params=kv_cache_params,
                                    attention_params=attention_params)

        if use_cache:
            hidden_states, presents = hidden_states

        if self.config.mapping.is_last_pp_rank():
            hidden_states = self.ln_f(hidden_states)
        else:
            hidden_states = send(hidden_states, self.config.mapping.next_pp_rank())

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class DeepseekMoeForCausalLM(DecoderModelForCausalLM):

    def __init__(self, config: PretrainedConfig):
        self.check_config(config)
        transformer = DeepseekMoeModel(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size,
                                           config.mapping.tp_size)
        if config.mapping.is_last_pp_rank():
            lm_head = ColumnLinear(config.hidden_size,
                                    vocab_size_padded,
                                    bias=config.bias,
                                    dtype=config.dtype,
                                    tp_group=config.mapping.tp_group,
                                    tp_size=config.mapping.tp_size,
                                    gather_output=True)
        else:
            lm_head = None
        self.quant_mode = config.quant_mode
        self.mapping = config.mapping
        super().__init__(config, transformer, lm_head)

    def check_config(self, config):
        config.set_if_not_exist('bias', False)
        # config.set_if_not_exist('clip_qkv', None)
        config.set_if_not_exist('rotary_base', 10000.0)
        config.set_if_not_exist('rotary_scaling', None)
        config.set_if_not_exist('moe_num_experts', 0)
        config.set_if_not_exist('moe_top_k', 0)
        config.set_if_not_exist('moe_tp_mode',
                                MoeConfig.ParallelismMode.TENSOR_PARALLEL)
        config.set_if_not_exist(
            'moe_normalization_mode',
            None)


