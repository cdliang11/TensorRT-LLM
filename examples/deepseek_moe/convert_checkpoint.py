# Author: Chengdong Liang
# Date: 2024-04-23

import argparse
import copy
import functools
import json
import os
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple

import numpy as np
import safetensors
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer)
from transformers.pytorch_utils import Conv1D

import tensorrt_llm
# from tensorrt_llm._utils import torch_to_numpy, release_gc, numpy_to_torch
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.layers import MoeConfig


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--tp_size',
                        type=int,
                        default=1,
                        help='N-way tensor parallelism size')
    parser.add_argument('--pp_size',
                        type=int,
                        default=1,
                        help='N-way pipeline parallelism size')
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--logits_dtype',
                        type=str,
                        default='float32',
                        choices=['float16', 'float32'])
    parser.add_argument(
        '--per_channel',
        default=False,
        action="store_true",
        help=
        'By default, we use a single static scaling factor for the GEMM\'s result. '
        'per_channel instead uses a different static scaling factor for each channel. '
        'The latter is usually more accurate, but a little slower.')
    parser.add_argument("--dataset_cache_dir",
                        type=str,
                        default=None,
                        help="cache dir to load the hugging face dataset")
    parser.add_argument(
        '--int8_kv_cache',
        default=False,
        action="store_true",
        help=
        'By default, we use dtype for KV cache. int8_kv_cache chooses int8 quantization for KV'
    )
    parser.add_argument(
        '--use_weight_only',
        default=False,
        action="store_true",
        help='Quantize weights for the various GEMMs to INT4/INT8.'
        'See --weight_only_precision to set the precision')
    parser.add_argument(
        '--weight_only_precision',
        const='int8',
        type=str,
        nargs='?',
        default='int8',
        choices=['int8'],
        help=
        'Define the precision for the weights when using weight-only quantization.'
        'You must also use --use_weight_only for that argument to have an impact.'
        'Only int8 is supported for now'
    )
    parser.add_argument('--output_dir',
                        type=str,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT-LLM checkpoint')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')
    parser.add_argument('--rotary_base', type=float, default=10000.0)
    parser.add_argument('--rotary_scaling', nargs=2, type=str, default=None)
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--n_positions', type=int, default=2048)
    parser.add_argument('--n_layer', type=int, default=32)
    parser.add_argument('--n_head', type=int, default=32)
    parser.add_argument('--n_kv_head', type=int, default=None)
    parser.add_argument('--n_embd', type=int, default=4096)
    parser.add_argument('--inter_size', type=int, default=11008)
    parser.add_argument('--max_seq_len', type=int, default=4096)
    parser.add_argument('--clip_qkv', type=int, default=None)
    parser.add_argument('--hidden_act', type=str, default='silu', help='Set to swiglu to use GLU in MoEs')
    parser.add_argument(
        '--moe_num_experts',
        default=0,
        type=int,
        help='Specify the number of experts to use for MOE layers')
    parser.add_argument(
        '--moe_top_k',
        default=0,
        type=int,
        help=
        'Specify the top_k value to use for MOE layers. Default to 1 if --moe_num_experts is set'
    )
    parser.add_argument(
        '--moe_tp_mode',
        default=MoeConfig.ParallelismMode.TENSOR_PARALLEL,
        type=int,
        help=
        'Controls how to distribute experts in TP. Check layers/moe.py for accepted values',
    )
    parser.add_argument(
        '--moe_renorm_mode',
        default=MoeConfig.ExpertScaleNormalizationMode.NONE,
        type=int,
        help=
        'Controls renormalization after gate logits. Check layers/moe.py for accepted values',
    )
    parser.add_argument(
        '--use_fused_mlp',
        default=False,
        action='store_true',
        help=
        'Enable horizontal fusion in GatedMLP, reduces layer input traffic and potentially improves performance. '
        'For FP8 PTQ, the downside is slight reduction of accuracy because one of the quantization scaling factors are discarded '
        '(0.45734 vs 0.45755 for LLaMA-v2 7B using ammo/examples/hf/instruct_eval/mmlu.py).'
    )
    parser.add_argument(
        '--disable_weight_only_quant_plugin',
        default=False,
        action="store_true",
        help=
        'By default, using plugin implementation for weight quantization. Enabling disable_weight_only_quant_plugin flag will use ootb implementation instead of plugin.'
        'You must also use --use_weight_only for that argument to have an impact.'
    )
    parser.add_argument(
        '--dense_context_fmha',
        default=False,
        action='store_true',
        help=
        'Enable dense fmha in context phase, otherwise sliding window attention.'
        'If dense_context_fmha=False, the sliding window size is the max attention window size.'
    )
    parser.add_argument(
        '--use_parallel_embedding',
        action="store_true",
        default=False,
        help=
        'By default embedding parallelism is disabled. By setting this flag, embedding parallelism is enabled'
    )
    parser.add_argument(
        '--embedding_sharding_dim',
        type=int,
        default=0,
        choices=[0, 1],
        help=
        'By default the embedding lookup table is sharded along vocab dimension (embedding_sharding_dim=0). '
        'To shard it along hidden dimension, set embedding_sharding_dim=1'
        'Note: embedding sharing is only enabled when embedding_sharding_dim = 0'
    )
    parser.add_argument(
        '--use_embedding_sharing',
        action="store_true",
        default=False,
        help=
        'Try to reduce the engine size by sharing the embedding lookup table between two layers.'
        'Note: the flag might not take effect when the criteria are not met.')
    parser.add_argument('--use_prompt_tuning',
                        action="store_true",
                        default=False)
    args = parser.parse_args()

    return args


def args_to_build_options(args):
    return {
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
        'share_embedding_table': args.use_embedding_sharing,
        'disable_weight_only_quant_plugin':
        args.disable_weight_only_quant_plugin
    }



def generate_int8(weights, act_range, is_qkv=False, multi_query_mode=False):
    """
     This function has two purposes:
      - compute quantized weights, scaled either per-tensor or per-column
      - compute scaling factors
      Depending on the GEMM API (CUTLASS/CUBLAS) the required scaling factors differ.
      CUTLASS uses two sets of scaling factors. One for the activation X, one for the weight W.
      CUBLAS only has one (we can't do per-row scaling). So we must provide pre-multiplied scaling factor.
      Here is the list of what we need (T means per-tensor, C per-column):
        - scale_x_orig_quant puts fp activation into the quantized range (i.e. [-128, 127], for int8). Used before the GEMM. (T)
        - scale_y_quant_orig puts quantized activation into the fp range. Used if the GEMM outputs int8. (T)
        - scale_w_quant_orig puts weights from quant range to fp range (used with CUTLASS) (T, C)
        - scale_y_accum_quant puts the GEMM result (XW) from accumulation range (int32)
          to quant range (int8) (used for CUBLAS) (T, C)
      Note that we don't do anything special about row-parallel GEMM. Theoretically, we could have per-GPU scaling factors too,
      but then the model would change depending on the number of GPUs used.
      For QKV projection, the behavior is special. Even if we have a single matrix to perform QKV projection, we consider it
      as three different matrices: Q, K, and V. So per-tensor actually means one scaling factor for each Q, K and V.
      For our GEMM implementation to respect this behavior, we use per-column mode and replicate values along columns.
    """

    # compute weight scaling factors for fp->int8 and int8->fp
    if is_qkv and not multi_query_mode:
        scale_w_orig_quant_t = 127. / act_range["w"].reshape(3, -1).max(
            dim=-1, keepdims=True)[0].cpu().numpy()
        scale_w_orig_quant_c = 127. / act_range["w"].reshape(3,
                                                             -1).cpu().numpy()
    elif is_qkv and multi_query_mode:
        hidden_dim = weights.shape[0]
        local_dim = act_range["w"].shape[0]
        kv_dim = (local_dim - hidden_dim) // 2
        scale_w_q = act_range["w"][0:hidden_dim]
        scale_w_k = act_range["w"][hidden_dim:hidden_dim + kv_dim]
        scale_w_v = act_range["w"][-kv_dim:]

        scale_w_qkv_t = torch.concat([
            scale_w_q.max(dim=0, keepdim=True)[0],
            scale_w_k.max(dim=0, keepdim=True)[0],
            scale_w_v.max(dim=0, keepdim=True)[0]
        ])

        scale_w_orig_quant_t = 127. / scale_w_qkv_t.cpu().numpy()
        scale_w_orig_quant_c = 127. / act_range["w"].cpu().numpy()
    else:
        scale_w_orig_quant_t = 127. / act_range["w"].max().cpu().numpy()
        scale_w_orig_quant_c = 127. / act_range["w"].cpu().numpy()
    scale_w_quant_orig_t = 1.0 / scale_w_orig_quant_t
    scale_w_quant_orig_c = 1.0 / scale_w_orig_quant_c

    scale_w_orig_quant_c = scale_w_orig_quant_c.astype(np.float32)
    scale_w_orig_quant_t = scale_w_orig_quant_t.astype(np.float32)
    # compute the rest of needed scaling factors
    scale_x_orig_quant_t = np.array(127. / act_range["x"].max().item())
    scale_y_orig_quant_t = np.array(127. / act_range["y"].max().item())
    scale_y_quant_orig_t = np.array(act_range["y"].max().item() / 127.)
    scale_y_accum_quant_t = scale_y_orig_quant_t / (scale_x_orig_quant_t *
                                                    scale_w_orig_quant_t)
    scale_y_accum_quant_c = scale_y_orig_quant_t / (scale_x_orig_quant_t *
                                                    scale_w_orig_quant_c)
    if is_qkv and not multi_query_mode:
        scale_y_accum_quant_t = np.broadcast_to(scale_y_accum_quant_t,
                                                scale_w_orig_quant_c.shape)
        scale_w_quant_orig_t = np.broadcast_to(scale_w_quant_orig_t,
                                               scale_w_orig_quant_c.shape)
    if is_qkv and multi_query_mode:
        scale_q_y_accum_t = np.broadcast_to(scale_y_accum_quant_t[0],
                                            scale_w_q.shape)
        scale_k_y_accum_t = np.broadcast_to(scale_y_accum_quant_t[1],
                                            scale_w_k.shape)
        scale_v_y_accum_t = np.broadcast_to(scale_y_accum_quant_t[2],
                                            scale_w_v.shape)
        scale_y_accum_quant_t = np.concatenate(
            [scale_q_y_accum_t, scale_k_y_accum_t, scale_v_y_accum_t])
        scale_w_quant_orig_t = np.concatenate([
            np.broadcast_to(scale_w_quant_orig_t[0], scale_w_q.shape),
            np.broadcast_to(scale_w_quant_orig_t[1], scale_w_k.shape),
            np.broadcast_to(scale_w_quant_orig_t[2], scale_w_v.shape)
        ])

    to_i8 = lambda x: x.round().clip(-127, 127).astype(np.int8)
    if weights.dtype == torch.bfloat16:
        weights = weights.to(torch.float32).numpy()
    else:
        weights = weights.numpy()

    if is_qkv and multi_query_mode:
        weight_int8 = to_i8(weights / scale_w_quant_orig_t)
    else:
        weight_int8 = to_i8(weights * scale_w_orig_quant_t)

    return {
        "weight.int8": weight_int8,
        "weight.int8.col": to_i8(weights * scale_w_orig_quant_c),
        "scale_x_orig_quant": scale_x_orig_quant_t.astype(np.float32),
        "scale_w_quant_orig": scale_w_quant_orig_t.astype(np.float32),
        "scale_w_quant_orig.col": scale_w_quant_orig_c.astype(np.float32),
        "scale_y_accum_quant": scale_y_accum_quant_t.astype(np.float32),
        "scale_y_accum_quant.col": scale_y_accum_quant_c.astype(np.float32),
        "scale_y_quant_orig": scale_y_quant_orig_t.astype(np.float32),
    }


@torch.no_grad()
def capture_activation_range(model,
                             tokenizer,
                             dataset,
                             num_samples=1,
                             seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    act_scales = defaultdict(lambda: {"x": None, "y": None, "w": None})

    tokenizer.pad_token = tokenizer.eos_token

    def stat_tensor(name, tensor, act_scales, key):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float()

        if act_scales[name][key] is None:
            act_scales[name][key] = comming_max
        else:
            act_scales[name][key] = torch.max(act_scales[name][key],
                                              comming_max)

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x, act_scales, "x")
        stat_tensor(name, y, act_scales, "y")

        if act_scales[name]["w"] is None:
            act_scales[name]["w"] = m.weight.abs().clip(
                1e-8, None).max(dim=1)[0].float()

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) or isinstance(m, Conv1D):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples), desc="calibrating model"):
        datapoint = dataset['train'][i:i + 1]
        line = copy.copy(datapoint['article'])
        line[0] = line[0] + ' TL;DR: '
        line[0] = line[0].strip()
        line[0] = line[0].replace(" n't", "n't")
        input_ids = tokenizer(line,
                              return_tensors="pt",
                              max_length=seq_len,
                              padding=True,
                              truncation=True).input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    return act_scales


def split(weight: torch.Tensor,
          tp_size: int,
          rank: int = 0,
          dim: int = 0) -> torch.Tensor:
    if tp_size == 1:
        return weight
    elif weight.ndim == 1:
        return torch.chunk(weight, tp_size)[rank].contiguous()
    else:
        return torch.chunk(weight, tp_size, dim=dim)[rank].contiguous()



def split_qkv_tp(qkv, n_head, n_kv_heads, n_hidden, tensor_parallel, rank):
    """
    Splits the QKV matrix according to tensor parallelism
    """
    kv_head_size = n_kv_heads * (n_hidden // n_head)
    q, k, v = torch.split(qkv, [n_hidden, kv_head_size, kv_head_size], dim=0)
    q = split(q, tensor_parallel, rank, dim=0)
    k = split(k, tensor_parallel, rank, dim=0)
    v = split(v, tensor_parallel, rank, dim=0)
    return torch.concatenate([q, k, v], dim=0).contiguous()



def split_matrix(weight: torch.Tensor, tp_size: int, rank: int,
                 dim: int) -> torch.Tensor:
    return split(weight, tp_size, rank, dim=dim)



def get_weight(params: Dict[str, torch.Tensor], prefix: str,
               dtype: torch.dtype) -> torch.Tensor:
    if f'{prefix}' in params:
        return params[f'{prefix}'].to(dtype).detach().cpu()
    elif f'{prefix}.weight' not in params:
        return None
    return params[f'{prefix}.weight'].to(dtype).detach().cpu()


def get_bias(params: Dict[str, torch.Tensor], prefix: str,
             dtype: torch.dtype) -> torch.Tensor:
    if f'{prefix}.bias' not in params:
        return None
    return params[f'{prefix}.bias'].to(dtype).detach().cpu()



def get_weight_and_bias(params: Dict[str, torch.Tensor], prefix: str,
                        dtype: torch.dtype) -> Tuple[torch.Tensor]:
    return get_weight(params, prefix, dtype), get_bias(params, prefix, dtype)


def get_tllm_linear_weight(
        weight: torch.Tensor,
        prefix: str,
        bias: Optional[torch.Tensor] = None,
        use_weight_only: bool = False,
        plugin_weight_only_quant_type: torch.dtype = torch.int8,
        postfix='weight',
        quant_scale_name=None) -> Dict[str, torch.Tensor]:
    results = {}
    if use_weight_only:
        if weight.dim() > 2:
            v = weight.transpose(1, 2).contiguous().clone()
        else:
            v = weight.t().contiguous().clone()
        processed_torch_weights, torch_weight_scales = \
            torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                v.cpu(), plugin_weight_only_quant_type)
        results[prefix + postfix] = processed_torch_weights
        if quant_scale_name is not None:
            results[quant_scale_name] = torch_weight_scales
        else:
            results[prefix + 'per_channel_scale'] = torch_weight_scales
    else:
        results[prefix + postfix] = weight.contiguous()

    if bias is not None:
        results[f'{prefix}bias'] = bias

    return results



def convert_hf_deepseek_moe(model_params: dict,
                   hf_config: AutoConfig,
                   mapping: Mapping,
                   dtype: str = 'float32',
                   use_weight_only: bool = False,
                   plugin_weight_only_quant_type: torch.dtype = torch.int8,
                   moe_config: MoeConfig = None,
                   int8_kv_cache=False,
                   act_range=[]):

    weights = {}
    tik = time.time()

    dtype = getattr(torch, dtype)
    num_hidden_layers = hf_config.num_hidden_layers
    num_head = hf_config.num_attention_heads
    num_kv_heads = hf_config.num_key_value_heads
    num_hidden = hf_config.hidden_size
    mlp_hidden_size = hf_config.intermediate_size
    moe_hidden_size = hf_config.moe_intermediate_size
    moe_first_k_dense_replace = hf_config.first_k_dense_replace
    layers_range = mapping.pp_layers(num_hidden_layers)
    multi_query_mode = (num_kv_heads != num_head)

    for l in layers_range:
        prefix = f'model.layers.{l}'
        tllm_prex = f'transformer.layers.{l-layers_range[0]}'

        q_weight = get_weight(model_params, prefix + '.self_attn.q_proj', dtype)
        k_weight = get_weight(model_params, prefix + '.self_attn.k_proj', dtype)
        v_weight = get_weight(model_params, prefix + '.self_attn.v_proj', dtype)

        qkv_w = torch.cat([q_weight, k_weight, v_weight], dim=0)
        # split_v = split_qkv_tp(qkv_weight, num_head, num_kv_heads, num_hidden,
        #                        mapping.tp_size, mapping.tp_rank)

        norm_prefix = '.norm_attn_norm'
        # Attention QKV (no bias)
        # qkv_w = get_weight(model_params, f'{prefix}{norm_prefix}.attn.Wqkv', dtype)
        qkv_w = split_qkv_tp(qkv_w, num_head, num_kv_heads, num_hidden,
                             mapping.tp_size, mapping.tp_rank)

        weights.update(
            get_tllm_linear_weight(qkv_w, f'{tllm_prex}.attention.qkv.', None,
                                   use_weight_only,
                                   plugin_weight_only_quant_type))

        del model_params[prefix + '.self_attn.q_proj.weight']
        del model_params[prefix + '.self_attn.k_proj.weight']
        del model_params[prefix + '.self_attn.v_proj.weight']

        # print(model_params.keys())


        # Attention dense (no bias)
        # attn_dense_weight = get_weight(model_params, f'{prefix}{norm_prefix}.attn.out_proj',
        #                                dtype)
        attn_dense_weight = get_weight(model_params, prefix + '.self_attn.o_proj', dtype)
        attn_dense_w = split_matrix(attn_dense_weight,
                                    mapping.tp_size,
                                    mapping.tp_rank,
                                    dim=1)
        weights.update(
            get_tllm_linear_weight(attn_dense_w, f'{tllm_prex}.attention.dense.',
                                   None, use_weight_only,
                                   plugin_weight_only_quant_type))

        del model_params[prefix + '.self_attn.o_proj.weight']

        # input layer_norm
        input_ln_weight = get_weight(model_params, prefix + '.input_layernorm', dtype)
        weights[f'{tllm_prex}.input_layernorm.weight'] = input_ln_weight

        del model_params[prefix + '.input_layernorm.weight']

        # post layer_norm
        post_ln_weight = get_weight(model_params, prefix + '.post_attention_layernorm', dtype)
        weights[f'{tllm_prex}.post_layernorm.weight'] = post_ln_weight

        del model_params[prefix + '.post_attention_layernorm.weight']

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


        if moe_config and moe_config.has_moe() and l >= moe_first_k_dense_replace:
            rank_experts = list(range(moe_config.num_experts))
            print(rank_experts)
            # up_proj->w3 gate_proj->w1 down_proj->w2
            for suffix in ['gate_proj', 'down_proj', 'up_proj']:
                model_params[f'model.layers.{l}.block_sparse_moe.experts.{suffix}.weight'] = \
                    torch.stack([model_params[f'model.layers.{l}.mlp.experts.{expert}.{suffix}.weight'].detach()
                                 for expert in rank_experts])
                for expert in rank_experts:
                    del model_params[f'model.layers.{l}.mlp.experts.{expert}.{suffix}.weight']
            print("MOE branch")
            w3 = model_params[f'model.layers.{l}.block_sparse_moe.experts.up_proj.weight']
            w2 = model_params[f'model.layers.{l}.block_sparse_moe.experts.down_proj.weight']
            w1 = model_params[f'model.layers.{l}.block_sparse_moe.experts.gate_proj.weight']
            print(w1.shape)
            print(w2.shape)
            print(w3.shape)
            if moe_config.tp_mode == moe_config.ParallelismMode.TENSOR_PARALLEL:
                w3 = split(w3, mapping.tp_size, mapping.tp_rank, dim=1)
                w2 = split(w2, mapping.tp_size, mapping.tp_rank, dim=2)
                w1 = split(w1, mapping.tp_size, mapping.tp_rank, dim=1)

            model_params[f'model.layers.{l}.block_sparse_moe.experts.w3w1.weight'] = torch.concat(
                [w3, w1], dim=-2)
            model_params[f'model.layers.{l}.block_sparse_moe.experts.w2.weight'] = w2

            ## w2
            moe_experts_w2_weights = get_weight(
                model_params, prefix + ".block_sparse_moe.experts.w2", dtype)
            weights.update(get_tllm_linear_weight(
                moe_experts_w2_weights,
                tllm_prex + ".mlp.experts_weight_2",
                None,
                use_weight_only,
                plugin_weight_only_quant_type,
                postfix='',
                quant_scale_name=tllm_prex + ".mlp.experts_scale_2"
            ))

            ## w3w1
            moe_experts_w3w1_weights = get_weight(
                model_params, prefix + '.block_sparse_moe.experts.w3w1', dtype
            )
            weights.update(get_tllm_linear_weight(
                moe_experts_w3w1_weights,
                tllm_prex + '.mlp.experts_weight_1',
                None,
                use_weight_only,
                plugin_weight_only_quant_type,
                postfix="",
                quant_scale_name=tllm_prex + ".mlp.experts_scale_1"
            ))
            print(tllm_prex + '.mlp.experts_weight_1')

            moe_experts_gate_weights = get_weight(model_params, f'{prefix}.mlp.gate', torch.float32)
            weights[f'{tllm_prex}.mlp.router.weight'] = moe_experts_gate_weights

            # shared_experts
            shared_experts_gate_weight = get_weight(model_params, prefix + '.mlp.shared_experts.up_proj', dtype)
            split_v = split_matrix(shared_experts_gate_weight,
                                   mapping.tp_size,
                                   mapping.tp_rank,
                                   dim=0)

            weights.update(get_tllm_linear_weight(split_v, tllm_prex + ".shared_experts.gate.",
                                                  None, use_weight_only, plugin_weight_only_quant_type))



            # fc
            shared_experts_fc_weight = get_weight(model_params, prefix + '.mlp.shared_experts.gate_proj', dtype)
            split_v = split_matrix(shared_experts_fc_weight,
                                   mapping.tp_size,
                                   mapping.tp_rank,
                                   dim=0)
            weights.update(get_tllm_linear_weight(split_v, tllm_prex + '.shared_experts.fc.', None,
                                                  use_weight_only, plugin_weight_only_quant_type,
                                                  ))

            shared_experts_proj_weight = get_weight(model_params, prefix + '.mlp.shared_experts.down_proj', dtype)
            split_v = split_matrix(shared_experts_proj_weight,
                                   mapping.tp_size,
                                   mapping.tp_rank,
                                   dim=1)
            weights.update(get_tllm_linear_weight(split_v, tllm_prex + '.shared_experts.proj.',
                                                  None, use_weight_only, plugin_weight_only_quant_type,
                                                  ))

            del model_params[f'model.layers.{l}.block_sparse_moe.experts.gate_proj.weight']
            del model_params[f'model.layers.{l}.block_sparse_moe.experts.up_proj.weight']
            del model_params[f'model.layers.{l}.block_sparse_moe.experts.down_proj.weight']
            del w3, w2, w1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        else:
            # 非 moe分支
            mlp_gate_weight = get_weight(model_params, f'{prefix}.mlp.up_proj', dtype)
            split_v = split_matrix(mlp_gate_weight,
                                   mapping.tp_size,
                                   mapping.tp_rank,
                                   dim=0)
            weights.update(get_tllm_linear_weight(split_v, f'{tllm_prex}.mlp.gate.',
                                                  None, use_weight_only, plugin_weight_only_quant_type,
                                                  ))

            del model_params[f'{prefix}.mlp.up_proj.weight']

            mlp_fc_weight = get_weight(model_params, prefix + '.mlp.gate_proj', dtype)
            split_v = split_matrix(mlp_fc_weight,
                                   mapping.tp_size,
                                   mapping.tp_rank,
                                   dim=0)
            weights.update(get_tllm_linear_weight(split_v, tllm_prex + '.mlp.fc.', None,
                                                  use_weight_only, plugin_weight_only_quant_type,
                                                  ))

            del model_params[f'{prefix}.mlp.gate_proj.weight']

            mlp_proj_weight = get_weight(model_params, prefix + '.mlp.down_proj', dtype)
            split_v = split_matrix(mlp_proj_weight, mapping.tp_size, mapping.tp_rank, dim=1)
            weights.update(get_tllm_linear_weight(split_v, tllm_prex + '.mlp.proj.',
                                                  None, use_weight_only, plugin_weight_only_quant_type,
                                                  ))
            del model_params[f'{prefix}.mlp.down_proj.weight']
            print("MLP branch")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    embed_w = get_weight(model_params, 'model.embed_tokens', dtype)
    # embed_w = get_weight(model_params, 'transformer.wte', dtype)
    lm_head = get_weight(model_params, 'lm_head', dtype)
    if mapping.is_first_pp_rank():
        # Embedding
        weights['transformer.vocab_embedding.weight'] = embed_w
    if mapping.is_last_pp_rank():
        if lm_head is None:
            lm_head = embed_w.clone()
        ln_f_w = get_weight(model_params, 'model.norm', dtype)
        # ln_f weight and bias
        weights['transformer.ln_f.weight'] = ln_f_w
        weights['lm_head.weight'] = split_matrix(lm_head,
                                                    mapping.tp_size,
                                                    mapping.tp_rank,
                                                    dim=0)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights



def execute(workers, func, hf_model):
    if workers == 1:
        for rank, f in enumerate(func):
            f(hf_model, rank)
    else:
        with ThreadPoolExecutor(max_workers=workers) as p:
            futures = [p.submit(f, hf_model, rank) for rank, f in enumerate(func)]
            exceptions = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    exceptions.append(e)
            assert len(
                exceptions
            ) == 0, "Checkpoint conversion failed, please check error log."


if __name__ == '__main__':
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    world_size = args.tp_size * args.pp_size

    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    quant_algo = None
    plugin_weight_only_quant_type = None
    if args.use_weight_only:
        assert args.weight_only_precision == 'int8', "Only int8 is supported"
        plugin_weight_only_quant_type = torch.int8
        quant_algo = "W8A16"

    if args.int8_kv_cache:
        kv_cache_quant_algo = "INT8"
    else:
        kv_cache_quant_algo = None

    hf_config = None
    if args.model_dir is not None:
        hf_config = AutoConfig.from_pretrained(args.model_dir,
                                               trust_remote_code=True)
        args.n_kv_head = hf_config.num_key_value_heads
        args.n_layer = hf_config.num_hidden_layers
        args.first_k_dense_replace = hf_config.first_k_dense_replace
        args.n_head = hf_config.num_attention_heads
        args.vocab_size = hf_config.vocab_size
        args.n_embd = hf_config.hidden_size
        args.inter_size = hf_config.intermediate_size
        args.moe_inter_size = hf_config.moe_intermediate_size
        args.max_seq_len = hf_config.max_position_embeddings
        args.moe_num_experts = getattr(hf_config, "n_routed_experts", 0)
        args.shared_num_experts = getattr(hf_config, "n_shared_experts", 0)
        args.moe_top_k = getattr(hf_config, "num_experts_per_tok", 0)
        if args.moe_num_experts and args.moe_top_k == 0:
            args.moe_top_k = 1
        # args.clip_qkv = hf_config.attn_config.clip_qkv
        args.hidden_act = 'swiglu'
        args.rotary_base = hf_config.rope_theta
        args.rotary_scaling = hf_config.rope_scaling
    args.moe_config = MoeConfig(args.moe_num_experts, args.moe_top_k,
                                    args.moe_tp_mode,
                                    args.moe_renorm_mode).validate()
    config = {
        'architecture': 'DeepseekMoeForCausalLM',
        'dtype': args.dtype,
        'logits_dtype': args.logits_dtype,
        'vocab_size': args.vocab_size,
        'hidden_size': args.n_embd,
        'intermediate_size': args.inter_size,
        'moe_intermediate_size': args.moe_inter_size,
        'num_hidden_layers': args.n_layer,
        'num_attention_heads': args.n_head,
        'num_key_value_heads': args.n_kv_head,
        'max_position_embeddings': args.max_seq_len,
        'position_embedding_type': 'rope_gpt_neox',
        'hidden_act': args.hidden_act,
        'rotary_base': args.rotary_base,
        'rotary_scaling': args.rotary_scaling,
        'quantization': {
            'quant_algo': quant_algo,
            'kv_cache_quant_algo': kv_cache_quant_algo,
            'exclude_modules': ['lm_head'],
        },
        'moe_config': {
            "num_experts": args.moe_num_experts,
            "top_k": args.moe_top_k,
            "tp_mode": args.moe_tp_mode,
            "normalization_mode": args.moe_renorm_mode
        },
        'mapping': {
            'world_size': world_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        },
        'clip_qkv': args.clip_qkv,
        'moe_num_experts': args.moe_num_experts,
        'moe_shared_num_experts': args.shared_num_experts,
        'moe_first_k_dense_replace': args.first_k_dense_replace,
        'moe_top_k': args.moe_top_k,
        'moe_tp_mode': args.moe_tp_mode,
        'moe_normalization_mode': args.moe_renorm_mode,
        'dense_context_fmha': args.dense_context_fmha,
    }

    print(config)

    if args.use_weight_only and (args.weight_only_precision == 'int8'):
            config['quantization']['quant_algo'] = 'W8A16'

    if args.use_weight_only and args.moe_config.has_moe():
        config['quantization']['exclude_modules'].append('router')

    if args.int8_kv_cache:
        config['quantization']['kv_cache_quant_algo'] = 'INT8'

    config.update(args_to_build_options(args))

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    def load_from_hf(model_dir):
        hf_model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                        trust_remote_code=True,
                                                        device_map="cpu",
                                                        torch_dtype=getattr(
                                                            torch, args.dtype),
                                                        config=hf_config)
        return hf_model

    def convert_and_save(hf_model, rank):
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=args.tp_size,
                          pp_size=args.pp_size)
        act_range = {}
        if args.int8_kv_cache:
            dataset = load_dataset("ccdv/cnn_dailymail",
                                   '3.0.0',
                                   cache_dir=args.dataset_cache_dir,
                                   trust_remote_code=True)
            act_range = capture_activation_range(
                hf_model,
                AutoTokenizer.from_pretrained(args.model_dir,
                                              padding_side='left',
                                              trust_remote_code=True), dataset)

        hf_model = dict(hf_model.named_parameters())
        weights = convert_hf_deepseek_moe(
                hf_model,
                hf_config,
                mapping,
                dtype=args.dtype,
                use_weight_only=args.use_weight_only,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type,
                moe_config=args.moe_config,
                int8_kv_cache=args.int8_kv_cache,
                act_range=act_range)

        safetensors.torch.save_file(
            weights, os.path.join(args.output_dir, f'rank{rank}.safetensors'))
        del weights
        # release_gc()

    if args.model_dir:
        hf_model = load_from_hf(args.model_dir)
        execute(args.workers, [convert_and_save] * world_size, hf_model)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')
