import itertools
import random

import numpy as np
import torch.cuda
from sglang.srt.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

# from sglang.srt.models.grok import Grok1ModelForCausalLM
from transformers import PretrainedConfig

from sgl_jax.sglang_grok.sglang_grok import Grok1ModelForCausalLM
from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool

config = PretrainedConfig(
    **{
        "architectures": ["Grok1ForCausalLM"],
        "embedding_multiplier_scale": 90.50966799187809,
        "output_multiplier_scale": 0.5,
        "vocab_size": 131072,
        "hidden_size": 8192,
        "intermediate_size": 1024,
        "moe_intermediate_size": 2048,
        "max_position_embeddings": 131072,
        "num_experts_per_tok": 2,
        "num_local_experts": 8,
        "residual_moe": True,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "num_hidden_layers": 2,
        "head_dim": 128,
        "rms_norm_eps": 1e-05,
        "final_logit_softcapping": 50,
        "attn_logit_softcapping": 30.0,
        "router_logit_softcapping": 30.0,
        "rope_theta": 208533496,
        "attn_temperature_len": 2,
        "sliding_window_size": -1,
        "global_attn_every_n": 1,
        "model_type": "git",
        "torch_dtype": "bfloat16",
        "rope_type": "original",
        "original_max_position_embeddings": 8192,
        "scaling_factor": 16.0,
        "extrapolation_factor": 1.0,
        "attn_factor": 1.0,
        "beta_fast": 8,
        "beta_slow": 1,
    }
)


class AttrDict(dict):
    def __getattr__(self, item):
        return self[item]


def main():
    np.random.seed(0)
    random.seed(0)

    init_distributed_environment(rank=0)
    initialize_model_parallel()

    text_ids = [[random.randint(0, 131071) for _ in range(20)]]

    input_ids = torch.tensor(
        list(itertools.chain(*text_ids)), dtype=torch.int32, device="cuda"
    )
    seq_lens = torch.tensor(
        [len(e) for e in text_ids], dtype=torch.int32, device="cuda"
    )
    positions = torch.tensor(
        list(itertools.chain(*[range(len(e)) for e in text_ids])),
        dtype=torch.int32,
        device="cuda",
    )

    # exclusive cumsum
    extend_start_loc = torch.cumsum(seq_lens, dim=0) - seq_lens

    quant_config = None
    model = Grok1ModelForCausalLM(config=config, quant_config=quant_config)
    model = model.model.cuda()
    for name, param in sorted(model.named_parameters()):
        if torch.is_floating_point(param):
            if "block_sparse_moe.experts.w13_weight" in name:
                data1 = np.random.randn(
                    param.shape[0], param.shape[2], param.shape[1] // 2
                )
                data2 = np.random.randn(
                    param.shape[0], param.shape[2], param.shape[1] // 2
                )
                data1 = data1.transpose((0, 2, 1))
                data2 = data2.transpose((0, 2, 1))
                data = np.concatenate((data1, data2), axis=1)
            else:
                data = np.random.randn(*param.shape)
            param.requires_grad_(False)
            param.copy_(torch.tensor(data, device=param.device))
        else:
            raise ValueError("Cannot random initialize non-floating parameters")
    cache_pool = MHATokenToKVPool(
        size=128,
        page_size=1,
        head_num=config.num_key_value_heads,
        head_dim=config.head_dim,
        layer_num=config.num_hidden_layers,
        dtype=torch.bfloat16,
        device="cuda",
        enable_memory_saver=False,
    )
    out_cache_loc = torch.zeros([sum(seq_lens)], dtype=torch.int32, device="cuda")
    fake_model_runner = AttrDict(device=torch.device("cuda"))
    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(seq_lens),
        input_ids=input_ids,
        seq_lens=seq_lens,
        positions=positions,
        out_cache_loc=out_cache_loc,
        extend_start_loc=extend_start_loc,
        token_to_kv_pool=cache_pool,
        attn_backend=TorchNativeAttnBackend(model_runner=fake_model_runner),
        req_pool_indices=torch.zeros([sum(seq_lens)], dtype=torch.int32, device="cuda"),
        req_to_token_pool=ReqToTokenPool(size=100, max_context_len=512),
        seq_lens_sum=None,
        extend_prefix_lens=torch.tensor([0], dtype=torch.int32, device="cuda"),
        extend_seq_lens=torch.tensor([sum(seq_lens)], dtype=torch.int32, device="cuda"),
    )
    outputs = model(batch.input_ids, batch.positions, batch)
    print(outputs)


if __name__ == "__main__":
    main()
