import flax.nnx
import jax
from jax import numpy as jnp
import torch
from sglang.srt.models.grok import Grok1ForCausalLM as STDModel, Grok1DecoderLayer as STDLayer
from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel
from sgl_jax.srt.models.grok import Grok1ForCausalLM as SRCModel, Grok1DecoderLayer as SRCLayer
from sgl_jax.srt.configs.model_config import ModelConfig

torch.manual_seed(0)

init_distributed_environment(rank=0)
initialize_model_parallel()

config = ModelConfig(model_path="./weights")

std_model = STDModel(config=config.hf_config)
src_model = SRCModel(config=config.hf_config, rngs=flax.nnx.Rngs(default=0))

std_decode: STDLayer = std_model.model.layers[0]
src_decode: SRCLayer = src_model.model.layers[0]
std_moe = std_decode.block_sparse_moe.cuda(3)
src_moe = src_decode.block_sparse_moe

std_moe.gate.weight[:] = torch.randn_like(std_moe.gate.weight)
std_moe.experts.w13_weight[:] = torch.randn_like(std_moe.experts.w13_weight)
std_moe.experts.w2_weight[:] = torch.randn_like(std_moe.experts.w2_weight)

print(std_moe.gate.weight.dtype)
print(f"{std_moe.experts.w13_weight.shape=}")
print(f"{std_moe.experts.w2_weight.shape=}")

src_moe.gate.weight.value = std_moe.gate.weight.cpu().numpy()
src_moe.experts.wi_0.value = std_moe.experts.w13_weight[:, :config.hf_config.moe_intermediate_size].transpose(1, 2).cpu().numpy()
src_moe.experts.wi_1.value = std_moe.experts.w13_weight[:, config.hf_config.moe_intermediate_size:].transpose(1, 2).cpu().numpy()
src_moe.experts.wo.value = std_moe.experts.w2_weight.transpose(1, 2).cpu().numpy()

std_input = torch.randn([1, config.hf_config.hidden_size]) / 500
src_input = jnp.asarray(std_input)

print(std_moe(std_input.cuda()))
print("====")
print(src_moe(src_input))