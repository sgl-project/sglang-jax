import jax  
import numpy as np  
from flax import nnx  
from sgl_jax.srt.multimodal.models.kimi_k2_5.kimi_vit import KimiK25VisionModel  
from sgl_jax.srt.multimodal.configs.config_registry import get_kimi_vl_config  
  
model_path = "/local/moonshotai/Kimi-K2.5"  
  
# 1. Build mesh — jax.devices() returns TPU cores on a TPU machine  
devices = jax.devices()  
mesh = jax.sharding.Mesh(np.array(devices), axis_names=("tensor",))  
  
# 2. Load config  
config = get_kimi_vl_config(model_path)  
config.model_path = model_path  
config.model_class = KimiK25VisionModel  
  
# 3. Create model structure (no memory allocated yet)  
with jax.set_mesh(mesh):  
    model = nnx.eval_shape(  
        lambda: KimiK25VisionModel(config, dtype=jnp.bfloat16, mesh=mesh)  
    )  
  
# 4. Sample params before loading  
before = model.visual.encoder.blocks[0].attn.qkv_proj.kernel[...].mean().item()  
  
# 5. Load weights — reads on CPU, shards to TPU  
model.load_weights(config)  
  
# 6. Verify values changed  
after = model.visual.encoder.blocks[0].attn.qkv_proj.kernel[...].mean().item()  
assert before != after, "Weights did not change — check your weight mappings"  
print(f"qkv_proj mean: {before:.6f} → {after:.6f}")
