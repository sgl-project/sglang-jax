import logging
import jax  
import numpy as np  
import jax.numpy as jnp
from flax import nnx  
from sgl_jax.srt.multimodal.models.kimi_k25.kimi_k25_vit import Kimi_K25_VisionModel  
from sgl_jax.srt.multimodal.configs.kimi.kimi_k25_config import (
    KimiK25ModelVitConfig,
)

logging.basicConfig(level=logging.INFO)
  
model_path = "/local/kimi"  
  
# 1. Build mesh â€” jax.devices() returns TPU cores on a TPU machine  
devices = jax.devices()  
mesh = jax.sharding.Mesh(np.array(devices), axis_names=("tensor",))  
  
# 2. Load config  
config = KimiK25ModelVitConfig
config.model_path = model_path  
config.model_class = Kimi_K25_VisionModel  
  
# 3. Create model structure (no memory allocated yet)  
with jax.set_mesh(mesh):  
    model = nnx.eval_shape(  
        lambda: Kimi_K25_VisionModel(config, dtype=jnp.bfloat16, mesh=mesh)  
    )  
  
# 4. Sample params before loading  
#before = model.vision_tower.encoder.blocks[0].attn.qkv_proj.kernel[...].mean().item()  
  
# 5. Load weights â€” reads on CPU, shards to TPU  
model.load_weights(config)  
  
# 6. Verify values changed  
after = model.vision_tower.encoder.blocks[0].attn.qkv_proj.kernel[...].mean().item()  
