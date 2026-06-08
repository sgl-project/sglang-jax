import logging
import os
import jax

# Set up logging immediately
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_kimi_int4_loading")

# Initialize JAX distributed as early as possible to avoid backend init issues
if "JAX_COORDINATOR_ADDRESS" in os.environ:
    logger.info("Initializing JAX distributed at startup...")
    try:
        jax.distributed.initialize()
        logger.info("JAX Distributed Initialized successfully at startup.")
        
        # Set TPU overrides after JAX init but before backend init
        process_id = int(os.environ.get("JAX_PROCESS_ID", 0))
        host0_name = "gke-tpu-4a99f854-2zmz"
        host1_name = "gke-tpu-4a99f854-ptwl"
        
        if process_id == 0:
            os.environ["TPU_WORKER_HOSTNAMES"] = host0_name
            os.environ["TPU_PROCESS_ADDRESSES"] = f"{host0_name}:8471"
            os.environ["TPU_HOSTNAME_OVERRIDE"] = host0_name
            os.environ["MEGASCALE_SLICE_ID"] = "0"
        elif process_id == 1:
            os.environ["TPU_WORKER_HOSTNAMES"] = host1_name
            os.environ["TPU_PROCESS_ADDRESSES"] = f"{host1_name}:8471"
            os.environ["TPU_HOSTNAME_OVERRIDE"] = host1_name
            os.environ["MEGASCALE_SLICE_ID"] = "1"
            
        os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
        os.environ["MEGASCALE_NUM_SLICES"] = "2"
        os.environ["MEGASCALE_COORDINATOR_ADDRESS"] = f"{host0_name}:9915"
        
        logger.info("TPU overrides set in Python: TPU_WORKER_HOSTNAMES=%s, TPU_HOSTNAME_OVERRIDE=%s, MEGASCALE_SLICE_ID=%s, TPU_HOST_BOUNDS=%s", 
                    os.environ["TPU_WORKER_HOSTNAMES"], os.environ["TPU_HOSTNAME_OVERRIDE"], os.environ["MEGASCALE_SLICE_ID"], os.environ["TPU_HOST_BOUNDS"])
    except Exception as e:
        logger.info(f"JAX distributed init failed or already initialized: {e}")

import numpy as np
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, AxisType

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.multimodal.models.kimi_k25.kimi_k25_vl_generation import KimiK25ForConditionalGeneration
from sgl_jax.srt.utils.quantization.quantization_utils import apply_linear_quantization, apply_moe_quantization

def main():
    model_path = "/dsk/models/kimi_original_new"
    
    # 1. Build TPU Mesh
    devices = jax.devices()
    logger.info("Available JAX devices: %d (%s)", len(devices), [d.platform for d in devices])
    mesh = Mesh(
        np.array(devices).reshape(2, 8),
        axis_names=("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )

    # 2. Create ModelConfig
    logger.info("Creating ModelConfig for: %s", model_path)
    model_config = ModelConfig(
        model_path=model_path,
        trust_remote_code=True,
        dtype="bfloat16"
    )
    
    # Configure ideal sharding for 16 devices (EP=2, TP=8)
    model_config.ep_size = 2
    if hasattr(model_config, "hf_config"):
        model_config.hf_config.ep_size = 2
        if hasattr(model_config.hf_config, "text_config") and model_config.hf_config.text_config is not None:
            model_config.hf_config.text_config.ep_size = 2
            model_config.hf_config.text_config.moe_intermediate_size = 2048
            model_config.hf_config.text_config.n_routed_experts = 16
    if hasattr(model_config, "hf_text_config") and model_config.hf_text_config is not None:
        model_config.hf_text_config.ep_size = 2
        model_config.hf_text_config.moe_intermediate_size = 2048
        model_config.hf_text_config.n_routed_experts = 16
    
    num_layers_env = os.environ.get("NUM_LAYERS")
    if num_layers_env is not None:
        num_layers = int(num_layers_env)
        logger.info("Patching ModelConfig to %d layers for custom verification...", num_layers)
        model_config.hf_text_config.num_hidden_layers = num_layers
        model_config.num_hidden_layers = num_layers
        if hasattr(model_config.hf_config, "text_config") and model_config.hf_config.text_config is not None:
            model_config.hf_config.text_config.num_hidden_layers = num_layers
        
    # 3. Instantiate ABSTRACT model under nnx.eval_shape to avoid HBM out-of-memory
    logger.info("Instantiating abstract KimiK25ForConditionalGeneration under nnx.eval_shape...")
    with jax.set_mesh(mesh):
        model = nnx.eval_shape(
            lambda: KimiK25ForConditionalGeneration(
                model_config.hf_config,
                dtype=model_config.dtype,
                mesh=mesh
            )
        )
    
    # 4. Apply Selective Quantization structure modifications on the abstract model
    logger.info("Applying selective linear quantization (is_static_input=True) on abstract model...")
    model = apply_linear_quantization(model_config, model, is_static_input=True)
    
    logger.info("Applying MoE quantization (is_static_input=True) on abstract model...")
    model = apply_moe_quantization(model_config, model, is_static_input=True)

    layers = model.model.layers
    moe_layer = layers[1]
    
    # 5. Load weights via WeightLoader
    logger.info("Loading weights from safetensors onto TPU...")
    model.load_weights(model_config)
    
    # 6. Verify loaded parameters are materialized and in JAX-native int4
    if hasattr(moe_layer.mlp, "wi_0"):
        param_after = moe_layer.mlp.wi_0.value
        logger.info("Parameter type after loading: %s", type(param_after))
        logger.info("Loaded Parameter shape: %s, dtype: %s", param_after.shape, param_after.dtype)
        
        assert not isinstance(param_after, jax.ShapeDtypeStruct), "Weight is still abstract after loading!"
        assert param_after.dtype == jnp.int4, f"Weight dtype is {param_after.dtype}, expected jnp.int4!"
        
    # 7. Run a dummy forward pass on EPMoE to verify dynamic compilation & device execution dtypes
    logger.info("Constructing dummy inputs to trigger GMM TPU computation kernel...")
    batch_seq = 16
    hidden_size = moe_layer.mlp.hidden_size
    num_experts_per_tok = moe_layer.mlp.num_experts_per_tok

    dummy_hidden_states = jax.random.normal(jax.random.PRNGKey(42), (batch_seq, hidden_size), dtype=jnp.bfloat16)
    dummy_topk_weights = jax.random.uniform(jax.random.key(43), (batch_seq, num_experts_per_tok), dtype=jnp.bfloat16)
    dummy_topk_ids = jax.random.randint(jax.random.key(44), (batch_seq, num_experts_per_tok), 0, moe_layer.mlp.num_experts)

    logger.info("Executing EPMoE forward pass (GMM Kernel on TPU)...")
    with jax.set_mesh(mesh):
        output = moe_layer.mlp(dummy_hidden_states, dummy_topk_weights, dummy_topk_ids)
        output.block_until_ready()
    logger.info("EPMoE Output successfully materialized on TPU - shape: %s, dtype: %s", output.shape, output.dtype)
        
    logger.info("SUCCESS: Kimi K2.5 Raw INT4 Weights Successfully Unpacked, Loaded and Verified!")

if __name__ == "__main__":
    main()
