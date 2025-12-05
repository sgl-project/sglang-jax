import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.kernels.fused_moe.v1.kernel import fused_ep_moe
from sgl_jax.srt.layers.fused_moe import FusedEPMoE


# Mock Config class
class MockConfig:
    hidden_size = 128


def test_fused_ep_moe_crash():
    print("--- Starting FusedEPMoE Crash Test ---")

    # Setup minimal mesh
    devices = jax.devices()
    if len(devices) < 2:
        print("Need at least 2 devices. Skipping.")
        return

        # Use 2 devices: data=1, tensor=2
        mesh_devices = np.array(devices[:2]).reshape(1, 2)
        # FORCE EXPLICIT AXIS TYPES to bypass the initialization error
        mesh = Mesh(
            mesh_devices,
            axis_names=("data", "tensor"),
            axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
        )

        print(f"Base Mesh: {mesh}")
    # Instantiate FusedEPMoE
    # ep_size=1 (so num_experts=2 means 2 experts per device? No, ep_size is partitions.)
    # Let's try ep_size=2. If world_size=2, then tp_size=1.
    # mesh is (1, 2).
    # If ep_size=2.
    # FusedEPMoE checks: world_size // ep_size -> 2 // 2 = 1 (tp_size).
    # moe_mesh will be (2, 1) -> ('expert', 'tensor').

    try:
        layer = FusedEPMoE(
            config=MockConfig(),
            num_experts=2,
            num_experts_per_tok=1,
            ep_size=2,
            mesh=mesh,
            intermediate_dim=128,
            bt=16,
            bf=16,
            bd1=32,
            bd2=32,
            btc=16,
            bfc=16,
            bd1c=16,
            bd2c=16,  # Set explicit tiles to avoid defaults causing issues
        )
        print("FusedEPMoE initialized successfully.")
        print(f"Layer MoE Mesh: {layer.moe_mesh}")
        print(f"Layer MoE Mesh Axis Names: {layer.moe_mesh.axis_names}")

        # Now try to call it to trigger the kernel assertion
        bs = 1
        seq = 16
        hidden = 128

        # Create dummy inputs
        # Need to shard inputs correctly for the call?
        # FusedEPMoE.__call__ takes hidden_states and router_logits.

        # For simplicity, just calling the kernel wrapper directly with the bad mesh to see assert fail
        # But let's try calling the layer.__call__

        hidden_states = jnp.zeros((seq, hidden), dtype=jnp.bfloat16)
        router_logits = jnp.zeros((seq, 2), dtype=jnp.float32)

        print("Calling layer(hidden_states, router_logits)...")
        output = layer(hidden_states, router_logits)
        print("Layer call finished (Unexpected if assert exists!).")

    except AssertionError as e:
        print(f"\n>>> CAUGHT EXPECTED ASSERTION ERROR: {e}")
    except Exception as e:
        print(f"\n>>> CAUGHT UNEXPECTED EXCEPTION: {type(e).__name__}: {e}")


if __name__ == "__main__":
    # Force CPU if needed, though we want to test mesh behavior which is backend agnostic regarding shape
    test_fused_ep_moe_crash()
