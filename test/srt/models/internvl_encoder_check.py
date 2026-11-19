"""InternVL vision tower (sgl_jax) vs HF get_image_features on the tiny checkpoint, CPU fp32."""
import os, sys, time
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np, torch, jax, jax.numpy as jnp
from transformers import AutoConfig, AutoProcessor, InternVLForConditionalGeneration
from PIL import Image

D = sys.argv[1]
cfg = AutoConfig.from_pretrained(D)
hf = InternVLForConditionalGeneration.from_pretrained(D, dtype=torch.float32).eval()
proc = AutoProcessor.from_pretrained(D)
rng = np.random.default_rng(0)
img = Image.fromarray((rng.random((600, 900, 3)) * 255).astype("uint8"))
text = proc.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":"x"}]}], add_generation_prompt=True)
enc = proc(text=text, images=[img], return_tensors="pt")
with torch.no_grad():
    ref = hf.model.get_image_features(enc["pixel_values"], vision_feature_layer=-1, vision_feature_select_strategy="default").pooler_output
ref = ref.reshape(-1, ref.shape[-1]).numpy()
print("hf feats", ref.shape)

from sgl_jax.srt.models.internvl import InternVLForConditionalGeneration as M
from sgl_jax.srt.multimodal.processors.internvl import crops_to_patches
mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("data", "tensor"),
                         axis_types=(jax.sharding.AxisType.Explicit,) * 2)
with jax.set_mesh(mesh):
    m = M(cfg, dtype=jnp.float32, mesh=mesh)
    # copy weights from HF into the JAX tower
    sd = {k: v.numpy() for k, v in hf.state_dict().items()}
    v = m.visual
    vt = "model.vision_tower"
    v.patch_embedding.inner.weight.value = jnp.asarray(sd[f"{vt}.embeddings.patch_embeddings.projection.weight"].transpose(1, 2, 3, 0).reshape(v.patch_dim, -1))
    v.patch_embedding.inner.bias.value = jnp.asarray(sd[f"{vt}.embeddings.patch_embeddings.projection.bias"])
    v.cls_token.value = jnp.asarray(sd[f"{vt}.embeddings.cls_token"].reshape(1, -1))
    v.position_embeddings.value = jnp.asarray(sd[f"{vt}.embeddings.position_embeddings"][0])
    for i, L in enumerate(v.layers):
        p = f"{vt}.encoder.layer.{i}"
        for a, b in [("q_proj", "attention.q_proj"), ("k_proj", "attention.k_proj"), ("v_proj", "attention.v_proj"),
                     ("projection_layer", "attention.projection_layer"), ("fc1", "mlp.fc1"), ("fc2", "mlp.fc2")]:
            getattr(L, a).inner.weight.value = jnp.asarray(sd[f"{p}.{b}.weight"].T)
            getattr(L, a).inner.bias.value = jnp.asarray(sd[f"{p}.{b}.bias"])
        for a in ("layernorm_before", "layernorm_after"):
            getattr(L, a).weight.value = jnp.asarray(sd[f"{p}.{a}.weight"]); getattr(L, a).bias.value = jnp.asarray(sd[f"{p}.{a}.bias"])
        L.lambda_1.value = jnp.asarray(sd[f"{p}.lambda_1"]); L.lambda_2.value = jnp.asarray(sd[f"{p}.lambda_2"])
    mp = "model.multi_modal_projector"
    v.mm_norm.weight.value = jnp.asarray(sd[f"{mp}.layer_norm.weight"]); v.mm_norm.bias.value = jnp.asarray(sd[f"{mp}.layer_norm.bias"])
    v.mm_fc1.inner.weight.value = jnp.asarray(sd[f"{mp}.linear_1.weight"].T); v.mm_fc1.inner.bias.value = jnp.asarray(sd[f"{mp}.linear_1.bias"])
    v.mm_fc2.inner.weight.value = jnp.asarray(sd[f"{mp}.linear_2.weight"].T); v.mm_fc2.inner.bias.value = jnp.asarray(sd[f"{mp}.linear_2.bias"])
    patches = crops_to_patches(enc["pixel_values"].numpy().astype(np.float32), 14)
    t0 = time.time()
    out = np.asarray(v.encode(jnp.asarray(patches)[None]))
    print("jax encode s", round(time.time() - t0, 1), out.shape)
out = out.reshape(-1, out.shape[-1])
d = np.abs(out - ref).max(); rel = d / np.abs(ref).max()
print(f"max|diff|={d:.3e} rel={rel:.3e}")
print("ENCODER_MATCH" if rel < 1e-4 else "ENCODER_MISMATCH")
