"""Qwen2-Audio audio tower (sgl_jax) vs HF on a tiny random model, CPU fp32 (incl. padded-frame masking)."""
import os, sys
os.environ["JAX_PLATFORMS"] = "cpu"
sys.argv = [sys.argv[0], sys.argv[1]]
import numpy as np, torch, jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location("cpa", os.path.join(os.path.dirname(__file__), "tiny_hf_parity_qwen2_audio.py"))
cpa = importlib.util.module_from_spec(spec); spec.loader.exec_module(cpa)
hf = cpa.build()
from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained(cpa.OUT)
tone, noise = cpa.clips()
enc = proc(text=["<|audio_bos|><|AUDIO|><|audio_eos|>"] * 2, audio=[tone, noise], sampling_rate=cpa.SR, return_tensors="pt", padding=True)
feats, fmask = enc["input_features"], enc["feature_attention_mask"]
with torch.no_grad():
    m = hf.model
    feat_len, out_len = m.audio_tower._get_feat_extract_output_lengths(fmask.sum(-1))
    max_seq = (feats.shape[-1] - 2) // 2 + 1
    valid = torch.arange(max_seq)[None] < feat_len[:, None]
    from transformers.masking_utils import create_bidirectional_mask
    am = create_bidirectional_mask(config=m.audio_tower.config, inputs_embeds=torch.zeros(2, max_seq, 1), attention_mask=valid.long())
    ref = m.multi_modal_projector(m.audio_tower(feats, attention_mask=am).last_hidden_state).numpy()
from sgl_jax.srt.models.qwen2_audio import Qwen2AudioForConditionalGeneration as M, audio_output_length
mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("data", "tensor"), axis_types=(jax.sharding.AxisType.Explicit,) * 2)
sd = {k: v.numpy() for k, v in hf.state_dict().items()}
with jax.set_mesh(mesh):
    jm = M(hf.config, dtype=jnp.float32, mesh=mesh)
    a, at = jm.audio, "model.audio_tower"
    for c in ("conv1", "conv2"):
        getattr(a, c).weight.value = jnp.asarray(sd[f"{at}.{c}.weight"]); getattr(a, c).bias.value = jnp.asarray(sd[f"{at}.{c}.bias"])
    a.embed_positions.value = jnp.asarray(sd[f"{at}.embed_positions.weight"])
    a.layer_norm.weight.value = jnp.asarray(sd[f"{at}.layer_norm.weight"]); a.layer_norm.bias.value = jnp.asarray(sd[f"{at}.layer_norm.bias"])
    a.projector.inner.weight.value = jnp.asarray(sd["model.multi_modal_projector.linear.weight"].T); a.projector.inner.bias.value = jnp.asarray(sd["model.multi_modal_projector.linear.bias"])
    for i, L in enumerate(a.layers):
        p = f"{at}.layers.{i}"
        for n in ("q_proj", "k_proj", "v_proj", "out_proj"):
            getattr(L, n).inner.weight.value = jnp.asarray(sd[f"{p}.self_attn.{n}.weight"].T)
            if n != "k_proj": getattr(L, n).inner.bias.value = jnp.asarray(sd[f"{p}.self_attn.{n}.bias"])
        for n in ("fc1", "fc2"):
            getattr(L, n).inner.weight.value = jnp.asarray(sd[f"{p}.{n}.weight"].T); getattr(L, n).inner.bias.value = jnp.asarray(sd[f"{p}.{n}.bias"])
        for n in ("self_attn_layer_norm", "final_layer_norm"):
            getattr(L, n).weight.value = jnp.asarray(sd[f"{p}.{n}.weight"]); getattr(L, n).bias.value = jnp.asarray(sd[f"{p}.{n}.bias"])
    mel = jnp.asarray(feats.numpy().transpose(0, 2, 1)); frames = jnp.asarray(fmask.sum(-1).numpy().astype(np.int32))
    got = np.asarray(a.encode(mel, frames))
worst = 0.0
for i in range(2):
    n = int(out_len[i]); assert n == audio_output_length(int(frames[i]))
    d = np.abs(got[i, :n] - ref[i, :n]).max() / np.abs(ref[i, :n]).max(); worst = max(worst, d)
    print(f"clip {i}: frames={int(frames[i])} tokens={n} rel max|diff|={d:.3e}")
print("ENCODER_MATCH" if worst < 1e-4 else "ENCODER_MISMATCH")
