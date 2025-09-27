import os
# os.environ["SGLANG_DISABLE_TRITON"] = "1"
# os.environ["SGLANG_DISABLE_FUSED_MOE"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # optional: easier debugging

import math
import unittest
from types import SimpleNamespace

import numpy as np


class TestGrokEquivalence(unittest.TestCase):
    def setUp(self):
        # No monkeypatch; we'll select attention backend per device (CPU vs TPU)
        pass

    def _load_config(self):
        import json
        from transformers import PretrainedConfig
        from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel

        init_distributed_environment(rank=0)
        initialize_model_parallel()

        cfg_path = "/data/hongzheng/sglang-jax/test/grok/config.json"
        with open(cfg_path, "r") as f:
            cfg_dict = json.load(f)
        return PretrainedConfig.from_dict(cfg_dict)

    def _build_jax_forward_batch(self, cfg, input_ids_np, seq_lens, positions_np, mesh=None):
        import jax
        import jax.numpy as jnp
        from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

        # Detect platform
        devices = jax.devices()
        has_tpu = any(d.platform == "tpu" for d in devices)

        total_tokens = int(seq_lens.sum())
        batch_size = int(len(seq_lens))

        # CPU simple attention backend (causal, per-sequence)
        class SimpleAttentionBackend:
            def get_forward_metadata(self, batch):
                return None

            def __call__(self, q, k, v, layer, forward_batch, **kwargs):
                # q,k,v: [T, H*D] folded, reshape into heads
                Hq = layer.q_head_num
                Hkv = layer.kv_head_num
                D = layer.head_dim
                qh = q.reshape(q.shape[0], Hq, D).astype(jnp.float32)
                kh = k.reshape(k.shape[0], Hkv, D).astype(jnp.float32)
                vh = v.reshape(v.shape[0], Hkv, D).astype(jnp.float32)
                if Hkv != Hq:
                    repeat = Hq // max(Hkv, 1)
                    kh = jnp.repeat(kh, repeats=repeat, axis=1)
                    vh = jnp.repeat(vh, repeats=repeat, axis=1)
                pos = forward_batch.positions.astype(jnp.int32)
                seq_lens_local = forward_batch.seq_lens
                seq_ids = []
                for i, L in enumerate(list(seq_lens_local)):
                    seq_ids.append(jnp.full((int(L),), i, dtype=jnp.int32))
                seq_ids = jnp.concatenate(seq_ids, axis=0)
                same_seq = (seq_ids[:, None] == seq_ids[None, :])
                causal = (pos[:, None] >= pos[None, :])
                mask = jnp.where(same_seq & causal, 0.0, -1e9)
                scale = layer.scaling if layer.scaling is not None else 1.0 / jnp.sqrt(D)
                q_t = jnp.transpose(qh, (1, 0, 2))
                k_t = jnp.transpose(kh, (1, 2, 0))
                logits = jnp.matmul(q_t, k_t) * scale + mask[None, :, :]
                probs = jax.nn.softmax(logits, axis=-1)
                v_t = jnp.transpose(vh, (1, 0, 2))
                out = jnp.matmul(probs, v_t)
                out = jnp.transpose(out, (1, 0, 2)).reshape(q.shape[0], Hq * D)
                return out.astype(q.dtype), None

        if has_tpu and mesh is not None:
            # TPU path: use FlashAttention + fused KV cache
            from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention
            from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool

            kv_heads = int(cfg.num_key_value_heads)
            head_dim = int(cfg.head_dim)
            num_layers = int(cfg.num_hidden_layers)
            page_size = 1

            token_to_kv_pool = MHATokenToKVPool(
                size=total_tokens,
                page_size=page_size,
                dtype=jnp.bfloat16,
                head_num=kv_heads,
                head_dim=head_dim,
                layer_num=num_layers,
                mesh=mesh,
            )

            attn_backend = FlashAttention(
                num_attn_heads=int(cfg.num_attention_heads),
                num_kv_heads=kv_heads,
                head_dim=head_dim,
                page_size=page_size,
                mesh=mesh,
            )

            cache_loc = np.arange(total_tokens, dtype=np.int32)
            batch_like = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                seq_lens=seq_lens,
                extend_seq_lens=seq_lens,
                cache_loc=cache_loc,
            )
            attn_backend.forward_metadata = attn_backend.get_forward_metadata(batch_like)

            fb = ForwardBatch(
                bid=0,
                forward_mode=ForwardMode.EXTEND,
                batch_size=batch_size,
                input_ids=jnp.asarray(input_ids_np),
                req_pool_indices=jnp.arange(batch_size, dtype=jnp.int32),
                seq_lens=jnp.asarray(seq_lens),
                out_cache_loc=jnp.zeros(total_tokens, dtype=jnp.int32),
                positions=jnp.asarray(positions_np),
                extend_start_loc=jnp.zeros(batch_size, dtype=jnp.int32),
                token_to_kv_pool=token_to_kv_pool,
                attn_backend=attn_backend,
                cache_loc=jnp.asarray(cache_loc),
                extend_prefix_lens=jnp.zeros(batch_size, dtype=jnp.int32),
                extend_seq_lens=jnp.asarray(seq_lens),
            )
            return fb
        else:
            # CPU path: use simple backend; no KV cache or mesh
            attn_backend = SimpleAttentionBackend()
            fb = ForwardBatch(
                bid=0,
                forward_mode=ForwardMode.EXTEND,
                batch_size=batch_size,
                input_ids=jnp.asarray(input_ids_np),
                req_pool_indices=jnp.arange(batch_size, dtype=jnp.int32),
                seq_lens=jnp.asarray(seq_lens),
                out_cache_loc=jnp.zeros(total_tokens, dtype=jnp.int32),
                positions=jnp.asarray(positions_np),
                extend_start_loc=jnp.zeros(batch_size, dtype=jnp.int32),
                token_to_kv_pool=None,
                attn_backend=attn_backend,
                cache_loc=jnp.arange(total_tokens, dtype=jnp.int32),
                extend_prefix_lens=jnp.zeros(batch_size, dtype=jnp.int32),
                extend_seq_lens=jnp.asarray(seq_lens),
            )
            return fb

    def test_jax_vs_torch_hidden_states_match(self):
        import importlib

        # Build shared config from JSON
        cfg = self._load_config()

        # Instantiate Torch model (upstream sglang PyTorch)
        try:
            torch_grok_mod = importlib.import_module("sglang.srt.models.grok")
            TorchGrok = getattr(torch_grok_mod, "Grok1ForCausalLM")
        except Exception as e:
            raise unittest.SkipTest(f"Cannot import Torch Grok1ForCausalLM: {e}")
        torch_model = TorchGrok(cfg)

        # Instantiate JAX model (this repo)
        jax_grok_mod = importlib.import_module("sgl_jax.srt.models.grok")
        JaxGrok = getattr(jax_grok_mod, "Grok1ForCausalLM")
        jax_model = JaxGrok(cfg)

        # Move PyTorch model to GPU:1
        import torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available for PyTorch")
        torch_device = torch.device("cuda:1")
        torch_model = torch_model.to(torch_device)

        # Copy weights from Torch to JAX using JAX loader mapping
        import jax
        import jax.numpy as jnp

        def torch_named_weights(m):
            for name, param in m.named_parameters():
                yield name, jnp.asarray(param.detach().cpu().numpy())

        def maybe_transpose_for_linear(name: str, arr):
            # Transpose typical Linear weights that come as (out, in) from PyTorch
            linear_keywords = [
                "qkv_proj.weight",
                "o_proj.weight",
                "gate_up_proj.weight",
                "down_proj.weight",
                "lm_head.weight",
                "gate.weight",
            ]
            if name.endswith(".weight") and any(k in name for k in linear_keywords):
                if arr.ndim == 2:
                    return arr.T
            return arr

        def torch_named_weights_transposed(m):
            for name, param in m.named_parameters():
                w = jnp.asarray(param.detach().cpu().numpy())
                yield name, maybe_transpose_for_linear(name, w)

        _ = jax_model.load_weights(
            torch_named_weights_transposed(torch_model),
            ignore_parent_name=False,
            check_hit_names=False,
            model_config=cfg,
        )

        # Ensure rotary caches in PyTorch model are float32 for custom op
        import torch
        def _ensure_fp32_rope_cache(module):
            for child in module.modules():
                for attr in ["cos_sin_cache", "cos_cached", "sin_cached"]:
                    if hasattr(child, attr):
                        t = getattr(child, attr)
                        if isinstance(t, torch.Tensor) and t.dtype != torch.float32:
                            setattr(child, attr, t.to(dtype=torch.float32))

        # Create synthetic input batch
        rng = np.random.default_rng(0)
        # Keep small lengths to limit KV cache size while using real FA
        seq_lens = np.array([3, 5], dtype=np.int32)
        total_tokens = int(seq_lens.sum())
        input_ids_np = rng.integers(low=0, high=cfg.vocab_size, size=(total_tokens,), dtype=np.int32)
        positions_np = np.concatenate([np.arange(L, dtype=np.int32) for L in seq_lens], axis=0)

        # Optionally create a single-device mesh for TPU only
        import jax as _jax
        from sgl_jax.test.test_utils import create_device_mesh
        tpu_devices = [d for d in _jax.devices() if d.platform == "tpu"]
        mesh = None
        if tpu_devices:
            mesh = create_device_mesh([1, 1, 1], [1, 1, 1], devices=[tpu_devices[0]])

        # Build JAX ForwardBatch (CPU simple backend by default; TPU uses FA)
        fb_jax = self._build_jax_forward_batch(cfg, input_ids_np, seq_lens, positions_np, mesh)

        # Torch forward batch (upstream likely manages its own attention backend)
        fb_torch = SimpleNamespace()
        fb_torch.seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=torch_device)
        fb_torch.positions = torch.tensor(positions_np, dtype=torch.int32, device=torch_device)

        class SimpleAttentionBackendTorch:
            def __init__(self, cfg):
                self.Hq = int(cfg.num_attention_heads)
                self.Hkv = int(cfg.num_key_value_heads)
                self.D = int(cfg.head_dim)
            def get_forward_metadata(self, batch):
                return None
            def forward(self, q, k, v, layer, forward_batch, *args, **kwargs):
                import torch
                T = q.shape[0]
                Hq = self.Hq
                Hkv = self.Hkv
                D = self.D

                # Reshape
                qh = q.view(T, Hq, D).float()
                kh = k.view(T, Hkv, D).float()
                vh = v.view(T, Hkv, D).float()

                # Replicate KV heads if needed
                if Hkv != Hq and Hkv > 0 and (Hq % Hkv == 0):
                    repeat = Hq // Hkv
                    kh = kh.repeat(1, repeat, 1)
                    vh = vh.repeat(1, repeat, 1)

                pos = forward_batch.positions.to(torch.int32)
                seq_lens = forward_batch.seq_lens.tolist()
                seq_ids = []
                for i, L in enumerate(seq_lens):
                    seq_ids.append(torch.full((int(L),), i, dtype=torch.int32, device=pos.device))
                seq_ids = torch.cat(seq_ids, dim=0)
                same_seq = (seq_ids[:, None] == seq_ids[None, :])
                causal = (pos[:, None] >= pos[None, :])
                mask = torch.where(same_seq & causal, torch.tensor(0.0, device=pos.device), torch.tensor(-1e9, device=pos.device))

                scale = getattr(layer, 'scaling', 1.0 / (D ** 0.5))

                q_t = qh.permute(1, 0, 2)
                k_t = kh.permute(1, 2, 0)
                logits = torch.matmul(q_t, k_t) * scale + mask.unsqueeze(0)
                probs = torch.softmax(logits, dim=-1)
                v_t = vh.permute(1, 0, 2)
                out = torch.matmul(probs, v_t).permute(1, 0, 2).contiguous().view(T, Hq * D)
                return out.to(dtype=q.dtype)

        fb_torch.attn_backend = SimpleAttentionBackendTorch(cfg)

        # Run base models to get hidden states
        input_ids_torch = torch.tensor(input_ids_np, dtype=torch.long, device=torch_device)

        if mesh is not None:
            with mesh:
                hs_jax = jax_model.model(
                    input_ids=fb_jax.input_ids,
                    positions=fb_jax.positions,
                    forward_batch=fb_jax,
                )
        else:
            hs_jax = jax_model.model(
                input_ids=fb_jax.input_ids,
                positions=fb_jax.positions,
                forward_batch=fb_jax,
            )
        torch_model = torch_model.to(dtype=torch.bfloat16)
        _ensure_fp32_rope_cache(torch_model)
        hs_torch = torch_model.model(
            input_ids=input_ids_torch,
            positions=fb_torch.positions,
            forward_batch=fb_torch,
        )

        hs_jax_np = np.asarray(hs_jax, dtype=np.float32)
        hs_torch_np = hs_torch.detach().cpu().numpy().astype(np.float32)

        self.assertEqual(hs_jax_np.shape, hs_torch_np.shape)
        max_abs_diff = float(np.max(np.abs(hs_jax_np - hs_torch_np)))
        self.assertLessEqual(max_abs_diff, 1e-3, msg=f"Max abs diff too large: {max_abs_diff}")


if __name__ == "__main__":
    unittest.main() 