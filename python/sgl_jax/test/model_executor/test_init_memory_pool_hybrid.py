"""init_memory_pool has_recurrent_state branch.

Detection now lives on ``ModelRunner.linear_recurrent_config`` (a @property
that combines ``isinstance(KimiLinearConfig)`` with the config's own
``is_linear_attn`` flag), so this file constructs real ``KimiLinearConfig``
instances for the detection cases and uses ``SimpleNamespace`` only for the
pool stubs / server args.
"""

import unittest
from types import SimpleNamespace

import jax
import numpy as np
from jax.sharding import Mesh


def _mesh():
    """Single-device mesh with the canonical "tensor" axis name; matches the
    sharding axis RecurrentStatePool partitions H / proj_size on."""
    return Mesh(np.array(jax.devices()), ("tensor",))


def _kimi_cfg(kda_layers=None, num_heads=2, head_dim=4, conv_kernel_size=4):
    """Build a KimiLinearConfig with the given linear_attn_config sub-dict.

    ``kda_layers=None`` produces a degenerate KimiLinearConfig (type matches
    but ``is_linear_attn`` is False), used to exercise the detection
    negative path where the hf_config carries no ``linear_attn_config``.

    NOTE: linear_attn_config is a `dict | None` in real Kimi-Linear HF config
    (see ). KimiLinearConfig's own __init__ asserts both
    ``kda_layers`` and ``full_attn_layers`` keys are present and non-None
    when the dict is given; we satisfy that with an empty ``full_attn_layers``
    list (length is irrelevant to the recurrent pool path).
    """
    from sgl_jax.srt.configs.kimi_linear import KimiLinearConfig

    if kda_layers is None:
        return KimiLinearConfig(num_hidden_layers=2)
    return KimiLinearConfig(
        num_hidden_layers=max(max(kda_layers, default=0) + 1, 2),
        linear_attn_config={
            "kda_layers": list(kda_layers),
            "full_attn_layers": [],
            "num_heads": num_heads,
            "head_dim": head_dim,
            "short_conv_kernel_size": conv_kernel_size,
        },
    )


def _runner_shim(hf_config):
    """SimpleNamespace-style shim that exposes ModelRunner.linear_recurrent_config
    so tests can drive the @property without instantiating a real ModelRunner.

    Mirrors the shim used in test_hybrid_linear_attn_backend.TestHybridConfigProperties
    to keep the two test files aligned.
    """
    from sgl_jax.srt.model_executor.model_runner import ModelRunner

    class _Runner:
        kimi_linear_config = ModelRunner.kimi_linear_config
        linear_recurrent_config = ModelRunner.linear_recurrent_config

        def __init__(self, hf):
            self.model_config = SimpleNamespace(hf_config=hf)

    return _Runner(hf_config)


class TestHasRecurrentStateDetection(unittest.TestCase):
    """Detection logic: ``runner.linear_recurrent_config is not None`` iff the
    hf_config is a ``KimiLinearConfig`` AND its ``is_linear_attn`` flag is True
    (i.e. ``linear_attn_config["kda_layers"]`` is a non-empty list)."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def test_detects_kimi_linear_kda_layers(self):
        runner = _runner_shim(_kimi_cfg(kda_layers=[0, 1, 2]))
        self.assertIsNotNone(runner.linear_recurrent_config)

    def test_non_kimi_linear_hf_config_returns_none(self):
        # Any object that is not a KimiLinearConfig instance must be rejected
        # by isinstance(), even if it duck-types a linear_attn_config attribute.
        hf = SimpleNamespace(linear_attn_config={"kda_layers": [0, 1]})
        runner = _runner_shim(hf)
        self.assertIsNone(runner.linear_recurrent_config)

    def test_degenerate_kimi_linear_returns_none(self):
        # Type matches but linear_attn_config is None → is_linear_attn=False
        # → property must return None (the C1 fix).
        runner = _runner_shim(_kimi_cfg(kda_layers=None))
        self.assertIsNotNone(runner.kimi_linear_config)  # type still matches
        self.assertIsNone(runner.linear_recurrent_config)

    def test_empty_kda_layers_returns_none(self):
        # is_linear_attn explicitly treats an empty kda_layers list as not
        # linear-attn-based (matches the per-config flag's semantics).
        runner = _runner_shim(_kimi_cfg(kda_layers=[]))
        self.assertIsNone(runner.linear_recurrent_config)


class TestInitMemoryPoolHybridBranch(unittest.TestCase):
    """D3 decision: 3 integration tests covering the full has_recurrent_state branch
    × KV pool type matrix (MHA / MLA). No 4-way over-coverage; MHA and MLA share
    the same MemoryPools wrapping path and only differ in the KV pool object.
    """

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def test_standard_model_no_recurrent_state(self):
        """linear_recurrent_config is None -> MemoryPools contains only
        token_to_kv_pool; req_to_token_pool is plain ReqToTokenPool."""
        from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _build_non_hybrid_memory_pools,
        )

        runner = _runner_shim(_kimi_cfg(kda_layers=None))
        self.assertIsNone(runner.linear_recurrent_config)

        kv_stub = SimpleNamespace(replace_buffer=lambda v: None)
        mp = _build_non_hybrid_memory_pools(token_to_kv_pool=kv_stub)
        self.assertIsInstance(mp, MemoryPools)
        self.assertEqual(set(mp._pools.keys()), {"token_to_kv_pool"})
        # Plain ReqToTokenPool path is unchanged (caller handles construction).

    def test_hybrid_with_mha_pool(self):
        """linear_recurrent_config truthy with MHA-like KV pool stub ->
        MemoryPools contains both pools; req_to_token_pool is HybridReqToTokenPool."""
        from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool, MemoryPools
        from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _build_hybrid_pools,
        )

        cfg = _kimi_cfg(kda_layers=[0, 1], num_heads=2, head_dim=4)
        # Stub mimics MHATokenToKVPool externally (any object with replace_buffer).
        mha_like_stub = SimpleNamespace(replace_buffer=lambda v: None)
        rsp, hybrid_pool, mp = _build_hybrid_pools(
            cfg=cfg,
            max_num_reqs=4,
            max_context_len=16,
            tp_size=1,
            token_to_kv_pool=mha_like_stub,
            mesh=_mesh(),
        )
        self.assertIsInstance(rsp, RecurrentStatePool)
        self.assertIsInstance(hybrid_pool, HybridReqToTokenPool)
        self.assertIsInstance(mp, MemoryPools)
        self.assertEqual(set(mp._pools.keys()), {"token_to_kv_pool", "recurrent_state_pool"})
        self.assertIs(mp.token_to_kv_pool, mha_like_stub)
        self.assertIs(mp.recurrent_state_pool, rsp)
        self.assertIs(hybrid_pool.recurrent_state_pool, rsp)
        # linear_attn_config dict subscript path produced expected layer ids.
        self.assertEqual(rsp.linear_recurrent_layer_ids, [0, 1])

    def test_hybrid_with_mla_pool(self):
        """Kimi-Linear real-world scenario: linear_recurrent_config truthy with
        MLA-like KV pool. Verifies MLA pool enters MemoryPools without special
        handling (transparent to wrapping layer). Same shape as MHA case; only
        KV pool type differs - this IS the proof that MHA/MLA pool is
        transparent to the MemoryPools wrapper."""
        from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _build_hybrid_pools,
        )

        cfg = _kimi_cfg(kda_layers=[0, 1], num_heads=2, head_dim=4)
        # Stub mimics MLATokenToKVPool externally (different shape/dim
        # internally, but MemoryPools sees only `replace_buffer`).
        mla_like_stub = SimpleNamespace(
            replace_buffer=lambda v: None,
            kv_lora_rank=512,  # MLA-specific attr; MemoryPools doesn't touch it
        )
        rsp, hybrid_pool, mp = _build_hybrid_pools(
            cfg=cfg,
            max_num_reqs=4,
            max_context_len=16,
            tp_size=1,
            token_to_kv_pool=mla_like_stub,
            mesh=_mesh(),
        )
        self.assertIsInstance(mp, MemoryPools)
        self.assertEqual(set(mp._pools.keys()), {"token_to_kv_pool", "recurrent_state_pool"})
        # MLA-specific attribute survived (MemoryPools doesn't strip it).
        self.assertEqual(mp.token_to_kv_pool.kv_lora_rank, 512)
        # Both pools wired correctly.
        self.assertIs(hybrid_pool.recurrent_state_pool, rsp)


class TestServerArgsForcedConstraints(unittest.TestCase):
    """has_recurrent_state model must assert both disable_radix_cache=True
    and disable_overlap_schedule=True (transparent failure, not silent override)."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def test_assert_radix_cache_disabled_when_user_left_it_enabled(self):
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _enforce_recurrent_state_server_constraints,
        )

        sa = SimpleNamespace(disable_radix_cache=False, disable_overlap_schedule=True)
        with self.assertRaises(AssertionError) as cm:
            _enforce_recurrent_state_server_constraints(sa)
        self.assertIn("disable-radix-cache", str(cm.exception))

    def test_both_disabled_passes(self):
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _enforce_recurrent_state_server_constraints,
        )

        sa = SimpleNamespace(disable_radix_cache=True, disable_overlap_schedule=True)
        # Should not raise.
        _enforce_recurrent_state_server_constraints(sa)

    def test_assert_overlap_schedule_disabled(self):
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _enforce_recurrent_state_server_constraints,
        )

        # Use disable_radix_cache=True so the radix-cache assert passes and
        # we hit the overlap-schedule assert specifically.
        sa = SimpleNamespace(disable_radix_cache=True, disable_overlap_schedule=False)
        with self.assertRaises(AssertionError) as cm:
            _enforce_recurrent_state_server_constraints(sa)
        self.assertIn("disable-overlap-schedule", str(cm.exception))


class TestForwardCallsReplaceAll(unittest.TestCase):
    """_forward must dispatch via self.memory_pools.replace_all(pool_updates)
    instead of the legacy _set_kv_cache_after_forward path.

    We only verify the contract surface (method existence + call shape) without
    invoking a real forward, since real forward needs model loading.
    """

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def test_set_kv_cache_after_forward_method_removed(self):
        """After , _set_kv_cache_after_forward must be deleted."""
        from sgl_jax.srt.model_executor import model_runner

        self.assertFalse(
            hasattr(model_runner.ModelRunner, "_set_kv_cache_after_forward"),
            "_set_kv_cache_after_forward should be removed; "
            "sharding fix now lives in each pool's replace_buffer.",
        )

    def test_jitted_run_model_donates_memory_pools(self):
        """The JIT signature must donate memory_pools, not token_to_kv_pool."""
        import inspect

        from sgl_jax.srt.model_executor import model_runner

        # Inspect the source of _build_jitted_run_model (or jitted_run_model wrapper)
        # for "donate_argnames" containing "memory_pools".
        src = inspect.getsource(model_runner)
        self.assertIn('donate_argnames=["memory_pools"]', src)
        self.assertNotIn('donate_argnames=["token_to_kv_pool"]', src)


class TestStateToKvRatioZeroGuard(unittest.TestCase):
    """D5 decision: ratio<=0 with has_recurrent_state must raise ValueError
    explicitly (not fall through to RecurrentStatePool's max_num_reqs assert,
    whose error message would not point at the ratio config)."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def test_zero_ratio_with_hybrid_raises_value_error(self):
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _check_state_to_kv_ratio_for_hybrid,
        )

        with self.assertRaises(ValueError) as ctx:
            _check_state_to_kv_ratio_for_hybrid(state_to_kv_ratio=0.0)
        msg = str(ctx.exception)
        self.assertIn("state_to_kv_ratio", msg)
        self.assertIn("0", msg)
        self.assertIn("--state-to-kv-ratio", msg)  # actionable hint

    def test_negative_ratio_raises_value_error(self):
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _check_state_to_kv_ratio_for_hybrid,
        )

        with self.assertRaises(ValueError):
            _check_state_to_kv_ratio_for_hybrid(state_to_kv_ratio=-0.5)

    def test_positive_ratio_passes(self):
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _check_state_to_kv_ratio_for_hybrid,
        )

        # No exception
        _check_state_to_kv_ratio_for_hybrid(state_to_kv_ratio=0.9)
        _check_state_to_kv_ratio_for_hybrid(state_to_kv_ratio=0.001)


class TestPhase2EndToEndSanity(unittest.TestCase):
    """surface integration: hybrid pool wiring + non-hybrid wrapping
    + JIT signature both work without raising at construction time."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def test_hybrid_construction_does_not_raise(self):
        """Constructing the full hybrid stack via the helpers does not raise."""
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _build_hybrid_pools,
        )

        cfg = _kimi_cfg(kda_layers=[0, 1, 2], num_heads=4, head_dim=8)
        kv_stub = SimpleNamespace(replace_buffer=lambda v: None)
        rsp, hybrid_pool, mp = _build_hybrid_pools(
            cfg=cfg,
            max_num_reqs=4,
            max_context_len=16,
            tp_size=1,
            token_to_kv_pool=kv_stub,
            mesh=_mesh(),
        )
        # All three pools share the same RecurrentStatePool instance.
        self.assertIs(rsp, hybrid_pool.recurrent_state_pool)
        self.assertIs(rsp, mp.recurrent_state_pool)
        self.assertIs(kv_stub, mp.token_to_kv_pool)

    def test_memory_pools_replace_all_with_kv_only(self):
        """Non-hybrid path: MemoryPools.replace_all dispatches to kv pool."""
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _build_non_hybrid_memory_pools,
        )

        captured = {}
        kv_stub = SimpleNamespace(replace_buffer=lambda v: captured.setdefault("v", v))
        mp = _build_non_hybrid_memory_pools(token_to_kv_pool=kv_stub)
        mp.replace_all({"token_to_kv_pool": ["dummy_layer"]})
        self.assertEqual(captured["v"], ["dummy_layer"])


class TestPerReqStateBytesFromConfig(unittest.TestCase):
    """_per_req_state_bytes_from_config wraps _compute_recurrent_per_req_bytes
    after resolving temporal_dtype / conv_dtype with the same env-var lookup
    that RecurrentStatePool uses, so the budget estimate matches the pool's
    actual allocation."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def test_default_dtypes_match_pool_construction(self):
        """Default temporal=f32 (4) + conv=bf16 (2). Compare against the
        canonical per_req formula at tp=1."""
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _per_req_state_bytes_from_config,
        )

        cfg = {
            "kda_layers": [0, 1, 2],
            "num_heads": 4,
            "head_dim": 16,
            "short_conv_kernel_size": 4,
        }
        bytes_ = _per_req_state_bytes_from_config(cfg, tp_size=1)
        proj_size = 4 * 16 + 2 * (4 * 16)  # = 192
        expected_recurrent = 3 * 4 * 16 * 16 * 4
        expected_conv = 3 * 3 * proj_size * 2
        self.assertEqual(bytes_, expected_recurrent + expected_conv)

    def test_tp_size_divides_per_device_bytes(self):
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _per_req_state_bytes_from_config,
        )

        cfg = {
            "kda_layers": [0, 1],
            "num_heads": 8,
            "head_dim": 16,
            "short_conv_kernel_size": 4,
        }
        tp1 = _per_req_state_bytes_from_config(cfg, tp_size=1)
        tp2 = _per_req_state_bytes_from_config(cfg, tp_size=2)
        # Both recurrent and conv components scale by 1/tp.
        self.assertEqual(tp1, tp2 * 2)


class TestStateToKvRatioActuallySplitsBudget(unittest.TestCase):
    """Structural lock-in: init_memory_pool must wire the budget split helpers
    in the hybrid branch BEFORE token_to_kv_pool / hybrid pool construction.
    Otherwise state_to_kv_ratio is dead code: KV pool keeps the full profile
    and the recurrent state pool starves at runtime."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def _src(self):
        import inspect

        from sgl_jax.srt.model_executor.model_runner import ModelRunner

        return inspect.getsource(ModelRunner.init_memory_pool)

    def test_split_helpers_are_called(self):
        src = self._src()
        self.assertIn("_split_state_kv_budget(", src)
        self.assertIn("_compute_max_num_reqs_from_state_budget(", src)
        self.assertIn("self._compute_cell_size()", src)
        self.assertIn("_per_req_state_bytes_from_config(", src)

    def test_split_happens_before_kv_pool_construction(self):
        """The kv_budget recompute must precede MHATokenToKVPool construction;
        otherwise the pool is sized against the full profile."""
        src = self._src()
        split_pos = src.find("_split_state_kv_budget(")
        mha_pool_pos = src.find("MHATokenToKVPool(")
        self.assertGreater(split_pos, 0, "split helper must be called in init_memory_pool")
        self.assertGreater(
            mha_pool_pos, 0, "MHATokenToKVPool must be constructed in init_memory_pool"
        )
        self.assertLess(
            split_pos,
            mha_pool_pos,
            "_split_state_kv_budget must be called BEFORE MHATokenToKVPool() "
            "so the pool size reflects kv_budget, not the full profile.",
        )

    def test_max_num_reqs_recomputed_before_build_hybrid_pools(self):
        """max_num_reqs override must happen before _build_hybrid_pools so
        RecurrentStatePool is sized against state_budget, not the default
        formula."""
        src = self._src()
        reqs_pos = src.find("_compute_max_num_reqs_from_state_budget(")
        build_pos = src.find("_build_hybrid_pools(")
        self.assertGreater(reqs_pos, 0)
        self.assertGreater(build_pos, 0)
        self.assertLess(
            reqs_pos,
            build_pos,
            "max_num_reqs must be recomputed from state_budget BEFORE "
            "_build_hybrid_pools(...) so RecurrentStatePool gets the "
            "ratio-derived size.",
        )


class TestMlaAndSwaHybridRaiseNotImplemented(unittest.TestCase):
    """MLA + recurrent_state and SWA + recurrent_state are out of scope for
    this initial wiring (per-token byte formulas differ from MHA). Lock-in
    that init_memory_pool fails fast with NotImplementedError instead of
    silently sizing pools wrong."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def test_init_memory_pool_source_guards_mla_and_swa(self):
        """Source-level lock-in: the guard raises NotImplementedError when
        is_hybrid (SWA) or use_mla_backend is True in the recurrent branch."""
        import inspect

        from sgl_jax.srt.model_executor.model_runner import ModelRunner

        src = inspect.getsource(ModelRunner.init_memory_pool)
        self.assertIn("self.is_hybrid or self.use_mla_backend", src)
        self.assertIn("NotImplementedError", src)


if __name__ == "__main__":
    unittest.main()
