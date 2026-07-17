"""RED coverage for GDN prefill implementation selection and dispatch."""

from __future__ import annotations

import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.gdn import ragged_gated_delta_rule_ref
from sgl_jax.srt.layers.attention.linear import gdn_backend
from sgl_jax.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.test_gdn_attention import create_test_data, gather_conv, gather_ssm


_SELECTOR_ENV = "SGLANG_JAX_GDN_PREFILL_IMPL"
_PALLAS_INTERPRET_ENV = "PALLAS_INTERPRET"
_ENV_UNSET = object()

# Keep this import-safe until I2 exports the optimized prefill implementation.
# I1 establishes ``ragged_gated_delta_rule_chunkwise`` as the stable public name.
_CHUNKWISE_CALLABLE_NAME = "ragged_gated_delta_rule_chunkwise"

mesh = create_device_mesh(
    ici_parallelism=[1, 1], dcn_parallelism=[1, 1], devices=[jax.devices()[0]]
)
jax.sharding.set_mesh(mesh)


@contextmanager
def _prefill_environment(selector=_ENV_UNSET, pallas_interpret=_ENV_UNSET):
    """Temporarily isolate only the selector and Pallas interpretation settings."""
    previous = {
        _SELECTOR_ENV: os.environ.get(_SELECTOR_ENV, _ENV_UNSET),
        _PALLAS_INTERPRET_ENV: os.environ.get(_PALLAS_INTERPRET_ENV, _ENV_UNSET),
    }
    requested = {
        _SELECTOR_ENV: selector,
        _PALLAS_INTERPRET_ENV: pallas_interpret,
    }
    try:
        for name, value in requested.items():
            if value is _ENV_UNSET:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is _ENV_UNSET:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class TestGDNPrefillImplementation(unittest.TestCase):
    """Selector behavior is fixed at backend initialization, never in a hot path."""

    def make_backend(self, *, head_k_dim=32, test_mesh=mesh):
        return GDNAttnBackend(
            num_k_heads=2,
            num_v_heads=4,
            head_k_dim=head_k_dim,
            head_v_dim=16,
            conv_kernel_size=3,
            mesh=test_mesh,
        )

    def expected_chunkwise_callable(self):
        from sgl_jax.srt.kernels import gdn

        callable_ = getattr(gdn, _CHUNKWISE_CALLABLE_NAME, None)
        self.assertIsNotNone(
            callable_,
            f"expected exported GDN chunkwise callable {_CHUNKWISE_CALLABLE_NAME!r}",
        )
        self.assertTrue(callable(callable_), "exported GDN chunkwise callable must be callable")
        return callable_

    def test_missing_selector_defaults_to_chunkwise_request(self):
        with _prefill_environment():
            backend = self.make_backend()

        self.assertEqual(backend.requested_impl, "chunkwise")

    def test_reference_request_binds_reference_callable(self):
        with _prefill_environment("reference"):
            backend = self.make_backend()

        self.assertEqual(backend.requested_impl, "reference")
        self.assertEqual(backend.effective_impl, "reference")
        self.assertIsNone(backend.fallback_reason)
        self.assertIs(backend._prefill_callable, ragged_gated_delta_rule_ref)

    def test_chunkwise_request_on_cpu_interpret_binds_chunkwise_callable(self):
        expected = self.expected_chunkwise_callable()
        with _prefill_environment("chunkwise", "true"):
            backend = self.make_backend()

        self.assertEqual(backend.requested_impl, "chunkwise")
        self.assertEqual(backend.effective_impl, "chunkwise")
        self.assertIsNone(backend.fallback_reason)
        self.assertIs(backend._prefill_callable, expected)

    def test_invalid_selector_is_rejected_during_initialization(self):
        with _prefill_environment("not-a-gdn-prefill-implementation"):
            with self.assertRaisesRegex(
                ValueError,
                r"SGLANG_JAX_GDN_PREFILL_IMPL.*not-a-gdn-prefill-implementation.*(chunkwise|reference)",
            ):
                self.make_backend()

    def test_chunkwise_request_on_cpu_falls_back_to_reference_with_platform_reason(self):
        with _prefill_environment("chunkwise"):
            backend = self.make_backend()

        self.assertEqual(backend.requested_impl, "chunkwise")
        self.assertEqual(backend.effective_impl, "reference")
        self.assertIsNotNone(backend.fallback_reason)
        self.assertIn("platform", backend.fallback_reason.lower())
        self.assertIn("cpu", backend.fallback_reason.lower())
        self.assertIs(backend._prefill_callable, ragged_gated_delta_rule_ref)

    def test_oversize_head_k_dimension_falls_back_with_shape_reason(self):
        with _prefill_environment("chunkwise", "true"):
            backend = self.make_backend(head_k_dim=257)

        self.assertEqual(backend.requested_impl, "chunkwise")
        self.assertEqual(backend.effective_impl, "reference")
        self.assertIsNotNone(backend.fallback_reason)
        self.assertIn("257", backend.fallback_reason)
        self.assertIn("256", backend.fallback_reason)
        self.assertIs(backend._prefill_callable, ragged_gated_delta_rule_ref)

    def test_tpu_decision_uses_only_the_supplied_mesh(self):
        fake_tpu_mesh = SimpleNamespace(
            shape={"data": 1, "tensor": 1},
            devices=np.asarray([SimpleNamespace(platform="tpu")], dtype=object),
        )
        with _prefill_environment("chunkwise"):
            with (
                mock.patch.object(
                    gdn_backend.jax,
                    "devices",
                    side_effect=AssertionError("selector must inspect the supplied mesh, not jax.devices"),
                ),
                mock.patch.object(
                    gdn_backend.jax,
                    "default_backend",
                    side_effect=AssertionError(
                        "selector must inspect the supplied mesh, not jax.default_backend"
                    ),
                ),
            ):
                backend = self.make_backend(test_mesh=fake_tpu_mesh)

        self.assertEqual(backend.requested_impl, "chunkwise")
        self.assertEqual(backend.effective_impl, "chunkwise")
        self.assertIsNone(backend.fallback_reason)
        self.assertIs(backend._prefill_callable, self.expected_chunkwise_callable())

    def test_environment_changes_after_initialization_do_not_change_saved_dispatch(self):
        with _prefill_environment("reference"):
            (
                forward_batch,
                pool,
                layer,
                q,
                k,
                v,
                a,
                b,
                initial_ssm,
                initial_conv,
                recurrent_indices,
            ) = create_test_data(
                "prefill",
                [3],
                num_k_heads=2,
                num_v_heads=4,
                head_k_dim=16,
                head_v_dim=16,
                conv_kernel_size=3,
                dtype=jnp.bfloat16,
                rng=np.random.default_rng(1),
                test_mesh=mesh,
                all_have_initial_state=True,
            )
            backend = forward_batch.attn_backend._backend

        saved_status = tuple(
            getattr(backend, field, None)
            for field in ("requested_impl", "effective_impl", "fallback_reason")
        )
        saved_callable_was_bound = hasattr(backend, "_prefill_callable")
        saved_callable = getattr(backend, "_prefill_callable", ragged_gated_delta_rule_ref)

        dispatch_calls = []

        def wrapped_reference(*args, **kwargs):
            dispatch_calls.append(True)
            return saved_callable(*args, **kwargs)

        backend._prefill_callable = wrapped_reference

        with _prefill_environment("invalid-after-initialization"):
            activation_sharding = NamedSharding(mesh, P("data", "tensor"))
            actual, (recurrent_buffer, conv_buffer_list) = layer(
                forward_batch,
                jax.device_put(q, activation_sharding),
                jax.device_put(k, activation_sharding),
                jax.device_put(v, activation_sharding),
                jax.device_put(a, activation_sharding),
                jax.device_put(b, activation_sharding),
                pool,
            )

        actual_ssm = gather_ssm(pool, recurrent_buffer, recurrent_indices)
        actual_conv = gather_conv(pool, conv_buffer_list[0], recurrent_indices)
        self.assertGreaterEqual(
            len(dispatch_calls),
            1,
            "forward_extend must invoke the saved _prefill_callable after initialization",
        )
        self.assertTrue(
            saved_callable_was_bound,
            "backend initialization must save the selected _prefill_callable",
        )
        self.assertIs(saved_callable, ragged_gated_delta_rule_ref)
        self.assertEqual(
            (backend.requested_impl, backend.effective_impl, backend.fallback_reason),
            saved_status,
        )
        self.assertIs(backend._prefill_callable, wrapped_reference)
        for name, actual_value, expected_shape in (
            ("output", actual, (q.shape[0], v.shape[1])),
            ("recurrent state", actual_ssm, initial_ssm.shape),
            ("conv state", actual_conv, initial_conv.shape),
        ):
            actual_value = np.asarray(actual_value)
            self.assertEqual(actual_value.shape, expected_shape)
            self.assertTrue(np.isfinite(actual_value).all(), f"{name} must be finite")

    def test_initialization_logs_requested_effective_and_fallback_status_once(self):
        with _prefill_environment("reference"):
            with self.assertLogs(level="INFO") as captured:
                self.make_backend()

        status_logs = [
            line
            for line in captured.output
            if all(field in line for field in ("requested_impl", "effective_impl", "fallback_reason"))
        ]
        self.assertEqual(len(status_logs), 1, "expected one selector status log per init")

    def make_numerical_fixture(
        self,
        selector,
        seq_lens,
        *,
        mode="prefill",
        seed=0,
        all_have_initial_state=False,
        test_mesh=mesh,
        num_k_heads=2,
        num_v_heads=4,
        head_k_dim=16,
        head_v_dim=16,
        conv_kernel_size=3,
    ):
        """Create one selector-isolated fixture without executing its numerical body."""
        with _prefill_environment(selector, "true"):
            values = create_test_data(
                mode,
                seq_lens,
                num_k_heads=num_k_heads,
                num_v_heads=num_v_heads,
                head_k_dim=head_k_dim,
                head_v_dim=head_v_dim,
                conv_kernel_size=conv_kernel_size,
                dtype=jnp.bfloat16,
                rng=np.random.default_rng(seed),
                test_mesh=test_mesh,
                all_have_initial_state=all_have_initial_state,
            )
        return SimpleNamespace(
            selector=selector,
            test_mesh=test_mesh,
            forward_batch=values[0],
            pool=values[1],
            layer=values[2],
            q=values[3],
            k=values[4],
            v=values[5],
            a=values[6],
            b=values[7],
            initial_ssm=values[8],
            initial_conv=values[9],
            recurrent_indices=values[10],
        )

    def assert_effective_dispatch(self, fixture, expected_impl):
        backend = fixture.forward_batch.attn_backend._backend
        self.assertTrue(
            hasattr(backend, "effective_impl"),
            "optimized-vs-reference prefill requires GDNAttnBackend.effective_impl",
        )
        self.assertEqual(
            backend.effective_impl,
            expected_impl,
            f"requested {fixture.selector!r} must be effective {expected_impl!r}",
        )
        self.assertEqual(
            getattr(backend, "requested_impl", None),
            fixture.selector,
            f"backend must report the requested implementation {fixture.selector!r}",
        )
        self.assertTrue(
            hasattr(backend, "_prefill_callable"),
            "optimized-vs-reference prefill requires a saved _prefill_callable",
        )
        if expected_impl == "chunkwise":
            self.assertIs(backend._prefill_callable, self.expected_chunkwise_callable())
        else:
            self.assertIs(
                backend._prefill_callable,
                ragged_gated_delta_rule_ref,
                "reference selection must completely bypass optimized prefill",
            )

    @staticmethod
    def pool_arrays(fixture):
        recurrent, conv_list = fixture.pool.get_linear_recurrent_layer_cache(
            fixture.layer.layer_id
        )
        return np.asarray(recurrent), np.asarray(conv_list[0])

    def assert_identical_fixtures(self, chunkwise, reference):
        for name in ("q", "k", "v", "a", "b"):
            np.testing.assert_array_equal(
                np.asarray(getattr(chunkwise, name)),
                np.asarray(getattr(reference, name)),
                err_msg=f"A/B {name} inputs must be identical",
            )
        for name, chunkwise_value, reference_value in (
            (
                "conv1d weight",
                chunkwise.layer.conv1d.weight.value,
                reference.layer.conv1d.weight.value,
            ),
            ("A_log", chunkwise.layer.A_log.value, reference.layer.A_log.value),
            ("dt_bias", chunkwise.layer.dt_bias.value, reference.layer.dt_bias.value),
        ):
            np.testing.assert_array_equal(
                np.asarray(chunkwise_value),
                np.asarray(reference_value),
                err_msg=f"A/B {name} parameters must be identical",
            )
        chunkwise_rec, chunkwise_conv = self.pool_arrays(chunkwise)
        reference_rec, reference_conv = self.pool_arrays(reference)
        np.testing.assert_array_equal(
            chunkwise_rec,
            reference_rec,
            err_msg="A/B initial recurrent pools must be identical",
        )
        np.testing.assert_array_equal(
            chunkwise_conv,
            reference_conv,
            err_msg="A/B initial conv pools must be identical",
        )
        chunkwise_meta = chunkwise.forward_batch.attn_backend.forward_metadata
        reference_meta = reference.forward_batch.attn_backend.forward_metadata
        for name in ("cu_q_lens", "recurrent_indices", "has_initial_state"):
            np.testing.assert_array_equal(
                np.asarray(getattr(chunkwise_meta, name)),
                np.asarray(getattr(reference_meta, name)),
                err_msg=f"A/B {name} metadata must be identical",
            )
        for name in ("recurrent_track_indices", "recurrent_track_mask"):
            chunkwise_value = getattr(chunkwise_meta, name)
            reference_value = getattr(reference_meta, name)
            self.assertEqual(
                chunkwise_value is None,
                reference_value is None,
                f"A/B {name} presence must be identical",
            )
            if chunkwise_value is not None:
                np.testing.assert_array_equal(
                    np.asarray(chunkwise_value),
                    np.asarray(reference_value),
                    err_msg=f"A/B {name} metadata must be identical",
                )

    def make_prefill_ab(self, seq_lens, *, configure=None, **kwargs):
        chunkwise = self.make_numerical_fixture("chunkwise", seq_lens, **kwargs)
        reference = self.make_numerical_fixture("reference", seq_lens, **kwargs)
        if configure is not None:
            configure(chunkwise)
            configure(reference)
        # This guard is deliberately before either layer call. Without it the
        # current backend would execute reference twice and create a false pass.
        self.assert_effective_dispatch(chunkwise, "chunkwise")
        self.assert_effective_dispatch(reference, "reference")
        self.assert_identical_fixtures(chunkwise, reference)
        return chunkwise, reference

    @staticmethod
    def execute_fixture(fixture):
        activation_sharding = NamedSharding(fixture.test_mesh, P("data", "tensor"))
        with _prefill_environment(fixture.selector, "true"):
            output, (recurrent_buffer, conv_buffer_list) = fixture.layer(
                fixture.forward_batch,
                jax.device_put(fixture.q, activation_sharding),
                jax.device_put(fixture.k, activation_sharding),
                jax.device_put(fixture.v, activation_sharding),
                jax.device_put(fixture.a, activation_sharding),
                jax.device_put(fixture.b, activation_sharding),
                fixture.pool,
            )
        return SimpleNamespace(
            output=output,
            recurrent_buffer=recurrent_buffer,
            conv_buffer=conv_buffer_list[0],
            conv_buffer_list=conv_buffer_list,
        )

    def assert_numerical_ab(self, chunkwise, reference, *, rtol=2e-2, atol=1e-2):
        for name, chunkwise_value, reference_value in (
            ("output", chunkwise.output, reference.output),
            ("full recurrent pool", chunkwise.recurrent_buffer, reference.recurrent_buffer),
            ("full conv pool", chunkwise.conv_buffer, reference.conv_buffer),
        ):
            chunkwise_np = np.asarray(chunkwise_value)
            reference_np = np.asarray(reference_value)
            self.assertTrue(np.isfinite(chunkwise_np).all(), f"chunkwise {name} must be finite")
            self.assertTrue(np.isfinite(reference_np).all(), f"reference {name} must be finite")
            np.testing.assert_allclose(
                chunkwise_np,
                reference_np,
                rtol=rtol,
                atol=atol,
                err_msg=f"chunkwise/reference {name} mismatch",
            )

    def expand_pool(self, fixture, *, size=8):
        old_recurrent, old_conv = self.pool_arrays(fixture)
        old_pool = fixture.pool
        expanded = RecurrentStatePool(
            linear_recurrent_layer_ids=[fixture.layer.layer_id],
            size=size,
            num_heads=old_pool.num_heads,
            head_dim=old_pool.head_dim,
            conv_kernel_size=old_pool.conv_kernel_size,
            mesh=fixture.test_mesh,
            dp_size=1,
            recurrent_partition_axis="tensor",
            conv_partition_axis="tensor",
            data_partition_axis="data",
            temporal_dtype=old_pool.temporal_dtype,
            conv_dtype=old_pool.conv_dtype,
            num_k_heads=old_pool.num_k_heads,
            head_k_dim=old_pool.head_k_dim,
        )
        recurrent = np.full(
            (expanded.total_slots,) + old_recurrent.shape[1:], 0.375, dtype=old_recurrent.dtype
        )
        conv = np.full(
            (expanded.total_slots,) + old_conv.shape[1:], -0.5, dtype=old_conv.dtype
        )
        recurrent[: old_recurrent.shape[0]] = old_recurrent
        conv[: old_conv.shape[0]] = old_conv
        # Make dummy slot 0 nontrivial so an accidental zero/write is observable.
        recurrent[0] = 0.625
        conv[0] = -0.75
        expanded.replace_buffer(
            (
                [jax.device_put(jnp.asarray(recurrent), expanded.recurrent_sharding)],
                [[jax.device_put(jnp.asarray(conv), expanded.conv_sharding)]],
            )
        )
        fixture.pool = expanded

    @staticmethod
    def set_metadata_indices(fixture, state_indices, track_indices=None, track_mask=None):
        data_sharding = NamedSharding(fixture.test_mesh, P("data"))
        metadata = fixture.forward_batch.attn_backend.forward_metadata
        metadata.recurrent_indices = jax.device_put(
            np.asarray(state_indices, dtype=np.int32), data_sharding
        )
        if track_indices is not None:
            metadata.recurrent_track_indices = jax.device_put(
                np.asarray(track_indices, dtype=np.int32), data_sharding
            )
            metadata.recurrent_track_mask = jax.device_put(
                np.asarray(track_mask, dtype=np.int32), data_sharding
            )

    def test_numerical_fresh_and_mixed_continuing_prefill(self):
        for case, has_initial_state in (
            ("fresh", [False, False]),
            ("mixed", [False, True]),
        ):
            with self.subTest(case=case):
                chunkwise, reference = self.make_prefill_ab(
                    [7, 5], seed=11, all_have_initial_state=has_initial_state
                )
                self.assert_numerical_ab(
                    self.execute_fixture(chunkwise), self.execute_fixture(reference)
                )

    def test_numerical_raw_gate_activation_matches_reference(self):
        def configure(fixture):
            token = np.arange(fixture.a.shape[0], dtype=np.float32)[:, None]
            head = np.asarray([-3.0, -0.75, 1.25, 3.5], dtype=np.float32)[None, :]
            raw_a = head + 1.75 * np.sin(token * 0.37 + head * 0.19)
            head_sharding = NamedSharding(fixture.test_mesh, P("data", "tensor"))
            param_sharding = NamedSharding(fixture.test_mesh, P("tensor"))
            fixture.a = jax.device_put(jnp.asarray(raw_a, dtype=jnp.bfloat16), head_sharding)
            fixture.layer.dt_bias.value = jax.device_put(
                jnp.asarray([-1.5, -0.2, 0.7, 1.4], dtype=jnp.bfloat16), param_sharding
            )
            fixture.layer.A_log.value = jax.device_put(
                jnp.asarray([-1.2, -0.4, 0.3, 0.8], dtype=jnp.float32), param_sharding
            )

        chunkwise, reference = self.make_prefill_ab(
            [65], seed=17, all_have_initial_state=True, configure=configure
        )
        self.assert_numerical_ab(self.execute_fixture(chunkwise), self.execute_fixture(reference))

    def test_numerical_partial_chunk_lengths_match_reference(self):
        for seq_len in (1, 63, 65, 127, 129):
            with self.subTest(seq_len=seq_len):
                chunkwise, reference = self.make_prefill_ab(
                    [seq_len], seed=23, all_have_initial_state=True
                )
                self.assert_numerical_ab(
                    self.execute_fixture(chunkwise), self.execute_fixture(reference)
                )

    def test_numerical_zero_length_preserves_dummy_and_unused_slots(self):
        def configure_padded(fixture):
            self.expand_pool(fixture)
            self.set_metadata_indices(fixture, [1, 0, 3])

        def copy_real_request_states(padded, control):
            padded_recurrent, padded_conv = self.pool_arrays(padded)
            control_recurrent, control_conv = self.pool_arrays(control)
            control_recurrent = control_recurrent.copy()
            control_conv = control_conv.copy()
            control_recurrent[[1, 2]] = padded_recurrent[[1, 3]]
            control_conv[[1, 2]] = padded_conv[[1, 3]]
            control.pool.replace_buffer(
                (
                    [
                        jax.device_put(
                            jnp.asarray(control_recurrent), control.pool.recurrent_sharding
                        )
                    ],
                    [[jax.device_put(jnp.asarray(control_conv), control.pool.conv_sharding)]],
                )
            )

        def effective_initial_states(fixture, request_rows, state_slots):
            recurrent, conv = self.pool_arrays(fixture)
            has_initial = np.asarray(
                fixture.forward_batch.attn_backend.forward_metadata.has_initial_state
            )[request_rows]
            return (
                np.where(
                    has_initial[:, None, None, None],
                    recurrent[state_slots],
                    np.zeros_like(recurrent[state_slots]),
                ),
                np.where(
                    has_initial[:, None, None],
                    conv[state_slots],
                    np.zeros_like(conv[state_slots]),
                ),
            )

        def assert_equivalent_control_setup(padded, control):
            for name in ("q", "k", "v", "a", "b"):
                np.testing.assert_array_equal(
                    np.asarray(getattr(padded, name)),
                    np.asarray(getattr(control, name)),
                    err_msg=f"padded/control {name} inputs must be identical",
                )
            for name, padded_value, control_value in (
                (
                    "conv1d weight",
                    padded.layer.conv1d.weight.value,
                    control.layer.conv1d.weight.value,
                ),
                ("A_log", padded.layer.A_log.value, control.layer.A_log.value),
                ("dt_bias", padded.layer.dt_bias.value, control.layer.dt_bias.value),
            ):
                np.testing.assert_array_equal(
                    np.asarray(padded_value),
                    np.asarray(control_value),
                    err_msg=f"padded/control {name} parameters must be identical",
                )
            padded_effective = effective_initial_states(padded, [0, 2], [1, 3])
            control_effective = effective_initial_states(control, [0, 1], [1, 2])
            for name, padded_value, control_value in zip(
                ("recurrent", "conv"), padded_effective, control_effective
            ):
                np.testing.assert_array_equal(
                    padded_value,
                    control_value,
                    err_msg=f"padded/control effective {name} initial states must be identical",
                )

        chunkwise = self.make_numerical_fixture(
            "chunkwise", [65, 0, 63], seed=29, all_have_initial_state=[True, True, False]
        )
        reference = self.make_numerical_fixture(
            "reference", [65, 0, 63], seed=29, all_have_initial_state=[True, True, False]
        )
        chunkwise_control = self.make_numerical_fixture(
            "chunkwise", [65, 63], seed=29, all_have_initial_state=[True, False]
        )
        reference_control = self.make_numerical_fixture(
            "reference", [65, 63], seed=29, all_have_initial_state=[True, False]
        )
        for padded, control in (
            (chunkwise, chunkwise_control),
            (reference, reference_control),
        ):
            configure_padded(padded)
            copy_real_request_states(padded, control)
            np.testing.assert_array_equal(
                np.asarray(padded.forward_batch.attn_backend.forward_metadata.cu_q_lens),
                np.asarray([0, 65, 65, 128], dtype=np.int32),
                err_msg="padded cu_q_lens must contain the duplicate zero-length boundary",
            )
            np.testing.assert_array_equal(
                np.asarray(control.forward_batch.attn_backend.forward_metadata.cu_q_lens),
                np.asarray([0, 65, 128], dtype=np.int32),
                err_msg="no-zero control cu_q_lens must contain only the two real requests",
            )
            assert_equivalent_control_setup(padded, control)

        # Build and validate both controls before this guard so fixture or RNG
        # mismatches cannot be hidden by the expected selector-only RED.
        self.assert_effective_dispatch(chunkwise, "chunkwise")
        self.assert_effective_dispatch(reference, "reference")
        self.assert_effective_dispatch(chunkwise_control, "chunkwise")
        self.assert_effective_dispatch(reference_control, "reference")
        self.assert_identical_fixtures(chunkwise, reference)
        self.assert_identical_fixtures(chunkwise_control, reference_control)
        chunkwise_before = self.pool_arrays(chunkwise)
        reference_before = self.pool_arrays(reference)
        chunkwise_result = self.execute_fixture(chunkwise)
        reference_result = self.execute_fixture(reference)
        chunkwise_control_result = self.execute_fixture(chunkwise_control)
        reference_control_result = self.execute_fixture(reference_control)
        self.assertEqual(chunkwise_result.output.shape[0], 65 + 63)
        self.assertEqual(reference_result.output.shape[0], 65 + 63)
        self.assert_numerical_ab(chunkwise_result, reference_result)
        self.assert_numerical_ab(chunkwise_control_result, reference_control_result)
        for name, padded_result, control_result in (
            ("chunkwise", chunkwise_result, chunkwise_control_result),
            ("reference", reference_result, reference_control_result),
        ):
            for value_name, padded_value, control_value in (
                ("real-token output", padded_result.output, control_result.output),
                (
                    "real-request recurrent state",
                    np.asarray(padded_result.recurrent_buffer)[[1, 3]],
                    np.asarray(control_result.recurrent_buffer)[[1, 2]],
                ),
                (
                    "real-request conv state",
                    np.asarray(padded_result.conv_buffer)[[1, 3]],
                    np.asarray(control_result.conv_buffer)[[1, 2]],
                ),
            ):
                padded_value = np.asarray(padded_value)
                control_value = np.asarray(control_value)
                self.assertTrue(np.isfinite(padded_value).all(), f"{name} padded {value_name}")
                self.assertTrue(np.isfinite(control_value).all(), f"{name} control {value_name}")
                np.testing.assert_allclose(
                    padded_value,
                    control_value,
                    rtol=2e-2,
                    atol=1e-2,
                    err_msg=f"{name} zero-length row contaminated {value_name}",
                )
        for name, result, before in (
            ("chunkwise", chunkwise_result, chunkwise_before),
            ("reference", reference_result, reference_before),
        ):
            np.testing.assert_array_equal(
                np.asarray(result.recurrent_buffer)[0],
                before[0][0],
                err_msg=f"{name} recurrent dummy slot 0 changed",
            )
            np.testing.assert_array_equal(
                np.asarray(result.conv_buffer)[0],
                before[1][0],
                err_msg=f"{name} conv dummy slot 0 changed",
            )
            np.testing.assert_array_equal(
                np.asarray(result.recurrent_buffer)[2],
                before[0][2],
                err_msg=f"{name} unrelated recurrent slot changed",
            )
            np.testing.assert_array_equal(
                np.asarray(result.conv_buffer)[2],
                before[1][2],
                err_msg=f"{name} unrelated conv slot changed",
            )

    def test_numerical_track_snapshot_respects_boundary_and_zero_length(self):
        def configure(fixture):
            self.expand_pool(fixture)
            # Row 0 is a valid boundary, row 1 is zero-length despite a true
            # track mask, and row 2 is a real request away from a boundary.
            self.set_metadata_indices(
                fixture,
                [1, 0, 3],
                track_indices=[4, 5, 6],
                track_mask=[1, 1, 0],
            )

        chunkwise, reference = self.make_prefill_ab(
            [65, 0, 63],
            seed=31,
            all_have_initial_state=[True, True, True],
            configure=configure,
        )
        chunkwise_before = self.pool_arrays(chunkwise)
        reference_before = self.pool_arrays(reference)
        chunkwise_result = self.execute_fixture(chunkwise)
        reference_result = self.execute_fixture(reference)
        self.assert_numerical_ab(chunkwise_result, reference_result)
        for name, result, before in (
            ("chunkwise", chunkwise_result, chunkwise_before),
            ("reference", reference_result, reference_before),
        ):
            recurrent = np.asarray(result.recurrent_buffer)
            conv = np.asarray(result.conv_buffer)
            np.testing.assert_array_equal(
                recurrent[4], recurrent[1], err_msg=f"{name} recurrent track snapshot mismatch"
            )
            np.testing.assert_array_equal(
                conv[4], conv[1], err_msg=f"{name} conv track snapshot mismatch"
            )
            for track_slot in (5, 6):
                np.testing.assert_array_equal(
                    recurrent[track_slot],
                    before[0][track_slot],
                    err_msg=f"{name} recurrent track slot {track_slot} changed",
                )
                np.testing.assert_array_equal(
                    conv[track_slot],
                    before[1][track_slot],
                    err_msg=f"{name} conv track slot {track_slot} changed",
                )

    def test_numerical_prefill_then_decode_continuation_matches_reference(self):
        chunkwise, reference = self.make_prefill_ab(
            [65], seed=37, all_have_initial_state=True
        )
        chunkwise_prefill = self.execute_fixture(chunkwise)
        reference_prefill = self.execute_fixture(reference)
        self.assert_numerical_ab(chunkwise_prefill, reference_prefill)

        chunkwise.pool.replace_buffer(
            ([chunkwise_prefill.recurrent_buffer], [chunkwise_prefill.conv_buffer_list])
        )
        reference.pool.replace_buffer(
            ([reference_prefill.recurrent_buffer], [reference_prefill.conv_buffer_list])
        )
        chunkwise_decode = self.make_numerical_fixture(
            "chunkwise", [1], mode="decode", seed=41, all_have_initial_state=True
        )
        reference_decode = self.make_numerical_fixture(
            "reference", [1], mode="decode", seed=41, all_have_initial_state=True
        )
        self.assert_effective_dispatch(chunkwise_decode, "chunkwise")
        self.assert_effective_dispatch(reference_decode, "reference")
        for decode_fixture, prefill_fixture in (
            (chunkwise_decode, chunkwise),
            (reference_decode, reference),
        ):
            decode_fixture.layer = prefill_fixture.layer
            decode_fixture.pool = prefill_fixture.pool
        self.assert_identical_fixtures(chunkwise_decode, reference_decode)
        self.assert_numerical_ab(
            self.execute_fixture(chunkwise_decode), self.execute_fixture(reference_decode)
        )

    def test_numerical_sharded_dp2_tp2_matches_reference(self):
        if jax.device_count() != 4:
            self.skipTest("sharded DP2/TP2 numerical RED requires exactly four forced CPU devices")

        from sgl_jax.test.test_gdn_attention_dp import (
            create_test_data as create_dp_test_data,
        )
        from sgl_jax.test.test_gdn_attention_dp import set_mesh

        sharded_mesh = set_mesh(tp_size=2, dp_size=2)
        lens_per_rank = {0: [65], 1: [63]}

        def make(selector):
            with _prefill_environment(selector, "true"):
                values = create_dp_test_data(
                    "prefill",
                    lens_per_rank,
                    num_k_heads=4,
                    num_v_heads=8,
                    head_k_dim=16,
                    head_v_dim=16,
                    conv_kernel_size=3,
                    dtype=jnp.bfloat16,
                    mesh=sharded_mesh,
                    dp_size=2,
                    seed=43,
                    has_initial_state_per_rank={0: [False], 1: [True]},
                )
            return SimpleNamespace(
                selector=selector,
                test_mesh=sharded_mesh,
                forward_batch=values[0],
                pool=values[1],
                layer=values[2],
                q=values[3],
                k=values[4],
                v=values[5],
                a=values[6],
                b=values[7],
                per_dp_infos=values[8],
                per_dp_token_padding=values[10],
            )

        chunkwise = make("chunkwise")
        reference = make("reference")
        self.assert_effective_dispatch(chunkwise, "chunkwise")
        self.assert_effective_dispatch(reference, "reference")
        self.assert_identical_fixtures(chunkwise, reference)
        chunkwise_result = self.execute_fixture(chunkwise)
        reference_result = self.execute_fixture(reference)

        chunkwise_valid = []
        reference_valid = []
        for dp_rank in range(2):
            valid = sum(chunkwise.per_dp_infos[dp_rank]["seq_lens"])
            offset = dp_rank * chunkwise.per_dp_token_padding
            chunkwise_valid.append(np.asarray(chunkwise_result.output)[offset : offset + valid])
            reference_valid.append(np.asarray(reference_result.output)[offset : offset + valid])
        chunkwise_result.output = np.concatenate(chunkwise_valid, axis=0)
        reference_result.output = np.concatenate(reference_valid, axis=0)
        self.assert_numerical_ab(
            chunkwise_result,
            reference_result,
            rtol=2e-2,
            atol=5e-2,
        )

    def run_decode(self, selector):
        with _prefill_environment(selector):
            (
                forward_batch,
                pool,
                layer,
                q,
                k,
                v,
                a,
                b,
                _initial_ssm,
                _initial_conv,
                recurrent_indices,
            ) = create_test_data(
                "decode",
                [1],
                num_k_heads=2,
                num_v_heads=4,
                head_k_dim=16,
                head_v_dim=16,
                conv_kernel_size=3,
                dtype=jnp.bfloat16,
                rng=np.random.default_rng(0),
                test_mesh=mesh,
                all_have_initial_state=True,
            )

            activation_sharding = NamedSharding(mesh, P("data", "tensor"))
            actual, (recurrent_buffer, conv_buffer_list) = layer(
                forward_batch,
                jax.device_put(q, activation_sharding),
                jax.device_put(k, activation_sharding),
                jax.device_put(v, activation_sharding),
                jax.device_put(a, activation_sharding),
                jax.device_put(b, activation_sharding),
                pool,
            )
            return (
                np.asarray(actual),
                np.asarray(gather_ssm(pool, recurrent_buffer, recurrent_indices)),
                np.asarray(gather_conv(pool, conv_buffer_list[0], recurrent_indices)),
            )

    def test_decode_matches_between_reference_and_chunkwise_requests(self):
        reference = self.run_decode("reference")
        chunkwise = self.run_decode("chunkwise")

        for name, reference_value, chunkwise_value in zip(
            ("output", "recurrent state", "conv state"), reference, chunkwise
        ):
            self.assertTrue(np.isfinite(reference_value).all(), f"reference {name} must be finite")
            self.assertTrue(np.isfinite(chunkwise_value).all(), f"chunkwise {name} must be finite")
            np.testing.assert_allclose(
                chunkwise_value,
                reference_value,
                rtol=2e-2,
                atol=1e-2,
                err_msg=f"decode {name} changed with the prefill implementation request",
            )


if __name__ == "__main__":
    unittest.main()
