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
