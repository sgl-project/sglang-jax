from types import SimpleNamespace

import pytest

from sgl_jax.srt.model_executor.model_runner_kv_cache_mixin import (
    _enforce_recurrent_state_server_constraints,
)


def test_recurrent_state_allows_overlap_schedule_enabled():
    server_args = SimpleNamespace(
        disable_radix_cache=True,
        disable_overlap_schedule=False,
    )

    _enforce_recurrent_state_server_constraints(server_args)


def test_recurrent_state_still_requires_radix_cache_disabled():
    server_args = SimpleNamespace(
        disable_radix_cache=False,
        disable_overlap_schedule=False,
    )

    with pytest.raises(AssertionError, match="disable-radix-cache"):
        _enforce_recurrent_state_server_constraints(server_args)
