"""Test for the PD-mode shutdown handler installed by the scheduler.

This test depends on `_make_disagg_shutdown` in `sgl_jax.srt.managers.scheduler`,
which is wired up in Stage 2 (scheduler PD-mode dispatch). Lives in its own
file so Stage 1 can be PR'd without depending on Stage 2 symbols.
"""

from unittest import mock


def test_disagg_shutdown_handler_unregisters_and_drains():
    """Stage 4 review I2: the shutdown handler must call
    unregister_prefill, stop the heartbeat, and run the manager's
    graceful_shutdown in that order.
    """

    from sgl_jax.srt.managers.scheduler import _make_disagg_shutdown

    scheduler = mock.MagicMock()
    scheduler.disagg_bootstrap_key = "bkey"
    scheduler.disagg_bootstrap_client = mock.MagicMock()
    scheduler.disagg_heartbeat = mock.MagicMock()
    scheduler.disagg_kv_manager = mock.MagicMock()

    fn = _make_disagg_shutdown(scheduler, "prefill")
    fn()
    scheduler.disagg_bootstrap_client.unregister_prefill.assert_called_once_with(
        "bkey"
    )
    scheduler.disagg_heartbeat.stop.assert_called_once()
    scheduler.disagg_kv_manager.graceful_shutdown.assert_called_once()
    scheduler.disagg_kv_manager.zmq_notifier.stop.assert_called_once()
    # Idempotent.
    fn()
    scheduler.disagg_bootstrap_client.unregister_prefill.assert_called_once()
