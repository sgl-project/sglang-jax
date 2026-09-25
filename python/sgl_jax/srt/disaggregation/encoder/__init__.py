"""Encoder disaggregation components and receive-backend initialization."""


def create_encoder_client(server_args, model_runner, token_buckets, apply_result):
    """Select the receive backend and prepare its storage before starting the client."""
    from sgl_jax.raiden import require_raiden_preloaded
    from sgl_jax.srt.disaggregation.encoder.client import EncoderClient
    from sgl_jax.srt.disaggregation.encoder.embedding_data import precompile_received_embeddings
    from sgl_jax.srt.disaggregation.encoder.raiden_pool import create_encoder_pool
    from sgl_jax.srt.disaggregation.encoder.raiden_receiver import RaidenReceiverBackend
    from sgl_jax.srt.disaggregation.host_ip import resolve_host_ip

    require_raiden_preloaded()
    transfer_timeout = server_args.encoder_request_timeout_seconds
    if transfer_timeout <= 0:
        raise ValueError("Raiden requires a positive encoder request timeout")
    host = resolve_host_ip(server_args.disaggregation_host_ip)
    channel_number = max(1, int(server_args.disaggregation_channel_number))
    pool = create_encoder_pool(server_args, model_runner.model_config, model_runner.mesh)
    if not server_args.disable_precompile:
        precompile_received_embeddings(pool.buffer, model_runner.model, token_buckets)
    backend = RaidenReceiverBackend(
        host=host,
        pool=pool,
        parallelism=channel_number,
        pool_size=server_args.encoder_transfer_pool_size,
        transfer_timeout_s=transfer_timeout,
    )
    control_timeout = server_args.encoder_control_timeout_seconds
    return EncoderClient(
        host=host,
        backend=backend,
        apply_result=apply_result,
        registration_workers=channel_number,
        registration_timeout=None if control_timeout <= 0 else control_timeout,
    )
