"""Errors shared by encoder scheduling and transfer backends."""


class EncoderBatchCapacityError(ValueError):
    """A batch cannot fit even when the transfer pool is otherwise empty."""
