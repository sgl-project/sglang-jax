from __future__ import annotations


class DFlashWorker:
    """Placeholder for the DFlash runtime worker.

    Stage2 adds the JAX draft model, config guards, and verify data structures.
    The end-to-end worker loop is intentionally left for the next stage because it
    needs target-verify attention metadata and KV commit/free handling.
    """

    def __init__(self, server_args, target_worker):
        raise NotImplementedError(
            "DFLASH worker runtime is not implemented in stage2. "
            "Use this stage to validate DFlashDraftModel loading and greedy verify "
            "helpers; end-to-end DFlash serving is a follow-up stage."
        )
