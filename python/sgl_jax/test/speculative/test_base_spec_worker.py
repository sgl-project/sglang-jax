import pytest

from sgl_jax.srt.speculative.base_spec_worker import BaseDraftWorker, BaseSpecWorker


class ConcreteDraftWorker(BaseDraftWorker):
    def draft(self, model_worker_batch):
        return ("draft", model_worker_batch)

    def draft_extend_for_prefill(self, model_worker_batch, target_hidden_states, next_token_ids):
        return ("prefill", model_worker_batch, target_hidden_states, next_token_ids)

    def draft_extend_for_decode(self, model_worker_batch, batch_output):
        return ("decode", model_worker_batch, batch_output)


class ConcreteSpecWorker(BaseSpecWorker):
    def __init__(self):
        self._target_worker = object()
        self._draft_worker = ConcreteDraftWorker()

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    def forward_batch_speculative_generation(self, model_worker_batch):
        return self.draft_worker.draft(model_worker_batch)

    def verify(self, model_worker_batch, verify_input, cur_allocate_lens=None):
        return ("verify", model_worker_batch, verify_input, cur_allocate_lens)


def test_base_draft_worker_requires_public_methods():
    with pytest.raises(TypeError):
        BaseDraftWorker()


def test_base_spec_worker_requires_public_methods():
    with pytest.raises(TypeError):
        BaseSpecWorker()


def test_concrete_workers_expose_contract():
    worker = ConcreteSpecWorker()

    assert worker.target_worker is worker._target_worker
    assert isinstance(worker.draft_worker, BaseDraftWorker)
    assert worker.forward_batch_speculative_generation("batch") == ("draft", "batch")
    assert worker.verify("batch", "verify_input", "alloc") == (
        "verify",
        "batch",
        "verify_input",
        "alloc",
    )
