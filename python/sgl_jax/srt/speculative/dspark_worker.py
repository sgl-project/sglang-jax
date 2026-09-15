from sgl_jax.srt.speculative.dflash_worker import DFlashWorker
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm


class DSparkWorker(DFlashWorker):
    """DSpark Markov drafting with fixed-width target verification."""

    algorithm = SpeculativeAlgorithm.DSPARK

    @staticmethod
    def _draft_model_class():
        from sgl_jax.srt.models.dspark import DSparkDraftModel

        return DSparkDraftModel

    def _validate_draft_model(self, draft_model):
        if not self.sample_from_anchor:
            raise ValueError("DSPARK requires sampling from the anchor position.")
        if self.draft_query_tokens != int(draft_model.config.block_size):
            raise ValueError("DSPARK requires verify width = checkpoint block_size + 1.")
        if draft_model.markov_head.vocab_size != self._target_vocab_size:
            raise ValueError("DSPARK draft and target must have the same vocabulary size.")
