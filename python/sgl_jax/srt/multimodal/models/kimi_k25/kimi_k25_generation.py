import logging  
import jax  
import jax.numpy as jnp  
from flax import nnx  
  
from sgl_jax.srt.hf_transformers_utils import get_hf_text_config  
from sgl_jax.srt.layers.embeddings import ParallelLMHead  
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor  
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools  
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch  
from sgl_jax.srt.multimodal.models.qwen2_5VL.qwen2_5_vl_generation import Qwen2_5_VL_Model  
  
logger = logging.getLogger(__name__)  
  
  
class KimiK25ForConditionalGeneration(nnx.Module):  
    ''' 
    Dummy LLM stage for KimiK2.5.  
    '''
  
    def __init__(self, config=None, dtype=None, mesh=None):  
        super().__init__()  
        self.mesh = mesh  
        self.config = config  
        self.text_config = get_hf_text_config(config) or config  
        self.dtype = dtype or jnp.bfloat16  
  
        self.model = Qwen2_5_VL_Model(self.text_config, mesh=mesh, dtype=self.dtype)  
        if not getattr(self.text_config, "tie_word_embeddings", False):  
            self.lm_head = ParallelLMHead(  
                self.text_config.vocab_size,  
                self.text_config.hidden_size,  
                dtype=self.dtype,  
                param_dtype=self.dtype,  
                kernel_axes=("tensor", None),  
            )  
        self.logits_processor = LogitsProcessor(  
            self.text_config.vocab_size, mesh=self.mesh  
        )  
  
    def load_weights(self, model_config):  
        logger.warning(  
            "KimiK25ForConditionalGeneration: dummy load_weights"  
        )  
  
    def get_embed_and_head(self):  
        if getattr(self.text_config, "tie_word_embeddings", False):  
            w = self.model.embed_tokens.embedding.value  
            return (w, w)  
        return (self.model.embed_tokens.embedding.value, self.lm_head.embedding.value)  
  
    def set_embed_and_head(  
        self,  
        embed_weight: jax.Array | None = None,  
        head_weight: jax.Array | None = None,  
    ) -> None:  
        if embed_weight is not None:  
            self.model.embed_tokens.embedding.value = embed_weight  
        if head_weight is not None:  
            self.lm_head.embedding.value = head_weight  
  
    def __call__(  
        self,  
        forward_batch: ForwardBatch,  
        memory_pools: MemoryPools,  
        logits_metadata: LogitsMetadata,  
    ):  
        token_to_kv_pool = memory_pools.token_to_kv_pool  
        hidden_states, layers_kv_fused, layers_callback_flag = self.model(  
            forward_batch, token_to_kv_pool  
        )  
        if not getattr(self.text_config, "tie_word_embeddings", False):  
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)  
        else:  
            output = self.logits_processor(  
                hidden_states, self.model.embed_tokens, logits_metadata  
            )  
        return output, layers_kv_fused, layers_callback_flag, None
