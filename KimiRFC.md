### \[RFC\] Support Kimi-K2.5 on SGL\_JAX

#### Summary

This RFC proposes adding inference support for **Kimi-K2.5** (moonshotai/Kimi-K2.5) in sglang-jax's multimodal pipeline. Kimi-K2.5 is a native vision-language model from Moonshot AI that integrates a DeepSeek-V3-style MoE LLM with a custom vision encoder.


#### 1\. Model Architecture

##### 1.1 LLM Backbone

The LLM is a **DeepSeek-V3-style** transformer (model\_type: "kimi\_k2") characterized by massive sparsity and memory-efficient attention:

* **Attention:** Uses **Multi-head Latent Attention (MLA)** to compress the KV cache.  
  * **Latent Ranks:** kv\_lora\_rank=512, q\_lora\_rank=1536.  
  * **Head Dimensions:** qk\_nope\_head\_dim=128, qk\_rope\_head\_dim=64, v\_head\_dim=128.  
* **Mixture-of-Experts (MoE):** 61 layers total.  
  * **Layer 0:** Dense (acts as a standardizer).  
  * **Layers 1–60:** Sparse MoE with **384 routed experts** and **1 shared expert**.  
  * **Routing:** Selects **top-8** experts per token using sigmoid scoring with bias correction.  
* **Positional Encoding:** Uses **YaRN RoPE** (factor=64.0, theta=50,000).  
  * **Crucial Note:** Unlike Qwen-VL, Kimi-K2.5 does **not** use MRoPE; vision tokens are treated as standard sequence elements.  
* **Capacity:** 1.04T total parameters (\~32B active), 262,144 token context window, and 163,840 vocabulary size.


##### 1.2 Vision Encoder: MoonViT-3D

Kimi-K2.5 uses the **MoonViT-3D** encoder, a 425M parameter model designed for unified image and video understanding:

* **Structure:** 27 layers, hidden dimension of 1152, and 16 attention heads.  
* **Positional Embedding:** KimiDividedFixedPosEmb (separate H, W, T embeddings).  
* **Spatio-Temporal Attention:** For video, it performs sequential spatial passes (within frames) and temporal passes (across frames) rather than full 3D attention.  
* **Patch Merger:** Reduces token density by pooling 2 times 2 spatial patches and applying temporal pooling before projecting to the LLM's 7168 hidden dimension.

#### 2\. Integration into sglang-jax

##### 2.1 Component Reuse

The following sglang-jax components are directly compatible:

* [DeepseekV3ForCausalLM](https://github.com/sgl-project/sglang-jax/blob/deepseekv3/python/sgl_jax/srt/models/deepseekv3.py#L392)   
* apply\_linear\_quantization (for bf16 → FP8 on-the-fly conversion).

##### 2.2 New Developments

* **kimi\_vit.py**: A full implementation of MoonViT-3D, including the divided positional embeddings and the 2 times 2 patch merger.  
* **kimi\_k2\_5\_generation.py**: A KimiK25LLMModel subclass that enables the multimodal prefill path (using input\_embedding instead of token IDs for the vision sequence).  
* **Stage Config:** A new YAML defining the Stage 0 (ViT) and Stage 1 (LLM) pipeline topology.


#### 3\. Core Model Implementation

##### 3.1 Kimi Vision Encoder (kimi\_vit.py)

This is the primary development task. The Kimi ViT architecture requires:

1. **KimiDividedFixedPosEmb:** Separate sinusoidal embeddings for height, width, and time, interpolated at runtime.  
2. **KimiVisionAttention:** Implements spatial\_temporal attention (sequential spatial and temporal passes).  
3. **KimiPatchMerger:** A projector that pools 2 times 2 spatial patches and applies temporal pooling, projecting from 4,608 to 7,168 hidden dimensions.

##### 3.2 Multimodal LLM Wrapper

The KimiK25LLMModel (a subclass of DeepseekV3Model) will override the execution call to inspect forward\_batch.input\_embedding. During the **prefill** phase, it will use the merged embeddings from Stage 0 instead of standard token embeddings.

#### 4\. End-to-End Data Flow

1. **Tokenization:** The MultimodalTokenizer processes inputs, replacing media placeholders with media\_placeholder\_token\_id=163605 and skipping MRoPE.  
2. **Stage 0:** The KimiVisionTransformer processes pixel values. The KimiPatchMerger reduces token counts and projects them to the LLM's hidden dimension.  
3. **Embedding Merge:** Vision embeddings are inserted into the text embedding sequence at the placeholder locations.  
4. **Stage 1:** The LLM performs a prefill using the merged embeddings, followed by standard autoregressive decoding for text generation.



#### 5\. Implementation Roadmap

1. **Weight Inspection:** Validate safetensors structure.  
2. **Config Setup:** Register YAML and dataclasses.  
3. **ViT Development:** Implement and isolate-test the Kimi Vision Encoder.  
4. **LLM Integration:** Wrap DeepseekV3 and wire the embedding injection path.  
5. **Runner Logic:** Generalize VitModelRunner to support the Kimi interface.  
6. **Validation:** Compare output logits against the HF reference implementation.

#### 6\. Milestones

##### 6.1 P0 Milestones

1. Vision Encoder Implementation \[Image support only\]  
2. Integration with LLM Backbone \[Deepseek v3\]  
3. E2E successful run on multihost  
4. Accuracy & sanity validations   
5. Baselining and comparison with B200 sglang oss version

##### 6.2 P1 Milestones

1. Video Input support  
2. Speculative decoding 
