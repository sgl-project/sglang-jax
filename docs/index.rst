SGL-JAX Documentation
=====================

SGL-JAX is a JAX-based inference engine for LLM and multimodal serving on TPU.
This documentation keeps the main project docs in a Sphinx-style tree, while
the model cookbook is maintained as a Mintlify-style recipe collection.

.. toctree::
   :caption: Getting Started
   :maxdepth: 1

   get_started/install
   get_started/gke-tpu-install
   get_started/sglang-jax-gpu-install

.. toctree::
   :caption: Basic Usage
   :maxdepth: 1

   basic_usage/index
   basic_usage/qwen
   basic_usage/mimo_v2_flash
   basic_usage/mimo_v2.5_pro
   basic_usage/grok2-skypilot-serving

.. toctree::
   :caption: Cookbook
   :maxdepth: 1

   cookbook_overview

.. toctree::
   :caption: Features
   :maxdepth: 1

   features/index
   features/attention_backend
   features/chunked_prefill
   features/dtype_config
   features/dynamic_continuous_batching
   features/global_jit_compile
   features/partial_rollout
   features/quantization
   features/radix_cache
   features/return_routed_experts
   features/run_in_pathways
   features/speculative_decoding
   features/structured_output

.. toctree::
   :caption: Architecture
   :maxdepth: 1

   architecture/index
   architecture/01-architecture-overview
   architecture/02-entrypoints-and-tokenization
   architecture/03-scheduler
   architecture/04-model-executor
   architecture/05-models
   architecture/06-layers-and-attention
   architecture/07-kv-cache
   architecture/08-pallas-kernels
   architecture/09-speculative-decoding
   architecture/10-lora
   architecture/11-quantization
   architecture/12-multimodal
   architecture/13-configuration-reference
   architecture/14-pd-disaggregation

.. toctree::
   :caption: Multimodal
   :maxdepth: 1

   multimodal/index
   multimodal/multimodal_usage

.. toctree::
   :caption: Developer Guide
   :maxdepth: 1

   developer_guide/index
   developer_guide/contribution_guide
   developer_guide/benchmark_and_profiling
   developer_guide/ci_architecture
   developer_guide/jax_tutorial
   developer_guide/release_process
   developer_guide/tpu_resources_guide
   developer_guide/how_to_join_community

.. toctree::
   :caption: References
   :maxdepth: 1

   references/index
