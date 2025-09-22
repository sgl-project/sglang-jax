import os

TP_SIZE = int(os.environ.get("TP_SIZE", 1))
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={TP_SIZE}"
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import AutoTokenizer

from sgl_jax.srt.configs.load_config import LoadConfig, LoadFormat
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.model_executor.model_runner import ModelRunner
from sgl_jax.srt.model_loader.loader import JAXModelLoader
from sgl_jax.srt.precision_tracer import precision_tracer
from sgl_jax.srt.sampling.sampling_batch_info import SamplingBatchInfo, SamplingMetadata
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.test.test_utils import create_device_mesh


class TestModelRunner(unittest.TestCase):
    """Test for ModelRunner."""

    def setUp(self):
        """Set up ModelRunner"""
        num_processes = int(os.environ.get("SGL_JAX_NUM_PROCESSES", 1))
        process_id = int(os.environ.get("SGL_JAX_PROCESS_ID", 0))
        coordinator_address = os.environ.get(
            "SGL_JAX_COORDINATOR_ADDRESS", "localhost:10000"
        )
        if num_processes > 1:
            jax.distributed.initialize(
                coordinator_address=coordinator_address,
                num_processes=num_processes,
                process_id=process_id,
            )

        print(f"{jax.device_count()=} {jax.local_device_count()=}")

        # Use create_device_mesh following test_qwen_model.py pattern
        jax_devices = jax.devices()
        self.tp_size = TP_SIZE
        if len(jax_devices) < self.tp_size:
            raise ValueError(
                f"TP_SIZE {self.tp_size} is greater than the number of devices {len(jax_devices)}"
            )
        elif len(jax_devices) > self.tp_size:
            jax_devices = jax_devices[: self.tp_size]
        self.mesh = create_device_mesh(
            devices=jax_devices,
            ici_parallelism=[1, self.tp_size, 1, 1],
            dcn_parallelism=[1, 1, 1, 1],
        )

        # Create RNG
        self.rng = nnx.Rngs(42)
        self.enable_precision_tracer = os.environ.get("ENABLE_PRECISION_TRACER", "1")
        if self.enable_precision_tracer == "1":
            print("precision tracer enabled")
        # Create model config for Qwen-7B
        self.model_path = os.environ.get("MODEL_PATH", "/models/Qwen-7B")
        self.model_config = ModelConfig(
            model_path=self.model_path, model_override_args="{}", dtype="bfloat16"
        )

        # Create load config and JAX loader
        self.load_config = LoadConfig(load_format=LoadFormat.JAX, download_dir="/tmp/")
        self.jax_loader = JAXModelLoader(self.load_config, self.rng, self.mesh)

        # Setup ModelRunner
        self._setup_model_runner()

    def _setup_model_runner(self):
        """Setup ModelRunner with minimal required attributes."""
        # Create simplified ModelRunner for testing

        server_args = ServerArgs(
            model_path=self.model_path,
            trust_remote_code=True,
            device=os.environ.get("JAX_PLATFORMS", "tpu"),
        )

        req_to_token_pool = ReqToTokenPool(
            size=128, max_context_len=8192, mesh=self.mesh, dtype=jnp.int32
        )

        self.model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=0.1,
            tp_size=self.tp_size,
            server_args=server_args,
            mesh=self.mesh,
            rngs=self.rng,
            req_to_token_pool=req_to_token_pool,
        )

    def _get_tokenizer(self):
        """Get tokenizer from local path if available, otherwise use HuggingFace"""
        model_path = Path(self.model_path)

        # Check if it's a local path and has tokenizer files
        if model_path.exists():
            tokenizer_files = ["tokenizer_config.json"]
            has_tokenizer = any(
                (model_path / file).exists() for file in tokenizer_files
            )

            if has_tokenizer:
                print(f"Using local tokenizer from: {model_path}")
                try:
                    return AutoTokenizer.from_pretrained(
                        str(model_path), trust_remote_code=True
                    )
                except Exception as e:
                    print(f"  Failed to load local tokenizer: {e}")

        # Use HuggingFace model with network error handling
        try:
            print(f"Loading tokenizer from HuggingFace: {self.model_path}")
            return AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
        except Exception as e:
            print(f"Failed to load tokenizer from HuggingFace: {e}")
            raise RuntimeError(
                f"Could not load tokenizer from local path or HuggingFace: {e}"
            )

    def _new_forward_batch(self, input_ids, positions):
        """Create a ForwardBatch for testing."""
        total_tokens = sum(len(ids) for ids in input_ids)
        req_pool_indices = self.model_runner.req_to_token_pool.alloc(len(input_ids))
        cache_loc_index = self.model_runner.token_to_kv_pool_allocator.alloc(
            total_tokens
        )
        # out_cache_loc = self.model_runner.token_to_kv_pool_allocator.alloc(len(input_ids))

        # write to req_to_token_pool
        pt = 0
        for i, input in enumerate(input_ids):
            self.model_runner.req_to_token_pool.write(
                (req_pool_indices[i], slice(0, len(input))),
                cache_loc_index[pt : pt + len(input)],
            )
            pt += len(input)

        worker_batch = ModelWorkerBatch(
            bid=0,
            forward_mode=ForwardMode.EXTEND,
            input_ids=jnp.array(input_ids).flatten(),
            real_input_ids_len=sum(input_ids),
            real_bs=len(input_ids),
            req_pool_indices=jnp.array(req_pool_indices),
            seq_lens=jnp.array([len(ids) for ids in input_ids]),
            out_cache_loc=cache_loc_index,
            cache_loc=cache_loc_index,
            positions=jnp.array(positions),
            extend_start_loc=jnp.array([0]),
            sampling_info=SamplingBatchInfo(
                temperatures=jnp.full((1, 1), 1.0),
                is_all_greedy=True,
                top_ps=jnp.full((1, 1), 1.0),
                top_ks=jnp.ones((1, 1)),
                min_ps=jnp.full((1, 1), 0.0),
            ),
        )
        return worker_batch

    def _update_forward_batch(self, forward_batch: ForwardBatch, output_ids: jax.Array):
        """Update the forward batch with the next token ids."""
        out_cache_loc = self.model_runner.token_to_kv_pool_allocator.alloc(
            len(output_ids)
        )

        forward_batch.forward_mode = ForwardMode.DECODE
        forward_batch.input_ids = output_ids.flatten()
        forward_batch.positions = jnp.array(
            [seq_len for seq_len in forward_batch.seq_lens]
        )  # Use current seq_len as position

        batch_size = forward_batch.batch_size
        for i in range(batch_size):
            # write to req_to_token_pool
            self.model_runner.req_to_token_pool.write(
                (
                    forward_batch.req_pool_indices[i],
                    slice(forward_batch.seq_lens[i], forward_batch.seq_lens[i] + 1),
                ),
                out_cache_loc[i],
            )

        forward_batch.out_cache_loc = jnp.array(out_cache_loc)
        forward_batch.seq_lens = jnp.array(
            [seq_len + 1 for seq_len in forward_batch.seq_lens]
        )

        token_indices_with_all_reqs = self.model_runner.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices
        ]
        cache_loc_list = []
        for seq_idx in range(forward_batch.seq_lens.shape[0]):
            seq_len = forward_batch.seq_lens[seq_idx]
            cache_loc_list.append(token_indices_with_all_reqs[seq_idx][:seq_len])
        forward_batch.cache_loc = jnp.concatenate(cache_loc_list, axis=0)

        forward_batch.extend_start_loc = None
        return forward_batch

    def test_forward(self):
        """Test complete forward pass."""
        # Step 1: Extend phase (prefill)
        tokenizer = self._get_tokenizer()
        if self.enable_precision_tracer == "1":
            precision_tracer.start_trace()
        text = "1+1=?"
        encoded = tokenizer.encode(text, return_tensors="pt")
        extend_input_ids = [encoded[0].tolist()]
        extend_positions = [list(range(len(extend_input_ids[0])))]

        model_worker_batch = self._new_forward_batch(extend_input_ids, extend_positions)
        extend_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        with self.mesh:
            extend_output = self.model_runner.forward(
                extend_batch,
                LogitsMetadata.from_model_worker_batch(model_worker_batch, self.mesh),
            )

        # Verify forward_pass_id incremented
        self.assertEqual(self.model_runner.forward_pass_id, 1)

        # Verify extend output shape
        self.assertIsInstance(extend_output, LogitsProcessorOutput)
        self.assertEqual(
            extend_output.next_token_logits.shape, (1, self.model_config.vocab_size)
        )  # (batch_size, vocab_size)

        print(
            f" Extend phase completed. Output shape: {extend_output.next_token_logits.shape}"
        )

        # Step 2: Multiple decode phases (generation)
        # Continue from the extend batch for proper KV cache continuity
        decode_outputs = []
        current_batch = extend_batch  # Use the same batch for continuity

        # Sample the first token from extend output
        current_token = self.model_runner.sampler(
            extend_output,
            sampling_metadata=SamplingMetadata.from_model_worker_batch(
                model_worker_batch.sampling_info
            ),
        )

        # Collect all generated tokens
        all_generated_tokens = [current_token]

        for step in range(10):  # Generate 10 tokens
            print(f"step {step} current_token: {current_token}")
            current_batch = self._update_forward_batch(current_batch, current_token)
            with self.mesh:
                decode_output = self.model_runner.forward(
                    current_batch,
                    LogitsMetadata.from_model_worker_batch(
                        model_worker_batch, self.mesh
                    ),
                )
            decode_outputs.append(decode_output)

            # Verify decode output shape
            self.assertIsInstance(decode_output, LogitsProcessorOutput)
            self.assertEqual(
                decode_output.next_token_logits.shape, (1, self.model_config.vocab_size)
            )
            # Verify forward_pass_id incremented correctly
            self.assertEqual(self.model_runner.forward_pass_id, 2 + step)

            # Sample next token for the next iteration
            current_token = self.model_runner.sampler(
                decode_output,
                sampling_metadata=SamplingMetadata.from_model_worker_batch(
                    model_worker_batch.sampling_info
                ),
            )
            all_generated_tokens.append(current_token)
            print(f"step {step} current_token added: {current_token}")

        if self.enable_precision_tracer == "1":
            print("Ending precision tracer session...")
            debug_file = precision_tracer.stop_trace()
            if debug_file:
                print(f"Precision trace saved to: {debug_file}")
            else:
                print("Precision trace not saved")
        # Verify all decode outputs have consistent shapes
        for output in decode_outputs:
            self.assertEqual(
                output.next_token_logits.shape, (1, self.model_config.vocab_size)
            )
            self.assertEqual(output.next_token_logits.dtype, jnp.bfloat16)
        self.assertEqual(current_token.shape, (1, 1))  # (batch_size, 1)
        # Assertions for final verification
        self.assertEqual(len(decode_outputs), 10)
        # Verify all outputs are from the same model runner instance
        self.assertEqual(self.model_runner.forward_pass_id, 11)  # 1 extend + 10 decode

        # Decode the complete generated sequence
        print(f"All generated tokens: {all_generated_tokens}")
        if hasattr(tokenizer, "decode"):
            try:
                # Concatenate all generated tokens
                all_tokens = []
                for token_batch in all_generated_tokens:
                    all_tokens.extend(token_batch[0].tolist())

                # Decode the complete sequence
                decoded_text = tokenizer.decode(all_tokens)
                print(f"Complete decoded text: {decoded_text}")

                # Also decode just the generated part (without input)
                input_tokens = extend_input_ids[0]
                print(f"Input tokens: {input_tokens}")
                print(f"All tokens: {all_tokens}")
                print(f"Input length: {len(input_tokens)}")
                print(f"All tokens length: {len(all_tokens)}")
                generated_tokens = all_tokens[len(input_tokens) :]
                print(f"Generated tokens: {generated_tokens}")
                generated_text = tokenizer.decode(generated_tokens)
                print(f"Generated text only: {generated_text}")
            except Exception as e:
                print(f"Could not decode tokens: {e}")


if __name__ == "__main__":
    unittest.main()
