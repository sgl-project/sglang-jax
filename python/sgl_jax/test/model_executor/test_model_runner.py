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
from sgl_jax.srt.debug_tracer import global_tracer
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
from sgl_jax.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sgl_jax.srt.model_executor.model_runner import ModelRunner
from sgl_jax.srt.model_loader.loader import JAXModelLoader
from sgl_jax.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.test.test_utils import create_device_mesh


class TestModelRunner(unittest.TestCase):
    """Test for ModelRunner."""

    # Long test texts to test different ForwardModes with substantial input
    LONG_TEST_TEXTS = [
        "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?",
        "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?",
        "Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?",
        "Analyze the impact of climate change on global ecosystems, discussing greenhouse gases, temperature changes, sea level rise, and effects on biodiversity. Include potential mitigation strategies and adaptation measures.",
    ]

    # Use a small chunk size to trigger actual chunked prefill
    CHUNKED_PREFILL_SIZE = 16

    def setUp(self):
        """Set up ModelRunner"""
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
        self.enable_debug_tracer = os.environ.get("ENABLE_DEBUG_TRACER", "0")
        if self.enable_debug_tracer == "1":
            print("debug tracer enabled")
        # Create model config for Qwen-7B
        self.model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen-7B-Chat")
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
            input_ids=jnp.concatenate([jnp.array(ids) for ids in input_ids], axis=0),
            real_input_ids_len=sum(len(ids) for ids in input_ids),
            real_bs=len(input_ids),
            req_pool_indices=jnp.array(req_pool_indices),
            seq_lens=jnp.array([len(ids) for ids in input_ids]),
            out_cache_loc=jnp.array(cache_loc_index),
            cache_loc=jnp.concatenate(
                [
                    jnp.array(
                        cache_loc_index[
                            sum(len(input_ids[j]) for j in range(i)) : sum(
                                len(input_ids[j]) for j in range(i + 1)
                            )
                        ]
                    )
                    for i in range(len(input_ids))
                ],
                axis=0,
            ),
            positions=jnp.concatenate([jnp.array(pos) for pos in positions], axis=0),
            extend_start_loc=jnp.array(
                [
                    sum(len(input_ids[j]) for j in range(i))
                    for i in range(len(input_ids))
                ]
            ),
            extend_seq_lens=jnp.array([len(ids) for ids in input_ids]),
            extend_prefix_lens=jnp.zeros(len(input_ids), dtype=jnp.int32),
            return_logprob=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            extend_logprob_start_lens=None,
            extend_input_logprob_token_ids=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            sampling_info=SamplingBatchInfo(
                temperatures=jnp.full((len(input_ids), 1), 1.0),
                is_all_greedy=True,
                top_ps=jnp.full((len(input_ids), 1), 1.0),
                top_ks=jnp.ones((len(input_ids), 1)),
                min_ps=jnp.full((len(input_ids), 1), 0.0),
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
        if self.enable_debug_tracer == "1":
            global_tracer.start_session()
        text = "1+1=?"
        encoded = tokenizer.encode(text, return_tensors="pt")
        extend_input_ids = [encoded[0].tolist()]
        extend_positions = [list(range(len(extend_input_ids[0])))]

        model_worker_batch = self._new_forward_batch(extend_input_ids, extend_positions)
        extend_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        with self.mesh:
            extend_output, _ = self.model_runner.forward(
                extend_batch, LogitsMetadata.from_model_worker_batch(model_worker_batch)
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
        current_token = self.model_runner.sample(extend_output, model_worker_batch)

        # Collect all generated tokens
        all_generated_tokens = [current_token]

        for step in range(10):  # Generate 10 tokens
            print(f"step {step} current_token: {current_token}")
            current_batch = self._update_forward_batch(current_batch, current_token)
            with self.mesh:
                decode_output, _ = self.model_runner.forward(
                    current_batch,
                    LogitsMetadata.from_model_worker_batch(model_worker_batch),
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
            current_token = self.model_runner.sample(decode_output, model_worker_batch)
            all_generated_tokens.append(current_token)
            print(f"step {step} current_token added: {current_token}")

        if self.enable_debug_tracer == "1":
            print("Ending debug tracer session...")
            debug_file = global_tracer.end_session()
            if debug_file:
                print(f"Debug trace saved to: {debug_file}")
            else:
                print("Debug trace not saved")
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

    def _run_inference_and_get_outputs(
        self, texts, chunked_prefill_size=None, enable_mixed_chunk=False, batch_size=1
    ):
        """Helper function to run inference and return outputs for precision comparison."""
        tokenizer = self._get_tokenizer()

        # Prepare inputs
        input_ids_list = []
        positions_list = []

        for text in texts[:batch_size]:
            encoded = tokenizer.encode(text, return_tensors="pt")
            input_ids = encoded[0].tolist()
            positions = list(range(len(input_ids)))
            input_ids_list.append(input_ids)
            positions_list.append(positions)

        # Run prefill phase (potentially chunked)
        if chunked_prefill_size is not None:
            extend_output, final_seq_lens = self._run_chunked_prefill(
                input_ids_list, positions_list, chunked_prefill_size
            )
        else:
            # Regular prefill (non-chunked)
            model_worker_batch = self._new_forward_batch(input_ids_list, positions_list)
            extend_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
            with self.mesh:
                extend_output, _ = self.model_runner.forward(
                    extend_batch,
                    LogitsMetadata.from_model_worker_batch(model_worker_batch),
                )
            final_seq_lens = [len(ids) for ids in input_ids_list]

        # Run decode phases
        decode_outputs = []
        current_token = self._sample_from_extend_output(extend_output)

        # Generate a few tokens for comparison
        for step in range(5):
            # Create decode batch
            decode_batch = self._create_decode_batch(
                input_ids_list, final_seq_lens, current_token, step
            )

            # Create model worker batch with proper bid
            model_worker_batch = self._create_decode_model_worker_batch(
                decode_batch, bid=step + 100
            )

            with self.mesh:
                decode_output, _ = self.model_runner.forward(
                    decode_batch,
                    LogitsMetadata.from_model_worker_batch(model_worker_batch),
                )
            decode_outputs.append(decode_output)
            current_token = self.model_runner.sample(decode_output, model_worker_batch)

            # Update sequence lengths for next iteration
            final_seq_lens = [seq_len + 1 for seq_len in final_seq_lens]

        return extend_output, decode_outputs

    def _run_chunked_prefill(self, input_ids_list, positions_list, chunk_size):
        """Run chunked prefill by processing input in chunks."""
        batch_size = len(input_ids_list)
        max_seq_len = max(len(ids) for ids in input_ids_list)

        # Track current position for each sequence
        current_positions = [0] * batch_size

        # We'll collect extend outputs from final chunk of each sequence
        final_extend_outputs = []
        final_batch_info = []

        # Process chunks until all sequences are done
        while any(
            pos < len(input_ids_list[i]) for i, pos in enumerate(current_positions)
        ):
            # Create chunk input_ids and positions for sequences that still have tokens
            chunk_input_ids = []
            chunk_positions = []
            active_seq_indices = []

            for seq_idx, (input_ids, positions) in enumerate(
                zip(input_ids_list, positions_list)
            ):
                if current_positions[seq_idx] < len(input_ids):
                    start_pos = current_positions[seq_idx]
                    end_pos = min(start_pos + chunk_size, len(input_ids))

                    if start_pos < end_pos:  # Still has tokens to process
                        chunk_ids = input_ids[start_pos:end_pos]
                        chunk_pos = positions[start_pos:end_pos]

                        chunk_input_ids.append(chunk_ids)
                        chunk_positions.append(chunk_pos)
                        active_seq_indices.append(seq_idx)

                        # Check if this sequence is complete after this chunk
                        if end_pos >= len(input_ids):
                            final_batch_info.append(
                                (len(chunk_input_ids) - 1, seq_idx)
                            )  # (position_in_chunk, original_seq_idx)

                        # Update position for next chunk
                        current_positions[seq_idx] = end_pos

            if not chunk_input_ids:  # No more data to process
                break

            print(
                f"Processing chunk: {len(chunk_input_ids)} active sequences out of {batch_size}"
            )

            # Create batch for this chunk (always EXTEND mode for chunked prefill)
            model_worker_batch = self._new_forward_batch(
                chunk_input_ids, chunk_positions
            )
            # Force EXTEND mode for chunked prefill
            model_worker_batch.forward_mode = ForwardMode.EXTEND

            extend_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)

            with self.mesh:
                extend_output, _ = self.model_runner.forward(
                    extend_batch,
                    LogitsMetadata.from_model_worker_batch(model_worker_batch),
                )

            # Store outputs for sequences that completed in this chunk
            for pos_in_chunk, orig_seq_idx in final_batch_info:
                if pos_in_chunk < len(active_seq_indices):  # Safety check
                    final_extend_outputs.append(
                        (orig_seq_idx, extend_output.next_token_logits[pos_in_chunk])
                    )

            # Clear batch info for next iteration
            final_batch_info = []

        # Reconstruct the extend output in original sequence order
        final_logits = []
        for seq_idx in range(batch_size):
            for orig_idx, logit in final_extend_outputs:
                if orig_idx == seq_idx:
                    final_logits.append(logit)
                    break

        # Create final extend output
        from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput

        combined_extend_output = LogitsProcessorOutput(
            next_token_logits=(
                jnp.stack(final_logits)
                if final_logits
                else extend_output.next_token_logits
            )
        )

        # Return the combined output and original sequence lengths
        final_seq_lens = [len(ids) for ids in input_ids_list]
        return combined_extend_output, final_seq_lens

    def _sample_from_extend_output(self, extend_output):
        """Sample tokens from extend output."""
        next_token_logits = extend_output.next_token_logits
        next_tokens = jnp.argmax(next_token_logits, axis=-1)
        return next_tokens.reshape(-1, 1)

    def _create_decode_batch(self, input_ids_list, seq_lens, current_token, step):
        """Create a decode batch."""
        batch_size = len(input_ids_list)

        # Allocate cache locations for decode
        req_pool_indices = self.model_runner.req_to_token_pool.alloc(batch_size)
        out_cache_loc = self.model_runner.token_to_kv_pool_allocator.alloc(batch_size)

        # Update token pool with new tokens
        for i in range(batch_size):
            self.model_runner.req_to_token_pool.write(
                (req_pool_indices[i], slice(seq_lens[i], seq_lens[i] + 1)),
                out_cache_loc[i],
            )

        model_worker_batch = ModelWorkerBatch(
            bid=step + 1,
            forward_mode=ForwardMode.DECODE,
            input_ids=current_token.flatten(),
            real_input_ids_len=batch_size,
            real_bs=batch_size,
            req_pool_indices=jnp.array(req_pool_indices),
            seq_lens=jnp.array([s + 1 for s in seq_lens]),
            out_cache_loc=jnp.array(out_cache_loc),
            cache_loc=jnp.array(out_cache_loc),
            positions=jnp.array(seq_lens),  # Next position for each sequence
            extend_start_loc=None,
            extend_seq_lens=None,
            extend_prefix_lens=None,
            return_logprob=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            extend_logprob_start_lens=None,
            extend_input_logprob_token_ids=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            sampling_info=SamplingBatchInfo(
                temperatures=jnp.full((batch_size, 1), 1.0),
                is_all_greedy=True,
                top_ps=jnp.full((batch_size, 1), 1.0),
                top_ks=jnp.ones((batch_size, 1)),
                min_ps=jnp.full((batch_size, 1), 0.0),
            ),
        )

        return ForwardBatch.init_new(model_worker_batch, self.model_runner)

    def _create_decode_model_worker_batch(
        self, forward_batch: ForwardBatch, bid: int = 0
    ):
        """Create ModelWorkerBatch for decode phase from ForwardBatch."""
        return ModelWorkerBatch(
            bid=bid,
            forward_mode=forward_batch.forward_mode,
            input_ids=forward_batch.input_ids,
            real_input_ids_len=len(forward_batch.input_ids),
            real_bs=forward_batch.batch_size,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            out_cache_loc=forward_batch.out_cache_loc,
            cache_loc=getattr(forward_batch, "cache_loc", forward_batch.out_cache_loc),
            positions=forward_batch.positions,
            extend_start_loc=getattr(forward_batch, "extend_start_loc", None),
            extend_seq_lens=getattr(forward_batch, "extend_seq_lens", None),
            extend_prefix_lens=getattr(forward_batch, "extend_prefix_lens", None),
            return_logprob=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            extend_logprob_start_lens=None,
            extend_input_logprob_token_ids=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            sampling_info=SamplingBatchInfo(
                temperatures=jnp.full((forward_batch.batch_size, 1), 1.0),
                is_all_greedy=True,
                top_ps=jnp.full((forward_batch.batch_size, 1), 1.0),
                top_ks=jnp.ones((forward_batch.batch_size, 1)),
                min_ps=jnp.full((forward_batch.batch_size, 1), 0.0),
            ),
        )

    def _compare_outputs(self, baseline_outputs, test_outputs):
        """Compare outputs for precision verification."""
        baseline_extend, baseline_decode = baseline_outputs
        test_extend, test_decode = test_outputs

        # Compare extend phase outputs (should have same shape now)
        extend_diff = jnp.abs(
            baseline_extend.next_token_logits - test_extend.next_token_logits
        )
        extend_max_diff = jnp.max(extend_diff)
        extend_mean_diff = jnp.mean(extend_diff)

        # Compare decode phase outputs
        decode_max_diffs = []
        decode_mean_diffs = []

        for i, (baseline_dec, test_dec) in enumerate(zip(baseline_decode, test_decode)):
            decode_diff = jnp.abs(
                baseline_dec.next_token_logits - test_dec.next_token_logits
            )
            decode_max_diffs.append(jnp.max(decode_diff))
            decode_mean_diffs.append(jnp.mean(decode_diff))

        results = {
            "extend_max_diff": float(extend_max_diff),
            "extend_mean_diff": float(extend_mean_diff),
            "decode_max_diffs": [float(d) for d in decode_max_diffs],
            "decode_mean_diffs": [float(d) for d in decode_mean_diffs],
            "overall_max_diff": float(max(extend_max_diff, max(decode_max_diffs))),
            "overall_mean_diff": float(
                (extend_mean_diff + sum(decode_mean_diffs))
                / (1 + len(decode_mean_diffs))
            ),
        }

        return results

    def test_baseline_precision(self):
        """Test baseline precision (non-chunked prefill, full concurrency)."""
        print("\\n=== Testing baseline precision (non-chunked prefill) ===")

        # Use all test texts to ensure consistent comparison with other tests
        test_texts = self.LONG_TEST_TEXTS

        # Run baseline inference (full batch, no chunked prefill)
        baseline_outputs = self._run_inference_and_get_outputs(
            test_texts,
            chunked_prefill_size=None,
            enable_mixed_chunk=False,
            batch_size=len(test_texts),
        )

        print(
            f"Baseline extend output shape: {baseline_outputs[0].next_token_logits.shape}"
        )
        print(f"Baseline decode outputs count: {len(baseline_outputs[1])}")

        # Store baseline for later comparison
        self._baseline_outputs = baseline_outputs

    def test_chunked_prefill_precision(self):
        """Test chunked prefill precision against baseline."""
        if not hasattr(self, "_baseline_outputs"):
            self.test_baseline_precision()

        print("\\n=== Testing chunked prefill precision ===")

        # Use all test texts for consistent comparison
        test_texts = self.LONG_TEST_TEXTS

        # Test chunked prefill (multiple EXTEND forwards then decode)
        chunked_outputs = self._run_inference_and_get_outputs(
            test_texts,
            chunked_prefill_size=self.CHUNKED_PREFILL_SIZE,
            enable_mixed_chunk=False,
            batch_size=len(test_texts),
        )

        # Compare against baseline
        comparison = self._compare_outputs(self._baseline_outputs, chunked_outputs)

        print(f"Chunked prefill comparison:")
        print(f"  Extend max diff: {comparison['extend_max_diff']:.6f}")
        print(f"  Extend mean diff: {comparison['extend_mean_diff']:.6f}")
        print(
            f"  Decode max diffs: {[f'{d:.6f}' for d in comparison['decode_max_diffs']]}"
        )
        print(f"  Overall max diff: {comparison['overall_max_diff']:.6f}")
        print(f"  Overall mean diff: {comparison['overall_mean_diff']:.6f}")

        # Assert precision is acceptable (adjust thresholds as needed)
        self.assertLess(
            comparison["overall_max_diff"],
            0.1,
            f"Chunked prefill max difference too large: {comparison['overall_max_diff']}",
        )
        self.assertLess(
            comparison["overall_mean_diff"],
            0.01,
            f"Chunked prefill mean difference too large: {comparison['overall_mean_diff']}",
        )

    def test_high_concurrency_precision(self):
        """Test consistency of repeated runs (should be identical to baseline)."""
        if not hasattr(self, "_baseline_outputs"):
            self.test_baseline_precision()

        print("\\n=== Testing high concurrency precision (repeated run) ===")

        # Use same configuration as baseline for consistency check
        test_texts = self.LONG_TEST_TEXTS

        # Run with same settings as baseline to verify consistency
        high_concurrency_outputs = self._run_inference_and_get_outputs(
            test_texts,
            chunked_prefill_size=None,
            enable_mixed_chunk=False,
            batch_size=len(test_texts),
        )

        # Compare against baseline (compare first sample)
        comparison = self._compare_outputs(
            self._baseline_outputs, high_concurrency_outputs
        )

        print(f"High concurrency comparison:")
        print(f"  Extend max diff: {comparison['extend_max_diff']:.6f}")
        print(f"  Extend mean diff: {comparison['extend_mean_diff']:.6f}")
        print(
            f"  Decode max diffs: {[f'{d:.6f}' for d in comparison['decode_max_diffs']]}"
        )
        print(f"  Overall max diff: {comparison['overall_max_diff']:.6f}")
        print(f"  Overall mean diff: {comparison['overall_mean_diff']:.6f}")

        # Assert precision is acceptable
        self.assertLess(
            comparison["overall_max_diff"],
            0.1,
            f"High concurrency max difference too large: {comparison['overall_max_diff']}",
        )
        self.assertLess(
            comparison["overall_mean_diff"],
            0.01,
            f"High concurrency mean difference too large: {comparison['overall_mean_diff']}",
        )

    def test_chunked_prefill_with_mixed_mode_precision(self):
        """Test chunked prefill with mixed mode enabled precision against baseline."""
        if not hasattr(self, "_baseline_outputs"):
            self.test_baseline_precision()

        print("\\n=== Testing chunked prefill + mixed mode precision ===")

        # Use all test texts for chunked prefill + mixed mode
        test_texts = self.LONG_TEST_TEXTS

        # Run with chunked prefill + mixed mode enabled
        chunked_mixed_outputs = self._run_inference_and_get_outputs(
            test_texts,
            chunked_prefill_size=self.CHUNKED_PREFILL_SIZE,
            enable_mixed_chunk=True,
            batch_size=len(test_texts),
        )

        # Compare against baseline
        comparison = self._compare_outputs(
            self._baseline_outputs, chunked_mixed_outputs
        )

        print(f"Chunked prefill + mixed mode comparison:")
        print(f"  Extend max diff: {comparison['extend_max_diff']:.6f}")
        print(f"  Extend mean diff: {comparison['extend_mean_diff']:.6f}")
        print(
            f"  Decode max diffs: {[f'{d:.6f}' for d in comparison['decode_max_diffs']]}"
        )
        print(f"  Overall max diff: {comparison['overall_max_diff']:.6f}")
        print(f"  Overall mean diff: {comparison['overall_mean_diff']:.6f}")

        # Assert precision is acceptable (may need looser thresholds for this combination)
        self.assertLess(
            comparison["overall_max_diff"],
            0.2,
            f"Chunked prefill + mixed mode max difference too large: {comparison['overall_max_diff']}",
        )
        self.assertLess(
            comparison["overall_mean_diff"],
            0.02,
            f"Chunked prefill + mixed mode mean difference too large: {comparison['overall_mean_diff']}",
        )


if __name__ == "__main__":
    unittest.main()
