import json
import os
import time
import unittest

import numpy as np

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.test.test_utils import CustomTestCase

LORA_SETS = [
    {
        "base": "Qwen/Qwen3-4B",
        "lora": [
            "y9760210/Qwen3-4B-lora_model",
        ],
    },
]

LORA_PATHS = ["Qwen3-4B-lora_model", None, "Qwen3-4B-lora_model"]
BASE_MODEL = "Qwen/Qwen3-4B"
DTYPE = "float32"
NPDTYPE = np.float32

VOCAB_SIZE = 151936

PROMPTS = [
    "AI is a field of computer science focused on",
    "Computer science is the study of",
    "Write a short story.",
]
THRESHOLD = 2e-3


class TestAlignLoRAAccuracy(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = "Qwen/Qwen3-4B"
        cls.lora_target_modules = ["all"]
        import os

        os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jit_cache"
        cls.single_prompt_prefill_logits_cpu_dump_filename = (
            f"./lora_data/sglangjax/single_prompt_prefill_logits_cpu.txt"
        )
        cls.single_prompt_decode_logits_cpu_dump_filename = (
            f"./lora_data/sglangjax/single_prompt_decode_logits_cpu.txt"
        )
        cls.multi_prompts_prefill_logits_tpu_dump_filename = (
            f"./lora_data/sglangjax/multi_prompts_prefill_logits_tpu.txt"
        )
        cls.multi_prompts_decode_logits_tpu_dump_filename = (
            f"./lora_data/sglangjax/multi_prompts_decode_logits_tpu.txt"
        )

    def get_sglang_jax_last_layer_logits_hidden_states(
        self,
        prompts,
        lora_paths,
        return_logits=True,
        prefill_return_logits_dump_filename=None,
        decode_return_logits_dump_filename=None,
        max_new_tokens=2,  # one forward for prefill, one forward for decode
        device="cpu",
        disable_overlap_schedule=False,
        enable_single_process=False,
        mem_fraction_static=0.2,
        model_layer_nums=None,
        return_hidden_states=False,
    ):
        if return_logits:
            os.environ["DUMP_LAST_LAYER_LOGITS_FILENAMES"] = (
                f"{prefill_return_logits_dump_filename},{decode_return_logits_dump_filename}"
            )
        if device == "cpu":
            os.environ["JAX_PLATFORMS"] = "cpu"
        else:
            del os.environ["JAX_PLATFORMS"]
        engine = Engine(
            model_path=self.model_path,
            trust_remote_code=True,
            tp_size=1,
            device=device,
            random_seed=3,
            node_rank=0,
            mem_fraction_static=mem_fraction_static,
            chunked_prefill_size=64,
            download_dir="/tmp",
            dtype=DTYPE,
            precompile_bs_paddings=[4],
            max_running_requests=4,
            skip_server_warmup=True,
            attention_backend="fa",
            precompile_token_paddings=[64],
            page_size=16,
            log_requests=False,
            enable_deterministic_sampling=True,
            lora_paths=LORA_SETS[0]["lora"],
            max_loras_per_batch=3,
            lora_target_modules=self.lora_target_modules,
            watchdog_timeout=3000,
            disable_overlap_schedule=disable_overlap_schedule,
            enable_single_process=enable_single_process,
            model_layer_nums=model_layer_nums,
        )
        sampling_params = {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
        }

        outputs = engine.generate(
            prompt=prompts,
            sampling_params=sampling_params,
            lora_path=lora_paths,
        )

        self.assertEqual(len(outputs), len(prompts))

        def get_content_from_files(
            prefill_dump_filename=None,
            decode_dump_filename=None,
        ):
            if not os.path.exists(prefill_dump_filename) or not os.path.exists(
                decode_dump_filename
            ):
                raise ValueError(
                    f"{prefill_dump_filename} or {decode_dump_filename} does not exist!"
                )

            if prefill_dump_filename:
                with open(prefill_dump_filename, "r") as prefill_f:
                    prefill_data = np.loadtxt(prefill_f)
                    if len(prompts) == 1:
                        prefill_reshaped_data = np.array(prefill_data, dtype=NPDTYPE).reshape(1, -1)
                        # Note: Due to accumulating batch in SGLangJax, multi prompts may be divided into several batches, so the
                        # number of items in the prefill file is difficult to calculate. Merging files does not take effect,
                        # because the order of result may not keep the same with order in prompts. So for multi prompts, we only return decode_logits.
                    else:
                        prefill_reshaped_data = None
            else:
                prefill_reshaped_data = None

            if decode_dump_filename:
                while True:
                    # Ensure the number of items in decode file to match the vocab_size * len(prompts)
                    with open(decode_dump_filename, "r") as decode_f:
                        decode_data = np.loadtxt(decode_f)
                        decode_reshaped_data = np.array(decode_data, dtype=NPDTYPE).reshape(
                            len(prompts), -1
                        )
                        if decode_reshaped_data.size == VOCAB_SIZE * len(prompts):
                            break
                        else:
                            time.sleep(5)
            else:
                decode_reshaped_data = None

            return prefill_reshaped_data, decode_reshaped_data

        if return_logits:
            prefill_logits, decode_logits = get_content_from_files(
                prefill_return_logits_dump_filename,
                decode_return_logits_dump_filename,
            )
        else:
            prefill_logits, decode_logits = None, None

        if return_hidden_states:
            raise NotImplemented

        engine.shutdown()

        return prefill_logits, decode_logits, None, outputs

    @classmethod
    def tearDownClass(cls):
        pass

    def test_01_single_prompt_align_last_layer_logits_with_hf_on_cpu(self):
        # command to generate hf results
        hf_generate_commands = """
cd test/srt/lora

# For prefill
python3 dump_hf_lora_output.py --model Qwen/Qwen3-4B \
--lora-path y9760210/Qwen3-4B-lora_model \
--prompt "AI is a field of computer science focused on" \
--max-new-tokens 1 \
--output lora_data/hf/single_prompt_prefill_output_cpu.json \
--use-cpu \
--dtype float32

# For decode
python3 dump_hf_lora_output.py --model Qwen/Qwen3-4B \
--lora-path y9760210/Qwen3-4B-lora_model \
--prompt "AI is a field of computer science focused on" \
--max-new-tokens 2 \
--output lora_data/hf/single_prompt_decode_output_cpu.json \
--use-cpu \
--dtype float32
"""
        hf_lora_prefill_logits_file = os.path.join(
            os.path.dirname(__file__), "lora_data/hf/single_prompt_prefill_output_cpu.json"
        )
        hf_lora_decode_logits_file = os.path.join(
            os.path.dirname(__file__), "lora_data/hf/single_prompt_decode_output_cpu.json"
        )

        try:
            assert os.path.exists(hf_lora_prefill_logits_file) and os.path.exists(
                hf_lora_decode_logits_file
            )
        except BaseException as e:
            print(
                f"{hf_lora_prefill_logits_file} or {hf_lora_decode_logits_file} does not exist, please generate them firstly with following commands: \n{hf_generate_commands}",
                flush=True,
            )
            raise

        """Compare logprobs from sglang-jax with HuggingFace reference."""
        print("=================== testing single prompt logits comparison ======================")

        try:
            with (
                open(hf_lora_prefill_logits_file, "r") as prefill_f,
                open(hf_lora_decode_logits_file, "r") as decode_f,
            ):
                hf_prefill_data = json.load(prefill_f)
                hf_decode_data = json.load(decode_f)
                hf_prefill_logits = np.array(
                    hf_prefill_data["results"][0]["last_token_logits"], dtype=NPDTYPE
                ).reshape(1, -1)
                hf_decode_logits = np.array(
                    hf_decode_data["results"][0]["last_token_logits"], dtype=NPDTYPE
                ).reshape(1, -1)
        except Exception as e:
            raise ValueError(
                f"Fail to load {hf_lora_prefill_logits_file} and {hf_lora_decode_logits_file} and meet err: {e}"
            )

        sgl_prefill_logits, sgl_decode_logits, _, _ = (
            self.get_sglang_jax_last_layer_logits_hidden_states(
                [PROMPTS[0]],
                LORA_PATHS[0],
                prefill_return_logits_dump_filename=self.single_prompt_prefill_logits_cpu_dump_filename,
                decode_return_logits_dump_filename=self.single_prompt_decode_logits_cpu_dump_filename,
                device="cpu",
                disable_overlap_schedule=True,
                mem_fraction_static=0.2,
            )
        )

        prefill_diff = np.abs(hf_prefill_logits - sgl_prefill_logits)
        prefill_max_diff = np.max(prefill_diff)
        prefill_mean_diff = np.mean(prefill_diff)

        decode_diff = np.abs(hf_decode_logits - sgl_decode_logits)
        decode_max_diff = np.max(decode_diff)
        decode_mean_diff = np.mean(decode_diff)

        print(f"\n  Prefill  Max diff:   {prefill_max_diff:.6e}", flush=True)
        print(f"    Prefill Mean diff:  {prefill_mean_diff:.6e}", flush=True)

        print(f"\n  Decode  Max diff:   {decode_max_diff:.6e}", flush=True)
        print(f"    Decode Mean diff:  {decode_mean_diff:.6e}", flush=True)

        tolerance = 1e-4
        np.testing.assert_allclose(
            hf_prefill_logits, sgl_prefill_logits, atol=tolerance, rtol=tolerance
        )
        np.testing.assert_allclose(
            hf_decode_logits, sgl_decode_logits, atol=tolerance, rtol=tolerance
        )

        print("\n✓ Single prompt logits comparison completed")

    def test_02_multi_prompts_align_last_layer_logits_with_hf_on_tpu(self):
        hf_generate_commands = """
cd test/srt/lora/

# For decode
python3 dump_hf_lora_output.py --model Qwen/Qwen3-4B \
--lora-path y9760210/Qwen3-4B-lora_model "" y9760210/Qwen3-4B-lora_model \
--prompt "AI is a field of computer science focused on" "Computer science is the study of" "Write a short story." \
--max-new-tokens 2 \
--output lora_data/hf/multi_prompts_decode_output_cpu.json \
--use-cpu \
--dtype float32
"""

        hf_lora_decode_logits_file = os.path.join(
            os.path.dirname(__file__), "lora_data/hf/multi_prompts_decode_output_cpu.json"
        )

        try:
            assert os.path.exists(hf_lora_decode_logits_file)
        except BaseException as e:
            print(
                f"{hf_lora_decode_logits_file} does not exist, please generate it firstly with following commands: \n{hf_generate_commands}",
                flush=True,
            )
            raise

        """Compare logprobs from sglang-jax with HuggingFace reference."""
        print("=================== testing multi prompts logits comparison ======================")

        try:
            with open(hf_lora_decode_logits_file, "r") as decode_f:
                hf_decode_data = json.load(decode_f)
                hf_decode_logits_list = []
                for i in range(len(PROMPTS)):
                    hf_decode_logits_list.append(hf_decode_data["results"][i]["last_token_logits"])
                hf_decode_logits_multi_prompts = np.array(
                    hf_decode_logits_list, dtype=NPDTYPE
                ).reshape(len(PROMPTS), -1)
        except Exception as e:
            raise ValueError(f"Fail to {hf_lora_decode_logits_file} and meet err: {e}")

        _, sgl_decode_logits_multi_prompts, _, _ = (
            self.get_sglang_jax_last_layer_logits_hidden_states(
                PROMPTS,
                LORA_PATHS,
                prefill_return_logits_dump_filename=self.multi_prompts_prefill_logits_tpu_dump_filename,
                decode_return_logits_dump_filename=self.multi_prompts_decode_logits_tpu_dump_filename,
                device="tpu",
                disable_overlap_schedule=True,
                mem_fraction_static=0.6,
            )
        )

        # Calculate differences (similar to original sglang)
        for i in range(len(PROMPTS)):
            hf_decode_logits = hf_decode_logits_multi_prompts[i, :]
            sgl_decode_logits = sgl_decode_logits_multi_prompts[i, :]
            print(f"{hf_decode_logits.shape=}, {hf_decode_logits[...,:20]}", flush=True)
            print(f"{sgl_decode_logits.shape=}, {sgl_decode_logits[...,:20]}", flush=True)

            decode_diff = np.abs(hf_decode_logits - sgl_decode_logits)
            decode_max_diff = np.max(decode_diff)
            decode_mean_diff = np.mean(decode_diff)

            print(f"\n  Decode  Max diff:   {decode_max_diff:.6e}", flush=True)
            print(f"    Decode Mean diff:  {decode_mean_diff:.6e}", flush=True)

            tolerance = 9e-2
            np.testing.assert_allclose(
                hf_decode_logits, sgl_decode_logits, atol=tolerance, rtol=tolerance
            )

        print("\n✓ Multi prompts logits comparison completed")

    def test_03_multi_prompts_align_last_layer_logits_with_hf_on_tpu_1_layer(self):
        # Note: The return_hidden_states has to be supported before this test.
        return

        hf_generate_commands = """
cd test/srt/lora/

# For decode
python3 dump_hf_lora_output.py --model Qwen/Qwen3-4B \
--lora-path y9760210/Qwen3-4B-lora_model "" y9760210/Qwen3-4B-lora_model \
--prompt "AI is a field of computer science focused on" "Computer science is the study of" "Write a short story." \
--max-new-tokens 2 \
--output lora_data/hf/multi_prompts_decode_output_cpu_1_layer.json \
--use-cpu \
--dtype float32 \
--num-layers 1
"""

        hf_lora_decode_logits_file = os.path.join(
            os.path.dirname(__file__), "lora_data/hf/multi_prompts_decode_output_cpu_1_layer.json"
        )

        print(f"==========={hf_lora_decode_logits_file}")

        try:
            assert os.path.exists(hf_lora_decode_logits_file)
        except BaseException as e:
            print(
                f"{hf_lora_decode_logits_file} does not exist, please generate it firstly with following commands: \n{hf_generate_commands}",
                flush=True,
            )
            raise

        """Compare logprobs from sglang-jax with HuggingFace reference."""
        print(
            "=================== testing multi prompts hidden states comparison ======================"
        )

        try:
            with open(hf_lora_decode_logits_file, "r") as decode_f:
                hf_decode_data = json.load(decode_f)
                hf_decode_hidden_states_list = []
                for i in range(len(PROMPTS)):
                    hf_decode_hidden_states_list.append(
                        hf_decode_data["results"][i]["last_layer_hidden_states"][-1]
                    )
                hf_decode_hidden_states_multi_prompts = np.array(
                    hf_decode_hidden_states_list, dtype=NPDTYPE
                ).reshape(len(PROMPTS), -1)
        except Exception as e:
            raise ValueError(f"Fail to {hf_lora_decode_logits_file} and meet err: {e}")

        _, _, _, sgl_decode_hidden_states_multi_prompts = (
            self.get_sglang_jax_last_layer_logits_hidden_states(
                PROMPTS,
                LORA_PATHS,
                return_logits=False,
                device="tpu",
                disable_overlap_schedule=True,
                mem_fraction_static=0.6,
                model_layer_nums=1,
                return_hidden_states=True,
            )
        )

        # Calculate differences (similar to original sglang)
        for i in range(len(PROMPTS)):
            # hf_prefill_logits = hf_prefill_logits[i,:]
            hf_decode_hidden_states = hf_decode_hidden_states_multi_prompts[i, :]
            # sgl_prefill_logits = sgl_prefill_logits[i,:]
            sgl_decode_hidden_states = sgl_decode_hidden_states_multi_prompts[i, :]
            # print(f"{hf_prefill_logits.shape=}, {hf_prefill_logits[...,:20]}", flush=True)
            print(
                f"{hf_decode_hidden_states.shape=}, {hf_decode_hidden_states[...,:20]}", flush=True
            )
            # print(f"{sgl_prefill_logits.shape=}, {sgl_prefill_logits[...,:20]}",flush=True)
            print(
                f"{sgl_decode_hidden_states.shape=}, {sgl_decode_hidden_states[...,:20]}",
                flush=True,
            )

            # prefill_diff = np.abs(hf_prefill_logits - sgl_prefill_logits)
            # prefill_max_diff = np.max(prefill_diff)
            # prefill_mean_diff = np.mean(prefill_diff)

            decode_diff = np.abs(hf_decode_hidden_states - sgl_decode_hidden_states)
            decode_max_diff = np.max(decode_diff)
            decode_mean_diff = np.mean(decode_diff)

            # print(f"\n  Prefill  Max diff:   {prefill_max_diff:.6e}",flush=True)
            # print(f"    Prefill Mean diff:  {prefill_mean_diff:.6e}",flush=True)

            print(f"\n  Decode  Max diff:   {decode_max_diff:.6e}", flush=True)
            print(f"    Decode Mean diff:  {decode_mean_diff:.6e}", flush=True)

            tolerance = 9e-2
            np.testing.assert_allclose(
                hf_decode_hidden_states, sgl_decode_hidden_states, atol=tolerance, rtol=tolerance
            )

        print("\n✓ Multi prompts logits comparison completed")


if __name__ == "__main__":
    unittest.main()
