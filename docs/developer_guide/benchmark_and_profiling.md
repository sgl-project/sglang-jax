# Benchmark and Profiling

## Benchmark

- Benchmark the latency of running a single static batch without a server. The arguments are the same as for `launch_server.py`.
  Note that this is a simplified test script without a dynamic batching server, so it may run out of memory for a batch size that a real server can handle. A real server truncates the prefill into several batches, while this simplified script does not.
  - With a server (please use `sgl_jax.launch_server` to launch a server first and run the following command.)
    ```bash
    python -m sgl_jax.bench_one_batch_server --base-url http://127.0.0.1:30000 --model-path Qwen/Qwen-7B-Chat --batch-size 32 --input-len 256 --output-len 32
    ```

- Benchmark online serving. Please use `sgl_jax.launch_server` to launch a server first and run the following command.

  ```bash
  python3 -m sgl_jax.bench_serving --backend sgl-jax --num-prompt 10
  ```
- A recommended benchmark before merging into main. Please save the following codes as script.sh and execute `./script.sh sglang-oai`.
  ```bash
  #!/bin/bash

  set -e

  if [ -z "$1" ]; then
      echo "Usage: $0 <engine>"
      echo "engine: sgl-jax, sglang-oai or vllm"
      exit 1
  fi

  backend=${1}
  num_prompts_per_concurrency=3
  input_seq_lens=(1024 4096 8192)
  output_seq_lens=(1 1024) # 1 for TTFT
  max_concurrencies=(8 16 32 64 128 256)

  for input_seq_len in "${input_seq_lens[@]}"; do
      for output_seq_len in "${output_seq_lens[@]}"; do
          echo "======================================="
          echo "Testing with ISL/OSL: $input_seq_len/$output_seq_len"
          echo "======================================="
          for max_concurrency in "${max_concurrencies[@]}"; do
              echo "benchmark on max_concurrency=$max_concurrency"
              num_prompts=$((num_prompts_per_concurrency * max_concurrency))
              uv run python -m sgl_jax.bench_serving \
                --backend ${backend} \
                --dataset-name random \
                --num-prompts ${num_prompts} \
                --random-input-len ${input_seq_len} \
                --random-output-len ${output_seq_len} \
                --max-concurrency ${max_concurrency} \
                --random-range-ratio 1 \
                --disable-ignore-eos \
                --warmup-requests 0
          done
      done
  done
  ```

## Profile with Jax Profiler

[Jax Profiler](https://docs.jax.dev/en/latest/profiling.html) is a convenient basic tool to inspect kernel execution time, call stack, and kernel overlap and occupancy.

### Profile a server with `sgl_jax.bench_serving`
```bash

# start server
python3 -m sgl_jax.launch_server --model-path Qwen/Qwen-7B-Chat --trust-remote-code  --dist-init-addr=0.0.0.0:10011 --nnodes=1  --tp-size=4 --device=tpu --random-seed=3 --node-rank=0 --mem-fraction-static=0.8 --max-prefill-tokens=8192 --download-dir=/tmp --dtype=bfloat16  --skip-server-warmup

# send request to start profile
curl -X POST 'http://127.0.0.1:30000/start_profile' -d '{"output_dir": "/home/profile", "num_steps": 5}' -H 'Content-Type: application/json'

# send profiling request from client
python3 -m sgl_jax.bench_serving --backend sgl-jax --dataset-name random --num-prompts 10 --random-input 512 --random-output 10 --random-range-ratio 1 --warmup-requests 0
```

Please make sure that the `output_dir` should be set at both server side, otherwise the trace file cannot be generated correctly.

### View traces

Trace files can be loaded and visualized from:

1. https://ui.perfetto.dev/ (any browser)
2. chrome://tracing (Chrome browser only)

If browser cannot open trace file due to its large size,
client can generate a small trace file (<100MB) by controlling number of prompts and lengths of prompt outputs.

### View traces with Tensorboard

View the profile file with tensorboard
```bash
tensorboard --logdir={profile_dir}
# click url to open the link
```

### View traces with XProf

[XProf](https://github.com/openxla/xprof) includes a suite of tools for JAX, TensorFlow, and PyTorch/XLA. These tools help you understand, debug and optimize programs to run on CPUs, GPUs and TPUs.

```bash
# To install the nightly version of profiler:
pip install xprof-nightly

# Without TensorBoard:
xprof --logdir=profiler/demo --port=6006

# With TensorBoard:
tensorboard --logdir=profiler/demo

```
