import subprocess
import os
import signal
import time
import sys

cmd = [
    sys.executable, "-u", "-m", "sgl_jax.launch_server",
    "--model-path", "/Users/jiongxuan/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B/",
    "--device", "cpu",
    "--trust-remote-code",
    "--port", "30000",
    "--host", "127.0.0.1",
    "--tp-size", "1",
    "--mem-fraction-static", "0.25",
    "--max-prefill-tokens", "512",
    "--disable-radix-cache",
    "--skip-server-warmup",
    "--disable-precompile",
    "--log-level", "debug"
]

env = os.environ.copy()
env["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jit_cache"
env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

print("Running command:", " ".join(cmd))

try:
    # Use Popen to capture output in real-time or just let it print to stdout
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1
    )
    
    start_time = time.time()
    while time.time() - start_time < 300:
        line = process.stdout.readline()
        if line:
            print(line, end="")
        if process.poll() is not None:
            break
        
    if process.poll() is None:
        print("\nTerminating process...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

except Exception as e:
    print(f"Error: {e}")
