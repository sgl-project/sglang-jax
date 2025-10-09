rsync -rvzz --checksum --delete --exclude "venv" --exclude ".git" --exclude "*.egg-info" --exclude "old_weights" --exclude "*.pyc" --exclude ".idea" --exclude ".ruff_cache" --exclude "weights/*.safetensors" --exclude "__pycache__" ./ sky-495b-jcyang:sky_workdir/sglang-jax/

