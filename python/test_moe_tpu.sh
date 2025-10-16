#\!/bin/bash
# Run MoE TopK tests on TPU

echo "JAX Configuration:"
python -c "import jax; print(f'Backend: {jax.default_backend()}'); print(f'Devices: {jax.devices()}')"

echo -e "\nRunning tests..."
PYTHONPATH=. python sgl_jax/test/test_moe_topk.py
