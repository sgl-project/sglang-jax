"""
Benchmark script to compare original vs optimized FlowUniPCMultistepScheduler.

Issue #845: The FlowUniPCMultistepScheduler involves many discontinuous operations in wan2.1

This benchmark demonstrates the performance improvement of the optimized version.
"""

import time
import jax
import jax.numpy as jnp
from functools import partial

# Import both versions
from flow_unipc_multistep_scheduler import FlowUniPCMultistepScheduler
from flow_unipc_multistep_scheduler_optimized import FlowUniPCMultistepSchedulerOptimized


def benchmark_original_scheduler(num_steps=20, batch_size=1, latent_shape=(4, 16, 16)):
    """Benchmark the original scheduler with step-by-step execution."""
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        solver_order=2,
        shift=1.0,
    )
    
    # Setup
    scheduler.set_timesteps(num_steps, latent_shape)
    sample = jax.random.normal(jax.random.PRNGKey(0), latent_shape)
    
    # Warmup
    for i in range(min(3, num_steps)):
        model_output = jax.random.normal(jax.random.PRNGKey(i), latent_shape)
        result = scheduler.step(model_output, scheduler.timesteps[i], sample)
        sample = result.prev_sample
    
    # Benchmark
    start_time = time.time()
    sample = jax.random.normal(jax.random.PRNGKey(0), latent_shape)
    
    for i in range(num_steps):
        model_output = jax.random.normal(jax.random.PRNGKey(i), latent_shape)
        result = scheduler.step(model_output, scheduler.timesteps[i], sample)
        sample = result.prev_sample
    
    # Ensure completion
    jax.block_until_ready(sample)
    elapsed = time.time() - start_time
    
    return elapsed


def benchmark_optimized_scan(num_steps=20, batch_size=1, latent_shape=(4, 16, 16)):
    """Benchmark the optimized scheduler with scan execution."""
    scheduler = FlowUniPCMultistepSchedulerOptimized(
        num_train_timesteps=1000,
        solver_order=2,
        shift=1.0,
    )
    
    # Setup
    scheduler.set_timesteps(num_steps, latent_shape)
    initial_sample = jax.random.normal(jax.random.PRNGKey(0), latent_shape)
    
    # Pre-generate all model outputs (simulating model forward passes)
    model_outputs_list = [
        jax.random.normal(jax.random.PRNGKey(i), latent_shape)
        for i in range(num_steps)
    ]
    
    # Warmup
    _ = scheduler.scan_steps(model_outputs_list[:3], initial_sample)
    jax.block_until_ready(_)
    
    # Benchmark
    start_time = time.time()
    result = scheduler.scan_steps(model_outputs_list, initial_sample)
    jax.block_until_ready(result)
    elapsed = time.time() - start_time
    
    return elapsed


def benchmark_optimized_step(num_steps=20, batch_size=1, latent_shape=(4, 16, 16)):
    """Benchmark the optimized scheduler with step-by-step execution."""
    scheduler = FlowUniPCMultistepSchedulerOptimized(
        num_train_timesteps=1000,
        solver_order=2,
        shift=1.0,
    )
    
    # Setup
    scheduler.set_timesteps(num_steps, latent_shape)
    sample = jax.random.normal(jax.random.PRNGKey(0), latent_shape)
    
    # Warmup
    for i in range(min(3, num_steps)):
        model_output = jax.random.normal(jax.random.PRNGKey(i), latent_shape)
        result = scheduler.step(model_output, scheduler.timesteps[i], sample)
        sample = result.prev_sample
    
    # Benchmark
    start_time = time.time()
    sample = jax.random.normal(jax.random.PRNGKey(0), latent_shape)
    
    for i in range(num_steps):
        model_output = jax.random.normal(jax.random.PRNGKey(i), latent_shape)
        result = scheduler.step(model_output, scheduler.timesteps[i], sample)
        sample = result.prev_sample
    
    # Ensure completion
    jax.block_until_ready(sample)
    elapsed = time.time() - start_time
    
    return elapsed


def main():
    print("=" * 70)
    print("FlowUniPCMultistepScheduler Performance Benchmark")
    print("Issue #845: Optimizing discontinuous operations in wan2.1")
    print("=" * 70)
    
    configs = [
        ("Small latent", (4, 16, 16)),
        ("Medium latent", (4, 32, 32)),
        ("Large latent", (4, 64, 64)),
    ]
    
    num_steps = 20
    
    for name, latent_shape in configs:
        print(f"\n{name}: {latent_shape}, Steps: {num_steps}")
        print("-" * 70)
        
        # Original
        try:
            t_original = benchmark_original_scheduler(num_steps, latent_shape=latent_shape)
            print(f"  Original (step-by-step):  {t_original:.4f}s")
        except Exception as e:
            print(f"  Original (step-by-step):  ERROR - {e}")
            t_original = None
        
        # Optimized step
        try:
            t_opt_step = benchmark_optimized_step(num_steps, latent_shape=latent_shape)
            print(f"  Optimized (step-by-step): {t_opt_step:.4f}s")
        except Exception as e:
            print(f"  Optimized (step-by-step): ERROR - {e}")
            t_opt_step = None
        
        # Optimized scan
        try:
            t_opt_scan = benchmark_optimized_scan(num_steps, latent_shape=latent_shape)
            print(f"  Optimized (scan):         {t_opt_scan:.4f}s")
        except Exception as e:
            print(f"  Optimized (scan):         ERROR - {e}")
            t_opt_scan = None
        
        # Speedup
        if t_original and t_opt_scan:
            speedup = t_original / t_opt_scan
            print(f"\n  Speedup (scan vs original): {speedup:.2f}x")
        
        if t_original and t_opt_step:
            speedup_step = t_original / t_opt_step
            print(f"  Speedup (step vs original): {speedup_step:.2f}x")
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("  - Original: Many host-device synchronizations per step")
    print("  - Optimized (step): Reduced overhead, same API")
    print("  - Optimized (scan): Single JIT compilation, fully continuous")
    print("=" * 70)


if __name__ == "__main__":
    main()
