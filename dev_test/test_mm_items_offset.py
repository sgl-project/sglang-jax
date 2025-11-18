import random
import argparse
import jax
import jax.numpy as jnp

from sgl_jax.srt.multimodal.processors.base_processor import BaseMultimodalProcessor


def run_fixed_example():
    input_ids = [1, 2, 3, 3, 3, 4, 3, 3]
    token = 3
    arr = jnp.array(input_ids, dtype=jnp.int32)

    ranges = BaseMultimodalProcessor.get_mm_items_offset(arr, token)
    print("[JAX][fixed] get_mm_items_offset:")
    print(" input=", input_ids)
    print(" token=", token)
    print(" ranges=", ranges)

    # Pair version fixed example
    S, E = 101, 102
    input_ids2 = [1, 2, S, 11, 12, 13, E, 4, 5, S, 21, 22, E, 6]
    arr2 = jnp.array(input_ids2, dtype=jnp.int32)
    ranges2 = BaseMultimodalProcessor.get_mm_items_offset_by_pair(arr2, S, E)
    print("[JAX][fixed] get_mm_items_offset_by_pair:")
    print(" input=", input_ids2)
    print(" start=", S, " end=", E)
    print(" ranges=", ranges2)


def run_random_cases(num_cases: int, max_len: int, vocab_size: int, seed: int):
    rng = random.Random(seed)

    print("[JAX] Random cases for get_mm_items_offset:")
    for i in range(num_cases):
        length = rng.randint(0, max_len)
        data = [rng.randint(0, vocab_size - 1) for _ in range(length)]
        token = 0 if length == 0 else rng.choice(data)
        arr = jnp.array(data, dtype=jnp.int32)
        ranges = BaseMultimodalProcessor.get_mm_items_offset(arr, token)
        print(f" case#{i}: token={token}, input={data}, ranges={ranges}")

    print("\n[JAX] Random cases for get_mm_items_offset_by_pair:")
    for i in range(num_cases):
        length = rng.randint(0, max_len)
        data = [rng.randint(0, vocab_size - 1) for _ in range(length)]
        # choose two (possibly distinct) tokens in vocab
        start_id = rng.randint(0, vocab_size - 1)
        end_id = rng.randint(0, vocab_size - 1)
        if end_id == start_id:
            end_id = (end_id + 1) % vocab_size
        arr = jnp.array(data, dtype=jnp.int32)
        ranges = BaseMultimodalProcessor.get_mm_items_offset_by_pair(arr, start_id, end_id)
        print(f" case#{i}: start={start_id}, end={end_id}, input={data}, ranges={ranges}")


def main():
    parser = argparse.ArgumentParser(description="Compare mm_items_offset in JAX (TPU)")
    parser.add_argument("--num-cases", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Print JAX device info (to confirm TPU/XLA environment)
    print("JAX devices:", jax.devices())

    run_fixed_example()
    run_random_cases(args.num_cases, args.max_len, args.vocab_size, args.seed)


if __name__ == "__main__":
    main()
