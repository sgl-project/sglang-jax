### Developer Guide: Why We Chose JAX and How to Learn It

Welcome to the `sglang-jax` project! This document is designed to help new developers, especially those unfamiliar with JAX, understand the core reasons behind our technical choices and provide a clear learning path.

#### Why We Chose JAX

Among the many deep learning frameworks available (like PyTorch and TensorFlow), we choose JAX as our core backend for three key advantages that are crucial for building a high-performance inference system for large language models.

**1. Native Distributed Semantics**

- **What it is:** Imagine a massive language model as a giant machine that requires hundreds of people to operate. You need to tell each person (a compute device like a GPU/TPU) exactly which part they are responsible for (a piece of the model's weights) and how they should coordinate. JAX's distributed features (`jax.sharding`) allow us to describe this complex division of labor and collaboration using remarkably clean and clear Python code.
- **Why it matters:** In other frameworks, distributed logic is often implemented as external libraries or complex abstractions, making the code more cumbersome. JAX, however, integrates the concept of "how to split computations across devices" into its core design. We can easily define a device "Mesh" and then, with simple annotations, tell JAX to "split this tensor along this dimension and distribute it to that group of devices." This level of clarity and expressiveness is invaluable when dealing with large models that require dozens or hundreds of accelerator cards. It simplifies writing and maintaining complex parallelism strategies (like tensor parallelism and expert parallelism).
- **In short:** JAX allows us to elegantly conduct an "orchestra" of hundreds or thousands of devices to perform the symphony of a large model.

**2. Better Compiler Support**

- **What it is:** Behind JAX is Google's XLA (Accelerated Linear Algebra) compiler. When you decorate your Python function with `jax.jit` (Just-In-Time), JAX doesn't just execute the Python code. Instead, it transforms your function into a computation graph and hands it over to XLA. XLA performs deep optimizations on this graph (like operator fusion, which merges multiple small computations into a single larger one) and compiles it into highly optimized machine code tailored for your specific hardware (GPU/TPU).
- **Why it matters:** This is like writing a recipe in plain language (Python code), while XLA is a Michelin-starred chef who can interpret it. The chef will rearrange and optimize all the steps, using the most efficient culinary techniques (machine code) to produce a final dish much faster and better than a regular cook (the standard Python interpreter). This deep compilation optimization is the key to JAX's superior performance. For inference tasks where every millisecond of latency counts, the performance boost from XLA is decisive.
- **In short:** JAX + XLA turns our Python code into "gold," compiling it into the fastest possible program that can run on the hardware.

**3. Functional Programming Paradigm**

- **What it is:** JAX encourages writing "pure functions." A pure function has two characteristics: 1) the same input always produces the same output, and 2) it has no "side effects," such as modifying variables outside its scope. All your code should be stateless. The model's weights and optimizer states must be passed explicitly as arguments to functions, which in turn return the new state.
- **Why it matters:**
  - **Predictability and Easy Debugging:** With no hidden state modifications, the behavior of the code becomes completely predictable. When a bug occurs, you don't have to worry about some unrelated part of the code secretly modifying your data, which greatly simplifies debugging.
  - **Born for Parallelism:** Pure functions have no dependencies or state contentions, making it incredibly easy for JAX to parallelize, vectorize, and distribute your code without worrying about thread safety.
  - **Composability:** `jit` (compilation), `vmap` (auto-vectorization), `grad` (auto-differentiation), `pmap` (parallelization)... The core of JAX is these function transformations that can be combined like Lego bricks. You can write a simple function for a single data point, use `vmap` to turn it into a function that handles a batch, `jit` to compile it, and `pmap` to distribute it across multiple devices. The entire process is clean and powerful.
- **In short:** The functional paradigm makes our code more robust, easier to debug, and perfectly suited for JAX's powerful compilation and parallelization capabilities.

---

#### How to Learn JAX and Pallas

We recommend following this path to systematically master JAX and Pallas.

**Phase 1: Master the JAX Core**

For a beginner, the most important thing is to understand the JAX "way of thinking."

1. **Start with the Official Tutorials:**

  - **Must-Read:** [JAX 101 Tutorial](https://jax.readthedocs.io/en/latest/jax-101/index.html). This is the best introductory material, bar none. Spend a few hours going through it completely and run every code example yourself.
  - **Core Concepts:** You need to focus on the differences between JAX and NumPy (especially immutability) and master JAX's three core function transformations:
    - `jax.jit()`: Understand how it compiles functions into high-performance code.
    - `jax.vmap()`: Grasp its magic of automatically converting a function that processes a single data point into one that handles a batch, saving you from writing manual loops.
    - `jax.grad()`: While not used directly for training in our inference project, understanding it is key to grasping the JAX ecosystem.
2. **Learning Resources:**

  - **Official Documentation:** The [JAX Documentation](https://jax.readthedocs.io/en/latest/) is your most reliable friend.
  - **Code Examples:** Read through well-regarded open-source JAX projects like [Flax](https://github.com/google/flax) (a neural network library for JAX) to see how others structure their JAX code.

**Phase 2: Learn Distributed Programming**

Once you have a handle on the basics of JAX, you can start learning how to manage multiple devices.

1. **Understand `pmap`:** `pmap` is the parallel version of `vmap` and serves as the entry point to understanding JAX's parallelism. It easily distributes data and computation across multiple devices.
2. **Master the Modern Distributed Approach: `jax.sharding`:**
  - This is the method we actually use in `sglang-jax` and is central to handling large models.
  - **Learning Path:**
    1. Read the official guide on [Distributed arrays and automatic parallelization](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html).
    2. Understand the concepts of `Mesh`, `PartitionSpec`, and `NamedSharding`.
    3. Try to write code yourself that distributes a simple matrix multiplication across a 2x2 virtual device mesh.

**Phase 3: Advanced Pallas (Optional)**

Pallas is an advanced topic. We **do not recommend** diving into Pallas until you have a deep understanding of JAX's core and distributed programming.

1. **What is Pallas?**

  - Pallas is an extension to JAX that allows you to write custom, low-level computation "kernels" using Python syntax.
  - If JAX lets you drive an F1 car, Pallas gives you the toolkit to design and build the engine for that car yourself.
  - In `sglang-jax`, we use Pallas to implement operators with extreme performance requirements, like FlashAttention. This allows us to bypass some of JAX's higher-level abstractions to communicate more directly with the hardware, squeezing out every last drop of performance.
2. **How to Learn Pallas?**

  - **When to start:** You should only consider Pallas when you've hit a performance bottleneck and are certain that standard JAX operators can no longer meet your needs.
  - **Resources:**
    - Read the [Pallas Official Documentation](https://jax.readthedocs.io/en/latest/pallas/index.html).
    - Study the Pallas tutorials and examples in the Google Cloud TPU documentation.
    - **Best Practice:** After you are fluent with JAX, the best way to learn is by reading the Pallas implementations within the `sglang-jax` project itself (e.g., the code in the `sgl_jax/srt/layers/attention/` directory). This is the best material for combining theory with practice.

After reviewing the documentation and information above, we strongly recommend reading [Scaling ML with JAX](https://jax-ml.github.io/scaling-book/) in its entirety. This book provides a comprehensive guide to the concepts and best practices for high-performance computing with JAX. It is crucial to not only read the material but also to actively implement and practice the concepts described within.

---

We hope this guide helps you embark on your JAX learning journey. If you invest the time to understand the philosophy behind its design, you will find it to be an incredibly powerful and elegant tool. Happy coding!
