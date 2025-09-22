# SGLang-JAX Style Guide

# Introduction
This style guide outlines the coding conventions for code developed for the SGLang-JAX project.

# General Code Style
* **Avoid code duplication:** If the same code snippet (more than five lines) appears multiple times, extract it into a shared function.
* **Keep files concise:** If a file exceeds 2,000 lines of code, split it into multiple smaller files.
* **Strive for function purity:** Write pure functions whenever possible. Avoid in-place modification of arguments.
* **Unit tests:** If a new feature is added or a bug is fixed, add corresponding unit tests using Python's `unittest` framework.
