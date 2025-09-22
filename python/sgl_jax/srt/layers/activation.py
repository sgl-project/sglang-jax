# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import jax
from flax import nnx


class GeluAndMul(nnx.Module):
    def __init__(self, approximate="tanh"):
        super().__init__()
        self.approximate = approximate

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.approximate == "tanh":
            out = jax.nn.gelu(x, approximate=True) * x
        elif self.approximate == "none":
            out = jax.nn.gelu(x, approximate=False) * x
        else:
            raise RuntimeError("GeluAndMul only support tanh or none")
        return out
