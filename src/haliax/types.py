# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


from typing import Any, Callable, Literal, Protocol, Tuple, TypeAlias, Union

import jax.numpy as jnp
import numpy as np
from jax.lax import Precision
from jaxtyping import PyTree

DType: TypeAlias = np.dtype

try:
    from jax.typing import DTypeLike
except ImportError:
    # Cribbed from jax.typing, for older versions of JAX
    class SupportsDType(Protocol):
        @property
        def dtype(self) -> DType: ...

    DTypeLike = Union[
        str,  # like 'float32', 'int32'
        type,  # like np.float32, np.int32, float, int
        np.dtype,  # like np.dtype('float32'), np.dtype('int32')
        SupportsDType,  # like jnp.float32, jnp.int32
    ]


Scalar = Union[float, int, jnp.ndarray]  # ndarray b/c array(1) is a scalar
IntScalar = Union[int, jnp.ndarray]

PrecisionLike = Union[None, str, Precision, Tuple[str, str], Tuple[Precision, Precision]]

GatherScatterModeStr = Literal["promise_in_bounds", "clip", "drop", "fill"]


FilterSpec = Union[bool, Callable[[Any], bool]]
"""
A filter specification. Typically used on a pytree to filter out certain subtrees. Boolean values are
treated as-is, while callables are called on each element of the pytree. If the callable returns True, the element
is kept, otherwise it is filtered out.
"""

FilterTree = FilterSpec | PyTree[FilterSpec]
