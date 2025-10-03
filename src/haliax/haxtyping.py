# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


@dataclass(frozen=True)
class DTypeCategory:
    """Represents a dtype category such as ``float`` or ``int``."""

    name: str
    category: Any

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return self.name


if TYPE_CHECKING:
    # ── STATIC ONLY: re‑export jaxtyping’s aliases so mypy/Pyright/PyCharm see them
    from jaxtyping import (
        Float32 as f32,
        Float64 as f64,
        Float16 as f16,
        BFloat16 as bf16,
        Int8 as i8,
        Int16 as i16,
        Int32 as i32,
        Int64 as i64,
        UInt8 as u8,
        UInt16 as u16,
        UInt32 as u32,
        UInt64 as u64,
        Bool as bool_,
        Complex64 as complex64,
        Complex128 as complex128,
        Float as Float,
        Int as Int,
        UInt as UInt,
    )

    # axes‑only helper
    from typing import Annotated as Named
    from .core import Axis, NamedArray

    def check_axes(**arrays: NamedArray) -> dict[str, Axis]: ...

else:
    # ── RUNTIME: custom wrappers for NamedArray, plus delegation to jaxtyping ──
    import jaxtyping as jt
    import jax.numpy as jnp
    from typing import Annotated
    import inspect
    import typing
    from dataclasses import dataclass, replace

    from .core import (
        Axis,
        NamedArray,
        NamedArrayAxes,
        NamedArrayAxesSpec,
        _parse_namedarray_axes,
    )

    def _with_dtype(axes: NamedArrayAxes, dtype):
        """Attach dtype to axes metadata if not already set."""
        return axes if axes.dtype is not None else replace(axes, dtype=dtype)

    def _make_dtype_wrapper(dtype):
        """Factory for f32, i32, etc."""

        class _Wrapper:
            def __class_getitem__(cls, item):
                # two‑arg form: (BaseType, axes_spec)
                if isinstance(item, tuple) and len(item) == 2:
                    base, axes_spec = item
                else:
                    base, axes_spec = NamedArray, item

                # Delegate non‑NamedArray to jaxtyping
                if base is not NamedArray:
                    # e.g. use jt.Float32 for jnp.float32
                    jaxt = getattr(jt, f"Float{dtype.itemsize*8}") if hasattr(dtype, "itemsize") else jt.Float
                    return jaxt[base, axes_spec]

                # Handle NamedArray path
                axes = _parse_namedarray_axes(axes_spec)
                return Annotated[NamedArray, _with_dtype(axes, dtype)]

        return _Wrapper

    # ── Build all dtype wrappers ─────────────────────────────────────────────
    f32 = _make_dtype_wrapper(jnp.float32)
    f64 = _make_dtype_wrapper(jnp.float64)
    f16 = _make_dtype_wrapper(jnp.float16)
    bf16 = _make_dtype_wrapper(jnp.bfloat16)

    i8 = _make_dtype_wrapper(jnp.int8)
    i16 = _make_dtype_wrapper(jnp.int16)
    i32 = _make_dtype_wrapper(jnp.int32)
    i64 = _make_dtype_wrapper(jnp.int64)

    u8 = _make_dtype_wrapper(jnp.uint8)
    u16 = _make_dtype_wrapper(jnp.uint16)
    u32 = _make_dtype_wrapper(jnp.uint32)
    u64 = _make_dtype_wrapper(jnp.uint64)

    bool_ = _make_dtype_wrapper(jnp.bool_)
    complex64 = _make_dtype_wrapper(jnp.complex64)
    complex128 = _make_dtype_wrapper(jnp.complex128)

    def _make_category_wrapper(name: str, category):
        """Like _make_dtype_wrapper but matches any dtype in the JAX category."""

        class _Wrapper:
            def __class_getitem__(cls, item):
                # same base/axes unpack logic
                if isinstance(item, tuple) and len(item) == 2:
                    base, axes_spec = item
                else:
                    base, axes_spec = NamedArray, item

                # non‑NamedArray → delegate to jaxtyping’s category wrapper
                if base is not NamedArray:
                    return getattr(jt, name)[base, axes_spec]

                # NamedArray path
                axes = _parse_namedarray_axes(axes_spec)
                cat = DTypeCategory(name, category)
                return Annotated[NamedArray, _with_dtype(axes, cat)]

        return _Wrapper

    # Build the category wrappers
    Float = _make_category_wrapper("float", jnp.floating)
    Complex = _make_category_wrapper("complex", jnp.complexfloating)
    Int = _make_category_wrapper("int", jnp.signedinteger)
    UInt = _make_category_wrapper("uInt", jnp.unsignedinteger)

    # ── Named: axes‑only helper ───────────────────────────────────────────────
    class _NamedHelper:
        @classmethod
        def __class_getitem__(self, axes_spec_: tuple[type[NamedArray], NamedArrayAxesSpec]):
            _, axes_spec = axes_spec_
            axes = _parse_namedarray_axes(axes_spec)
            return Annotated[NamedArray, axes]

    Named = _NamedHelper

    __all__ = [
        "Named",
        "f32",
        "f64",
        "f16",
        "bf16",
        "i8",
        "i16",
        "i32",
        "i64",
        "u8",
        "u16",
        "u32",
        "u64",
        "bool_",
        "complex64",
        "complex128",
    ]

    GenericName = str  # capitalized axis name used for generics

    def _is_generic(name: str) -> bool:
        return bool(name and name[0].isupper())

    def _resolve_unordered(
        arr: NamedArray, spec: NamedArrayAxesSpec, generics: dict[GenericName, Axis]
    ) -> None:
        axes = _parse_namedarray_axes(spec)
        remaining = {ax.name: ax for ax in arr.axes}

        # match explicit axis names first
        for name in axes.before:
            if _is_generic(name):
                continue
            if name not in remaining:
                raise ValueError(f"Axis {name} not found in {arr.axes}")
            remaining.pop(name)

        # assign generics
        leftover = sorted(remaining.items())
        for name in axes.before:
            if not _is_generic(name):
                continue
            if name in generics:
                expected = generics[name]
                if expected.name not in remaining:
                    raise ValueError(
                        f"Generic axis {name} expected {expected.name} but not present"
                    )
                ax = remaining.pop(expected.name)
                if ax.size != expected.size:
                    raise ValueError(
                        f"Generic axis {name} size mismatch: {ax.size} vs {expected.size}"
                    )
            else:
                if not leftover:
                    raise ValueError(f"Not enough axes to resolve generic {name}")
                ax_name, ax = leftover.pop(0)
                remaining.pop(ax_name, None)
                generics[name] = ax

        if not axes.subset and remaining:
            raise ValueError(f"Unexpected axes {list(remaining)} for spec {spec}")


    def check_axes(**arrays: NamedArray) -> dict[str, Axis]:
        """Validate NamedArray arguments against their type annotations and resolve generic axes."""

        frame = inspect.currentframe()
        assert frame is not None
        caller = frame.f_back
        if caller is None:
            raise RuntimeError("check_axes must be called from within a function")

        fn = None
        for obj in caller.f_globals.values():
            if inspect.isfunction(obj) and obj.__code__ is caller.f_code:
                fn = obj
                break
        if fn is None:
            raise RuntimeError("Unable to locate calling function")

        hints = typing.get_type_hints(fn, include_extras=True)
        generics: dict[str, Axis] = {}

        for name, arr in arrays.items():
            spec = hints.get(name)
            if spec is None:
                continue
            ann = _parse_namedarray_axes(spec)
            if ann.ordered:
                if not arr.matches_axes(ann):
                    raise ValueError(f"Array {name} does not match annotation {ann}")
            else:
                _resolve_unordered(arr, spec, generics)

        return generics

    __all__.append("check_axes")
