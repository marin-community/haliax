# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp

from .axis import Axis, AxisSelector, axis_name, dslice as HaliaxDSlice
from .core import (
    NamedArray,
    NamedArrayAxesSpec,
    NamedOrNumeric,
    SliceSpec,
    _compute_new_axes_and_slices_for_index,
    _convert_index_expr_to_dict,
    named,
)
from .jax_utils import is_pallas_dslice
from .util import ensure_tuple

try:  # pragma: no cover - optional dependency depending on JAX version
    from jax.experimental.pallas import dslice as PallasDSlice  # type: ignore
except ImportError:  # pragma: no cover - pallas may not be available
    PallasDSlice = None


class _AxisMetadata:
    """Minimal structure supplying just axis metadata for index helpers."""

    def __init__(self, axes: Sequence[Axis]):
        self.axes = tuple(axes)

    def axis_indices(self, axis: AxisSelector) -> int | None:
        name = axis_name(axis)
        for i, ax in enumerate(self.axes):
            if ax.name == name:
                return i
        return None


def _is_trivial_index(idx: Any) -> bool:
    return isinstance(idx, slice) and idx.start is None and idx.stop is None and idx.step is None


def _slice_length(start: int, stop: int, step: int) -> int:
    if step == 0:
        raise ValueError("slice step cannot be zero")
    if step > 0:
        if start >= stop:
            return 0
        return (stop - start + step - 1) // step
    if start <= stop:
        return 0
    step_abs = -step
    return (start - stop - step - 1) // step_abs


def _normalize_slice_value(value: Any) -> Any:
    if isinstance(value, range):
        return slice(value.start, value.stop, value.step)
    if isinstance(value, HaliaxDSlice) or is_pallas_dslice(value):
        return value
    return value


def _dslice_params(value: Any) -> tuple[Any, int, Any] | None:
    if isinstance(value, HaliaxDSlice):
        return value.start, value.size, 1
    if is_pallas_dslice(value):
        start = getattr(value, "start")
        size = getattr(value, "size")
        step = getattr(value, "step", 1)
        return start, size, step
    return None


def _make_dslice(start: Any, size: int, step: Any = 1, *, prefer_pallas: bool = False):
    if step != 1:
        if PallasDSlice is not None:
            return PallasDSlice(start, size, step)
        return slice(start, start + size * step, step)
    if prefer_pallas and PallasDSlice is not None:
        return PallasDSlice(start, size)
    return HaliaxDSlice(start, size)


def _axes_after_prefix(axes: Sequence[Axis], indices: Sequence[Any]) -> tuple[Axis, ...]:
    view_axes: list[Axis] = []
    for axis, sel in zip(axes, indices):
        if isinstance(sel, int):
            continue
        if isinstance(sel, slice):
            start, stop, step = sel.indices(axis.size)
            length = _slice_length(start, stop, step)
            view_axes.append(Axis(axis.name, length))
        else:
            params = _dslice_params(sel)
            if params is not None:
                _start, size, _step = params
                if not isinstance(size, int):
                    size = int(size)
                view_axes.append(Axis(axis.name, size))
                continue
            else:
                raise TypeError(
                    "Slice references currently only support integer or slice prefixes; "
                    f"got {type(sel)} for axis {axis}"
                )
    return tuple(view_axes)


def _combine_index(
    current: slice | int,
    new: Any,
    *,
    base_axis: Axis,
    view_axis: Axis,
) -> Any:
    if isinstance(current, int):
        raise ValueError(f"Axis {base_axis.name} is already fixed by the slice reference")

    new = _normalize_slice_value(new)

    ds_params = _dslice_params(current)
    if ds_params is not None:
        start0, size0, step0 = ds_params
        view_length = size0

        if isinstance(new, slice):
            start1, stop1, step1 = new.indices(view_length)
            new_size = _slice_length(start1, stop1, step1)
            return _make_dslice(start0 + start1 * step0, new_size, step0 * step1)

        if isinstance(new, (HaliaxDSlice,)) or is_pallas_dslice(new):
            n_params = _dslice_params(new)
            assert n_params is not None
            start1, size1, step1 = n_params
            return _make_dslice(start0 + start1 * step0, size1, step0 * step1)

        if isinstance(new, int):
            if new < -view_length or new >= view_length:
                raise IndexError(f"Index {new} out of bounds for axis {view_axis.name}")
            if new < 0:
                new += view_length
            return start0 + new * step0

        if isinstance(new, NamedArray):
            data = jnp.where(new.array < 0, new.array + view_length, new.array)
            transformed = data * step0 + start0
            return NamedArray(transformed, new.axes)

        if isinstance(new, jnp.ndarray):
            data = jnp.where(new < 0, new + view_length, new)
            return data * step0 + start0

        if isinstance(new, (list, tuple)):
            converted: list[int] = []
            for item in new:
                if not isinstance(item, int):
                    raise TypeError("Only integer lists/tuples are supported for slice references")
                if item < -view_length or item >= view_length:
                    raise IndexError(f"Index {item} out of bounds for axis {view_axis.name}")
                if item < 0:
                    item += view_length
                converted.append(start0 + item * step0)
            return converted

        raise TypeError(
            "Slice references only support integers, slices, dslice, NamedArray, jnp.ndarray, or lists/tuples thereof"
            f"; got {type(new)}"
        )

    # Fallback to python slice semantics for basic slices
    start0, stop0, step0 = current.indices(base_axis.size)
    view_length = _slice_length(start0, stop0, step0)

    if isinstance(new, slice):
        start1, stop1, step1 = new.indices(view_length)
        return slice(start0 + start1 * step0, start0 + stop1 * step0, step0 * step1)

    if isinstance(new, (HaliaxDSlice,)) or is_pallas_dslice(new):
        n_params = _dslice_params(new)
        assert n_params is not None
        start1, size1, step1 = n_params
        return _make_dslice(start0 + start1 * step0, size1, step0 * step1)

    if isinstance(new, int):
        if new < -view_length or new >= view_length:
            raise IndexError(f"Index {new} out of bounds for axis {view_axis.name}")
        if new < 0:
            new += view_length
        return start0 + new * step0

    if isinstance(new, NamedArray):
        data = jnp.where(new.array < 0, new.array + view_length, new.array)
        transformed = data * step0 + start0
        return NamedArray(transformed, new.axes)

    if isinstance(new, jnp.ndarray):
        data = jnp.where(new < 0, new + view_length, new)
        return data * step0 + start0

    if isinstance(new, (list, tuple)):
        converted: list[int] = []
        for item in new:
            if not isinstance(item, int):
                raise TypeError("Only integer lists/tuples are supported for slice references")
            if item < -view_length or item >= view_length:
                raise IndexError(f"Index {item} out of bounds for axis {view_axis.name}")
            if item < 0:
                item += view_length
            converted.append(start0 + item * step0)
        return converted

    raise TypeError(
        "Slice references only support integers, slices, dslice, NamedArray, jnp.ndarray, or lists/tuples thereof"
        f"; got {type(new)}"
    )


def _combine_indices(
    axes: Sequence[Axis],
    prefix: Sequence[Any],
    selectors: Mapping[AxisSelector, Any],
) -> tuple[Any, ...]:
    if not selectors:
        return tuple(prefix)

    current = list(prefix)
    view_axes = _axes_after_prefix(axes, prefix)
    view_positions = {ax.name: i for i, ax in enumerate(view_axes)}
    mapping = [i for i, sel in enumerate(prefix) if not isinstance(sel, int)]

    for axis_sel, new_value in selectors.items():
        name = axis_name(axis_sel)
        view_pos = view_positions.get(name)
        if view_pos is None:
            raise ValueError(f"Axis {name} is not available in this slice reference")
        base_pos = mapping[view_pos]
        base_axis = axes[base_pos]
        view_axis = view_axes[view_pos]
        combined = _combine_index(current[base_pos], new_value, base_axis=base_axis, view_axis=view_axis)
        current[base_pos] = combined

    return tuple(current)


def _indices_to_selector(axes: Sequence[Axis], indices: Sequence[Any]) -> dict[AxisSelector, Any]:
    selector: dict[AxisSelector, Any] = {}
    for axis, idx in zip(axes, indices):
        if _is_trivial_index(idx):
            continue
        selector[axis] = idx
    return selector


def _is_haliax_or_pallas_dslice(value: Any) -> bool:
    return isinstance(value, HaliaxDSlice) or is_pallas_dslice(value)


def _to_pallas_dslice_if_available(value: Any) -> Any:
    if is_pallas_dslice(value):
        return value
    if isinstance(value, HaliaxDSlice):
        if PallasDSlice is not None:
            return PallasDSlice(value.start, value.size)
        return value
    return value


@dataclass(frozen=True)
class NamedRef:
    """Named wrapper around :class:`jax.Ref` that preserves axis metadata."""

    _ref: jax.Ref
    _axes: tuple[Axis, ...]
    _prefix: tuple[Any, ...]

    def __post_init__(self):
        if len(self._axes) != len(self._prefix):
            raise ValueError("Prefix entries must align with axes")

    @property
    def dtype(self):
        return self._ref.dtype

    @property
    def axes(self) -> tuple[Axis, ...]:
        return _axes_after_prefix(self._axes, self._prefix)

    @property
    def shape(self) -> Mapping[str, int]:
        return {ax.name: ax.size for ax in self.axes}

    @property
    def named_shape(self) -> Mapping[str, int]:
        return self.shape

    @property
    def ndim(self) -> int:
        return len(self.axes)

    def value(self) -> NamedArray:
        _, axes_spec, index_tuple = self._prepare(Ellipsis)
        result = self._ref[tuple(index_tuple)]
        return named(result, axes_spec)

    def unsliced(self) -> "NamedRef":
        full_prefix = tuple(slice(None) for _ in self._axes)
        if self._prefix == full_prefix:
            return self
        return NamedRef(self._ref, self._axes, full_prefix)

    def _prepare(self, idx: SliceSpec | Ellipsis | None) -> tuple[tuple[Any, ...], tuple[str, ...], list[Any]]:
        if idx is Ellipsis or idx is None:
            selectors: Mapping[AxisSelector, Any] = {}
        else:
            selectors = _convert_index_expr_to_dict(idx)
        combined = _combine_indices(self._axes, self._prefix, selectors)
        selector_dict = _indices_to_selector(self._axes, combined)
        array_info = _AxisMetadata(self._axes)
        new_axes, ordered = _compute_new_axes_and_slices_for_index(array_info, selector_dict)
        index_tuple: list[Any] = []
        for axis, item in zip(self._axes, ordered):
            selector_value = selector_dict.get(axis)
            if selector_value is not None and _is_haliax_or_pallas_dslice(selector_value):
                index_tuple.append(_to_pallas_dslice_if_available(selector_value))
            else:
                index_tuple.append(item.array if isinstance(item, NamedArray) else item)
        return combined, ensure_tuple(new_axes), index_tuple

    def __getitem__(self, idx: SliceSpec | Ellipsis = Ellipsis) -> NamedArray:
        _, axes_spec, index_tuple = self._prepare(idx)
        result = self._ref[tuple(index_tuple)]
        return named(result, axes_spec)

    def __setitem__(self, idx: SliceSpec | Ellipsis, value: NamedOrNumeric) -> None:
        _, axes_spec, index_tuple = self._prepare(idx)
        if isinstance(value, NamedArray):
            desired = axes_spec
            current_names = tuple(axis_name(ax) for ax in value.axes)
            if set(current_names) != set(desired):
                raise ValueError(
                    f"Value axes {current_names} do not match target axes {desired}; broadcasting is not yet supported"
                )
            if current_names != desired:
                value = value.rearrange(desired)
            payload = value.array
        else:
            payload = jnp.asarray(value)
        self._ref[tuple(index_tuple)] = payload

    def slice(self, selector: Mapping[AxisSelector, Any]) -> NamedRef:
        normalized = {key: _normalize_slice_value(val) for key, val in selector.items()}
        combined = _combine_indices(self._axes, self._prefix, normalized)
        for idx in combined:
            if not (
                isinstance(idx, (slice, int))
                or _is_haliax_or_pallas_dslice(idx)
                or jnp.isscalar(idx)
                or (hasattr(idx, "shape") and getattr(idx, "ndim", 0) == 0)
            ):
                raise TypeError("Slice references only support simple integer/slice prefixes")
        return replace(self, _prefix=combined)

    def unsafe_buffer_pointer(self):  # pragma: no cover
        return self._ref.unsafe_buffer_pointer()


def new_ref(value: NamedArray | jax.Array | Any, axes: NamedArrayAxesSpec | None = None) -> NamedRef:
    if isinstance(value, NamedArray):
        base_axes = value.axes
        impl = jax.new_ref(value.array)
    else:
        array = jnp.asarray(value)
        if axes is None:
            raise ValueError("axes must be provided when creating a NamedRef from raw arrays")
        base_axes = named(array, axes).axes
        impl = jax.new_ref(array)
    prefix = tuple(slice(None) for _ in base_axes)
    return NamedRef(impl, base_axes, prefix)


def freeze(ref: NamedRef) -> NamedArray:
    _, axes_spec, index_tuple = ref._prepare(Ellipsis)
    ref_module = getattr(jax, "ref", None)
    if ref_module is None or not hasattr(ref_module, "freeze"):
        return ref.value()

    frozen = ref_module.freeze(ref._ref)
    view = frozen[tuple(index_tuple)] if len(index_tuple) > 0 else frozen
    return named(view, axes_spec)


def get(ref: NamedRef, idx: SliceSpec | Ellipsis = Ellipsis) -> NamedArray:
    return ref[idx]


def swap(ref: NamedRef, idx: SliceSpec | Ellipsis, value: NamedOrNumeric) -> NamedArray:
    _, axes_spec, index_tuple = ref._prepare(idx)
    if isinstance(value, NamedArray):
        desired = axes_spec
        current_names = tuple(axis_name(ax) for ax in value.axes)
        if set(current_names) != set(desired):
            raise ValueError(
                f"Value axes {current_names} do not match target axes {desired}; broadcasting is not yet supported"
            )
        if current_names != desired:
            value = value.rearrange(desired)
        payload = value.array
    else:
        payload = jnp.asarray(value)

    ref_module = getattr(jax, "ref", None)
    if ref_module is None or not hasattr(ref_module, "swap"):
        previous = ref[idx]
        ref[idx] = value
        return previous

    out = ref_module.swap(ref._ref, tuple(index_tuple), payload)
    return named(out, axes_spec)


__all__ = ["NamedRef", "new_ref", "freeze", "get", "swap"]


def _namedref_flatten(ref: NamedRef):
    return (ref._ref,), (ref._axes, ref._prefix)


def _namedref_unflatten(aux, children):
    (axes, prefix) = aux
    (ref_impl,) = children
    return NamedRef(ref_impl, tuple(axes), tuple(prefix))


jax.tree_util.register_pytree_node(NamedRef, _namedref_flatten, _namedref_unflatten)
