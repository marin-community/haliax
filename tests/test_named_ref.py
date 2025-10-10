import jax
import jax.numpy as jnp
import pytest

import haliax as hax


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


def test_named_ref_basic_get_set(key):
    X = hax.Axis("x", 4)
    Y = hax.Axis("y", 3)
    array = hax.random.uniform(key, (X, Y))
    ref = hax.new_ref(array)

    assert ref.shape == array.shape
    assert ref.axes == array.axes

    slice_val = ref[{"x": slice(1, 3)}]
    assert slice_val.axes == (X.resize(2), Y)

    new_block = hax.ones_like(slice_val) * 2.0
    ref[{"x": slice(1, 3)}] = new_block
    updated = ref.value()
    assert jnp.allclose(updated[{"x": slice(1, 3)}].array, jnp.asarray(new_block.array))


def test_named_ref_scalar_update():
    X = hax.Axis("x", 5)
    ref = hax.new_ref(hax.zeros(X))
    ref[{"x": 2}] = 3.14
    assert pytest.approx(ref.value()[{"x": 2}].array.item()) == 3.14


def test_slice_ref_composition():
    Layer = hax.Axis("layer", 6)
    Head = hax.Axis("head", 8)
    cache = hax.zeros((Layer, Head))
    cache_ref = hax.new_ref(cache)

    layers_1_4 = cache_ref.slice({"layer": slice(1, 4)})
    assert layers_1_4.axes == (Layer.resize(3), Head)

    layers_1_4[{"layer": 0}] = hax.arange(Head).astype(jnp.float32)
    value = cache_ref.value()
    expected_row = jnp.arange(Head.size, dtype=jnp.float32)
    assert jnp.allclose(value[{"layer": 1}].array, expected_row)

    single_layer = layers_1_4.slice({"layer": 0})
    assert single_layer.axes == (Head,)
    single_layer[{"head": 0}] = 7.0
    updated = cache_ref.value()
    assert updated[{"layer": 1, "head": 0}].array == pytest.approx(7.0)


def test_freeze_returns_named_array():
    X = hax.Axis("x", 3)
    ref = hax.new_ref(hax.arange(X))
    frozen = hax.freeze(ref)
    assert isinstance(frozen, hax.NamedArray)
    assert frozen.axes == ref.axes
    assert jnp.allclose(frozen.array, ref.value().array)


def test_swap_returns_previous_value():
    X = hax.Axis("x", 4)
    ref = hax.new_ref(hax.zeros(X))
    prev = hax.swap(ref, {"x": slice(1, 3)}, hax.ones(X.resize(2)))
    assert isinstance(prev, hax.NamedArray)
    assert prev.axes == (X.resize(2),)
    assert jnp.allclose(prev.array, 0.0)
    assert jnp.allclose(ref.value()[{"x": slice(1, 3)}].array, jnp.ones((2,), dtype=jnp.float32))


def test_named_ref_jit_plumbing():
    X = hax.Axis("x", 5)
    ref = hax.new_ref(hax.zeros(X))

    @jax.jit
    def write_and_read(ref):
        ref[{"x": 1}] = 4.2
        return ref[{"x": 1}]

    out = write_and_read(ref)
    assert isinstance(out, hax.NamedArray)
    assert out.axes == ()
    assert pytest.approx(out.array.item()) == 4.2
    assert jnp.allclose(ref.value()[{"x": 1}].array, jnp.asarray(4.2))


def test_named_ref_is_pytree_leaf():
    X = hax.Axis("x", 3)
    ref = hax.new_ref(hax.zeros(X))

    leaves = jax.tree_util.tree_leaves(ref)
    assert len(leaves) == 1
    assert leaves[0] is ref._ref

    structure = jax.tree_util.tree_structure(ref)
    rebuilt = jax.tree_util.tree_unflatten(structure, leaves)

    assert isinstance(rebuilt, hax.NamedRef)
    assert rebuilt.axes == ref.axes
    assert rebuilt._prefix == ref._prefix


def test_with_scan():
    X = hax.Axis("x", 5)
    ref = hax.new_ref(hax.zeros(X))

    @jax.jit
    def foo(ref, xs):
        def scan_fn(_, x):
            ref_slice = ref.slice({"x": x})
            ref_slice[...] = (x * x).astype(ref_slice.dtype)
            return None, x * 2
        
        return hax.scan(scan_fn, X)(None, xs)[1]

    out = foo(ref, jnp.arange(X.size))

    assert jnp.all(ref.value().array == jnp.arange(X.size) ** 2)

    assert jnp.all(out == jnp.arange(X.size) * 2)