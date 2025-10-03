import jax.numpy as jnp

import haliax as hax
import haliax.haxtyping as ht


def foo(a: ht.f32[{"B", "embed"}], x: ht.i32[{"pos", "B"}]):
    resolved = hax.check_axes(a=a, x=x)
    return resolved


def test_generic_axis_resolution():
    res = foo(
        hax.zeros({"batch": 12, "embed": 4}),
        hax.zeros({"pos": 4, "batch": 12}, dtype=jnp.int32),
    )
    assert res["B"] == hax.Axis("batch", 12)


def test_generic_axis_mismatch():
    try:
        foo(
            hax.zeros({"batch": 12, "embed": 4}),
            hax.zeros({"pos": 4, "time": 12}, dtype=jnp.int32),
        )
    except ValueError:
        pass
    else:
        assert False
