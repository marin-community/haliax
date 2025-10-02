# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import equinox as eqx
import jax
import pytest
from equinox import filter_grad

import haliax as hax
from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.scan import BlockSeq, ScanCheckpointPolicy, Stacked


def test_unstacked():
    class Module(eqx.Module):
        named: hax.NamedArray
        array: jax.Array
        static: int = eqx.static_field()

        def __call__(self, x, *, key):
            return x + self.array + self.static

        @staticmethod
        def init(named, array, static):
            return Module(named=named, array=array, static=static)

    Block = hax.Axis("block", 4)
    E = hax.Axis("E", 10)

    initial_named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))

    m = Stacked.init(Block, Module)(named=initial_named, array=jax.numpy.ones(Block.size), static=1)

    assert m.stacked.named.axes == (Block, E)
    assert m.stacked.array.shape == (Block.size,)
    assert m.stacked.static == 1

    unstacked = m.unstacked()

    assert len(unstacked) == Block.size

    for i, module in enumerate(unstacked):
        assert module.named.axes == (E,)
        assert module.array.shape == ()
        assert module.static == 1

        assert hax.all(module.named == m.stacked.named["block", i])
        assert hax.all(module.array == m.stacked.array[i])


def test_vmap():
    class Module(eqx.Module):
        weight: hax.NamedArray

        def __call__(self, x):
            return x + self.weight

        @staticmethod
        def init(weight):
            return Module(weight=weight)

    Block = hax.Axis("block", 4)
    E = hax.Axis("E", 10)

    weights = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))
    m = Stacked.init(Block, Module)(weight=weights)

    x = hax.random.uniform(jax.random.PRNGKey(1), (E,))
    y = m.vmap(x)

    assert y.axes == (Block, E)
    assert hax.all(y == weights + x)


def test_seq_and_stacked_give_same_results():
    class Module(eqx.Module):
        named: hax.NamedArray
        array: jax.Array
        static: int = eqx.static_field()

        def __call__(self, x, *, key):
            return x + self.array + self.static + hax.random.normal(key, x.axes)

        @staticmethod
        def init(named, array, static):
            return Module(named=named, array=array, static=static)

    Block = hax.Axis("block", 4)
    E = hax.Axis("E", 10)

    initial_named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))

    m = Stacked.init(Block, Module)(named=initial_named, array=jax.numpy.ones(Block.size), static=1)
    m_seq = BlockSeq.init(Block, Module)(named=initial_named, array=jax.numpy.ones(Block.size), static=1)

    x = hax.random.uniform(jax.random.PRNGKey(1), (E,))
    y = m.fold(x, key=jax.random.split(jax.random.PRNGKey(2), Block.size))
    y_seq = m_seq.fold(x, key=jax.random.split(jax.random.PRNGKey(2), Block.size))
    assert hax.all(hax.isclose(y, y_seq, atol=1e-5))

    with pytest.raises(ValueError):
        m.scan(x, key=jax.random.split(jax.random.PRNGKey(2), Block.size))

    with pytest.raises(ValueError):
        m_seq.scan(x, key=jax.random.split(jax.random.PRNGKey(2), Block.size))


def test_using_scan():
    class Module(eqx.Module):
        named: hax.NamedArray
        array: jax.Array
        static: int = eqx.static_field()

        def __call__(self, x, *, key):
            return x + self.array + self.static + hax.random.normal(key, x.axes), x * 2

        @staticmethod
        def init(named, array, static):
            return Module(named=named, array=array, static=static)

    Block = hax.Axis("block", 4)
    E = hax.Axis("E", 10)

    initial_named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))

    m = Stacked.init(Block, Module)(named=initial_named, array=jax.numpy.ones(Block.size), static=1)

    x = hax.random.uniform(jax.random.PRNGKey(1), (E,))
    y, intermediates = m.scan(x, key=jax.random.split(jax.random.PRNGKey(2), Block.size))

    assert y.axes == (E,)
    assert intermediates.axes == (Block, E)


def test_scan_with_aux_named_args():
    class Module(eqx.Module):
        named: hax.NamedArray
        array: jax.Array
        static: int = eqx.static_field()

        def __call__(self, x, y, *, key):
            return x + self.array + self.static + hax.random.normal(key, x.axes), x * 2 + y

        @staticmethod
        def init(named, array, static):
            return Module(named=named, array=array, static=static)

    Block = hax.Axis("block", 4)
    E = hax.Axis("E", 10)

    initial_named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))
    initial_y = hax.random.uniform(jax.random.PRNGKey(1), (E,))

    m = Stacked.init(Block, Module)(named=initial_named, array=jax.numpy.ones(Block.size), static=1)
    m_seq = BlockSeq.init(Block, Module)(named=initial_named, array=jax.numpy.ones(Block.size), static=1)

    x = hax.random.uniform(jax.random.PRNGKey(1), (E,))
    z, z_scan = m.scan(x, initial_y, key=jax.random.split(jax.random.PRNGKey(2), Block.size))
    z_seq, z_seq_scan = m_seq.scan(x, initial_y, key=jax.random.split(jax.random.PRNGKey(2), Block.size))
    assert hax.all(hax.isclose(z, z_seq, atol=1e-5))

    assert hax.all(hax.isclose(z_scan, z_seq_scan, atol=1e-5))


def test_stacked_to_state_dict():
    class Module(eqx.Module):
        named: hax.NamedArray
        array: jax.Array
        static: int = eqx.static_field()

        def __call__(self, x, *, key):
            return x + self.array + self.static + hax.random.normal(key, x.axes)

        @staticmethod
        def init(named, array, static):
            return Module(named=named, array=array, static=static)

    Block = hax.Axis("block", 4)
    E = hax.Axis("E", 10)

    initial_named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))

    m = Stacked.init(Block, Module)(named=initial_named, array=jax.numpy.ones(Block.size), static=1)

    state_dict = m.to_state_dict()
    m2 = m.from_state_dict(state_dict)
    input = hax.random.uniform(jax.random.PRNGKey(1), (E,))
    key = jax.random.split(jax.random.PRNGKey(2), Block.size)

    y = m.fold(input, key=key)
    y2 = m2.fold(input, key=key)

    assert hax.all(hax.equal(y, y2))


Block = hax.Axis("block", 4)
E = hax.Axis("E", 10)


@pytest.mark.parametrize(
    "name,policy,expected_scan_shapes,check_offloading",
    [
        (
            "disabled",
            ScanCheckpointPolicy(disable=True),
            [(E.size,), (Block.size, E.size), (Block.size, E.size)],
            None,
        ),
        ("carry_true", True, [(E.size,), (Block.size, E.size)], None),
        (
            "carry",
            ScanCheckpointPolicy(save_carries=True, save_block_internals=False),
            [(E.size,), (Block.size, E.size)],
            None,
        ),
        (
            "everything",
            ScanCheckpointPolicy(save_carries=True, save_inputs=True, save_block_internals=True),
            [(E.size,), (Block.size, E.size), (Block.size, E.size)],
            None,
        ),
        (
            "internals",
            ScanCheckpointPolicy(save_carries=False, save_block_internals=True),
            [(E.size,), (Block.size, E.size), (Block.size, E.size)],
            None,
        ),
        (
            "cos",
            ScanCheckpointPolicy(save_carries=False, save_block_internals=["cos"]),
            [(E.size,), (Block.size, E.size)],
            None,
        ),
        (
            "sin",
            ScanCheckpointPolicy(save_carries=True, save_block_internals=["sin"]),
            [(E.size,), (Block.size, E.size), (Block.size, E.size)],
            None,
        ),
        ("simple", ScanCheckpointPolicy(simple=True), [(E.size,), (Block.size, E.size)], None),
        ("nested", ScanCheckpointPolicy(simple=True, nested=2), [(E.size,), (2, E.size)], None),
        (
            "sin_offload",
            ScanCheckpointPolicy(save_carries=True, offload_block_internals=["sin"]),
            [(E.size,), (Block.size, E.size), (Block.size, E.size)],
            ["sin"],
        ),
    ],
)
def test_checkpoint_carries(name, policy, expected_scan_shapes, check_offloading):
    class Module(eqx.Module):
        named: hax.NamedArray

        def __call__(self, x):
            y = tree_checkpoint_name(hax.sin(x + self.named), "sin")
            y = tree_checkpoint_name(hax.cos(y + x), "cos")
            return y + x

        @staticmethod
        def init(named):
            return Module(named=named)

    initial_named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))

    m = Stacked.init(
        Block,
        Module,
        gradient_checkpointing=policy,  # type: ignore
    )(named=initial_named)

    def loss_fn(m, x):
        y = m.fold(x)
        return hax.sum(y).scalar()

    grad_fn = filter_grad(loss_fn)

    jaxpr = jax.make_jaxpr(grad_fn)(m, hax.random.uniform(jax.random.PRNGKey(1), (E,)))
    closed_call = next(eqn for eqn in jaxpr.jaxpr.eqns if eqn.primitive in [jax.lax.scan_p])
    out_shapes = [out.aval.shape for out in closed_call.outvars]

    print(name)
    print(jaxpr)
    from jax._src.ad_checkpoint import saved_residuals

    for residual in saved_residuals(loss_fn, m, hax.random.uniform(jax.random.PRNGKey(1), (E,))):
        print(residual)

    assert out_shapes == expected_scan_shapes, f"{name}: Expected {expected_scan_shapes}, got {out_shapes}"

    # Add check for offloading if specified
    if check_offloading is not None:
        for name in check_offloading:
            print(f"Checking offloading for {name}")
            target = None
            found_saved = False
            for expr in jaxpr.jaxpr.eqns:
                if expr.primitive.name == "scan":
                    inner_jaxpr = expr.params["jaxpr"]
                    for eqn in inner_jaxpr.eqns:
                        if eqn.primitive.name == "name":
                            this_name = eqn.params["name"]
                            if this_name == name:
                                # TODO in theory we can save more than one thing with the same name
                                # not gonna worry about that for now
                                target = eqn.outvars[0]
                        elif eqn.primitive.name == "device_put":
                            if eqn.invars[0] == target:
                                found_saved = True
                                break
                    # found scan
                    break

            assert target is not None, f"Could not find named value for {name}"
            assert found_saved, f"Could not find offloaded value for {name}"


def test_fold_via():
    class Module(eqx.Module):
        w: hax.NamedArray

        def __call__(self, x):
            return x + self.w

        def intermediate(self, x):
            return x + 2 * self.w

        @staticmethod
        def init(named):
            return Module(w=named)

    Block = hax.Axis("block", 3)
    E = hax.Axis("E", 5)

    named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))
    m = Stacked.init(Block, Module)(named=named)

    x = hax.random.uniform(jax.random.PRNGKey(1), (E,))
    result = m.fold_via(Module.intermediate)(x)

    expected = x + 2 * hax.sum(named, Block)
    assert hax.all(hax.isclose(result, expected))


def test_scan_via():
    class Module(eqx.Module):
        w: hax.NamedArray

        def with_output(self, x):
            out = x + self.w
            return out, 2 * self.w

        @staticmethod
        def init(named):
            return Module(w=named)

    Block = hax.Axis("block", 4)
    E = hax.Axis("E", 6)

    named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))
    m = Stacked.init(Block, Module)(named=named)

    x = hax.random.uniform(jax.random.PRNGKey(1), (E,))
    carry, outs = m.scan_via(Module.with_output)(x)

    expected_carry = x + hax.sum(named, Block)
    expected_outs = 2 * named

    assert hax.all(hax.isclose(carry, expected_carry))
    assert hax.all(hax.isclose(outs, expected_outs))


def test_scan_via_multi_args():
    class Module(eqx.Module):
        w: hax.NamedArray

        def with_output(self, x, y, z, *, static1, static2):
            assert static1 is True
            assert static2 is False
            out = x + self.w + y + z
            return out, 2 * self.w + y

        @staticmethod
        def init(named):
            return Module(w=named)

    Block = hax.Axis("block", 4)
    E = hax.Axis("E", 6)

    named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))
    m = Stacked.init(Block, Module)(named=named)

    x = hax.random.uniform(jax.random.PRNGKey(1), (E,))
    y = hax.random.uniform(jax.random.PRNGKey(2), (E,))
    z = 3.0  # scalar that shouldn't be scanned

    carry, outs = m.scan_via(Module.with_output)(x, y, z, static1=True, static2=False)

    # compute expected values via a reference Python loop
    expected_carry = x
    expected_outs_list = []
    for i in range(Block.size):
        expected_outs_list.append(2 * named["block", i] + y)
        expected_carry = expected_carry + named["block", i] + y + z

    expected_outs = hax.stack(Block, expected_outs_list)

    assert hax.all(hax.isclose(carry, expected_carry))
    assert hax.all(hax.isclose(outs, expected_outs))


def test_scan_via_static_args():
    class Module(eqx.Module):
        w: hax.NamedArray

        def with_output(self, x, static1, *, static2):
            assert static1 == 1.0
            assert static2 is False
            out = x + self.w + static1
            return out, 2 * self.w

        @staticmethod
        def init(named):
            return Module(w=named)

    Block = hax.Axis("block", 4)
    E = hax.Axis("E", 6)

    named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))
    m = Stacked.init(Block, Module)(named=named)

    x = hax.random.uniform(jax.random.PRNGKey(1), (E,))

    carry, outs = m.scan_via(Module.with_output)(x, 1.0, static2=False)

    expected_carry = x
    expected_outs_list = []
    for i in range(Block.size):
        expected_outs_list.append(2 * named["block", i])
        expected_carry = expected_carry + named["block", i] + 1.0  # True -> 1.0

    expected_outs = hax.stack(Block, expected_outs_list)

    assert hax.all(hax.isclose(carry, expected_carry))
    assert hax.all(hax.isclose(outs, expected_outs))


def test_scan_via_doesnt_scan_scalars():
    class Module(eqx.Module):
        w: hax.NamedArray

        def with_output(self, x, scalar):
            out = x + self.w + scalar
            return out, x * scalar

        @staticmethod
        def init(named):
            return Module(w=named)

    Block = hax.Axis("block", 4)
    E = hax.Axis("E", 6)

    named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))
    m = Stacked.init(Block, Module)(named=named)

    x = hax.random.uniform(jax.random.PRNGKey(1), (E,))
    scalar = 4.0

    carry, outs = m.scan_via(Module.with_output)(x, scalar)

    expected_carry = x
    expected_outs_list = []
    for i in range(Block.size):
        expected_outs_list.append(expected_carry * scalar)
        expected_carry = expected_carry + named["block", i] + scalar

    expected_outs = hax.stack(Block, expected_outs_list)

    assert hax.all(hax.isclose(carry, expected_carry))
    assert hax.all(hax.isclose(outs, expected_outs))


def test_fold_via_multi_args():
    class Module(eqx.Module):
        w: hax.NamedArray

        def intermediate(self, x, y, z, *, static1, static2):
            assert static1 is True
            assert static2 is False
            return x + 2 * self.w + y + z

        @staticmethod
        def init(named):
            return Module(w=named)

    Block = hax.Axis("block", 3)
    E = hax.Axis("E", 5)

    named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))
    m = Stacked.init(Block, Module)(named=named)

    x = hax.random.uniform(jax.random.PRNGKey(1), (E,))
    y = hax.random.uniform(jax.random.PRNGKey(2), (E,))
    z = 3.0  # scalar that shouldn't be scanned

    result = m.fold_via(Module.intermediate)(x, y, z, static1=True, static2=False)

    expected = x
    for i in range(Block.size):
        expected = expected + 2 * named["block", i] + y + z

    assert hax.all(hax.isclose(result, expected))


def test_fold_via_static_args():
    class Module(eqx.Module):
        w: hax.NamedArray

        def intermediate(self, x, static1, *, static2):
            assert static1 is True
            assert static2 is False
            return x + 2 * self.w + static1

        @staticmethod
        def init(named):
            return Module(w=named)

    Block = hax.Axis("block", 3)
    E = hax.Axis("E", 5)

    named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))
    m = Stacked.init(Block, Module)(named=named)

    x = hax.random.uniform(jax.random.PRNGKey(1), (E,))

    result = m.fold_via(Module.intermediate)(x, True, static2=False)

    expected = x
    for i in range(Block.size):
        expected = expected + 2 * named["block", i] + 1.0

    assert hax.all(hax.isclose(result, expected))


def test_fold_via_doesnt_reduce_scalars():
    class Module(eqx.Module):
        w: hax.NamedArray

        def intermediate(self, x, scalar):
            return x + 2 * self.w + scalar

        @staticmethod
        def init(named):
            return Module(w=named)

    Block = hax.Axis("block", 3)
    E = hax.Axis("E", 5)

    named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))
    m = Stacked.init(Block, Module)(named=named)

    x = hax.random.uniform(jax.random.PRNGKey(1), (E,))
    scalar = 4.0

    result = m.fold_via(Module.intermediate)(x, scalar)

    expected = x
    for i in range(Block.size):
        expected = expected + 2 * named["block", i] + scalar

    assert hax.all(hax.isclose(result, expected))


def test_vmap_via():
    class Module(eqx.Module):
        w: hax.NamedArray

        def transform(self, x):
            return x + self.w

        @staticmethod
        def init(named):
            return Module(w=named)

    Block = hax.Axis("block", 4)
    E = hax.Axis("E", 6)

    named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))
    m = Stacked.init(Block, Module)(named=named)

    x = hax.random.uniform(jax.random.PRNGKey(1), (E,))
    outs = m.vmap_via(Module.transform)(x)

    expected_outs = x + named

    assert hax.all(hax.isclose(outs, expected_outs))


def test_vmap_via_multi_args():
    class Module(eqx.Module):
        w: hax.NamedArray

        def transform(self, x, y, z, *, static1, static2):
            assert static1 is True
            assert static2 is False
            return x + self.w + y + z

        @staticmethod
        def init(named):
            return Module(w=named)

    Block = hax.Axis("block", 4)
    E = hax.Axis("E", 6)

    named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))
    m = Stacked.init(Block, Module)(named=named)

    x = hax.random.uniform(jax.random.PRNGKey(1), (E,))
    y = hax.random.uniform(jax.random.PRNGKey(2), (E,))
    z = 3.0  # scalar that shouldn't be vmapped

    outs = m.vmap_via(Module.transform)(x, y, z, static1=True, static2=False)

    expected_outs = x + named + y + z

    assert hax.all(hax.isclose(outs, expected_outs))


def test_vmap_via_static_args():
    class Module(eqx.Module):
        w: hax.NamedArray

        def transform(self, x, static1, *, static2):
            assert static1 == 1.0
            assert static2 is False
            return x + self.w + static1

        @staticmethod
        def init(named):
            return Module(w=named)

    Block = hax.Axis("block", 4)
    E = hax.Axis("E", 6)

    named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))
    m = Stacked.init(Block, Module)(named=named)

    x = hax.random.uniform(jax.random.PRNGKey(1), (E,))

    outs = m.vmap_via(Module.transform)(x, 1.0, static2=False)

    expected_outs = x + named + 1.0

    assert hax.all(hax.isclose(outs, expected_outs))


def test_vmap_via_doesnt_vmap_scalars():
    class Module(eqx.Module):
        w: hax.NamedArray

        def transform(self, x, scalar):
            return x + self.w + scalar

        @staticmethod
        def init(named):
            return Module(w=named)

    Block = hax.Axis("block", 4)
    E = hax.Axis("E", 6)

    named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))
    m = Stacked.init(Block, Module)(named=named)

    x = hax.random.uniform(jax.random.PRNGKey(1), (E,))
    scalar = 4.0

    outs = m.vmap_via(Module.transform)(x, scalar)

    expected_outs = x + named + scalar

    assert hax.all(hax.isclose(outs, expected_outs))


def test_vmap_via_blockseq():
    class Module(eqx.Module):
        w: hax.NamedArray

        def transform(self, x):
            return x + self.w

        @staticmethod
        def init(named):
            return Module(w=named)

    Block = hax.Axis("block", 4)
    E = hax.Axis("E", 6)

    named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))
    m = BlockSeq.init(Block, Module)(named=named)

    x = hax.random.uniform(jax.random.PRNGKey(1), (E,))
    outs = m.vmap_via(Module.transform)(x)

    expected_outs = x + named

    assert hax.all(hax.isclose(outs, expected_outs))


def test_vmap_via_blockseq_multi_args():
    class Module(eqx.Module):
        w: hax.NamedArray

        def transform(self, x, y, z, *, static1, static2):
            assert static1 is True
            assert static2 is False
            return x + self.w + y + z

        @staticmethod
        def init(named):
            return Module(w=named)

    Block = hax.Axis("block", 4)
    E = hax.Axis("E", 6)

    named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))
    m = BlockSeq.init(Block, Module)(named=named)

    x = hax.random.uniform(jax.random.PRNGKey(1), (E,))
    y = hax.random.uniform(jax.random.PRNGKey(2), (E,))
    z = 3.0  # scalar that shouldn't be vmapped

    outs = m.vmap_via(Module.transform)(x, y, z, static1=True, static2=False)

    expected_outs = x + named + y + z

    assert hax.all(hax.isclose(outs, expected_outs))


def test_vmap_via_consistency():
    """Test that vmap_via gives the same results as vmap for Stacked."""

    class Module(eqx.Module):
        w: hax.NamedArray

        def transform(self, x):
            return x + self.w

        def __call__(self, x):
            return self.transform(x)

        @staticmethod
        def init(named):
            return Module(w=named)

    Block = hax.Axis("block", 4)
    E = hax.Axis("E", 6)

    named = hax.random.uniform(jax.random.PRNGKey(0), (Block, E))
    m = Stacked.init(Block, Module)(named=named)

    x = hax.random.uniform(jax.random.PRNGKey(1), (E,))

    # Test vmap_via
    outs_via = m.vmap_via(Module.transform)(x)

    # Test direct vmap
    outs_direct = m.vmap(x)

    assert hax.all(hax.isclose(outs_via, outs_direct))
