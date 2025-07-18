from enum import Enum
import math
import prickle
import pytest
import numpy as np

from common import *

np.random.seed(1234)

@prickle.jit
def prickle_add(dst, a, b):
    with prickle.gpu:
        dst[0] = a[0] + b[0]

@prickle.jit
def prickle_sub(dst, a, b):
    with prickle.gpu:
        dst[0] = a[0] - b[0]

@prickle.jit
def prickle_mul(dst, a, b):
    with prickle.gpu:
        dst[0] = a[0] * b[0]

@prickle.jit
def prickle_div_int(dst, a, b):
    with prickle.gpu:
        dst[0] = a[0] // b[0]

@prickle.jit
def prickle_div(dst, a, b):
    with prickle.gpu:
        dst[0] = a[0] / b[0]

@prickle.jit
def prickle_rem(dst, a, b):
    with prickle.gpu:
        dst[0] = a[0] % b[0]

@prickle.jit
def prickle_pow(dst, a, b):
    with prickle.gpu:
        dst[0] = a[0] ** b[0]

@prickle.jit
def prickle_abs(dst, a, b):
    with prickle.gpu:
        dst[0] = abs(a[0]) + abs(b[0])

@prickle.jit
def prickle_bit_and(dst, a, b):
    with prickle.gpu:
        dst[0] = a[0] & b[0]

@prickle.jit
def prickle_bit_or(dst, a, b):
    with prickle.gpu:
        dst[0] = a[0] | b[0]

@prickle.jit
def prickle_bit_xor(dst, a, b):
    with prickle.gpu:
        dst[0] = a[0] ^ b[0]

@prickle.jit
def prickle_bit_shl(dst, a, b):
    with prickle.gpu:
        dst[0] = a[0] << b[0]

@prickle.jit
def prickle_bit_shr(dst, a, b):
    with prickle.gpu:
        dst[0] = a[0] >> b[0]

@prickle.jit
def prickle_aug_ops(dst, a, b):
    with prickle.gpu:
        dst[0] += a[0]
        dst[0] -= b[0]

@prickle.jit
def prickle_max(dst, a, b):
    with prickle.gpu:
        dst[0] = prickle.max(a[0], b[0])

@prickle.jit
def prickle_min(dst, a, b):
    with prickle.gpu:
        dst[0] = prickle.min(a[0], b[0])

def arith_binop_dtype(fn, ldtype, rdtype, compile_only, backend):
    a = np.random.randint(1, 10, (1,)).astype(ldtype)
    b = np.random.randint(1, 10, (1,)).astype(rdtype)
    dst = np.zeros((1,), dtype=rdtype)
    if compile_only:
        s = prickle.print_compiled(fn, [dst, a, b], par_opts(backend, {}))
        assert len(s) != 0
    else:
        dst_device = np.zeros_like(dst)
        fn(dst_device, a, b, opts=par_opts(backend, {}))
        fn(dst, a, b, opts=seq_opts(backend))
        assert np.allclose(dst, dst_device, atol=1e-5)

bitwise_funs = [
    prickle_bit_and, prickle_bit_or, prickle_bit_xor, prickle_bit_shl, prickle_bit_shr
]
arith_funs = [
    prickle_add, prickle_sub, prickle_mul, prickle_div_int, prickle_div, prickle_rem,
    prickle_pow, prickle_abs, prickle_aug_ops, prickle_max, prickle_min
] + bitwise_funs
signed_int_tys = [np.int8, np.int16, np.int32, np.int64]
unsigned_int_tys = [np.uint8, np.uint16, np.uint32, np.uint64]
float_tys = [np.float16, np.float32, np.float64]
arith_tys = signed_int_tys + unsigned_int_tys + float_tys

def is_float_dtype(dtype):
    return dtype in float_tys

def is_untyped_dtype(dtype):
    return dtype in unsigned_int_tys

def is_invalid_div_or_rem_call(fn, ldtype, rdtype):
    return ((fn.__name__ == "prickle_div_int" or fn.__name__ == "prickle_rem") and
        (is_float_dtype(ldtype) or is_float_dtype(rdtype)))

# There is no 'pow' implementation for 16-bit floats in CUDA
def is_invalid_pow_call(fn, ldtype, rdtype):
    return (fn.__name__ == "prickle_pow" and
        ((not is_float_dtype(ldtype) and not is_float_dtype(rdtype)) or
        (ldtype == np.float16 and rdtype == np.float16)))

# When subtraction of untyped integers overflows, we get a warning that causes
# tests to fail. As we pick random values, we do not know until the numbers
# have been picked whether the test should fail or pass.
def is_untyped_subtraction(fn, ldtype, rdtype):
    return ((fn.__name__ == "prickle_sub" or fn.__name__ == "prickle_aug_ops") and
        (is_untyped_dtype(ldtype) and is_untyped_dtype(rdtype)))

class RunType(Enum):
    ShouldPass = 0
    ShouldFail = 1
    Skip = 2

def set_expected_behavior_binop(fn, ldtype, rdtype, backend):
    if is_invalid_div_or_rem_call(fn, ldtype, rdtype):
        return RunType.ShouldFail, r"Invalid type .* of integer arithmetic operation"
    elif is_invalid_pow_call(fn, ldtype, rdtype):
        return RunType.ShouldFail, r"Invalid type .* of floating-point arithmetic operation"
    elif fn in bitwise_funs and (is_float_dtype(ldtype) or is_float_dtype(rdtype)):
        return RunType.ShouldFail, r"Invalid type .* of bitwise operation"
    elif backend == prickle.CompileBackend.Metal and \
         (ldtype == np.float64 or rdtype == np.float64):
        return RunType.ShouldFail, r"Metal does not support double-precision floating-point numbers."
    elif is_untyped_subtraction(fn, ldtype, rdtype):
        return RunType.Skip, None
    else:
        return RunType.ShouldPass, None

def bin_arith_helper(fn, ldtype, rdtype, compile_only, backend):
    rt, msg_regex = set_expected_behavior_binop(fn, ldtype, rdtype, backend)
    if rt == RunType.ShouldPass:
        arith_binop_dtype(fn, ldtype, rdtype, compile_only, backend)
    elif rt == RunType.ShouldFail:
        with pytest.raises(TypeError) as e_info:
            arith_binop_dtype(fn, ldtype, rdtype, compile_only, backend)
        assert e_info.match(msg_regex)
    else:
        pass

@pytest.mark.parametrize('fn', arith_funs)
@pytest.mark.parametrize('dtype', arith_tys)
@pytest.mark.parametrize('backend', compiler_backends)
def test_bin_arith(fn, dtype, backend):
    run_if_backend_is_enabled(
        backend,
        lambda: bin_arith_helper(fn, dtype, dtype, False, backend)
    )

@pytest.mark.parametrize('fn', arith_funs)
@pytest.mark.parametrize('dtype', arith_tys)
@pytest.mark.parametrize('backend', compiler_backends)
def test_bin_arith_compile(fn, dtype, backend):
    bin_arith_helper(fn, dtype, dtype, True, backend)

# All allowed pairs of types in arithmetic operations, where the LHS should be
# coerced to the RHS type.
arith_ty_pairs = [
    (np.int8, np.int16),
    (np.int8, np.int32),
    (np.int8, np.int64),
    (np.int16, np.int32),
    (np.int16, np.int64),
    (np.int32, np.int64),
    (np.uint8, np.uint16),
    (np.uint8, np.uint32),
    (np.uint8, np.uint64),
    (np.uint16, np.uint32),
    (np.uint16, np.uint64),
    (np.uint32, np.uint64),
    (np.float16, np.float32),
    (np.float16, np.float64),
    (np.float32, np.float64),
]

@pytest.mark.parametrize('fn', arith_funs)
@pytest.mark.parametrize('dtypes', arith_ty_pairs)
@pytest.mark.parametrize('backend', compiler_backends)
def test_bin_arith_mixed_types(fn, dtypes, backend):
    run_if_backend_is_enabled(
        backend,
        lambda: bin_arith_helper(fn, dtypes[0], dtypes[1], False, backend)
    )

@prickle.jit
def prickle_cos(dst, src):
    with prickle.gpu:
        dst[0] = prickle.cos(src[0])

@prickle.jit
def prickle_sin(dst, src):
    with prickle.gpu:
        dst[0] = prickle.sin(src[0])

@prickle.jit
def prickle_tanh(dst, src):
    with prickle.gpu:
        dst[0] = prickle.tanh(src[0])

@prickle.jit
def prickle_atan2(dst, src):
    with prickle.gpu:
        dst[0] = prickle.atan2(src[0], src[0])

@prickle.jit
def prickle_sqrt(dst, src):
    with prickle.gpu:
        dst[0] = prickle.sqrt(src[0])

def arith_unop_dtype(fn, dtype, compile_only, backend):
    src = np.array([0.5], dtype=dtype)
    dst = np.zeros_like(src)
    if compile_only:
        s = prickle.print_compiled(fn, [src, dst], par_opts(backend, {}))
        assert len(s) != 0
    else:
        dst_device = np.zeros_like(dst)
        fn(dst_device, src, opts=par_opts(backend, {}))
        fn(dst, src, opts=seq_opts(backend))
        assert np.allclose(dst, dst_device, atol=1e-5)

float_funs = [prickle_cos, prickle_sin, prickle_tanh, prickle_atan2, prickle_sqrt]
float_tys = [np.float16, np.float32, np.float64]

def set_expected_behavior_unop(fn, dtype, backend):
    if backend == prickle.CompileBackend.Cuda:
        if fn.__name__ == "prickle_tanh" and dtype == np.float16:
            return RunType.ShouldFail, r"Operation tanh not supported for 16-bit floats.*"
        elif fn.__name__ == "prickle_atan2" and dtype != np.float64:
            return RunType.ShouldFail, r"Operation atan2 is only supported for 64-bit floats.*"
    elif backend == prickle.CompileBackend.Metal:
        if dtype == np.float64:
            return RunType.ShouldFail, r"Metal does not support double-precision floating-point numbers."
    return RunType.ShouldPass, None

def float_unop_helper(fn, dtype, compile_only, backend):
    rt, msg_regex = set_expected_behavior_unop(fn, dtype, backend)
    if rt == RunType.ShouldPass:
        arith_unop_dtype(fn, dtype, compile_only, backend)
    elif rt == RunType.ShouldFail:
        with pytest.raises(TypeError) as e_info:
            arith_unop_dtype(fn, dtype, compile_only, backend)
        assert e_info.match(msg_regex)
    else:
        pass

@pytest.mark.parametrize('fn', float_funs)
@pytest.mark.parametrize('dtype', float_tys)
@pytest.mark.parametrize('backend', compiler_backends)
def test_float_unop_arith(fn, dtype, backend):
    run_if_backend_is_enabled(
        backend,
        lambda: float_unop_helper(fn, dtype, False, backend)
    )

@pytest.mark.parametrize('fn', float_funs)
@pytest.mark.parametrize('dtype', float_tys)
@pytest.mark.parametrize('backend', compiler_backends)
def test_float_unop_arith_compile(fn, dtype, backend):
    float_unop_helper(fn, dtype, True, backend)
