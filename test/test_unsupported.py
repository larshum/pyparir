# This file ensures that features that are not yet supported by the compiler
# are correctly reported as errors already when parsing the Python function.

import numpy as np
import prickle
import pytest
import torch

from common import *

def test_while_else_rejected():
    with pytest.raises(RuntimeError) as e_info:
        @prickle.jit
        def while_fun(x, y, N):
            i = 0
            while i < N:
                y[i] = x[i]
                i += 1
            else:
                y[i] = 0.0
    assert e_info.match(r".*lines 16-20.*")

def test_for_else_rejected():
    with pytest.raises(RuntimeError) as e_info:
        @prickle.jit
        def for_else(x, y, N):
            for i in range(N):
                y[i] = x[i]
            else:
                y[0] += 1
    assert e_info.match(r".*lines 27-30.*")

def test_with_unsupported_context():
    with pytest.raises(RuntimeError) as e_info:
        @prickle.jit
        def with_context():
            with 5:
                pass
    assert e_info.match(r".*lines 37-38.*")

def test_with_as():
    with pytest.raises(RuntimeError) as e_info:
        @prickle.jit
        def with_as():
            with prickle.gpu as x:
                a = x + 1
    assert e_info.match(r".*lines 45-46.*")

def test_dict_with_non_string_keys():
    @prickle.jit
    def dict_arg(a):
        with prickle.gpu:
            a["x"] = a["y"]

    with pytest.raises(RuntimeError) as e_info:
        dict_arg({'x': 2, 'y': 4, 3: 5})
    assert e_info.match(r"(.*non-string key.*)|(Found no enabled GPU backends.*)")

def test_dict_with_int_key():
    @prickle.jit
    def dict_arg(a):
        with prickle.gpu:
            a["x"] = a[2]

    with pytest.raises(RuntimeError) as e_info:
        dict_arg({'x': 2, 2: 4})
    assert e_info.match(r"(.*non-string key.*)|(Found no enabled GPU backends.*)")
