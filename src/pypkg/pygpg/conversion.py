import numpy as np
import sympy
import sympytorch
from functools import partial
from sympy.printing.lambdarepr import NumPyPrinter

from ._timeout import threading_timeoutable as timeoutable


@timeoutable()
def timed_simplify(model, ratio=1.0):
    return sympy.simplify(model, ratio=ratio)


@timeoutable()
def model_cleanup(sympy_model):
    # replace Pi, torch cannot convert
    sympy_model = sympy_model.subs(sympy.pi, sympy.Float(3.14159265))
    # replace bad symbols (with 1.0, an arbitrary number)
    s_inf = sympy.Symbol("inf")
    sympy_model = sympy_model.subs(s_inf, sympy.Float(1.0))
    sympy_model = sympy_model.replace(sympy.zoo, sympy.Float(1.0))
    sympy_model = sympy_model.subs(sympy.conjugate, sympy.Float(1.0))
    sympy_model = sympy_model.subs(sympy.AccumBounds, sympy.Float(1.0))
    sympy_model = sympy_model.subs(sympy.StrictLessThan, sympy.Float(1.0))
    sympy_model = sympy_model.subs(sympy.StrictGreaterThan, sympy.Float(1.0))
    sympy_model = sympy_model.replace(sympy.oo, sympy.Float(1.0))
    sympy_model = sympy_model.replace(sympy.S.Infinity, sympy.Float(1.0))
    sympy_model = sympy_model.replace(
        sympy.S.NegativeInfinity,
        sympy.Float(1.0),
    )
    sympy_model = sympy_model.replace(sympy.I, sympy.Float(1.0))
    sympy_model = sympy_model.replace(sympy.nan, sympy.Float(1.0))
    return sympy_model


# Adapted from Pierre-Alexandre Kamienny's symbolic regression helpers.


@timeoutable()
def sympy_to_torch(sympy_model, partialize=False):
    torch_model = sympytorch.SymPyModule(expressions=[sympy_model])

    if not partialize:
        return torch_model

    def wrapper_fn(_torch_model, x, constants=None):
        local_dict = {}
        for d in range(x.shape[1]):
            local_dict[f"x_{d}"] = x[:, d]
        if constants is not None:
            for d in range(constants.shape[0]):
                local_dict[f"C_{d}"] = constants[d]
        return _torch_model(**local_dict)

    return partial(wrapper_fn, torch_model)


class MyNumPyPrinter(NumPyPrinter):
    def _print_Max(self, expr):
        # find the constant
        idx_c = -1
        for i, arg in enumerate(expr.args):
            try:
                _ = float(arg)
                idx_c = i
                break
            except ValueError:
                continue
        assert idx_c != -1
        idx_other = 1 - idx_c
        result = "clip({}, {}, numpy.inf)".format(
            self._print(expr.args[idx_other]),
            self._print(expr.args[idx_c]),
        )
        return result


@timeoutable()
def sympy_to_numpy_fn(sympy_model):
    def wrapper_fn(_sympy_model, X, extra_local_dict=None):
        if extra_local_dict is None:
            extra_local_dict = {}

        local_dict = {}
        for d in range(X.shape[1]):
            local_dict[f"x_{d}"] = X[:, d]
        local_dict.update(extra_local_dict)
        variables = [f"x_{d}" for d in range(X.shape[1])]
        variables_symbols = sympy.symbols(" ".join(variables))
        extra_symbols = list(extra_local_dict.keys())
        if len(extra_symbols) > 0:
            extra_symbols = sympy.symbols(" ".join(extra_symbols))
        else:
            extra_symbols = ()

        np_fn = sympy.lambdify(
            (*variables_symbols, *extra_symbols),
            _sympy_model,
            modules=[{"clip": np.clip}, "numpy"],
            printer=MyNumPyPrinter,
        )
        return np_fn(**local_dict)

    return partial(wrapper_fn, sympy_model)
