import sympy, sympytorch
from functools import partial
from stopit import threading_timeoutable as timeoutable

@timeoutable()
def timed_simplify(model, ratio=1.0):
  return sympy.simplify(model)


"""
Code by Pierre-Alexandre Kamienny
https://github.com/pakamienny/e2e_transformer/blob/a4208974779e2109593297ef1992d0b14dd91b00/symbolicregression/envs/simplifiers.py
"""
def sympy_to_torch(sympy_model, partialize=False):
  torch_model = sympytorch.SymPyModule(expressions=[sympy_model])

  if not partialize:
    return torch_model

  def wrapper_fn(_torch_model, x, constants=None):
    local_dict = {}
    for d in range(x.shape[1]):
      local_dict["x_{}".format(d)]=x[:, d]
    if constants is not None:
      for d in range(constants.shape[0]):
        local_dict["C_{}".format(d)]=constants[d]
    return _torch_model(**local_dict)

  return partial(wrapper_fn, torch_model)

def sympy_to_numpy_fn(sympy_model):
  def wrapper_fn(_sympy_model, X, extra_local_dict={}):
    local_dict = {}
    for d in range(X.shape[1]):
      local_dict["x_{}".format(d)]=X[:, d]
    local_dict.update(extra_local_dict)
    variables_symbols = sympy.symbols(" ".join(["x_{}".format(d) for d in range(X.shape[1])]))
    extra_symbols = list(extra_local_dict.keys())
    if len(extra_symbols)>0:
      extra_symbols = sympy.symbols(" ".join(extra_symbols))
    else:
      extra_symbols=()
    np_fn =  sympy.lambdify((*variables_symbols, *extra_symbols), _sympy_model, modules="numpy")
    return np_fn(**local_dict)
  return partial(wrapper_fn, sympy_model)