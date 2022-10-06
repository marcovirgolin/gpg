import sympy, torch, sympytorch
from functools import partial


"""
Code by Pierre-Alexandre Kamienny
https://github.com/pakamienny/e2e_transformer/blob/a4208974779e2109593297ef1992d0b14dd91b00/symbolicregression/envs/simplifiers.py
"""
def sympy_to_torch(sympy_model, dtype=torch.float32):
  mod = sympytorch.SymPyModule(expressions=[sympy_model])
  mod.to(dtype)
  def wrapper_fn(_mod, x, constants=None):
      local_dict = {}
      for d in range(x.shape[1]):
          local_dict["x_{}".format(d)]=x[:, d]
      if constants is not None:
          for d in range(constants.shape[0]):
              local_dict["C_{}".format(d)]=constants[d]
      return _mod(**local_dict)
  return partial(wrapper_fn, mod)

def sympy_to_numpy_fn(expr):
  def wrapper_fn(_expr, x, extra_local_dict={}):
      local_dict = {}
      for d in range(x.shape[1]):
          local_dict["x_{}".format(d)]=x[:, d]
      local_dict.update(extra_local_dict)
      variables_symbols = sympy.symbols(" ".join(["x_{}".format(d) for d in range(x.shape[1])]))
      extra_symbols = list(extra_local_dict.keys())
      if len(extra_symbols)>0:
          extra_symbols = sympy.symbols(" ".join(extra_symbols))
      else:
          extra_symbols=()
      np_fn =  sympy.lambdify((*variables_symbols, *extra_symbols), _expr, modules="numpy")
      return np_fn(**local_dict)

  return partial(wrapper_fn, expr)