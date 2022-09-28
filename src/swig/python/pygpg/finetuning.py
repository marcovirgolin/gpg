from pygpg import conversion as C


def bfgs():
  return None

def finetune(sympy_model, tuner='bfgs'):
  torchified = C.sympy_to_torch(sympy_model)
  # runs and returns optimized sympy_model
  if tuner == "bfgs":
    result = bfgs()
  # blah
  return sympy_model

