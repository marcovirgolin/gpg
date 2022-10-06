from pygpg import conversion as C
from scipy.optimize import minimize
from sympy import preorder_traversal, Float, Symbol
import numpy as np
from functorch import grad
from functools import partial
import torch
import time

class TimedFun:
  def __init__(self, fun, verbose=False, stop_after=3):
    self.fun_in = fun
    self.started = False
    self.stop_after = stop_after
    self.best_fun_value = np.infty
    self.best_x = None
    self.loss_history=[]
    self.verbose = verbose

  def fun(self, x, *args):
    if self.started is False:
      self.started = time.time()
    elif abs(time.time() - self.started) >= self.stop_after:
      self.loss_history.append(self.best_fun_value)
      raise ValueError("Time is over.")
    self.fun_value = self.fun_in(x, *args)
    self.loss_history.append(self.fun_value)
    if self.best_x is None:
      self.best_x=x
    elif self.fun_value < self.best_fun_value:
      self.best_fun_value=self.fun_value
      self.best_x=x
    self.x = x
    return self.fun_value


def optimize(c_sympy_model, init_coeffs, X : np.array, y: np.array, optimizer="bfgs", max_runtime=10, dtype=torch.float64):
  torch_model = C.sympy_to_torch(c_sympy_model, dtype=dtype)
  X=torch.tensor(X, dtype=dtype, requires_grad=False)
  y=torch.tensor(y, dtype=dtype, requires_grad=False)
  func = partial(torch_model, X)

  def torch_objective(coeffs):
    if not isinstance(coeffs, torch.Tensor):
      coeffs = torch.tensor(coeffs, dtype=dtype, requires_grad=True)
    p = func(coeffs)
    print("p:",p[:10])
    print(y[:10])
    
    mse = (y - p).square().mean()
    print(mse)
    print(np.mean(np.square(y.detach().numpy()-p.detach().numpy())))
    quit()

    return mse

  def np_objective(coeffs):
    return torch_objective(coeffs).item()

  def np_gradient(coeffs):
    if not isinstance(coeffs, torch.Tensor):
      coeffs = torch.tensor(coeffs, dtype=dtype, requires_grad=True)
    grad_obj = grad(torch_objective)(coeffs)
    return grad_obj.detach().numpy()

  print("\n\nStart obj", np_objective(init_coeffs), "for init coeffs", init_coeffs)
  timed_np_obj = TimedFun(np_objective, stop_after=max_runtime)
  minimize(timed_np_obj.fun, init_coeffs, method=optimizer, jac=np_gradient, options={"disp":True})
  tuned_coeffs = timed_np_obj.best_x

  return tuned_coeffs

def prep_coeffs(sympy_model):
  new_model = sympy_model
  coeffs = list()
  c_id = 0
  for n in preorder_traversal(sympy_model):
    if isinstance(n, Float):
      new_model = new_model.subs(n, f"C_{c_id}")
      coeffs.append(float(n))
      c_id += 1
  return new_model, coeffs

def insert_coeffs(c_sympy_model, coeffs):
  new_model = c_sympy_model
  for n in preorder_traversal(c_sympy_model):
    if isinstance(n, Symbol) and str(n).startswith("C_"):
      c_id = int(str(n).split("_")[1])
      new_model = new_model.subs(n, Float(coeffs[c_id]))
  return new_model

def finetune(sympy_model, X, y, tuner='bfgs'):
  c_sympy_model, coeffs = prep_coeffs(sympy_model)
  print(c_sympy_model)
  # runs and returns optimized sympy_model
  tuned_coeffs = optimize(c_sympy_model, coeffs, X, y, tuner)
  new_model = insert_coeffs(c_sympy_model, tuned_coeffs)
  print(coeffs, "->", tuned_coeffs)
  return new_model

