from pygpg import conversion as C
import numpy as np
import torch
import sympy
from copy import deepcopy
import re

"""
Fine-tunes a sympy model. Returns the fine-tuned model and the number of steps used.
If it terminates prematurely, the number of steps used is returned as well.
"""
def finetune(sympy_model, X, y, learning_rate=1.0, n_steps=100, 
  tol_grad=1e-9, tol_change=1e-9):
  
  best_torch_model, best_loss = None, np.infty

  if not isinstance(X, torch.TensorType):
      X = torch.tensor(X)
  if not isinstance(y, torch.TensorType):
      y = torch.tensor(y.reshape((-1,))) 

  # workaround to have identical constants be treated as different ones
  str_model = str(sympy_model)
  sympy_model = sympy.sympify(str(sympy_model))
  for el in sympy.preorder_traversal(sympy_model):
    if isinstance(el, sympy.Float):
      f = float(el)
      str_model = str_model.replace(str(f), str(f+np.random.normal(0,1e-5)), 1)
  sympy_model = sympy.sympify(str_model)

  expr_vars = set(re.findall(r'\bx_[0-9]+', str(sympy_model)))
  try:
    torch_model = C.sympy_to_torch(sympy_model, timeout=5)
  except TypeError:
    print("[!] Warning: invalid conversion from sympy to torch pre fine-tuning")
    return sympy_model, 0
  if torch_model is None:
    print("[!] Warning: failed to convert from sympy to torch within a reasonable time")
    return sympy_model, 0

  x_args = {x: X[:, int(x.lstrip("x_"))] for x in expr_vars}

  try: # optimizer might get an empty parameter list
    optimizer = torch.optim.LBFGS(
      torch_model.parameters(), 
      line_search_fn=None,
      lr=learning_rate,
      tolerance_grad=tol_grad, 
      tolerance_change=tol_change)
  except ValueError:
    return sympy_model, 0

  prev_loss = np.infty
  batch_x = x_args
  batch_y = y
  steps_done = 0
  for _ in range(n_steps):
    steps_done += 1
    optimizer.zero_grad()
    try:
      p = torch_model(**batch_x).squeeze(-1)
    except TypeError:
      print("[!] Warning: error during forward call of torch model while fine-tuning")
      return sympy_model, steps_done
    loss = (p-batch_y).pow(2).mean().div(2)
    loss.retain_grad()
    loss.backward()
    optimizer.step(lambda: loss)
    loss_val = loss.item()
    if loss_val < best_loss:
      best_torch_model = deepcopy(torch_model)
      best_loss = loss_val
    if abs(loss_val - prev_loss) < tol_change:
      break
    prev_loss = loss_val
  
  result = best_torch_model.sympy()[0] if best_torch_model else sympy_model
  result = C.timed_simplify(result, timeout=5)
  if result is None:
    return sympy_model, steps_done
  return result, steps_done
