from pygpg import conversion as C
import numpy as np
import torch
import sympy
from copy import deepcopy
import re


def finetune(sympy_model, X, y, learning_rate=1.0, n_steps=100, 
  tol_grad=1e-9, tol_change=1e-9, batch_size=None):

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
    torch_model = C.sympy_to_torch(sympy_model)
  except TypeError:
    print("[!] Wearning: invalid conversion from sympy to torch pre fine-tuning")
    return sympy_model
  x_args = {x: X[:, int(x.lstrip("x_"))] for x in expr_vars}

  try: # optimizer might get an empty parameter list
    optimizer = torch.optim.LBFGS(
      torch_model.parameters(), 
      line_search_fn=None,
      lr=learning_rate,
      tolerance_grad=tol_grad, 
      tolerance_change=tol_change)
  except ValueError:
    return sympy_model

  prev_loss = np.infty
  batch_idx = 0
  batch_x = x_args
  batch_y = y
  for _ in range(n_steps):
    optimizer.zero_grad()
    permutation = torch.randperm(X.size()[0])
    if batch_size is not None:
      indices = permutation[batch_idx*batch_size:(batch_idx+1)*batch_size]
      batch_x = {x: x_args[x][indices] for x in expr_vars}
      batch_y = y[indices]
      batch_idx += 1
    try:
      p = torch_model(**batch_x).squeeze(-1)
    except TypeError:
      print("[!] Warning: error during forward call of torch model while fine-tuning")
      return sympy_model
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
  result = C.timed_simplify(result)
  return result
