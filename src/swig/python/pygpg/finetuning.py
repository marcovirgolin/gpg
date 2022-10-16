from pygpg import conversion as C
import numpy as np
import torch
from copy import deepcopy
import re


def finetune(sympy_model, X, y, learning_rate=1, n_steps=1000, 
  tol_grad=1e-7, tol_change=1e-9):

  best_torch_model, best_loss = None, np.infty
  if not isinstance(X, torch.TensorType):
      X = torch.tensor(X)
  if not isinstance(y, torch.TensorType):
      y = torch.tensor(y) 

  expr_vars = set(re.findall(r'\bx_[0-9]+', str(sympy_model)))
  torch_model = C.sympy_to_torch(sympy_model)
  x_args = {x: X[:, int(x.lstrip("x_"))] for x in expr_vars}
  optimizer = torch.optim.LBFGS(
    torch_model.parameters(), 
    line_search_fn=None,
    lr=learning_rate,
    tolerance_grad=tol_grad, 
    tolerance_change=tol_change)

  prev_loss = np.infty
  for _ in range(n_steps):
      optimizer.zero_grad()
      p = torch_model(**x_args).squeeze(-1)
      loss = (p-y).pow(2).mean().div(2)
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
  return result
