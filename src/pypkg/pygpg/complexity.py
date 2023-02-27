# Mirrors complexity.cpp
import numpy as np
import sympy

def compute_complexity(model, complexity_metric="node_count"):
  if complexity_metric == "node_count":
    c = 0
    for _ in sympy.preorder_traversal(model):
      c += 1
    return c
  else:
    raise ValueError("Unrecognized complexity metric", complexity_metric)
  
def get_num_coefficients(model):
  c = 0
  sp_model = sympy.sympify(str(model))
  for el in sympy.preorder_traversal(sp_model):
    if isinstance(el, sympy.Float):
      c += 1
  return c

def determine_rci_best(errors, complexities, rci=0.1) -> int:
  assert(len(errors) == len(complexities))

  min_e, max_e = min(errors), max(errors)
  rang_e = max_e - min_e
  min_c, max_c = min(complexities), max(complexities)
  rang_c = max_c - min_c

  rel_accuracies = [1.0-(e - min_e)/rang_e for e in errors]
  rel_simplicities = [1.0-(c - min_c)/rang_c for c in complexities]

  best_score = -np.inf
  best_idx = None
  for i in range(len(errors)):
    score_i = (1.0 - rci)*rel_accuracies[i] + rci*rel_simplicities[i]
    if score_i > best_score:
      best_score = score_i
      best_idx = i

  return best_idx
    
  
  