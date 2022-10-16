# Mirrors complexity.cpp
import numpy as np

def compute_complexity(model, complexity_metric="node_count"):
  if complexity_metric == "node_count":
    return model.count_ops(visual=False)
  else:
    raise ValueError("Unrecognized complexity metric", complexity_metric)

def determine_rci_best(errors, complexities, rci=0.1) -> int:
  assert(len(errors) == len(complexities))

  min_e, max_e = min(errors), max(errors)
  rang_e = max_e - min_e
  min_c, max_c = min(complexities), max(complexities)
  rang_c = max_c - min_c

  rel_accuracies = [(e - min_e)/rang_e for e in errors]
  rel_simplicities = [(c - min_c)/rang_c for c in complexities]

  best_score = -np.inf
  best_idx = None
  for i in range(len(errors)):
    score_i = (1.0 - rci)*rel_accuracies[i] + rci*rel_simplicities[i]
    if score_i > best_score:
      best_score = score_i
      best_idx = i

  return best_idx
    
  
  