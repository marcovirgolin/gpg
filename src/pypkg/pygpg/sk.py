from pygpg import conversion, imputing, complexity
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
import sys, os
import inspect
import numpy as np
import sympy

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'pygpg'))

# load cpp interface
import _pb_gpg

class GPGRegressor(BaseEstimator, RegressorMixin):

  def __init__(self, **kwargs):
    # store parameters internally
    for k in kwargs:
      setattr(self, k, kwargs[k])

  #def __del__(self):
  #  if hasattr(self, "_gpg_cpp"):
  #    del self._gpg_cpp

  def _create_cpp_option_string(self):
    # build string of options for cpp
    kwargs = self.get_params()
    s = ""
    for k in kwargs:
      # skip python-only params
      if k == "_pb_gpg":
        print(k)
      if k in ["finetune", "model", "_pb_gpg"]:
        continue

      # handle bool flags for c++ 
      if type(kwargs[k]) == bool:
        if kwargs[k] == True:
          s += f" -{k}"
      else:
        s += f" -{k} {kwargs[k]}"

    # add "lib" flag to differntiate from CLI calls
    s = s[1:] + " -lib"

    # init cpp interface object
    # & pass options for internal setup
    return s


  def get_params(self, deep=True):
    attributes = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
    attributes = [x for x in attributes if not 
      ((x[0].startswith("__") and x[0].endswith("__")) or 
      x[0] in ["_estimator_type", "_gpg_cpp"])]

    dic = {}
    for a in attributes:
      dic[a[0]] = a[1]

    return dic

  def set_params(self, **parameters):
    for parameter, value in parameters.items():
      setattr(self, parameter, value)

    return self


  def fit(self, X, y):
    # setup cpp interface
    cpp_options = self._create_cpp_option_string()

    # impute if needed
    if np.isnan(X).any():
      self.imputer, X = imputing.fit_and_apply_imputation(X)
      # fix non-contiguous memory block for SWIG
      X = X.copy()

    models = _pb_gpg.evolve(cpp_options, X, y)

    # extract the model as a sympy and store it internally
    self.model = self._pick_best_model(X, y, models)
    

  def _pick_best_model(self, X, y, models):
    # simplify (with stopping)
    if hasattr(self, "verbose") and self.verbose:
      print(f"simplifying {len(models)} models...")
    simplified_models = list()
    for m in models:
      simpl_m = conversion.timed_simplify(m, ratio=1.0, timeout=5)
      if simpl_m is None:
        simpl_m = sympy.sympify(m) # do not simplify, just sympify
      simplified_models.append(simpl_m)
    # proceed with simplified models  
    models = simplified_models

    # cleanup
    models = [conversion.model_cleanup(m, timeout=5) for m in models]
    models = [m for m in models if m is not None]

    # finetune  
    if hasattr(self, "finetune") and self.finetune:
      import finetuning as ft
      if hasattr(self, "verbose") and self.verbose:
        print(f"finetuning {len(models)} models...")

      if len(X) > 10000:
        print("[!] Warning: finetuning on large datasets (>10,000 obs.) can be slow, skipping...")
      else:
        models = [ft.finetune(m, X, y) for m in models]

    # suppose all models failed
    if len(models) == 0:
      # then we return a constant model, the mean over the training data
      models = [sympy.sympify(np.mean(y))]

    # pick best
    errs = list()
    max_err = 0
    for i, m in enumerate(models):
      p = self.predict(X, model=m)
      if np.isnan(p).any():
        # convert this model to a constant, i.e., the mean over the training y
        models[i] = sympy.sympify(np.mean(y))
        p = np.array([np.mean(y)]*len(y))
      err = mean_squared_error(y, p)
      if err > max_err:
        max_err = err
      errs.append(err)
    # adjust errs
    errs = [err if not np.isnan(err) else max_err + 1e-6 for err in errs]

    if hasattr(self, "rci") and len(models) > 1:
      complexity_metric = "component_count" if not hasattr(self, "compl") else self.compl
      compls = [complexity.compute_complexity(m, complexity_metric) for m in models]
      best_idx = complexity.determine_rci_best(errs, compls, self.rci)
    else:
      best_idx = np.argmin(errs)
    
    return models[best_idx]

    
  def predict(self, X, model=None):
    if model is None:
      # assume implicitly wanted the best one found at fit
      model = self.model

    # deal with a model that was simplified to a simple constant
    if type(model) == sympy.Float or type(model) == sympy.Integer:
      prediction = np.array([float(model)]*X.shape[0])
      return prediction

    f = conversion.sympy_to_numpy_fn(model, timeout=5)
    if f is None:
      print("[!] Warning: failed to convert sympy model to numpy, returning NaN as prediction")
      return float("nan")

    if np.isnan(X).any():
      assert(hasattr(self, "imputer"))
      X = self.imputer.transform(X)

    try:
      prediction = f(X)
    except:
      print("[!] Warning: failed to evaluate sympy model, returning NaN as prediction")
      return float("nan")
    
    # can still happen for certain classes of sympy 
    # (e.g., sympy.core.numbers.Zero)
    if type(prediction) in [int, float, np.int64, np.float64]:
      prediction = np.array([float(prediction)]*X.shape[0])
    if len(prediction) != X.shape[0]:
      prediction = np.array([prediction[0]]*X.shape[0])

    return prediction