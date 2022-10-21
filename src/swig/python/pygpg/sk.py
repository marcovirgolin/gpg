from pygpg import pyface as _f
from pygpg import conversion, imputing, complexity
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
import sys, os
import inspect
import numpy as np
import sympy

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'pygpg'))

class GPGRegressor(BaseEstimator, RegressorMixin):

  def __init__(self, **kwargs):
    # store parameters internally
    for k in kwargs:
      setattr(self, k, kwargs[k])

    self.set_params_cpp(kwargs)
  
  def set_params_cpp(self, kwargs):
    # build string
    s = ""
    for k in kwargs:
      # skip python-only params
      if k == "finetune":
        continue

      # handle bool flags for c++ 
      if type(kwargs[k]) == bool:
        if kwargs[k] == True:
          s += f" -{k}"
      else:
        s += f" -{k} {kwargs[k]}"
    # add "lib" flag to differntiate from CLI calls
    s = s[1:] + " -lib"

    # pass string to cpp for internal setup
    _f.setup(s)


  def get_params(self, deep=True):
    attributes = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
    attributes = [a for a in attributes if '_' not in a[0]]

    dic = {}
    for a in attributes:
      dic[a[0]] = a[1]

    return dic

  def set_params(self, **parameters):
    for parameter, value in parameters.items():
      setattr(self, parameter, value)

    self.set_params_cpp(parameters)

    return self


  def fit(self, X, y):
    # impute if needed
    if np.isnan(X).any():
      self.imputer, X = imputing.fit_and_apply_imputation(X)
      # fix non-contiguous memory block for SWIG
      X = X.copy()

    Xy = np.hstack((X, y.reshape((-1,1))))
    _f.fit(Xy)

    # extract the model as a sympy and store it internally
    self.model = self._pick_best_model(X, y)
    

  def _pick_best_model(self, X, y):
    # get all models from cpp
    models = _f.models().split("\n")
    models = [sympy.simplify(m) for m in models]
    # finetune  
    if hasattr(self, "finetune") and self.finetune:
      import finetuning as ft
      models = [ft.finetune(m, X, y) for m in models]

    # pick best
    errs = [mean_squared_error(y, self.predict(X, model=m)) for m in models]
    if hasattr(self, "rci"):
      complexity_metric = "node_count" if not hasattr(self, "compl") else self.compl
      compls = [complexity.compute_complexity(m, complexity_metric) for m in models]
      best_idx = complexity.determine_rci_best(errs, compls, self.rci)
    else:
      best_idx = np.argmin(errs)
    
    return models[best_idx]

    
  def _predict_cpp(self, X):
    num_obs = len(X)
    X_prime = np.hstack((X, np.zeros(num_obs).reshape((-1,1))))
    _f.predict(X_prime)
    prediction = X_prime[:,X_prime.shape[1]-1]
    return prediction

  def predict(self, X, model=None, cpp=False):
    # add extra dimension to accomodate prediction
    if cpp:
      if hasattr(self, "finetune") and self.finetune:
        raise ValueError("Cannot use cpp=True if finetuning has taken place")
      if model is not None:
        raise ValueError("Conflict: called predict from cpp but passing a sympy model")
      return self._predict_cpp(X)


    if model is None:
      # assume implicitly wanted the best one found at fit
      model = self.model

    # deal with a model that was simplified to a simple constant
    if type(model) == sympy.Float:
      prediction = np.array([float(model)]*X.shape[0])
      return prediction

    f = conversion.sympy_to_numpy_fn(model)

    if np.isnan(X).any():
      assert(hasattr(self, "imputer"))
      X = self.imputer.transform(X)

    prediction = f(X)
    return prediction
