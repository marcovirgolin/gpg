from pygpg import pyface as _f
from pygpg import conversion
import sys, os
import numpy as np
import sympy

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'pygpg'))

class GPGRegressor():

  def __init__(self, verbose=True, **kwargs):
    # python-only
    self.finetune = False
    if "finetune" in kwargs:
      if kwargs["finetune"] == True:
        self.finetune = True
      kwargs.remove("finetune")

    # to pass to c++
    s = ""
    for k in kwargs:
      if type(kwargs[k]) == bool:
        if kwargs[k] == True:
          s += f" -{k}"
      else:
        s += f" -{k} {kwargs[k]}"
    s = s[1:] + " -lib"
    if verbose:
      s += " -verbose"
      
    _f.setup(s)


  def fit(self, X, y):
    Xy = np.hstack((X, y.reshape((-1,1))))
    _f.fit(Xy)
    # extract the model as a sympy and store it internally
    self.model = _f.model()
    self.model = sympy.simplify(self.model)
    if self.finetune:
      import finetuning as ft
      self.model = ft.finetune(self.model)
    
  def _predict_cpp(self, X):
    num_obs = len(X)
    X_prime = np.hstack((X, np.zeros(num_obs).reshape((-1,1))))
    _f.predict(X_prime)
    prediction = X_prime[:,X_prime.shape[1]-1]
    return prediction

  def predict(self, X, use_cpp=False):
    # add extra dimension to accomodate prediction
    if use_cpp:
      return self._predict_cpp(X)
    f = conversion.sympy_to_numpy_fn(self.model)
    prediction = f(X)
    return prediction

  def model(self):
    return self.model
    