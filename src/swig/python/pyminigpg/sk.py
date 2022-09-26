from pyminigpg import pyface as _f
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'pyminigpg'))

class GPGRegressor():

  def __init__(self, verbose=True, **kwargs):
    s = ""
    for k in kwargs:
      if type(kwargs[k]) == bool:
        print(k)
        if kwargs[k] == True:
          s += f" -{k}"
        else:
          s += f" -{k} 0"
      else:
        s += f" -{k} {kwargs[k]}"
    s = s[1:] + " -lib"
    if verbose:
      s += " -verbose"
      
    _f.setup(s)


  def fit(self, X, y):
    Xy = np.hstack((X, y.reshape((-1,1))))
    _f.fit(Xy)

  def predict(self, X):
    # add extra dimension to accomodate prediction
    num_obs = len(X)
    X = np.hstack((X, np.zeros(num_obs).reshape((-1,1))))
    _f.predict(X)
    prediction = X[:,X.shape[1]-1]
    return prediction