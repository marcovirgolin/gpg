from pyminigpg import pyface as _f
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'pyminigpg'))

class GPGRegressor():

  def __init__(self):
    _f.setup()
    pass

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