import numpy as np
from pygpg.sk import GPGRegressor as GPG2
from pyGPGOMEA import GPGOMEARegressor as GPG1
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
import sympy as sp

np.random.seed(42)


X = np.random.randn(1024, 3)*10

def grav_law(X):
  return 6.67 * X[:,0]*X[:,1]/(np.square(X[:,2]))

y = grav_law(X)

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler as SS
from sklearn.model_selection import train_test_split


BUDGET = 10000
X, y = load_boston(return_X_y=True)


def run_exp(seed, new=False):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
  s = SS()
  X_train = s.fit_transform(X_train)
  X_test = s.transform(X_test)
  y_train = s.fit_transform(y_train.reshape((-1,1)))
  y_test = s.transform(y_test.reshape((-1,1)))

  if new:
    g = GPG2(t=-1, g=-1, e=BUDGET, fset="+,-,*,/", ff="ac", d=5,
      rci=0.0, finetune=True, verbose=False, random_state=seed, cmp=1.0)
  else:
    g = GPG1(time=-1, generations=-1, evaluations=BUDGET,
      functions="+_-_*_p/", initmaxtreeheight=5,
      ims="4_1", seed=seed, coeffmut="0.1_0.5_1.0_9999", popsize=64)
  g.fit(X_train,y_train)
  #print(g.model if new else sp.simplify(g.get_model().replace("p/", "/")))
  train_err = mean_squared_error(y_train, g.predict(X_train))
  test_err = mean_squared_error(y_test, g.predict(X_test))
  print(seed, new, train_err, test_err)
  return test_err


err_gpg1 = [run_exp(i, True) for i in range(10)]
err_gpg2 = [run_exp(i, False) for i in range(10)]

from scipy.stats import mannwhitneyu

print(np.median(err_gpg1),np.median(err_gpg2))
_, p_val = mannwhitneyu(err_gpg1, err_gpg2)
print(p_val)