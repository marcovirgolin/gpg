import numpy as np
from pygpg.sk import GPGRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone

np.random.seed(42)


X = np.random.randn(512, 3)*10

def grav_law(X):
  return 6.67 * X[:,0]*X[:,1]/(np.square(X[:,2]))

y = grav_law(X)

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler as SS
from sklearn.model_selection import train_test_split

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)
s = SS()
X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)

g = GPGRegressor(t=-1, g=-1, e=1000000, fset="+,-,*,/,cos,log", 
  rci=0.0, finetune=True, verbose=True, random_state=42)
g.fit(X_train,y_train)
print(g.model)
p = g.predict(X_test)
print(r2_score(y_test, p), mean_squared_error(y_test, p))


"""
m = g.model()
print(m)


import sympy

def _coeffs_to_symbs(model):
  model2 = model
  coeff_init_values = list()
  coeff_symbols = list()
  symb_counter = 0
  for node in sympy.preorder_traversal(model):
      if isinstance(node, sympy.Float):
        symb = "theta_"+str(symb_counter)
        model2 = model2.subs(node, symb)
        coeff_init_values.append(node)
        coeff_symbols.append(symb)
        symb_counter += 1
  return model2, coeff_symbols, coeff_init_values

print(_coeffs_to_symbs(m))

ms, cs, ci = _coeffs_to_symbs(m)


def _get_gradient(model, coeff_symbols):
  for i, symb in enumerate(coeff_symbols):
    deriv = sympy.diff(model, symb)
    print('grad[{}]'.format(i), '=', sympy.ccode(deriv) + ';')

_get_gradient(ms, cs)
g = sympy.lambdify(cs, ms, modules="numpy")
print(g(0.99,0.1))
"""