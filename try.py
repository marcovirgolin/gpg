import numpy as np
from pygpg.sk import GPGRegressor
from sklearn.metrics import r2_score

np.random.seed(42)

g = GPGRegressor(pop=1000, g=20, d=2, fit="ac", nolink=False, s=2)


X = np.random.randn(1000, 3)

def grav_law(X):
  return 6.67 * X[:,0]*X[:,1]/(np.square(X[:,2]))

y = grav_law(X)

g.fit(X,y)
p = g.predict(X)

print(r2_score(y, p))
print(np.mean(np.square(y-p)))
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