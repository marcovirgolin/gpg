import numpy as np
from pygpg.sk import GPGRegressor
from pygpg.complexity import compute_complexity
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

X = np.random.randn(128, 3)*10

def grav_law(X : np.ndarray) -> np.ndarray:
    """Ground-truth function for the gravity law."""
    return 6.67 * X[:,0]*X[:,1]/(np.square(X[:,2])) + np.random.randn(X.shape[0])*0.1 # some noise

y = grav_law(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=RANDOM_SEED)

gpg = GPGRegressor(
  e=50_000,                   # 50,000 evaluations limit for search
  t=-1,                       # no time limit,
  g=-1,                       # no generation limit,
  d=3,                        # maximum tree depth
  finetune=True,              # whether to fine-tune the coefficients after the search
  finetune_max_evals=10_000,  # 10,000 evaluations limit for fine-tuning
  verbose=True,               # print progress
  random_state=RANDOM_SEED,   # for reproducibility
)
#gpg = GPGRegressor(
#  t=10, g=-1, e=100000, disable_ims=False, pop=512,
#  fset="+,-,*,/,sqrt,log,sin,cos", ff="ac",
#  nolink=False, feat_sel=False, no_large_fos=True, bs=512,
#  d=5, rci=0.05, finetune=False, verbose=True, tour=4, random_state=42, cmp=0.0
#)

gpg.fit(X_train,y_train)
print(
  "Best found:",
  gpg.model, 
  "(complexity: {})".format(compute_complexity(gpg.model, complexity_metric="node_count")))
print("Train\t\tR2: {}\t\tMSE: {}".format(
  np.round(r2_score(y_train, gpg.predict(X_train)), 3),
  np.round(mean_squared_error(y_train, gpg.predict(X_train)), 3),
))
print("Test\t\tR2: {}\t\tMSE: {}".format(
  np.round(r2_score(y_test, gpg.predict(X_test)), 3),
  np.round(mean_squared_error(y_test, gpg.predict(X_test)), 3),
))

quit()




from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import GridSearchCV

g = clone(g)
hyper_params = [
    {
      'cmp': (0.1,), 'd': (5,), 'e': (500000,), 'feat_sel': (10,), 'fset': ('+,-,*,/,log,cos,sqrt',), 'g': (-1,), 'random_state': (23654,), 'rci': (0.1,), 't': (7200,)
    },
]
grid_est = GridSearchCV(g, param_grid=hyper_params, cv=100,
                verbose=2, n_jobs=40, scoring='r2', error_score=0.0)


grid_est.fit(X_train, y_train)
p = grid_est.predict(X_test)
print(r2_score(y_test, p), mean_squared_error(y_test, p))


print(g.model)

quit()


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