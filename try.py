import numpy as np
from pygpg.sk import GPGRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV

np.random.seed(42)


X = np.random.randn(24, 3)*10

def grav_law(X):
  return 6.67 * X[:,0]*X[:,1]/(np.square(X[:,2]))

y = grav_law(X)

#from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler as SS
from sklearn.model_selection import train_test_split

#X, y = load_boston(return_X_y=True)


"""
Load 618_fri_c3_1000_50
"""

hyper_params = [
    { # 1
     'd' : (3,), 'rci' : (0.0,),
    },
    { # 2
     'd' : (4,), 'rci' : (0.0, 0.1),
    },
    { # 2
     'd' : (5,), 'rci' : (0.0, 0.1,),
    },
    { # 1
     'd' : (6,), 'rci' : (0.1,),  'no_univ_exc_leaves_fos' : (True,),
    },
]

# load poker
import pandas as pd
df = pd.read_csv("../pmlb/datasets/618_fri_c3_1000_50/618_fri_c3_1000_50.tsv.gz", compression='gzip', sep='\t')
X = df.drop(columns=['target']).to_numpy()
y = df['target'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23654)
s = SS()
X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)
y_train = s.fit_transform(y_train.reshape((-1,1)))
y_test = s.transform(y_test.reshape((-1,1)))


from sklearn.base import clone

#g = GPGRegressor(t=7200, g=-1, e=500000, disable_ims=True, pop=1024, fset="+,-,*,/,sqrt,log,sin,cos", ff="ac",
#  nolink=False, feat_sel=10, no_large_fos=True, bs=100,
#  d=4, rci=0.0, finetune=False, verbose=True, tour=4, random_state=42, cmp=0.0)

#
g = GPGRegressor(t=7200, g=-1, e=499500, tour=4, d=5,
        disable_ims=True, pop=1024, nolink=True, feat_sel=16,
        no_large_fos=True, no_univ_exc_leaves_fos=False,
        finetune=True, 
        verbose=True,
        bs=2048,
        finetune_max_evals=500,
        oset='+,-,*,/,log,sqrt,sin,cos', cmp=0.1, rci=0.0,
        random_state=23654)  
from sklearn.model_selection import KFold
cv = KFold(n_splits=10, shuffle=True,random_state=23654)

from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV
grid_est = GridSearchCV(g, param_grid=hyper_params, cv=cv, verbose=0, n_jobs=1, scoring='r2', error_score=0.0)
grid_est.fit(X_train, y_train)
g = grid_est.best_estimator_
#g.fit(X_train,y_train)
print(g.model)
p = g.predict(X_test)
print(r2_score(y_train, g.predict(X_train)), mean_squared_error(y_train, g.predict(X_train)))
print(r2_score(y_test, p), mean_squared_error(y_test, p))
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