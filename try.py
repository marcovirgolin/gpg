import numpy as np
from pyminigpg.sk import GPGRegressor
from sklearn.metrics import r2_score

g = GPGRegressor(pop=1000, g=20, d=2, fit="ac")


X = np.random.randn(1000, 3)

def grav_law(X):
  return 6.67 * X[:,0]*X[:,1]/(np.square(X[:,2]))

y = grav_law(X)

g.fit(X,y)
p = g.predict(X)

print(r2_score(y, p))

m = g.model()
print(m)