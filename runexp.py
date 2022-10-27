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


BUDGET = 500000
X, y = load_boston(return_X_y=True)


def run_exp(seed, new=False):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
  s = SS()
  X_train = s.fit_transform(X_train)
  X_test = s.transform(X_test)
  y_train = s.fit_transform(y_train.reshape((-1,1)))
  y_test = s.transform(y_test.reshape((-1,1)))

  if new:
    #g = GPG1(time=-1, generations=-1, evaluations=BUDGET,
    #  functions="+_-_*_p/_plog_sqrt", initmaxtreeheight=6,
    #  ims=False, seed=seed, coeffmut="0.1_0.5_1.0_9999", popsize=1024)
    #g = GPG2(t=-1, g=-1, e=BUDGET, fset="+,-,*,/", ff="ac", d=5,
    #  rci=0.0, finetune=True, verbose=False, random_state=seed, cmp=1.0)
    from feyn import QLattice
    ql = QLattice(random_seed=seed)
    Xy = np.concatenate((X_train, y_train), axis=1)
    import pandas as pd
    df = pd.DataFrame(Xy, columns=([f"x_{i}" for i in range(X_train.shape[1])] + ["y"]))
    models = ql.auto_run(df, output_name="y")
    g = models[0]
    X_test = pd.DataFrame(X_test, columns=([f"x_{i}" for i in range(X_train.shape[1])]))
    X_train = df
  else:
    #g = GPG1(time=-1, generations=-1, evaluations=BUDGET,
    #  functions="+_-_*_p/_plog_sqrt", initmaxtreeheight=6,
    #  ims="4_1", seed=seed, coeffmut="0.1_0.5_1.0_9999", popsize=64)
    g = GPG2(t=5, g=-1, e=BUDGET, fset="+,-,*,/,sqrt,log,sin,cos", ff="ac", d=4,
      feat_sel=10, disable_ims=True, pop=1024, nolink=True, tour=4,
      rci=0.0, finetune=True, verbose=True, random_state=seed, cmp=0.0)
    g.fit(X_train,y_train)
  train_err = mean_squared_error(y_train, g.predict(X_train))
  test_err = mean_squared_error(y_test, g.predict(X_test))
  print(seed, new, train_err, test_err)
  return train_err, test_err


N_RUNS = 20
#errs1 = [run_exp(i, True) for i in range(N_RUNS)]
#train_errs1 = [x[0] for x in errs1]
#test_errs1 = [x[1] for x in errs1]

train_feyn = [0.1141255357251836, 0.1862778094561644, 0.16903595042464167, 0.16071080059121243, 0.18130955052537953, 0.1779948894072094, 0.1955346102152719, 0.15292365456243293, 0.16315972632826325, 0.17713748616460706, 0.19476962553868682, 0.1356799356154494, 0.18948335165918276, 0.16500415632714152, 0.13072138855270443, 0.16869228050668314, 0.1780020509873894, 0.16830849199453374, 0.17104688248306274, 0.17171615927789122]
test_feyn = [0.28007780385128367, 0.14367585379335418, 0.19736760597929295, 0.1634082674840623, 0.206352426681073, 0.16226205702800384, 0.20966355295050246, 0.2217937825129509, 0.2201829941129696, 0.17766873801260308, 0.3930206859736083, 0.22163186021936343, 0.202714960500208, 0.1734788610045545, 0.32242821304487945, 0.26591886451091346, 0.1413847616525383, 0.2594779435682536, 0.2226005604361091, 0.2852211011724652]

train_RT = [0.15131220691569258, 0.20377106134128134, 0.1942946567697649, 0.16677998293955382, 0.18545708048476728, 0.20288948978384558, 0.1882305409531301, 0.16024806576937975, 0.16757119418037486, 0.19912523761398085, 0.16982988629044998, 0.15772278939690684, 0.2182756832707645, 0.18432955470178022, 0.13321949333018804, 0.16486399079177466, 0.20649403327552035, 0.19908901841824195, 0.15902058150109719, 0.16711610574902197]
test_RT = [0.314105782015446, 0.1588266794262907, 0.22442234809627326, 0.18432894260758376, 0.233517896612445, 0.22980714452544965, 0.19647437540057855, 0.33380932255073775, 0.3077472992905316, 0.19078581142846493, 0.28545908083030613, 0.3793816963311337, 0.22010592121747424, 0.18469159210394903, 0.340647596739066, 0.2716393774817233, 0.14686976825401352, 0.16879739094538365, 0.19407957954374697, 0.2991357552000382]

train_errs1 = train_RT
test_errs1 = test_RT

errs2 = [run_exp(i, False) for i in range(N_RUNS)]
train_errs2 = [x[0] for x in errs2]
test_errs2 = [x[1] for x in errs2]

from scipy.stats import mannwhitneyu

print("Train")
print(train_errs1)
print(train_errs2)
print(np.median(train_errs1),np.median(train_errs2))
_, p_val = mannwhitneyu(train_errs1, train_errs2)
print(p_val)
print("----")
print("Test")
print(test_errs1)
print(test_errs2)
print(np.median(test_errs1),np.median(test_errs2))
_, p_val = mannwhitneyu(test_errs1, test_errs2)
print(p_val)


