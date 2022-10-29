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
    g = GPG2(t=10, g=-1, e=BUDGET, fset="+,-,*,/,sin,cos,log,sqrt", ff="ac", d=3,
      feat_sel=10, disable_ims=True, pop=512, nolink=True, tour=4,
      no_large_fos=True, no_univ_exc_leaves_fos=True,
      rci=0.1, finetune=True, verbose=True, random_state=seed, cmp=0.1)
    #g = GPG2(t=3, g=-1, e=499500, tour=4,
    #    disable_ims=True, pop=512, nolink=True,
    #    no_large_fos=True, no_univ_exc_leaves_fos=True,
    #    fset='+,-,*,/,log,sqrt,sin,cos', cmp=0.1, rci=0.1,
    #    random_state=seed)
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

train_RT = [0.13385794179578572, 0.19455830251060613, 0.16205064263165378, 0.17624577477107367, 0.17338649670934628, 0.17697111110530436, 0.16756849082989012, 0.1494228386418852, 0.15719580746436745, 0.158701371308787, 0.15725717175872997, 0.1450574119457481, 0.1900426323547044, 0.16668962751693042, 0.1265065279689986, 0.15324109732661195, 0.17798083705744264, 0.1703114552721424, 0.15500993641618782, 0.16859590008693495]
test_RT = [0.2756704469039293, 0.14819850816075752, 0.2295488052279981, 0.17331047629147542, 0.2186395584690041, 0.15762461515551504, 0.16242520892794504, 0.2988775116981199, 0.23447881250971148, 0.15813552725013588, 0.2428481663939028, 0.2625502379049222, 0.15189421846541673, 0.2717466703736266, 0.33200921539776446, 0.32581470657907685, 0.18380802638541258, 0.2073567802671587, 0.22016878329037184, 0.31134081227309646]
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