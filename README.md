# gpg
Re-implementation of GP-GOMEA (Python scikit-learn-compatible interface, C++ backend).
This version of the code features only GP-GOMEA and no other algorithms (differently from the [previous repo](https://github.com/marcovirgolin/GP-GOMEA)) and focuses on symbolic regression alone.
Also, this version uses dependencies that are easier and less finicky to install (see [environment.yml](environment.yml)).

## Installation
Installation requires [git](https://github.com/git-guides/install-git) and [conda](https://www.anaconda.com/download).
Run the following bash commands from a folder of your choice:
```bash
git clone https://github.com/marcovirgolin/gpg.git
cd gpg
conda env create -f environment.yml
conda activate gpg
make
```

## Usage
You can try `gpg` out with the following code snippet (or simply run `try.py` if you like):
```python
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
  e=50_000,                   # 50,000 evaluations limit
  t=-1,                       # no time limit,
  g=-1,                       # no generation limit,
  d=3,                        # maximum tree depth
  verbose=True,               # print progress
  random_state=RANDOM_SEED,   # for reproducibility
)
gpg.fit(X_train,y_train)

print(
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
```

## Differences w.r.t. previous version
This version has some differences compared to the code in the [previous repo](https://github.com/marcovirgolin/GP-GOMEA).
Here's a list:
- Protected operators are not used here (expressions that evaluate to NaN for some training points are assigned a worst-case fitness `INF`)
- Functions/variables/constants can be sampled with custom probabilities (by default, uniform with binary operators twice as likely as unary operators)
- Tournament selection can be used to speed up convergence within GOM.
- Models returned from the C++ code are simplified and (optionally) fine-tuned in Python
- Elite at multiple levels of complexity (expression size) are stored and returned to Python (a "best one" is selected using the `rci` parameter)
- If the IMS is disabled and the population converges before the budget is exhausted, then a new population is started which includes a random elite from those found before
- A simple feature selection mechanism is included (if desired)
- Models obtained from C++ are converted to `sympy` and can be further processed as such
- The scikit-learn interface includes imputation in case of incomplete data
- The scikit-learn interface includes coefficient fine-tuning with `sympy-torch` and L-BFGS


## Results on SRBench
Running this version on SRBench (`gpg`) leads to expressions that are as compact but more accurate than those of the original `GP-GOMEA`, in much less time!

<img src=pics/srbench.png alt="blackbox_results" width=800px />
<img src=pics/srbench_harmonic.png alt="harmonic_means" width=800px />


## Research
If you use our code for academic purposes, please support our research by citing:
```
@article{virgolin2021improving,
  title={Improving model-based genetic programming for symbolic regression of small expressions},
  author={Virgolin, Marco and Alderliesten, Tanja and Witteveen, Cees and Bosman, Peter A. N.},
  journal={Evolutionary Computation},
  volume={29},
  number={2},
  pages={211--237},
  year={2021},
  publisher={MIT Press}
}
```

## Branches
- `swig` and `pybind` are the same, with the exception that the first uses SWIG and the second uses pybind to realize the python interface. `pybind` is now default and, probably, `swig` will no longer be supported/updated.
- `vector_repr` represents an expression as a vector of strings instead of a tree of nodes. This version may be slightly faster (matters only when the number of observations in the data set is relatively small). However it needs to be [fixed](https://github.com/marcovirgolin/gpg/issues/10).
