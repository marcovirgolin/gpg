# gpg
Re-implementation of GP-GOMEA (Python scikit-learn-compatible interface, C++ backend).
This version of the code features only GP-GOMEA and no other algorithms (differently from the [previous repo](https://github.com/marcovirgolin/GP-GOMEA)) and focuses on symbolic regression alone.
Also, this version uses dependencies that are easier and less finicky to install (see [environment.yml](environment.yml)). 

## Differences w.r.t. previous version
This version has some differences compared to the code in the [previous repo](https://github.com/marcovirgolin/GP-GOMEA).
Here's a list:
- Protected operators are not used here (expressions that evaluate to NaN for some training points are assigned a worst-case fitness `INF`)
- Models returned from the C++ code are simplified and (optionally) fine-tuned in Python
- Elite at multiple levels of complexity (expression size) are stored and returned to Python (a "best one" is selected using the `rci` parameter)
- If the IMS is disabled and the population converges before the budget is exhausted, then a new population is started which includes a random elite from those found before

## Results on SRBench
Running this version on SRBench (GP-GOMEAv2) leads to expressions that are as compact but more accurate than those of the original GP-GOMEA.

<img src=pics/srbench.png alt="blackbox_results" width=800px />
<img src=pics/srbench_pareto.png alt="pareto_results" width=800px />

The hyper-parameter options used are:

```python
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

est = GPGR(t=2*60*60, g=-1, e=499500,
        tour=4, d=4,
        disable_ims=True, pop=1024, feat_sel=16,
        no_large_fos=True, no_univ_exc_leaves_fos=False,
        finetune=True, 
        bs=2048,
        fset='+,-,*,/,log,sqrt,sin,cos', 
        cmp=0.0, 
        rci=0.0,
        random_state=0
        )
```

## Branches
- `swig` and `pybind` are the same, with the exception that the first uses SWIG and the second uses pybind to realize the python interface.
- `vector_repr` represents an expression as a vector of strings instead of a tree of nodes. This version may be slightly faster (matters only when the number of observations in the data set is relatively small). However it performs less well on SRBench (unclear why, must be investigated).
