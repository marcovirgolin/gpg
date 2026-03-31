from . import complexity, conversion, imputing
from importlib import import_module
import numpy as np
import pandas as pd
import sympy
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error


class GPGRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, **kwargs):
        self.imputer = None
        self.model = None
        self.n_features_in_ = None
        self._init_params = dict(kwargs)
        # store parameters internally
        for k in kwargs:
            setattr(self, k, kwargs[k])

    @staticmethod
    def _cpp_module():
        return import_module("pygpg._pb_gpg")

    @staticmethod
    def _coerce_feature_matrix(X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array-like object")
        return X

    @staticmethod
    def _coerce_target_vector(y):
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        y = np.asarray(y).reshape(-1)
        return y

    def _create_cpp_option_string(self):
        # build string of options for cpp
        kwargs = self.get_params()
        s = ""
        for k in kwargs:
            # skip python-only params and internal sklearn/class attributes
            if (
                k in ["finetune", "model", "finetune_max_evals"]
                or k.startswith("_")
            ):
                continue

            # handle bool flags for c++
            if isinstance(kwargs[k], bool):
                if kwargs[k]:
                    s += f" -{k}"
            else:
                s += f" -{k} {kwargs[k]}"

        # add "lib" flag to differntiate from CLI calls
        s = s[1:] + " -lib"

        # init cpp interface object
        # & pass options for internal setup
        return s

    def get_params(self, deep=True):
        _ = deep
        return dict(self._init_params)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            self._init_params[parameter] = value

        return self

    def fit(self, X, y):
        # setup cpp interface
        cpp_options = self._create_cpp_option_string()

        X = self._coerce_feature_matrix(X)
        y = self._coerce_target_vector(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of rows")
        self.n_features_in_ = X.shape[1]

        # impute if needed
        if np.isnan(X).any():
            self.imputer, X = imputing.fit_and_apply_imputation(X)
            # fix non-contiguous memory block for SWIG
            X = X.copy()
        else:
            self.imputer = None

        X = np.ascontiguousarray(X)
        y = np.ascontiguousarray(y)

        models = self._cpp_module().evolve(cpp_options, X, y)

        # extract the model as a sympy and store it internally
        self.model = self._pick_best_model(X, y, models)
        return self

    def _finetune_multiple_models(self, models, X, y):
        from . import finetuning as ft

        if hasattr(self, "verbose") and self.verbose:
            print(f"finetuning {len(models)} models...")

        if len(X) > 10000:
            print(
                "[!] Warning: finetuning on large datasets (>10,000 obs.) "
                "can be slow, skipping..."
            )
        else:
            if hasattr(self, "finetune_max_evals"):
                # Scatter finetuning across models by coefficient count.
                num_coeffs = [
                    complexity.get_num_coefficients(m) for m in models
                ]
                tot = sum(num_coeffs)
                finetune_num_steps = [
                    int(self.finetune_max_evals * (n / tot))
                    for n in num_coeffs
                ]
                while sum(finetune_num_steps) < self.finetune_max_evals:
                    finetune_num_steps[np.random.randint(len(models))] += 1
            else:
                finetune_num_steps = [100] * len(models)

            for i, m in enumerate(models):
                models[i], steps_done = ft.finetune(
                    m,
                    X,
                    y,
                    n_steps=finetune_num_steps[i],
                )
                steps_leftover = finetune_num_steps[i] - steps_done
                # scatter steps left over all models
                if i + 1 < len(models):
                    models_left = len(models) - i - 1
                    steps_left_per_model = int(steps_leftover / models_left)
                    reminder = steps_leftover % models_left
                    for j in range(i + 1, len(models)):
                        finetune_num_steps[j] += steps_left_per_model
                    # and scatter remainder
                    for j in range(reminder):
                        random_index = np.random.randint(i + 1, len(models))
                        finetune_num_steps[random_index] += 1

    def _pick_best_model(self, X, y, models):
        # simplify (with stopping)
        if hasattr(self, "verbose") and self.verbose:
            print(f"simplifying {len(models)} models...")
        simplified_models = list()
        for m in models:
            simpl_m = conversion.timed_simplify(m, ratio=1.0, timeout=5)
            if simpl_m is None:
                simpl_m = sympy.sympify(m)  # do not simplify, just sympify
            simplified_models.append(simpl_m)
        # proceed with simplified models
        models = simplified_models

        # cleanup
        models = [conversion.model_cleanup(m, timeout=5) for m in models]
        models = [m for m in models if m is not None]

        # finetune
        if hasattr(self, "finetune") and self.finetune:
            self._finetune_multiple_models(models, X, y)

        # pick best
        errs = list()
        max_err = 0
        for i, m in enumerate(models):
            p = self.predict(X, model=m)
            if np.isnan(p).any():
                # Convert broken models to the mean target value.
                models[i] = sympy.sympify(np.mean(y))
                p = np.array([np.mean(y)] * len(y))
            err = mean_squared_error(y, p)
            if err > max_err:
                max_err = err
            errs.append(err)
        # adjust errs
        errs = [err if not np.isnan(err) else max_err + 1e-6 for err in errs]

        if hasattr(self, "rci") and len(models) > 1:
            complexity_metric = (
                "node_count" if not hasattr(self, "compl") else self.compl
            )
            compls = [
                complexity.compute_complexity(m, complexity_metric)
                for m in models
            ]
            best_idx = complexity.determine_rci_best(errs, compls, self.rci)
        else:
            best_idx = np.argmin(errs)

        return models[best_idx]

    def predict(self, X, model=None):
        if model is None:
            # assume implicitly wanted the best one found at fit
            model = self.model
            if model is None:
                raise NotFittedError(
                    "This GPGRegressor instance is not fitted yet. "
                    "Call 'fit' before using this estimator."
                )

        X = self._coerce_feature_matrix(X)

        # deal with a model that was simplified to a simple constant
        if isinstance(model, (sympy.Float, sympy.Integer)):
            prediction = np.array([float(model)] * X.shape[0])
            return prediction

        f = conversion.sympy_to_numpy_fn(model, timeout=5)
        if f is None:
            print(
                "[!] Warning: failed to convert sympy model to numpy, "
                "returning NaN as prediction"
            )
            return float("nan")

        if np.isnan(X).any():
            if self.imputer is None:
                raise ValueError(
                    "X contains NaN values, but no imputer was fitted "
                    "during training"
                )
            X = self.imputer.transform(X)

        X = np.ascontiguousarray(X)

        try:
            prediction = f(X)
        except (ArithmeticError, IndexError, KeyError, TypeError, ValueError):
            print(
                "[!] Warning: failed to evaluate sympy model, "
                "returning NaN as prediction"
            )
            return float("nan")

        # can still happen for certain classes of sympy
        # (e.g., sympy.core.numbers.Zero)
        if isinstance(prediction, (int, float, np.integer, np.floating)):
            prediction = np.array([float(prediction)] * X.shape[0])
        if np.ndim(prediction) == 0:
            prediction = np.array([float(prediction)] * X.shape[0])
        elif len(prediction) != X.shape[0]:
            prediction = np.array([prediction[0]] * X.shape[0])

        return prediction
