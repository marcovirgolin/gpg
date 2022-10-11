from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

def fit_and_apply_imputation(X, max_iter=10, random_state=42, sample_posterior=True):
  imp = IterativeImputer(max_iter=max_iter, random_state=random_state, sample_posterior=sample_posterior)
  X = imp.fit_transform(X)
  return imp, X