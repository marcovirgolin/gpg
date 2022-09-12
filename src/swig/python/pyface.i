%module pyface

%{
#define SWIG_FILE_WITH_INIT
#include "pyface.hpp"
%}

%include "numpy.i"

%init %{
  import_array();
%}

%apply ( double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double * X_n_y, int n_obs, int n_feats_plus_label)}
%apply ( double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double * X_n_p, int n_obs, int n_feats_plus_one)}


%include "pyface.hpp"