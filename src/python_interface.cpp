#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <iostream>

#include "util.hpp"
#include "myeig.hpp"
#include "globals.hpp"
#include "ims.hpp"

namespace py = pybind11; 
using namespace std;

py::list evolve(string options, myeig::Mat &X, myeig::Vec &y) {
  // 1. SETUP
  auto opts = split_string(options, " ");
  int argc = opts.size()+1;
  char * argv[argc];
  string title = "gpg";
  argv[0] = (char*) title.c_str();
  for (int i = 1; i < argc; i++) {
    argv[i] = (char*) opts[i-1].c_str();
  }
  g::read_options(argc, argv);

  // initialize evolution handler 
  IMS * ims = new IMS();

  // set training set
  g::fit_func->set_Xy(X, y);
  // set terminals
  g::set_terminals(g::lib_tset);
  g::apply_feature_selection(g::lib_feat_sel_number);
  g::set_terminal_probabilities(g::lib_tset_probs);
  print("terminal set: ",g::str_terminal_set()," (probs: ",g::lib_tset_probs,")");
  // set batch size
  g::set_batch_size(g::lib_batch_size);
  print("batch size: ", g::batch_size);

  // 2. RUN
  ims->run();

  // 3. OUTPUT
  if (ims->elites_per_complexity.empty()) {
    throw runtime_error("Not models found, something went wrong");
  }
  py::list models;
  for (auto it = ims->elites_per_complexity.begin(); it != ims->elites_per_complexity.end(); it++) {
    string model_repr = it->second->to_infix_notation();
    models.append(model_repr);
  }

  // 4. CLEANUP
  delete ims;

  return models;
}

PYBIND11_MODULE(_pb_gpg, m) {
  m.doc() = "pybind11-based interface for gpg"; // optional module docstring
  m.def("evolve", &evolve, "Runs gpg evolution in C++");
}