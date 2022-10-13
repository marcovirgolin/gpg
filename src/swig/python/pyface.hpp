#include "../../util.hpp"
#include "../../myeig.hpp"
#include "../../globals.hpp"
#include "../../ims.hpp"


using namespace std;

Node * best = NULL;
IMS * ims = NULL;

pair<myeig::Mat, myeig::Vec> _assemble_Xy(double * X_n_y, int n_obs, int n_feats_plus_label) {
  myeig::Mat X(n_obs, n_feats_plus_label-1);
  myeig::Vec y(n_obs);
  for(int i = 0; i < n_obs; i++) {
    double * row = &X_n_y[i*(n_feats_plus_label)];
    for(int j = 0; j < n_feats_plus_label - 1; j++) {
      X(i,j) = (float) row[j];
    }
    y(i) = X_n_y[i*(n_feats_plus_label)+n_feats_plus_label-1];
  }
  return make_pair(X,y);
}

void _include_prediction_back(Vec & prediction, double * X_n_p, int n_feats_plus_one) {
  int n_obs = prediction.size();

  for(int i = 0; i < n_obs; i++) {
    X_n_p[i*(n_feats_plus_one)+n_feats_plus_one-1] = (double) prediction(i);
  }
}

void setup(char * options) {
  string str_options = string(options);
  auto opts = split_string(str_options, " ");
  int argc = opts.size()+1;
  char * argv[argc];
  string title = "minigpg";
  argv[0] = (char*) title.c_str();
  for (int i = 1; i < argc; i++) {
    argv[i] = (char*) opts[i-1].c_str();
  }
  g::read_options(argc, argv);
  ims = new IMS();
}

void fit(double * X_n_y, int n_obs, int n_feats_plus_label) {
  // update training set
  auto Xy = _assemble_Xy(X_n_y, n_obs, n_feats_plus_label);
  auto X = Xy.first;
  auto y = Xy.second;

  g::fit_func->set_Xy(X, y);
  // set terminals
  g::set_terminals(g::lib_tset);
  g::set_terminal_probabilities(g::lib_tset_probs);
  print("terminal set: ",g::str_terminal_set()," (probs: ",g::lib_tset_probs,")");
  // batch size
  if (g::lib_batch_size == "auto") {
    g::batch_size = g::fit_func->X_train.rows();
  } else {
    g::batch_size = stoi(g::lib_batch_size);
  }
  print("batch size: ",g::lib_batch_size);
  // run ims
  ims->run();
}

void predict(double * X_n_p, int n_obs, int n_feats_plus_one) {
  if (ims->elites_per_complexity.empty()) {
    throw runtime_error("Not fitted");
  }
  // assemble Mat
  myeig::Mat X = _assemble_Xy(X_n_p, n_obs, n_feats_plus_one).first;

  // select elite
  Node * elite = ims->select_elite(g::rel_compl_importance);

  // mock prediction
  myeig::Vec prediction = elite->get_output(X);
  _include_prediction_back(prediction, X_n_p, n_feats_plus_one);

  return;
}

void model(char * model_str) {
  if (ims->elites_per_complexity.empty()) {
    throw runtime_error("Not fitted");
  }

  Node * elite = ims->select_elite(g::rel_compl_importance);
  string model_repr = elite->human_repr();
  sprintf(model_str, model_repr.c_str());
  return;
}

void models(char * models_str) {
  if (ims->elites_per_complexity.empty()) {
    throw runtime_error("Not fitted");
  }
  string models_repr = "";
  for (auto it = ims->elites_per_complexity.begin(); it != ims->elites_per_complexity.end(); it++) {
    string model_repr = it->second->human_repr();
    models_repr += model_repr + "\n";
  }
  sprintf(models_str, models_repr.c_str());
  return;
}