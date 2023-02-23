#ifndef GLOBALS_H
#define GLOBALS_H

#include <random>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <map>
#include "myeig.hpp"
#include "operator.hpp"
#include "fitness.hpp"
#include "cmdparser.hpp"
#include "feature_selection.hpp"
#include "rng.hpp"

using namespace std;
using namespace myeig;

namespace g {

  // ALL fitness functions 
  vector<Fitness*> all_fitness_functions = {
    new MAEFitness(), new MSEFitness(), new AbsCorrFitness()
  };

  // budget
  int pop_size;
  int max_generations;
  int max_time;
  int max_evaluations;
  long long max_component_evaluations;
  bool disable_ims = false;

  // representation
  int max_depth;
  int max_arity = 0; // set automatically once chosen the operators
  string init_strategy;
  vector<string> operators;
  vector<string> terminals;
  Vec cumul_fset_probs;
  Vec cumul_tset_probs;
  string lib_tset; // used when `fit` is called when using as lib
  string lib_tset_probs; // used when `fit` is called when using as lib
  string complexity_type;
  float rel_compl_importance=0.0;
  int lib_feat_sel_number = -1; // used when `fit` is called when using as lib

  // problem
  Fitness * fit_func = NULL;
  string path_to_training_set;
  string lib_batch_size; // used when `fit` is called when using as lib
  int batch_size;

  // variation
  int max_init_attempts = 10000;
  bool no_linkage;
  float cmut_eps;
  float cmut_prob;
  float cmut_temp;
  bool no_large_subsets=false;
  bool no_univariate=false;

  // selection
  int tournament_size;
  bool tournament_stochastic = false;
  bool diversity_promotion = false;

  // other
  int random_state = -1;
  bool verbose = true;
  bool _call_as_lib = false;

  // Functions
  void set_fit_func(string fit_func_name) { 
    bool found = false;
    for (auto * f : all_fitness_functions) {
      if (f->name() == fit_func_name) {
        found = true;
        fit_func = f->clone();
      }
    }
    if (!found) {
      throw runtime_error("Unrecognized fitness function: "+fit_func_name);
    }
  }

  void set_operators(string setting) {
    assert(operators.empty());
    vector<string> desired_funct_symbs = split_string(setting);
    for (string sym : desired_funct_symbs) {
      // check sym is found
      bool found = all_operators.count(sym);
      if (!found) {
        throw runtime_error("Unrecognized function: "+sym);
      }
      operators.push_back(sym);
      int arity = all_operators[sym].second;
      if (arity > max_arity) {
        max_arity = arity;
      }
    }
  }

  Vec _compute_custom_cumul_probs_operator_set(string setting, vector<string> & func_or_term_set) {

    auto str_v = split_string(setting);
    if (str_v.size() != func_or_term_set.size()) {
      throw runtime_error("Size of the probabilities for function or terminal set does not match the size of the respective set: "+setting);
    }

    Vec result(func_or_term_set.size());
    float cumul_prob = 0;
    for (int i = 0; i < str_v.size(); i++) {
      cumul_prob += stof(str_v[i]);
      result[i] = cumul_prob;
    }

    if (abs(1.0-result[result.size()-1]) > 1e-3) {
      throw runtime_error("The probabilties for the respective operator set do not sum to 1: "+setting);
    }

    return result;
  }

  void set_operator_probabilities(string setting) {
      
    if (setting == "auto") {
      // set unary operators to have half the chance other ones (which are normally binary)
      Veci arities(operators.size());
      int num_unary = 0;
      for(int i = 0; i < operators.size(); i++) {
        int arity = all_operators[operators[i]].second;
        arities[i] = arity;
        if (arities[i] == 1) {
          num_unary++;
        }
      }

      int num_other = operators.size() - num_unary;
      float p_unary = 1.0 / (2.0*num_other + num_unary);
      float p_other = 1.0 / (num_other + 0.5*num_unary);

      float cumul_prob = 0;
      cumul_fset_probs = Vec(operators.size());
      for (int i = 0; i < arities.size(); i++) {
        if (arities[i] == 1) {
          cumul_prob += p_unary;
        } else {
          cumul_prob += p_other;
        }
        cumul_fset_probs[i] = cumul_prob;
      }
      return;
    }

    // else, use what provided
    cumul_fset_probs = _compute_custom_cumul_probs_operator_set(setting, operators);
  }

  void set_terminals(string setting) {
    assert(terminals.empty());

    if (setting == "auto") {
      assert(fit_func);
      for(int i = 0; i < fit_func->X_train.cols(); i++) {
        terminals.push_back("x_"+to_string(i));
      }
      terminals.push_back("erc");
    }
    else {
      vector<string> desired_terminal_symbs = split_string(setting);
      // the following checks that the terminal is either a variable, a constant or erc
      // {but could be skipped, and simply do terminals = split_string(setting)}
      for (string sym : desired_terminal_symbs) {
        try {
          if (sym.size() > 2 && sym.substr(0, 2) == "x_") {
            // variable
            int i = stoi(sym.substr(2,sym.size()));
            if (i < 0) {
              throw runtime_error("Variable index must be non-negative: "+sym);
            }
            else if (i > fit_func->X_train.cols()-1) {
              throw runtime_error("Variable index is too large: "+sym);
            }
            terminals.push_back("x_"+to_string(i));
          } else if (sym == "erc") {
            terminals.push_back("erc");
          } else {
            // constant
            float c = stof(sym);
            terminals.push_back(to_string(c));
          }
        } catch(std::invalid_argument const& ex) {
          throw runtime_error("Unrecognized terminal: "+sym);
        }
      }
    }
  }

  void set_terminal_probabilities(string setting) {
    if (setting == "auto") {
      cumul_tset_probs = Vec(terminals.size());
      float p = 1.0 / terminals.size();
      float cumul_p = 0;
      for (int i = 0; i < terminals.size(); i++) {
        cumul_p += p;
        cumul_tset_probs[i] = cumul_p;
      }
      return;
    }
    // else, use what provided
    cumul_tset_probs = _compute_custom_cumul_probs_operator_set(setting, terminals);
  }

  string str_terminal_set() {
    string str = "";
    for (string el : terminals) {
      str += el +",";
    }
    str = str.substr(0, str.size()-1);
    return str;
  }

  void set_batch_size(string lib_batch_size) {
    if (lib_batch_size == "auto") {
      batch_size = fit_func->X_train.rows();
    } else {
      int n = fit_func->X_train.rows();
      batch_size = stoi(lib_batch_size);
      if (batch_size > n) {
        print("[!] Warning: batch size is larger than the number of training examples. Setting it to ", n);
        batch_size = n;
      }
    }
  }

  void apply_feature_selection(int num_feats_to_keep) {
    // check if nothing needs to be done
    if (num_feats_to_keep == -1) {
      return;
    }
    int num_features = 0;
    for(string s : terminals) {
      if (s[0] == 'x')
        num_features++;
    }
    if (num_features <= num_feats_to_keep)
      return;

    // proceed with feature selection
    Veci indices_to_keep = feature_selection(fit_func->X_train, fit_func->y_train, num_feats_to_keep);
    vector<int> indices_to_remove; indices_to_remove.reserve(terminals.size());
    for(int i = 0; i < terminals.size(); i++) {
      if (terminals[i][0] != 'x')
        continue; // ignore constants
      
      auto end = indices_to_keep.data() + indices_to_keep.size();
      if (find(indices_to_keep.data(), end, i) == end) {
        indices_to_remove.push_back(i);
      }
    }

    // remove those terminals from the search (from back to front not to screw up indexing)
    for(int i = indices_to_remove.size() - 1; i >= 0; i--) {
      int idx = indices_to_remove[i];
      terminals.erase(terminals.begin() + idx);
    }

    // gotta update also prob of sampling terminals
    if (lib_tset_probs != "auto") {
      vector<string> prob_str = split_string(lib_tset_probs);
      for(int i = indices_to_remove.size() - 1; i >= 0; i--) {
        int idx = indices_to_remove[i];
        prob_str.erase(prob_str.begin() + idx);
      }
      lib_tset_probs = "";
      for(int i = 0; i < prob_str.size(); i++) {
        lib_tset_probs += prob_str[i];
        if (i < prob_str.size()-1)
          lib_tset_probs += ",";
      }
    }

  }

  void reset() {
    operators.clear();
    terminals.clear();
    if (fit_func)
      delete fit_func;
    fit_func = NULL;
  }

  void read_options(int argc, char** argv) {
    reset();
    cli::Parser parser(argc, argv);

    // budget
    parser.set_optional<int>("pop", "population_size", 1000, "Population size");
    parser.set_optional<int>("g", "generations", 20, "Budget of generations (-1 for disabled)");
    parser.set_optional<int>("t", "time", -1, "Budget of time (-1 for disabled)");
    parser.set_optional<int>("e", "evaluations", -1, "Budget of evaluations (-1 for disabled)");
    parser.set_optional<long>("ce", "component_evaluations", -1, "Budget of component evaluations (-1 for disabled)");
    parser.set_optional<bool>("disable_ims", "disable_ims", false, "Whether to disable the IMS (default is false)");
    // initialization
    parser.set_optional<string>("is", "initialization_strategy", "hh", "Strategy to sample the initial population");
    parser.set_optional<int>("d", "depth", 4, "Maximum depth that the trees can have");
    // problem & representation
    parser.set_optional<string>("ff", "fitness_function", "ac", "Fitness function");
    parser.set_optional<string>("oset", "operator_set", "+,-,*,/,sin,cos,log", "Operator set");
    parser.set_optional<string>("oset_probs", "operator_set_probabilities", "auto", "Probabilities of sampling each element of the operator set (same order as oset)");
    parser.set_optional<string>("tset", "terminal_set", "auto", "Terminal set");
    parser.set_optional<string>("tset_probs", "terminal_set_probabilities", "auto", "Probabilities of sampling each element of the terminal set (same order as tset)");
    parser.set_optional<string>("train", "training_set", "./train.csv", "Path to the training set (needed only if calling as CLI)");
    parser.set_optional<string>("bs", "batch_size", "auto", "Batch size (default is 'auto', i.e., the entire training set)");
    parser.set_optional<string>("compl", "complexity_type", "component_count", "Measure to score the complexity of candidate sotluions (default is component_count)");
    parser.set_optional<float>("rci", "rel_compl_imp", 0.0, "Relative importance of complexity over accuracy to select the final elite (default is 0.0)");
    parser.set_optional<int>("feat_sel", "feature_selection", 10, "Max. number of feature to consider (if -1, all features are considered)");
    // variation
    parser.set_optional<float>("cmp", "coefficient_mutation_probability", 0.1, "Probability of applying coefficient mutation");
    parser.set_optional<float>("cmt", "coefficient_mutation_temperature", 0.05, "Temperature of coefficient mutation");
    parser.set_optional<bool>("nolink", "no_linkage", false, "Disables computing linkage when building the linkage tree FOS, essentially making it random");
    parser.set_optional<bool>("no_large_fos", "no_large_fos", false, "Whether to discard subsets in the FOS with size > half the size of the genotype (default is false)");
    parser.set_optional<bool>("no_univ_fos", "no_univ_fos", false, "Whether to discard univariate subsets in the FOS (default is false)");
    // selection
    parser.set_optional<int>("tour", "tournament_size", 2, "Tournament size (if tournament selection is active)");
    parser.set_optional<bool>("diversity", "diversity_promotion", false, "Whether to use diversity promotion (default is false)");
    // other
    parser.set_optional<int>("random_state", "random_state", -1, "Random state (seed)");
    parser.set_optional<bool>("verbose", "verbose", false, "Verbose");
    parser.set_optional<bool>("lib", "call_as_lib", false, "Whether the code is called as a library (e.g., from Python)");

    // set options
    parser.run_and_exit_if_error();

    // verbose (MUST BE FIRST)
    verbose = parser.get<bool>("verbose");
    if (!verbose) {
      cout.rdbuf(NULL);
    }

    // random_state
    random_state = parser.get<int>("random_state");
    if (random_state >= 0){
      Rng::set_seed(random_state);
      print("random state: ", random_state);
    } else {
      print("random state: not set");
    }
    
    // budget
    disable_ims = parser.get<bool>("disable_ims");
    if (disable_ims) {
      pop_size = parser.get<int>("pop");
      print("pop. size: ",pop_size);
    } else {
      pop_size = 64;
      print("IMS active");
    }
   
    max_generations = parser.get<int>("g");
    max_time = parser.get<int>("t");
    max_evaluations = parser.get<int>("e");
    max_component_evaluations = parser.get<long>("ce");
    print("budget: ", 
      max_generations > -1 ? max_generations : INF, " generations, ", 
      max_time > -1 ? max_time : INF, " time [s], ", 
      max_evaluations > -1 ? max_evaluations : INF, " evaluations, ", 
      max_component_evaluations > -1 ? max_component_evaluations : INF, " component evaluations" 
    );

    // initialization
    init_strategy = parser.get<string>("is");
    print("initialization strategy: ", init_strategy);
    max_depth = parser.get<int>("d");
    print("max. depth: ", max_depth);
    
    // variation
    cmut_prob = parser.get<float>("cmp");
    cmut_temp = parser.get<float>("cmt");
    print("coefficient mutation probability: ", cmut_prob, ", temperature: ",cmut_temp);

    no_linkage = parser.get<bool>("nolink");
    no_large_subsets = parser.get<bool>("no_large_fos");
    no_univariate = parser.get<bool>("no_univ_fos");
    print("compute linkage: ", no_linkage ? "false" : "true", " (FOS trimming-no large: ",no_large_subsets,", no univ.: ",no_univariate,")");

    // selection
    tournament_size = parser.get<int>("tour");
    print("tournament size: ", tournament_size);
    diversity_promotion = parser.get<bool>("diversity");
    print("diversity promotion: ", diversity_promotion ? "true" : "false");

    // problem
    string fit_func_name = parser.get<string>("ff");
    set_fit_func(fit_func_name);
    print("fitness function: ", fit_func_name);

    _call_as_lib = parser.get<bool>("lib");
    if (!_call_as_lib) {
      // then it expects a training set
      path_to_training_set = parser.get<string>("train");
      // load up
      if (!exists(path_to_training_set)) {
        throw runtime_error("Training set not found at path "+path_to_training_set);
      }
      Mat Xy = load_csv(path_to_training_set);
      Mat X = remove_column(Xy, Xy.cols()-1);
      Vec y = Xy.col(Xy.cols()-1);
      fit_func->set_Xy(X, y);
    } 
    lib_batch_size = parser.get<string>("bs");
    if (!_call_as_lib) {
      set_batch_size(lib_batch_size);
      print("batch size: ", g::batch_size);
    }

    // representation
    string oset = parser.get<string>("oset");
    set_operators(oset);
    string oset_p = parser.get<string>("oset_probs");
    set_operator_probabilities(oset_p);
    print("operator set: ",oset," (probabs: ",oset_p,")");
    
    lib_tset = parser.get<string>("tset");
    lib_feat_sel_number = parser.get<int>("feat_sel");
    lib_tset_probs = parser.get<string>("tset_probs");
    if (!_call_as_lib) {
      set_terminals(lib_tset);
      apply_feature_selection(lib_feat_sel_number);
      set_terminal_probabilities(lib_tset_probs);
      print("terminal set: ",str_terminal_set()," (probs: ",lib_tset_probs, (lib_feat_sel_number > -1 ? ", feat.selection : "+to_string(lib_feat_sel_number) : ""), ")");
    } 

    complexity_type = parser.get<string>("compl");
    rel_compl_importance = parser.get<float>("rci");
    print("complexity type: ",complexity_type," (rel. importance: ",rel_compl_importance,")");


    
    // other
    cout << std::setprecision(NUM_PRECISION);
  }

  void clear_globals() {
    for(auto * f : all_fitness_functions) {
      delete f;
    }
    reset();
  }


}

#endif
