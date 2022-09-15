#ifndef GLOBALS_H
#define GLOBALS_H

#include <random>
#include <chrono>
#include <sstream>
#include "myeig.hpp"
#include "operator.hpp"
#include "fitness.hpp"
#include "cmdparser.hpp"

using namespace std;
using namespace myeig;

namespace g {


// budget
int pop_size = 1000;
int max_generations = 100000;
int max_time = -1;
int max_evaluations = -1;
long max_node_evaluations = -1;

// representation
int max_depth;
string init_strategy;
vector<Op*> functions;
vector<Op*> terminals;


// problem
Fitness * fit_func = NULL;
string path_to_training_set;

// variation
bool no_linkage = false;
float cmut_eps = 1e-5;
float cmut_prob = 0.0;
float cmut_temp = 0.01;


// selection
int tournament_size = 2;
bool tournament_stochastic = false;

// other
int seed = -1;
bool verbose = true;
bool _call_as_lib = false;

// Functions
void set_terminal_set(string setting) {
  if (option == "auto") {
    for(int i = 0; i < X.cols(); i++) {
      terminals.push_back(new Feat(i));
    }
    terminals.push_back(new Const());
  }
  else {
    throw runtime_error("Unrecognized setting for terminal set: "+setting);
  }
}


void read_options(int argc, char** argv) {
  cli::Parser parser(argc, argv);
  
  // budget
  parser.set_optional<int>("pop", "population_size", 1000, "Population size");
  parser.set_optional<int>("g", "generations", 20, "Budget of generations (-1 for disabled)");
  parser.set_optional<int>("t", "time", -1, "Budget of time (-1 for disabled)");
  parser.set_optional<int>("e", "evaluations", -1, "Budget of evaluations (-1 for disabled)");
  parser.set_optional<long>("ne", "node evaluations", -1, "Budget of node evaluations (-1 for disabled)");
  // initialization
  parser.set_optional<string>("is", "initialization_strategy", "hh", "Strategy to sample the initial population");
  parser.set_optional<int>("d", "depth", 3, "Maximum depth trees can have");
  // problem
  parser.set_optional<string>("fit", "fitness_function", "ac", "Fitness function");
  parser.set_optional<string>("fset", "function_set", "+,-,*,/", "Function set");
  parser.set_optional<string>("tset", "terminal_set", "auto", "Terminal set");
  parser.set_optional<string>("train", "training_set", "./train.csv", "Path to the training set");
  // variation
  parser.set_optional<float>("cmp", "coefficient_mutation_probability", 0.1, "Probability of applying coefficient mutation to a coefficient node");
  parser.set_optional<float>("cmt", "coefficient_mutation_temperature", 0.05, "Temperature of coefficient mutation");
  parser.set_optional<bool>("nolink", "no_linkage", false, "Disables computing linkage when building the linkage tree FOS, essentially making it random");
  // other
  parser.set_optional<int>("s", "seed", -1, "Random seed");
  parser.set_optional<bool>("v", "verbose", true, "Verbose");
  parser.set_optional<bool>("lib", "call_as_lib", false, "Whether the code is called as a library (e.g., from Python)");
  
  // set options
  parser.run_and_exit_if_error();

  // verbose (MUST BE FIRST)
  verbose = parser.get<bool>("v");
  if (!verbose) {
    cout.rdbuf(NULL);
  }

  // seed
  seed = parser.get<int>("s");
  if (seed > 0){
    srand((unsigned int) seed);
    print("seed: ",seed);
  } else {
    print("seed: not set");
  }
  
  // budget
  pop_size = parser.get<int>("pop");
  print("pop. size: ",pop_size);

  max_generations = parser.get<int>("g");
  max_time = parser.get<int>("t");
  max_evaluations = parser.get<int>("e");
  max_node_evaluations = parser.get<long>("ne");
  print("budget: ", 
    max_generations > -1 ? max_generations : INF, " generations, ", 
    max_time > -1 ? max_time : INF, " time [s], ", 
    max_evaluations > -1 ? max_evaluations : INF, " evaluations, ", 
    max_node_evaluations > -1 ? max_node_evaluations : INF, " node evaluations" 
  );

  // initialization
  init_strategy = parser.get<string>("is");
  print("initialization strategy: ", init_strategy);
  max_depth = parser.get<int>("d");
  print("max. depth: ", max_depth);
  
  // variation
  cmut_prob = parser.get<float>("cmp");

  no_linkage = parser.get<bool>("nolink");
  print("compute linkage: ", no_linkage ? "false" : "true");

  // problem
  string fit_func_name = parser.get<string>("fit");
  if (fit_func_name == "ac")
    fit_func = new AbsCorrFitness();
  else if (fit_func_name == "mse")
    fit_func = new MSEFitness();
  else if (fit_func_name == "mae")
    fit_func = new MAEFitness();
  else
    throw runtime_error("Unrecognized fitness function name: "+fit_func_name);

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

  // representation
  functions = {new Add(), new Sub(), new Mul(), new Div()};
  set_terminal_set(parser.get<string>("ts"));
  
  /*Mat X(10,3);
  X << 1, 2, 1,
       3, 4, 5,
       5, 6, 7,
       8, 1, 0,
       4, 4, 2, 
       1, 1, 1,
       4, 3, 2, 
       2, 2, 2,
       0, 0, 9, 
       12, 32, 2;
  Vec y(10);
  y << 1, 0, 1, 9, 2, 3, 1, 4, 5, 2;
  fit_func->set_Xy(X, y);*/

  // other
  
}


void clear_globals() {
  for(auto * f : functions) {
    delete f;
  }
  for(auto * t : terminals) {
    delete t; 
  }
  if (fit_func)
    delete fit_func;
}


}

#endif