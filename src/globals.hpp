#ifndef GLOBALS_H
#define GLOBALS_H

#include <random>
#include <chrono>
#include "myeig.hpp"
#include "operator.hpp"
#include "fitness.hpp"
#include "cmdparser.hpp"

using namespace std;
using namespace myeig;

namespace g {

// general
int seed = -1;

// evolution
int pop_size = 1000;

// representation
int max_depth = 5;
string init_strategy = "hh";
vector<Op*> functions;
vector<Op*> terminals;

// termination criteria
int max_generations = 100000;
int max_time = -1;
int max_evaluations = -1;

// fitness
Fitness * fit_func = NULL;

// variation
bool no_linkage = false;
float cmut_eps = 1e-5;
float cmut_prob = 0.0;
float cmut_temp = 0.01;


// selection
int tournament_size = 2;
bool tournament_stochastic = false;


// Random stuff


// Functions

void read_options(int argc, char** argv) {
  cli::Parser parser(argc, argv);
  parser.set_optional<int>("s", "seed", -1, "Random seed");
  parser.set_optional<int>("pop", "population_size", 1000, "Population size");
  parser.set_optional<string>("fit", "fitness_function", "ac", "Fitness function");
  parser.set_optional<string>("fset", "function_set", "+,-,*,/", "Function set");
  parser.set_optional<string>("tset", "terminal_set", "x_i,c", "Terminal set");
  parser.set_optional<float>("cmp", "coefficient_mutation_probability", 0.1, "Probability of applying coefficient mutation to a coefficient node");
  parser.set_optional<float>("cmt", "coefficient_mutation_temperature", 0.05, "Temperature of coefficient mutation");
  parser.set_optional<bool>("nolink", "no_linkage", false, "Disables computing linkage when building the linkage tree FOS, essentially making it random");
  
  // set options
  parser.run_and_exit_if_error();

  // seed
  seed = parser.get<int>("s");
  if (seed > 0){
    srand((unsigned int) seed);
    print("seed: ",seed);
  } else {
    print("seed: not set");
  }
  
  // pop size
  pop_size = parser.get<int>("pop");
  print("pop. size: ",pop_size);



  cmut_prob = parser.get<float>("cmp");

  // linkage
  no_linkage = parser.get<bool>("nolink");
  print("compute linkage: ", no_linkage ? "false" : "true");

  functions = {new Add(), new Sub(), new Mul(), new Div()};
  terminals = {new Feat(0), new Feat(1), new Const()};
  fit_func = new AbsCorrFitness();

  Mat X(10,3);
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
  fit_func->set_Xy(X, y);
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