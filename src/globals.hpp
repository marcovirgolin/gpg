#ifndef GLOBALS_H
#define GLOBALS_H

#include <random>
#include <chrono>
#include "myeig.hpp"
#include "operator.hpp"
#include "fitness.hpp"

using namespace std;
using namespace myeig;

namespace g {

// general
int seed = 42;

// evolution
int pop_size = 100;

// representation
int max_depth = 4;
string init_strategy = "hh";
vector<Op*> functions;
vector<Op*> terminals;

// termination criteria
int max_generations = 100;
int max_time = -1;
int max_evaluations = -1;

// fitness
Fitness * fit_func = NULL;

// variation
float cmut_eps = 1e-5;


// Random stuff

void set_options() {
  pop_size = 1000;
  max_generations = 100;
  seed = 42;
  srand((unsigned int) seed);

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