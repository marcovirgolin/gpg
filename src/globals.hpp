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
int pop_size = 1000;

// representation
int max_depth = 5;
string init_strategy = "hh";
vector<Op*> functions;
vector<Op*> terminals;

// termination criteria
int max_generations = 100;
int max_time = -1;
int max_evaluations = -1;

// fitness
Fitness * fitness = NULL;



// Random stuff

void set_options() {
  max_generations = 10;
  seed = 42;
  srand((unsigned int) seed);

  functions = {new Add(), new Sub(), new Mul(), new Div()};
  terminals = {new Feat(0), new Feat(1)};

  fitness = new AbsCorrFitness();

  Mat X(3,2);
  X << 1, 2,
       3, 4,
       5, 6;
  Vec y(3);
  y << 1, 0, 1;
  fitness->set_Xy(X, y);

}

void clear_globals() {
  for(auto * f : functions) {
    delete f;
  }
  for(auto * t : terminals) {
    delete t; 
  }
  if (fitness)
    delete fitness;
}


}

#endif