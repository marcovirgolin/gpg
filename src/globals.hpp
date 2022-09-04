#ifndef GLOBALS_H
#define GLOBALS_H

#include <random>
#include "myeig.hpp"

using namespace myeig;

namespace g {

int max_generations = 100;
int max_time = -1;
int max_evaluations = -1;
int seed = 42;

Mat X_train, X_val;
Vec y_train, y_val;



// Random stuff

void set_options() {
  max_generations = 10;
  seed = 42;
  srand((unsigned int) seed);
}


}

#endif