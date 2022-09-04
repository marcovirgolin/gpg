#ifndef FITNESS_H
#define FITNESS_H

#include "globals.hpp"
#include "myeig.hpp"
#include "node.hpp"

using namespace myeig;

struct Fitness {

  virtual float get_fitness(Node * n, Mat X, Vec y) {
    throw runtime_error("Not implemented");
  }

  Vec get_fitnesses(vector<Node*> population, Mat X, Vec y, bool compute=true) {
    Vec fitnesses(population.size());
    for(int i = 0; i < population.size(); i++) {
      if (compute)
        fitnesses[i] = get_fitness(population[i], X, y);
      else
        fitnesses[i] = population[i]->fitness;
    }
    return fitnesses;
  }

};

struct MAEFitness : Fitness {

  float get_fitness(Node * n, Mat X, Vec y) override {
    Vec out = n->get_output(X);

    float fitness = (y - out).abs().mean();
    if (isnan(fitness))
      fitness = INF;
    n->fitness = fitness;

    return fitness;
  }

};

struct MSEFitness : Fitness {

  float get_fitness(Node * n, Mat X, Vec y) override {
    Vec out = n->get_output(X);

    float fitness = (y-out).square().mean();
    if (isnan(fitness))
      fitness = INF;
    n->fitness = fitness;

    return fitness;
  }

};

struct AbsCorrFitness : Fitness {

  float get_fitness(Node * n, Mat X, Vec y) override {
    Vec out = n->get_output(X);

    float fitness = abs(corr(y, out));
    if (isnan(fitness))
      fitness = INF;
    n->fitness = fitness;

    return fitness;
  }

};








#endif