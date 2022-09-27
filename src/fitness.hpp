#ifndef FITNESS_H
#define FITNESS_H

#include "myeig.hpp"
#include "node.hpp"
#include "util.hpp"

using namespace myeig;

struct Fitness {

  virtual ~Fitness() {};

  Mat X_train, X_val;
  Vec y_train, y_val;

  virtual string name() {
    throw runtime_error("Not implemented");
  }

  virtual Fitness * clone() {
    throw runtime_error("Not implemented");
  }

  virtual float get_fitness(Node * n, Mat & X, Vec & y) {
    throw runtime_error("Not implemented");
  }

  // shorthand for training set
  float get_fitness(Node * n, Mat * X=NULL, Vec * y=NULL) {
    if (!X)
      X = & this->X_train;
    if (!y)
      y = & this->y_train;
    return get_fitness(n, *X, *y);
  }

  Vec get_fitnesses(vector<Node*> population, Mat * X=NULL, Vec * y=NULL, bool compute=true) {
    
    Vec fitnesses(population.size());
    for(int i = 0; i < population.size(); i++) {
      if (compute)
        fitnesses[i] = get_fitness(population[i], X, y);
      else
        fitnesses[i] = population[i]->fitness;
    }
    return fitnesses;
  }

  void set_X(Mat & X, string type="train") {
    if (type == "train")
      X_train = X;
    else if (type=="val")
      X_val = X;
    else
      throw runtime_error("Unrecognized X type "+type);
  }

  void set_y(Vec & y, string type="train") {
    if (type == "train")
      y_train = y;
    else if (type=="val")
      y_val = y;
    else
      throw runtime_error("Unrecognized y type "+type);
  }

  void set_Xy(Mat & X, Vec & y, string type="train") {
    set_X(X, type);
    set_y(y, type);
  }

};

struct MAEFitness : Fitness {

  string name() override {
    return "mae";
  }
  
  Fitness * clone() override {
    return new MAEFitness();
  }

  float get_fitness(Node * n, Mat & X, Vec & y) override {
    Vec out = n->get_output(X);

    float fitness = (y - out).abs().mean();
    if (isnan(fitness) || fitness < 0) // the latter can happen due to float overflow
      fitness = INF;
    n->fitness = fitness;

    return fitness;
  }

};

struct MSEFitness : Fitness {

  string name() override {
    return "mse";
  }

  Fitness * clone() override {
    return new MSEFitness();
  }

  float get_fitness(Node * n, Mat & X, Vec & y) override {
    Vec out = n->get_output(X);

    float fitness = (y-out).square().mean();
    if (isnan(fitness) || fitness < 0) // the latter can happen due to float overflow
      fitness = INF;
    n->fitness = fitness;

    return fitness;
  }

};

struct AbsCorrFitness : Fitness {

  string name() override {
    return "ac";
  }

  Fitness * clone() override {
    return new AbsCorrFitness();
  }

  float get_fitness(Node * n, Mat & X, Vec & y) override {
    Vec out = n->get_output(X);

    float fitness = 1.0-abs(corr(y, out));
    // Below, the < 0 can happen due to float overflow, while 
    // the = 0 is meant to penalize constants as much as broken solutions
    if (isnan(fitness) || fitness <= 0) 
      fitness = INF;
    n->fitness = fitness;

    return fitness;
  }

};








#endif