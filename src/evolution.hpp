#ifndef EVOLUTION_H
#define EVOLUTION_H

#include "util.hpp"
#include "node.hpp"
#include "variation.hpp"
#include "globals.hpp"

struct Evolution {

  vector<Node*> population;
  Node * elite = NULL;

  ~Evolution() {
    for(auto * t : population) {
      t->clear();
    }
    elite->clear();
  }

  bool check_n_set_elite(Node * tree) {
    if (!elite) {
      elite = tree->clone();
      return true;
    } 
    // accept if better fitness
    if (tree->fitness < elite->fitness) {
      elite->clear();
      elite = tree->clone();
      return true;
    }
    return false;
  } 

  void run() {
    // initialize population
    population.reserve(g::pop_size);
    for(int i = 0; i < g::pop_size; i++) {
      auto * tree = generate_tree(g::functions, g::terminals, g::max_depth, g::init_strategy);
      g::fitness->get_fitness(tree);
      check_n_set_elite(tree);
      population.push_back(tree);
    }

    for(int i = 0; i < g::max_generations; i++) {
      print("gen: ",i+1, " elite fitness: ", elite->fitness);
    }

  }

};

#endif