#ifndef EVOLUTION_H
#define EVOLUTION_H

#include "util.hpp"
#include "node.hpp"
#include "variation.hpp"
#include "fos.hpp"
#include "globals.hpp"

struct Evolution {

  vector<Node*> population;
  Node * elite = NULL;
  FOSBuilder * fb = NULL;
  int gen_number = 0;

  Evolution() {
    fb = new FOSBuilder();
  }

  ~Evolution() {
    for(auto * t : population) {
      t->clear();
    }
    if (elite)
      elite->clear();
    if (fb)
      delete fb;
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

  void init_pop() {
    population.reserve(g::pop_size);
    for(int i = 0; i < g::pop_size; i++) {
      auto * tree = generate_tree(g::functions, g::terminals, g::max_depth, g::init_strategy);
      g::fitness->get_fitness(tree);
      check_n_set_elite(tree);
      population.push_back(tree);
    }
  } 

  void generation() {
    // build linkage tree fos
    auto fos = fb->build_linkage_tree(population);
    print("gen: ",++gen_number, " elite fitness: ", elite->fitness);
  }

  void run() {
    init_pop();

    for(int i = 0; i < g::max_generations; i++) {
      generation();
    }

  }

};

#endif