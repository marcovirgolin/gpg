#ifndef EVOLUTION_H
#define EVOLUTION_H

#include <Python.h>
#include <unordered_map>

#include "util.hpp"
#include "individual.hpp"
#include "variation.hpp"
#include "selection.hpp"
#include "fos.hpp"
#include "complexity.hpp"
#include "globals.hpp"

struct Evolution {
  
  vector<Individual*> population;
  FOSBuilder * fb = NULL;
  int gen_number = 0;
  int pop_size = 0;

  Evolution(int pop_size) {
    this->pop_size = pop_size;
    fb = new FOSBuilder();
    init_pop();
  }

  ~Evolution() {
    for (Individual * indiv : population)
      delete indiv;
    population.clear();
    if (fb)
      delete fb;
  }

 

  void init_pop() {
    unordered_set<string> already_generated;
    population.reserve(pop_size);
    int init_attempts = 0;
    while (population.size() < pop_size) {
      Individual * indiv = variation::generate_indiv(g::max_depth, g::init_strategy);
      string str_indiv = indiv->to_prefix_notation();
      if (init_attempts < g::max_init_attempts && already_generated.find(str_indiv) != already_generated.end()) {
        delete indiv;
        init_attempts++;
        if (init_attempts == g::max_init_attempts) {
          print("[!] Warning: could not initialize a syntactically-unique population within ", init_attempts, " attempts");
        }
        continue;
      } 
      already_generated.insert(str_indiv);
      g::fit_func->get_fitness(indiv);
      population.push_back(indiv);
    }
  } 

  void gomea_generation() {
    // build linkage tree FOS
    auto fos = fb->build_linkage_tree(population);

    // perform GOM
    vector<Individual*> offspring_population; 
    offspring_population.reserve(pop_size);
    for(int i = 0; i < pop_size; i++) {
      auto * offspring = variation::gom(population[i], population, fos);
      //check_n_set_elite(offspring);
      offspring_population.push_back(offspring);
    }

    // replace parent with offspring population
    for (Individual * indiv : population)
      delete indiv;
    population = offspring_population;
    ++gen_number;
  }

  void run() {
    throw runtime_error("Not implemented, please use IMS (with max runs 1 if you want a single population)");
  }

};

#endif