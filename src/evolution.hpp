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

      /*auto genome = indiv->genome;
      print(genome);
      print(indiv->to_prefix_notation());
      print(indiv->to_infix_notation());
      print(indiv->get_num_components());
      auto active_indices = indiv->get_active_indices();
      print(active_indices);
      exit(0);*/
    }
  } 

  void gomea_generation() {
    // build linkage tree FOS
    auto fos = fb->build_linkage_tree(population);

    // diversity promotion
    Vec chance_random_accept = Vec::Zero(pop_size);
    if (g::diversity_promotion) {
      // get parent fitnesses
      vector<float> fitnesses; fitnesses.reserve(pop_size);
      float max_fitness = -INF; 
      float min_fitness = INF;
      for (Individual * indiv : population){
        fitnesses.push_back(indiv->fitness);
        if (indiv->fitness > max_fitness)
          max_fitness = indiv->fitness;
        if (indiv->fitness < min_fitness)
          min_fitness = indiv->fitness;
      }
      for(int i = 0; i < pop_size; i++) {
        chance_random_accept[i] = (fitnesses[i] - min_fitness) / (max_fitness - min_fitness);
      }
    }

    // perform GOM
    vector<Individual*> offspring_population; 
    offspring_population.reserve(pop_size);
    for(int i = 0; i < pop_size; i++) {
      auto * offspring = variation::gom(population[i], population, fos, chance_random_accept[i]);
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