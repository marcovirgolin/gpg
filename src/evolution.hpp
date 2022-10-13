#ifndef EVOLUTION_H
#define EVOLUTION_H

#include <Python.h>
#include <unordered_map>

#include "util.hpp"
#include "node.hpp"
#include "variation.hpp"
#include "selection.hpp"
#include "fos.hpp"
#include "complexity.hpp"
#include "globals.hpp"

struct Evolution {
  
  vector<Node*> population;
  Node * elite = NULL;
  FOSBuilder * fb = NULL;
  int gen_number = 0;
  int pop_size = 0;
  unordered_map<float, Node*> elites_per_complexity;

  Evolution(int pop_size) {
    this->pop_size = pop_size;
    fb = new FOSBuilder();
    init_pop();
  }

  ~Evolution() {
    clear_population(population);
    if (elite)
      elite->clear();
    for (auto it = elites_per_complexity.begin(); it != elites_per_complexity.end(); it++) {
      it->second->clear();
    }
    elites_per_complexity.clear();
    if (fb)
      delete fb;
  }

  bool converged(vector<Node*> & population, bool ignore_constants=false) {
    auto rp = rand_perm(population.size());
    auto nodes_first = population[rp[0]]->subtree();
    for(int i = 1; i < population.size(); i++) {
      auto nodes_i = population[rp[i]]->subtree();
      for(int j = 0; j < nodes_first.size(); j++) {
        if (ignore_constants && 
          nodes_first[j]->op->type() == OpType::otConst && 
          nodes_i[j]->op->type() == OpType::otConst)
          continue;
        if (nodes_first[j]->op->sym() != nodes_i[j]->op->sym()) {
          return false;
        }
      }
    }
    return true;
  }

  bool check_n_set_elite(Node * tree) {

    float c = compute_complexity(tree);

    // replace elites
    vector<float> obsolete_complexities; obsolete_complexities.reserve(elites_per_complexity.size());
    for(auto it = elites_per_complexity.begin(); it != elites_per_complexity.end(); it++) {
      if (c <= it->first && tree->fitness < it->second->fitness) {
        obsolete_complexities.push_back(it->first);
      }
    }
    if (obsolete_complexities.empty()) {
      // TODO return false;
    }
    for(float oc : obsolete_complexities) {
      elites_per_complexity[oc]->clear();
      elites_per_complexity.erase(oc);
    }
    elites_per_complexity[c] = tree->clone();
    // TODO: return true;

    // TODO: delete what follows
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

  void clear_population(vector<Node*> & population) {
    for(auto * tree : population) {
      tree->clear();
    }
    population.clear();
  }

  void init_pop() {
    population.reserve(pop_size);
    for(int i = 0; i < pop_size; i++) {
      auto * tree = generate_tree(g::max_depth, g::init_strategy);
      g::fit_func->get_fitness(tree);
      //check_n_set_elite(tree);
      population.push_back(tree);
    }
  } 

  void gomea_generation() {
    // build linkage tree fos
    auto fos = fb->build_linkage_tree(population);

    // perform GOM
    vector<Node*> offspring_population; 
    offspring_population.reserve(pop_size);
    for(int i = 0; i < pop_size; i++) {
      auto * offspring = efficient_gom(population[i], population, fos);
      //check_n_set_elite(offspring);
      offspring_population.push_back(offspring);
    }

    // replace parent with offspring population
    clear_population(population);
    population = offspring_population;
    ++gen_number;
  }

  void ga_generation() {
    vector<Node*> offspring_population; 
    offspring_population.reserve(pop_size);
    for(int i = 0; i < pop_size; i++) {
      auto * cr_offspring = crossover(population[i], population[randu()*population.size()]);
      auto * mut_offspring = mutation(cr_offspring, g::functions, g::terminals, 0.75);
      cr_offspring->clear();
      mut_offspring = coeff_mut(mut_offspring, false);
      // compute fitness
      g::fit_func->get_fitness(mut_offspring);
      //check_n_set_elite(mut_offspring);
      // add to off pop
      offspring_population.push_back(mut_offspring);
    }

    // selection
    auto selection = popwise_tournament(offspring_population, pop_size, g::tournament_size, g::tournament_stochastic);
    
    // clean up
    clear_population(population);
    clear_population(offspring_population);
    population = selection;
    ++gen_number;
  }

  void run() {

    for(int i = 0; i < g::max_generations; i++) {
      if(g::_call_as_lib && PyErr_CheckSignals() == -1) {
        exit(1);
      }

      // update mini batch
      bool is_updated = g::fit_func->update_batch(g::batch_size);
      if (is_updated && elite) {
        elite->clear();
        elite = NULL;
        // TODO: remove elite and keep map
        for (auto it = elites_per_complexity.begin(); it != elites_per_complexity.end(); it++) {
          it->second->clear();
        }
        elites_per_complexity.clear();
      }
      
      gomea_generation();
      print("gen: ",gen_number, " elite fitness: ", elite->fitness); // TODO: remove elite
      if (converged(population, true)) {
        print("population converged");
        break;
      }

    }

    // if abs corr, append linear scaling terms
    if (g::fit_func->name() == "ac") {
      elite = append_linear_scaling(elite); 
      for (auto it = elites_per_complexity.begin(); it != elites_per_complexity.end(); it++) {
        elites_per_complexity[it->first] = append_linear_scaling(it->second);
      }
    }

    // TODO: remove this
    for (auto it = elites_per_complexity.begin(); it != elites_per_complexity.end(); it++) {
      print(it->first, " ", it->second->fitness, ":", it->second->human_repr());
    }

    if (!g::_call_as_lib) {
      print(elite->human_repr());
    }
  }

};

#endif