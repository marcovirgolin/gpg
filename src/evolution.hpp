#ifndef EVOLUTION_H
#define EVOLUTION_H

#include "util.hpp"
#include "node.hpp"
#include "variation.hpp"
#include "selection.hpp"
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
    clear_population(population);
    if (elite)
      elite->clear();
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
    population.reserve(g::pop_size);
    for(int i = 0; i < g::pop_size; i++) {
      auto * tree = generate_tree(g::functions, g::terminals, g::max_depth, g::init_strategy);
      g::fit_func->get_fitness(tree);
      check_n_set_elite(tree);
      population.push_back(tree);
    }
  } 

  void gomea_generation() {
    // build linkage tree fos
    auto fos = fb->build_linkage_tree(population);

    // perform GOM
    vector<Node*> offspring_population; 
    offspring_population.reserve(g::pop_size);
    for(int i = 0; i < g::pop_size; i++) {
      auto * offspring = efficient_gom(population[i], population, fos);
      check_n_set_elite(offspring);
      offspring_population.push_back(offspring);
    }

    // replace parent with offspring population
    clear_population(population);
    population = offspring_population;
  }

  void ga_generation() {
    vector<Node*> offspring_population; 
    offspring_population.reserve(g::pop_size);
    for(int i = 0; i < g::pop_size; i++) {
      auto * cr_offspring = crossover(population[i], population[randu()*population.size()]);
      auto * mut_offspring = mutation(cr_offspring, g::functions, g::terminals, 0.75);
      cr_offspring->clear();
      mut_offspring = coeff_mut(mut_offspring, false);
      // compute fitness
      g::fit_func->get_fitness(mut_offspring);
      check_n_set_elite(mut_offspring);
      // add to off pop
      offspring_population.push_back(mut_offspring);
    }

    // selection
    auto selection = popwise_tournament(offspring_population, g::pop_size, g::tournament_size, g::tournament_stochastic);
    
    // clean up
    clear_population(population);
    clear_population(offspring_population);
    population = selection;
  }

  void run() {
    init_pop();

    for(int i = 0; i < g::max_generations; i++) {
      gomea_generation();
      print("gen: ",++gen_number, " elite fitness: ", elite->fitness);
      if (converged(population, true)) {
        print("population converged");
        break;
      }
    }

    elite->print_subtree();

  }

};

#endif