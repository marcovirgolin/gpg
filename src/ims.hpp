#ifndef IMS_H
#define IMS_H

#include <Python.h>
#include <unordered_map>

#include "globals.hpp"
#include "util.hpp"
#include "evolution.hpp"
#include "myeig.hpp"

using namespace std;
using namespace myeig;

struct IMS {

  int MAX_POP_SIZE = (int) pow(2,20);
  int SUB_GENs = 4;

  vector<Evolution*> evolutions;
  int macro_generations = 0;
  unordered_map<float, Node*> elites_per_complexity;

  ~IMS() {
    for (Evolution * e : evolutions) {
      delete e;
    }
    reset_elites();
  }

  Node * select_elite(float rel_compl_importance=0.0) {
    // get relative fitness among elites
    float min_fit = INF;
    float max_fit = NINF;
    float min_compl = INF;
    float max_compl = NINF;
    vector<Node*> ordered_elites; ordered_elites.reserve(elites_per_complexity.size());
    vector<float> ordered_fitnesses; ordered_fitnesses.reserve(elites_per_complexity.size());
    vector<float> ordered_complexities; ordered_complexities.reserve(elites_per_complexity.size());

    for(auto it = elites_per_complexity.begin(); it != elites_per_complexity.end(); ++it) {

      float f = it->second->fitness;
      float c = it->first;
      
      if (f > max_fit)
        max_fit = f;
      if (f < min_fit)
        min_fit = f;
      if (c > max_compl)
        max_compl = c;
      if (c < min_compl)
        min_compl = c;
      
      ordered_elites.push_back(it->second);
      ordered_fitnesses.push_back(f);
      ordered_complexities.push_back(c);
    }

    // get best
    int best_idx = 0;
    float best_score = 0;
    for(int i = 0; i < ordered_fitnesses.size(); i++) {
      // normalize fitness & compl (with higher=better)
      float fit_score = 1.0 - (ordered_fitnesses[i] - min_fit)/(max_fit - min_fit);
      float compl_score = 1.0 - (ordered_complexities[i] - min_compl)/(max_compl - min_compl);

      // incl. penalty
      float score = (1.0 - rel_compl_importance) * fit_score + rel_compl_importance * compl_score;
      if (score > best_score) {
        best_score = score;
        best_idx = i;
      }
    }
    return ordered_elites[best_idx];
  }

  bool initialize_new_evolution() {
    // if it is the first evolution
    int pop_size;
    if (evolutions.empty()) {
      evolutions.reserve(10);
      pop_size = g::pop_size;
    } else {
      pop_size = evolutions[evolutions.size()-1]->population.size() * 2;
    }
    // skip if new pop.size is too large
    if (pop_size > MAX_POP_SIZE) {
      return false;
    }
    // or skip if options set not to use IMS and we already have 1 evolution
    if (g::disable_ims && evolutions.size() > 0) {
      return false;
    }
    Evolution * evo = new Evolution(pop_size);

    if (g::disable_ims && elites_per_complexity.size() > 0) {
      // if this was a re-start of the single population that converged before
      // inject a random elite by replacing a random solution

      // get random elite
      auto it = elites_per_complexity.begin();
      std::advance(it, randi(elites_per_complexity.size()));
      Node * an_elite = it->second;

      int repl_idx = randi(evo->population.size());
      evo->population[repl_idx]->clear();
      evo->population[repl_idx] = an_elite->clone();
      print(" + injecting an elite into re-started population");
    }

    evolutions.push_back(evo);
    print(" + init. new evolution with pop.size: ",pop_size);
    return true;
  }

  bool approximately_converged(Vec & fitnesses, float upper_quantile=0.9) {
    sort(fitnesses.data(), fitnesses.data() + fitnesses.size());
    if (fitnesses[0] == fitnesses[fitnesses.size() * upper_quantile]) {
      return true;
    }
    return false;
  }

  void terminate_obsolete_evolutions() {
    int largest_obsolete_idx = -1;
    for(int i = evolutions.size() - 1; i >= 0; i--) {
      auto fitnesses_i = g::fit_func->get_fitnesses(evolutions[i]->population, false);
      float med_fit_i = median(fitnesses_i);

      // if there is only one evolution & it converged, terminate it
      if (g::disable_ims && approximately_converged(fitnesses_i)) {
        largest_obsolete_idx = i;
      }

      for (int j = i-1; j >= 0; j--) {
        auto fitnesses_j = g::fit_func->get_fitnesses(evolutions[j]->population, false);
        float med_fit_j = median(fitnesses_j);
        if (med_fit_j > med_fit_i || approximately_converged(fitnesses_j)) {
          // will have to terminate j and previous
          largest_obsolete_idx = j;
          break;
        }
      }
      // got something, stop checking
      if (largest_obsolete_idx >= 0) {
        print(" - terminating evolutions with pop.size <= ", evolutions[largest_obsolete_idx]->pop_size);
        break;
      }
    }

    // terminate all previous
    for (int i = 0; i <= largest_obsolete_idx; i++) {
      // free memory
      delete evolutions[i];
    }
    // resize evolutions array
    if (largest_obsolete_idx > -1) {
      evolutions = vector<Evolution*>(evolutions.begin() + largest_obsolete_idx + 1, evolutions.end());
    }
  }

  void reset_elites() {
    for(auto it = elites_per_complexity.begin(); it != elites_per_complexity.end(); it++) {
      it->second->clear();
    }
    elites_per_complexity.clear();
  }

  void reevaluate_elites() {
    for(auto it = elites_per_complexity.begin(); it != elites_per_complexity.end(); it++) {
      g::fit_func->get_fitness(it->second);
    }
  }

  void update_elites(vector<Node*>& population) {
    for (Node * tree : population){
      float c = compute_complexity(tree);
      // determine if to insert this among elites and eliminate now-obsolete elites
      bool worse_or_equal_than_existing = false;
      vector<float> obsolete_complexities; obsolete_complexities.reserve(elites_per_complexity.size());
      for(auto it = elites_per_complexity.begin(); it != elites_per_complexity.end(); it++) {
        if (c >= it->first && tree->fitness >= it->second->fitness) {
          // this tree is equal or worse than an existing elite
          worse_or_equal_than_existing = true;
          break;
        }
        // check if a previous elite became obsolete
        if (c <= it->first && tree->fitness < it->second->fitness) {
          obsolete_complexities.push_back(it->first);
        }  
      }

      if (worse_or_equal_than_existing)
        continue;

      // remove obsolete elites
      for(float oc : obsolete_complexities) {
        elites_per_complexity[oc]->clear();
        elites_per_complexity.erase(oc);
      }

      // save this tree as a new elite
      elites_per_complexity[c] = tree->clone();
      //print("\tfound new equation with fitness ", tree->fitness, " and complexity ", c);
    }

  }

  void run() {

    auto start_time = tick();

    // initialize the first evolution
    initialize_new_evolution();
    
    bool stop = false;
    while(!stop) {

      // macro generation

      // update mini batch
      bool mini_batch_changed = g::fit_func->update_batch(g::batch_size);
      if (mini_batch_changed){
        reevaluate_elites();
      }

      int curr_num_evos = evolutions.size();
      for (int i = 0; i < curr_num_evos + 1; i++) {
        
        // check should stop
        if(g::_call_as_lib && PyErr_CheckSignals() == -1) {
          exit(1);
        }
        if (
          (g::max_generations > 0 && macro_generations == g::max_generations) ||
          (g::max_time > 0 && tock(start_time) >= g::max_time) ||
          (g::max_evaluations > 0 && g::fit_func->evaluations >= g::max_evaluations) ||
          (g::max_node_evaluations > 0 && g::fit_func->node_evaluations >= g::max_node_evaluations)
        ) {
          stop = true;
          break;
        }

        // find evo that must perform a generation
        bool should_perform_gen = false;
        if (i == 0 ||  evolutions[i-1]->gen_number > 0 && evolutions[i-1]->gen_number % SUB_GENs == 0){
          should_perform_gen = true;
          if (i > 0)
            evolutions[i-1]->gen_number = 0; // reset counter
        }

        if (!should_perform_gen)
          continue;

        // must be initialized
        if (i == evolutions.size()) {
          bool possible = initialize_new_evolution();
          if (!possible)
            continue;
        }

        // perform generation
        evolutions[i]->gomea_generation();

        // update elites
        update_elites(evolutions[i]->population);
        //print("\tperformed evo with pop.size: ",evolutions[i]->pop_size);
      }

      // decide if some evos should terminate
      terminate_obsolete_evolutions();

      // update macro gen
      macro_generations += 1;
      float curr_best_fit = select_elite(0.0)->fitness;
      print(" ~ macro generation: ", macro_generations, ", curr. best fit: ",curr_best_fit);
    }

    // finished

    // if abs corr, append linear scaling terms
    if (g::fit_func->name() == "ac") {
      for (auto it = elites_per_complexity.begin(); it != elites_per_complexity.end(); it++) {
        elites_per_complexity[it->first] = append_linear_scaling(it->second);
      }
    }

    if (!g::_call_as_lib) { // TODO: remove false
      print("\nAll elites found:");
      for (auto it = elites_per_complexity.begin(); it != elites_per_complexity.end(); it++) {
        print(it->first, " ", it->second->fitness, ":", it->second->human_repr());
      }
      print("\nBest w.r.t. complexity for chosen importance:");
      print(this->select_elite(g::rel_compl_importance)->human_repr());
    }
    
  }

};







#endif