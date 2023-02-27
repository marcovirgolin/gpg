#ifndef VARIATION_H
#define VARIATION_H

#include "globals.hpp"
#include "individual.hpp"
#include "operator.hpp"
#include "util.hpp"
#include "selection.hpp"
#include "fos.hpp"
#include "rng.hpp"

#include <vector>

using namespace std;

namespace variation {

  string _sample_component(const vector<string> & components, Vec & cumul_probs) {
    double r = Rng::randu();
    int i = 0;
    while (r > (double) cumul_probs[i]) {
      i++;
    }
    return components[i];  
  }

  string _sample_operator() {
    return _sample_component(g::operators, g::cumul_fset_probs);
  }

  string _sample_terminal() {
    string t = _sample_component(g::terminals, g::cumul_tset_probs);
    if (t == "erc") {
      t = to_string(roundd(Rng::randu()*10 - 5, NUM_PRECISION));
    }
    return t;
  }

  void _gen_indiv_recursive(vector<string> & genome, int max_arity, int max_depth_left, int actual_depth_left, float terminal_prob=.25) {
    if (max_depth_left > 0) {
      if (actual_depth_left > 0 && Rng::randu() < 1.0-terminal_prob) {
        genome.push_back(_sample_operator());
      } else {
        genome.push_back(_sample_terminal());
      }
      for(int i = 0; i < max_arity; i++){
        _gen_indiv_recursive(genome, max_arity, max_depth_left-1, actual_depth_left-1, terminal_prob);
      }
    } else {
      genome.push_back(_sample_terminal());
    }
  }

  Individual * generate_indiv(int max_depth, string init_type="hh") {

    // compute total number of components
    int max_arity = 0;
    for(auto it = all_operators.begin(); it != all_operators.end(); it++) {
      if (it->second.second > max_arity){
        max_arity = it->second.second;
      }
    }
    int tot_num_components = (pow(max_arity, max_depth + 1) - 1) / (max_arity - 1);

    Individual * indiv = new Individual(g::max_arity);
    indiv->genome.reserve(tot_num_components+1);

    // build individual
    int actual_depth = max_depth;
    if (init_type == "rhh" || init_type == "hh") {
      if (init_type == "rhh")
        actual_depth = Rng::randu() * max_depth;
      
      bool is_full = Rng::randu() < .5;

      if (is_full)
        _gen_indiv_recursive(indiv->genome, max_arity, max_depth, actual_depth, 0.0);
      else
        _gen_indiv_recursive(indiv->genome, max_arity, max_depth, actual_depth);

    } else {
      throw runtime_error("Unrecognized init_type "+init_type);
    }

    return indiv;
  }


/* void setPTChildNode(short * building_individual, int current_height, bool full, bool use_bbs, int GOMEA_index) {
      pt_index_trace++;

      const vector<short> * functions;
      const vector<short> * terminals;
      functions = &FUNCTIONS;
      terminals = &TERMINALS;

      if (current_height > 0) {
          if (full || current_height == max_depths[GOMEA_index] || randomRealUniform01() >= 0.5) {
              building_individual[pt_index_trace] = functions->at(randomInt(functions->size()));
              for (int i = 0; i < maximum_arity; i++) {
                  setPTChildNode(building_individual, current_height - 1, full, use_bbs, GOMEA_index);
              }
          } else {
              building_individual[pt_index_trace] = terminals->at(randomInt(terminals->size()));
              for (int i = 0; i < maximum_arity; i++) {
                  setPTChildNode(building_individual, current_height - 1, full, use_bbs, GOMEA_index);
              }
          }
      } else {
          building_individual[pt_index_trace] = terminals->at(randomInt(terminals->size()));
      }
  }


  Node * _grow_tree_recursive(int max_arity, int max_depth_left, int actual_depth_left, int curr_depth, float terminal_prob=.25) {
    Node * n = NULL;

    if (max_depth_left > 0) {
      if (actual_depth_left > 0 && Rng::randu() < 1.0-terminal_prob) {
        n = new Node(_sample_operator());
      } else {
        n = new Node(_sample_terminal());
      }

      for (int i = 0; i < max_arity; i++) {
        Node * c = _grow_tree_recursive(max_arity,
          max_depth_left - 1, actual_depth_left - 1, curr_depth + 1, terminal_prob);
        n->append(c);
      }
    } else {
      n = new Node(_sample_terminal());
    }

    assert(n != NULL);

    return n;
  }

  Node * generate_tree(int max_depth, string init_type="hh") {

    int max_arity = 0;
    for(Op * op : g::operators) {
      int op_arity = op->arity();
      if (op_arity > max_arity)
        max_arity = op_arity;
    }

    Node * tree = NULL;
    int actual_depth = max_depth;

    if (init_type == "rhh" || init_type == "hh") {
      if (init_type == "rhh")
        actual_depth = Rng::randu() * max_depth;
      
      bool is_full = Rng::randu() < .5;

      if (is_full)
        tree = _grow_tree_recursive(max_arity, max_depth, actual_depth, -1, 0.0);
      else
        tree = _grow_tree_recursive(max_arity, max_depth, actual_depth, -1);

    } else {
      throw runtime_error("Unrecognized init_type "+init_type);
    }

    assert(tree);

    return tree;
  }
  */

  set<int> coeff_mut(Individual * indiv) {
    // NOTE: this method changes constants IN PLACE, does not create a copy of the individual

    set<int> changed_indices;
    
    if (g::cmut_prob > 0 && g::cmut_temp > 0) {
      // apply coeff mut to all components that are constants
      for(int i = 0; i < indiv->genome.size(); i++) {
        string sym = indiv->genome[i];
        // check it is a constant & should be mutated
        if (all_operators.count(sym) || sym[0] == 'x' || Rng::randu() >= g::cmut_prob) {
          continue;
        }
        float c = stof(sym);
        float std = g::cmut_temp*abs(c);
        if (std < g::cmut_eps)
          std = g::cmut_eps;
        float mutated_c = roundd(c + Rng::randn()*std, NUM_PRECISION); 
        indiv->genome[i] = to_string(mutated_c);
        changed_indices.insert(i);
      }
    }
    return changed_indices;
  }

  vector<int> _sample_crossover_mask(int num_components) {
    auto crossover_mask = Rng::rand_perm(num_components);
    int k = 1+sqrt(num_components)*abs(Rng::randn());
    k = min(k, num_components);
    crossover_mask.erase(crossover_mask.begin() + k, crossover_mask.end());
    assert(crossover_mask.size() == k);
    return crossover_mask;
  }

  Individual * crossover(Individual * parent, Individual * donor) {
    Individual * indiv = parent->clone();
    
    // sample a crossover mask
    auto crossover_mask = _sample_crossover_mask(indiv->genome.size());
    for(int i : crossover_mask) {
      indiv->genome[i] = donor->genome[i];
    }

    return indiv;
  }

  Individual * mutation(Individual * parent, float prob_op = 0.75) {
    Individual * indiv = parent->clone();

    // sample a crossover mask
    auto crossover_mask = _sample_crossover_mask(indiv->genome.size());
    for(int i : crossover_mask) {
      string sym = indiv->genome[i];
      if (all_operators.count(sym) && Rng::randu() < prob_op) {
        indiv->genome[i] = _sample_operator();
      }
      else {
        indiv->genome[i] = _sample_terminal();
      }
    }

    return indiv;
  }

/*
  Individual * gom(Individual * parent, const vector<Individual*> & population, vector<vector<int>> & fos) {
    Individual * indiv = parent->clone();
    vector<string> backup_genome = parent->genome;
    float backup_fitness = parent->fitness;

    auto random_fos_order = Rng::rand_perm(fos.size());

    bool ever_improved = false;
    for(int i = 0; i < fos.size(); i++) {
      auto crossover_mask = fos[random_fos_order[i]];
      // fetch donor
      vector<string> donor_genome = population[Rng::randi(population.size())]->genome;

      bool something_changed = false;
      for(int & idx : crossover_mask) {
        if (indiv->genome[idx] == donor_genome[idx]) {
          continue;
        }
        something_changed = true;
        // TODO: instead of 'something changed', store which indices changed
        // then, implement (recursive, in Individual) a method to get which indices are active
        // check that at least one that changed is active to avoid unnecessary evaluations!
        indiv->genome[idx] = donor_genome[idx];
      }

      if (!something_changed) {
        // for efficiency, no coeff mut if nothing changed
        continue;
      }

      // apply coeff mut
      coeff_mut(indiv, false);

      // check is not worse
      float new_fitness = g::fit_func->get_fitness(indiv);
      if (new_fitness > backup_fitness) {
        // undo
        indiv->genome = backup_genome;
        indiv->fitness = backup_fitness;
      } else {
        // retain
        backup_genome = indiv->genome;
        backup_fitness = new_fitness;
      }
    }

    // variant of forced improvement that is potentially less aggressive, & less expensive to carry out
    if (g::tournament_size > 1 && !ever_improved) {
      // make a tournament between tournament size - 1 candidates + indiv
      vector<Individual*> tournament_candidates; tournament_candidates.reserve(g::tournament_size - 1);
      for(int i = 0; i < g::tournament_size - 1; i++) {
        tournament_candidates.push_back(population[Rng::randi(population.size())]);
      }
      tournament_candidates.push_back(indiv);
      Individual * winner = selection::tournament(tournament_candidates, g::tournament_size);
      // copy that
      indiv->genome = winner->genome;
      indiv->fitness = winner->fitness;
      delete winner;
    }
    
    return indiv;
  }
  */

 Individual * gom(Individual * parent, const vector<Individual*> & population, vector<vector<int>> & fos) {
    Individual * indiv = parent->clone();
    vector<string> backup_genome = parent->genome;
    float backup_fitness = parent->fitness;

    auto random_fos_order = Rng::rand_perm(fos.size());

    bool ever_improved = false;
    for(int i = 0; i < fos.size(); i++) {
      auto crossover_mask = fos[random_fos_order[i]];
      // fetch donor
      vector<string> donor_genome = population[Rng::randi(population.size())]->genome;

      set<int> changed_indices;
      for(int & idx : crossover_mask) {
        if (indiv->genome[idx] == donor_genome[idx]) {
          continue;
        }
        indiv->genome[idx] = donor_genome[idx];
        changed_indices.insert(idx);
      }

      // apply coeff mut
      set<int> coeff_mutated_indices = coeff_mut(indiv);
      // update changed indices
      set_union(changed_indices.begin(), changed_indices.end(), coeff_mutated_indices.begin(), coeff_mutated_indices.end(), 
        inserter(changed_indices, changed_indices.begin()));

      // check if somethnig changed
      bool something_changed = false;
      set<int> active_indices = indiv->get_active_indices();
      for(int idx : changed_indices) {
        if (active_indices.count(idx)) {
          something_changed = true;
          break;
        }
      }
      if (!something_changed) {
        // keep diversity in genome
        backup_genome = indiv->genome;
        continue; // no need to re-evaluate
      }

      // check is not worse
      float new_fitness = g::fit_func->get_fitness(indiv);
      if (new_fitness > backup_fitness) {
        // undo
        indiv->genome = backup_genome;
        indiv->fitness = backup_fitness;
      } else {
        // retain
        if (new_fitness < backup_fitness) {
          ever_improved = true;
          backup_fitness = new_fitness;
        }
        backup_genome = indiv->genome;
      }
    }

    // variant of forced improvement that is potentially less aggressive, & less expensive to carry out
    if (g::tournament_size > 1 && !ever_improved) {
      // make a tournament between tournament size - 1 candidates + indiv
      vector<Individual*> tournament_candidates; tournament_candidates.reserve(g::tournament_size - 1);
      for(int i = 0; i < g::tournament_size - 1; i++) {
        tournament_candidates.push_back(population[Rng::randi(population.size())]);
      }
      tournament_candidates.push_back(indiv);
      Individual * winner = selection::tournament(tournament_candidates, g::tournament_size);
      // copy that
      indiv->genome = winner->genome;
      indiv->fitness = winner->fitness;
      delete winner;
    }
    
    return indiv;
  }

  void set_linear_scaling(Individual * indiv) {
    // compute intercept and scaling coefficients, set them
    Vec p = indiv->eval(g::fit_func->X_train);

    pair<float,float> intc_slope = linear_scaling_coeffs(g::fit_func->y_train, p);
    
    indiv->linear_scaling_intercept = intc_slope.first;
    indiv->linear_scaling_slope = intc_slope.second;
  }

}


#endif