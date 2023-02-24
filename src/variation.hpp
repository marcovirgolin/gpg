#ifndef VARIATION_H
#define VARIATION_H

#include "globals.hpp"
#include "node.hpp"
#include "operator.hpp"
#include "util.hpp"
#include "selection.hpp"
#include "fos.hpp"
#include "rng.hpp"

#include <vector>

using namespace std;

Op * _sample_operator(vector<Op *> & operators, Vec & cumul_probs) {
  double r = Rng::randu();
  int i = 0;
  while (r > (double) cumul_probs[i]) {
    i++;
  }
  return operators[i]->clone();  
}

Op * _sample_function() {
  return _sample_operator(g::functions, g::cumul_fset_probs);
}

Op * _sample_terminal() {
  return _sample_operator(g::terminals, g::cumul_tset_probs);
}

Node * _grow_tree_recursive(int max_arity, int max_depth_left, int actual_depth_left, int curr_depth, float terminal_prob=.25) {
  Node * n = NULL;

  if (max_depth_left > 0) {
    if (actual_depth_left > 0 && Rng::randu() < 1.0-terminal_prob) {
      n = new Node(_sample_function());
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
  for(Op * op : g::functions) {
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

Node * coeff_mut(Node * parent, bool return_copy=true, vector<int> * changed_indices = NULL, vector<Op*> * backup_ops = NULL) {
  Node * tree = parent;
  if (return_copy) {
    tree = parent->clone();
  }
  
  if (g::cmut_prob > 0 && g::cmut_temp > 0) {
    // apply coeff mut to all nodes that are constants
    vector<Node*> nodes = tree->subtree();
    for(int i = 0; i < nodes.size(); i++) {
      Node * n = nodes[i];
      if (
        n->op->type() == OpType::otConst &&
        Rng::randu() < g::cmut_prob
        ) {

        float prev_c = ((Const*)n->op)->c;
        float std = g::cmut_temp*abs(prev_c);
        if (std < g::cmut_eps)
          std = g::cmut_eps;
        float mutated_c = roundd(prev_c + Rng::randn()*std, NUM_PRECISION); 
        ((Const*)n->op)->c = mutated_c;
        if (changed_indices != NULL) {
          changed_indices->push_back(i);
          backup_ops->push_back(new Const(prev_c));
        }
      }
    }
  }
  return tree;
}

vector<int> _sample_crossover_mask(int num_nodes) {
  auto crossover_mask = Rng::rand_perm(num_nodes);
  int k = 1+sqrt(num_nodes)*abs(Rng::randn());
  k = min(k, num_nodes);
  crossover_mask.erase(crossover_mask.begin() + k, crossover_mask.end());
  assert(crossover_mask.size() == k);
  return crossover_mask;
}

Node * crossover(Node * parent, Node * donor) {
  Node * offspring = parent->clone();
  auto nodes = offspring->subtree();
  auto d_nodes = donor->subtree();

  // sample a crossover mask
  auto crossover_mask = _sample_crossover_mask(nodes.size());
  for(int i : crossover_mask) {
    delete nodes[i]->op;
    nodes[i]->op = d_nodes[i]->op->clone();
  }

  return offspring;
}

Node * mutation(Node * parent, vector<Op*> & functions, vector<Op*> & terminals, float prob_fun = 0.75) {
  Node * offspring = parent->clone();
  auto nodes = offspring->subtree();

  // sample a crossover mask
  auto crossover_mask = _sample_crossover_mask(nodes.size());
  for(int i : crossover_mask) {
    delete nodes[i]->op;
    if (nodes[i]->children.size() > 0 && Rng::randu() < prob_fun) {
      nodes[i]->op = _sample_function();
    }
    else {
      nodes[i]->op = _sample_terminal();
    }
  }

  return offspring;
}

/*Node * gom(Node * parent, vector<Node*> & population, vector<vector<int>> & fos) {
  Node * offspring = parent->clone();
  Node * backup = parent->clone();

  float backup_fitness = backup->fitness;

  vector<Node*> offspring_nodes = offspring->subtree();

  auto random_fos_order = rand_perm(fos.size());

  for(int i = 0; i < fos.size(); i++) {
    auto crossover_mask = fos[random_fos_order[i]];
    // fetch donor
    Node * donor = population[randu()*population.size()];
    vector<Node*> donor_nodes = donor->subtree();

    for(int & idx : crossover_mask) {
      delete offspring_nodes[idx]->op;
      offspring_nodes[idx]->op = donor_nodes[idx]->op->clone();
    }

    // apply coeff mut
    coeff_mut(offspring, false);

    // check is not worse
    float new_fitness = g::fit_func->get_fitness(offspring);
    if (new_fitness > backup_fitness) {
      // undo
      offspring->clear();
      offspring = backup->clone();
      offspring_nodes = offspring->subtree();
    } else {
      // retain
      backup->clear();
      backup = offspring->clone();
      backup_fitness = new_fitness;
    }
  }
  
  return offspring;
}*/


Node * efficient_gom(Node * parent, vector<Node*> & population, vector<vector<int>> & fos) {
  Node * offspring = parent->clone();
  float backup_fitness = parent->fitness;
  vector<Node*> offspring_nodes = offspring->subtree();

  auto random_fos_order = Rng::rand_perm(fos.size());

  bool ever_improved = false;
  for(int fos_idx = 0; fos_idx < fos.size(); fos_idx++) {
    
    auto crossover_mask = fos[random_fos_order[fos_idx]];
    bool change_is_meaningful = false;
    vector<Op*> backup_ops; backup_ops.reserve(crossover_mask.size());
    vector<int> effectively_changed_indices; effectively_changed_indices.reserve(crossover_mask.size());

    Node * donor = population[Rng::randi(population.size())];
    vector<Node*> donor_nodes = donor->subtree();

    for(int & idx : crossover_mask) {
      // check if swap is not necessary
      if (offspring_nodes[idx]->op->sym() == donor_nodes[idx]->op->sym()) {
        // might need to swap if the node is a constant that might be optimized
        if (g::cmut_prob <= 0 || g::cmut_temp <= 0 || donor_nodes[idx]->op->type() != OpType::otConst)
          continue;
      }

      // then execute the swap
      Op * replaced_op = offspring_nodes[idx]->op;
      offspring_nodes[idx]->op = donor_nodes[idx]->op->clone();
      backup_ops.push_back(replaced_op);
      effectively_changed_indices.push_back(idx);
    }

    // apply coeff mut
    coeff_mut(offspring, false, &effectively_changed_indices, &backup_ops);

    // check if at least one change was meaningful
    for(int i : effectively_changed_indices) {
      Node * n = offspring_nodes[i];
      if (!n->is_intron()) {
        change_is_meaningful = true;
        break;
      }
    }

    // assume nothing changed
    float new_fitness = backup_fitness;
    if (change_is_meaningful) {
      // gotta recompute
      new_fitness = g::fit_func->get_fitness(offspring);
    }

    // check is not worse
    if (new_fitness > backup_fitness) {
      // undo
      for(int i = 0; i < effectively_changed_indices.size(); i++) {
        int changed_idx = effectively_changed_indices[i];
        Node * off_n = offspring_nodes[changed_idx];
        Op * back_op = backup_ops[i];
        delete off_n->op;
        off_n->op = back_op->clone();
        offspring->fitness = backup_fitness;
      }
    } else if (new_fitness < backup_fitness) {
      // it improved
      backup_fitness = new_fitness;
      ever_improved = true;
    }

    // discard backup
    for(Op * op : backup_ops) {
      delete op;
    }

  }

  // variant of forced improvement that is potentially less aggressive, & less expensive to carry out
  if(g::tournament_size > 1 && !ever_improved) {
    // make a tournament between tournament size - 1 candidates + offspring
    vector<Node*> tournament_candidates; tournament_candidates.reserve(g::tournament_size - 1);
    for(int i = 0; i < g::tournament_size - 1; i++) {
      tournament_candidates.push_back(population[Rng::randi(population.size())]);
    }
    tournament_candidates.push_back(offspring);
    Node * winner = tournament(tournament_candidates, g::tournament_size);
    offspring->clear();
    offspring = winner;
  }
  
  return offspring;
}

Node * append_linear_scaling(Node * tree) {
  // compute intercept and scaling coefficients, append them to the root
  Node * add_n, * mul_n, * slope_n, * interc_n;

  Vec p = tree->get_output(g::fit_func->X_train);

  pair<float,float> intc_slope = linear_scaling_coeffs(g::fit_func->y_train, p);
  
  if (intc_slope.second == 0){
    add_n = new Node(new Add());
    interc_n = new Node(new Const(intc_slope.first));
    add_n->append(interc_n);
    add_n->append(tree);
    return add_n;
  }

  mul_n = new Node(new Mul());
  slope_n = new Node(new Const(intc_slope.second));
  add_n = new Node(new Add());
  interc_n = new Node(new Const(intc_slope.first));
  mul_n->append(slope_n);
  mul_n->append(tree);
  add_n->append(interc_n);
  add_n->append(mul_n);

  // bring fitness info to new root
  add_n->fitness = tree->fitness;

  return add_n;

}



#endif