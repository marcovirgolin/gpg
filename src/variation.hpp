#ifndef VARIATION_H
#define VARIATION_H

#include "globals.hpp"
#include "node.hpp"
#include "operator.hpp"
#include "util.hpp"

#include <vector>

using namespace std;


Node * _grow_tree_recursive(vector<Op*> functions, vector<Op*> terminals, int max_arity, int max_depth_left, int actual_depth_left, int curr_depth, float terminal_prob=.5) {
  Node * n = NULL;

  if (max_depth_left > 0) {
    if (actual_depth_left > 0 && randu() < 1.0-terminal_prob) {
      n = new Node(functions[randu() * functions.size()]->clone());
    } else {
      n = new Node(terminals[randu() * terminals.size()]->clone());
    }

    for (int i = 0; i < max_arity; i++) {
      Node * c = _grow_tree_recursive(functions, terminals, max_arity,
        max_depth_left - 1, actual_depth_left - 1, curr_depth + 1, terminal_prob);
      n->append(c);
    }
  } else {
    n = new Node(terminals[randu() * terminals.size()]->clone());
  }

  assert(n != NULL);

  return n;
}


Node * generate_tree(vector<Op*> functions, vector<Op*> terminals, int max_depth, string init_type="rhh") {

  int max_arity = 0;
  for(Op * op : functions) {
    int op_arity = op->arity();
    if (op_arity > max_arity)
      max_arity = op_arity;
  }

  Node * tree = NULL;
  int actual_depth = max_depth;

  if (init_type == "rhh" || init_type == "hh") {
    if (init_type == "rhh")
      actual_depth = randu() * max_depth;
    
    bool is_full = randu() < .5;

    if (is_full)
      tree = _grow_tree_recursive(functions, terminals, max_arity, max_depth, actual_depth, -1, 0.0);
    else
      tree = _grow_tree_recursive(functions, terminals, max_arity, max_depth, actual_depth, -1);

  } else {
    throw runtime_error("Unrecognized init_type "+init_type);
  }

  assert(tree);

  return tree;
}

Node * coeff_mut(Node * parent, bool return_copy=true) {
  Node * tree = parent;
  if (return_copy) {
    tree = parent->clone();
  }
  
  if (g::cmut_prob > 0 && g::cmut_temp > 0) {
    // apply coeff mut to all nodes that are constants
    vector<Node*> nodes = tree->subtree();
    for(Node * n : nodes) {
      if (
        n->op->type() == OpType::otConst &&
        randu() < g::cmut_prob
        ) {

        float c = ((Const*)n->op)->c;
        float std = g::cmut_temp*abs(c);
        if (std < g::cmut_eps)
          std = g::cmut_eps;
        float mutated_c = c * randn()*std; 
        ((Const*)n->op)->c = mutated_c;

      }
    }
  }

  return tree;
}

vector<int> _sample_crossover_mask(int num_nodes) {
  auto crossover_mask = rand_perm(num_nodes);
  int k = 1+sqrt(num_nodes)*abs(randn());
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
    if (nodes[i]->children.size() > 0 && randu() < prob_fun) {
      nodes[i]->op = functions[randu()*functions.size()]->clone();
    }
    else {
      nodes[i]->op = terminals[randu()*terminals.size()]->clone();
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

  auto random_fos_order = rand_perm(fos.size());

  for(int fos_idx = 0; fos_idx < fos.size(); fos_idx++) {
    
    auto crossover_mask = fos[random_fos_order[fos_idx]];
    bool change_is_meaningful = false;
    vector<Op*> backup_ops; backup_ops.reserve(crossover_mask.size());
    vector<int> effectively_changed_indices; effectively_changed_indices.reserve(crossover_mask.size());

    int where = randu()*population.size();
    if (where >= population.size())
      print("\t",population.size(), "idx:", where);
    Node * donor = population[where];
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
    coeff_mut(offspring, false);

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
    } else {
      // retain
      backup_fitness = new_fitness;
    }

    // discard backup
    for(Op * op : backup_ops) {
      delete op;
    }

  }
  
  return offspring;
}



#endif