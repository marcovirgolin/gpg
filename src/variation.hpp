#ifndef VARIATION_H
#define VARIATION_H

#include "globals.hpp"
#include "node.hpp"
#include "operator.hpp"
#include <vector>
#include <random>

using namespace std;


Node * _grow_tree_recursive(vector<Op*> functions, vector<Op*> terminals, int max_arity, int max_depth_left, int actual_depth_left, int curr_depth, float terminal_prob=.5) {
  Node * n = NULL;

  if (max_depth_left > 0) {

    if (actual_depth_left > 0 && rand() < terminal_prob) {
      n = new Node(functions[rand() * functions.size()]->clone());
    } else {
      n = new Node(terminals[rand() & terminals.size()]->clone());
    }

    for (size_t i = 0; i < max_arity; i++) {
      Node * c = _grow_tree_recursive(functions, terminals, max_arity,
        max_depth_left - 1, actual_depth_left - 1, curr_depth + 1);
      n->append(c);
    }

  } else {
    n = new Node(terminals[rand()*terminals.size()]->clone());
  }

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

  if (init_type == "rhh" || init_type == "hh") {
    if (init_type == "rhh")
      max_depth = rand() * max_depth;
    
    bool is_full = rand() < .5;

    if (is_full)
      tree = _grow_tree_recursive(functions, terminals, max_arity, max_depth, max_depth, -1, 0.0);
    else
      tree = _grow_tree_recursive(functions, terminals, max_arity, max_depth, max_depth, -1);

  } else {
    throw runtime_error("Unrecognized init_type "+init_type);
  }

  assert(tree);

  return tree;
}

#endif