#ifndef COMPLEXITY_H
#define COMPLEXITY_H

#include "node.hpp"
#include "operator.hpp"
#include "globals.hpp"

float compute_nodecount(Node * tree) {
  auto nodes = tree->subtree();
  int count_introns = 0;
  for (auto * n : nodes)
    if (n->is_intron()) 
      count_introns++;
  return nodes.size() - count_introns;
}

float compute_complexity(Node * tree) {
  if (g::complexity_type == "node_count") {
    return compute_nodecount(tree);
  } 
  throw std::runtime_error("Unrecognized complexity type: " + g::complexity_type);
}


#endif