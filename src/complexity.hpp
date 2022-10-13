#ifndef COMPLEXITY_H
#define COMPLEXITY_H

#include "node.hpp"
#include "operator.hpp"
#include "globals.hpp"

float compute_complexity(Node * tree) {
  if (g::complexity_type == "node_count") {
    return tree->get_num_nodes(true);
  } 
  throw std::runtime_error("Unrecognized complexity type: " + g::complexity_type);
}


#endif