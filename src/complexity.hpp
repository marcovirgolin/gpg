#ifndef COMPLEXITY_H
#define COMPLEXITY_H

#include "individual.hpp"
#include "operator.hpp"
#include "globals.hpp"

float compute_complexity(Individual * indiv) {
  if (g::complexity_type == "component_count") {
    return indiv->get_num_components();
  } 
  throw std::runtime_error("Unrecognized complexity type: " + g::complexity_type);
}


#endif