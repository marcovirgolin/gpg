#ifndef EVOLUTION_H
#define EVOLUTION_H

#include "util.hpp"
#include "node.hpp"
#include "globals.hpp"

struct Evolution {

  vector<Node*> population;

  void run() {
    for(int i = 0; i < g::max_generations; i++) {
      print("gen: ",i+1);
    }
  }

};

#endif