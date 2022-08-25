#ifndef EVOLUTION_H
#define EVOLUTION_H

#include "util.hpp"
#include "node.hpp"

struct Evolution {

  vector<Node*> population;

  void run() {
    for(int i = 0; i < 10; i++) {
      print("gen: ",i);
    }
  }

};

#endif