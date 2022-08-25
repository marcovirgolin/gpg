#ifndef TESTS_H
#define TESTS_H

#include "util.hpp"
#include "node.hpp"
#include "operator.hpp"

struct Test {

  void run_all() {
    depth();
    subtree();
    operators();
  }

  void depth() {
    Node n1;
    Node n2;
    Node n3;

    n3.parent = &n2;
    n2.parent = &n1;

    assert(n1.get_depth() == 0);
    assert(n2.get_depth() == 1);
    assert(n3.get_depth() == 2);
  }

  void subtree() {
    Node n1, n2, n3, n4, n5;
    n1.fitness = 1;
    n2.fitness = 2;
    n3.fitness = 3;
    n4.fitness = 4;
    n5.fitness = 5;

    n1.children.push_back(&n2);
    n2.children.push_back(&n3);
    n2.children.push_back(&n5);
    n1.children.push_back(&n4);

    auto subtree = n1.get_subtree();
    string collected_fitnesses = "";
    for(Node * n : subtree) {
      collected_fitnesses += to_string((int)n->fitness);
    }
    
    assert(collected_fitnesses == "12354");
  }

  void operators() {
    
    auto op = Add();
    assert(op.sym == "+");
    assert(op.apply(1,2) == 3);
    
    auto op2 = Sub();
    assert(op2.sym == "-");
    assert(op2.apply(2,3)==-1);
  }

};


#endif