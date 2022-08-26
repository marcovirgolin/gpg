#ifndef TESTS_H
#define TESTS_H

#include "util.hpp"
#include "node.hpp"
#include "operator.hpp"
#include "myeig.hpp"

using namespace std;
using namespace myeig;

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

    // Generic ref
    Op * op;

    // Toy data
    Mat X(3,2);
    X << 1, 2,
         3, 4,
         5, 6;
    Vec expected(3);
    Vec result(3);

    // Add
    op = new Add();
    assert(op->sym() == "+");
    expected << 3, 7, 11;
    result = op->apply(X);
    assert(result.isApprox(expected));
    delete op;

    // Sub
    op = new Sub();
    print(op->sym());
    expected << -1, -1, -1;
    result = op->apply(X);
    delete op;

    // Neg
    op = new Neg();
    print(op->sym());
    expected << -1, -3, -5;
    result = op->apply(X.col(0));
    delete op;

    
  }

};


#endif