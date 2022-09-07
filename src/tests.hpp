#ifndef TESTS_H
#define TESTS_H

#include "myeig.hpp"
#include "util.hpp"
#include "node.hpp"
#include "operator.hpp"
#include "fitness.hpp"
#include "variation.hpp"

using namespace std;
using namespace myeig;

struct Test {

  void run_all() {
    depth();
    subtree();
    gen_tree();
    operators();
    node_output();
    fitness();
    math();
  }

  Node * _generate_mock_tree() {
    // Builds x_0 * (x_1 + x_1) aka [* x_0 + x_1 x_1]
    Node * add_node = new Node(new Add());

    Node * mul_node = new Node(new Mul());

    Node * feat0_node = new Node(new Feat(0));

    Node * feat1_node = new Node(new Feat(1));

    Node * feat1_node2 = feat1_node->clone();

    add_node->append(feat1_node);
    add_node->append(feat1_node2);

    mul_node->append(feat0_node);
    mul_node->append(add_node);

    return mul_node;
  }

  void depth() {
    Node n1(new Add());
    Node n2(new Add());
    Node n3(new Add());

    n3.parent = &n2;
    n2.parent = &n1;

    assert(n1.depth() == 0);
    assert(n2.depth() == 1);
    assert(n3.depth() == 2);
  }

  void subtree() {
    Node n1(new Add()), n2(new Add()), n3(new Add()), n4(new Add()), n5(new Add());
    n1.fitness = 1;
    n2.fitness = 2;
    n3.fitness = 3;
    n4.fitness = 4;
    n5.fitness = 5;

    n1.children.push_back(&n2);
    n2.children.push_back(&n3);
    n2.children.push_back(&n5);
    n1.children.push_back(&n4);

    auto subtree = n1.subtree();
    string collected_fitnesses = "";
    for(Node * n : subtree) {
      collected_fitnesses += to_string((int)n->fitness);
    }
    
    assert(collected_fitnesses == "12354");

  }

  void gen_tree() {
    vector<Op*> functions = {new Add(), new Sub(), new Mul()};
    vector<Op*> terminals = {new Feat(0), new Feat(1)};
    for(int height = 10; height >= 0; height--){
      // create full trees, check their height is correct
      for(int trial=0; trial < 10; trial++){
        auto * t = _grow_tree_recursive(functions, terminals, 2, height, height, -1, 0.0);
        assert(t->height() == height);
        t->clear();
      }
    }
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
    expected << -1, -1, -1;
    result = op->apply(X);
    delete op;

    // Neg
    op = new Neg();
    expected << -1, -3, -5;
    Mat temp = X.col(0);
    result = op->apply(temp);
    delete op;
    
  }

  void node_output() {
    // Toy data
    Mat X(3,2);
    X << 1, 2,
         3, 4,
         5, 6;
    Vec expected(3);
    Vec result(3);

    Node * mock_tree = _generate_mock_tree();
    expected << 4, 24, 60;
    result = mock_tree->get_output(X);
    mock_tree->clear();

    assert(result.isApprox(expected));
  }

  void fitness() {
    auto * mock_tree = _generate_mock_tree();

    Mat X(3,2);
    X << 1, 2,
         3, 4,
         5, 6;
    Vec y(3);
    y << 1, 0, 1;

    Vec out = mock_tree->get_output(X);
    // out = [4, 24, 60]

    // mae = mean(|1-4|,|0-24|,|1-60|)
    float expected = (3+24+59)/3.0;

    Fitness * f = new MAEFitness();
    float res = f->get_fitness(mock_tree, X, y);

    assert(res == expected);

  }

  void math() {
    // correlation
    Vec x(5); 
    x << 1, 2, 3, 4, 5;
    Vec y(5);
    y << 2, 4, 6, 8, 10;

    assert( corr(x, y) == 1.0 );

    y = -1 * y;
    assert( corr(x, y) == -1.0 );

    y << 0, 0, 0, 0, 0;
    assert( corr(x, y) == 0.0 );

    // nan & infs
    assert(isnan(NAN));
    assert(INF > 99999999.9);
    assert(NINF < -9999999.9);
  }

};


#endif