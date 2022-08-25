#ifndef OPERATOR_H
#define OPERATOR_H

using namespace std;

struct Op {

  string sym = "none";

  int apply(int a, int b) {
    throw runtime_error("Not implemented");
  }

};

struct Add : Op {

  string sym = "+";

  int apply(int a, int b) {
    return a + b;
  }

};

struct Sub : Op {

  string sym = "-";

  int apply(int a, int b) {
    return a - b;
  }

};

#endif