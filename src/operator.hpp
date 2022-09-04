#ifndef OPERATOR_H
#define OPERATOR_H

#include "myeig.hpp"

using namespace std;
using namespace myeig;

struct Op {

  virtual ~Op(){};

  virtual Op * clone() {
    throw runtime_error("Not implemented");  
  }

  virtual int arity() {
    throw runtime_error("Not implemented");
  }

  virtual string sym() {
    throw runtime_error("Not implemented");
  }

  virtual Vec apply(Mat X) {
    throw runtime_error("Not implemented");
  }

};

struct Add : Op {

  Op * clone() override {
    return new Add();
  }

  int arity() override {
    return 2;
  }

  string sym() override {
    return "+";
  }

  Vec apply(Mat X) override {
    return X.rowwise().sum();
  }

};

struct Neg : Op {

  Op * clone() override {
    return new Neg();
  }

  int arity() override {
    return 1;
  }

  string sym() override {
    return "¬";
  }

  Vec apply(Mat X) override {
    return -X.col(0);
  }

};

struct Sub : Op {

  Op * clone() override {
    return new Sub();
  }

  int arity() override {
    return 2;
  }

  string sym() override {
    return "-";
  }

  Vec apply(Mat X) override {
    return X.col(0)-X.col(1);
  }

};

struct Mul : Op {

  Op * clone() override {
    return new Mul();
  }

  int arity() override {
    return 2;
  }

  string sym() override {
    return "*";
  }

  Vec apply(Mat X) override {
    return X.col(0)*X.col(1);
  }

};

struct Inv : Op {

  Op * clone() override {
    return new Inv();
  }

  int arity() override {
    return 1;
  }

  string sym() override {
    return "1/";
  }

  Vec apply(Mat X) override {
    return 1/X.col(0);
  }

};

struct Div : Op {

  Op * clone() override {
    return new Div();
  }

  int arity() override {
    return 2;
  }

  string sym() override {
    return "/";
  }

  Vec apply(Mat X) override {
    return X.col(0)/X.col(1);
  }

};

struct Feat : Op {

  int id;
  Feat(int id) {
    this->id = id;
  }

  Op * clone() override {
    return new Feat(this->id);
  }

  int arity() override {
    return 0;
  }

  string sym() override {
    return "x_"+to_string(id);
  }

  Vec apply(Mat X) override {
    return X.col(id);
  }

};

#endif