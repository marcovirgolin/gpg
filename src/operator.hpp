#ifndef OPERATOR_H
#define OPERATOR_H

#include "myeig.hpp"
#include "util.hpp"

using namespace std;
using namespace myeig;

enum OpType {
  otFun, otFeat, otConst
};

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

  virtual OpType type() {
    throw runtime_error("Not implemented");
  }

  virtual Vec apply(Mat & X) {
    throw runtime_error("Not implemented");
  }

};

struct Fun : Op {
  OpType type() override {
    return OpType::otFun;
  }
};

struct Add : Fun {

  Op * clone() override {
    return new Add();
  }

  int arity() override {
    return 2;
  }

  string sym() override {
    return "+";
  }

  Vec apply(Mat & X) override {
    return X.rowwise().sum();
  }

};

struct Neg : Fun {

  Op * clone() override {
    return new Neg();
  }

  int arity() override {
    return 1;
  }

  string sym() override {
    return "¬";
  }

  Vec apply(Mat & X) override {
    return -X.col(0);
  }

};

struct Sub : Fun {

  Op * clone() override {
    return new Sub();
  }

  int arity() override {
    return 2;
  }

  string sym() override {
    return "-";
  }

  Vec apply(Mat & X) override {
    return X.col(0)-X.col(1);
  }

};

struct Mul : Fun {

  Op * clone() override {
    return new Mul();
  }

  int arity() override {
    return 2;
  }

  string sym() override {
    return "*";
  }

  Vec apply(Mat & X) override {
    return X.col(0)*X.col(1);
  }

};

struct Inv : Fun {

  Op * clone() override {
    return new Inv();
  }

  int arity() override {
    return 1;
  }

  string sym() override {
    return "1/";
  }

  Vec apply(Mat & X) override {
    // division by 0 is undefined thus conver to NAN
    Vec denom = X.col(0);
    replace(denom, 0, NAN);
    return 1/denom;
  }

};

struct Div : Fun {

  Op * clone() override {
    return new Div();
  }

  int arity() override {
    return 2;
  }

  string sym() override {
    return "/";
  }

  Vec apply(Mat & X) override {
    // division by 0 is undefined thus conver to NAN
    Vec denom = X.col(1);
    replace(denom, 0, NAN);
    return X.col(0)/denom;
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

  OpType type() override {
    return OpType::otFeat;
  }

  Vec apply(Mat & X) override {
    return X.col(id);
  }

};

struct Const : Op {

  float c;
  Const(float c=NAN) {
    this->c=NAN;
  }

  Op * clone() override {
    return new Const(this->c);
  }

  void _sample() {
    this->c = randu()*10 - 5;
  }

  int arity() override {
    return 0;
  }

  string sym() override {
    if (isnan(c))
      _sample();
    return to_string(c);
  }

  OpType type() override {
    return OpType::otConst;
  }

  Vec apply(Mat & X) override {
    if (isnan(c))
      _sample();
    Vec c_vec = Vec::Constant(X.rows(), c);
    return c_vec;
  }

};

#endif