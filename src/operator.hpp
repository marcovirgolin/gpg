#ifndef OPERATOR_H
#define OPERATOR_H

#include "myeig.hpp"
#include "util.hpp"

using namespace std;
using namespace myeig;

// IMPORTANT: All operators need to be defined in globals.h's `all_operators` field to be accessible

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

  virtual string human_repr(vector<string> & args) {
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

  string _human_repr_binary_between(vector<string> & args) {
    return "(" + args[0] + this->sym() + args[1] + ")";
  }

  string _human_repr_unary_before(vector<string> & args) {
    return this->sym()+ "(" + args[0] + ")";
  }

  string _human_repr_common(vector<string> & args) {
    if (arity() == 2) {
      return _human_repr_binary_between(args);
    } else if (arity() == 1) {
      return _human_repr_unary_before(args);
    } else {
      throw runtime_error("Not implemented");
    }
  }

  virtual string human_repr(vector<string> & args) override {
    return _human_repr_common(args);
  }
};

struct Term : Op {
  virtual string human_repr(vector<string> & args) override {
    return sym();
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
    // division by 0 is undefined thus convert to NAN
    Vec denom = X.col(1);
    replace(denom, 0, NAN);
    return X.col(0)/denom;
  }

};

struct Sin : Fun {

  Op * clone() override {
    return new Sin();
  }

  int arity() override {
    return 1;
  }

  string sym() override {
    return "sin";
  }

  Vec apply(Mat & X) override {
    return X.sin();
  }

};

struct Cos : Fun {

  Op * clone() override {
    return new Cos();
  }

  int arity() override {
    return 1;
  }

  string sym() override {
    return "cos";
  }

  Vec apply(Mat & X) override {
    return X.cos();
  }

};

struct Log : Fun {

  Op * clone() override {
    return new Log();
  }

  int arity() override {
    return 1;
  }

  string sym() override {
    return "log";
  }

  Vec apply(Mat & X) override {
    // Log of x < 0 is undefined (and =0 is -inf) thus convert to NAN
    Vec x = X.col(0);
    replace(x, 0, NAN, "<=");
    return x.log();
  }

};

struct Feat : Term {

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

struct Const : Term {

  float c;
  Const(float c=NAN) {
    this->c=c;
    if (abs(this->c) < 1e-6) {
      this->c = 0;
    }
  }

  Op * clone() override {
    return new Const(this->c);
  }

  void _sample() {
    this->c = randu()*10 - 5;
    if (abs(this->c) < 1e-6) {
      this->c = 0;
    }
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