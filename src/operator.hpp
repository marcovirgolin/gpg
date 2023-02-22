#ifndef OPERATOR_H
#define OPERATOR_H

#include "myeig.hpp"
#include "util.hpp"
#include "rng.hpp"
#include <map>

using namespace std;
using namespace myeig;

map<string, pair<function<Vec(const vector<Vec> &)>, int>> all_operators = {
  {"+", {[](const vector<Vec> &args){return args[0] + args[1];}, 2}},
  {"-", {[](const vector<Vec> &args){return args[0] - args[1];}, 2}},
  {"*", {[](const vector<Vec> &args){return args[0].cwiseProduct(args[1]);}, 2}},
  {"/", {[](const vector<Vec> &args){return args[0].cwiseQuotient(args[1]);}, 2}},
  {"neg", {[](const vector<Vec> &args){return -args[0];}, 1}},
  {"**2", {[](const vector<Vec> &args){return args[0].square();}, 1}},
  {"**3", {[](const vector<Vec> &args){return args[0].cube();}, 1}},
  {"pow", {[](const vector<Vec> &args){return args[0].pow(args[1]);}, 2}},
  {"sqrt", {[](const vector<Vec> &args){return args[0].sqrt();}, 1}},
  {"log", {[](const vector<Vec> &args){return args[0].log();}, 1}},
  {"exp", {[](const vector<Vec> &args){return args[0].exp();}, 1}},
  {"abs", {[](const vector<Vec> &args){return args[0].abs();}, 1}},
  {"sin", {[](const vector<Vec> &args){return args[0].sin();}, 1}},
  {"cos", {[](const vector<Vec> &args){return args[0].cos();}, 1}},
  {"tan", {[](const vector<Vec> &args){return args[0].tan();}, 1}},
  {"asin", {[](const vector<Vec> &args){return args[0].asin();}, 1}},
  {"acos", {[](const vector<Vec> &args){return args[0].acos();}, 1}},
  {"atan", {[](const vector<Vec> &args){return args[0].atan();}, 1}},
};

string op_to_infix_repr(string op_sym, const vector<string> & args) {
  if (op_sym == "+") {
    return "("+args[0]+" + "+args[1]+")";
  } else if (op_sym == "-") {
    return "("+args[0]+" - "+args[1]+")";
  } else if (op_sym == "*") {
    return "("+args[0]+" * "+args[1]+")";
  } else if (op_sym == "/") {
    return "("+args[0]+" / "+args[1]+")";
  } else if (op_sym == "neg") {
    return "-"+args[0];
  } else if (op_sym == "**2") {
    return "("+args[0]+"**2)";
  } else if (op_sym == "**3") {
    return "("+args[0]+"**3)";
  } else if (op_sym == "pow") {
    return "("+args[0]+"**"+args[1]+")";
  } else if (op_sym == "sqrt") {
    return "sqrt("+args[0]+")";
  } else if (op_sym == "log") {
    return "log("+args[0]+")";
  } else if (op_sym == "exp") {
    return "exp("+args[0]+")";
  } else if (op_sym == "abs") {
    return "abs("+args[0]+")";
  } else if (op_sym == "sin") {
    return "sin("+args[0]+")";
  } else if (op_sym == "cos") {
    return "cos("+args[0]+")";
  } else if (op_sym == "tan") {
    return "tan("+args[0]+")";
  } else if (op_sym == "asin") {
    return "asin("+args[0]+")";
  } else if (op_sym == "acos") {
    return "acos("+args[0]+")";
  } else if (op_sym == "atan") {
    return "atan("+args[0]+")";
  } else {
    throw runtime_error("unknown operator: "+op_sym);
  }
}

#endif