#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include <vector>
#include <stack>
#include "operator.hpp"
#include "util.hpp"

using namespace std;

struct Individual {
  float fitness;
  vector<string> genome;

  Individual() {
    fitness = INF;
  }

  Individual * clone() {
    Individual * indiv = new Individual();
    indiv->fitness = fitness;
    indiv->genome = genome;
    return indiv;
  }

  int get_num_components(bool excl_introns=false) {
    if (!excl_introns)
      return genome.size();
    int num_components = 0;
    _recursive_get_num_components(genome, num_components);
    return num_components;
  }

  int _recursive_get_num_components(const vector<string> &stack, int & num_components, int idx=0) {
    num_components++;
    if (all_operators.count(stack[idx])) {
      auto op = all_operators[stack[idx]];
      for (int i = 0; i < op.second; i++) {
        int jdx = _recursive_get_num_components(stack, num_components, idx + 1);
        idx = jdx;
      }
    }
    return idx;
  }

  string to_prefix_notation() {
    string s = "";
    _recursive_to_prefix_notation(genome, s);
    // remove trailing comma
    s = s.substr(0, s.size()-1);
    return s;
  }

  int _recursive_to_prefix_notation(const vector<string> &stack, string & prefix_notation, int idx=0) {
    prefix_notation += stack[idx] + ",";
    if (all_operators.count(stack[idx])) {
      auto op = all_operators[stack[idx]];
      for (int i = 0; i < op.second; i++) {
        int jdx = _recursive_to_prefix_notation(stack, prefix_notation, idx + 1);
        idx = jdx;
      }
    }
    return idx;
  }

  string to_infix_notation() {
    string pre = to_prefix_notation();
    vector<string> pre_v = split_string(pre);

    // CODE ADAPTED FROM https://www.geeksforgeeks.org/prefix-infix-conversion/
    stack<string> s;
    int length = pre_v.size();
    // reading from right to left
    for(int i = length-1; i >= 0; i--) {
      string sym = pre_v[i];
      // check if symbol is operator
      if (all_operators.count(sym)) {
        // pop operands from stack based on arity
        int arity = all_operators[sym].second;
        vector<string> children; children.reserve(4);
        for(int j = 0; j < arity; j++) {
          string c = s.top(); s.pop();
          children.push_back(c);
        }
        string temp = op_to_infix_repr(sym, children);
        s.push(temp);
      }
      // if symbol is an operand
      else {
        // push the operand to the stack
        s.push(pre_v[i]);
      }
    }
    // Stack now contains the Infix expression
    return s.top();
  }

  Vec eval(const Mat &D) {
    auto [res, _] = _recursive_eval(genome, D);
    return res;
  }

  pair<Vec, int> _recursive_eval(vector<string> &stack, const Mat &D, int idx=0) {
    if (all_operators.count(stack[idx])) {
      auto op = all_operators[stack[idx]];
      vector<Vec> args; args.reserve(2);
      for (int i = 0; i < op.second; i++) {
        auto [res, jdx] = _recursive_eval(stack, D, idx + 1);
        args.push_back(res);
        idx = jdx;
      }
      return {op.first(args), idx};
    } else if (stack[idx][0] == 'x') {
      int feat_idx = stoi(stack[idx].substr(2));
      return {D.col(feat_idx), idx};
    } else {
      /*float c;
      // initialize ERC
      if (stack[idx] == "nan" || stack[idx] == "erc") {
        c = roundd(Rng::randu()*10 - 5, NUM_PRECISION);
        genome[idx] = to_string(c); // note that genome == stack
      }
      else {
        c = stof(stack[idx]);
      }*/
      float c = stof(stack[idx]);
      return {Vec::Constant(D.rows(), c), idx};
    }
  }


};

#endif