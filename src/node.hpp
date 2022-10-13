#ifndef NODE_H
#define NODE_H

#include <vector>
#include "operator.hpp"
#include "util.hpp"

using namespace std;

struct Node {

  Node * parent = NULL;
  vector<Node*> children;
  Op * op = NULL;
  float fitness;

  //Node() {};

  Node(Op * op) {
    this->op = op;
  }

  virtual ~Node() noexcept(false) {
    if (op)
      delete op;
  }

  void clear() {
    auto nodes = subtree();
    for (int i = 1; i < nodes.size(); i++) {
      delete nodes[i];
    }
    delete this;
  }

  Node * clone() {
    Node * new_node = new Node(this->op->clone());
    new_node->fitness = this->fitness;
    for(Node * c : this->children) {
      Node * new_c = c->clone();
      new_node->append(new_c);
    }
    return new_node;
  }

  void append(Node * c) {
    this->children.push_back(c);
    c->parent = this;
  }

  vector<Node*>::iterator detach(Node * c){
    auto it = children.begin();
    for(auto it = children.begin(); it < children.end(); it++){
      if ((*it)==c){
        break;
      }
    }
    assert(it != children.end());
    children.erase(it);
    c->parent = NULL;
    return it;
  }

  Node * detach(int idx) {
    assert(idx < children.size());
    auto it = children.begin() + idx;
    Node * c = children[idx];
    children.erase(it);
    c->parent = NULL;
    return c;
  }

  void insert(Node * c, vector<Node*>::iterator it) {
    children.insert(it, c);
    c->parent = this;
  }

  int depth() {
    int depth = 0;
    auto * curr = this;
    while(curr->parent) {
      depth++;
      curr = curr->parent;
    }
    return depth;
  }

  int height() {
    int max_child_depth = 0;
    _height_recursive(max_child_depth);
    int h = max_child_depth - depth();
    assert(h >= 0);
    return h;
  }

  int get_num_nodes(bool excl_introns=false) {
    auto nodes = this->subtree();
    int n = nodes.size();
    if (!excl_introns) 
      return n;

    int num_introns = 0;
    for(Node * n : nodes) {
      if (n->is_intron()) {
        num_introns++;
      }
    }
    return n - num_introns;
  }

  void _height_recursive(int & max_child_depth) {
    if (op->arity() == 0) {
      int d = this->depth();
      if (d > max_child_depth)
        max_child_depth = d;
    }

    for (int i = 0; i < op->arity(); i++)
      children[i]->_height_recursive(max_child_depth);
  }

  vector<Node*> subtree() {
    vector<Node*> subtree;
    subtree.reserve(64);
    _subtree_recursive(subtree);
    return subtree;
  }

  void _subtree_recursive(vector<Node*> &subtree) {
    subtree.push_back(this);
    for(Node * child : children) {
      child->_subtree_recursive(subtree);
    }
  }

  int position_among_siblings() {
    if (!parent)
      return 0;
    int i = 0;
    for (Node * s : parent->children) {
      if (s == this)
        return i;
      i++;
    }
    throw runtime_error("Unreachable code");
  }

  bool is_intron() {
    Node * p = parent;
    if(!p)
      return false;
    Node * n = this;
    while (p) {
      if (n->position_among_siblings() >= p->op->arity())
        return true;
      n = p;
      p = n->parent;
    }
    return false;
  }

  Vec get_output(Mat & X) {
    int a = op->arity();
    if (a == 0)
      return op->apply(X);

    Mat C(X.rows(), a);
    for(int i = 0; i < a; i++)
      C.col(i) = children[i]->get_output(X);

    return (op->apply(C) * pow(10.0, NUM_PRECISION)) / (float) pow(10.0,NUM_PRECISION);
  }

  void print_subtree() {
    vector<Node*> nodes = this->subtree();
    string str = "[";
    for(Node * n : nodes) {
      str += n->op->sym() + ", ";
    }
    str.erase(str.end()-2, str.end());
    str += "]";
    print(str);
  }


  void _human_repr_recursive(string & expr) {
    int arity = op->arity();
    vector<string> args; args.reserve(arity);
    for(int i = 0; i < arity; i++) {
      children[i]->_human_repr_recursive(expr);
      args.push_back(expr);
    }
    expr = op->human_repr(args);
  }

  string human_repr() {
    string result = "";
    _human_repr_recursive(result);
    return result;
  }

};


#endif