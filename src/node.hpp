#ifndef NODE_H
#define NODE_H

#include <vector>

using namespace std;

struct Node {

  Node * parent = NULL;
  vector<Node*> children;
  float fitness;

  int get_depth() {
    int depth = 0;
    auto * curr = this;
    while(curr->parent) {
      depth++;
      curr = curr->parent;
    }
    return depth;
  }

  vector<Node*> get_subtree() {
    vector<Node*> subtree;
    subtree.reserve(64);
    _get_subtree_recursive(subtree);
    return subtree;
    
  }

  void _get_subtree_recursive(vector<Node*> &subtree) {
    subtree.push_back(this);
    for(Node * child : children) {
      child->_get_subtree_recursive(subtree);
    }
  }

};


#endif