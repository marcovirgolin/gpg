#ifndef SELECTION_H
#define SELECTION_H

#include <vector>
#include "node.hpp"
#include "util.hpp"
#include "globals.hpp"
#include "rng.hpp"

using namespace std;

Node * tournament(vector<Node*> & candidates, int tournament_size) {
  auto rp = Rng::rand_perm(candidates.size());
  Node * winner = candidates[rp[0]];
  for(int i = 1; i < tournament_size; i++) {
    if (candidates[rp[i]]->fitness <= winner->fitness)
      winner = candidates[rp[i]];
  }
  return winner->clone();
}

vector<Node*> popwise_tournament(vector<Node*> & population, int selection_size, int tournament_size, bool stochastic=false) {
  int pop_size = population.size();
  vector<Node*> selected; selected.reserve(selection_size);
  
  if (stochastic) {
    while(selected.size() < selection_size) {
      selected.push_back(tournament(population, tournament_size));
    }
  }
  
  // else proceed with deterministic

  assert(  ((float)pop_size) / tournament_size == (float) pop_size / tournament_size );

  int n_selected_per_round = pop_size / tournament_size;
  int n_rounds = selection_size / n_selected_per_round;

  for(int i = 0; i < n_rounds; i++){
    // get a random permutation 
    auto perm = Rng::rand_perm(pop_size);

    // apply tournaments
    for(int j = 0; j < n_selected_per_round; j++) {
      // one tournament instance
      Node * winner = population[perm[j*tournament_size]];
      for(int k=j*tournament_size + 1; k < (j+1)*tournament_size; k++){
        if (population[perm[k]]->fitness < winner->fitness) {
          winner = population[perm[k]];
        }
      }
      selected.push_back(winner->clone());
    }
  }
  return selected;
}

#endif