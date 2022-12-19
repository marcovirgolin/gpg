#ifndef FEATURESELECTION_H
#define FEATURESELECTION_H

#include "myeig.hpp"

using namespace std;
using namespace myeig;

Veci feature_selection(Mat & X, Vec & y, int to_retain=10) {

  int num_features = X.cols();
  Veci result;

  if (to_retain >= num_features) {
    // simply return list of all feature indices
    result = Veci(num_features);
    for(int i = 0; i < num_features; i++) {
      result[i] = i;
    }
    return result;
  }

  // else, 
  int num_retained = 0;
  // initialize result (indices of features to retain) to -1
  result = Veci::Zero(to_retain) - 1;

  // pre-compute absolute spearman correlation to target
  Vec ascs(num_features);
  for(int i = 0; i < num_features; i++) {
    Vec feat = X.col(i);
    ascs[i] = abs(spearcorr(feat, y));
  }

  // pre-compute inter-feature abs pearson correlation
  Mat P(num_features,num_features);
  for(int i = 0; i < num_features; i++) {
    Vec feat_i = X.col(i);
    for(int j = i+1; j < num_features; j++) {
      Vec feat_j = X.col(j);
      P(i,j) = abs(corr(feat_i, feat_j));
      P(j,i) = P(i,j);
    }
    P(i,i) = 0.0; // do not consider corr w.r.t. itself
  }

  // initialize scores to abs spearson correlation
  Vec scores = ascs;

  // then scores will be updated as we include more features in retained
  int best_sc_idx = -1; // this keeps track of what is the last inserted idx
  while(num_retained < to_retain) {

    // update scores by subtracting max abs pears corr with last inserted idx
    if (best_sc_idx > 0)
      for(int i = 0; i < num_features; i++)
        scores[i] = scores[i] > NINF ? scores[i] - P(i,best_sc_idx) / ((float)num_features) : scores[i];

    // get next to include w.r.t. score
    best_sc_idx = argmax(scores);
    // lower score for this feature so that it is not selected again
    scores[best_sc_idx] = NINF;

    // add this feature
    result[num_retained] = best_sc_idx;
    num_retained++;
  }

  return result;
}


#endif