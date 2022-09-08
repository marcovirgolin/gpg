#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <random>
#include <chrono>
#include "myeig.hpp"

using namespace std;
using namespace myeig;

typedef std::chrono::steady_clock Clock;

template<typename T>
void print(vector<T> & v) {
  for (auto & el : v) {
    cout << el << " ";
  }
  cout << endl;
}

template<class... Args>
void print(Args... args)
{
  (cout << ... << args) << "\n";
}

float corr(Vec x, Vec y) {
  float mean_x = x.mean();
  float mean_y = y.mean();

  Vec res_x = x - mean_x;
  Vec res_y = y - mean_y;

  float numerator = (res_x*res_y).sum();
  float denominator = sqrt(res_x.square().sum()) * sqrt(res_y.square().sum());

  if (denominator == 0)
    return 0;

  float result = denominator != 0 ? numerator / denominator : 0;

  return result;
}

float roundd(float x, int num_dec) {
  return round(x * num_dec) / (float) num_dec;
}

float randu() {
  return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}


float randn() {
  // Marsiglia algorithm
  float x, y, rsq;
  do {
    x = 2.0 * randu() - 1.0;
    y = 2.0 * randu() - 1.0;
    rsq = x * x + y * y;
  } while( rsq >= 1. || rsq == 0. );
  float f = sqrt( -2.0 * log(rsq) / rsq );
  return x * f;
}

auto tick() {
  return Clock::now();
}

float tock(chrono::time_point<Clock> tick) {
  auto now = Clock::now();
  auto duration = now - tick;
  return chrono::duration_cast<chrono::milliseconds>(duration).count() / 1000.0;
}

template<typename T>
vector<int> argsort(const vector<T> &array) {
  // credit: https://gist.github.com/HViktorTsoi/58eabb4f7c5a303ced400bcfa816f6f5
  vector<int> indices(array.size());
  iota(indices.begin(), indices.end(), 0);
  sort(indices.begin(), indices.end(),
      [&array](int left, int right) -> bool {
        // sort indices according to corresponding array element
        return array[left] < array[right];
      });
  return indices;
}

vector<int> rand_perm(int num_elements) {
  vector<float> rand_vec;
  rand_vec.reserve(num_elements);
  for(int i = 0; i < num_elements; i++) {
    rand_vec.push_back(randu());
  }
  return argsort(rand_vec);
}

void replace(Vec & x, float what, float with) {
  for(int i = 0; i < x.size(); i++)
    if (x[i] == what)
      x[i] = with;
}

#endif