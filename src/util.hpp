#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <random>
#include <chrono>
#include "myeig.hpp"

using namespace std;
using namespace myeig;

typedef std::chrono::steady_clock Clock;

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

float randu() {
  return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

auto tick() {
  return Clock::now();
}

float tock(chrono::time_point<Clock> tick) {
  auto now = Clock::now();
  auto duration = now - tick;
  return chrono::duration_cast<chrono::milliseconds>(duration).count() / 1000.0;
}

#endif