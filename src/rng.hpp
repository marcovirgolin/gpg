// Code from Johannes Koch
#ifndef RNG_H
#define RNG_H

#pragma once

#include "xoshiro.hpp"
#include "myeig.hpp"
#include "util.hpp"
#include <mutex>
#include <random>

/**
 * Rationale for doing it this way:
 * - I want to have a globally accessible seedable rng with nice
 *   properties (fast, long period, actually statistically random)
 * - Xoshiro (https://prng.di.unimi.it/) fits the bill according to
 *   my understanding
 * - To allow thread safety and still have determinism irrespective of
 *   execution order, the same seed is used on all threads
 * - To be able to use the same seed everywhere, it must be a global
 * - To have thread safety and performance, each thread has it's own
 *   thread_local rng
 *
 * !!! Important !!!
 * C++ assignment/copy construction is a pain, so always call
 * `Rng::get()` directly or sugar it like this:
 *
 *   auto& thread_rng(){ return Rng::get(); }
 */

using namespace std;
using namespace myeig;

/// A rust Rng like interface for a seeded thread local rng
class Rng
{
private:
  /// Not going to be an issue for my use case but here for correctness
  inline static std::mutex seed_mutex;
  /// The global seed
  inline static uint64_t seed = 0;
  Rng(){};

  inline static uniform_real_distribution<double> unif_distr = uniform_real_distribution<double>(0.0, 1.0);
  inline static normal_distribution<double> norm_distr = normal_distribution<double>(0.0, 1.0);

public:
  // Prevent auto generation of copy constructor and assignment operator (because we want a singleton)
  Rng(const Rng&) = delete;
  void operator=(const Rng&) = delete;

  // Returns the current rng seed or sets a random seed if the seed has not
  // been set up to this point
  static uint64_t get_seed()
  {
    std::scoped_lock<std::mutex> lock(Rng::seed_mutex);
    if (Rng::seed == 0) {
      std::random_device rd;
      Rng::seed = rd();
    }
    return seed;
  };

  // Sets the rng seed, only effective if called before the first
  // Rng::get. If the seed was never set, std::random_device is used for
  // rng seeding. 
  // @param seed
  static void set_seed(uint64_t seed)
  {
    std::scoped_lock<std::mutex> lock(Rng::seed_mutex);
    Rng::seed = seed;
  };

  // Returns the thread local rng
  static Xoshiro::Xoshiro256PP& get()
  {
    thread_local static Xoshiro::Xoshiro256PP instance(Rng::get_seed());
    return instance;
  };

  // Returns a random number in the range [0,1)
  static double randu()
  {
    return unif_distr(Rng::get());
  };

  static int randi(int min_inclusive, int max_exclusive) {
    int range = max_exclusive - min_inclusive;
    return min_inclusive + randu()*range;
  }

  static int randi(int max_exclusive) {
    return randi(0, max_exclusive);
  }


  // Returns a random number from the normal distribution
  static double randn(double mean=0.0, double stdev=1.0)
  {
    return norm_distr(Rng::get())*stdev + mean;
  };

  // Returns a matrix unitialized with the random uniform distribution (values between 0 incl. and 1 excl.)
  static Mat randu_mat(int n_rows, int n_cols)
  {
    Mat m = Mat(n_rows, n_cols);
    for(int i = 0; i < n_rows; i++) {
      for(int j = 0; j < n_cols; j++) {
        m(i, j) = randu();
      }
    }
    return m;
  }

  // Returns a matrix unitialized with the random normal distribution
  static Mat randn_mat(int n_rows, int n_cols, double mean=0.0, double stdev=1.0)
  {
    Mat m = Mat(n_rows, n_cols);
    for(int i = 0; i < n_rows; i++) {
      for(int j = 0; j < n_cols; j++) {
        m(i, j) = randn(mean, stdev);
      }
    }
    return m;
  }

  // Returns a random permutation of the numbers 0 to num_elements-1
  static vector<int> rand_perm(int num_elements) {
    vector<double> rand_vec;
    rand_vec.reserve(num_elements);
    for(int i = 0; i < num_elements; i++) {
      rand_vec.push_back(Rng::randu());
    }
    return argsort(rand_vec);
  }

};

#endif /* RNG_H */
