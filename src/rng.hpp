// Code from Johannes Koch
#ifndef GOMEA_RNG
#define GOMEA_RNG

#pragma once

#include "xoshiro.hpp"
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
 * `ThreadRng::get()` directly or sugar it like this:
 *
 *   auto& thread_rng(){ return ThreadRng::get(); }
 */

/// A rust ThreadRng like interface for a seeded thread local rng
class ThreadRng
{
private:
  /// Not going to be an issue for my use case but here for correctness
  inline static std::mutex seed_mutex;
  /// The global seed
  inline static uint64_t seed = 0;
  ThreadRng(){};

public:
  ThreadRng(const ThreadRng&) = delete;
  void operator=(const ThreadRng&) = delete;

  /// Returns the current rng seed or sets a random seed if the seed has not
  /// been set up to this point
  static uint64_t get_seed()
  {
    std::scoped_lock<std::mutex> lock(ThreadRng::seed_mutex);
    if (ThreadRng::seed == 0) {
      std::random_device rd;
      ThreadRng::seed = rd();
    }
    return seed;
  };

  /// Sets the rng seed, only effective if called before the first
  /// ThreadRng::get. If the seed was never set, std::random_device is used for
  /// rng seeding.
  /// @param seed
  static void set_seed(uint64_t seed)
  {
    std::scoped_lock<std::mutex> lock(ThreadRng::seed_mutex);
    ThreadRng::seed = seed;
  };

  /// Returns the thread local rng
  static Xoshiro::Xoshiro256PP& get()
  {
    thread_local static Xoshiro::Xoshiro256PP instance(ThreadRng::get_seed());
    return instance;
  };
};

#endif /* GOMEA_RNG */