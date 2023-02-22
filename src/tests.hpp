#ifndef TESTS_H
#define TESTS_H

#include "myeig.hpp"
#include "util.hpp"
#include "individual.hpp"
#include "operator.hpp"
#include "fitness.hpp"
#include "variation.hpp"

using namespace std;
using namespace myeig;

struct Test {

  void run_all() {
    math();
  }


  void math() {
    // correlation
    Vec x(5); 
    x << 1, 2, 3, 4, 5;
    Vec y(5);
    y << 2, 4, 6, 8, 10;

    assert( corr(x, y) == 1.0 );

    y = -1 * y;
    assert( corr(x, y) == -1.0 );

    y << 0, 0, 0, 0, 0;
    assert( corr(x, y) == 0.0 );

    // nan & infs
    assert(isnan(NAN));
    assert(INF > 99999999.9);
    assert(NINF < -9999999.9);

    // round
    assert(roundd(0.1, 0)==0);
    assert(roundd(0.5, 0)==1);
    assert(roundd(0.0004, 7)==(float)0.0004);
    assert(roundd(0.0004, 3)==0);
    assert(roundd(0.0027, 3)==(float)0.003);

    // sort_order
    Vec a(5);
    a << 10., .1, 3., 2., 11.;
    Veci order_of_a(5);
    Veci rank_of_a(5);
    order_of_a << 1, 3, 2, 0, 4;
    rank_of_a << 3, 0, 2, 1, 4;
    Veci o = sort_order(a);
    assert(order_of_a.isApprox(o));
    Veci r = ranking(a);
    assert(rank_of_a.isApprox(r));
  }

};


#endif