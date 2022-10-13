#ifndef MYEIG_H
#define MYEIG_H

#define INF std::numeric_limits<float>::infinity()
#define NINF -1*std::numeric_limits<float>::infinity()

#include <Eigen/Dense>
#include <limits>

namespace myeig {
  typedef Eigen::ArrayXXf Mat;
  typedef Eigen::ArrayXf Vec;
  typedef Eigen::ArrayXi Veci;
}

#endif