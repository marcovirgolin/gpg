#include <iostream>
#include "nicknames.hpp"
#include "util.hpp"
#include "evolution.hpp"
#include "node.hpp"
#include "tests.hpp"

#include <Eigen/Dense>


using namespace nnames;

int main(int argc, char** argv){
  print("ciao",2);

  auto evo = Evolution();
  evo.run();

  auto t = Test();
  t.run_all();

  mat a(2,2);
  vec v = vec::Zero(10);
  print(v.transpose());

}