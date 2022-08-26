#include <iostream>
#include "myeig.hpp"
#include "util.hpp"
#include "evolution.hpp"
#include "node.hpp"
#include "tests.hpp"

using namespace myeig;

int main(int argc, char** argv){
  print("ciao",2);

  auto evo = Evolution();
  evo.run();

  auto t = Test();
  t.run_all();

  Mat a(2,2);
  Vec v = Vec::Zero(10);
  print(v.transpose());

}