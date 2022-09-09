#include <iostream>

#include "globals.hpp"
#include "myeig.hpp"
#include "util.hpp"
#include "evolution.hpp"
#include "node.hpp"
#include "tests.hpp"

using namespace myeig;

int main(int argc, char** argv){
  g::read_options(argc, argv);
  
  auto t = Test();
  t.run_all();

  auto start_time = tick();
  auto evo = Evolution();
  evo.run();
  print("Runtime: ",tock(start_time),"s");

  g::clear_globals();

}