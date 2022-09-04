#include <iostream>

#include "globals.hpp"
#include "myeig.hpp"
#include "util.hpp"
#include "evolution.hpp"
#include "node.hpp"
#include "tests.hpp"

using namespace myeig;

int main(int argc, char** argv){

  g::set_options();

  //auto evo = Evolution();
  //evo.run();

  auto t = Test();
  t.run_all();

}