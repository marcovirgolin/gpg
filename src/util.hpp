#ifndef UTIL_H
#define UTIL_H

#include <iostream>

using namespace std;

template<class... Args>
void print(Args... args)
{
  (cout << ... << args) << "\n";
}

#endif