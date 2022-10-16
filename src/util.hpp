#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <iomanip>
#include <random>
#include <fstream>
#include <chrono>
#include <iterator>
#include "myeig.hpp"

using namespace std;
using namespace myeig;

typedef std::chrono::steady_clock Clock;

template<typename T>
void print(vector<T> & v) {
  for (auto & el : v) {
    cout << el << " ";
  }
  cout << endl;
}

template<class... Args>
void print(Args... args)
{
  (cout << ... << args) << "\n";
}

float roundd(float x, int num_dec) {
  return round(x * pow(10.0,num_dec)) / (float) pow(10.0,num_dec);
}

Vec roundd(Vec & x, int num_dec) {
  return (x * pow(10.0,num_dec)).round() / (float) pow(10.0,num_dec);
}

float median(Vec & x) {
  int n = x.size();
  if (n == 0) {
    throw runtime_error("Attempted to get median of empty array");
  }
  if (n == 1) {
    return x[0];
  }
  if (n % 2 == 0) {
    return 0.5*(x[n/2]+x[n/2+1]);
  } 
  return x[n/2];
}

Veci sort_order(Vec & x) {
  Veci indices(x.size());
  for(int i = 0; i < x.size(); i++)
    indices[i] = i;

  sort(indices.begin(), indices.end(), [&](int i, int j) {return x[i] < x[j];});
  
  return indices;
}

Veci ranking(Vec & x) {
  Veci o = sort_order(x);
  Veci r(o.size());
  for (int i = 0; i < o.size(); i++) {
    r[o[i]] = i;
  }
  return r;
}

float corr(Vec & x, Vec & y) {
  float mean_x = x.mean();
  float mean_y = y.mean();

  Vec res_x = x - mean_x;
  Vec res_y = y - mean_y;

  float numerator = (res_x*res_y).sum();
  float denominator = sqrt(res_x.square().sum()) * sqrt(res_y.square().sum());

  if (denominator == 0 || isnan(denominator) || isinf(abs(denominator)))
    return 0;

  float result = denominator != 0 ? numerator / denominator : 0;

  return result;
}

float spearcorr(Vec & x, Vec & y) {
  Veci r_x = ranking(x);
  Veci r_y = ranking(y);
  Vec r_x_f = r_x.cast<float>();
  Vec r_y_f = r_y.cast<float>();
  return corr(r_x_f, r_y_f);
}

int argmax(Vec & x) {
  int idx = -1;
  float best = NINF;
  for(int i = 0; i < x.size(); i++) {
    if (x[i] > best) {
      best = x[i];
      idx = i;
    }
  }
  return idx;
}

int argmin(Vec & x) {
  int idx = -1;
  float best = INF;
  for(int i = 0; i < x.size(); i++) {
    if (x[i] < best) {
      best = x[i];
      idx = i;
    }
  }
  return idx;
}

pair<float, float> linear_scaling_coeffs(Vec & y, Vec & p) {

  float interc, slope;
  float y_mean = y.mean();
  float p_mean = p.mean();

  Vec y_res = y - y_mean;
  Vec p_res = p - p_mean;

  float denominator = p_res.square().sum();

  if (denominator == 0 || isnan(denominator) || isinf(abs(denominator))) {
    slope = 0;
    interc = y_mean;
  } else {
    slope = (y_res*p_res).sum() / denominator;
    interc = y_mean - slope * p_mean;
  }

  return make_pair(interc, slope);    
}

double randu() {
  return (double) rand() / (double) (RAND_MAX + 1.0);
}

int randi(int min_inclusive, int max_exclusive) {
  int range = max_exclusive - min_inclusive;
  return min_inclusive + randu()*range;
}

int randi(int max_exclusive) {
  return randi(0, max_exclusive);
}

float randn() {
  // Marsiglia algorithm
  float x, y, rsq;
  do {
    x = 2.0 * randu() - 1.0;
    y = 2.0 * randu() - 1.0;
    rsq = x * x + y * y;
  } while( rsq >= 1. || rsq == 0. );
  float f = sqrt( -2.0 * log(rsq) / rsq );
  return x * f;
}

auto tick() {
  return Clock::now();
}

float tock(chrono::time_point<Clock> tick) {
  auto now = Clock::now();
  auto duration = now - tick;
  return chrono::duration_cast<chrono::milliseconds>(duration).count() / 1000.0;
}

template<typename T>
vector<int> argsort(const vector<T> &array) {
  // credit: https://gist.github.com/HViktorTsoi/58eabb4f7c5a303ced400bcfa816f6f5
  vector<int> indices(array.size());
  iota(indices.begin(), indices.end(), 0);
  sort(indices.begin(), indices.end(),
      [&array](int left, int right) -> bool {
        // sort indices according to corresponding array element
        return array[left] < array[right];
      });
  return indices;
}

vector<int> rand_perm(int num_elements) {
  vector<float> rand_vec;
  rand_vec.reserve(num_elements);
  for(int i = 0; i < num_elements; i++) {
    rand_vec.push_back(randu());
  }
  return argsort(rand_vec);
}

Vec replace(Vec & x, float what, float with, string condition="=") {
  // returns a copy 
  Vec r(x);
  if (condition == "=") {
    for(int i = 0; i < x.size(); i++)
      if (r[i] == what)
        r[i] = with;
  } else if (condition == "<=") {
    for(int i = 0; i < x.size(); i++)
      if (r[i] <= what)
        r[i] = with;
  } else if (condition == ">=") {
    for(int i = 0; i < x.size(); i++)
      if (r[i] >= what)
        r[i] = with;
  } else if (condition == "<") {
    for(int i = 0; i < x.size(); i++)
      if (r[i] < what)
        r[i] = with;
  } else if (condition == ">") {
    for(int i = 0; i < x.size(); i++)
      if (r[i] > what)
        r[i] = with;
  } else {
    throw runtime_error("Unrecognized condition: " + condition);
  }
  return r;
}
  
vector<int> create_range(int n) {
  vector<int> x; x.reserve(n);
  for(int i = 0; i < n; i++)
    x.push_back(i);
  return x;
}

Mat remove_column(Mat & X, int col_idx) {
  // returns a new matrix without that column (the original is not modified)
  Mat R(X.rows(), X.cols()-1);
  for(int i = 0; i < X.rows(); i++) {
    for(int j = 0; j < X.cols(); j++) {
      if (j == col_idx)
        continue;
      R(i,j) = X(i,j);
    }
  }
  return R;
}

Mat load_csv(const std::string & path, char separator=',') {
  // parser from https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix
  ifstream indata;
  indata.open(path);
  string line;
  vector<float> values;
  uint rows = 0;
  while (getline(indata, line)) {
    stringstream line_stream(line);
    string cell;
    while (getline(line_stream, cell, separator)) {
      values.push_back(stof(cell));
    }
    ++rows;
  }
  int cols = values.size()/rows;

  Mat R(rows, cols);
  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      R(i,j) = values[j + i*(cols)];
    }
  }
  return R;
}

bool exists(string & file_path)
{
    std::ifstream file(file_path.c_str());
    return file.good();
}

string replace(string& source, const string& what, string with)
{
  // Solution by Ingmar: https://stackoverflow.com/a/29752943
  string new_string;
  new_string.reserve(source.length());  // avoids a few memory allocations

  string::size_type lastPos = 0;
  string::size_type findPos;

  while(string::npos != (findPos = source.find(what, lastPos)))
  {
    new_string.append(source, lastPos, findPos - lastPos);
    new_string += with;
    lastPos = findPos + what.length();
  }

  // Care for the rest after last occurrence
  new_string += source.substr(lastPos);

  return new_string;
}

vector<string> split_string(string & original, string delimiter=",") {
  vector<string> result;
  string changed_string = original;
  if (delimiter != " ") {
    changed_string = replace(original, delimiter, " ");
  }
  istringstream iss(changed_string);
  result = {istream_iterator<string>
      {iss}, istream_iterator<string>
      {}};
  return result;
}


#endif