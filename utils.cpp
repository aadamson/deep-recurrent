#ifndef UTILS_CPP
#define UTILS_CPP
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <iterator>
#include <cassert>
#include <sys/stat.h>
#include "Eigen/Dense"

#define uint unsigned int

using namespace Eigen;
using namespace std;

namespace Eigen {
istream& operator>>(istream& s, MatrixXd& m) {
  for (uint i=0; i<m.rows(); i++)
    for (uint j=0; j<m.cols(); j++)
      s >> m(i,j);
  return s;
}
istream& operator>>(istream& s, VectorXd& m) {
  for (uint i=0; i<m.size(); i++)
    s >> m(i);
  return s;
}
}

MatrixXd confusion_matrix(const vector<string> & y_hat, const vector<string> & y, uint num_classes) {
  MatrixXd conf_mat = MatrixXd::Zero(num_classes, num_classes);
  for (int i = 0; i < y.size(); i++) {
    conf_mat(stoi(y[i]), stoi(y_hat[i])) += 1;
  }
  return conf_mat;
}

MatrixXd softmax(const MatrixXd &x) {
  RowVectorXd m = x.colwise().maxCoeff();
  MatrixXd t = (x.rowwise() - m).array().exp();
  return t.array().rowwise() / t.colwise().sum().array();
}

MatrixXd smaxentp(const MatrixXd &y, const MatrixXd &r) {
  return y-r;
}

MatrixXd relu(const MatrixXd &x) {
  return x.array().max(0);
}

MatrixXd relup(const MatrixXd &x) { 
  return (x.array() > 0).cast<double>();
}

MatrixXd sigmoid(const MatrixXd &x) {
  return 1.0 / (1.0 + (-x).array().exp());
}

MatrixXd sigmoidp(const MatrixXd &x) { 
  return x.array() * (1.0 - x.array());
}

MatrixXd fast_sigmoid(const MatrixXd &x) {
  return x.array() / (1.0 + x.array().abs());
}

MatrixXd fast_sigmoidp(const MatrixXd &x) { 
  return 1.0 / (1.0 + x.array().abs()).pow(2);
}

MatrixXd _tanh(const MatrixXd &x) {
  return (2.0 / (1.0 + (-2.0 * x).array().exp())) - 1.0;
}

MatrixXd _tanhp(const MatrixXd &x) { 
  return 1.0 - x.array().pow(2);
}

MatrixXd clip(const MatrixXd &x) {
  return x.array().min(1e10).max(-1e10);
}

double str2double(const string& s) {
  istringstream i(s);
  double x;
  if (!(i >> x))
    return 0;
  return x;
}

class LookupTable {
  public:
    void load(string fname, uint n, uint d, bool noUnknown=false);
    //ColXpr operator[](string word);
    VectorXd operator[](string word);
    void gradAdd(string word, VectorXd v);
    void update();

  private:
    double lr;
    map<string, uint> table; // word -> index
    MatrixXd data;            // index -> vector representation
    MatrixXd gdata;           // gradients
    MatrixXd adata;           // adagrad past
    set<uint> modifiedCols;
};

void LookupTable::load(string fname, uint n, uint d, bool noUnknown) {
  ifstream in(fname.c_str());
  assert(in.is_open());
  string line;
  if (noUnknown) n++;
  data = MatrixXd(d,n);
  gdata = adata = MatrixXd::Zero(d,n);
  adata.fill(1e-6);
  uint j=0;
  while(std::getline(in, line)) {
    std::istringstream ss(line);
    std::istream_iterator<std::string> begin(ss), end;

    //putting all the tokens in the vector
    std::vector<std::string> tokens(begin, end); 
    for (uint i=0; i<d; i++)
      data(i,j) = str2double(tokens[i+1]);
    table[tokens[0]] = j;
    j++;
    if (j == n)
      break;
  }
  
  if (noUnknown) {
    VectorXd v = data.rowwise().mean();
    data.col(n-1) = v;
    table["*UNKNOWN*"] = n-1;
  }

  double min = data.minCoeff();
  //cout << "Lookup table min: " << min << endl;
  //data = data.array()-min;
}

VectorXd LookupTable::operator[](string word) {
  map<string,uint>::iterator it;

  // this might not be the best place for this,
  // if i'm calling this frequently
  if (word == "-LRB-")
    word = "(";
  else if (word == "-RRB-")
    word = ")";
  else if (word == "-LSB-")
    word = "(";
  else if (word == "-RSB-")
    word = ")";
  else if (word == "-LCB-")
    word = "(";
  else if (word == "-RCB-")
    word = ")";

  it = table.find(word);
  if (it != table.end()) // exists
    return data.col(table[word]);
  else
    return data.col(table["*UNKNOWN*"]);
}

void LookupTable::gradAdd(string word, VectorXd v) {
  map<string,uint>::iterator it;
  if (word == "-LRB-")
    word = "(";
  else if (word == "-RRB-")
    word = ")";
  else if (word == "-LSB-")
    word = "(";
  else if (word == "-RSB-")
    word = ")";
  else if (word == "-LCB-")
    word = "(";
  else if (word == "-RCB-")
    word = ")";
 
  it = table.find(word);
  if (it != table.end()) {// exists
    gdata.col(table[word]) += v;
    modifiedCols.insert(table[word]);
  } else {
    gdata.col(table["*UNKNOWN*"]) += v;
    modifiedCols.insert(table[word]);
  }
}

void LookupTable::update() {
  lr = 0.001;
  for (auto i : modifiedCols) {
    adata.col(i) = (adata.col(i).cwiseProduct(adata.col(i)) + 
                  gdata.col(i).cwiseProduct(gdata.col(i))).cwiseSqrt();
    data.col(i) -= lr*gdata.col(i).cwiseQuotient(adata.col(i));
    gdata.col(i).setZero();
  }
  modifiedCols.clear();
}

// index of max in a vector
uint argmax(const VectorXd& x) {
  double max = x(0);
  uint maxi = 0;
  for (uint i=1; i<x.size(); i++) {
    if (x(i) > max) {
      max = x(i);
      maxi = i;
    }
  }
  return maxi;
}

// this is used for randomly initializing an Eigen matrix
double urand(double dummy) {
  double min = -0.01, max = 0.01;
  return (double(rand())/RAND_MAX)*(max-min) + min;
}

// KFY shuffle (uniformly randomly) of a vector
template <class T>
void shuffle(vector<T>& v) {
  for (uint i=v.size()-1; i>0; i--) {
    uint j = (rand() % i);
    T tmp = v[i];
    v[i] = v[j];
    v[j] = tmp;
  }
}

template <class T, class T2>
void shuffle(vector<T>& v, vector<T2>& w) {
  assert(w.size() == v.size());
  for (uint i=v.size()-1; i>0; i--) {
    uint j = (rand() % i);
    T tmp = v[i];
    v[i] = v[j];
    v[j] = tmp;
    T2 tmp2 = w[i];
    w[i] = w[j];
    w[j] = tmp2;
  }
}

bool isWhitespace(std::string str) {
  for(uint i=0; i<str.size(); i++) {
    if (!isspace(str[i]))
      return false;
  }
  return true;
}

vector<string> split(const string &s, char delim) {
  stringstream ss(s);
  string item;
  vector<string> elems;
  while (getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

string filename(string filepath) {
  unsigned int a = filepath.rfind('.');
  unsigned int b = filepath.rfind('/');
  string fname = filepath.substr(b+1,a-b-1);
  return fname;
}

bool conditional_mkdir(const string &path) {
  struct stat s;
  if (stat(path.c_str(), &s) == 0 && S_ISDIR(s.st_mode)) return true;
  string mkcmd =  "mkdir -p " + path;
  system(mkcmd.c_str());
  
  return true; 
}
#endif
