#ifndef DATA_UTILS_UTILS
#define DATA_UTILS_UTILS

#include <vector>
#include <map>
#include <iterator>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstdio>

using namespace std;
namespace DataUtils {
  int read_sentences(vector<vector<string > > &X,
                              vector<vector<string> > &T, 
                              string fname) {
    int num_labels = 1;
    map<string, int> label_token_to_num;
    label_token_to_num["0"] = 0;
    ifstream in(fname.c_str());
    string line;
    vector<string> x;
    vector<string> t; // individual sentences and labels
    while(std::getline(in, line)) {
      if (isWhitespace(line)) {
        if (x.size() != 0) {
          X.push_back(x);
          T.push_back(t);
          x.clear();
          t.clear();
        }
      } else {
        string token, part, label;
        uint i = line.find_first_of('\t');
        token = line.substr(0, i);
        // uint j = line.find_first_of('\t', i+1);
        // part = line.substr(i+1,j-i-1);
        // //cout << part << endl;
        i = line.find_last_of('\t');
        label = line.substr(i+1, line.size()-i-1);
        if (label_token_to_num.find(label) == label_token_to_num.end()) {
          cout << "Label " << label << " given index "<< num_labels << endl;
          label_token_to_num[label] = num_labels++;
        }

        x.push_back(token);
        t.push_back(to_string(label_token_to_num[label]));
      }
    }
    if (x.size() != 0) {
      X.push_back(x);
      T.push_back(t);
      x.clear();
      t.clear();
    }

    return num_labels;
  }
}


#endif
