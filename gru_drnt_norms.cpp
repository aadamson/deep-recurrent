#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <iterator>
#include <cassert>
#include <thread>
#include <getopt.h>
#include "Eigen/Dense"
#include "utils.cpp"
#include "data_utils/utils.cpp"
#include "gru_drnt.cpp"

void print_results(vector<string> sentence, MatrixXd predictions) {
  for (int i = 0; i < sentence.size(); i++) {
    string token = sentence[i];
    VectorXd prediction = predictions.col(i);

    cout << left << setw(16) << setfill(' ') << token;
    for (int j = 0; j < prediction.size(); j++) {
      cout << prediction(j) << "\t";
    }
    cout << endl;
  }
}

int main(int argc, char **argv) {
  cout << setprecision(3);

  // Set default arguments
  string model_path = "";
  string embeddings_file = "embeddings-original.EMBEDDING_SIZE=25.txt";
  int seed = 135;
  int embeddings_tokens = 268810;
  int nx = 25;
  int ny = 2;
  int nh = 25;

  int c;

  while (1) {
    static struct option long_options[] =
      {
        {"model",  required_argument, 0, 'a'},
        {"emb",    required_argument, 0, 'i'},
        {"nt",     required_argument, 0, 'j'},
        {"nx",     required_argument, 0, 'k'},
        {"ny",     required_argument, 0, 'l'},
        {"nh",     required_argument, 0, 'm'},
      };
    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long (argc, argv, "a:i:j:k:l:m:",
                     long_options, &option_index);    

    /* Detect the end of the options. */
    if (c == -1)
      break;

    switch (c) {
      case 0:
        /* If this option set a flag, do nothing else now. */
        if (long_options[option_index].flag != 0)
          break;
        printf ("option %s", long_options[option_index].name);
        if (optarg)
          printf (" with arg %s", optarg);
        printf ("\n");
        break;

      case 'a':
        model_path = optarg;
        break;

      case 'i':
        embeddings_file = optarg;
        break;

      case 'j':
        embeddings_tokens = stoi(optarg);
        break;

      case 'k':
        nx = stoi(optarg);
        break;

      case 'l':
        ny = stoi(optarg);
        break;

      case 'm':
        nh = stoi(optarg);
        break;        

      case '?':
        /* getopt_long already printed an error message. */
        break;

      default:
        abort ();
    }
  }

  srand(seed);

  LookupTable LT;
  
  GRURNN brnn(nx, nh, ny, LT, 0, 0, 0, 0, 0, false);
  brnn.load(model_path);

  brnn.print_norms();
 
  return 0;
}
