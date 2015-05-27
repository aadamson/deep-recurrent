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

int main(int argc, char **argv) {
  cout << setprecision(3);

  // Set default arguments
  int seed     = 135;
  float lambda = 1e-6;
  float lr     = 0.05;
  float mr     = 0.7;
  float null_class_weight = 0.5;
  float dropout_prob = 0.0;
  string embeddings_file = "embeddings-original.EMBEDDING_SIZE=25.txt";
  int embeddings_tokens = 268810;
  int nx = 25;
  string data  = "";
  string outdir = "models/";
  int num_epochs = 40;

  int c;

  while (1) {
    static struct option long_options[] =
      {
        {"seed",   required_argument, 0, 'a'},
        {"lr",     required_argument, 0, 'b'},
        {"mr",     required_argument, 0, 'c'},
        {"weight", required_argument, 0, 'd'},
        {"data",   required_argument, 0, 'f'},
        {"dr",     required_argument, 0, 'g'},
        {"lambda", required_argument, 0, 'h'},
        {"emb",    required_argument, 0, 'i'},
        {"nt",     required_argument, 0, 'j'},
        {"nx",     required_argument, 0, 'k'},
        {"epochs", required_argument, 0, 'l'},          
      };
    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long (argc, argv, "a:b:c:d:f:g:h:i:j:k:",
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
        seed = stoi(optarg);
        break;

      case 'b':
        lr = stof(optarg);
        break;

      case 'c':
        mr = stof(optarg);
        break;

      case 'd':
        null_class_weight = stof(optarg);
        break;

      case 'f':
        data = string(optarg);
        break;

      case 'g':
        dropout_prob = stof(optarg);
        break;

      case 'h':
        lambda = stof(optarg);
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
        num_epochs = stoi(optarg);
        break;        

      case '?':
        /* getopt_long already printed an error message. */
        break;

      default:
        abort ();
    }
  }

  srand(seed);

  string model_dir = outdir + filename(data) + "/gru_drnt";

  if (!conditional_mkdir(model_dir)) {
    cerr << "Failed to create model output directory " << model_dir << endl;
    cerr << "Exiting" << endl;
    return -1;
  }

  LookupTable LT;
  // i used mikolov's word2vec (300d) for my experiments, not CW
  LT.load(embeddings_file, embeddings_tokens, nx, false);
  vector<vector<string> > X;
  vector<vector<string> > T;
  int ny = DataUtils::read_sentences(X, T, data);

  vector<vector<string> > trainX, validX, testX;
  vector<vector<string> > trainL, validL, testL;

  DataUtils::generate_splits(X, T, trainX, trainL, validX, validL, testX, testL, 0.8, 0.1, 0.1);

  cout << "Total dataset size: " << X.size() << endl;
  cout << "Training set size: " << trainX.size() << endl;
  cout << "Validation set size: " << validX.size() << endl;
  cout << "Test set size: " << testX.size() << endl;
  
  GRURNN brnn(nx, 20, ny, LT, lambda, lr, mr, null_class_weight, dropout_prob, true);
  string outpath = model_dir + "/" + brnn.model_name();
  auto results = brnn.train(trainX, trainL, validX, validL, testX, testL, num_epochs, 80, outpath);
  
  return 0;
}
