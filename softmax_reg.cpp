//my softmax reg
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
#include "Eigen/Dense"
#include "utils.cpp"
#include "data_utils/utils.cpp"
#include <getopt.h>
#include "model.cpp"

#define uint unsigned int

#define DROPOUT
#define ETA 0.01
#define NORMALIZE false // keeping this false throughout my own experiments
#define OCLASS_WEIGHT 0.5
#define layers 3 // number of EXTRA (not all) hidden layers

#define MR 0.7
uint fold = -1;

using namespace Eigen;
using namespace std;

double LAMBDA = (layers > 2) ? 1e-3 : 1e-4;  // L2 regularizer on weights
double LAMBDAH = (layers > 2) ? 1e-5 : 1e-4; //L2 regularizer on activations
double DROP;

#ifdef DROPOUT
Matrix<double, -1, 1> dropout(Matrix<double, -1, 1> x, double p=DROP);
#endif

class SoftmaxRegression : public Model {
public:
  SoftmaxRegression(uint nx, uint ny, LookupTable &LT);
  void save(string fname);
  void load(string fname);
  double cost(const vector<string> &sent, const vector<string> &labels);

  Matrix<double, 6, 2> train(vector<vector<string> > &sents,
                             vector<vector<string> > &labels,
                             vector<vector<string> > &validX,
                             vector<vector<string> > &validL,
                             vector<vector<string> > &testX,
                             vector<vector<string> > &testL);
  bool is_nan();
  void update();
  string model_name();
  Matrix<double, 3, 2> testSequential(vector<vector<string> > &sents,
                                      vector<vector<string> > &labels);
  LookupTable *LT;
  

private:
 MatrixXd forward(const vector<string> &sent);
  double backward(const vector<string> &s, const vector<string> &labels);  

  // Parameters
  MatrixXd W;
  VectorXd b;

  // Gradients
  MatrixXd gW;
  VectorXd gb;

  // Velocities
  MatrixXd vW;
  VectorXd vb;

  uint nx, ny;
  uint epoch;

  double lr;
};

MatrixXd SoftmaxRegression::forward(const vector<string> &sent) {
  VectorXd dropper;
  uint T = sent.size();

  MatrixXd x = MatrixXd(nx, T);
  for (uint i=0; i<T; i++)
    x.col(i) = (*LT)[sent[i]];

  return softmax(b*RowVectorXd::Ones(T) + W*x);
}

double SoftmaxRegression::backward(const vector<string> &sent, const vector<string> &labels) {
  double cost = 0;
  uint T = sent.size();

  MatrixXd x = MatrixXd(nx, T);
  for (uint i=0; i<T; i++)
    x.col(i) = (*LT)[sent[i]];

  // Make predictions
  MatrixXd y = forward(sent);

  // Create one-hot vectors
  MatrixXd yi = MatrixXd::Zero(ny, T);
  for (uint i=0; i<T; i++) {
    int label_idx = stoi(labels[i]);
    cost += -log(y(label_idx, i));
    yi(label_idx, i) = 1;
  }

  // CE through softmax error vector
  MatrixXd delta = smaxentp(y, yi);
  for (uint i = 0; i < T; i++) {
    if (labels[i] == "0") {
      delta.col(i) *= OCLASS_WEIGHT;
    }
  }

  gb.noalias() += delta * RowVectorXd::Ones(T).transpose();
  gW.noalias() += delta * x.transpose();

  return (1.0 / T) * cost;
}

SoftmaxRegression::SoftmaxRegression(uint nx, uint ny, LookupTable &LT) {
  lr = ETA;

  this->LT = &LT;

  this->nx = nx;
  this->ny = ny;

  // Initialize the parameters randomly
  W = MatrixXd(ny,nx).unaryExpr(ptr_fun(urand));
  b = VectorXd(ny).unaryExpr(ptr_fun(urand));

  gW = MatrixXd::Zero(ny,nx);
  gb = VectorXd::Zero(ny);

  vW = MatrixXd::Zero(ny,nx);
  vb = VectorXd::Zero(ny);
}


double SoftmaxRegression:: cost(const vector<string> &sent, const vector<string> &labels){
  return 0; // not relevant 
}

bool SoftmaxRegression::is_nan() {
  return false; //not relevant
}

void SoftmaxRegression::update() {

  double lambda = LAMBDA;
  double mr = MR;
  double norm = 0;

  // update params
  b.noalias() -= lr*gb;
  W.noalias() -= lr*gW;

  // reset gradients
  gb.setZero();
  gW.setZero();

  lr *= 0.999;
}

void SoftmaxRegression::load(string fname) {
  ifstream in(fname.c_str());

  in >> nx >> ny;

  in >> W >> b;
}

void SoftmaxRegression::save(string fname) {
  ofstream out(fname.c_str());

  out << nx << " " << ny << endl;

  out << W << endl;
  out << b << endl;
}

Matrix<double, 6, 2>
SoftmaxRegression::train(vector<vector<string> > &sents,
                         vector<vector<string> > &labels,
                         vector<vector<string> > &validX,
                         vector<vector<string> > &validL,
                         vector<vector<string> > &testX,
                         vector<vector<string> > &testL) {
  uint MAXEPOCH = 200;
  uint MINIBATCH = 80;

  ostringstream strS;
  strS << "models/smr_" << DROP << "_"
  << MAXEPOCH << "_" << lr << "_" << LAMBDA << "_"
  << MR << "_" << fold;
  string fname = strS.str();

  vector<uint> perm;
  for (uint i=0; i<sents.size(); i++)
    perm.push_back(i);

  Matrix<double, 3, 2> bestVal, bestTest;
  bestVal << 0,0,0,0,0,0;

  for (epoch=0; epoch<MAXEPOCH; epoch++) {
    shuffle(perm);
    for (int i=0; i<sents.size(); i++) {
      double cost = backward(sents[perm[i]], labels[perm[i]]);
      if ((i+1) % MINIBATCH == 0 || i == sents.size()-1) {
        //cout << "Cost: " << cost << endl;
        update();
      }
    }
    if (epoch % 5 == 0) {
      Matrix<double, 3, 2> resVal, resTest, resVal2, resTest2;
      cout << "Epoch " << epoch << endl;
      cout << b << endl;

      // diagnostic
      /*
        cout << Wf.norm() << " " << Wb.norm() << " "
             << Vf.norm() << " " << Vb.norm() << " "
             << Wfo.norm() << " " << Wbo.norm() << endl;
        for (uint l=0; l<layers; l++) {
          cout << WWff[l].norm() << " " << WWfb[l].norm() << " "
               << WWbb[l].norm() << " " << WWbf[l].norm() << " "
               << VVf[l].norm() << " " << VVb[l].norm() << " "
               << WWfo[l].norm() << " " << WWbo[l].norm() << endl;
        }
      */
      cout << "Training results" << endl;
      printResults(testSequential(sents, labels));

      //cout << "P, R, F1:\n" << testSequential(sents, labels) << endl;
      resVal = testSequential(validX, validL);
      resTest = testSequential(testX, testL);

      cout << "Validation results" << endl;
      printResults(resVal);

      cout << "Test results" << endl;
      printResults(resTest);
      // cout << "  " << " Prop " << "  Bin  "
      // cout << "P " << reVal()
      // cout << "P, R, F1:\n" << resVal << endl;
      // cout << "P, R, F1" << endl;
      // cout << resTest  << endl<< endl;
      if (bestVal(2,0) < resVal(2,0)) {
        bestVal = resVal;
        bestTest = resTest;
        save(fname);
      }
    }
  }
  Matrix<double, 6, 2> results;
  results << bestVal, bestTest;
  return results;
}

// returns soft (precision, recall, F1) per expression
// counts proportional overlap & binary overlap
Matrix<double, 3, 2> SoftmaxRegression::testSequential(vector<vector<string> > &sents,
                                                       vector<vector<string> > &labels) {
  uint nExprPredicted = 0;
  double nExprPredictedCorrectly = 0;
  uint nExprTrue = 0;
  double precNumerProp = 0, precNumerBin = 0;
  double recallNumerProp = 0, recallNumerBin = 0;
  for (uint i=0; i<sents.size(); i++) { // per sentence
    vector<string> labelsPredicted;
    MatrixXd y_pred = forward(sents[i]);

    for (uint j=0; j<sents[i].size(); j++) {
      uint maxi = argmax(y_pred.col(j));
      labelsPredicted.push_back(to_string(maxi));
    }
    assert(labelsPredicted.size() == y_pred.cols());

    string y, t, py="", pt="";
    uint match = 0;
    uint exprSize = 0;
    vector<pair<uint,uint> > pred, tru;
    int l1=-1, l2=-1;

    if (labels[i].size() != labelsPredicted.size())
      cout << labels[i].size() << " " << labelsPredicted.size() << endl;
    for (uint j=0; j<labels[i].size(); j++) { // per token in a sentence
      t = labels[i][j];
      y = labelsPredicted[j];

      if (t != "0") {
        //nExprTrue++;
        if (t != pt) {
          if (l1 != -1) {
            tru.push_back(make_pair(l1, j));
          }
          l1 = j;
        }
      } else {
        if (l1 != -1)
          tru.push_back(make_pair(l1, j));
        l1 = -1;
      }

      if (y != "0") {
        if (y != py) {
          if (l2 != -1) {
            nExprPredicted++;
            pred.push_back(make_pair(l2, j));
          }
          l2 = j;
        }
      } else {
        if (l2 != -1) {
          nExprPredicted++;
          pred.push_back(make_pair(l2, j));
        }
        l2 = -1;
      }

//      if ((y == "B") || ((y == "I") && ((py == "") || (py == "O")))) {
//        nExprPredicted++;
//        if (l2 != -1)
//          pred.push_back(make_pair(l2,j));
//        l2 = j;
//      } else if (y == "I") {
//        assert(l2 != -1);
//      } else if (y == "O") {
//        if (l2 != -1)
//          pred.push_back(make_pair(l2,j));
//        l2 = -1;
//      } else {
//        cout << y << endl;
//        assert(false);
//      }

      py = y;
      pt = t;
    }
    if ((l1 != -1) && (l1 != labels[i].size()))
      tru.push_back(make_pair(l1,labels[i].size()));
    if ((l2 != -1) && (l2 != labels[i].size()))
      pred.push_back(make_pair(l2,labels[i].size()));

    vector<bool> trum = vector<bool>(tru.size(),false);
    vector<bool> predm = vector<bool>(pred.size(),false);
    for (uint a=0; a<tru.size(); a++) {
      pair<uint,uint> truSpan = tru[a];
      nExprTrue++;
      for (uint b=0; b<pred.size(); b++) {
        pair<uint,uint> predSpan = pred[b];

        uint lmax, rmin;
        if (truSpan.first > predSpan.first)
          lmax = truSpan.first;
        else
          lmax = predSpan.first;
        if (truSpan.second < predSpan.second)
          rmin = truSpan.second;
        else
          rmin = predSpan.second;

        uint overlap = 0;
        if (rmin > lmax)
          overlap = rmin-lmax;
        if (predSpan.second == predSpan.first) cout << predSpan.first << endl;
        assert(predSpan.second != predSpan.first);
        precNumerProp += (double)overlap/(predSpan.second-predSpan.first);
        recallNumerProp += (double)overlap/(truSpan.second-truSpan.first);
        if (!predm[b] && overlap > 0) {
          precNumerBin += (double)(overlap>0);
          predm[b] = true;
        }
        if (!trum[a] && overlap>0) {
          recallNumerBin += 1;
          trum[a]=true;
        }
      }
    }

  }
  double precisionProp = (nExprPredicted==0) ? 1 : precNumerProp/nExprPredicted;
  double recallProp = recallNumerProp/nExprTrue;
  double f1Prop = (2*precisionProp*recallProp)/(precisionProp+recallProp);
  double precisionBin = (nExprPredicted==0) ? 1 : precNumerBin/nExprPredicted;
  double recallBin = recallNumerBin/nExprTrue;
  double f1Bin = (2*precisionBin*recallBin)/(precisionBin+recallBin);
  Matrix<double, 3, 2> results;
  results << precisionProp, precisionBin,
      recallProp, recallBin,
      f1Prop, f1Bin;
  return results;
}

#ifdef DROPOUT
Matrix<double, -1, 1> dropout(Matrix<double, -1, 1> x, double p) {
  for (uint i=0; i<x.size(); i++) {
    if ((double)rand()/RAND_MAX < p)
      x(i) = 0;
  }
  return x;
}
#endif

string SoftmaxRegression::model_name() {
  ostringstream strS;
  strS << "softmax_reg" << layers << lr << "_" << LAMBDA << "_" << MR ;
  string fname = strS.str();
  return fname;
}

int main(int argc, char **argv) {
  fold = atoi(argv[1]); // between 0-9
  cout << setprecision(6);
  //===
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

  int c;

  while (1) {
    static struct option long_options[] =
      {

        //CAN ALSO REWIRE LAMBDA, lr ETA, mr MR


        {"seed",   required_argument, 0, 'a'},
        // {"lr",     required_argument, 0, 'b'},
        // {"mr",     required_argument, 0, 'c'},
        // {"weight", required_argument, 0, 'd'},
        {"data",   required_argument, 0, 'f'},
        // {"dr",     required_argument, 0, 'g'},
        // {"lambda", required_argument, 0, 'h'},
        {"emb",    required_argument, 0, 'i'},
        {"nt",     required_argument, 0, 'j'},
        {"nx",     required_argument, 0, 'k'},          
      };
    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long (argc, argv, "a:f:i:j:k:",//"a:b:c:d:f:g:h:i:j:k:",
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

      // case 'b':
      //   lr = stof(optarg);
      //   break;

      // case 'c':
      //   mr = stof(optarg);
      //   break;

      // case 'd':
      //   null_class_weight = stof(optarg);
      //   break;

      case 'f':
        data = string(optarg);
        break;

      // case 'g':
      //   dropout_prob = stof(optarg);
      //   break;

      // case 'h':
      //   lambda = stof(optarg);
      //   break;

      case 'i':
        embeddings_file = optarg;
        break;

      case 'j':
        embeddings_tokens = stoi(optarg);
        break;

      case 'k':
        nx = stoi(optarg);
        break;

      case '?':
        /* getopt_long already printed an error message. */
        break;

      default:
        abort ();
    }
  }


  //----
  srand(seed);

  LookupTable LT;
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

  SoftmaxRegression srm(nx,ny,LT);
  auto results = srm.train(trainX, trainL, validX, validL, testX, testL);

  srm.save("model.txt");

  return 0;
}

