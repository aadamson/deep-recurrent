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

class SoftmaxRegression {
public:
  SoftmaxRegression(uint nx, uint ny, LookupTable &LT);
  Matrix<double, 6, 2> train(vector<vector<string> > &sents,
                             vector<vector<string> > &labels,
                             vector<vector<string> > &validX,
                             vector<vector<string> > &validL,
                             vector<vector<string> > &testX,
                             vector<vector<string> > &testL);
  void update();
  Matrix<double, 3, 2> testSequential(vector<vector<string> > &sents,
                                      vector<vector<string> > &labels);
  LookupTable *LT;
  void save(string fname);
  void load(string fname);

private:
  MatrixXd forward(const vector<string> &s, int index=-1);
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

MatrixXd SoftmaxRegression::forward(const vector<string> &s, int index) {
  VectorXd dropper;
  uint T = s.size();

  MatrixXd x = MatrixXd(nx, T);
  for (uint i=0; i<T; i++)
    x.col(i) = (*LT)[s[i]];

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

void printResults(Matrix<double, 3, 2> results) {
  cout << "   " << " Prop " << "  Bin  " << endl;
  cout << "Pr " << results(0, 0) << " " << results(0, 1) << endl;
  cout << "Re " << results(1, 0) << " " << results(1, 1) << endl;
  cout << "F1 " << results(2, 0) << " " << results(2, 1) << endl;
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

int main(int argc, char **argv) {
  fold = atoi(argv[1]); // between 0-9
  srand(135);
  cout << setprecision(6);

  LookupTable LT;
  // i used mikolov's word2vec (300d) for my experiments, not CW
  LT.load("embeddings-original.EMBEDDING_SIZE=25.txt", 268810, 25, false);
  vector<vector<string> > X;
  vector<vector<string> > T;
  int ny = DataUtils::read_sentences(X, T, argv[2]); // dse.txt or ese.txt

  unordered_map<string, set<uint> > sentenceIds;
  set<string> allDocs;
  ifstream in("sentenceid.txt");
  string line;
  uint numericId = 0;
  while(getline(in, line)) {
    vector<string> s = split(line, ' ');
    assert(s.size() == 3);
    string strId = s[2];

    if (sentenceIds.find(strId) != sentenceIds.end()) {
      sentenceIds[strId].insert(numericId);
    } else {
      sentenceIds[strId] = set<uint>();
      sentenceIds[strId].insert(numericId);
    }
    numericId++;
  }

  vector<vector<string> > trainX, validX, testX;
  vector<vector<string> > trainL, validL, testL;
  vector<bool> isUsed(X.size(), false);

  ifstream in4("datasplit/doclist.mpqaOriginalSubset");
  while(getline(in4, line))
    allDocs.insert(line);

  ifstream in2("datasplit/filelist_train"+to_string(fold));
  while(getline(in2, line)) {
    for (const auto &id : sentenceIds[line]) {
      trainX.push_back(X[id]);
      trainL.push_back(T[id]);
    }
    allDocs.erase(line);
  }
  ifstream in3("datasplit/filelist_test"+to_string(fold));
  while(getline(in3, line)) {
    for (const auto &id : sentenceIds[line]) {
      testX.push_back(X[id]);
      testL.push_back(T[id]);
    }
    allDocs.erase(line);
  }

  uint validSize = 0;
  for (const auto &doc : allDocs) {
    for (const auto &id : sentenceIds[doc]) {
      validX.push_back(X[id]);
      validL.push_back(T[id]);
    }
  }

  cout << X.size() << " " << trainX.size() << " " << testX.size() << endl;
  cout << "Valid size: " << validX.size() << endl;

  Matrix<double, 6, 2> best = Matrix<double, 6, 2>::Zero();
  double bestDrop;
  for (DROP=0; DROP<0.1; DROP+=0.2) { // can use this loop for CV
    SoftmaxRegression srm(25,ny,LT);
    auto results = srm.train(trainX, trainL, validX, validL, testX, testL);
    if (best(2,0) < results(2,0)) { // propF1 on val set
      best = results;
      bestDrop = DROP;
    }
    srm.save("model.txt");
  }
  cout << "Best: " << endl;
  cout << "Drop: " << bestDrop << endl;
  cout << best << endl;

  return 0;
}
