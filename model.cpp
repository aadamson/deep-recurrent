#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "utils.cpp"
#include "Eigen/Dense"

#define uint unsigned int

using namespace std;
using namespace Eigen;

class Model {
public:
  virtual void save(string fname) = 0;
  virtual void load(string fname) = 0;
  
  virtual MatrixXd forward(const vector<string> &sent) = 0;
  virtual double backward(const vector<string> &sent, 
                          const vector<string> &labels) = 0;
  virtual void update() = 0;

  virtual string model_name() = 0;

  Matrix<double, 6, 2> train(vector<vector<string> > &sents,
                             vector<vector<string> > &labels,
                             vector<vector<string> > &validX,
                             vector<vector<string> > &validL,
                             vector<vector<string> > &testX,
                             vector<vector<string> > &testL,
                             uint max_epoch,
                             uint mini_batch);
  Matrix<double, 3, 2> testSequential(vector<vector<string> > &sents,
                                      vector<vector<string> > &labels);
};

void printResults(Matrix<double, 3, 2> results) {
  cout << "   " << " Prop " << "  Bin  " << endl;
  cout << "Pr " << results(0, 0) << " " << results(0, 1) << endl;
  cout << "Re " << results(1, 0) << " " << results(1, 1) << endl;
  cout << "F1 " << results(2, 0) << " " << results(2, 1) << endl;
}

// returns soft (precision, recall, F1) per expression
// counts proportional overlap & binary overlap
// returns soft (precision, recall, F1) per expression
// counts proportional overlap & binary overlap
Matrix<double, 3, 2> Model::testSequential(vector<vector<string> > &sents,
                                           vector<vector<string> > &labels) {
  vector<string> y_hat;
  vector<string> y_truth;
  uint ny = 0;

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
      ny = y_pred.rows();
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

      y_hat.push_back(y);
      y_truth.push_back(t);

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

  MatrixXd conf_mat = confusion_matrix(y_hat, y_truth, ny);
  cout << "Confusion matrix" << endl;
  cout << conf_mat << endl;

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

Matrix<double, 6, 2>
Model::train(vector<vector<string> > &sents,
             vector<vector<string> > &labels,
             vector<vector<string> > &validX,
             vector<vector<string> > &validL,
             vector<vector<string> > &testX,
             vector<vector<string> > &testL,
             uint max_epoch,
             uint mini_batch) {
  vector<uint> perm;
  for (uint i=0; i<sents.size(); i++)
    perm.push_back(i);

  Matrix<double, 3, 2> bestVal, bestTest;
  bestVal << 0,0,0,0,0,0;

  for (uint epoch = 0; epoch < max_epoch; epoch++) {
    shuffle(perm);
    for (int i = 0; i < sents.size(); i++) {
      backward(sents[perm[i]], labels[perm[i]]);
      if ((i+1) % mini_batch == 0 || i == sents.size()-1)
        update();
    }
    if (epoch % 10 == 0) {
      Matrix<double, 3, 2> resVal, resTest, resVal2, resTest2;
      cout << "Epoch " << epoch << endl;

      cout << "Training results" << endl;
      printResults(testSequential(sents, labels));

      resVal = testSequential(validX, validL);
      cout << "Validation results" << endl;
      printResults(resVal);

      resTest = testSequential(testX, testL);
      cout << "Test results" << endl;
      printResults(resTest);

      cout << endl;
      
      if (bestVal(2,0) < resVal(2,0)) {
        bestVal = resVal;
        bestTest = resTest;
        save("models/" + model_name());
      }
    }
  }
  Matrix<double, 6, 2> results;
  results << bestVal, bestTest;
  return results;
}
