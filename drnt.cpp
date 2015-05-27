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
#include "model.cpp"

#define NORMALIZE false // keeping this false throughout my own experiments
#define layers 3 // number of EXTRA (not all) hidden layers

using namespace Eigen;
using namespace std;

double LAMBDA = (layers > 2) ? 1e-5 : 1e-4;  // L2 regularizer on weights
//double LAMBDAH = (layers > 2) ? 1e-5 : 1e-4; //L2 regularizer on activations
double LAMBDAH = 0; //L2 regularizer on activations

Matrix<double, -1, 1> dropout(Matrix<double, -1, 1> x, double p);

class RNN : public Model {
public:
  RNN(uint nx, uint nh, uint ny, LookupTable &LT, float lambda, float lr, float mr, float null_class_weight, float dropout = 0.0, bool error_signal = false);

  void save(string fname);
  void load(string fname);
  MatrixXd forward(const vector<string> &sent);
  double backward(const vector<string> &sent, 
                  const vector<string> &labels);
  double cost(const vector<string> &sent, const vector<string> &labels);
  void update();
  bool is_nan();
  string model_name();

  LookupTable *LT;

private:
  MatrixXd (*f)(const MatrixXd& x);
  MatrixXd (*fp)(const MatrixXd& x);

  MatrixXd hf, hb, hhf[layers], hhb[layers];

  // recurrent network params
  // WW(f/b)o is forward/backward matrix if output layer
  MatrixXd Wo, Wfo, Wbo, WWfo[layers], WWbo[layers];
  VectorXd bo;
  MatrixXd Wf, Vf, Wb, Vb;
  VectorXd bhf, bhb;

  MatrixXd WWff[layers], WWfb[layers], WWbb[layers], WWbf[layers];
  MatrixXd VVf[layers], VVb[layers];
  VectorXd bbhf[layers], bbhb[layers];


  // recurrent network gradients
  MatrixXd gWo, gWfo, gWbo, gWWfo[layers], gWWbo[layers];
  VectorXd gbo;
  MatrixXd gWf, gVf, gWb, gVb;
  VectorXd gbhf, gbhb;

  MatrixXd gWWff[layers], gWWfb[layers], gWWbb[layers], gWWbf[layers];
  MatrixXd gVVf[layers], gVVb[layers];
  VectorXd gbbhf[layers], gbbhb[layers];

  // recurrent network velocities
  MatrixXd vWo, vWfo, vWbo, vWWfo[layers], vWWbo[layers];
  VectorXd vbo;
  MatrixXd vWf, vVf, vWb, vVb;
  VectorXd vbhf, vbhb;

  MatrixXd vWWff[layers], vWWfb[layers], vWWbb[layers], vWWbf[layers];
  MatrixXd vVVf[layers], vVVb[layers];
  VectorXd vbbhf[layers], vbbhb[layers];

  uint nx, nh, ny;
  uint epoch;

  float lambda, lr, mr, null_class_weight, dropout_prob, error_signal;
};

RNN::RNN(uint nx, uint nh, uint ny, LookupTable &LT, float lambda, float lr, float mr, float null_class_weight, float dropout, bool error_signal) :
  LT(&LT), nx(nx), nh(nh), ny(ny), lambda(lambda), lr(lr), mr(mr), null_class_weight(null_class_weight), dropout_prob(dropout), error_signal(error_signal)
{
  f = &relu;
  fp = &relup;

  // init randomly
  Wf = MatrixXd(nh,nx).unaryExpr(ptr_fun(urand));
  Vf = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
  bhf = VectorXd(nh).unaryExpr(ptr_fun(urand));

  Wb = MatrixXd(nh,nx).unaryExpr(ptr_fun(urand));
  Vb = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
  bhb = VectorXd(nh).unaryExpr(ptr_fun(urand));

  for (uint l=0; l<layers; l++) {
    WWff[l] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
    WWfb[l] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
    VVf[l] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
    bbhf[l] = VectorXd(nh).unaryExpr(ptr_fun(urand));

    WWbb[l] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
    WWbf[l] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
    VVb[l] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
    bbhb[l] = VectorXd(nh).unaryExpr(ptr_fun(urand));
  }

  Wfo = MatrixXd(ny,nh).unaryExpr(ptr_fun(urand));
  Wbo = MatrixXd(ny,nh).unaryExpr(ptr_fun(urand));
  for (uint l=0; l<layers; l++) {
    WWfo[l] = MatrixXd(ny,nh).unaryExpr(ptr_fun(urand));
    WWbo[l] = MatrixXd(ny,nh).unaryExpr(ptr_fun(urand));
  }
  Wo = MatrixXd(ny,nx).unaryExpr(ptr_fun(urand));
  bo = VectorXd(ny).unaryExpr(ptr_fun(urand));

  gWf = MatrixXd::Zero(nh,nx);
  gVf = MatrixXd::Zero(nh,nh);
  gbhf = VectorXd::Zero(nh);

  gWb = MatrixXd::Zero(nh,nx);
  gVb = MatrixXd::Zero(nh,nh);
  gbhb = VectorXd::Zero(nh);

  for (uint l=0; l<layers; l++) {
    gWWff[l] = MatrixXd::Zero(nh,nh);
    gWWfb[l] = MatrixXd::Zero(nh,nh);
    gVVf[l] = MatrixXd::Zero(nh,nh);
    gbbhf[l] = VectorXd::Zero(nh);

    gWWbb[l] = MatrixXd::Zero(nh,nh);
    gWWbf[l] = MatrixXd::Zero(nh,nh);
    gVVb[l] = MatrixXd::Zero(nh,nh);
    gbbhb[l] = VectorXd::Zero(nh);
  }


  gWfo = MatrixXd::Zero(ny,nh);
  gWbo = MatrixXd::Zero(ny,nh);
  for (uint l=0; l<layers; l++) {
    gWWfo[l] = MatrixXd::Zero(ny,nh);
    gWWbo[l] = MatrixXd::Zero(ny,nh);
  }
  gWo = MatrixXd::Zero(ny,nx);
  gbo = VectorXd::Zero(ny);

  vWf = MatrixXd::Zero(nh,nx);
  vVf = MatrixXd::Zero(nh,nh);
  vbhf = VectorXd::Zero(nh);

  vWb = MatrixXd::Zero(nh,nx);
  vVb = MatrixXd::Zero(nh,nh);
  vbhb = VectorXd::Zero(nh);

  for (uint l=0; l<layers; l++) {
    vWWff[l] = MatrixXd::Zero(nh,nh);
    vWWfb[l] = MatrixXd::Zero(nh,nh);
    vVVf[l] = MatrixXd::Zero(nh,nh);
    vbbhf[l] = VectorXd::Zero(nh);

    vWWbb[l] = MatrixXd::Zero(nh,nh);
    vWWbf[l] = MatrixXd::Zero(nh,nh);
    vVVb[l] = MatrixXd::Zero(nh,nh);
    vbbhb[l] = VectorXd::Zero(nh);
  }


  vWfo = MatrixXd::Zero(ny,nh);
  vWbo = MatrixXd::Zero(ny,nh);
  for (uint l=0; l<layers; l++) {
    vWWfo[l] = MatrixXd::Zero(ny,nh);
    vWWbo[l] = MatrixXd::Zero(ny,nh);
  }
  vWo = MatrixXd::Zero(ny,nx);
  vbo = VectorXd::Zero(ny);
}

double RNN::cost(const vector<string> &sent, const vector<string> &labels) {
  double cost = 0.0;
  uint T = sent.size();

  // Build the input x from the embeddings
  MatrixXd x = MatrixXd(nx, T);
  for (uint i = 0; i < T; i++) 
    x.col(i) = (*LT)[sent[i]];

  MatrixXd y_hat = forward(sent);
  // Create one-hot vectors
  MatrixXd y = MatrixXd::Zero(ny, T);
  for (uint i=0; i<T; i++) {
    int label_idx = stoi(labels[i]);
    y(label_idx, i) = 1;
    cost += log(y_hat(label_idx, i));
  }

  return -cost;
}

MatrixXd RNN::forward(const vector<string> &sent) {
  VectorXd dropper;
  uint T = sent.size();
  MatrixXd x = MatrixXd(nx, T);

  for (uint i=0; i<T; i++)
    x.col(i) = (*LT)[sent[i]];

  hf = MatrixXd::Zero(nh, T);
  hb = MatrixXd::Zero(nh, T);

  for (uint l=0; l<layers; l++) {
    hhf[l] = MatrixXd::Zero(nh, T);
    hhb[l] = MatrixXd::Zero(nh, T);
  }

  MatrixXd Wfx = Wf*x + bhf*RowVectorXd::Ones(T);
  dropper = dropout(VectorXd::Ones(nh), dropout_prob);
  for (uint i=0; i<T; i++) {
    hf.col(i) = (i==0) ? f(Wfx.col(i)) : f(Wfx.col(i) + Vf*hf.col(i-1));
    hf.col(i) = hf.col(i).cwiseProduct(dropper);
  }

  MatrixXd Wbx = Wb*x + bhb*RowVectorXd::Ones(T);
  dropper = dropout(VectorXd::Ones(nh), dropout_prob);
  for (uint i=T-1; i!=(uint)(-1); i--) {
    hb.col(i) = (i==T-1) ? f(Wbx.col(i)) : f(Wbx.col(i) + Vb*hb.col(i+1));
    hb.col(i) = hb.col(i).cwiseProduct(dropper);
  }

  for (uint l=0; l<layers; l++) {
    MatrixXd *xf, *xb; // input to this layer (not to all network)
    xf = (l == 0) ? &hf : &(hhf[l-1]);
    xb = (l == 0) ? &hb : &(hhb[l-1]);

    MatrixXd WWffxf = WWff[l]* *xf + bbhf[l]*RowVectorXd::Ones(T);
    MatrixXd WWfbxb = WWfb[l]* *xb;
    dropper = dropout(VectorXd::Ones(nh), dropout_prob);
    for (uint i=0; i<T; i++) {
      hhf[l].col(i) = (i==0) ? f(WWffxf.col(i) + WWfbxb.col(i))
                             : f(WWffxf.col(i) + WWfbxb.col(i) +
                                 VVf[l]*hhf[l].col(i-1));
      hhf[l].col(i) = hhf[l].col(i).cwiseProduct(dropper);
    }

    MatrixXd WWbfxf = WWbf[l]* *xf + bbhb[l]*RowVectorXd::Ones(T);
    MatrixXd WWbbxb = WWbb[l]* *xb;
    dropper = dropout(VectorXd::Ones(nh), dropout_prob);
    for (uint i=T-1; i!=(uint)(-1); i--) {
      hhb[l].col(i) = (i==T-1) ? f(WWbbxb.col(i) + WWbfxf.col(i))
                               : f(WWbbxb.col(i) + WWbfxf.col(i) +
                                   VVb[l]*hhb[l].col(i+1));
      hhb[l].col(i) = hhb[l].col(i).cwiseProduct(dropper);
    }
  }

  // output layer uses the last hidden layer
  // you can experiment with the other version by changing this
  // (backward pass needs to change as well of course)
  return softmax(bo*RowVectorXd::Ones(T) + WWfo[layers-1]*hhf[layers-1] +
              WWbo[layers-1]*hhb[layers-1]);
}

double RNN::backward(const vector<string> &sent, const vector<string> &labels) {
  double cost = 0.0;
  uint T = sent.size();

  // Build the input x from the embeddings
  MatrixXd x = MatrixXd(nx, T);
  for (uint i = 0; i < T; i++) 
    x.col(i) = (*LT)[sent[i]];

  // Create one-hot vectors
  MatrixXd y = MatrixXd::Zero(ny, T);
  for (uint i=0; i<T; i++) {
    int label_idx = stoi(labels[i]);
    y(label_idx, i) = 1;
  }
  MatrixXd y_hat = forward(sent);

  // Get error vector propagated by cross-entropy error passed through softmax
  // output layer
  MatrixXd delta_y = smaxentp(y_hat, y);
  // We dampen the error propagated by mis-classifying null-class tokens so that
  // the model doesn't simply learn the prior
  for (uint i = 0; i < T; i++) {
    if (labels[i] == "0")
      delta_y.col(i) *= null_class_weight;
  } 

  // Calculate output layer gradients
  gWWfo[layers-1].noalias() += delta_y * hhf[layers-1].transpose();
  gWWbo[layers-1].noalias() += delta_y * hhb[layers-1].transpose();
  gbo.noalias() += delta_y * VectorXd::Ones(T);

  MatrixXd deltaf[layers];
  MatrixXd deltab[layers];
  for (uint l = 0; l < layers; l++) {
    deltaf[l] = MatrixXd::Zero(nh, T);
    deltab[l] = MatrixXd::Zero(nh, T);
  }
  // Create error vectors propagated by hidden units
  // Note that ReLU'(x) = ReLU'(ReLU(x)). In general, we assume that for the
  // hidden unit non-linearity f, f(z) = f'(f(z))
  for (uint l = layers-1; l != (uint)(-1); l--) {
    MatrixXd cur_hf = (l == 0) ? hf : hhf[l-1];
    MatrixXd cur_hb = (l == 0) ? hb : hhb[l-1];

    MatrixXd fpf = fp(hhf[l]);
    MatrixXd fpb = fp(hhb[l]);

    if (l != layers-1) {
      // Update error vectors by propagating back from layer above
      deltaf[l].noalias() += fpf.cwiseProduct(WWff[l+1].transpose() * deltaf[l+1] + WWbf[l+1].transpose() * deltab[l+1]);
      deltab[l].noalias() += fpb.cwiseProduct(WWfb[l+1].transpose() * deltaf[l+1] + WWbb[l+1].transpose() * deltab[l+1]);

      if (error_signal) {
        // Add supervised error signal (i.e. WW(f/b)o * delta_y)
        deltaf[l].noalias() += fpf.cwiseProduct(WWfo[l].transpose() * delta_y); 
        deltab[l].noalias() += fpb.cwiseProduct(WWbo[l].transpose() * delta_y);
      } 
    } else {
      deltaf[l].noalias() += fpf.cwiseProduct(WWfo[l].transpose() * delta_y);
      deltab[l].noalias() += fpb.cwiseProduct(WWbo[l].transpose() * delta_y);
    }

    // Update error vectors by propagating back from neighbor. Note that e.g.
    // the rightmost forward neighbor does not have a neighbor to the right,
    // so there is nothing to propagate back
    for (uint t = 1; t < T; t++) {
      deltab[l].col(t) += fpb.col(t).cwiseProduct(VVb[l].transpose() * deltab[l].col(t-1));
      gVVb[l].noalias() += deltab[l].col(t-1) * hhb[l].col(t).transpose();
    }

    for (uint t = T-2; t != (uint)(-1); t--) {
      deltaf[l].col(t) += fpf.col(t).cwiseProduct(VVf[l].transpose() * deltaf[l].col(t+1));
      gVVf[l].noalias() += deltaf[l].col(t+1) * hhf[l].col(t).transpose();
    }

    gWWff[l].noalias() += deltaf[l] * cur_hf.transpose();
    gWWfb[l].noalias() += deltaf[l] * cur_hb.transpose();
    gbbhf[l].noalias() += deltaf[l] * VectorXd::Ones(T);

    gWWbf[l].noalias() += deltab[l] * cur_hf.transpose();
    gWWbb[l].noalias() += deltab[l] * cur_hb.transpose();
    gbbhb[l].noalias() += deltab[l] * VectorXd::Ones(T);
  }

  // Calculate error vectors for input layer
  MatrixXd deltaf_i = MatrixXd::Zero(nh, T);
  MatrixXd deltab_i = MatrixXd::Zero(nh, T);

  MatrixXd fpf = fp(hf);
  MatrixXd fpb = fp(hb);

  deltaf_i.noalias() += fpf.cwiseProduct(WWff[0].transpose() * deltaf[0] + WWbf[0].transpose() * deltab[0]);
  deltab_i.noalias() += fpb.cwiseProduct(WWfb[0].transpose() * deltaf[0] + WWbb[0].transpose() * deltab[0]);

#ifdef ERROR_SIGNAL
  // Add supervised error signal (i.e. WW(f/b)o * delta_y)
  deltaf_i.noalias() += fpf.cwiseProduct(Wfo.transpose() * delta_y); 
  deltab_i.noalias() += fpb.cwiseProduct(Wbo.transpose() * delta_y); 
#endif

  for (uint t = T-2; t < (uint)(-1); t--) {
    deltaf_i.col(t) += fpf.col(t).cwiseProduct(Vf.transpose() * deltaf_i.col(t+1));
    gVf.noalias() += deltaf_i.col(t+1) * hf.col(t).transpose();
  }
  for (uint t = 1; t < T; t++) {
    deltab_i.col(t) += fpb.col(t).cwiseProduct(Vb.transpose() * deltab_i.col(t-1));
    gVb.noalias() += deltab_i.col(t-1) * hb.col(t).transpose();
  }

  // Calculate input layer gradients
  gWf.noalias() += deltaf_i * x.transpose();
  gWb.noalias() += deltab_i * x.transpose();

  gbhf.noalias() += deltaf_i * VectorXd::Ones(T);
  gbhb.noalias() += deltab_i * VectorXd::Ones(T);

  return cost;
}

void RNN::update() {
  double norm = 0;

  // regularize
  gbo.noalias() += lambda*bo;
  for (uint l=layers-1; l<layers; l++) {
    gWWfo[l].noalias() += (lambda)*WWfo[l];
    gWWbo[l].noalias() += (lambda)*WWbo[l];
  }

  norm += 0.1* (gWo.squaredNorm() + gbo.squaredNorm());
  for (uint l=0; l<layers; l++)
    norm+= 0.1*(gWWfo[l].squaredNorm() + gWWbo[l].squaredNorm());

  gWf.noalias() += lambda*Wf;
  gVf.noalias() += lambda*Vf;
  gWb.noalias() += lambda*Wb;
  gVb.noalias() += lambda*Vb;
  gbhf.noalias() += lambda*bhf;
  gbhb.noalias() += lambda*bhb;

  norm += gWf.squaredNorm() + gVf.squaredNorm()
          + gWb.squaredNorm() + gWf.squaredNorm()
          + gbhf.squaredNorm() + gbhb.squaredNorm();

  for (uint l=0; l<layers; l++) {
    gWWff[l].noalias() += lambda*WWff[l];
    gWWfb[l].noalias() += lambda*WWfb[l];
    gWWbf[l].noalias() += lambda*WWbf[l];
    gWWbb[l].noalias() += lambda*WWbb[l];
    gVVf[l].noalias() += lambda*VVf[l];
    gVVb[l].noalias() += lambda*VVb[l];
    gbbhf[l].noalias() += lambda*bbhf[l];
    gbbhb[l].noalias() += lambda*bbhb[l];

    norm += gWWff[l].squaredNorm() + gWWfb[l].squaredNorm()
            + gWWbf[l].squaredNorm() + gWWbb[l].squaredNorm()
            + gVVf[l].squaredNorm() + gVVb[l].squaredNorm()
            + gbbhf[l].squaredNorm() + gbbhb[l].squaredNorm();

  }

  // update velocities
  vbo = 0.1*lr*gbo + mr*vbo;
  for (uint l=layers-1; l<layers; l++) {
    vWWfo[l] = 0.1*lr*gWWfo[l] + mr*vWWfo[l];
    vWWbo[l] = 0.1*lr*gWWbo[l] + mr*vWWbo[l];
  }

  if (NORMALIZE)
    norm = (norm > 25) ? sqrt(norm/25) : 1;
  else
    norm = 1;

  vWf  = lr*gWf/norm  + mr*vWf;
  vVf  = lr*gVf/norm  + mr*vVf;
  vWb  = lr*gWb/norm  + mr*vWb;
  vVb  = lr*gVb/norm  + mr*vVb;
  vbhf = lr*gbhf/norm + mr*vbhf;
  vbhb = lr*gbhb/norm + mr*vbhb;

  for (uint l=0; l<layers; l++) {
    vWWff[l] = lr*gWWff[l]/norm + mr*vWWff[l];
    vWWfb[l] = lr*gWWfb[l]/norm + mr*vWWfb[l];
    vVVf[l]  = lr*gVVf[l]/norm  + mr*vVVf[l];
    vWWbb[l] = lr*gWWbb[l]/norm + mr*vWWbb[l];
    vWWbf[l] = lr*gWWbf[l]/norm + mr*vWWbf[l];
    vVVb[l]  = lr*gVVb[l]/norm  + mr*vVVb[l];
    vbbhf[l] = lr*gbbhf[l]/norm + mr*vbbhf[l];
    vbbhb[l] = lr*gbbhb[l]/norm + mr*vbbhb[l];
  }

  // update params
  bo.noalias() -= vbo;
  for (uint l=layers-1; l<layers; l++) {
    WWfo[l].noalias() -= vWWfo[l];
    WWbo[l].noalias() -= vWWbo[l];
  }

  Wf.noalias()  -= vWf;
  Vf.noalias()  -= vVf;
  Wb.noalias()  -= vWb;
  Vb.noalias()  -= vVb;
  bhf.noalias() -= vbhf;
  bhb.noalias() -= vbhb;

  for (uint l=0; l<layers; l++) {
    WWff[l].noalias() -= vWWff[l];
    WWfb[l].noalias() -= vWWfb[l];
    VVf[l].noalias()  -= vVVf[l];
    WWbb[l].noalias() -= vWWbb[l];
    WWbf[l].noalias() -= vWWbf[l];
    VVb[l].noalias()  -= vVVb[l];
    bbhf[l].noalias() -= vbbhf[l];
    bbhb[l].noalias() -= vbbhb[l];
  }

  // reset gradients
  gbo.setZero();
  for (uint l=layers-1; l<layers; l++) {
    gWWfo[l].setZero();
    gWWbo[l].setZero();
  }

  gWf.setZero();
  gVf.setZero();
  gWb.setZero();
  gVb.setZero();
  gbhf.setZero();
  gbhb.setZero();

  for (uint l=0; l<layers; l++) {
    gWWff[l].setZero();
    gWWfb[l].setZero();
    gVVf[l].setZero();
    gWWbb[l].setZero();
    gWWbf[l].setZero();
    gVVb[l].setZero();
    gbbhf[l].setZero();
    gbbhb[l].setZero();
  }

  lr *= 0.999;
  //cout << Wuo << endl;
}

bool RNN::is_nan() {
  return (bbhf[0](0) != bbhf[0](0));
}

void RNN::load(string fname) {
  ifstream in(fname.c_str());

  in >> nx >> nh >> nh >> ny;

  in >> Wf >> Vf >> bhf
  >> Wb >> Vb >> bhb;

  for (uint l=0; l<layers; l++) {
    in >> WWff[l] >> WWfb[l] >> VVf[l] >> bbhf[l]
    >> WWbb[l] >> WWbf[l] >> VVb[l] >> bbhb[l];
  }

  in >> Wfo >> Wbo;
  for (uint l=0; l<layers; l++)
    in >> WWfo[l] >> WWbo[l];
  in >> Wo >> bo;
}

void RNN::save(string fname) {
  ofstream out(fname.c_str());

  out << nx << " " << nh << " " << nh << " " << ny << endl;

  out << Wf << endl;
  out << Vf << endl;
  out << bhf << endl;

  out << Wb << endl;
  out << Vb << endl;
  out << bhb << endl;

  for (uint l=0; l<layers; l++) {
    out << WWff[l] << endl;
    out << WWfb[l] << endl;
    out << VVf[l] << endl;
    out << bbhf[l] << endl;

    out << WWbb[l] << endl;
    out << WWbf[l] << endl;
    out << VVb[l]  << endl;
    out << bbhb[l] << endl;
  }

  out << Wfo << endl;
  out << Wbo << endl;
  for (uint l=0; l<layers; l++) {
    out << WWfo[l] << endl;
    out << WWbo[l] << endl;
  }
  out << Wo << endl;
  out << bo << endl;
}

Matrix<double, -1, 1> dropout(Matrix<double, -1, 1> x, double p) {
  for (uint i=0; i<x.size(); i++) {
    if ((double)rand()/RAND_MAX < p)
      x(i) = 0;
  }
  return x;
}

string RNN::model_name() {
  ostringstream strS;
  strS << "drnt_layers_" << layers << "_nh_" << nh
  << "_dr_" << dropout_prob << "_lr_"<< lr << "_error_signal_" << error_signal
  << "_lambda_" << lambda << "_mr_" << mr << "_weight_" << null_class_weight;
  string fname = strS.str();
  return fname;
}

