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
#include "model.cpp"

#define DROPOUT
#define ETA 0.005
#define NORMALIZE false // keeping this false throughout my own experiments
#define OCLASS_WEIGHT 0.5
#define layers 2 // number of EXTRA (not all) hidden layers

#define MR 0.7
uint fold = -1;

using namespace Eigen;
using namespace std;

double LAMBDA = (layers > 2) ? 1e-5 : 1e-4;  // L2 regularizer on weights
//double LAMBDAH = (layers > 2) ? 1e-5 : 1e-4; //L2 regularizer on activations
double LAMBDAH = 0; //L2 regularizer on activations
double DROP;

#ifdef DROPOUT
Matrix<double, -1, 1> dropout(Matrix<double, -1, 1> x, double p=DROP);
#endif

class RNN : public Model {
public:
  RNN(uint nx, uint nhf, uint nhb, uint ny, LookupTable &LT);

  void save(string fname);
  void load(string fname);

  MatrixXd forward(const vector<string> &sent);
  double backward(const vector<string> &sent, 
                  const vector<string> &labels);
  void update();

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

  uint nx, nhf, nhb, ny;
  uint epoch;

  double lr;
};

MatrixXd RNN::forward(const vector<string> &sent) {
  VectorXd dropper;
  uint T = sent.size();
  MatrixXd x = MatrixXd(nx, T);

  for (uint i=0; i<T; i++)
    x.col(i) = (*LT)[sent[i]];

  hf = MatrixXd::Zero(nhf, T);
  hb = MatrixXd::Zero(nhb, T);

  for (uint l=0; l<layers; l++) {
    hhf[l] = MatrixXd::Zero(nhf, T);
    hhb[l] = MatrixXd::Zero(nhb, T);
  }

  MatrixXd Wfx = Wf*x + bhf*RowVectorXd::Ones(T);
  dropper = dropout(VectorXd::Ones(nhf));
  for (uint i=0; i<T; i++) {
    hf.col(i) = (i==0) ? f(Wfx.col(i)) : f(Wfx.col(i) + Vf*hf.col(i-1));
#ifdef DROPOUT
    hf.col(i) = hf.col(i).cwiseProduct(dropper);
#endif
  }

  MatrixXd Wbx = Wb*x + bhb*RowVectorXd::Ones(T);
  dropper = dropout(VectorXd::Ones(nhb));
  for (uint i=T-1; i!=(uint)(-1); i--) {
    hb.col(i) = (i==T-1) ? f(Wbx.col(i)) : f(Wbx.col(i) + Vb*hb.col(i+1));
#ifdef DROPOUT
    hb.col(i) = hb.col(i).cwiseProduct(dropper);
#endif
  }

  for (uint l=0; l<layers; l++) {
    MatrixXd *xf, *xb; // input to this layer (not to all network)
    xf = (l == 0) ? &hf : &(hhf[l-1]);
    xb = (l == 0) ? &hb : &(hhb[l-1]);

    MatrixXd WWffxf = WWff[l]* *xf + bbhf[l]*RowVectorXd::Ones(T);
    MatrixXd WWfbxb = WWfb[l]* *xb;
    dropper = dropout(VectorXd::Ones(nhf));
    for (uint i=0; i<T; i++) {
      hhf[l].col(i) = (i==0) ? f(WWffxf.col(i) + WWfbxb.col(i))
                             : f(WWffxf.col(i) + WWfbxb.col(i) +
                                 VVf[l]*hhf[l].col(i-1));
#ifdef DROPOUT
      hhf[l].col(i) = hhf[l].col(i).cwiseProduct(dropper);
#endif
    }

    MatrixXd WWbfxf = WWbf[l]* *xf + bbhb[l]*RowVectorXd::Ones(T);
    MatrixXd WWbbxb = WWbb[l]* *xb;
    dropper = dropout(VectorXd::Ones(nhb));
    for (uint i=T-1; i!=(uint)(-1); i--) {
      hhb[l].col(i) = (i==T-1) ? f(WWbbxb.col(i) + WWbfxf.col(i))
                               : f(WWbbxb.col(i) + WWbfxf.col(i) +
                                   VVb[l]*hhb[l].col(i+1));
#ifdef DROPOUT
      hhb[l].col(i) = hhb[l].col(i).cwiseProduct(dropper);
#endif
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
      delta_y.col(i) *= OCLASS_WEIGHT;
  }

  // Calculate output layer gradients
  gWWfo[layers-1].noalias() += delta_y * hhf[layers-1].transpose();
  gWWbo[layers-1].noalias() += delta_y * hhb[layers-1].transpose();
  gbo.noalias() += delta_y * VectorXd::Ones(T);

  MatrixXd deltaf[layers];
  MatrixXd deltab[layers];
  for (uint l = 0; l < layers; l++) {
    deltaf[l] = MatrixXd::Zero(nhf, T);
    deltab[l] = MatrixXd::Zero(nhb, T);
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
      // Update error vectors by propagating back from layer above and add supervised error signal (i.e. WW(f/b)o * delta_y)
      deltaf[l].noalias() += fpf.cwiseProduct(WWfo[l].transpose() * delta_y + WWff[l+1].transpose() * deltaf[l+1] + WWbf[l+1].transpose() * deltab[l+1]);
      deltab[l].noalias() += fpb.cwiseProduct(WWbo[l].transpose() * delta_y + WWfb[l+1].transpose() * deltaf[l+1] + WWbb[l+1].transpose() * deltab[l+1]);
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
  MatrixXd deltaf_i = MatrixXd::Zero(nhf, T);
  MatrixXd deltab_i = MatrixXd::Zero(nhb, T);

  MatrixXd fpf = fp(hf);
  MatrixXd fpb = fp(hb);

  deltaf_i.noalias() += fpf.cwiseProduct(Wfo.transpose() * delta_y + WWff[0].transpose() * deltaf[0] + WWbf[0].transpose() * deltab[0]);
  deltab_i.noalias() += fpb.cwiseProduct(Wbo.transpose() * delta_y + WWfb[0].transpose() * deltaf[0] + WWbb[0].transpose() * deltab[0]);

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

RNN::RNN(uint nx, uint nhf, uint nhb, uint ny, LookupTable &LT) {
  lr = ETA;

  this->LT = &LT;

  this->nx = nx;
  this->nhf = nhf;
  this->nhb = nhb;
  this->ny = ny;

  f = &relu;
  fp = &relup;

  // init randomly
  Wf = MatrixXd(nhf,nx).unaryExpr(ptr_fun(urand));
  Vf = MatrixXd(nhf,nhf).unaryExpr(ptr_fun(urand));
  bhf = VectorXd(nhf).unaryExpr(ptr_fun(urand));

  Wb = MatrixXd(nhb,nx).unaryExpr(ptr_fun(urand));
  Vb = MatrixXd(nhb,nhb).unaryExpr(ptr_fun(urand));
  bhb = VectorXd(nhb).unaryExpr(ptr_fun(urand));

  for (uint l=0; l<layers; l++) {
    WWff[l] = MatrixXd(nhf,nhf).unaryExpr(ptr_fun(urand));
    WWfb[l] = MatrixXd(nhf,nhb).unaryExpr(ptr_fun(urand));
    VVf[l] = MatrixXd(nhf,nhf).unaryExpr(ptr_fun(urand));
    bbhf[l] = VectorXd(nhf).unaryExpr(ptr_fun(urand));

    WWbb[l] = MatrixXd(nhb,nhb).unaryExpr(ptr_fun(urand));
    WWbf[l] = MatrixXd(nhb,nhf).unaryExpr(ptr_fun(urand));
    VVb[l] = MatrixXd(nhb,nhb).unaryExpr(ptr_fun(urand));
    bbhb[l] = VectorXd(nhb).unaryExpr(ptr_fun(urand));
  }

  Wfo = MatrixXd(ny,nhf).unaryExpr(ptr_fun(urand));
  Wbo = MatrixXd(ny,nhb).unaryExpr(ptr_fun(urand));
  for (uint l=0; l<layers; l++) {
    WWfo[l] = MatrixXd(ny,nhf).unaryExpr(ptr_fun(urand));
    WWbo[l] = MatrixXd(ny,nhb).unaryExpr(ptr_fun(urand));
  }
  Wo = MatrixXd(ny,nx).unaryExpr(ptr_fun(urand));
  bo = VectorXd(ny).unaryExpr(ptr_fun(urand));

  gWf = MatrixXd::Zero(nhf,nx);
  gVf = MatrixXd::Zero(nhf,nhf);
  gbhf = VectorXd::Zero(nhf);

  gWb = MatrixXd::Zero(nhb,nx);
  gVb = MatrixXd::Zero(nhb,nhb);
  gbhb = VectorXd::Zero(nhb);

  for (uint l=0; l<layers; l++) {
    gWWff[l] = MatrixXd::Zero(nhf,nhf);
    gWWfb[l] = MatrixXd::Zero(nhf,nhb);
    gVVf[l] = MatrixXd::Zero(nhf,nhf);
    gbbhf[l] = VectorXd::Zero(nhf);

    gWWbb[l] = MatrixXd::Zero(nhb,nhb);
    gWWbf[l] = MatrixXd::Zero(nhb,nhf);
    gVVb[l] = MatrixXd::Zero(nhb,nhb);
    gbbhb[l] = VectorXd::Zero(nhb);
  }


  gWfo = MatrixXd::Zero(ny,nhf);
  gWbo = MatrixXd::Zero(ny,nhb);
  for (uint l=0; l<layers; l++) {
    gWWfo[l] = MatrixXd::Zero(ny,nhf);
    gWWbo[l] = MatrixXd::Zero(ny,nhb);
  }
  gWo = MatrixXd::Zero(ny,nx);
  gbo = VectorXd::Zero(ny);

  vWf = MatrixXd::Zero(nhf,nx);
  vVf = MatrixXd::Zero(nhf,nhf);
  vbhf = VectorXd::Zero(nhf);

  vWb = MatrixXd::Zero(nhb,nx);
  vVb = MatrixXd::Zero(nhb,nhb);
  vbhb = VectorXd::Zero(nhb);

  for (uint l=0; l<layers; l++) {
    vWWff[l] = MatrixXd::Zero(nhf,nhf);
    vWWfb[l] = MatrixXd::Zero(nhf,nhb);
    vVVf[l] = MatrixXd::Zero(nhf,nhf);
    vbbhf[l] = VectorXd::Zero(nhf);

    vWWbb[l] = MatrixXd::Zero(nhb,nhb);
    vWWbf[l] = MatrixXd::Zero(nhb,nhf);
    vVVb[l] = MatrixXd::Zero(nhb,nhb);
    vbbhb[l] = VectorXd::Zero(nhb);
  }


  vWfo = MatrixXd::Zero(ny,nhf);
  vWbo = MatrixXd::Zero(ny,nhb);
  for (uint l=0; l<layers; l++) {
    vWWfo[l] = MatrixXd::Zero(ny,nhf);
    vWWbo[l] = MatrixXd::Zero(ny,nhb);
  }
  vWo = MatrixXd::Zero(ny,nx);
  vbo = VectorXd::Zero(ny);
}

void RNN::update() {
  double lambda = LAMBDA;
  double mr = MR;
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

  vWf = lr*gWf/norm + mr*vWf;
  vVf = lr*gVf/norm + mr*vVf;
  vWb = lr*gWb/norm + mr*vWb;
  vVb = lr*gVb/norm + mr*vVb;
  vbhf = lr*gbhf/norm + mr*vbhf;
  vbhb = lr*gbhb/norm + mr*vbhb;

  for (uint l=0; l<layers; l++) {
    vWWff[l] = lr*gWWff[l]/norm + mr*vWWff[l];
    vWWfb[l] = lr*gWWfb[l]/norm + mr*vWWfb[l];
    vVVf[l] = lr*gVVf[l]/norm + mr*vVVf[l];
    vWWbb[l] = lr*gWWbb[l]/norm + mr*vWWbb[l];
    vWWbf[l] = lr*gWWbf[l]/norm + mr*vWWbf[l];
    vVVb[l] = lr*gVVb[l]/norm + mr*vVVb[l];
    vbbhf[l] = lr*gbbhf[l]/norm + mr*vbbhf[l];
    vbbhb[l] = lr*gbbhb[l]/norm + mr*vbbhb[l];
  }

  // update params
  bo.noalias() -= vbo;
  for (uint l=layers-1; l<layers; l++) {
    WWfo[l].noalias() -= vWWfo[l];
    WWbo[l].noalias() -= vWWbo[l];
  }

  Wf.noalias() -= vWf;
  Vf.noalias() -= vVf;
  Wb.noalias() -= vWb;
  Vb.noalias() -= vVb;
  bhf.noalias() -= vbhf;
  bhb.noalias() -= vbhb;

  for (uint l=0; l<layers; l++) {
    WWff[l].noalias() -= vWWff[l];
    WWfb[l].noalias() -= vWWfb[l];
    VVf[l].noalias() -= vVVf[l];
    WWbb[l].noalias() -= vWWbb[l];
    WWbf[l].noalias() -= vWWbf[l];
    VVb[l].noalias() -= vVVb[l];
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

void RNN::load(string fname) {
  ifstream in(fname.c_str());

  in >> nx >> nhf >> nhb >> ny;

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

  out << nx << " " << nhf << " " << nhb << " " << ny << endl;

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

#ifdef DROPOUT
Matrix<double, -1, 1> dropout(Matrix<double, -1, 1> x, double p) {
  for (uint i=0; i<x.size(); i++) {
    if ((double)rand()/RAND_MAX < p)
      x(i) = 0;
  }
  return x;
}
#endif

string RNN::model_name() {
  ostringstream strS;
  strS << "drnt_" << layers << "_" << nhf << "_"
  << nhb << "_" << DROP << "_"
  << lr << "_" << LAMBDA << "_"
  << MR << "_" << fold;
  string fname = strS.str();
  return fname;
}

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

  vector<vector<string> > trainX, validX, testX;
  vector<vector<string> > trainL, validL, testL;

  DataUtils::generate_splits(X, T, trainX, trainL, validX, validL, testX, testL, 0.8, 0.1, 0.1);

  cout << "Total dataset size: " << X.size() << endl;
  cout << "Training set size: " << trainX.size() << endl;
  cout << "Validation set size: " << validX.size() << endl;
  cout << "Test set size: " << testX.size() << endl;

  Matrix<double, 6, 2> best = Matrix<double, 6, 2>::Zero();
  double bestDrop;
  for (DROP=0; DROP<0.1; DROP+=0.2) { // can use this loop for CV
    RNN brnn(25,25,25,ny,LT);
    auto results = brnn.train(trainX, trainL, validX, validL, testX, testL, 200, 80);
    if (best(2,0) < results(2,0)) { // propF1 on val set
      best = results;
      bestDrop = DROP;
    }
    brnn.save("models/" + brnn.model_name());
  }
  cout << "Best: " << endl;
  cout << "Drop: " << bestDrop << endl;
  cout << best << endl;

  return 0;
}

