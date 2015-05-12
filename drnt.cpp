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
#include "model.h"

#define DROPOUT
#define ETA 0.001
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
  MatrixXd x = MatrixXd(nx, T);

  for (uint i=0; i<T; i++)
    x.col(i) = (*LT)[sent[i]];

  MatrixXd dhb = MatrixXd::Zero(nhb, T);
  MatrixXd dhf = MatrixXd::Zero(nhf, T);

  MatrixXd dhhf[layers], dhhb[layers];
  for (uint l=0; l<layers; l++) {
    dhhf[l] = MatrixXd::Zero(nhf, T);
    dhhb[l] = MatrixXd::Zero(nhb, T);
  }

  // Create one-hot vectors
  MatrixXd yi = MatrixXd::Zero(ny, T);
  for (uint i=0; i<T; i++) {
    int label_idx = stoi(labels[i]);
    yi(label_idx, i) = 1;
  }

  MatrixXd y = forward(sent);

  MatrixXd gpyd = smaxentp(y,yi);
  for (uint i=0; i<T; i++)
    if (labels[i] == "0")
      gpyd.col(i) *= OCLASS_WEIGHT;

  for (uint l=layers-1; l<layers; l++) {
    gWWfo[l].noalias() += gpyd * hhf[l].transpose();
    gWWbo[l].noalias() += gpyd * hhb[l].transpose();
  }
  gbo.noalias() += gpyd*VectorXd::Ones(T);

  dhf.noalias() += Wfo.transpose() * gpyd;
  dhb.noalias() += Wbo.transpose() * gpyd;
  for (uint l=0; l<layers; l++) {
    dhhf[l].noalias() += WWfo[l].transpose() * gpyd;
    dhhb[l].noalias() += WWbo[l].transpose() * gpyd;
  }

  // activation regularize
  dhf.noalias() += LAMBDAH*hf;
  dhb.noalias() += LAMBDAH*hb;
  for (uint l=0; l<layers; l++) {
    dhhf[l].noalias() += LAMBDAH*hhf[l];
    dhhb[l].noalias() += LAMBDAH*hhb[l];
  }

  for (uint l=layers-1; l != (uint)(-1); l--) {
    MatrixXd *dxf, *dxb, *xf, *xb;
    dxf = (l == 0) ? &dhf : &(dhhf[l-1]);
    dxb = (l == 0) ? &dhb : &(dhhb[l-1]);
    xf = (l == 0) ? &hf : &(hhf[l-1]);
    xb = (l == 0) ? &hb : &(hhb[l-1]);

    MatrixXd fphdh = MatrixXd::Zero(nhf,T);
    for (uint i=T-1; i != (uint)(-1); i--) {
      // i^th column of drop across ReLU unit is relu(hf_i^l) \cdot deltaf_i^{L}
      fphdh.col(i) = fp(hhf[l].col(i)).cwiseProduct(dhhf[l].col(i));
      if (i > 0) {
        // dVf_l += deltaf_i^l * hf_(i-1)^l.T
        gVVf[l].noalias() += fphdh.col(i) * hhf[l].col(i-1).transpose();
        dhhf[l].col(i-1).noalias() += VVf[l].transpose() * fphdh.col(i);
      }
    }
    gWWff[l].noalias() += fphdh * xf->transpose();
    gWWfb[l].noalias() += fphdh * xb->transpose();
    gbbhf[l].noalias() += fphdh * VectorXd::Ones(T);
    dxf->noalias() += WWff[l].transpose() * fphdh;
    dxb->noalias() += WWfb[l].transpose() * fphdh;

    fphdh = MatrixXd::Zero(nhb,T);
    for (uint i=0; i < T; i++) {
      fphdh.col(i) = fp(hhb[l].col(i)).cwiseProduct(dhhb[l].col(i));
      if (i < T-1) {
        // dh_{t+1}^l_back += V^l_back.T * relu(
        dhhb[l].col(i+1).noalias() += VVb[l].transpose() * fphdh.col(i);
        gVVb[l].noalias() += fphdh.col(i) * hhb[l].col(i+1).transpose();
      }
    }
    gWWbb[l].noalias() += fphdh * xb->transpose();
    gWWbf[l].noalias() += fphdh * xf->transpose();
    gbbhb[l].noalias() += fphdh * VectorXd::Ones(T);
    dxf->noalias() += WWbf[l].transpose() * fphdh;
    dxb->noalias() += WWbb[l].transpose() * fphdh;
  }

  for (uint i=T-1; i != 0; i--) {
    VectorXd fphdh = fp(hf.col(i)).cwiseProduct(dhf.col(i));
    gWf.noalias() += fphdh * x.col(i).transpose();
    gVf.noalias() += fphdh * hf.col(i-1).transpose();
    gbhf.noalias() += fphdh;
    dhf.col(i-1).noalias() += Vf.transpose() * fphdh;
  }
  VectorXd fphdh = fp(hf.col(0)).cwiseProduct(dhf.col(0));
  gWf.noalias() += fphdh * x.col(0).transpose();
  gbhf.noalias() += fphdh;

  for (uint i=0; i < T-1; i++) {
    VectorXd fphdh = fp(hb.col(i)).cwiseProduct(dhb.col(i));
    gWb.noalias() += fphdh * x.col(i).transpose();
    gVb.noalias() += fphdh * hb.col(i+1).transpose();
    gbhb.noalias() += fphdh;
    dhb.col(i+1).noalias() += Vb.transpose() * fphdh;
  }
  fphdh = fp(hb.col(T-1)).cwiseProduct(dhb.col(T-1));
  gWb.noalias() += fphdh * x.col(T-1).transpose();
  gbhb.noalias() += fphdh;

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
  strS << "models/drnt_" << layers << "_" << nhf << "_"
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
    RNN brnn(25,25,25,ny,LT);
    auto results = brnn.train(trainX, trainL, validX, validL, testX, testL, 200, 80);
    if (best(2,0) < results(2,0)) { // propF1 on val set
      best = results;
      bestDrop = DROP;
    }
    brnn.save("model.txt");
  }
  cout << "Best: " << endl;
  cout << "Drop: " << bestDrop << endl;
  cout << best << endl;

  return 0;
}

