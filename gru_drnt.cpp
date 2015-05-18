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

#define ERROR_SIGNAL
#define NORMALIZE false // keeping this false throughout my own experiments
#define layers 3 // number of EXTRA (not all) hidden layers

using namespace Eigen;
using namespace std;

double LAMBDA = (layers > 2) ? 1e-5 : 1e-4;  // L2 regularizer on weights
//double LAMBDAH = (layers > 2) ? 1e-5 : 1e-4; //L2 regularizer on activations
double LAMBDAH = 0; //L2 regularizer on activations

Matrix<double, -1, 1> dropout(Matrix<double, -1, 1> x, double p);

class GRURNN : public Model {
public:
  GRURNN(uint nx, uint nh, uint ny, LookupTable &LT, float lr, float mr, float null_class_weight, float dropout = 0.0);

  void save(string fname);
  void load(string fname);
  MatrixXd forward(const vector<string> &sent);
  double backward(const vector<string> &sent, 
                  const vector<string> &labels);
  void update();
  bool is_nan();
  string model_name();
  void grad_check(vector<string> &sentence, vector<string> &labels);
  MatrixXd numerical_gradient(MatrixXd &parameter, vector<string> &sentence, vector<string> &labels);
  double cost(const vector<string> &sent, const vector<string> &labels);

  LookupTable *LT;

private:
  MatrixXd (*f)(const MatrixXd& x);
  MatrixXd (*fp)(const MatrixXd& x);
  MatrixXd (*f2)(const MatrixXd& x);
  MatrixXd (*f2p)(const MatrixXd& x);

  MatrixXd hf, hb, hhf[layers], hhb[layers], // Activations
           htf, htb, hhtf[layers], hhtb[layers], // h tilde activations
           zf, zb, zzf[layers], zzb[layers], // Update gate activations
           rf, rb, rrf[layers], rrb[layers]; // Reset gate activations

  // recurrent network params
  // WW(f/b)o is forward/backward matrix if output layer
  MatrixXd Wo, Wfo, Wbo, WWfo[layers], WWbo[layers];
  VectorXd bo;

  // Input layer parameters
  MatrixXd Wf, Vf, Wb, Vb;
  VectorXd bhf, bhb;

  MatrixXd Wrf, Vrf, Wrb, Vrb;
  VectorXd brhf, brhb;

  MatrixXd Wzf, Vzf, Wzb, Vzb;
  VectorXd bzhf, bzhb;

  // Hidden layer parameters
  MatrixXd WWff[layers], WWfb[layers], WWbb[layers], WWbf[layers];
  MatrixXd VVf[layers], VVb[layers];
  VectorXd bbhf[layers], bbhb[layers];

  // Reset gate parameters
  MatrixXd WWrff[layers], WWrfb[layers], WWrbb[layers], WWrbf[layers];
  MatrixXd VVrf[layers], VVrb[layers];
  VectorXd bbrhf[layers], bbrhb[layers];

  // Update gate parameters
  MatrixXd WWzff[layers], WWzfb[layers], WWzbb[layers], WWzbf[layers];
  MatrixXd VVzf[layers], VVzb[layers];
  VectorXd bbzhf[layers], bbzhb[layers];

  // Gradients
  MatrixXd gWo, gWfo, gWbo, gWWfo[layers], gWWbo[layers];
  VectorXd gbo;

  // Input layer gradients
  MatrixXd gWf, gVf, gWb, gVb;
  VectorXd gbhf, gbhb;

  MatrixXd gWrf, gVrf, gWrb, gVrb;
  VectorXd gbrhf, gbrhb;

  MatrixXd gWzf, gVzf, gWzb, gVzb;
  VectorXd gbzhf, gbzhb;

  // Hidden layer gradients
  MatrixXd gWWff[layers], gWWfb[layers], gWWbb[layers], gWWbf[layers];
  MatrixXd gVVf[layers],  gVVb[layers];
  VectorXd gbbhf[layers], gbbhb[layers];

  // Reset gate gradients
  MatrixXd gWWrff[layers], gWWrfb[layers], gWWrbb[layers], gWWrbf[layers];
  MatrixXd gVVrf[layers],  gVVrb[layers];
  VectorXd gbbrhf[layers], gbbrhb[layers];

  // Update gate gradients
  MatrixXd gWWzff[layers], gWWzfb[layers], gWWzbb[layers], gWWzbf[layers];
  MatrixXd gVVzf[layers],  gVVzb[layers];
  VectorXd gbbzhf[layers], gbbzhb[layers];

  // Velocities
  MatrixXd vWo, vWfo, vWbo, vWWfo[layers], vWWbo[layers];
  VectorXd vbo;

  // Input layer velocities
  MatrixXd vWf, vVf, vWb, vVb;
  VectorXd vbhf, vbhb;

  MatrixXd vWrf, vVrf, vWrb, vVrb;
  VectorXd vbrhf, vbrhb;

  MatrixXd vWzf, vVzf, vWzb, vVzb;
  VectorXd vbzhf, vbzhb;

  // Hidden layer velocities
  MatrixXd vWWff[layers], vWWfb[layers], vWWbb[layers], vWWbf[layers];
  MatrixXd vVVf[layers],  vVVb[layers];
  VectorXd vbbhf[layers], vbbhb[layers];

  // Reset gate velocities
  MatrixXd vWWrff[layers], vWWrfb[layers], vWWrbb[layers], vWWrbf[layers];
  MatrixXd vVVrf[layers],  vVVrb[layers];
  VectorXd vbbrhf[layers], vbbrhb[layers];

  // Update gate velocities
  MatrixXd vWWzff[layers], vWWzfb[layers], vWWzbb[layers], vWWzbf[layers];
  MatrixXd vVVzf[layers],  vVVzb[layers];
  VectorXd vbbzhf[layers], vbbzhb[layers];

  uint nx, nh, ny;
  uint epoch;

  float lr, mr, null_class_weight, dropout_prob;
};

GRURNN::GRURNN(uint nx, uint nh, uint ny, LookupTable &LT, float lr, float mr, float null_class_weight, float dropout) :
  LT(&LT), nx(nx), nh(nh), ny(ny), lr(lr), mr(mr), null_class_weight(null_class_weight), dropout_prob(dropout)
{
  f = &relu;
  fp = &relup;

  f2 = &sigmoid;
  f2p = &sigmoidp;

  // init randomly
  Wf = MatrixXd(nh,nx).unaryExpr(ptr_fun(urand));
  Vf = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
  bhf = VectorXd(nh).unaryExpr(ptr_fun(urand));

  Wb = MatrixXd(nh,nx).unaryExpr(ptr_fun(urand));
  Vb = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
  bhb = VectorXd(nh).unaryExpr(ptr_fun(urand));

  Wrf = MatrixXd(nh,nx).unaryExpr(ptr_fun(urand));
  Vrf = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
  brhf = VectorXd(nh).unaryExpr(ptr_fun(urand));

  Wrb = MatrixXd(nh,nx).unaryExpr(ptr_fun(urand));
  Vrb = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
  brhb = VectorXd(nh).unaryExpr(ptr_fun(urand));

  Wzf = MatrixXd(nh,nx).unaryExpr(ptr_fun(urand));
  Vzf = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
  bzhf = VectorXd(nh).unaryExpr(ptr_fun(urand));

  Wzb = MatrixXd(nh,nx).unaryExpr(ptr_fun(urand));
  Vzb = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
  bzhb = VectorXd(nh).unaryExpr(ptr_fun(urand));

  for (uint l=0; l<layers; l++) {
    WWff[l] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
    WWfb[l] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
    VVf[l] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
    bbhf[l] = VectorXd(nh).unaryExpr(ptr_fun(urand));

    WWbb[l] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
    WWbf[l] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
    VVb[l] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
    bbhb[l] = VectorXd(nh).unaryExpr(ptr_fun(urand));

    WWrff[l] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
    WWrfb[l] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
    VVrf[l] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
    bbrhf[l] = VectorXd(nh).unaryExpr(ptr_fun(urand));

    WWrbb[l] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
    WWrbf[l] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
    VVrb[l] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
    bbrhb[l] = VectorXd(nh).unaryExpr(ptr_fun(urand));

    WWzff[l] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
    WWzfb[l] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
    VVzf[l] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
    bbzhf[l] = VectorXd(nh).unaryExpr(ptr_fun(urand));

    WWzbb[l] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
    WWzbf[l] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
    VVzb[l] = MatrixXd(nh,nh).unaryExpr(ptr_fun(urand));
    bbzhb[l] = VectorXd(nh).unaryExpr(ptr_fun(urand));
  }

  Wfo = MatrixXd(ny,nh).unaryExpr(ptr_fun(urand));
  Wbo = MatrixXd(ny,nh).unaryExpr(ptr_fun(urand));
  for (uint l=0; l<layers; l++) {
    WWfo[l] = MatrixXd(ny,nh).unaryExpr(ptr_fun(urand));
    WWbo[l] = MatrixXd(ny,nh).unaryExpr(ptr_fun(urand));
  }
  Wo = MatrixXd(ny,nx).unaryExpr(ptr_fun(urand));
  bo = VectorXd(ny).unaryExpr(ptr_fun(urand));

  // Initialize gradients to zero
  gWf = MatrixXd::Zero(nh,nx);
  gVf = MatrixXd::Zero(nh,nh);
  gbhf = VectorXd::Zero(nh);

  gWb = MatrixXd::Zero(nh,nx);
  gVb = MatrixXd::Zero(nh,nh);
  gbhb = VectorXd::Zero(nh);

  // Reset gate gradients for input layer
  gWrf = MatrixXd::Zero(nh,nx);
  gVrf = MatrixXd::Zero(nh,nh);
  gbrhf = VectorXd::Zero(nh);

  gWrb = MatrixXd::Zero(nh,nx);
  gVrb = MatrixXd::Zero(nh,nh);
  gbrhb = VectorXd::Zero(nh);

  // Update gate gradients for input layer
  gWzf = MatrixXd::Zero(nh,nx);
  gVzf = MatrixXd::Zero(nh,nh);
  gbzhf = VectorXd::Zero(nh);

  gWzb = MatrixXd::Zero(nh,nx);
  gVzb = MatrixXd::Zero(nh,nh);
  gbzhb = VectorXd::Zero(nh);

  // Gradients for hidden layer
  for (uint l=0; l<layers; l++) {
    // Output gate gradients
    gWWff[l] = MatrixXd::Zero(nh,nh);
    gWWfb[l] = MatrixXd::Zero(nh,nh);
    gVVf[l] = MatrixXd::Zero(nh,nh);
    gbbhf[l] = VectorXd::Zero(nh);

    gWWbb[l] = MatrixXd::Zero(nh,nh);
    gWWbf[l] = MatrixXd::Zero(nh,nh);
    gVVb[l] = MatrixXd::Zero(nh,nh);
    gbbhb[l] = VectorXd::Zero(nh);

    // Reset gate gradients
    gWWrff[l] = MatrixXd::Zero(nh,nh);
    gWWrfb[l] = MatrixXd::Zero(nh,nh);
    gVVrf[l] = MatrixXd::Zero(nh,nh);
    gbbrhf[l] = VectorXd::Zero(nh);

    gWWrbb[l] = MatrixXd::Zero(nh,nh);
    gWWrbf[l] = MatrixXd::Zero(nh,nh);
    gVVrb[l] = MatrixXd::Zero(nh,nh);
    gbbrhb[l] = VectorXd::Zero(nh);

    // Update gate gradients
    gWWzff[l] = MatrixXd::Zero(nh,nh);
    gWWzfb[l] = MatrixXd::Zero(nh,nh);
    gVVzf[l] = MatrixXd::Zero(nh,nh);
    gbbzhf[l] = VectorXd::Zero(nh);

    gWWzbb[l] = MatrixXd::Zero(nh,nh);
    gWWzbf[l] = MatrixXd::Zero(nh,nh);
    gVVzb[l] = MatrixXd::Zero(nh,nh);
    gbbzhb[l] = VectorXd::Zero(nh);
  }

  gWfo = MatrixXd::Zero(ny,nh);
  gWbo = MatrixXd::Zero(ny,nh);
  for (uint l=0; l<layers; l++) {
    gWWfo[l] = MatrixXd::Zero(ny,nh);
    gWWbo[l] = MatrixXd::Zero(ny,nh);
  }
  gWo = MatrixXd::Zero(ny,nx);
  gbo = VectorXd::Zero(ny);

  // Initialize velocities to zero
  vWf = MatrixXd::Zero(nh,nx);
  vVf = MatrixXd::Zero(nh,nh);
  vbhf = VectorXd::Zero(nh);

  vWb = MatrixXd::Zero(nh,nx);
  vVb = MatrixXd::Zero(nh,nh);
  vbhb = VectorXd::Zero(nh);

  // Reset gate velocities for input layer
  vWrf = MatrixXd::Zero(nh,nx);
  vVrf = MatrixXd::Zero(nh,nh);
  vbrhf = VectorXd::Zero(nh);

  vWrb = MatrixXd::Zero(nh,nx);
  vVrb = MatrixXd::Zero(nh,nh);
  vbrhb = VectorXd::Zero(nh);

  // Update gate velocities for input layer
  vWzf = MatrixXd::Zero(nh,nx);
  vVzf = MatrixXd::Zero(nh,nh);
  vbzhf = VectorXd::Zero(nh);

  vWzb = MatrixXd::Zero(nh,nx);
  vVzb = MatrixXd::Zero(nh,nh);
  vbzhb = VectorXd::Zero(nh);

  // Velocities for hidden layer
  for (uint l=0; l<layers; l++) {
    // Output gate velocities
    vWWff[l] = MatrixXd::Zero(nh,nh);
    vWWfb[l] = MatrixXd::Zero(nh,nh);
    vVVf[l] = MatrixXd::Zero(nh,nh);
    vbbhf[l] = VectorXd::Zero(nh);

    vWWbb[l] = MatrixXd::Zero(nh,nh);
    vWWbf[l] = MatrixXd::Zero(nh,nh);
    vVVb[l] = MatrixXd::Zero(nh,nh);
    vbbhb[l] = VectorXd::Zero(nh);

    // Reset gate velocities
    vWWrff[l] = MatrixXd::Zero(nh,nh);
    vWWrfb[l] = MatrixXd::Zero(nh,nh);
    vVVrf[l] = MatrixXd::Zero(nh,nh);
    vbbrhf[l] = VectorXd::Zero(nh);

    vWWrbb[l] = MatrixXd::Zero(nh,nh);
    vWWrbf[l] = MatrixXd::Zero(nh,nh);
    vVVrb[l] = MatrixXd::Zero(nh,nh);
    vbbrhb[l] = VectorXd::Zero(nh);

    // Update gate velocities
    vWWzff[l] = MatrixXd::Zero(nh,nh);
    vWWzfb[l] = MatrixXd::Zero(nh,nh);
    vVVzf[l] = MatrixXd::Zero(nh,nh);
    vbbzhf[l] = VectorXd::Zero(nh);

    vWWzbb[l] = MatrixXd::Zero(nh,nh);
    vWWzbf[l] = MatrixXd::Zero(nh,nh);
    vVVzb[l] = MatrixXd::Zero(nh,nh);
    vbbzhb[l] = VectorXd::Zero(nh);
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

MatrixXd GRURNN::forward(const vector<string> &sent) {
  VectorXd dropper;
  uint T = sent.size();
  MatrixXd x = MatrixXd(nx, T);

  for (uint i=0; i<T; i++)
    x.col(i) = (*LT)[sent[i]];

  hf = MatrixXd::Zero(nh, T);
  hb = MatrixXd::Zero(nh, T);
  htf = MatrixXd::Zero(nh, T);
  htb = MatrixXd::Zero(nh, T);
  rf = MatrixXd::Zero(nh, T);
  rb = MatrixXd::Zero(nh, T);
  zf = MatrixXd::Zero(nh, T);
  zb = MatrixXd::Zero(nh, T);

  for (uint l=0; l<layers; l++) {
    hhf[l] = MatrixXd::Zero(nh, T);
    hhb[l] = MatrixXd::Zero(nh, T);
    hhtf[l] = MatrixXd::Zero(nh, T);
    hhtb[l] = MatrixXd::Zero(nh, T);
    rrf[l] = MatrixXd::Zero(nh, T);
    rrb[l] = MatrixXd::Zero(nh, T);
    zzf[l] = MatrixXd::Zero(nh, T);
    zzb[l] = MatrixXd::Zero(nh, T);
  }

  // Forward units at the input layer
  MatrixXd Wzfx = Wzf * x + bzhf * RowVectorXd::Ones(T);
  MatrixXd Wrfx = Wrf * x + brhf * RowVectorXd::Ones(T);
  MatrixXd Wfx = Wf * x + bhf * RowVectorXd::Ones(T);
  dropper = dropout(VectorXd::Ones(nh), dropout_prob);
  for (uint t = 0; t < T; t++) {
    if (t == 0) {
      rf.col(t) = f2(Wrfx.col(t));
      zf.col(t) = f2(Wzfx.col(t));
      htf.col(t) = f(Wfx.col(t));
      hf.col(t) = (VectorXd::Ones(nh) - zf.col(t)).cwiseProduct(htf.col(t));
    } else {
      rf.col(t) = f2(Wrfx.col(t) + Vrf * hf.col(t-1));
      zf.col(t) = f2(Wzfx.col(t) + Vzf * hf.col(t-1));
      htf.col(t) = f(Wfx.col(t) + rf.col(t).cwiseProduct(Vf * hf.col(t-1)));
      hf.col(t) = zf.col(t).cwiseProduct(hf.col(t-1)) + (VectorXd::Ones(nh) - zf.col(t)).cwiseProduct(htf.col(t));
    }
    //hf.col(t) = hf.col(t).cwiseProduct(dropper);
  }

  // Backward units at the input layer
  MatrixXd Wzbx = Wzb * x + bzhb * RowVectorXd::Ones(T);
  MatrixXd Wrbx = Wrb * x + brhb * RowVectorXd::Ones(T);
  MatrixXd Wbx = Wb * x + bhb * RowVectorXd::Ones(T);
  dropper = dropout(VectorXd::Ones(nh), dropout_prob);
  for (uint t = T-1; t != (uint)(-1); t--) {
    if (t == T-1) {
      rb.col(t) = f2(Wrbx.col(t));
      zb.col(t) = f2(Wzbx.col(t));
      htb.col(t) = f(Wbx.col(t));
      hb.col(t) = (VectorXd::Ones(nh) - zb.col(t)).cwiseProduct(htb.col(t));
    } else {
      rb.col(t) = f2(Wrbx.col(t) + Vrb * hb.col(t+1));
      zb.col(t) = f2(Wzbx.col(t) + Vzb * hb.col(t+1));
      htb.col(t) = f(Wbx.col(t) + rb.col(t).cwiseProduct(Vb * hb.col(t+1)));
      hb.col(t) = zb.col(t).cwiseProduct(hb.col(t+1)) + (VectorXd::Ones(nh) - zb.col(t)).cwiseProduct(htb.col(t));
    }
    //hb.col(t) = hb.col(t).cwiseProduct(dropper);
  }

  for (uint l = 0; l < layers; l++) {
    MatrixXd *xf, *xb; // input to this layer (not to all network)
    xf = (l == 0) ? &hf : &(hhf[l-1]);
    xb = (l == 0) ? &hb : &(hhb[l-1]);

    MatrixXd WWffxf = WWff[l]* *xf + bbhf[l]*RowVectorXd::Ones(T);
    MatrixXd WWfbxb = WWfb[l]* *xb;

    MatrixXd WWrffxf = WWrff[l]* *xf + bbrhf[l]*RowVectorXd::Ones(T);
    MatrixXd WWrfbxb = WWrfb[l]* *xb;

    MatrixXd WWzffxf = WWzff[l]* *xf + bbzhf[l]*RowVectorXd::Ones(T);
    MatrixXd WWzfbxb = WWzfb[l]* *xb;

    dropper = dropout(VectorXd::Ones(nh), dropout_prob);
    for (uint t = 0; t < T; t++) {
      if (t == 0) {
        rrf[l].col(t)  = f2(WWrffxf.col(t) + WWrfbxb.col(t));
        zzf[l].col(t)  = f2(WWzffxf.col(t) + WWzfbxb.col(t));
        hhtf[l].col(t) = f(WWffxf.col(t) + WWfbxb.col(t));
        hhf[l].col(t)  = (VectorXd::Ones(nh) - zzf[l].col(t)).cwiseProduct(hhtf[l].col(t));
      } else {
        rrf[l].col(t)  = f2(WWrffxf.col(t) + WWrfbxb.col(t) + VVrf[l]*hhf[l].col(t-1));
        zzf[l].col(t)  = f2(WWzffxf.col(t) + WWzfbxb.col(t) + VVzf[l]*hhf[l].col(t-1));
        hhtf[l].col(t) = f(WWffxf.col(t) + WWfbxb.col(t) + rrf[l].col(t).cwiseProduct(VVf[l]*hhf[l].col(t-1)));
        hhf[l].col(t)  = zzf[l].col(t).cwiseProduct(hhf[l].col(t-1)) + (VectorXd::Ones(nh) - zzf[l].col(t)).cwiseProduct(hhtf[l].col(t));
      }
      //hhf[l].col(t) = hhf[l].col(t).cwiseProduct(dropper);
    }

    MatrixXd WWbfxf = WWbf[l]* *xf + bbhb[l]*RowVectorXd::Ones(T);
    MatrixXd WWbbxb = WWbb[l]* *xb;

    MatrixXd WWrbfxf = WWrbf[l]* *xf + bbrhb[l]*RowVectorXd::Ones(T);
    MatrixXd WWrbbxb = WWrbb[l]* *xb;

    MatrixXd WWzbfxf = WWzbf[l]* *xf + bbzhb[l]*RowVectorXd::Ones(T);
    MatrixXd WWzbbxb = WWzbb[l]* *xb;

    dropper = dropout(VectorXd::Ones(nh), dropout_prob);
    for (uint t = T-1; t != (uint)(-1); t--) {
      if (t == T-1) {
        rrb[l].col(t)  = f2(WWrbfxf.col(t) + WWrbbxb.col(t));
        zzb[l].col(t)  = f2(WWzbfxf.col(t) + WWzbbxb.col(t));
        hhtb[l].col(t) = f(WWbfxf.col(t) + WWbbxb.col(t));
        hhb[l].col(t)  = (VectorXd::Ones(nh) - zzb[l].col(t)).cwiseProduct(hhtb[l].col(t));
      } else {
        rrb[l].col(t)  = f2(WWrbfxf.col(t) + WWrbbxb.col(t) + VVrb[l]*hhb[l].col(t+1));
        zzb[l].col(t)  = f2(WWzbfxf.col(t) + WWzbbxb.col(t) + VVzb[l]*hhb[l].col(t+1));
        hhtb[l].col(t) = f(WWbfxf.col(t) + WWbbxb.col(t) + rrb[l].col(t).cwiseProduct(VVb[l]*hhb[l].col(t+1)));
        hhb[l].col(t)  = zzb[l].col(t).cwiseProduct(hhb[l].col(t+1)) + (VectorXd::Ones(nh) - zzb[l].col(t)).cwiseProduct(hhtb[l].col(t));
      }
      //hhb[l].col(t) = hhb[l].col(t).cwiseProduct(dropper);
    }
  }

  // output layer uses the last hidden layer
  // you can experiment with the other version by changing this
  // (backward pass needs to change as well of course)
  return softmax(bo*RowVectorXd::Ones(T) + WWfo[layers-1]*hhf[layers-1] +
              WWbo[layers-1]*hhb[layers-1]);
}

double GRURNN::cost(const vector<string> &sent, const vector<string> &labels) {
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

  //cost = -(y.transpose() * MatrixXd(y_hat.array().log())).sum();
  return -cost;
}

double GRURNN::backward(const vector<string> &sent, const vector<string> &labels) {
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
  }
  
  //cout << "Finished forward pass" << endl;

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

  MatrixXd deltaf[layers+1];
  MatrixXd deltab[layers+1];
  for (uint l = 0; l < layers; l++) {
    deltaf[l] = MatrixXd::Zero(nh, T);
    deltab[l] = MatrixXd::Zero(nh, T);
  }
  deltaf[layers] = WWfo[layers-1].transpose() * delta_y;
  deltab[layers] = WWbo[layers-1].transpose() * delta_y;
  // Create error vectors propagated by hidden units
  // Note that ReLU'(x) = ReLU'(ReLU(x)). In general, we assume that for the
  // hidden unit non-linearity f, f(z) = f'(f(z))
  for (uint l = layers-1; l != (uint)(-1); l--) {
    MatrixXd cur_hf = (l == 0) ? hf : hhf[l-1];
    MatrixXd cur_hb = (l == 0) ? hb : hhb[l-1];

    MatrixXd dhhtfdrrf = MatrixXd::Zero(nh, T);
    MatrixXd dhhfdzzf = MatrixXd::Zero(nh, T);

    // Calculate dhf_{t+1}^{(i)} / dhf_t^{(i)}
    MatrixXd VVzf_f2pzzf = VVzf[l].transpose() * f2p(zzf[l]);
    MatrixXd VVrf_f2prrf = VVrf[l].transpose() * f2p(rrf[l]);
    MatrixXd VVf_fphhtf = VVf[l].transpose() * fp(hhtf[l]);
    MatrixXd VVf_hhf = VVf[l] * hhf[l];
    MatrixXd dhhfdhhtf = (MatrixXd::Ones(nh,T) - zzf[l]);
    for (uint t = T-2; t != (uint)(-1); t--) {
      VectorXd dhft1 = VectorXd::Zero(nh);
      dhft1 += zzf[l].col(t+1);
      dhft1 += VVzf_f2pzzf.col(t+1).cwiseProduct(hhf[l].col(t) - hhtf[l].col(t+1));
      dhft1 += dhhfdhhtf.col(t+1).cwiseProduct(rrf[l].col(t+1).cwiseProduct(VVf_fphhtf.col(t+1))
                                                   + VVrf_f2prrf.col(t+1).cwiseProduct(VVf_hhf.col(t)).cwiseProduct(fp(hhtf[l].col(t+1))));
      deltaf[l+1].col(t) += dhft1.cwiseProduct(deltaf[l+1].col(t+1));
      dhhfdzzf.col(t+1) = (hhf[l].col(t) - hhtf[l].col(t+1));
      dhhtfdrrf.col(t+1) = fp(hhtf[l].col(t+1)).cwiseProduct(VVf[l]*hhf[l].col(t));
    }
    MatrixXd dJdhhtf = dhhfdhhtf.cwiseProduct(deltaf[l+1]); // verified by grad check
    MatrixXd dJdzzf = dhhfdzzf.cwiseProduct(deltaf[l+1]); // verified by grad check
    MatrixXd dJdrrf = dhhtfdrrf.cwiseProduct(dJdhhtf); // verified by grad check

    // Update local gradients TODO: update V matrices
    gWWff[l].noalias() += fp(hhtf[l]).cwiseProduct(dJdhhtf) * cur_hf.transpose();
    gWWfb[l].noalias() += fp(hhtf[l]).cwiseProduct(dJdhhtf) * cur_hb.transpose();
    gbbhf[l].noalias() += fp(hhtf[l]).cwiseProduct(dJdhhtf) * VectorXd::Ones(T);

    gWWzff[l].noalias() += f2p(zzf[l]).cwiseProduct(dJdzzf) * cur_hf.transpose();
    gWWzfb[l].noalias() += f2p(zzf[l]).cwiseProduct(dJdzzf) * cur_hb.transpose();
    gbbzhf[l].noalias() += f2p(zzf[l]).cwiseProduct(dJdzzf) * VectorXd::Ones(T);

    gWWrff[l].noalias() += f2p(rrf[l]).cwiseProduct(dJdrrf) * cur_hf.transpose();
    gWWrfb[l].noalias() += f2p(rrf[l]).cwiseProduct(dJdrrf) * cur_hb.transpose();
    gbbrhf[l].noalias() += f2p(rrf[l]).cwiseProduct(dJdrrf) * VectorXd::Ones(T);

    for (uint t = 1; t < T; t++) {
      gVVf[l].noalias() += rrf[l].col(t).cwiseProduct(dJdhhtf.col(t)) * hhf[l].col(t-1).transpose();
      gVVrf[l].noalias() += dJdrrf.col(t) * hhf[l].col(t-1).transpose();
      gVVzf[l].noalias() += dJdzzf.col(t) * hhf[l].col(t-1).transpose();
    }

    // Propagate error downwards
    deltaf[l].noalias() += WWzff[l].transpose() * dJdzzf;
    deltaf[l].noalias() += WWrff[l].transpose() * dJdrrf;
    deltaf[l].noalias() += WWff[l].transpose() * dJdhhtf;

    deltab[l].noalias() += WWzfb[l].transpose() * dJdzzf;
    deltab[l].noalias() += WWrfb[l].transpose() * dJdrrf;
    deltab[l].noalias() += WWfb[l].transpose() * dJdhhtf;

    MatrixXd dhhtbdrrb = MatrixXd::Zero(nh, T);
    MatrixXd dhhbdzzb = MatrixXd::Zero(nh, T);

    // Calculate dhf_{t+1}^{(i)} / dhf_t^{(i)}
    MatrixXd VVzb_f2pzzb = VVzb[l].transpose() * f2p(zzb[l]);
    MatrixXd VVrb_f2prrb = VVrb[l].transpose() * f2p(rrb[l]);
    MatrixXd VVb_fphhtb = VVb[l].transpose() * fp(hhtb[l]);
    MatrixXd VVb_hhb = VVb[l] * hhb[l];
    MatrixXd dhhbdhhtb = (MatrixXd::Ones(nh,T) - zzb[l]);
    for (uint t = 1; t < T; t++) {
      VectorXd dhft1 = VectorXd::Zero(nh);
      dhft1 += zzb[l].col(t-1);
      dhft1 += VVzb_f2pzzb.col(t-1).cwiseProduct(hhb[l].col(t) - hhtf[l].col(t-1));
      dhft1 += dhhbdhhtb.col(t-1).cwiseProduct(rrb[l].col(t-1).cwiseProduct(VVb_fphhtb.col(t-1))
                                                   + VVrb_f2prrb.col(t-1).cwiseProduct(VVb_hhb.col(t)).cwiseProduct(fp(hhtb[l].col(t-1))));
      deltab[l+1].col(t) += dhft1.cwiseProduct(deltab[l+1].col(t-1));
      dhhbdzzb.col(t-1) = (hhb[l].col(t) - hhtb[l].col(t-1));
      dhhtbdrrb.col(t-1) = fp(hhtb[l].col(t-1)).cwiseProduct(VVb[l]*hhb[l].col(t));
      //cout << "Calculated dhft1 for back node " << t << " in layer " << l << endl;
    }
    MatrixXd dJdhhtb = dhhbdhhtb.cwiseProduct(deltab[l+1]); // verified by grad check
    MatrixXd dJdzzb = dhhbdzzb.cwiseProduct(deltab[l+1]); // verified by grad check
    MatrixXd dJdrrb = dhhtbdrrb.cwiseProduct(dJdhhtb); // verified by grad check

    // Update local gradients TODO: update V matrices
    gWWbf[l].noalias() += fp(hhtb[l]).cwiseProduct(dJdhhtb) * cur_hf.transpose();
    gWWbb[l].noalias() += fp(hhtb[l]).cwiseProduct(dJdhhtb) * cur_hb.transpose();
    gbbhb[l].noalias() += fp(hhtb[l]).cwiseProduct(dJdhhtb) * VectorXd::Ones(T);

    gWWzbf[l].noalias() += f2p(zzb[l]).cwiseProduct(dJdzzb) * cur_hf.transpose();
    gWWzbb[l].noalias() += f2p(zzb[l]).cwiseProduct(dJdzzb) * cur_hb.transpose();
    gbbzhb[l].noalias() += f2p(zzb[l]).cwiseProduct(dJdzzb) * VectorXd::Ones(T);

    gWWrbf[l].noalias() += f2p(rrb[l]).cwiseProduct(dJdrrb) * cur_hf.transpose();
    gWWrbb[l].noalias() += f2p(rrb[l]).cwiseProduct(dJdrrb) * cur_hb.transpose();
    gbbrhb[l].noalias() += f2p(rrb[l]).cwiseProduct(dJdrrb) * VectorXd::Ones(T);

    for (uint t = 0; t < T-1; t++) {
      gVVb[l].noalias() += rrb[l].col(t).cwiseProduct(dJdhhtb.col(t)) * hhb[l].col(t+1).transpose();
      gVVrb[l].noalias() += dJdrrb.col(t) * hhb[l].col(t+1).transpose();
      gVVzb[l].noalias() += dJdzzb.col(t) * hhb[l].col(t+1).transpose();
    }

    // Propagate error downwards
    if (l > 0) {
      deltaf[l].noalias() += WWzbf[l].transpose() * dJdzzb;
      deltaf[l].noalias() += WWrbf[l].transpose() * dJdrrb;
      deltaf[l].noalias() += WWbf[l].transpose() * dJdhhtb;

      deltab[l].noalias() += WWzbb[l].transpose() * dJdzzb;
      deltab[l].noalias() += WWrbb[l].transpose() * dJdrrb;
      deltab[l].noalias() += WWbb[l].transpose() * dJdhhtb;
    }

    #ifdef ERROR_SIGNAL
      // Add supervised error signal (i.e. WW(f/b)o * delta_y)
      if (layers != 0) {
        deltaf[l].noalias() += WWfo[l].transpose() * delta_y; 
        deltab[l].noalias() += WWbo[l].transpose() * delta_y;
      } else {
        deltaf[l].noalias() += Wfo.transpose() * delta_y; 
        deltab[l].noalias() += Wbo.transpose() * delta_y;
      }

    #endif
  }

  // Update gradients at input layer
  MatrixXd dhtfdrf = MatrixXd::Zero(nh, T);
  MatrixXd dhfdzf = MatrixXd::Zero(nh, T);

  // Calculate dhf_{t+1}^{(i)} / dhf_t^{(i)}
  MatrixXd Vzf_f2pzf = Vzf.transpose() * f2p(zf);
  MatrixXd Vrf_f2prf = Vrf.transpose() * f2p(rf);
  MatrixXd Vf_fphtf = Vf.transpose() * fp(htf);
  MatrixXd Vf_hf = Vf * hf;
  MatrixXd dhfdhtf = (MatrixXd::Ones(nh,T) - zf);
  for (uint t = T-2; t != (uint)(-1); t--) {
    VectorXd dhft1 = VectorXd::Zero(nh);
    dhft1 += zf.col(t+1);
    dhft1 += Vzf_f2pzf.col(t+1).cwiseProduct(hf.col(t) - htf.col(t+1));
    dhft1 += dhfdhtf.col(t+1).cwiseProduct(rf.col(t+1).cwiseProduct(Vf_fphtf.col(t+1))
                                                 + Vrf_f2prf.col(t+1).cwiseProduct(Vf_hf.col(t)).cwiseProduct(fp(htf.col(t+1))));
    deltaf[0].col(t) += dhft1.cwiseProduct(deltaf[0].col(t+1));
    dhfdzf.col(t+1)    = (hf.col(t) - htf.col(t+1));
    dhtfdrf.col(t+1) = fp(htf.col(t+1)).cwiseProduct(Vf*hf.col(t));
  }
  MatrixXd dJdhtf = dhfdhtf.cwiseProduct(deltaf[0]); // verified by grad check
  MatrixXd dJdzf = dhfdzf.cwiseProduct(deltaf[0]); // verified by grad check
  MatrixXd dJdrf = dhtfdrf.cwiseProduct(dJdhtf); // verified by grad check

  // Update local gradients TODO: update V matrices
  gWf.noalias() += fp(htf).cwiseProduct(dJdhtf) * x.transpose();
  gbhf.noalias() += fp(htf).cwiseProduct(dJdhtf) * VectorXd::Ones(T);

  gWzf.noalias() += f2p(zf).cwiseProduct(dJdzf) * x.transpose();
  gbzhf.noalias() += f2p(zf).cwiseProduct(dJdzf) * VectorXd::Ones(T);

  gWrf.noalias() += f2p(rf).cwiseProduct(dJdrf) * x.transpose();
  gbrhf.noalias() += f2p(rf).cwiseProduct(dJdrf) * VectorXd::Ones(T);

  for (uint t = 1; t < T; t++) {
    gVf.noalias()  += rf.col(t).cwiseProduct(dJdhtf.col(t)) * hf.col(t-1).transpose();
    gVrf.noalias() += dJdrf.col(t) * hf.col(t-1).transpose();
    gVzf.noalias() += dJdzf.col(t) * hf.col(t-1).transpose();
  }


  MatrixXd dhtbdrb = MatrixXd::Zero(nh, T);
  MatrixXd dhbdzb = MatrixXd::Zero(nh, T);

  // Calculate dhf_{t+1}^{(i)} / dhf_t^{(i)}
  MatrixXd Vzb_f2pzb = Vzb.transpose() * f2p(zb);
  MatrixXd Vrb_f2prb = Vrb.transpose() * f2p(rb);
  MatrixXd Vb_fphtb = Vb.transpose() * fp(htb);
  MatrixXd Vb_hb = Vb * hb;
  MatrixXd dhbdhtb = (MatrixXd::Ones(nh,T) - zb);
  for (uint t = 1; t < T; t++) {
    VectorXd dhbt1 = VectorXd::Zero(nh);
    dhbt1 += zb.col(t-1);
    dhbt1 += Vzb_f2pzb.col(t-1).cwiseProduct(hb.col(t) - htb.col(t-1));
    dhbt1 += dhbdhtb.col(t-1).cwiseProduct(rb.col(t-1).cwiseProduct(Vb_fphtb.col(t-1))
                                                 + Vrb_f2prb.col(t-1).cwiseProduct(Vb_hb.col(t)).cwiseProduct(fp(htb.col(t-1))));
    deltab[0].col(t) += dhbt1.cwiseProduct(deltab[0].col(t-1));
    dhbdzb.col(t-1)    = (hb.col(t) - htb.col(t-1));
    dhtbdrb.col(t-1) = fp(htb.col(t-1)).cwiseProduct(Vb*hb.col(t));
  }
  MatrixXd dJdhtb= dhbdhtb.cwiseProduct(deltab[0]); // verified by grad check
  MatrixXd dJdzb = dhbdzb.cwiseProduct(deltab[0]); // verified by grad check
  MatrixXd dJdrb = dhtbdrb.cwiseProduct(dJdhtb); // verified by grad check

  // Update local gradients TODO: update V matrices
  gWb.noalias()  += fp(htb).cwiseProduct(dJdhtb) * x.transpose();
  gbhb.noalias() += fp(htb).cwiseProduct(dJdhtb) * VectorXd::Ones(T);

  gWzb.noalias()  += f2p(zb).cwiseProduct(dJdzb) * x.transpose();
  gbzhb.noalias() += f2p(zb).cwiseProduct(dJdzb) * VectorXd::Ones(T);

  gWrb.noalias()  += f2p(rb).cwiseProduct(dJdrb) * x.transpose();
  gbrhb.noalias() += f2p(rb).cwiseProduct(dJdrb) * VectorXd::Ones(T);

  for (uint t = 0; t < T-1; t++) {
    gVb.noalias()  += rb.col(t).cwiseProduct(dJdhtb.col(t)) * hb.col(t+1).transpose();
    gVrb.noalias() += dJdrb.col(t) * hb.col(t+1).transpose();
    gVzb.noalias() += dJdzb.col(t) * hb.col(t+1).transpose();
  }

  return cost;
}

void GRURNN::update() {
  double lambda = LAMBDA;
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

  gWf.noalias()  += lambda*Wf;
  gVf.noalias()  += lambda*Vf;
  gWb.noalias()  += lambda*Wb;
  gVb.noalias()  += lambda*Vb;
  gbhf.noalias() += lambda*bhf;
  gbhb.noalias() += lambda*bhb;

  gWrf.noalias()  += lambda*Wrf;
  gVrf.noalias()  += lambda*Vrf;
  gWrb.noalias()  += lambda*Wrb;
  gVrb.noalias()  += lambda*Vrb;
  gbrhf.noalias() += lambda*brhf;
  gbrhb.noalias() += lambda*brhb;

  gWzf.noalias()  += lambda*Wzf;
  gVzf.noalias()  += lambda*Vzf;
  gWzb.noalias()  += lambda*Wzb;
  gVzb.noalias()  += lambda*Vzb;
  gbzhf.noalias() += lambda*bzhf;
  gbzhb.noalias() += lambda*bzhb;

  norm += gWf.squaredNorm() + gVf.squaredNorm()
          + gWb.squaredNorm() + gVb.squaredNorm()
          + gbhf.squaredNorm() + gbhb.squaredNorm();

  norm += gWrf.squaredNorm() + gVrf.squaredNorm()
          + gWrb.squaredNorm() + gVrb.squaredNorm()
          + gbrhf.squaredNorm() + gbrhb.squaredNorm();

  norm += gWzf.squaredNorm() + gVzf.squaredNorm()
          + gWzb.squaredNorm() + gVzb.squaredNorm()
          + gbrhf.squaredNorm() + gbrhb.squaredNorm();        

  for (uint l=0; l<layers; l++) {
    gWWff[l].noalias() += lambda*WWff[l];
    gWWfb[l].noalias() += lambda*WWfb[l];
    gWWbf[l].noalias() += lambda*WWbf[l];
    gWWbb[l].noalias() += lambda*WWbb[l];
    gVVf[l].noalias()  += lambda*VVf[l];
    gVVb[l].noalias()  += lambda*VVb[l];
    gbbhf[l].noalias() += lambda*bbhf[l];
    gbbhb[l].noalias() += lambda*bbhb[l];

    gWWrff[l].noalias() += lambda*WWrff[l];
    gWWrfb[l].noalias() += lambda*WWrfb[l];
    gWWrbf[l].noalias() += lambda*WWrbf[l];
    gWWrbb[l].noalias() += lambda*WWrbb[l];
    gVVrf[l].noalias()  += lambda*VVrf[l];
    gVVrb[l].noalias()  += lambda*VVrb[l];
    gbbrhf[l].noalias() += lambda*bbrhf[l];
    gbbrhb[l].noalias() += lambda*bbrhb[l];

    norm += gWWff[l].squaredNorm() + gWWfb[l].squaredNorm()
            + gWWbf[l].squaredNorm() + gWWbb[l].squaredNorm()
            + gVVf[l].squaredNorm() + gVVb[l].squaredNorm()
            + gbbhf[l].squaredNorm() + gbbhb[l].squaredNorm();

   norm += gWWrff[l].squaredNorm() + gWWrfb[l].squaredNorm()
            + gWWrbf[l].squaredNorm() + gWWrbb[l].squaredNorm()
            + gVVrf[l].squaredNorm() + gVVrb[l].squaredNorm()
            + gbbrhf[l].squaredNorm() + gbbrhb[l].squaredNorm();

   norm += gWWzff[l].squaredNorm() + gWWzfb[l].squaredNorm()
            + gWWrbf[l].squaredNorm() + gWWzbb[l].squaredNorm()
            + gVVzf[l].squaredNorm() + gVVzb[l].squaredNorm()
            + gbbzhf[l].squaredNorm() + gbbzhb[l].squaredNorm();
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

  vWrf  = lr*gWrf/norm  + mr*vWrf;
  vVrf  = lr*gVrf/norm  + mr*vVrf;
  vWrb  = lr*gWrb/norm  + mr*vWrb;
  vVrb  = lr*gVrb/norm  + mr*vVrb;
  vbrhf = lr*gbrhf/norm + mr*vbrhf;
  vbrhb = lr*gbrhb/norm + mr*vbrhb;

  vWzf  = lr*gWzf/norm  + mr*vWzf;
  vVzf  = lr*gVzf/norm  + mr*vVzf;
  vWzb  = lr*gWzb/norm  + mr*vWzb;
  vVzb  = lr*gVzb/norm  + mr*vVzb;
  vbzhf = lr*gbzhf/norm + mr*vbzhf;
  vbzhb = lr*gbzhb/norm + mr*vbzhb;


  for (uint l=0; l<layers; l++) {
    vWWff[l] = lr*gWWff[l]/norm + mr*vWWff[l];
    vWWfb[l] = lr*gWWfb[l]/norm + mr*vWWfb[l];
    vVVf[l]  = lr*gVVf[l]/norm  + mr*vVVf[l];
    vWWbb[l] = lr*gWWbb[l]/norm + mr*vWWbb[l];
    vWWbf[l] = lr*gWWbf[l]/norm + mr*vWWbf[l];
    vVVb[l]  = lr*gVVb[l]/norm  + mr*vVVb[l];
    vbbhf[l] = lr*gbbhf[l]/norm + mr*vbbhf[l];
    vbbhb[l] = lr*gbbhb[l]/norm + mr*vbbhb[l];

    vWWrff[l] = lr*gWWrff[l]/norm + mr*vWWrff[l];
    vWWrfb[l] = lr*gWWrfb[l]/norm + mr*vWWrfb[l];
    vVVrf[l]  = lr*gVVrf[l]/norm  + mr*vVVrf[l];
    vWWrbb[l] = lr*gWWrbb[l]/norm + mr*vWWrbb[l];
    vWWrbf[l] = lr*gWWrbf[l]/norm + mr*vWWrbf[l];
    vVVrb[l]  = lr*gVVrb[l]/norm  + mr*vVVrb[l];
    vbbrhf[l] = lr*gbbrhf[l]/norm + mr*vbbrhf[l];
    vbbrhb[l] = lr*gbbrhb[l]/norm + mr*vbbrhb[l];

    vWWzff[l] = lr*gWWzff[l]/norm + mr*vWWzff[l];
    vWWzfb[l] = lr*gWWzfb[l]/norm + mr*vWWzfb[l];
    vVVzf[l]  = lr*gVVzf[l]/norm  + mr*vVVzf[l];
    vWWzbb[l] = lr*gWWzbb[l]/norm + mr*vWWzbb[l];
    vWWzbf[l] = lr*gWWzbf[l]/norm + mr*vWWzbf[l];
    vVVzb[l]  = lr*gVVzb[l]/norm  + mr*vVVzb[l];
    vbbzhf[l] = lr*gbbzhf[l]/norm + mr*vbbzhf[l];
    vbbzhb[l] = lr*gbbzhb[l]/norm + mr*vbbzhb[l];
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

  Wrf.noalias()  -= vWrf;
  Vrf.noalias()  -= vVrf;
  Wrb.noalias()  -= vWrb;
  Vrb.noalias()  -= vVrb;
  brhf.noalias() -= vbrhf;
  brhb.noalias() -= vbrhb;

  Wzf.noalias()  -= vWzf;
  Vzf.noalias()  -= vVzf;
  Wzb.noalias()  -= vWzb;
  Vzb.noalias()  -= vVzb;
  bzhf.noalias() -= vbzhf;
  bzhb.noalias() -= vbzhb;

  for (uint l=0; l<layers; l++) {
    WWff[l].noalias() -= vWWff[l];
    WWfb[l].noalias() -= vWWfb[l];
    VVf[l].noalias()  -= vVVf[l];
    WWbb[l].noalias() -= vWWbb[l];
    WWbf[l].noalias() -= vWWbf[l];
    VVb[l].noalias()  -= vVVb[l];
    bbhf[l].noalias() -= vbbhf[l];
    bbhb[l].noalias() -= vbbhb[l];

    WWrff[l].noalias() -= vWWrff[l];
    WWrfb[l].noalias() -= vWWrfb[l];
    VVrf[l].noalias()  -= vVVrf[l];
    WWrbb[l].noalias() -= vWWrbb[l];
    WWrbf[l].noalias() -= vWWrbf[l];
    VVrb[l].noalias()  -= vVVrb[l];
    bbrhf[l].noalias() -= vbbrhf[l];
    bbrhb[l].noalias() -= vbbrhb[l];

    WWzff[l].noalias() -= vWWzff[l];
    WWzfb[l].noalias() -= vWWzfb[l];
    VVzf[l].noalias()  -= vVVzf[l];
    WWzbb[l].noalias() -= vWWzbb[l];
    WWzbf[l].noalias() -= vWWzbf[l];
    VVzb[l].noalias()  -= vVVzb[l];
    bbzhf[l].noalias() -= vbbzhf[l];
    bbzhb[l].noalias() -= vbbzhb[l];
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

  gWrf.setZero();
  gVrf.setZero();
  gWrb.setZero();
  gVrb.setZero();
  gbrhf.setZero();
  gbrhb.setZero();

  gWzf.setZero();
  gVzf.setZero();
  gWzb.setZero();
  gVzb.setZero();
  gbzhf.setZero();
  gbzhb.setZero();

  for (uint l=0; l<layers; l++) {
    gWWff[l].setZero();
    gWWfb[l].setZero();
    gVVf[l].setZero();
    gWWbb[l].setZero();
    gWWbf[l].setZero();
    gVVb[l].setZero();
    gbbhf[l].setZero();
    gbbhb[l].setZero();

    gWWrff[l].setZero();
    gWWrfb[l].setZero();
    gVVrf[l].setZero();
    gWWrbb[l].setZero();
    gWWrbf[l].setZero();
    gVVrb[l].setZero();
    gbbrhf[l].setZero();
    gbbrhb[l].setZero();

    gWWzff[l].setZero();
    gWWzfb[l].setZero();
    gVVzf[l].setZero();
    gWWzbb[l].setZero();
    gWWzbf[l].setZero();
    gVVzb[l].setZero();
    gbbzhf[l].setZero();
    gbbzhb[l].setZero();
  }

  lr *= 0.999;
  //cout << Wuo << endl;
}

bool GRURNN::is_nan() {
  return (bbhf[0](0) != bbhf[0](0));
}

void GRURNN::load(string fname) {
  // ifstream in(fname.c_str());

  // in >> nx >> nh >> nh >> ny;

  // in >> Wf >> Vf >> bhf
  // >> Wb >> Vb >> bhb;

  // for (uint l=0; l<layers; l++) {
  //   in >> WWff[l] >> WWfb[l] >> VVf[l] >> bbhf[l]
  //   >> WWbb[l] >> WWbf[l] >> VVb[l] >> bbhb[l];
  // }

  // in >> Wfo >> Wbo;
  // for (uint l=0; l<layers; l++)
  //   in >> WWfo[l] >> WWbo[l];
  // in >> Wo >> bo;
}

void GRURNN::save(string fname) {
  // ofstream out(fname.c_str());

  // out << nx << " " << nh << " " << nh << " " << ny << endl;

  // out << Wf << endl;
  // out << Vf << endl;
  // out << bhf << endl;

  // out << Wb << endl;
  // out << Vb << endl;
  // out << bhb << endl;

  // for (uint l=0; l<layers; l++) {
  //   out << WWff[l] << endl;
  //   out << WWfb[l] << endl;
  //   out << VVf[l] << endl;
  //   out << bbhf[l] << endl;

  //   out << WWbb[l] << endl;
  //   out << WWbf[l] << endl;
  //   out << VVb[l]  << endl;
  //   out << bbhb[l] << endl;
  // }

  // out << Wfo << endl;
  // out << Wbo << endl;
  // for (uint l=0; l<layers; l++) {
  //   out << WWfo[l] << endl;
  //   out << WWbo[l] << endl;
  // }
  // out << Wo << endl;
  // out << bo << endl;
}

Matrix<double, -1, 1> dropout(Matrix<double, -1, 1> x, double p) {
  for (uint i=0; i<x.size(); i++) {
    if ((double)rand()/RAND_MAX < p)
      x(i) = 0;
  }
  return x;
}

string GRURNN::model_name() {
  ostringstream strS;
  strS << "gru_drnt_" << layers << "_" << nh << "_"
  << nh << "_" << dropout_prob << "_"
  << lr << "_" << LAMBDA << "_" << mr ;
  string fname = strS.str();
  return fname;
}

void GRURNN::grad_check(vector<string> &sentence, vector<string> &labels) {
  //clear_gradients();
  backward(sentence, labels);
  // for (uint l = 0; l < layers; l++) {
  //   cout << "Grad checking WWzff[" << l << "]" << endl;
  //   cout << "Analytic gradient:" << endl;
  //   cout << gWWzff[l] << endl;
  //   cout << "Numerical gradient:" << endl;
  //   cout << numerical_gradient(WWzff[l], sentence, labels) << endl;
  //   cout << endl;
  // }

  // for (uint l = 0; l < layers; l++) {
  //   cout << "Grad checking WWff[" << l << "]" << endl;
  //   cout << "Analytic gradient:" << endl;
  //   cout << gWWff[l] << endl;
  //   cout << "Numerical gradient:" << endl;
  //   cout << numerical_gradient(WWff[l], sentence, labels) << endl;
  //   cout << endl;
  // }

  for (uint l = 0; l < layers; l++) {
    cout << "Grad checking VVf[" << l << "]" << endl;
    cout << "Analytic gradient:" << endl;
    cout << gVVf[l] << endl;
    cout << "Numerical gradient:" << endl;
    cout << numerical_gradient(VVf[l], sentence, labels) << endl;
    cout << endl;
  }

  // for (uint l = 0; l < layers; l++) {
  //   cout << "Grad checking WWrff[" << l << "]" << endl;
  //   cout << "Analytic gradient:" << endl;
  //   cout << gWWrff[l] << endl;
  //   cout << "Numerical gradient:" << endl;
  //   cout << numerical_gradient(WWrff[l], sentence, labels) << endl;
  //   cout << endl;
  // }

  // for (uint l = layers - 1; l < layers; l++) {
  //   cout << "Grad checking WWzff[" << l << "]" << endl;
  //   cout << "Analytic gradient:" << endl;
  //   cout << gWWzff[l] << endl;
  //   cout << "Numerical gradient:" << endl;
  //   cout << numerical_gradient(WWzff[l], sentence, labels) << endl;
  //   cout << endl;
  // }
}

MatrixXd GRURNN::numerical_gradient(MatrixXd &parameter, vector<string> &sentence, vector<string> &labels) {
  double h = 1e-5;
  MatrixXd grad = MatrixXd::Zero(parameter.rows(), parameter.cols());
  for (int i = 0; i < parameter.rows(); i++) {
    for (int j = 0; j < parameter.cols(); j++) {
      double old_value = parameter(i, j);

      parameter(i, j) = old_value + h;
      double right_cost = cost(sentence, labels);
      parameter(i, j) = old_value - h;
      double left_cost = cost(sentence, labels);
      grad(i, j) = (right_cost - left_cost) / (2*h);

      parameter(i, j) = old_value;
    }
  }

  return grad;
}

int main(int argc, char **argv) {
  cout << setprecision(3);

  // Set default arguments
  int seed     = 135;
  float lr     = 0.05;
  float mr     = 0.7;
  float null_class_weight = 0.5;
  float dropout_prob = 0.0;
  string data  = "";

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
      };
    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long (argc, argv, "a:b:c:d:f:g:",
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

      case '?':
        /* getopt_long already printed an error message. */
        break;

      default:
        abort ();
    }
  }

  srand(seed);

  LookupTable LT;
  // i used mikolov's word2vec (300d) for my experiments, not CW
  LT.load("embeddings-original.EMBEDDING_SIZE=25.txt", 268810, 25, false);
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

  Matrix<double, 6, 2> best = Matrix<double, 6, 2>::Zero();
  GRURNN brnn(25,25,ny,LT, lr, mr, null_class_weight, dropout_prob);
  // for (int i = 0; i < 10; i++)
  //   brnn.grad_check(trainX[i], trainL[i]);
  // //cout << brnn.backward(trainX[1], trainL[1]) << endl;
  auto results = brnn.train(trainX, trainL, validX, validL, testX, testL, 200, 80);
  
  return 0;
}

