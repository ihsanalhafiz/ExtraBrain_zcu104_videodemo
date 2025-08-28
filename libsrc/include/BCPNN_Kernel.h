#ifndef __Bcpnn_Kernel_included
#define __Bcpnn_Kernel_included

#include <vector>
#include <cstring>
#include <stdint.h>
#include <stddef.h>
#include "ap_int.h"
#include "ap_fixed.h"

#ifndef WORD_SIZE
#define WORD_SIZE 16
#endif
#ifndef INTEGER_SIZE
#define INTEGER_SIZE 4
#endif
typedef ap_fixed<WORD_SIZE, INTEGER_SIZE > fixed_hls;

#define ABSENT 0
#define SILENT 1
#define ACTIVE 2

#define UNSUPERVISED 0
#define SUPERVISED 1
#define INFERENCES 2

const int NumberPop = 3;
const float eps_hls = 1e-4;
const float EPS_GLB = 1e-4;

// Layer Population Input
const int H_in = H_IN;
const int M_in = M_IN;
const int N_in = H_in * M_in;

// Layer Population Hidden
const int H_hid = H_HID;
const int M_hid = M_HID;
const int N_hid = H_hid * M_hid;

// Layer Population Output
const int H_ut = H_UT;
const int M_ut = M_UT;
const int N_ut = H_ut * M_ut;

const int nactHi_pop = NACTHI;
const int nsilHi_pop = NSILHI;
const float fgain = 1.0;
const float tauzjdt = 1.0;
const float tauzidt = 1.0;
const float again_hls = 1.0;

// Layer Projection Input to Hidden
const int axoHi_ih = H_in;
const int axoNi_ih = axoHi_ih*M_in;
const int nactHi_ih = nactHi_pop;
const int denHi_ih = nactHi_ih + nsilHi_pop;
const int denNi_ih = denHi_ih*M_in;

// Layer Projection Hidden to Output
const int axoHi_hu = H_hid;
const int axoNi_hu = axoHi_hu*M_hid;
const int nactHi_hu = axoHi_hu;
const int denHi_hu = nactHi_hu + 0;
const int denNi_hu = denHi_hu*M_hid;

// enum for modeOps
enum modeOps {
    UNSUPERVISED_TRAIN = 0,
    SUPERVISED_TRAIN = 1,
    INFERENCES_MODE = 2
};

void BCPNN_Kernel(float *input_hbm, float *label_hbm, float *output_hbm, int modeOps, float *rndPoisson_hid_hbm, int *Hihjhi_ih_hbm, int *Chjhi_ih_hbm, char *needsupdbw_hbm,
                float *Pj_ih_hbm, float *Pi_ih_hbm, float *Pji_ih_hbm, float *Bj_ih_hbm, float *Wji_ih_hbm, float *Wji_ih_hbm1, float *Wji_ih_hbm2,
                float *Pj_hu_hbm, float *Pi_hu_hbm, float *Pji_hu_hbm, float *Bj_hu_hbm, float *Wji_hu_hbm,
                float *constant_hbm);
void BCPNN_infer_fixed(fixed_hls *input_hbm, fixed_hls *output_hbm, int16_t *rndPoisson_hid_hbm, int16_t *Hihjhi_ih_hbm,
                  fixed_hls *Bj_ih_hbm, fixed_hls *Wji_ih_hbm, fixed_hls *Bj_hu_hbm, fixed_hls *Wji_hu_hbm,
                  fixed_hls nampl, int16_t nfreq,fixed_hls igain0,
                  fixed_hls igain2, fixed_hls bwgain1,fixed_hls bwgain2,
                  fixed_hls taumdt0,fixed_hls taumdt1,fixed_hls taumdt2);
void BCPNN_infer_half(half *input_hbm, half *output_hbm, int16_t *rndPoisson_hid_hbm, int16_t *Hihjhi_ih_hbm,
                  half *Bj_ih_hbm, half *Wji_ih_hbm, half *Bj_hu_hbm, half *Wji_hu_hbm,
                  half nampl, int16_t nfreq,half igain0,
                  half igain2, half bwgain1,half bwgain2,
                  half taumdt0,half taumdt1,half taumdt2);
void BCPNN_infer_float(float *input_hbm, float *output_hbm, int *rndPoisson_hid_hbm, int *Hihjhi_ih_hbm,
                  float *Bj_ih_hbm, float *Wji_ih_hbm, float *Bj_hu_hbm, float *Wji_hu_hbm,
                  float nampl, int nfreq,float igain0,
                  float igain2, float bwgain1,float bwgain2,
                  float taumdt0,float taumdt1,float taumdt2);
#endif // __Bcpnn_Kernel_included