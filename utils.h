// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef UTILS_H
#define UTILS_H

#include "global_params.h"
#include <complex.h>

void read_gauge(REAL * U, char * path);
void compute_Fmunu(REAL * U, REAL * Fmunu);
void matmul(REAL * A, REAL * B, REAL * C, int dag);
// In-place-safe inverse of one dense N x N interleaved-complex matrix
// (re,im,re,im,... row-major), computed in double precision via Gauss-Jordan
// with partial pivoting.  No external (MKL/LAPACK) dependency.
void invert_dense_block(const REAL * src, REAL * dst, int N);
void copy_gauge(REAL * output, REAL * input);
void rescale_multi_ferm(REAL * ferm, REAL c, long num_ferm);
void copy_multi_ferm(REAL * output, REAL * input, long num_ferm);
REAL multi_half_ferm_norm(REAL * ferm, long num_ferm);
void multi_half_ferm_norm_rhs(REAL * ferm, long num_ferm, REAL * norms);
void zero_multi_half_ferm(REAL * output, long num_ferm);
void copy_multi_half_ferm(REAL * output, REAL * input, long num_ferm);

complex REAL cdot_ferm(REAL * a, REAL * b, long length);
void caxpy_ferm(REAL * output, REAL * input, complex REAL a, long length);

// The (REAL) cast forces the products up to this TU's working precision before
// they hit the accumulators: a no-op for same-precision operands, and for
// mixed/float operands in the double TU (mixed fgmres: float V basis, float w)
// it gives double products over float reads (promote-on-load, like DDalphaAMG's
// process_multi_inner_product_MP -- but theirs multiplies in float first).
#define DOTC_KERNEL(sum_re, sum_im, a, b, index) { \
  sum_re += (REAL) (a)[index] * (b)[index]; \
  sum_re += (REAL) (a)[index + num_rhs] * (b)[index + num_rhs]; \
  sum_im += (REAL) (a)[index] * (b)[index + num_rhs]; \
  sum_im -= (REAL) (a)[index + num_rhs] * (b)[index]; \
}

#define CAXPY_KERNEL(output, input, a_re, a_im, index) { \
  (output)[index] += (input)[index] * a_re; \
  (output)[index + num_rhs] += (input)[index + num_rhs] * a_re; \
  (output)[index] -= (input)[index + num_rhs] * a_im; \
  (output)[index + num_rhs] += (input)[index] * a_im; \
}

#endif // UTILS_H
