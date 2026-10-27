// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

// `./qc densecheck` -- correctness gate for dense.c.
//
// Every routine is checked against something that shares no code with it: the
// GEMM against a naive triple loop, the two factorizations against the
// residual of the system they claim to have solved.  Residuals are reported
// in the LAPACK style, scaled by n * eps * ||A|| * ||X||, so the bar does not
// move with the conditioning of whatever random matrix came out of the
// generator -- which matters here because the Hermitian test matrix is
// deliberately *indefinite* (that is why Bunch-Kaufman is needed and Cholesky
// would fail), so its condition number is not under our control.
//
// Shapes cover the production dims (64 and 96 for the per-site coarse blocks,
// 512 for the even-Schur complement, and the bottom's order) and deliberately
// unround ones, since every blocking edge in dense.c -- the MR/NR microkernel
// tile, the MC/NC/KC cache blocks, the NB factorization panel -- is a place
// where a multiple-of-everything shape would hide a bug.

// System headers before test.h, which pulls in global_params.h and its checked
// allocator macros -- see the note at the top of dense.c.
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "test.h"
#include "dense.h"
#include "comm.h"

// Knuth MMIX linear congruential generator; see test/kernel_tests.c
static unsigned long dt_seed = 12345;
static double dt_rand(void) {                     // uniform on (-0.5, 0.5)
  dt_seed = dt_seed * 6364136223846793005UL + 1442695040888963407UL;
  return (double) ((dt_seed >> 40) & 0xffffff) / 16777216.0 - 0.5;
}
static double complex dt_randz(void) { return dt_rand() + I * dt_rand(); }

static double norm_inf(const double complex * A, long m, long n, long lda) {
  double best = 0;
  for (long i = 0; i < m; i ++) {
    double s = 0;
    for (long j = 0; j < n; j ++) s += cabs(A[i * lda + j]);
    if (s > best) best = s;
  }
  return best;
}

static int dt_fail;

// The build is -ffast-math, which tells the compiler no NaN can occur -- so it
// folds `got == got` to true and a NaN verdict prints as PASS.  This one did,
// before the check was written this way.  Route the value through a volatile
// and test the exponent/mantissa bits, which constant folding cannot reach.
static int is_nan(double v) {
  volatile double g = v;
  unsigned long long bits;
  memcpy(&bits, (const void *) &g, sizeof bits);
  return ((bits >> 52) & 0x7ff) == 0x7ff && (bits & 0xfffffffffffffULL) != 0;
}

static void verdict(const char * what, double got, double bar) {
  const int ok = !is_nan(got) && got < bar;
  print0("[densecheck] %-34s %9.3e  (bar %.0e)  => %s\n",
         what, got, bar, ok ? "PASS" : "FAIL");
  if (!ok) dt_fail = 1;
}

// --- GEMM vs a naive triple loop -------------------------------------------
static void check_gemm(long M, long N, long K) {
  double complex * A = malloc(sizeof(double complex) * (size_t) M * K);
  double complex * B = malloc(sizeof(double complex) * (size_t) K * N);
  double complex * C = malloc(sizeof(double complex) * (size_t) M * N);
  double complex * R = malloc(sizeof(double complex) * (size_t) M * N);
  for (long i = 0; i < M * K; i ++) A[i] = dt_randz();
  for (long i = 0; i < K * N; i ++) B[i] = dt_randz();
  for (long i = 0; i < M * N; i ++) C[i] = 1234.5;       // must be overwritten
  dense_gemm(M, N, K, A, K, B, N, C, N);
  for (long i = 0; i < M; i ++)
    for (long j = 0; j < N; j ++) {
      double complex s = 0;
      for (long p = 0; p < K; p ++) s += A[i * K + p] * B[p * N + j];
      R[i * N + j] = s;
    }
  // The generator emits 24-bit mantissas, so every product fits in 48 bits and
  // no partial sum here reaches 2^53: the fp64 GEMM is exact, and a correct
  // backend matches the reference bitwise regardless of summation order.  The
  // 0.000e+00 this prints is that, not a self-comparison; the fp64 apply
  // checks below are exact the same way.
  double dmax = 0, nmax = 0;
  for (long i = 0; i < M * N; i ++) {
    const double d = cabs(C[i] - R[i]);
    if (d > dmax) dmax = d;
    if (cabs(R[i]) > nmax) nmax = cabs(R[i]);
  }
  char tag[64];
  snprintf(tag, sizeof tag, "gemm %ldx%ldx%ld", M, N, K);
  verdict(tag, dmax / (nmax > 0 ? nmax : 1), 1e-11);
  free(A); free(B); free(C); free(R);
}

// --- LU: solve, then measure the residual of the original system ------------
static void check_lu(long n, long nrhs) {
  const size_t nn = (size_t) n * n;
  double complex * A  = malloc(sizeof(double complex) * nn);
  double complex * A0 = malloc(sizeof(double complex) * nn);
  double complex * B  = malloc(sizeof(double complex) * (size_t) n * nrhs);
  double complex * B0 = malloc(sizeof(double complex) * (size_t) n * nrhs);
  int * ipiv = malloc(sizeof(int) * (size_t) n);
  for (size_t i = 0; i < nn; i ++) A0[i] = A[i] = dt_randz();
  for (long i = 0; i < n * nrhs; i ++) B0[i] = B[i] = dt_randz();

  const int info = dense_getrf(A, n, ipiv);
  if (info) {
    print0("[densecheck] getrf n=%ld reported singular (info=%d)\n", n, info);
    dt_fail = 1;
  }
  dense_getrs(A, n, ipiv, B, nrhs);

  // A0 and B0 are column-major; norm_inf wants row-major, and for these
  // purposes the transpose norm is just as good a scale factor.
  const double na = norm_inf(A0, n, n, n), nx = norm_inf(B, nrhs, n, n);
  double rmax = 0;
  for (long c = 0; c < nrhs; c ++)
    for (long i = 0; i < n; i ++) {
      double complex s = 0;
      for (long p = 0; p < n; p ++) s += A0[i + p * n] * B[p + c * n];
      const double d = cabs(s - B0[i + c * n]);
      if (d > rmax) rmax = d;
    }
  char tag[64];
  snprintf(tag, sizeof tag, "lu solve n=%ld nrhs=%ld", n, nrhs);
  verdict(tag, rmax / (n * 2.220446049250313e-16 * na * nx), 1e2);
  free(A); free(A0); free(B); free(B0); free(ipiv);
}

// --- Bunch-Kaufman: invert, mirror the way csap_cinv does, check A A^-1 = I
// The mirror is part of what is under test, not decoration: dense_hetri only
// promises the lower triangle, and coarse_sap.c's exact-Hermiticity contract
// rests on the upper being its conjugate.  Checking the residual of the
// *mirrored* matrix is therefore the check that matches how it is used; a
// separate "is it Hermitian" assertion would only re-test the mirror loop.
static void check_hetri(long n) {
  const size_t nn = (size_t) n * n;
  double complex * A  = malloc(sizeof(double complex) * nn);
  double complex * A0 = malloc(sizeof(double complex) * nn);
  int * ipiv = malloc(sizeof(int) * (size_t) n);
  for (long i = 0; i < n; i ++) {              // random Hermitian, indefinite:
    for (long j = 0; j < i; j ++) {            // real diagonal, no dominance
      const double complex v = dt_randz();
      A0[i * n + j] = v;
      A0[j * n + i] = conj(v);
    }
    A0[i * n + i] = dt_rand();
  }
  for (size_t i = 0; i < nn; i ++) A[i] = A0[i];

  int info = dense_hetrf(A, n, ipiv);
  if (info) {
    print0("[densecheck] hetrf n=%ld reported singular (info=%d)\n", n, info);
    dt_fail = 1;
  }
  info = dense_hetri(A, n, ipiv);
  if (info) {
    print0("[densecheck] hetri n=%ld reported singular (info=%d)\n", n, info);
    dt_fail = 1;
  }
  for (long i = 0; i < n; i ++)                   // mirror, as csap_cinv does
    for (long j = i + 1; j < n; j ++)
      A[i * n + j] = conj(A[j * n + i]);

  const double na = norm_inf(A0, n, n, n), nx = norm_inf(A, n, n, n);
  double rmax = 0;
  for (long i = 0; i < n; i ++)
    for (long j = 0; j < n; j ++) {
      double complex s = 0;
      for (long p = 0; p < n; p ++) s += A0[i * n + p] * A[p * n + j];
      const double d = cabs(s - (i == j ? 1.0 : 0.0));
      if (d > rmax) rmax = d;
    }
  char tag[64];
  snprintf(tag, sizeof tag, "hermitian inverse n=%ld", n);
  verdict(tag, rmax / (n * 2.220446049250313e-16 * na * nx), 1e2);
  free(A); free(A0); free(ipiv);
}

// --- the bottom-inverse apply, both storage precisions ----------------------
static void check_apply(long n_out, long n_in, long nrhs) {
  double complex * INV = malloc(sizeof(double complex) * (size_t) n_out * n_in);
  double complex * B   = malloc(sizeof(double complex) * (size_t) n_in * nrhs);
  double complex * X   = malloc(sizeof(double complex) * (size_t) n_out * nrhs);
  double complex * R   = malloc(sizeof(double complex) * (size_t) n_out * nrhs);
  for (long i = 0; i < n_out * n_in; i ++) INV[i] = dt_randz();
  for (long i = 0; i < n_in * nrhs; i ++) B[i] = dt_randz();
  for (long i = 0; i < n_out * nrhs; i ++) X[i] = 987.6;  // must be overwritten
  dense_apply_d(INV, n_out, n_out, n_in, nrhs, B, X);
  for (long l = 0; l < nrhs; l ++)
    for (long r = 0; r < n_out; r ++) {
      double complex s = 0;
      for (long c = 0; c < n_in; c ++)
        s += INV[r + c * n_out] * B[c + l * n_in];
      R[r + l * n_out] = s;
    }
  double dmax = 0, nmax = 0;
  for (long i = 0; i < n_out * nrhs; i ++) {
    const double d = cabs(X[i] - R[i]);
    if (d > dmax) dmax = d;
    if (cabs(R[i]) > nmax) nmax = cabs(R[i]);
  }
  char tag[64];
  snprintf(tag, sizeof tag, "apply fp64 %ldx%ldx%ld", n_out, nrhs, n_in);
  verdict(tag, dmax / (nmax > 0 ? nmax : 1), 1e-12);

  // Row slabs, which is how the bottom solve actually calls it: each thread
  // gets a slab whose row count is not its leading dimension.  An unequal
  // split, so a stride bug cannot hide behind two identical halves.
  for (long i = 0; i < n_out * nrhs; i ++) X[i] = 987.6;
  const long cut = n_out / 3;
  dense_apply_d(INV,       n_out, cut,         n_in, nrhs, B, X);
  dense_apply_d(INV + cut, n_out, n_out - cut, n_in, nrhs, B, X + cut);
  dmax = 0;
  for (long i = 0; i < n_out * nrhs; i ++) {
    const double d = cabs(X[i] - R[i]);
    if (d > dmax) dmax = d;
  }
  snprintf(tag, sizeof tag, "apply fp64 slabbed %ld+%ld", cut, n_out - cut);
  verdict(tag, dmax / (nmax > 0 ? nmax : 1), 1e-12);

  float complex * If = malloc(sizeof(float complex) * (size_t) n_out * n_in);
  float complex * Bf = malloc(sizeof(float complex) * (size_t) n_in * nrhs);
  float complex * Xf = malloc(sizeof(float complex) * (size_t) n_out * nrhs);
  for (long i = 0; i < n_out * n_in; i ++) If[i] = (float complex) INV[i];
  for (long i = 0; i < n_in * nrhs; i ++) Bf[i] = (float complex) B[i];
  for (long i = 0; i < n_out * nrhs; i ++) Xf[i] = 987.6f;
  dense_apply_f(If, n_out, n_out, n_in, nrhs, Bf, Xf);
  dmax = 0;
  for (long i = 0; i < n_out * nrhs; i ++) {
    const double d = cabs((double complex) Xf[i] - R[i]);
    if (d > dmax) dmax = d;
  }
  snprintf(tag, sizeof tag, "apply fp32 %ldx%ldx%ld", n_out, nrhs, n_in);
  verdict(tag, dmax / (nmax > 0 ? nmax : 1), 1e-5);
  free(INV); free(B); free(X); free(R); free(If); free(Bf); free(Xf);
}

void test_dense(void) {
  dt_fail = 0;
  print0("[densecheck] backend: %s\n",
#if DENSE_BLAS_MKL
         "MKL"
#elif DENSE_BLAS_CBLAS
         "external CBLAS/LAPACKE"
#else
         "built-in (dense.c)"
#endif
        );
  // M, N, K chosen to straddle every blocking edge: 64^3 fits one cache block;
  // 100x130x77 is a multiple of no tile dimension; 257x129x193 crosses MC and
  // KC; 130x300x140 crosses NC and KC.
  check_gemm(64, 64, 64);
  check_gemm(100, 130, 77);
  check_gemm(257, 129, 193);
  check_gemm(130, 300, 140);

  check_lu(64, 16);
  check_lu(100, 1);
  check_lu(129, 17);
  check_lu(512, 16);

  check_hetri(64);                  // per-odd-site block, narrow coarse level
  check_hetri(96);                  // per-odd-site block, prod coarse level
  check_hetri(129);                 // odd order, crosses the NB panel
  check_hetri(512);                 // even-Schur complement

  check_apply(256, 256, 16);
  check_apply(100, 133, 16);

  print0("[densecheck] %s\n", dt_fail ? "FAILED" : "all checks passed");
}
