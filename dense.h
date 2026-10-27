// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef DENSE_H
#define DENSE_H
// Dense complex linear algebra, fp64, single-threaded.
//
// These entry points are the whole of qc's dense-matrix work outside its own
// stencil kernels.  They exist so the default build needs no external BLAS or
// LAPACK: MKL is x86-only and an icx spelling, and building without it used to
// mean falling back on a different *algorithm* (iterative block and bottom
// solves) rather than the same one at a different speed.  DENSE_BLAS=mkl or
// =cblas re-points the bodies at a vendor library; every call site above is
// unaware of which is in play.
//
// Two storage conventions, one per routine, each matching its single caller so
// that neither call site needs an index rewrite:
//
//   dense_gemm                    row-major
//   dense_getrf / dense_getrs     column-major   (coarse.c, the bottom inverse)
//   dense_hetrf / dense_hetri     row-major, lower triangle  (coarse_sap.c)
//   dense_apply_d / dense_apply_f column-major   (coarse.c, the bottom apply)
//
// Do not "unify" the two triangular conventions without rewriting the callers.
// A row-major lower-triangle Hermitian buffer is not a column-major lower one:
// read column-major it is the conjugate, with the triangles swapped.
//
// Nothing here is threaded.  Every caller is already inside an OpenMP region or
// splits the work itself (coarse_sap.c parallelizes over SAP blocks, coarse.c
// over inverse-column chunks and output row slabs), which is also why the MKL
// backend is linked -qmkl=sequential.  Keep that contract.

#include <complex.h>

// C = A B, no transposes.  A is M x K, B is K x N, C is M x N, all row-major
// with the given leading dimensions (elements per row).  C is written, not
// accumulated.
void dense_gemm(long M, long N, long K,
                const double complex * A, long lda,
                const double complex * B, long ldb,
                double complex * C, long ldc);

// LU with partial pivoting, column-major, order n, leading dimension n.
// A is overwritten by L (unit lower) and U (upper); ipiv[i] is the 0-based row
// interchanged with row i at step i.  Returns 0, or the 1-based index of the
// first exactly-zero pivot -- LAPACK's `info` convention, so a vendor backend
// forwards it unchanged.
int dense_getrf(double complex * A, long n, int * ipiv);

// Solve A X = B in place for nrhs columns given dense_getrf's output.
// B is n x nrhs column-major, leading dimension n.
void dense_getrs(const double complex * A, long n, const int * ipiv,
                 double complex * B, long nrhs);

// Bunch-Kaufman A = L D L^H for a Hermitian A, row-major, order n, leading
// dimension n.  Only the lower triangle is read; the strict upper is left
// alone.  D is block diagonal with 1x1 and 2x2 blocks.  ipiv follows LAPACK:
// ipiv[k] >= 0 means a 1x1 pivot that interchanged rows k and ipiv[k];
// ipiv[k] = ipiv[k+1] = -(p+1) means a 2x2 pivot in rows k,k+1 that
// interchanged row k+1 with row p.  Returns 0, or the 1-based index of a zero
// block pivot.
int dense_hetrf(double complex * A, long n, int * ipiv);

// Explicit inverse from dense_hetrf's output, in place.  The lower triangle is
// the inverse's lower triangle; the strict upper is unspecified (the caller
// mirrors it, which is what makes the stored result exactly Hermitian).
// Returns 0, or nonzero if D is singular.
int dense_hetri(double complex * A, long n, const int * ipiv);

// X (rows x nrhs) = INV (rows x n_in) B (n_in x nrhs), all column-major.  `ld`
// is the leading dimension shared by INV and X, which is not `rows`: the caller
// splits a taller matrix into row slabs across threads and hands each one a
// slab.  B's leading dimension is n_in.
//
// Not dense_gemm: nrhs is num_rhs (16), far too skinny for a blocked GEMM, and
// the point of the shape is that the matrix streams past exactly once for all
// the right-hand sides at once.  Two precisions rather than one template
// because BOTTOM_INV_FP32 picks between them and both can be live in a binary.
void dense_apply_d(const double complex * INV, long ld, long rows, long n_in,
                   long nrhs, const double complex * B, double complex * X);
void dense_apply_f(const float complex * INV, long ld, long rows, long n_in,
                   long nrhs, const float complex * B, float complex * X);

#endif // DENSE_H
