// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

// Dense complex linear algebra -- see dense.h for the storage conventions and
// for why qc carries its own instead of requiring MKL.
//
// Layout note that makes the Hermitian half of this file readable: LAPACK's
// column-major lower triangle A(I,J), I >= J, and our row-major lower triangle
// A[i*n+j], j <= i, have *identical index semantics* and differ only in where
// the element sits in memory.  So the Bunch-Kaufman code below transcribes
// LAPACK's zhetf2 index-for-index; what changes is loop order, since our
// contiguous direction is j (a row of the triangle) where theirs is i.

// System headers first: global_params.h redefines malloc/calloc as the checked
// allocators, and a later <stdlib.h> would then see its own declarations
// rewritten.  A build with AVX2 or AVX512 hides that -- <immintrin.h> pulls
// <stdlib.h> in ahead of the macros -- so only a scalar build catches it.
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Backend selection.  Both external choices expose the same CBLAS/LAPACKE API
// and differ only in the header and the link flags, so there is one external
// path with a two-line include block.  #if/#else rather than a composite
// #define so the live predicate reads off the source.
#ifndef DENSE_BLAS_MKL
#define DENSE_BLAS_MKL 0
#endif
#ifndef DENSE_BLAS_CBLAS
#define DENSE_BLAS_CBLAS 0
#endif
#if DENSE_BLAS_MKL || DENSE_BLAS_CBLAS
#define DENSE_EXTERNAL 1
#else
#define DENSE_EXTERNAL 0
#endif

// The vendor headers belong in the system-header group for a second reason
// beyond the malloc one above: global_params.h defines short object macros --
// V, Vh, Nc, Ns, Lx -- and <lapack.h> declares parameters named V, so including
// it afterwards rewrites those declarations into numeric constants and the
// header stops parsing.  MKL dodges it only because mkl.h happens not to use a
// bare V.  Nothing in this block needs global_params.h.
#if DENSE_EXTERNAL
#if DENSE_BLAS_MKL
#define MKL_Complex16 double _Complex
#define MKL_Complex8  float _Complex
#include <mkl.h>
#else
#define LAPACK_COMPLEX_CUSTOM
#define lapack_complex_double double _Complex
#define lapack_complex_float  float _Complex
#include <cblas.h>
#include <lapacke.h>
#endif
// Every ipiv crossing this interface is an int array; an ILP64 library would
// write 8-byte pivots into it.  qc never needs 64-bit dimensions, so pin LP64.
_Static_assert(sizeof(lapack_int) == sizeof(int),
               "ILP64 BLAS/LAPACKE linked; dense.c's pivot arrays are int (LP64)");
#endif

#include "global_params.h"      // checked malloc/calloc; AVX2 / AVX512
#include "dense.h"

#if DENSE_EXTERNAL
// ---------------------------------------------------------------------------
// Vendor backend: forward, and let the library own the work.  ipiv crosses
// only between dense_hetrf and dense_hetri (and between dense_getrf and
// dense_getrs), never between backends, so LAPACK's 1-based pivot convention
// needs no translation -- it just has to be self-consistent within the
// backend in play.
// ---------------------------------------------------------------------------

void dense_gemm(long M, long N, long K,
                const double complex * A, long lda,
                const double complex * B, long ldb,
                double complex * C, long ldc) {
  const double complex one = 1.0, zero = 0.0;
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) M, (int) N,
              (int) K, &one, A, (int) lda, B, (int) ldb, &zero, C, (int) ldc);
}

int dense_getrf(double complex * A, long n, int * ipiv) {
  return (int) LAPACKE_zgetrf(LAPACK_COL_MAJOR, (lapack_int) n, (lapack_int) n,
                              A, (lapack_int) n, (lapack_int *) ipiv);
}

void dense_getrs(const double complex * A, long n, const int * ipiv,
                 double complex * B, long nrhs) {
  LAPACKE_zgetrs(LAPACK_COL_MAJOR, 'N', (lapack_int) n, (lapack_int) nrhs,
                 A, (lapack_int) n, (const lapack_int *) ipiv, B,
                 (lapack_int) n);
}

int dense_hetrf(double complex * A, long n, int * ipiv) {
  return (int) LAPACKE_zhetrf(LAPACK_ROW_MAJOR, 'L', (lapack_int) n, A,
                              (lapack_int) n, (lapack_int *) ipiv);
}

int dense_hetri(double complex * A, long n, const int * ipiv) {
  return (int) LAPACKE_zhetri(LAPACK_ROW_MAJOR, 'L', (lapack_int) n, A,
                              (lapack_int) n, (const lapack_int *) ipiv);
}

void dense_apply_d(const double complex * INV, long ld, long rows, long n_in,
                   long nrhs, const double complex * B, double complex * X) {
  const double complex one = 1.0, zero = 0.0;
  cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, (int) rows,
              (int) nrhs, (int) n_in, &one, INV, (int) ld, B, (int) n_in,
              &zero, X, (int) ld);
}

void dense_apply_f(const float complex * INV, long ld, long rows, long n_in,
                   long nrhs, const float complex * B, float complex * X) {
  const float complex one = 1.0f, zero = 0.0f;
  cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, (int) rows,
              (int) nrhs, (int) n_in, &one, INV, (int) ld, B, (int) n_in,
              &zero, X, (int) ld);
}

#else
// ===========================================================================
// Native implementation
// ===========================================================================

// Microkernel tile, in complex elements.  MR*NR complex accumulators occupy
// 2*MR*NR doubles, and the whole point is to keep them in registers: at NR=8 a
// row of accumulators is exactly one AVX-512 double vector, so MR=6 gives 12
// live vectors -- enough independent FMA chains to cover the pipeline latency.
// A 16-register AVX2 file cannot hold that, so it takes a squarer, smaller tile.
#if AVX512
enum { MR = 6, NR = 8 };
#else
enum { MR = 4, NR = 4 };
#endif

// One accumulator row.  Naming the j direction as a vector type is what keeps
// the tile in registers: written as a plain loop the compiler unrolls j first
// (its trip count is the constant NR) and then vectorizes i instead, whose
// stride is NR, so the accumulators land in memory behind gathers.  This is a
// compiler-generic construct, not an intrinsic, so it lowers to whatever the
// target has -- zmm, ymm, NEON, or a scalar sequence.  It is fp64 regardless of
// the hierarchy's REAL, which is why it is not REAL_VEC.
typedef double vec_t __attribute__((vector_size(sizeof(double) * NR)));

// Cache blocking.  Ap (MC x KC) is sized for L2 and Bp (KC x NC) for L3; the
// packed buffers are allocated per call at the *actual* blocked size, so the
// many small (64^3) GEMMs in the coarse-SAP setup pay kilobytes, not megabytes.
enum { MC = 120, NC = 256, KC = 128 };

// Panel width for the blocked factorizations.
enum { NB = 64 };

static inline long dmin(long a, long b) { return a < b ? a : b; }
static inline long roundup(long v, long m) { return (v + m - 1) / m * m; }

// --- packing ---------------------------------------------------------------
// Both packers emit the same shape: consecutive k-steps, each holding the
// tile's real parts then its imaginary parts.  Splitting re/im here is what
// lets the microkernel's inner loop be four plain FMAs over contiguous vectors;
// interleaved complex does not vectorize.  Short tiles are zero-padded so the
// microkernel never needs an edge case, and the zeros contribute exact zeros.

// Ap <- A[i0:i0+m, p0:p0+k] (at == 0), or its conjugate transpose taken from a
// source stored the other way round (at != 0), which is what the backward
// triangular solve against L^H needs.  MR-row panels.
static void pack_a(const double complex * A, long lda, long i0, long p0,
                   long m, long k, int at, double * Ap) {
  double * d = Ap;
  for (long ii = 0; ii < m; ii += MR)
    for (long p = 0; p < k; p ++) {
      for (long i = 0; i < MR; i ++) {
        double complex v = 0.0;
        if (ii + i < m)
          v = at ? conj(A[(p0 + p) * lda + (i0 + ii + i)])
                 :      A[(i0 + ii + i) * lda + (p0 + p)];
        d[i] = creal(v); d[MR + i] = cimag(v);
      }
      d += 2 * MR;
    }
}

// Bp <- B[p0:p0+k, j0:j0+n] (bt == 0), or its conjugate transpose taken from a
// source stored the other way round (bt != 0), which is what the Hermitian
// trailing update needs.  NR-column panels.
static void pack_b(const double complex * B, long ldb, long p0, long j0,
                   long k, long n, int bt, double * Bp) {
  double * d = Bp;
  for (long jj = 0; jj < n; jj += NR)
    for (long p = 0; p < k; p ++) {
      for (long j = 0; j < NR; j ++) {
        double complex v = 0.0;
        if (jj + j < n)
          v = bt ? conj(B[(j0 + jj + j) * ldb + (p0 + p)])
                 :      B[(p0 + p) * ldb + (j0 + jj + j)];
        d[j] = creal(v); d[NR + j] = cimag(v);
      }
      d += 2 * NR;
    }
}

// --- microkernel -----------------------------------------------------------
// C tile (mr x nr, row-major, leading dimension ldc) from k steps of one packed
// A panel against one packed B panel.  `mode` is 0 to store, 1 to add, 2 to
// subtract -- the factorizations want the trailing update C -= A B.
static void micro(long k, const double * ap, const double * bp,
                  double complex * C, long ldc, long mr, long nr, int mode) {
  vec_t cr[MR], ci[MR];
  for (long i = 0; i < MR; i ++) cr[i] = ci[i] = (vec_t) {0};
  for (long p = 0; p < k; p ++) {
    const double * a = ap + p * 2 * MR;
    const vec_t br = *(const vec_t *) (bp + p * 2 * NR);
    const vec_t bi = *(const vec_t *) (bp + p * 2 * NR + NR);
    for (long i = 0; i < MR; i ++) {           // constant trip count, unrolled
      cr[i] += a[i] * br - a[MR + i] * bi;
      ci[i] += a[i] * bi + a[MR + i] * br;
    }
  }
  // Staged through a scalar copy so the ragged edge below, whose bounds are
  // runtime values, never subscripts cr/ci dynamically -- one dynamic index
  // makes the whole array unpromotable and puts it back in memory.
  double crs[MR][NR] __attribute__((aligned(sizeof(vec_t))));
  double cis[MR][NR] __attribute__((aligned(sizeof(vec_t))));
  for (long i = 0; i < MR; i ++) {
    *(vec_t *) crs[i] = cr[i];
    *(vec_t *) cis[i] = ci[i];
  }
  for (long i = 0; i < mr; i ++)
    for (long j = 0; j < nr; j ++) {
      const double complex v = crs[i][j] + I * cis[i][j];
      double complex * c = &C[i * ldc + j];
      if      (mode == 0) *c  = v;
      else if (mode == 1) *c += v;
      else                *c -= v;
    }
}

// --- GEMM core (row-major) -------------------------------------------------
// C (M x N) from A (M x K) and B (K x N); at / bt take the conjugate transpose
// of that operand instead, its source then being stored the other way round.
// mode as in micro(); a K loop past its first block always accumulates,
// whatever the caller asked for the first one.
static void gemm_core(long M, long N, long K,
                      const double complex * A, long lda, int at,
                      const double complex * B, long ldb, int bt,
                      double complex * C, long ldc, int mode) {
  if (M <= 0 || N <= 0) return;
  if (K <= 0) {                                  // nothing to accumulate
    if (mode == 0)
      for (long i = 0; i < M; i ++)
        for (long j = 0; j < N; j ++) C[i * ldc + j] = 0.0;
    return;
  }
  const long mc = dmin(M, (long) MC), nc = dmin(N, (long) NC), kc = dmin(K, (long) KC);
  // Ap's length is rounded to a whole vector so that Bp, which follows it in
  // the same block, keeps the 64-byte alignment alloc_try hands out: the
  // microkernel reads the B panel as vec_t, and every offset into it is a
  // multiple of NR.
  const long apn = roundup(roundup(mc, MR) * kc * 2, (long) NR);
  const long bpn = roundup(nc, NR) * kc * 2;
  double * buf = malloc(sizeof(double) * (apn + bpn));
  double * Ap = buf, * Bp = buf + apn;
  for (long jc = 0; jc < N; jc += NC) {
    const long nb = dmin((long) NC, N - jc);
    for (long pc = 0; pc < K; pc += KC) {
      const long kb = dmin((long) KC, K - pc);
      const int  md = (pc == 0) ? mode : (mode == 2 ? 2 : 1);
      pack_b(B, ldb, pc, jc, kb, nb, bt, Bp);
      for (long ic = 0; ic < M; ic += MC) {
        const long mb = dmin((long) MC, M - ic);
        pack_a(A, lda, ic, pc, mb, kb, at, Ap);
        for (long jr = 0; jr < nb; jr += NR)
          for (long ir = 0; ir < mb; ir += MR)
            micro(kb, Ap + (ir / MR) * kb * 2 * MR,
                  Bp + (jr / NR) * kb * 2 * NR,
                  C + (ic + ir) * ldc + (jc + jr), ldc,
                  dmin((long) MR, mb - ir), dmin((long) NR, nb - jr), md);
      }
    }
  }
  free(buf);
}

void dense_gemm(long M, long N, long K,
                const double complex * A, long lda,
                const double complex * B, long ldb,
                double complex * C, long ldc) {
  gemm_core(M, N, K, A, lda, 0, B, ldb, 0, C, ldc, 0);
}

// Column-major C -= A B.  Column-major C = A B is row-major C^T = B^T A^T over
// the very same buffers, so this is one argument swap, not a second kernel.
static void gemm_cm_sub(long M, long N, long K,
                        const double complex * A, long lda,
                        const double complex * B, long ldb,
                        double complex * C, long ldc) {
  gemm_core(N, M, K, B, ldb, 0, A, lda, 0, C, ldc, 2);
}

// --- LU, column-major ------------------------------------------------------

// Solve L Y = B in place: L unit lower jb x jb, B jb x nrhs, both column-major.
static void trsm_l_unit(const double complex * L, long ldl, long jb,
                        double complex * B, long ldb, long nrhs) {
  for (long c = 0; c < nrhs; c ++)
    for (long k = 0; k < jb; k ++) {
      const double complex t = B[k + c * ldb];
      if (t != 0.0)
        for (long i = k + 1; i < jb; i ++)
          B[i + c * ldb] -= L[i + k * ldl] * t;
    }
}

// Solve U X = B in place: U upper non-unit jb x jb, both column-major.
static void trsm_u(const double complex * U, long ldu, long jb,
                   double complex * B, long ldb, long nrhs) {
  for (long c = 0; c < nrhs; c ++)
    for (long k = jb - 1; k >= 0; k --) {
      const double complex t = B[k + c * ldb] / U[k + k * ldu];
      B[k + c * ldb] = t;
      if (t != 0.0)
        for (long i = 0; i < k; i ++)
          B[i + c * ldb] -= U[i + k * ldu] * t;
    }
}

static void swap_rows_cm(double complex * A, long lda, long r1, long r2,
                         long c0, long c1) {
  if (r1 == r2) return;
  for (long c = c0; c < c1; c ++) {
    const double complex t = A[r1 + c * lda];
    A[r1 + c * lda] = A[r2 + c * lda];
    A[r2 + c * lda] = t;
  }
}

int dense_getrf(double complex * A, long n, int * ipiv) {
  int info = 0;
  for (long j = 0; j < n; j += NB) {
    const long jb = dmin((long) NB, n - j);
    // Unblocked LU of the panel A[j:n, j:j+jb] -- right-looking within the
    // panel, so the pivot search always sees fully updated entries.
    for (long k = j; k < j + jb; k ++) {
      long p = k;
      double best = fabs(creal(A[k + k * n])) + fabs(cimag(A[k + k * n]));
      for (long i = k + 1; i < n; i ++) {
        const double v = fabs(creal(A[i + k * n])) + fabs(cimag(A[i + k * n]));
        if (v > best) { best = v; p = i; }
      }
      ipiv[k] = (int) p;
      if (A[p + k * n] == 0.0) { if (!info) info = (int) (k + 1); continue; }
      swap_rows_cm(A, n, k, p, j, j + jb);
      const double complex r = 1.0 / A[k + k * n];
      for (long i = k + 1; i < n; i ++) A[i + k * n] *= r;
      for (long c = k + 1; c < j + jb; c ++) {
        const double complex t = A[k + c * n];
        if (t != 0.0)
          for (long i = k + 1; i < n; i ++) A[i + c * n] -= A[i + k * n] * t;
      }
    }
    for (long k = j; k < j + jb; k ++) {         // swaps outside the panel
      swap_rows_cm(A, n, k, ipiv[k], 0, j);
      swap_rows_cm(A, n, k, ipiv[k], j + jb, n);
    }
    if (j + jb < n) {
      trsm_l_unit(A + j + j * n, n, jb, A + j + (j + jb) * n, n, n - j - jb);
      gemm_cm_sub(n - j - jb, n - j - jb, jb, A + (j + jb) + j * n, n,
                  A + j + (j + jb) * n, n, A + (j + jb) + (j + jb) * n, n);
    }
  }
  return info;
}

void dense_getrs(const double complex * A, long n, const int * ipiv,
                 double complex * B, long nrhs) {
  for (long k = 0; k < n; k ++) swap_rows_cm(B, n, k, ipiv[k], 0, nrhs);
  for (long j = 0; j < n; j += NB) {             // L Y = P B
    const long jb = dmin((long) NB, n - j);
    trsm_l_unit(A + j + j * n, n, jb, B + j, n, nrhs);
    if (j + jb < n)
      gemm_cm_sub(n - j - jb, nrhs, jb, A + (j + jb) + j * n, n, B + j, n,
                  B + (j + jb), n);
  }
  for (long j = n; j > 0; ) {                    // U X = Y
    const long jb = dmin((long) NB, j);
    j -= jb;
    trsm_u(A + j + j * n, n, jb, B + j, n, nrhs);
    if (j > 0)
      gemm_cm_sub(j, nrhs, jb, A + j * n, n, B + j, n, B, n);
  }
}

// --- Bunch-Kaufman, row-major lower ----------------------------------------
#define ALO(i, j) A[(long) (i) * n + (long) (j)]

static inline double cabs1(double complex z) {
  return fabs(creal(z)) + fabs(cimag(z));
}

// Symmetric interchange of rows/columns kk and kp (kk < kp).
//
// The finished columns to the left of kk are permuted along with the trailing
// submatrix, which is the one place this departs from LAPACK's zhetf2.  LAPACK
// leaves them alone, so its stored factor is a *product* of permutations and
// unit lower triangular matrices and its solve has to replay the interchanges
// one pivot at a time -- inherently level-2.  Permuting everything instead
// gives the textbook form P A P^H = L D L^H with a single unit lower
// triangular L, so the inverse becomes: one blocked triangular solve, D, one
// more, and then this same interchange sequence run backwards over the result.
// That is what makes dense_hetri level-3.  It costs O(n^2) of extra swapping
// over the whole factorization.
//
// The reflected leg of the interchange picks up a conjugation, which is the
// only place this differs from the real symmetric case.
static void swap_sym(double complex * A, long n, long kk, long kp) {
  if (kk == kp) return;
  for (long j = 0; j < kk; j ++) {
    const double complex t = ALO(kk, j);
    ALO(kk, j) = ALO(kp, j);
    ALO(kp, j) = t;
  }
  for (long j = kk + 1; j < kp; j ++) {
    const double complex t = ALO(j, kk);
    ALO(j, kk) = conj(ALO(kp, j));
    ALO(kp, j) = conj(t);
  }
  ALO(kp, kk) = conj(ALO(kp, kk));
  const double r = creal(ALO(kk, kk));
  ALO(kk, kk) = creal(ALO(kp, kp));
  ALO(kp, kp) = r;
  for (long i = kp + 1; i < n; i ++) {
    const double complex t = ALO(i, kk);
    ALO(i, kk) = ALO(i, kp);
    ALO(i, kp) = t;
  }
}

int dense_hetrf(double complex * A, long n, int * ipiv) {
  const double alpha = (1.0 + sqrt(17.0)) / 8.0;
  int info = 0;
  // Two columns of scratch: the trailing update reads the pivot column(s) as
  // they were before the multipliers overwrite them, and our contiguous
  // direction is the row, so the update loops i outer / j inner and needs
  // every multiplier up front rather than one column at a time.
  double complex * wk = malloc(sizeof(double complex) * 2 * (size_t) n);
  double complex * wk1 = wk + n;
  long k = 0;
  while (k < n) {
    long kstep = 1, kp, imax = 0;
    const double absakk = fabs(creal(ALO(k, k)));
    double colmax = 0.0;
    for (long i = k + 1; i < n; i ++) {
      const double v = cabs1(ALO(i, k));
      if (v > colmax) { colmax = v; imax = i; }
    }
    if (absakk == 0.0 && colmax == 0.0) {
      kp = k;
      if (!info) info = (int) (k + 1);
      ALO(k, k) = creal(ALO(k, k));
    } else if (absakk >= alpha * colmax) {
      kp = k;
    } else {
      double rowmax = 0.0;
      for (long j = k; j < imax; j ++) {
        const double v = cabs1(ALO(imax, j));
        if (v > rowmax) rowmax = v;
      }
      for (long i = imax + 1; i < n; i ++) {
        const double v = cabs1(ALO(i, imax));
        if (v > rowmax) rowmax = v;
      }
      if (absakk * rowmax >= alpha * colmax * colmax)      kp = k;
      else if (fabs(creal(ALO(imax, imax))) >= alpha * rowmax) kp = imax;
      else { kp = imax; kstep = 2; }
    }
    const long kk = k + kstep - 1;
    // For a 2x2 pivot kk is k+1, so column k is one of the "left" columns
    // swap_sym now handles -- no separate fixup for the pivot block.
    if (kp != kk) {
      swap_sym(A, n, kk, kp);
      if (kstep == 2) ALO(k, k) = creal(ALO(k, k));
    }

    if (kstep == 1) {
      // A22 -= r * w w^H, then w *= r.  Update before scaling so the unscaled
      // column is still in place, exactly as zher/zdscal are ordered in zhetf2.
      const double d = creal(ALO(k, k));
      if (d != 0.0) {
        const double r = 1.0 / d;
        for (long i = k + 1; i < n; i ++) wk[i] = conj(ALO(i, k));
        for (long i = k + 1; i < n; i ++) {
          const double complex s = r * ALO(i, k);
          for (long j = k + 1; j <= i; j ++) ALO(i, j) -= s * wk[j];
          ALO(i, i) = creal(ALO(i, i));
        }
        for (long i = k + 1; i < n; i ++) ALO(i, k) *= r;
      } else if (!info) {
        info = (int) (k + 1);
      }
      ipiv[k] = (int) kp;
    } else {
      // 2x2 pivot block E = [[a, conj(b)], [b, d]], with a and d real.
      // The multipliers are L = C E^-1 where C is the (n-k-2) x 2 block below
      // the pivot, so each row of C multiplies E^-1 from the left:
      //     L_j0 = ( d*C_j0 - b*C_j1       ) / det
      //     L_j1 = ( a*C_j1 - conj(b)*C_j0 ) / det
      // Note which of the two off-diagonal entries is conjugated.  The D solve
      // in hetrs() below applies the same E^-1 to a column vector instead, and
      // so conjugates the other one; getting these two the same way round is
      // the classic error, and it is what this file got wrong first time.
      const double a = creal(ALO(k, k)), d = creal(ALO(k + 1, k + 1));
      const double complex b = ALO(k + 1, k);
      const double det = a * d - creal(b * conj(b));
      for (long j = k + 2; j < n; j ++) {
        const double complex c0 = ALO(j, k), c1 = ALO(j, k + 1);
        wk[j]  = (d * c0 -       b  * c1) / det;
        wk1[j] = (a * c1 - conj(b) * c0) / det;
      }
      for (long i = k + 2; i < n; i ++) {
        const double complex a0 = ALO(i, k), a1 = ALO(i, k + 1);
        for (long j = k + 2; j <= i; j ++)
          ALO(i, j) -= a0 * conj(wk[j]) + a1 * conj(wk1[j]);
        ALO(i, i) = creal(ALO(i, i));
      }
      for (long j = k + 2; j < n; j ++) {
        ALO(j, k) = wk[j]; ALO(j, k + 1) = wk1[j];
      }
      ipiv[k] = ipiv[k + 1] = (int) (-(kp + 1));
    }
    k += kstep;
  }
  free(wk);
  return info;
}

// Lower triangle of (L D L^H)^{-1} from dense_hetrf's factors, into X, which
// enters as the identity (n x n row-major, rows are the matrix index).
//
// Blocked, and that is the whole point: an unblocked version of this was 10x
// slower than MKL's zhetri and cost more than the n^3 GEMM beside it, because
// level-2 work does not care how few flops it has.
//
// Each of the two triangular solves is then bounded to the columns that can be
// nonzero, and those two bounds together are a factor of three.  L^{-1} is unit
// lower triangular, so the forward solve never reaches past the panel it is in;
// and only the lower triangle of the Hermitian result is promised, so neither
// does the backward one.  That is n^3/3 -- what LAPACK's zhetri spends, except
// level-3 rather than level-2.
//
// It is also why the identity enters *unpermuted*.  Solving against Q I instead
// would scatter the zeros and leave no triangle to exploit, so the interchanges
// come off the front here and go back on symmetrically in dense_hetri.
static void hetri_core(const double complex * A, long n, const int * ipiv,
                       double complex * X) {
  const long ldb = n;
  #define BRW(i) (X + (long) (i) * ldb)
  // Which indices are the second row of a 2x2 pivot.  Two things depend on it,
  // and both are silent if missed: the stored A[k+1][k] of a 2x2 block is D's
  // off-diagonal, *not* an L entry (L has an implicit zero there), and a block
  // boundary drawn between the two rows of a pivot would hand that same entry
  // to the trailing GEMM as though it were part of L.
  char * is2nd = calloc((size_t) n, 1);
  for (long k = 0; k < n; ) {
    if (ipiv[k] >= 0) k += 1;
    else { is2nd[k + 1] = 1; k += 2; }
  }
  #define LENT(i, k) (((i) == (k) + 1 && is2nd[i]) ? 0.0 : ALO(i, k))

  for (long j = 0; j < n; ) {                 // L Y = I, Y unit lower
    // NB wide, pulled back by one if that boundary would split a 2x2 pivot.
    long jb = dmin((long) NB, n - j);
    if (j + jb < n && is2nd[j + jb]) jb --;
    const long nc = j + jb;                   // past the panel Y is still zero
    for (long k = j; k < j + jb; k ++)
      for (long i = k + 1; i < j + jb; i ++) {
        const double complex l = LENT(i, k);
        if (l != 0.0)
          for (long c = 0; c < nc; c ++) BRW(i)[c] -= l * BRW(k)[c];
      }
    if (j + jb < n)
      gemm_core(n - j - jb, nc, jb, &ALO(j + jb, j), n, 0,
                BRW(j), ldb, 0, BRW(j + jb), ldb, 2);
    j += jb;
  }

  long k = 0;
  while (k < n) {                             // D Z = Y
    if (ipiv[k] >= 0) {
      const double r = 1.0 / creal(ALO(k, k));
      for (long c = 0; c <= k; c ++) BRW(k)[c] *= r;
      k += 1;
    } else {
      const double complex b = ALO(k + 1, k);
      const double a = creal(ALO(k, k)), d = creal(ALO(k + 1, k + 1));
      const double complex det = a * d - b * conj(b);
      for (long c = 0; c <= k + 1; c ++) {
        const double complex y0 = BRW(k)[c], y1 = BRW(k + 1)[c];
        BRW(k)[c]     = ( d * y0 - conj(b) * y1) / det;
        BRW(k + 1)[c] = (-b * y0 +      a  * y1) / det;
      }
      k += 2;
    }
  }

  for (long jend = n; jend > 0; ) {         // L^H X = Z, unit upper
    long j = jend - dmin((long) NB, jend);
    if (j > 0 && is2nd[j]) j --;            // same rule, from the other side
    const long jb = jend - j;
    const long nc = jend;                   // the strict upper is not promised
    if (jend < n)
      gemm_core(jb, nc, n - jend, &ALO(jend, j), n, 1,
                BRW(jend), ldb, 0, BRW(j), ldb, 2);
    for (long i = jend - 1; i >= j; i --)
      for (long p = i + 1; p < jend; p ++) {
        const double complex l = conj(LENT(p, i));
        if (l != 0.0)
          for (long c = 0; c < nc; c ++) BRW(i)[c] -= l * BRW(p)[c];
      }
    jend = j;
  }

  free(is2nd);
  #undef LENT
  #undef BRW
}

int dense_hetri(double complex * A, long n, const int * ipiv) {
  for (long k = 0; k < n; k ++)                  // a zero block pivot is fatal
    if (ipiv[k] >= 0 && creal(ALO(k, k)) == 0.0) return (int) (k + 1);
  // The factors are needed until the last inverse column is solved, so the
  // result cannot overwrite A as it goes.  n^2 fp64 complex is 9.4 MB at the
  // largest shape here (n = 768), freed immediately.
  double complex * X = calloc((size_t) n * n, sizeof(double complex));
  for (long i = 0; i < n; i ++) X[i * n + i] = 1.0;
  hetri_core(A, n, ipiv, X);
  // A^{-1} = Q^H (L D L^H)^{-1} Q, so the interchanges the factorization
  // applied go back on here, walked in reverse.  Each one is the very swap_sym
  // the factorization made, which is also why this reads the lower triangle
  // only.
  for (long k = n; k > 0; ) {
    if (ipiv[k - 1] >= 0) { swap_sym(X, n, k - 1, ipiv[k - 1]);      k -= 1; }
    else                  { swap_sym(X, n, k - 1, -ipiv[k - 2] - 1); k -= 2; }
  }
  memcpy(A, X, sizeof(double complex) * (size_t) n * n);
  free(X);
  return 0;
}

#undef ALO

// --- bottom-inverse apply --------------------------------------------------
// X = INV B, column-major, nrhs columns.  Blocked so the accumulator stays in
// L1 while the inverse streams past once: at nrhs = 16 and 64 rows that is
// 16 kB against an inverse that is tens of megabytes.
//
// The loop nest is the same shape as the GEMM microkernel, and for the same
// reason.  B is transposed up front into split re/im rows so the innermost loop
// runs over the nrhs lanes contiguously with the matrix entry broadcast --
// four plain FMAs over unit-stride vectors.  Accumulating into X directly
// instead puts the lane index on the outside and leaves the inner loop walking
// interleaved re/im pairs, which does not vectorize; that cost a factor of ~5.
#define DENSE_APPLY_BODY(T, R)                                                \
  enum { RB = 64 };                                                           \
  R * acc = malloc(sizeof(R) * 2 * (size_t) RB * (size_t) nrhs);              \
  R * bt  = malloc(sizeof(R) * 2 * (size_t) n_in * (size_t) nrhs);            \
  for (long c = 0; c < n_in; c ++)                                            \
    for (long l = 0; l < nrhs; l ++) {                                        \
      const T b = B[c + l * n_in];                                            \
      bt[2 * c * nrhs + l]        = (R) creal(b);                             \
      bt[2 * c * nrhs + nrhs + l] = (R) cimag(b);                             \
    }                                                                         \
  for (long r0 = 0; r0 < rows; r0 += RB) {                                    \
    const long rb = dmin((long) RB, rows - r0);                               \
    memset(acc, 0, sizeof(R) * 2 * (size_t) rb * (size_t) nrhs);              \
    for (long c = 0; c < n_in; c ++) {                                        \
      const R * bre = bt + 2 * c * nrhs, * bim = bre + nrhs;                  \
      const R * col = (const R *) (INV + r0 + c * ld);                        \
      for (long r = 0; r < rb; r ++) {                                        \
        const R ar = col[2 * r], ai = col[2 * r + 1];                         \
        R * a = acc + 2 * r * nrhs;                                           \
        for (long l = 0; l < nrhs; l ++) {          /* vector loop */         \
          a[l]        += ar * bre[l] - ai * bim[l];                           \
          a[nrhs + l] += ar * bim[l] + ai * bre[l];                           \
        }                                                                     \
      }                                                                       \
    }                                                                         \
    for (long l = 0; l < nrhs; l ++)                                          \
      for (long r = 0; r < rb; r ++)                                          \
        X[r0 + r + l * ld] = acc[2 * r * nrhs + l]                            \
                           + I * acc[2 * r * nrhs + nrhs + l];                \
  }                                                                           \
  free(acc); free(bt);

void dense_apply_d(const double complex * INV, long ld, long rows, long n_in,
                   long nrhs, const double complex * B, double complex * X) {
  DENSE_APPLY_BODY(double complex, double)
}

void dense_apply_f(const float complex * INV, long ld, long rows, long n_in,
                   long nrhs, const float complex * B, float complex * X) {
  DENSE_APPLY_BODY(float complex, float)
}

#undef DENSE_APPLY_BODY

#endif // DENSE_EXTERNAL
