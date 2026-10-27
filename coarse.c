// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include "coarse.h"
#include "coarse_kernel.h"   // coarse_dslash_site, coarse_off,
                             //   DC_KERNEL / DC_MR
#include "clover_ferm.h"     // fine operator D, used to build D_c
#include "utils.h"           // DOTC_KERNEL / CAXPY_KERNEL, complex.h
#include "comm.h"            // allreduce_sum_real (fgmres_coarse global
                             //   reductions)
#include "omp.h"             // omp_get_thread_num / omp_get_max_threads
// solver.h (red_black_schwarz, the SAP smoother) comes in via coarse.h

// Number of complex components (re/im pairs) one aggregate contributes per
// test vector: all block sites x 2 spins (of the tau spin-pair) x Nc colors.
#define agg_components (block_sites * 2 * Nc)

// Fill `comp[0..agg_components-1]` with the component offsets of aggregate
// (cl, tau): each entry is ((loc*Ns + d)*Nc + c)*2, i.e. the index into `nv`
// of this component's real part before the "* num_null_vecs + vec" stride.
// The imaginary part sits one stride (num_null_vecs) higher.
static void aggregate_components(int cl, int tau, long * comp) {
  int x0, y0, z0, t0;
  coarse_block_origin(cl, &x0, &y0, &z0, &t0);
  int m = 0;
  for (int t = t0; t < t0 + BLOCK_AGG; t ++)
    for (int z = z0; z < z0 + BLOCK_AGG; z ++)
      for (int y = y0; y < y0 + BLOCK_AGG; y ++)
        for (int x = x0; x < x0 + BLOCK_AGG; x ++) {
          int parity = (x ^ y ^ z ^ t) & 1;
          long loc = (long) parity * Vh + site_half(x / 2, y, z, t);
          for (int ds = 0; ds < 2; ds ++) {
            int d = 2 * tau + ds;            // spins {0,1} (tau=0) or {2,3}
            for (int c = 0; c < Nc; c ++)
              comp[m ++] = ((loc * Ns + d) * Nc + c) * 2;
          }
        }
}

// Complex inner product <v_i, v_j> = sum_k conj(v_i)_k (v_j)_k over the
// components listed in `comp`, returned via *pr (real) and *pi (imag).
static inline void agg_dot(const REAL * nv, const long * comp, int i, int j,
                           REAL * pr, REAL * pi) {
  // accumulate in double even in a single-precision build: this is setup code
  // (run once), and the reduced summation error gives a cleaner basis.
  double sr = 0, si = 0;
  for (int k = 0; k < agg_components; k ++) {
    long re_i = comp[k] * num_null_vecs + i, im_i = re_i + num_null_vecs;
    long re_j = comp[k] * num_null_vecs + j, im_j = re_j + num_null_vecs;
    sr += (double) nv[re_i] * nv[re_j] + (double) nv[im_i] * nv[im_j];
    si += (double) nv[re_i] * nv[im_j] - (double) nv[im_i] * nv[re_j];
  }
  *pr = (REAL) sr; *pi = (REAL) si;
}

// Complex dot <v_i, v_j> over a component-major aggregate scratch: R/I hold
// the re/im parts as N contiguous rows of M reals (row i = null-vector i's M
// aggregate components).  Columns i,j are then unit-stride length-M rows, so
// the reduction vectorizes with packed loads (no gather).  Accumulates in
// double, like agg_dot.
static inline void agg_dot_packed(const REAL * Rb, const REAL * Ib, int M,
                                  int i, int j, REAL * pr, REAL * pi) {
  const REAL * ri = Rb + (long) i * M, * ii = Ib + (long) i * M;
  const REAL * rj = Rb + (long) j * M, * ij = Ib + (long) j * M;
  double sr = 0, si = 0;
  for (int k = 0; k < M; k ++) {
    sr += (double) ri[k] * rj[k] + (double) ii[k] * ij[k];
    si += (double) ri[k] * ij[k] - (double) ii[k] * rj[k];
  }
  *pr = (REAL) sr; *pi = (REAL) si;
}

// --- per-aggregate Cholesky-QR ------------------------------------------------
// Same factorization as the global bootstrap pass (bootstrap.c): the Cholesky
// factor R of the aggregate Gram matrix G = V^H V is exactly the triangular
// factor classical GS produces (positive real diagonal), so V <- V R^{-1}
// yields an orthonormal basis, and both the Gram outer-product and the
// R^{-1} apply run unit-stride over the vector index -- i.e. directly in the
// native vec-fastest nv/nvc layout, with no gather/scatter transpose.
//
// Conditioning: Gram/Cholesky/R^{-1} are done in double, which is sound only
// up to cond(V) ~ 1e7 within the aggregate; squaring the condition number in
// G is what costs the accuracy, and one pass is not enough when the
// aggregate's columns are near-dependent -- which can happen even where the
// global basis is well conditioned.  So min/max of R's diagonal (a collapse
// proxy) below AGG_CHOLQR_RATIO_TOL triggers a second pass (CholeskyQR2),
// and a collapsed pivot falls back to MGS, which fills a fresh orthonormal
// direction instead.
#define AGG_CHOLQR_RATIO_TOL 1e-3
// A pivot d (squared norm of the orthogonal residual of a column) is a
// breakdown when it collapses relative to that column's original squared norm
// g0 -- the column is (near-)dependent on the earlier ones.  Below this ratio
// the direction carries no independent content, so completing the QR would
// only amplify roundoff (or, when d has gone slightly negative, produce NaN).
// Falls to the MGS path, which replaces such a column with an orthonormal
// complement so the aggregate basis stays full rank.  The threshold sits in
// the gap between a legitimately present (if ill-conditioned) column,
// d/g0 >= ~1e-6, and a numerically dependent one, which -- with the vectors
// stored in single precision -- collapses only to d/g0 ~ 1e-14 (float eps
// squared).  Well-conditioned aggregates have d ~ g0 and never trip this,
// so uniform hierarchies stay bit-identical.
#define AGG_CHOLQR_BREAK 1e-10

// -ffast-math (-ffinite-math-only) lets the compiler assume no NaN/Inf, so
// `d > 0` / isnan(d) are unreliable for detecting a collapsed pivot -- it
// folds `!(d > 0)` to `d <= 0`, which is false for a NaN d and lets the
// poison through.  Test the IEEE sign/exponent bits directly (memcpy is not
// an FP op, so the assumption cannot touch it): finite, nonzero, positive.
static inline int agg_finite_pos_d(double x) {
  unsigned long long u; memcpy(&u, &x, 8);
  return (u & 0x8000000000000000ull) == 0
      && (u & 0x7FF0000000000000ull) != 0x7FF0000000000000ull
      && (u & 0x7FFFFFFFFFFFFFFFull) != 0;
}
static inline int agg_finite_pos_r(REAL x) {
  if (sizeof(REAL) == 4) {
    unsigned int u; memcpy(&u, &x, 4);
    return (u & 0x80000000u) == 0 && (u & 0x7F800000u) != 0x7F800000u
        && (u & 0x7FFFFFFFu) != 0;
  } else {
    unsigned long long u; memcpy(&u, &x, 8);
    return (u & 0x8000000000000000ull) == 0
        && (u & 0x7FF0000000000000ull) != 0x7FF0000000000000ull
        && (u & 0x7FFFFFFFFFFFFFFFull) != 0;
  }
}

// Overwrite the packed column j (component-major rows Rb/Ib, M components
// each, columns 0..j-1 already orthonormal) with a unit vector orthogonal to
// all of them: start from a canonical direction e_c, project out 0..j-1,
// normalize; if e_c lands mostly inside the existing span, advance the
// cursor and retry.  The span is only j-dimensional in M >> j components, so
// a fresh direction is found almost immediately.  This is the "inert" surplus
// column: it carries no near-null content but keeps P full rank, so the
// coarse on-site operator block stays invertible (a zero column would make
// it singular).
static void agg_fill_orthonormal(REAL * Rb, REAL * Ib, int M, int j, int * cursor) {
  REAL * rj = Rb + (long) j * M, * ij = Ib + (long) j * M;
  for (int attempt = 0; attempt < M; attempt ++) {
    int c = *cursor; *cursor = (*cursor + 1) % M;
    for (int k = 0; k < M; k ++) { rj[k] = 0; ij[k] = 0; }
    rj[c] = 1;                               // e_c (real part) at component c
    for (int i = 0; i < j; i ++) {           // project out v_0..v_{j-1}
      const REAL * ri = Rb + (long) i * M, * ii = Ib + (long) i * M;
      REAL pr = ri[c], pi = -ii[c];          // <v_i, e_c> = conj(v_i[c])
      for (int k = 0; k < M; k ++) {
        REAL ar = ri[k], ai = ii[k];
        rj[k] -= pr * ar - pi * ai;
        ij[k] -= pr * ai + pi * ar;
      }
    }
    double nr = 0.0;
    for (int k = 0; k < M; k ++) nr += (double) rj[k] * rj[k] + (double) ij[k] * ij[k];
    if (nr > 0.25) {                         // enough weight outside the span
      REAL inv = (REAL) (1.0 / sqrt(nr));
      for (int k = 0; k < M; k ++) { rj[k] *= inv; ij[k] *= inv; }
      return;
    }
  }
}

// One CholeskyQR pass over the aggregate whose M component real-part offsets
// are listed in `base` (imag part num_null_vecs above each).  Returns 0 on
// Cholesky breakdown (*diag_ratio untouched); serial -- parallelism is over
// aggregates in the callers.
static int agg_cholqr_pass(REAL * nv, const long * base, int M, int N,
                           double * diag_ratio) {
  // N = number of columns (test vectors) to orthonormalize per aggregate:
  // nv0 for the fine (L0->L1) transition, the per-transition count for coarse
  // levels.  Gram matrix G = V^H V, upper triangle, double accumulation.
  // The N x N work arrays are on the heap: this runs on worker threads, whose
  // stacks a few hundred columns would overflow.
  double (*G)[N][N] = malloc(sizeof(double[4][N][N]));  // Gr, Gi, Rr, Ri
  double (*Gr)[N] = G[0], (*Gi)[N] = G[1];
  memset(G, 0, sizeof(double[2][N][N]));
  for (int k = 0; k < M; k ++) {
    const REAL * ar = nv + base[k], * ai = ar + N;
    for (int i = 0; i < N; i ++) {
      double xr = ar[i], xi = ai[i];
      for (int j = i; j < N; j ++) {
        // Re <V[i],V[j]>
        Gr[i][j] += xr * (double) ar[j] + xi * (double) ai[j];
        // Im <V[i],V[j]>
        Gi[i][j] += xr * (double) ai[j] - xi * (double) ar[j];
      }
    }
  }

  // Cholesky G = R^H R (R upper triangular, real positive diagonal)
  double (*Rr)[N] = G[2], (*Ri)[N] = G[3];
  double dmin = 0.0, dmax = 0.0;
  for (int i = 0; i < N; i ++) {
    double g0 = Gr[i][i];                 // original squared norm of column i
    double d = g0;
    for (int k = 0; k < i; k ++) d -= Rr[k][i] * Rr[k][i] + Ri[k][i] * Ri[k][i];
    // Breakdown: non-finite/non-positive (bit test, robust under
    // -ffast-math), or collapsed relative to g0 (near-dependent).
    // Either way fall back to MGS, which fills a fresh orthonormal direction.
    // The bit test short-circuits before the relative compare,
    // so d is finite there.
    if (!agg_finite_pos_d(d) || d < AGG_CHOLQR_BREAK * g0) {
      free(G);
      return 0;
    }
    double rii = sqrt(d);
    if (i == 0) { dmin = dmax = rii; }
    else        { if (rii < dmin) dmin = rii; if (rii > dmax) dmax = rii; }
    Rr[i][i] = rii; Ri[i][i] = 0.0;
    double inv = 1.0 / rii;
    for (int j = i + 1; j < N; j ++) {
      double sr = Gr[i][j], si = Gi[i][j];
      for (int k = 0; k < i; k ++) {       // -= conj(R_ki) R_kj
        sr -= Rr[k][i] * Rr[k][j] + Ri[k][i] * Ri[k][j];
        si -= Rr[k][i] * Ri[k][j] - Ri[k][i] * Rr[k][j];
      }
      Rr[i][j] = sr * inv; Ri[i][j] = si * inv;
    }
  }
  *diag_ratio = dmin / dmax;

  // W = R^{-1} (upper triangular back-substitution, row-wise)
  REAL (*W)[N][N] = malloc(sizeof(REAL[2][N][N]));
  REAL (*Wr)[N] = W[0], (*Wi)[N] = W[1];
  for (int i = 0; i < N; i ++) {
    double wr[MAX_NULL_VECS], wi[MAX_NULL_VECS];
    wr[i] = 1.0 / Rr[i][i]; wi[i] = 0.0;
    for (int j = i + 1; j < N; j ++) {
      double sr = 0.0, si = 0.0;
      for (int k = i; k < j; k ++) {       // sum W_ik R_kj
        sr += wr[k] * Rr[k][j] - wi[k] * Ri[k][j];
        si += wr[k] * Ri[k][j] + wi[k] * Rr[k][j];
      }
      double inv = 1.0 / Rr[j][j];
      wr[j] = -sr * inv; wi[j] = -si * inv;
    }
    for (int j = 0; j < i; j ++) {
      Wr[i][j] = 0;
      Wi[i][j] = 0;
    }
    for (int j = i; j < N; j ++) {
      Wr[i][j] = (REAL) wr[j];
      Wi[i][j] = (REAL) wi[j];
    }
  }

  // V <- V W, per component row (1xN GEMV, unit-stride over j)
  for (int k = 0; k < M; k ++) {
    REAL * ar = nv + base[k], * ai = ar + N;
    REAL xr[MAX_NULL_VECS], xi[MAX_NULL_VECS];
    for (int i = 0; i < N; i ++) {
      xr[i] = ar[i];
      xi[i] = ai[i];
      ar[i] = 0;
      ai[i] = 0;
    }
    for (int c = 0; c < N; c ++) {
      REAL br = xr[c], bi = xi[c];
      for (int j = c; j < N; j ++) {
        ar[j] += br * Wr[c][j] - bi * Wi[c][j];
        ai[j] += br * Wi[c][j] + bi * Wr[c][j];
      }
    }
  }
  free(G); free(W);
  return 1;
}

// MGS fallback for one aggregate, used only on Cholesky breakdown.  In the
// native layout the null-vector index is the fastest index, so a
// fixed-vector reduction strides over components and the vectorizer is
// forced to gather -- so gather the aggregate once into contiguous
// component-major scratch (Rb/Ib, row = one null vector over its M
// components), run the MGS there with unit-stride packed loads, and scatter
// back.  *Rbp/*Ibp = the caller's per-thread scratch (N*M REALs each),
// allocated lazily here since breakdowns are rare.
static void agg_mgs(REAL * nv, const long * base, int M, int N,
                    REAL ** Rbp, REAL ** Ibp) {
  if (!*Rbp) {
    *Rbp = malloc(sizeof(REAL) * (long) N * M);
    *Ibp = malloc(sizeof(REAL) * (long) N * M);
  }
  REAL * Rb = *Rbp, * Ib = *Ibp;

  // gather: each component's num_null_vecs columns are contiguous in nv;
  // transpose into component-major rows
  for (int k = 0; k < M; k ++) {
    long b = base[k];
    for (int i = 0; i < N; i ++) {
      Rb[(long) i * M + k] = nv[b + i];
      Ib[(long) i * M + k] = nv[b + N + i];
    }
  }

  // modified Gram-Schmidt over the null-vector index, now unit-stride
  int fill_cursor = 0;
  for (int j = 0; j < N; j ++) {
    REAL * rj = Rb + (long) j * M, * ij = Ib + (long) j * M;
    REAL g0r, g0i;
    agg_dot_packed(Rb, Ib, M, j, j, &g0r, &g0i);   // original squared norm
    for (int i = 0; i < j; i ++) {
      REAL pr, pi;
      agg_dot_packed(Rb, Ib, M, i, j, &pr, &pi);   // pr + i*pi = <v_i, v_j>
      const REAL * ri = Rb + (long) i * M, * ii = Ib + (long) i * M;
      // v_j <- v_j - <v_i,v_j> v_i   (v_i already normalized)
      for (int k = 0; k < M; k ++) {
        REAL ar = ri[k], ai = ii[k];
        rj[k] -= pr * ar - pi * ai;
        ij[k] -= pr * ai + pi * ar;
      }
    }
    // normalize v_j, or -- if the residual has collapsed (near-dependent
    // column, bit test robust under -ffast-math) -- replace it with an
    // orthonormal complement so the aggregate basis stays full rank
    REAL nr, ni;
    agg_dot_packed(Rb, Ib, M, j, j, &nr, &ni);     // ni ~ 0
    if (!agg_finite_pos_r(nr) || nr < (REAL) AGG_CHOLQR_BREAK * g0r)
      agg_fill_orthonormal(Rb, Ib, M, j, &fill_cursor);
    else {
      REAL inv = (REAL) (1.0 / sqrt((double) nr));
      for (int k = 0; k < M; k ++) { rj[k] *= inv; ij[k] *= inv; }
    }
  }

  // scatter back into the native layout
  for (int k = 0; k < M; k ++) {
    long b = base[k];
    for (int i = 0; i < N; i ++) {
      nv[b + i]     = Rb[(long) i * M + k];
      nv[b + N + i] = Ib[(long) i * M + k];
    }
  }
}

// Orthonormalize one aggregate: CholeskyQR with adaptive second pass and MGS
// fallback.  *rmin tracks the smallest successful-pass diag ratio, *n2/*nfb
// count second passes and fallbacks (all per-thread reduction copies).
static void agg_orthonormalize(REAL * nv, const long * base, int M, int N,
                               REAL ** Rbp, REAL ** Ibp,
                               double * rmin, long * n2, long * nfb) {
  double ratio = 0.0;
  if (!agg_cholqr_pass(nv, base, M, N, &ratio)) {
    agg_mgs(nv, base, M, N, Rbp, Ibp); (*nfb) ++;
    return;
  }
  if (ratio < *rmin) *rmin = ratio;
  if (ratio < AGG_CHOLQR_RATIO_TOL) {
    (*n2) ++;
    if (!agg_cholqr_pass(nv, base, M, N, &ratio)) {
      agg_mgs(nv, base, M, N, Rbp, Ibp); (*nfb) ++;
    }
  }
}

// One summary line per orthonormalization call.  rmin sentinel 2.0 (ratios
// are always <= 1) means no aggregate completed a CholQR pass.
static void agg_ortho_report(const char * tag, double rmin, long n2, long nfb) {
  // Aggregates are rank-local (they never straddle a face), so the
  // orthonormalization itself needs no communication; only this diagnostic
  // is reduced so the single rank-0 line reflects the whole lattice (global
  // min ratio, total counts).  No-ops at MPI=0.  Called outside parallel
  // regions (FUNNELED-safe).
  allreduce_min_d(&rmin, 1);
  double cnt[2] = { (double) n2, (double) nfb };
  allreduce_sum_d(cnt, 2);
  n2 = (long) cnt[0]; nfb = (long) cnt[1];
  // Silent while healthy.  The ratio only matters as it approaches the
  // tolerance that triggers a second pass, so report it one decade out and
  // stay quiet above that -- a pass-2 or a fallback speaks for itself in the
  // counts below, and the sentinel rmin (no aggregate completed a pass at
  // all) always reports.
  if (rmin >= 10 * AGG_CHOLQR_RATIO_TOL && rmin < 2.0 && !n2 && !nfb) return;
  if (rmin < 2.0) print0("%s: cholqr min diag ratio %.2e", tag, rmin);
  else            print0("%s: cholqr (no aggregate completed a pass)", tag);
  if (n2)  print0("  pass-2 aggregates %ld", n2);
  if (nfb) print0("  mgs fallbacks %ld", nfb);
  print0("\n");
}

// Per-aggregate orthonormalization of the P columns via CholeskyQR
// (see above).  Aggregates (coarse site x spin-pair) are mutually
// independent, so the loop is a plain parallel-for;
// nullvec_orthonormality_error checks the result.
void local_orthonormalize(REAL * nv) {
  const int N = num_null_vecs, M = agg_components;
  double rmin = 2.0; long n2 = 0, nfb = 0;
  #pragma omp parallel reduction(min:rmin) reduction(+:n2,nfb)
  {
    REAL * Rb = NULL, * Ib = NULL;   // MGS-fallback scratch, lazily allocated
    #pragma omp for collapse(2) schedule(dynamic)
    for (int cl = 0; cl < Vc; cl ++)
      for (int tau = 0; tau < 2; tau ++) {
        long comp[agg_components], base[agg_components];
        aggregate_components(cl, tau, comp);
        // fold in the vec stride
        for (int k = 0; k < M; k ++) base[k] = comp[k] * N;
        agg_orthonormalize(nv, base, M, N, &Rb, &Ib, &rmin, &n2, &nfb);
      }
    free(Rb); free(Ib);
  }
  agg_ortho_report("local_ortho", rmin, n2, nfb);
}

// Index of the real part of coarse variable (cl, v) for right-hand side rhs.
// Imag part is one num_rhs stride higher.  coarse_idx uses the L1 macro width
// N_coarse; coarse_idx_lvl takes an explicit per-level width (nvar) for
// coarse fermions whose variable count differs from N_coarse
// (COARSE_NULL_VECS != nv0).
static inline long coarse_idx(int cl, int v, int rhs) {
  return ((long) (cl * N_coarse + v) * 2) * num_rhs + rhs;
}
static inline long coarse_idx_lvl(long cl, int v, int rhs, int nvar) {
  return ((cl * (long) nvar + v) * 2) * num_rhs + rhs;
}

// --- prolongation / restriction profiling ----------------------------------
// Flops per call, per fine (site, spin, color) x rhs -- P and R are not
// identical:
//   prolong P: fine = sum_j nv[j] * coarse[j], a reduction over the
//     num_null_vecs axis -- j=0 initializes the accumulator (complex
//     multiply, 6 flops) and j>0 accumulate (8-flop MAC), so
//     8*num_null_vecs - 2 (TRANSFER_FLOPS_P).
//   restrict R: coarse[j] = sum_fine conj(nv[j]) * fine, a reduction over the
//     fine sites -- every fine-site term is an 8-flop MAC and the single
//     init-onto-zero per coarse variable is a per-coarse (not per-fine)
//     saving, negligible, so 8*num_null_vecs (TRANSFER_FLOPS).
static double transfer_P_secs = 0, transfer_R_secs = 0;
static long   transfer_P_calls = 0, transfer_R_calls = 0;
// apply_R_even calls (eo outer solve): half-volume restricts, half the flops
// of a full call -- tracked so the GFlop/s rows stay honest.
static long   transfer_R_half_calls = 0;
// apply_P_par calls (fused first-color SAP prolong): half-volume prolongs,
// same bookkeeping -- the other half's flops run inside the SAP sweep and are
// added to the smoother's tally there.
static long   transfer_P_half_calls = 0;
#define TRANSFER_FLOPS \
  ((double) V * Ns * Nc * num_rhs * num_null_vecs * 8.0)
#define TRANSFER_FLOPS_P \
  ((double) V * Ns * Nc * num_rhs * (num_null_vecs * 8.0 - 2.0))

void transfer_stats_reset(void) {
  transfer_P_secs = transfer_R_secs = 0;
  transfer_P_calls = transfer_R_calls = 0;
  transfer_R_half_calls = 0;
  transfer_P_half_calls = 0;
}

// --- V-cycle phase profiling -----------------------------------------------
// Wall time spent in each phase of mg_vcycle, summed over all invocations.
// The phases a level can run are kept per level (indexed by level_t.depth,
// which is in scope wherever they accumulate), so the report can attribute
// them; a hierarchy of four or more levels otherwise sums its intermediate
// levels into one unreadable number.  Costs an array index next to a
// gettimeofday pair -- these are per-phase brackets, one per cycle, never
// per site.
static double vc_coarse = 0, vc_D = 0, vc_smooth = 0;
static long   vc_calls = 0, vc_citers = 0;
static double vc_R[NUM_LEVELS], vc_P[NUM_LEVELS], vc_vec[NUM_LEVELS];
// vc_smooth split by level: depth-0 (fine SAP) vs deeper (coarse SAP).
// The fine half stays a scalar -- there is only one depth 0 -- and the report
// prefers the SAP's own timer to this bracket anyway.
static double vc_smooth_fine = 0, vc_smooth_coarse[NUM_LEVELS];
// vc_vec_flops pairs with vc_vec (the eta-Dpsi / psi+=smooth loops at
// depth >= 1). (The depth-0 residual Dslash has no counter: it is folded
// into the fine SAP, so its time and flops land in vc_smooth_fine / the SAP
// row.)  The coarse-SAP flops that pair with vc_smooth_coarse live in
// coarse_sap.c -- coarse_sap_flops().
static double vc_vec_flops[NUM_LEVELS];
// Modeled depth>=1 transfer flops (accumulated in apply_{R,P}_coarse), paired
// with that level's vc_R + vc_P.
static double ctrans_flops[NUM_LEVELS];
// depth-0 mg_cycle invocations = top-level preconditioner cycles (one per
// fgmres Arnoldi step).  vc_calls counts mg_cycle at all levels, so it is
// larger -- a K-cycle's coarse correction runs several recursive
// mg_cycle(depth>=1) calls.
static long vc_calls_fine = 0;

void vcycle_stats_reset(void) {
  vc_coarse = vc_D = vc_smooth = vc_smooth_fine = 0;
  for (int d = 0; d < NUM_LEVELS; d ++) {
    vc_R[d] = vc_P[d] = vc_vec[d] = 0;
    vc_smooth_coarse[d] = vc_vec_flops[d] = ctrans_flops[d] = 0;
  }
  vc_calls = vc_citers = vc_calls_fine = 0;
  coarse_sap_flops_reset();
#if COARSE_SAP_PROFILE
  csap_profile_reset();     // stage timers -> solve-only, like the rows above
#endif
}
// Prolongate the single coarse site `cl` into its block of fine sites.
// Factored out of apply_P so build_coarse_op can prolong one block at a time
// (windowed construction) without re-walking the whole coarse lattice.
// nt selects the store instruction only (non-temporal vs cached) -- both call
// sites pass a constant, so each folds to straight-line stores.  The cached
// variant exists for the fused first-color SAP prolong (apply_P_block_cached),
// whose consumer reads the block back immediately.
static inline void apply_P_site_impl(REAL * fine_out, REAL * coarse_in,
                                     REAL * nv, int cl, int nt) {
  int x0, y0, z0, t0;
  coarse_block_origin(cl, &x0, &y0, &z0, &t0);
  for (int t = t0; t < t0 + BLOCK_AGG; t ++)
    for (int z = z0; z < z0 + BLOCK_AGG; z ++)
      for (int y = y0; y < y0 + BLOCK_AGG; y ++)
        for (int x = x0; x < x0 + BLOCK_AGG; x ++) {
          int parity = (x ^ y ^ z ^ t) & 1;
          long loc = (long) parity * Vh + site_half(x / 2, y, z, t);
          for (int d = 0; d < Ns; d ++) {
            int tau = d / 2;                       // spin aggregate
            // fine_out[fbase(c)] = sum_j nv[j] * coarse_in[v=nv*tau+j],
            // with the contraction over j as the outer loop and the
            // unit-stride rhs axis inside, so each step is a complex AXPY of
            // a broadcast nv scalar against a contiguous coarse vector.
            // The SIMD is written explicitly (REAL_VEC over num_rhs_vec
            // blocks, re then im) so it lives in the IR rather than
            // depending on the auto-vectorizer.
            //
            // The j-contraction accumulates in registers, not in fine_out,
            // and is then stored once (non-temporally at nt).
            // This (a) avoids re-reading/writing fine_out per j (without
            // restrict the compiler cannot prove out doesn't alias in/nv),
            // and (b) at nt skips the write-allocate RFO, since apply_P
            // overwrites fine_out and the line is write-only there.
            // NT stores are drained by the sfence in apply_P.
            //
            // The three colors of this spin are interleaved inside the j-loop
            // (they contract the same coarse vector against different nv
            // columns).  With num_rhs_vec == 1 a single (re,im) accumulator
            // pair is a serial 2-FMA-per-j dependency chain, so it runs at
            // FMA latency rather than throughput; 3 colors give 6 independent
            // FMA chains, enough issue parallelism to hide that latency, and
            // share each coarse_in load 3 ways.  Each (d,c) output still
            // accumulates its j-terms in the exact original order,
            // so results are unchanged.
            long fbase0 = ((loc * Ns + d) * Nc) * 2;      // c = 0; c stride 2
            long vbase0 = fbase0 * num_null_vecs;         // nv re index base
            REAL_VEC acc_re[Nc][num_rhs_vec], acc_im[Nc][num_rhs_vec];
            {   // j = 0 SETs the accumulators
              REAL_VEC * in_re = (REAL_VEC *) (coarse_in
                                    + coarse_idx(cl, num_null_vecs * tau, 0));
              REAL_VEC * in_im = in_re + num_rhs_vec;
              for (int c = 0; c < Nc; c ++) {
                REAL_VEC vr = SET1(nv[vbase0 + 2 * c * num_null_vecs]);
                REAL_VEC vi = SET1(nv[vbase0 + (2 * c + 1) * num_null_vecs]);
                for (int r = 0; r < num_rhs_vec; r ++) {
                  acc_re[c][r] = in_re[r] * vr - in_im[r] * vi;
                  acc_im[c][r] = in_im[r] * vr + in_re[r] * vi;
                }
              }
            }
            for (int j = 1; j < num_null_vecs; j ++) {
              REAL_VEC * in_re = (REAL_VEC *) (coarse_in
                                + coarse_idx(cl, num_null_vecs * tau + j, 0));
              REAL_VEC * in_im = in_re + num_rhs_vec;
              for (int c = 0; c < Nc; c ++) {
                REAL_VEC vr = SET1(nv[vbase0 + 2 * c * num_null_vecs + j]);
                REAL_VEC vi = SET1(nv[vbase0 + (2 * c + 1) * num_null_vecs
                                      + j]);
                for (int r = 0; r < num_rhs_vec; r ++) {
                  acc_re[c][r] += in_re[r] * vr - in_im[r] * vi;
                  acc_im[c][r] += in_im[r] * vr + in_re[r] * vi;
                }
              }
            }
            for (int c = 0; c < Nc; c ++) {
              REAL_VEC * out_re = (REAL_VEC *) (fine_out
                                                + (fbase0 + 2 * c) * num_rhs);
              REAL_VEC * out_im = out_re + num_rhs_vec;
              for (int r = 0; r < num_rhs_vec; r ++) {
                if (nt) {
                  NT_STORE(acc_re[c][r], &out_re[r]);
                  NT_STORE(acc_im[c][r], &out_im[r]);
                } else {
                  out_re[r] = acc_re[c][r];
                  out_im[r] = acc_im[c][r];
                }
              }
            }
          }
        }
}
static void apply_P_site(REAL * fine_out, REAL * coarse_in, REAL * nv,
                         int cl) {
  apply_P_site_impl(fine_out, coarse_in, nv, cl, 1);
}

// One aggregation block of fine_out = P * coarse_in, with cached stores: the
// fused first-color SAP prolong (see sap_fuse_prolong_next in solver.h) calls
// this at the top of a sweep-0 block visit, and the fold seeds read the block
// straight back from cache.  Accumulation arithmetic and store values are
// identical to apply_P_site -- only the store instruction differs -- so the
// prolonged field is bit-identical.  Block coords are in aggregation units
// (== SAP blocks; the registration gate guarantees the blockings coincide).
void apply_P_block_cached(REAL * fine_out, REAL * coarse_in, REAL * nv,
                          int bx, int by, int bz, int bt) {
  apply_P_site_impl(fine_out, coarse_in, nv,
                    ((bt * Lzc + bz) * Lyc + by) * Lxc + bx, 0);
}

void apply_P(REAL * fine_out, REAL * coarse_in, REAL * nv) {
  // Each coarse site owns a disjoint block of fine sites, so parallelizing
  // over coarse sites gives non-overlapping writes into fine_out.
  struct timeval t0, t1; gettimeofday(&t0, NULL);
  #pragma omp parallel
  {
    #pragma omp for schedule(dynamic) nowait
    for (int cl = 0; cl < Vc; cl ++)
      apply_P_site(fine_out, coarse_in, nv, cl);
    // apply_P_site streams fine_out with non-temporal stores; fence each
    // thread before the region-end barrier so the V-cycle consumer sees a
    // drained fine_out.
    NT_FENCE();
  }
  gettimeofday(&t1, NULL);
  transfer_P_secs += time_used(t0, t1); transfer_P_calls ++;
}

// apply_P restricted to aggregation blocks of one local block parity
// ((bx^by^bz^bt) & 1 == par).  The fused first-color SAP prolong bulk-prolongs
// only the smoother's second-processed block color with this (its blocks are
// read as neighbors by the first color's fold seeds, so they must be
// materialized up front); the first color is prolonged in-visit via
// apply_P_block_cached.  Counted as a half call in the transfer stats
// (transfer_P_half_calls, like apply_R_even); the in-visit half is added to
// the smoother's flop tally instead, so the books stay exact.
static void apply_P_par(REAL * fine_out, REAL * coarse_in, REAL * nv,
                        int par) {
  struct timeval t0, t1; gettimeofday(&t0, NULL);
  #pragma omp parallel
  {
    #pragma omp for schedule(dynamic) nowait
    for (int cl = 0; cl < Vc; cl ++) {
      int xb = cl % Lxc, rr = cl / Lxc;
      int yb = rr % Lyc; rr /= Lyc;
      int zb = rr % Lzc; rr /= Lzc;
      if (((xb ^ yb ^ zb ^ rr) & 1) == par)
        apply_P_site(fine_out, coarse_in, nv, cl);
    }
    NT_FENCE();
  }
  gettimeofday(&t1, NULL);
  transfer_P_secs += time_used(t0, t1);
  transfer_P_calls ++; transfer_P_half_calls ++;
}

// Prolong a batch of unit coarse probes onto block q of `fine_out`, where
// probe `s` (s = 0..nb-1) excites coarse variable (b0 + s) in rhs lane s.
// For a unit probe apply_P_site's num_null_vecs-deep interpolation collapses
// to a single term -- P*e is one chiral-masked null-vector column -- so write
// that column directly.  Bit-identical to apply_P_site(unit probe): its sum
// is 0 + .. + nv[j]*1 + .. + 0 = nv[j], and every non-matching lane/aggregate
// is exactly 0.  Overwrites all num_rhs lanes of block q (as apply_P_site
// CSETs them all); sites outside block q are untouched, preserving the
// "fine = 0 off-block" invariant the windowed D needs.
static void prolong_unit_block(REAL * fine_out, REAL * nv, int q, int b0,
                               int nb) {
  int x0, y0, z0, t0;
  coarse_block_origin(q, &x0, &y0, &z0, &t0);
  for (int t = t0; t < t0 + BLOCK_AGG; t ++)
    for (int z = z0; z < z0 + BLOCK_AGG; z ++)
      for (int y = y0; y < y0 + BLOCK_AGG; y ++)
        for (int x = x0; x < x0 + BLOCK_AGG; x ++) {
          int parity = (x ^ y ^ z ^ t) & 1;
          long loc = (long) parity * Vh + site_half(x / 2, y, z, t);
          for (int d = 0; d < Ns; d ++) {
            int tau = d / 2;                    // spin aggregate of this spin
            for (int c = 0; c < Nc; c ++) {
              long fbase = ((loc * Ns + d) * Nc + c) * 2;
              long vbase = fbase * num_null_vecs;
              REAL * out = fine_out + fbase * num_rhs;
              for (int k = 0; k < num_rhs; k ++) {
                int b = b0 + k;                   // coarse variable in lane k
                // nonzero only if that variable belongs to this spin aggregate
                if (k < nb && b / num_null_vecs == tau) {
                  int j = b % num_null_vecs;      // its null-vector column
                  out[k]           = nv[vbase + j];
                  out[k + num_rhs] = nv[vbase + num_null_vecs + j];
                } else {
                  out[k]           = 0;
                  out[k + num_rhs] = 0;
                }
              }
            }
          }
        }
}

// Restrict onto the single coarse site `cl` (reads its block of fine sites);
// also the per-site body of the windowed construction.  num_parities = 2:
// full restrict.  num_parities = 1: even sites only (eo outer solve, where
// fine_in is the even-site half-field whose odd part is identically zero) --
// bit-identical to the full restrict of the zero-padded field, since the
// skipped odd terms contribute exact zeros and even-site loc values index
// the half-field unchanged.
// One REAL_VEC of (REAL)((double) x * s) -- the promote-multiply-round the
// normalize applies per element, in explicit vector form.  Written with
// intrinsics (the only ISA-specific code in this file) for two reasons: the
// value reaching the staged NT store below must be a computation -- a plain
// tile->vnt copy loop gets loop-idiom-recognized as memcpy, which silently
// drops the non-temporal hint (verified in the disassembly; a computed-value
// store survives, apply_P_site_impl precedent) -- and the explicit converts
// pin the exact promote-multiply-round rounding independent of fast-math.
//
// The scale is per rhs, so the scalar broadcast becomes a lane-indexed vector.
// Which lanes a REAL_VEC covers follows from the layout (num_rhs reals then
// num_rhs imags per complex element, so lane = offset % num_rhs) and from
// veclen dividing num_rhs in every build: vector i spans lane block
// i % num_rhs_vec, whose coefficients are the veclen doubles at
// rinv + v*veclen.  So the whole per-lane cost is num_rhs_vec prebuilt
// constants, chosen by a mask -- and on AVX-512 + MG_SINGLE num_rhs_vec is 1,
// making the constant loop-invariant: one vector load hoisted out of the
// loop, replacing one broadcast hoisted out of the loop.  The MG_SINGLE
// variants promote in halves, so they take the matching half of the
// coefficient block for each.
#if AVX512
#if MG_SINGLE
typedef struct { __m512d lo, hi; } fuse_scale_t;
static inline fuse_scale_t fuse_scale_build(const double * rinv, int v) {
  fuse_scale_t s;
  s.lo = _mm512_loadu_pd(rinv + (long) v * veclen);
  s.hi = _mm512_loadu_pd(rinv + (long) v * veclen + 8);
  return s;
}
static inline REAL_VEC fuse_r_scale_vec(REAL_VEC x, fuse_scale_t s) {
  __m256 rlo = _mm512_cvtpd_ps(_mm512_mul_pd(
                   _mm512_cvtps_pd(_mm512_castps512_ps256(x)), s.lo));
  __m256 rhi = _mm512_cvtpd_ps(_mm512_mul_pd(
                   _mm512_cvtps_pd(_mm512_extractf32x8_ps(x, 1)), s.hi));
  return _mm512_insertf32x8(_mm512_castps256_ps512(rlo), rhi, 1);
}
#else
typedef struct { __m512d s; } fuse_scale_t;
static inline fuse_scale_t fuse_scale_build(const double * rinv, int v) {
  fuse_scale_t s; s.s = _mm512_loadu_pd(rinv + (long) v * veclen); return s;
}
static inline REAL_VEC fuse_r_scale_vec(REAL_VEC x, fuse_scale_t s) {
  return _mm512_mul_pd(x, s.s);
}
#endif
#elif AVX2
#if MG_SINGLE
typedef struct { __m256d lo, hi; } fuse_scale_t;
static inline fuse_scale_t fuse_scale_build(const double * rinv, int v) {
  fuse_scale_t s;
  s.lo = _mm256_loadu_pd(rinv + (long) v * veclen);
  s.hi = _mm256_loadu_pd(rinv + (long) v * veclen + 4);
  return s;
}
static inline REAL_VEC fuse_r_scale_vec(REAL_VEC x, fuse_scale_t s) {
  __m128 rlo = _mm256_cvtpd_ps(_mm256_mul_pd(
                   _mm256_cvtps_pd(_mm256_castps256_ps128(x)), s.lo));
  __m128 rhi = _mm256_cvtpd_ps(_mm256_mul_pd(
                   _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1)), s.hi));
  return _mm256_set_m128(rhi, rlo);
}
#else
typedef struct { __m256d s; } fuse_scale_t;
static inline fuse_scale_t fuse_scale_build(const double * rinv, int v) {
  fuse_scale_t s; s.s = _mm256_loadu_pd(rinv + (long) v * veclen); return s;
}
static inline REAL_VEC fuse_r_scale_vec(REAL_VEC x, fuse_scale_t s) {
  return _mm256_mul_pd(x, s.s);
}
#endif
#else
typedef struct { double s; } fuse_scale_t;
static inline fuse_scale_t fuse_scale_build(const double * rinv, int v) {
  fuse_scale_t s; s.s = rinv[v]; return s;    // veclen 1: lane block == lane
}
static inline REAL_VEC fuse_r_scale_vec(REAL_VEC x, fuse_scale_t s) {
  return (REAL) ((double) x * s.s);
}
#endif

// Implementation shared by the plain restrict, the fused pass's NT-publish
// variant, and the face restrict.  `stage` and `face` are compile-time literals
// at every call site (static inline, so the dead branches, the tile and the
// bounds switch all fold away and the plain codegen is untouched -- same trick
// as apply_P_site_impl).
//
// stage: per site, scale w by rinv into the 64B-aligned L1 tile, publish the
// tile to vnt with non-temporal stores (the basis slot's next readers are a
// coarse solve away -- DRAM either way -- so the cached-store write-allocate
// RFO is pure loss), and run the column loop on the tile.  Bit-identical to
// scaling into vnt with cached stores and restricting that: the tile holds the
// exact stored bits.  Callers of the staged path must NT_FENCE() per thread
// before their region join (NT stores are weakly ordered).
//
// face: restrict over one fine layer of the block instead of all of it -- `fd`
// names the face, and the block's `fd` coordinate is pinned to its high (+ dir)
// or low (- dir) layer.  Everything else is shared with the full-block walk, so
// a face restrict visits its sites and accumulates into each coarse variable in
// exactly the order the full-block restrict does; see apply_R_face for why that
// makes the two bit-identical on a face-supported input.
//
// always_inline, not plain inline: left to its own judgement the compiler will
// past some size emit one out-of-line body shared by all three call sites,
// which puts `stage` and `face` back on the runtime path -- a branch per site,
// and a tile materialized even in the plain restrict.  That specialization is
// what makes the sharing free, so it has to be required rather than hoped for.
static inline __attribute__((always_inline))
void apply_R_site_impl(REAL * coarse_out, REAL * fine_in,
                       REAL * nv, int cl, int num_parities,
                       int stage, const REAL * w,
                       const double * rinv, REAL * vnt,
                       int face, int fd) {
  REAL tile[Ns * Nc * 2 * num_rhs] __attribute__((aligned(64)));
  // Per-lane scale constants, one per lane block (see fuse_r_scale_vec).
  // Built once per site, and only on the staged path -- `stage` is a
  // compile-time literal, so the plain restrict never materializes these.
  fuse_scale_t sc[num_rhs_vec];
  if (stage)
    for (int v = 0; v < num_rhs_vec; v ++) sc[v] = fuse_scale_build(rinv, v);
  // zero this coarse site's variables (N_coarse vars, re+im over rhs, all
  // contiguous: coarse_idx(cl,v,0) is the base of v's 2*num_rhs-real block).
  for (int v = 0; v < N_coarse; v ++) {
    REAL * out = coarse_out + coarse_idx(cl, v, 0);
    for (int k = 0; k < 2 * num_rhs; k ++) out[k] = 0;
  }
  int x0, y0, z0, t0;
  coarse_block_origin(cl, &x0, &y0, &z0, &t0);
  // Full block, unless `face` pins one coordinate to a single fine layer.
  // `face` is a compile-time literal, so the full-block path keeps the constant
  // trip counts and never materializes the switch.
  int xlo = 0, xhi = BLOCK_AGG, ylo = 0, yhi = BLOCK_AGG;
  int zlo = 0, zhi = BLOCK_AGG, tlo = 0, thi = BLOCK_AGG;
  if (face)
    switch (fd) {
      case 0: xlo = BLOCK_AGG - 1; break;   case 4: xhi = 1; break;  // +x / -x
      case 1: ylo = BLOCK_AGG - 1; break;   case 5: yhi = 1; break;  // +y / -y
      case 2: zlo = BLOCK_AGG - 1; break;   case 6: zhi = 1; break;  // +z / -z
      case 3: tlo = BLOCK_AGG - 1; break;   case 7: thi = 1; break;  // +t / -t
    }
  // Walk the two checkerboard parities (even, then odd) -- the fine field is
  // stored parity-major (loc's leading term), so each parity is one
  // contiguous half; x_start picks the first x in this row with
  // (x^y^z^t)&1 == parity and x += 2 stays on it.
  for (int parity = 0; parity < num_parities; parity ++)
    for (int t = t0 + tlo; t < t0 + thi; t ++)
      for (int z = z0 + zlo; z < z0 + zhi; z ++)
        for (int y = y0 + ylo; y < y0 + yhi; y ++) {
          int xa = x0 + xlo;
          int x_start = xa + ((parity ^ xa ^ y ^ z ^ t) & 1);
          for (int x = x_start; x < x0 + xhi; x += 2) {
            long loc = (long) parity * Vh + site_half(x / 2, y, z, t);
            long site_base = loc * (long) Ns * Nc * 2 * num_rhs;
            REAL * site_in;
            if (stage) {
              // Scale w into the L1 tile and NT-publish the same computed
              // vector to the basis slot -- one loop, one value, two stores
              // (cached to the tile the column loop reads, NT to vnt).  See
              // fuse_r_scale_vec for why this must be explicit vector code.
              // All three pointers are 64B-aligned: site_base is a multiple
              // of Ns*Nc*2*num_rhs reals and the buffers come from the
              // aligning malloc.
              const REAL_VEC * wv = (const REAL_VEC *) (w + site_base);
              REAL_VEC * tv = (REAL_VEC *) tile;
              REAL_VEC * vv = (REAL_VEC *) (vnt + site_base);
              for (long i = 0; i < (long) Ns * Nc * 2 * num_rhs / veclen;
                   i ++) {
                REAL_VEC v = fuse_r_scale_vec(wv[i], sc[i % num_rhs_vec]);
                tv[i] = v;
                NT_STORE(v, &vv[i]);
              }
              site_in = tile;
            } else
              site_in = fine_in + site_base;
            for (int d = 0; d < Ns; d ++) {
              int tau = d / 2;
              // Hand-vectorized over num_rhs lanes (explicit REAL_VEC -- see
              // apply_P_site).  Accumulate conj(nv[j]) * fine_in into each
              // coarse variable; the coarse-site accumulators stay
              // L1-resident.
              //
              // The three colors of this spin are interleaved inside the
              // j-loop: they all accumulate into the same coarse variables,
              // so doing them per-j shares one out load/store (via an
              // explicit register temp, since without restrict the compiler
              // would otherwise re-load out around each store) across 3
              // colors -- 1/3 the L1 read-modify-write traffic and 1/3 the
              // per-j index arithmetic.  Each coarse variable still receives
              // its (site, d, c) contributions in c-ascending order.
              long fbase0 = ((loc * Ns + d) * Nc) * 2;    // c = 0; c stride 2
              long vbase0 = fbase0 * num_null_vecs;
              REAL_VEC * in0_re = (REAL_VEC *) (site_in + (long) d * Nc * 2
                                                          * num_rhs);
              for (int j = 0; j < num_null_vecs; j ++) {
                REAL_VEC * out_re = (REAL_VEC *) (coarse_out
                                + coarse_idx(cl, num_null_vecs * tau + j, 0));
                REAL_VEC * out_im = out_re + num_rhs_vec;
                REAL_VEC o_re[num_rhs_vec], o_im[num_rhs_vec];
                for (int r = 0; r < num_rhs_vec; r ++) {
                  o_re[r] = out_re[r];
                  o_im[r] = out_im[r];
                }
                for (int c = 0; c < Nc; c ++) {
                  REAL_VEC vr  = SET1( nv[vbase0 + 2 * c * num_null_vecs + j]);
                  REAL_VEC nvi = SET1(-nv[vbase0 + (2 * c + 1) * num_null_vecs
                                      + j]);   // conj(v): -imag
                  REAL_VEC * in_re = in0_re + 2 * c * num_rhs_vec;
                  REAL_VEC * in_im = in_re + num_rhs_vec;
                  for (int r = 0; r < num_rhs_vec; r ++) {
                    o_re[r] += in_re[r] * vr - in_im[r] * nvi;
                    o_im[r] += in_im[r] * vr + in_re[r] * nvi;
                  }
                }
                for (int r = 0; r < num_rhs_vec; r ++) {
                  out_re[r] = o_re[r];
                  out_im[r] = o_im[r];
                }
              }
            }
          }
        }
}

// Plain restrict onto coarse site cl (stage = face = 0 folds the staging and
// the face windowing away -- see apply_R_site_impl).
static void apply_R_site(REAL * coarse_out, REAL * fine_in, REAL * nv, int cl,
                         int num_parities) {
  apply_R_site_impl(coarse_out, fine_in, nv, cl, num_parities,
                    0, NULL, NULL, NULL, 0, 0);
}

void apply_R(REAL * coarse_out, REAL * fine_in, REAL * nv) {
  // Each coarse site writes only its own coarse variables -> parallelize over
  // coarse sites with disjoint writes.
  struct timeval t0, t1; gettimeofday(&t0, NULL);
  #pragma omp parallel for schedule(dynamic)
  for (int cl = 0; cl < Vc; cl ++)
    apply_R_site(coarse_out, fine_in, nv, cl, 2);
  gettimeofday(&t1, NULL);
  transfer_R_secs += time_used(t0, t1); transfer_R_calls ++;
}

// Even-site-only restrict for the eo outer solve: fine_in is the even
// half-field (odd part implicitly zero), so only the even parity pass
// runs -- bit-identical to apply_R on the zero-padded field at half the work.
void apply_R_even(REAL * coarse_out, REAL * fine_in, REAL * nv) {
  struct timeval t0, t1; gettimeofday(&t0, NULL);
  #pragma omp parallel for schedule(dynamic)
  for (int cl = 0; cl < Vc; cl ++)
    apply_R_site(coarse_out, fine_in, nv, cl, 1);
  gettimeofday(&t1, NULL);
  transfer_R_secs += time_used(t0, t1);
  transfer_R_calls ++; transfer_R_half_calls ++;
}

// Restrict onto coarse site `cl` over only its one-layer fine face in
// direction `fd` (0:+x,1:+y,2:+z,3:+t,4:-x..7:-t).  Used for neighbor
// couplings in the windowed build: there the windowed D*probe is supported
// only on the neighbor block's face toward the probed block, and the rest of
// the block -- which apply_R_site would contract over -- is exactly 0.0.
// Summing over only the face is therefore bit-identical to the full-block
// apply_R_site (adding 0.0 changes nothing), at a fraction of the cost.
// The bit-identity needs the two to agree on accumulation order as well as on
// the set of contributions, since floating-point summation is not associative.
// That holds by construction: this is the same kernel under face = 1, so the
// only difference is which sites the walk visits.
static void apply_R_face(REAL * coarse_out, REAL * fine_in, REAL * nv,
                         int cl, int fd) {
  apply_R_site_impl(coarse_out, fine_in, nv, cl, 2,
                    0, NULL, NULL, NULL, 1, fd);
}

int coarse_neighbor(int cl, int dir) {
  int xb = cl % Lxc;        int r = cl / Lxc;
  int yb = r % Lyc;             r /= Lyc;
  int zb = r % Lzc;             r /= Lzc;
  int tb = r;
  xb = (xb + coarse_off[dir][0] + Lxc) % Lxc;
  yb = (yb + coarse_off[dir][1] + Lyc) % Lyc;
  zb = (zb + coarse_off[dir][2] + Lzc) % Lzc;
  tb = (tb + coarse_off[dir][3] + Ltc) % Ltc;
  return ((tb * Lzc + zb) * Lyc + yb) * Lxc + xb;
}

// Apply the fine Wilson-clover D to `fine` at the single full-lattice site
// (x,y,z,t), writing into `out`.  This is exactly one iteration of D()'s
// BLOCK_LOOPS body in clover_ferm.c (same mass/clover/hop arithmetic), so
// evaluating it over a chosen set of sites reproduces D() on those sites
// bit-for-bit.
static inline void D_site_local(REAL * out, REAL * fine, REAL * U, REAL * F,
                                int clover, int x, int y, int z, int t) {
  int site_size = Nc * Ns * num_rhs * 2;
  int parity = (x ^ y ^ z ^ t) & 1;
  long loc = (long) parity * Vh + site_half(x / 2, y, z, t);
  for (int i = 0; i < site_size; i ++)
    out[loc * site_size + i] = -fine[loc * site_size + i];
  wilson_kernel(out  + (long) parity * ferm_size * num_rhs / 2,
                fine + (long) (1 - parity) * ferm_size * num_rhs / 2,
                U, x / 2, y, z, t, parity);
  if (clover)
    clover_kernel(out  + (long) parity * ferm_size * num_rhs / 2,
                  fine + (long) parity * ferm_size * num_rhs / 2,
                  F + parity * Vh * 2 * 6 * 6,
                  x / 2, y, z, t);
  for (int i = 0; i < site_size; i ++)
    out[loc * site_size + i] *= -1;
}

// Off-diagonal (hopping-only) fine D at a single shell site, including only
// the fine hop direction(s) in `mask` (bit order 0:+x..7:-t).  D's diagonal
// is the identity + clover, both site-local; at a coarse-block shell site
// `fine` is zero, so D_site_local there reduces to exactly -H_dir.  This is
// bit-identical to D_site_local at a shell site (the masked-out directions
// read zeros there: x + 0 == x) but skips their FMAs.  `out` is overwritten
// (not accumulated).
static inline void D_hop_local(REAL * out, REAL * fine, REAL * U,
                               int x, int y, int z, int t, int mask) {
  int site_size = Nc * Ns * num_rhs * 2;
  int parity = (x ^ y ^ z ^ t) & 1;
  long loc = (long) parity * Vh + site_half(x / 2, y, z, t);
  for (int i = 0; i < site_size; i ++)
    out[loc * site_size + i] = 0;
  wilson_kernel_dir_c(out  + (long) parity * ferm_size * num_rhs / 2,
                      fine + (long) (1 - parity) * ferm_size * num_rhs / 2,
                      U, x / 2, y, z, t, parity, mask);
  for (int i = 0; i < site_size; i ++)
    out[loc * site_size + i] *= -1;
}

// Zero the multi-rhs fine fermion `field` over all fine sites of coarse block
// `cl` (the sites apply_R_site(cl) reads and apply_P_site(cl) writes).
static void zero_fine_block(REAL * field, int cl) {
  int site_size = Nc * Ns * num_rhs * 2;
  int x0, y0, z0, t0;
  coarse_block_origin(cl, &x0, &y0, &z0, &t0);
  for (int t = t0; t < t0 + BLOCK_AGG; t ++)
    for (int z = z0; z < z0 + BLOCK_AGG; z ++)
      for (int y = y0; y < y0 + BLOCK_AGG; y ++)
        for (int x = x0; x < x0 + BLOCK_AGG; x ++) {
          int parity = (x ^ y ^ z ^ t) & 1;
          long loc = (long) parity * Vh + site_half(x / 2, y, z, t);
          memset(field + loc * site_size, 0, site_size * sizeof(REAL));
        }
}

// Smallest divisor of L that is >= 3 (always exists since the build requires
// the coarse extents >= 3).  Used to color the coarse sites for the parallel
// probe loop below: two sites whose coordinates differ by a multiple of
// s_d >= 3 in every direction (the wrap distance is also a multiple, since
// s_d | L) have disjoint windowed footprints, so same-color probes never
// collide.
static int coarse_color_stride(int L) {
  for (int s = 3; s <= L; s ++) if (L % s == 0) return s;
  return L;
}

// flat color index of coarse site cl under per-dim strides (sx,sy,sz,st)
static inline int coarse_color(int cl, int sx, int sy, int sz, int st) {
  int xb = cl % Lxc;        int r = cl / Lxc;
  int yb = r % Lyc;             r /= Lyc;
  int zb = r % Lzc;             r /= Lzc;
  int tb = r;
  return (((tb % st) * sz + (zb % sz)) * sy + (yb % sy)) * sx + (xb % sx);
}

// Order the coarse sites by color so the probe loop can run in parallel.  The
// scratch buffers (fine/Dfine/res) are full-lattice sized for global
// indexing, and replicating them per thread would multiply large buffers by
// the thread count.  So keep one shared set and color the sites such that any
// two same-color probes have disjoint footprints (>= 3 coarse sites apart,
// with wrap, in every direction) -- concurrent, disjoint reads/writes,
// bit-identical to the serial order, no extra memory.  Sites of color c are
// qorder[coff[c] .. coff[c+1]).  Returns the color count; the caller frees
// qorder and coff.
static int coarse_color_order(int ** qorder_out, int ** coff_out) {
  int sx = coarse_color_stride(Lxc), sy = coarse_color_stride(Lyc);
  int sz = coarse_color_stride(Lzc), st = coarse_color_stride(Ltc);
  int ncolor = sx * sy * sz * st;
  int * qorder = malloc(sizeof(int) * Vc);    // coarse sites grouped by color
  int * coff   = calloc(ncolor + 1, sizeof(int));
  for (int q = 0; q < Vc; q ++) coff[1 + coarse_color(q, sx, sy, sz, st)] ++;
  for (int c = 0; c < ncolor; c ++) coff[c + 1] += coff[c];
  int * fill = malloc(sizeof(int) * ncolor);
  memcpy(fill, coff, sizeof(int) * ncolor);
  for (int q = 0; q < Vc; q ++)
    qorder[fill[coarse_color(q, sx, sy, sz, st)] ++] = q;
  free(fill);
  *qorder_out = qorder;
  *coff_out   = coff;
  return ncolor;
}

// Store one restricted coarse site p into the coupling block C:
// C[a][b] = res[p,a] for the num_rhs-batched columns b = b0 .. b0+nb-1 (lane
// s carries column b0+s).  Serves the self coupling (C = self[q], p = q),
// the local neighbor couplings (C = nbr[p][dir]), and the seam pass alike.
static void store_coupling(REAL * C, const REAL * res, int p, int b0, int nb) {
  for (int s = 0; s < nb; s ++) {
    int b = b0 + s;
    for (int a = 0; a < N_coarse; a ++) {
      long ri = coarse_idx(p, a, s);
      long m = (long) (a * N_coarse + b) * 2;
      C[m]     = res[ri];
      C[m + 1] = res[ri + num_rhs];
    }
  }
}

#if FERM_GHOSTS
// Does coarse direction `dir` (2:+z 6:-z 3:+t 7:-t -- the only decomposable
// dirs) from coarse block `cl` cross a rank face?  True exactly when `cl`
// sits on the corresponding decomposed-dimension boundary; such couplings are
// filled by the seam pass (row-local pull), not the local probe pass.
// 0 for x/y (never decomposed) and at undecomposed grids.
static inline int coarse_dir_cross_rank(int cl, int dir) {
  int r = cl / Lxc / Lyc;
  int zb = r % Lzc;  int tb = r / Lzc;
  switch (dir) {
    case 2: return DECOMP_Z && zb == Lzc - 1;   // +z
    case 6: return DECOMP_Z && zb == 0;         // -z
    case 3: return DECOMP_T && tb == Ltc - 1;   // +t
    case 7: return DECOMP_T && tb == 0;         // -t
  }
  return 0;
}

// --- cross-rank neighbor couplings (row-local pull) ------------------------
// The local probe pass fills self[q] and every local neighbor coupling, but a
// coupling D_c[p,q] with p local and q one hop across a rank face cannot be
// built by probing q -- q belongs to another rank.  It is built here instead,
// on the owner rank of p: one halo of `nv` supplies q's boundary null
// vectors, hence P*e_q on the seam layer, and D_c[p,q] = R_p D (P e_q) is
// then exactly the windowed shell hop -- D over block p's face toward q,
// reading the off-rank fermion from the ghost scratch and the seam gauge from
// U's tail ghost (the backward links; forward seam links are local).
// Bit-identical to the single-rank probe of q: apply_R_face == apply_R_site
// on the face-supported shell, and D_hop_local == D_site_local at an
// off-block shell site.
//
// Shared by both fine builders: it uses only D_hop_local and apply_R_face,
// which the full-stencil and direct builds evaluate identically.  On entry
// `fine` must be all-zero (each build restores that invariant after its last
// probe) and Dc's local couplings must already be filled; Dfine needs no
// zeroing, since the hops below overwrite every site apply_R_face then reads.
static void coarse_seam_pass(coarse_op * Dc, REAL * nv, REAL * U_kappa,
                             REAL * fine, REAL * Dfine, REAL * res,
                             long matsz) {
  if (!(DECOMP_Z || DECOMP_T)) return;

  long nv_site        = (long) Ns * Nc * 2 * num_null_vecs;      // REALs/site
  long nv_ghost_block = (long) GHOST_SITES * nv_site;
  // wk_fghost parity block
  long fg_block       = (long) GHOST_SITES * Nc * Ns * 2 * num_rhs;
  REAL * nv_ghost = malloc(sizeof(REAL) * 2 * nv_ghost_block);
  REAL * seam     = malloc(sizeof(REAL) * 2 * fg_block);

  // halo the null-vector z/t boundary layers (one parity half at a time; nv's
  // two parities are Vh contiguous sites each, like a fermion half-field)
  long nv_half = (long) Vh * nv_site;
  halo_ferm_half(nv, nv_ghost, nv_site * sizeof(REAL));
  halo_ferm_half(nv + nv_half, nv_ghost + nv_ghost_block,
                 nv_site * sizeof(REAL));

  // local boundary blocks p and which decomposed dirs cross a face (bitmask);
  // one entry per block (its dirs run sequentially so a corner block's two
  // faces never race on the shared Dfine edge)
  static const int crdirs[4] = {2, 6, 3, 7};
  int * cr_p     = malloc(sizeof(int) * Vc);
  int * cr_fmask = malloc(sizeof(int) * Vc);
  int ncr = 0;
  for (int p = 0; p < Vc; p ++) {
    int fm = 0;
    for (int k = 0; k < 4; k ++)
      if (coarse_dir_cross_rank(p, crdirs[k])) fm |= 1 << crdirs[k];
    if (fm) { cr_p[ncr] = p; cr_fmask[ncr] = fm; ncr ++; }
  }

  for (int b0 = 0; b0 < N_coarse; b0 += num_rhs) {
    int nb = (N_coarse - b0 < num_rhs) ? (N_coarse - b0) : num_rhs;

    // Reconstruct P*e_q on the seam from the null-vector ghost (unit-probe
    // prolong: lane s carries coarse variable b0+s, nonzero only on its spin
    // aggregate tau = d/2, column j = (b0+s) % num_null_vecs).  Fill both
    // parity blocks; a result site of parity P reads wk_fghost[1-P].
    memset(seam, 0, sizeof(REAL) * 2 * fg_block);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int par = 0; par < 2; par ++)
      for (long gi = 0; gi < GHOST_SITES; gi ++) {
        const REAL * ng = nv_ghost + (long) par * nv_ghost_block + gi * nv_site;
        REAL * sg = seam + (long) par * fg_block
                  + gi * (long) Nc * Ns * 2 * num_rhs;
        for (int d = 0; d < Ns; d ++) {
          int tau = d / 2;
          for (int c = 0; c < Nc; c ++)
            for (int reim = 0; reim < 2; reim ++) {
              const REAL * ngv = ng + ((d * Nc + c) * 2 + reim)
                                * (long) num_null_vecs;
              REAL * sgv = sg + ((d * Nc + c) * 2 + reim) * (long) num_rhs;
              for (int s = 0; s < nb; s ++) {
                int b = b0 + s;
                if (b / num_null_vecs == tau) sgv[s] = ngv[b % num_null_vecs];
              }
            }
        }
      }
    wilson_set_fghost(seam,            0);
    wilson_set_fghost(seam + fg_block, 1);

    #pragma omp parallel for schedule(dynamic)
    for (int ci = 0; ci < ncr; ci ++) {
      int p = cr_p[ci];
      int px0, py0, pz0, pt0;
      coarse_block_origin(p, &px0, &py0, &pz0, &pt0);
      for (int k = 0; k < 4; k ++) {
        int fd = crdirs[k];
        if (!(cr_fmask[ci] & (1 << fd))) continue;
        // block p's single fine layer facing q (the fd-direction face)
        int zlo = 0, zhi = BLOCK_AGG, tlo = 0, thi = BLOCK_AGG;
        switch (fd) {
          case 2: zlo = BLOCK_AGG - 1; break;   // +z
          case 6: zhi = 1;            break;    // -z
          case 3: tlo = BLOCK_AGG - 1; break;   // +t
          case 7: thi = 1;            break;    // -t
        }
        // D over that face: the single hop toward q (mask = 1<<fd) reads the
        // off-rank P*e_q from the ghost scratch; all other dirs read zeros.
        for (int t = pt0 + tlo; t < pt0 + thi; t ++)
          for (int z = pz0 + zlo; z < pz0 + zhi; z ++)
            for (int y = py0; y < py0 + BLOCK_AGG; y ++)
              for (int x = px0; x < px0 + BLOCK_AGG; x ++)
                D_hop_local(Dfine, fine, U_kappa, x, y, z, t, 1 << fd);
        // restrict block p over just that face (== full-block restrict of the
        // face-supported shell) -> D_c[p, neighbor(p,fd)] = nbr[p][fd]
        apply_R_face(res, Dfine, nv, p, fd);
        store_coupling(Dc->nbr + ((long) p * 8 + fd) * matsz, res, p, b0, nb);
      }
    }
  }
  wilson_zero_fghost();            // leave the scratch clean for later sweeps
  free(nv_ghost); free(seam); free(cr_p); free(cr_fmask);
}
#endif

// Reference build: probe each coarse block with unit coarse vectors, prolong,
// apply the full fine D over the block and its 1-site shell, restrict.  This is
// the plain statement of D_c = P^H D P, and it is the oracle the direct build
// below is checked against (test_coarse_direct) -- production runs the direct
// build, which reaches the same matrix with a fraction of the shell work.
void build_coarse_op(coarse_op * Dc, REAL * nv,
                     REAL * U_kappa, REAL * F_csw, int clover) {
  // need distinct +mu/-mu coarse neighbors (else a single probe's restricted
  // footprint would write two coupling slots from one shared neighbor)
  if (Lxc < 3 || Lyc < 3 || Lzc < 3 || Ltc < 3) {
    comm_die("build_coarse_op: coarse dims must be >= 3 (got %d %d %d %d)",
             Lxc, Lyc, Lzc, Ltc);
  }
  long matsz = (long) N_coarse * N_coarse * 2;          // REALs per coupling
  Dc->self = calloc((long) Vc * matsz, sizeof(REAL));
  Dc->nbr  = calloc((long) Vc * 8 * matsz, sizeof(REAL));

  REAL * probe = malloc(sizeof(REAL) * coarse_ferm_size * num_rhs);
  REAL * fine  = malloc(sizeof(REAL) * (long) ferm_size * num_rhs);
  REAL * Dfine = malloc(sizeof(REAL) * (long) ferm_size * num_rhs);
  REAL * res   = malloc(sizeof(REAL) * coarse_ferm_size * num_rhs);

  // Invariant: `fine` is zero everywhere except the block being probed.  Only
  // block q is prolonged (apply_P_site), and the windowed D below reads `fine`
  // over block q and its halo, all of which must be the genuine P*probe.
  // Zero once here; each probe restores block q afterwards.
  memset(fine, 0, sizeof(REAL) * (long) ferm_size * num_rhs);

  // The windowed local pass evaluates D over each probed block + a 1-fine-site
  // shell.  Under decomposition that shell crosses rank faces; the ghost-aware
  // kernel reads those off-rank neighbors from wk_fghost, so it must hold zeros
  // to match the single-rank "shell outside the block is 0".  (Cross-rank
  // neighbor couplings are filled separately by the seam pass at the end.)
  wilson_zero_fghost();

  int * qorder, * coff;
  int ncolor = coarse_color_order(&qorder, &coff);

  // Probe each coarse site q with unit coarse vectors e_{q,b}.  Because only
  // site q is excited, the restricted result `res` at any coarse site p
  // equals the block D_c[p, q]; p = q gives the self coupling, p a face
  // neighbor gives the corresponding neighbor coupling.  Columns are batched
  // num_rhs at a time.
  for (int c = 0; c < ncolor; c ++) {
    #pragma omp parallel for schedule(dynamic)
    for (int qi = coff[c]; qi < coff[c + 1]; qi ++) {
      int q = qorder[qi];
      int qx0, qy0, qz0, qt0;
      coarse_block_origin(q, &qx0, &qy0, &qz0, &qt0);
      for (int b0 = 0; b0 < N_coarse; b0 += num_rhs) {
        int nb = (N_coarse - b0 < num_rhs) ? (N_coarse - b0) : num_rhs;
        // zero only site q's probe variables (the only ones apply_P_site(q)
        // reads); a full memset would clobber the other threads sharing this
        // buffer.  Site q's N_coarse complex vars x num_rhs are contiguous from
        // coarse_idx(q,0,0).
        memset(probe + coarse_idx(q, 0, 0), 0,
               sizeof(REAL) * N_coarse * 2 * num_rhs);
        for (int s = 0; s < nb; s ++)
          probe[coarse_idx(q, b0 + s, s)] = 1.0;
  
        apply_P_site(fine, probe, nv, q);           // fine = P*probe on block q
  
        // Windowed D = P^H D P column.  The probe is supported only on block q,
        // so D*fine is nonzero only on block q and the one-hop shell just
        // outside its 8 faces (which lie in the face-neighbor blocks).
        // Zero those 9 blocks of Dfine, then evaluate D's per-site body on
        // block q and the shell: any site >=2 steps outside block q has no fine
        // neighbor inside the block, so D*fine is identically zero there and
        // stays zeroed.  apply_R then reads exactly those 9 blocks, so this
        // equals a full-lattice D bit-for-bit.
        zero_fine_block(Dfine, q);
        for (int dir = 0; dir < 8; dir ++)
          zero_fine_block(Dfine, coarse_neighbor(q, dir));
        for (int dt = -1; dt <= BLOCK_AGG; dt ++)
          for (int dz = -1; dz <= BLOCK_AGG; dz ++)
            for (int dy = -1; dy <= BLOCK_AGG; dy ++)
              for (int dx = -1; dx <= BLOCK_AGG; dx ++) {
                int nout = (dx < 0) + (dx >= BLOCK_AGG)
                         + (dy < 0) + (dy >= BLOCK_AGG)
                         + (dz < 0) + (dz >= BLOCK_AGG)
                         + (dt < 0) + (dt >= BLOCK_AGG);
                if (nout >= 2) continue;             // edge/corner: D*fine = 0
                int x = (qx0 + dx + Lx) % Lx;
                int y = (qy0 + dy + Ly) % Ly;
                int z = (qz0 + dz + Lz) % Lz;
                int t = (qt0 + dt + Lt) % Lt;
                D_site_local(Dfine, fine, U_kappa, F_csw, clover, x, y, z, t);
              }
  
        // restrict only the 9 coarse sites the extraction reads.  Cross-rank
        // neighbors wrap onto the wrong (off-rank) block here, so skip them --
        // their couplings are built by the seam pass below from the off-rank
        // null vectors.  (No-op at undecomposed grids: every dir is local.)
        apply_R_site(res, Dfine, nv, q, 2);
        for (int dir = 0; dir < 8; dir ++) {
#if FERM_GHOSTS
          if (coarse_dir_cross_rank(q, dir)) continue;
#endif
          apply_R_site(res, Dfine, nv, coarse_neighbor(q, dir), 2);
        }
  
        // self coupling: read at p = q
        store_coupling(Dc->self + (long) q * matsz, res, q, b0, nb);
  
        // neighbor couplings: source q seen from neighbor p = q + offset[dirp]
        // fills D_c[p, q] = nbr[p][dirp ^ 4] (since q = p + offset[dirp ^ 4]).
        for (int dirp = 0; dirp < 8; dirp ++) {
#if FERM_GHOSTS
          if (coarse_dir_cross_rank(q, dirp)) continue;  // built by the seam pass
#endif
          int p = coarse_neighbor(q, dirp);
          store_coupling(Dc->nbr + ((long) p * 8 + (dirp ^ 4)) * matsz,
                         res, p, b0, nb);
        }
        // restore the "fine is zero outside the probed block" invariant
        zero_fine_block(fine, q);
      }
    }
  }

#if FERM_GHOSTS
  coarse_seam_pass(Dc, nv, U_kappa, fine, Dfine, res, matsz);
#endif

  free(qorder); free(coff);
  free(probe); free(fine); free(Dfine); free(res);
}

// --- direct-assembly build (production) ------------------------------------
// Identical to build_coarse_op in every respect -- coloring, scratch buffers,
// probe/prolong/restrict, self+neighbor extraction and storage -- except the
// fine D evaluation: block-interior sites (nout==0) use the full D_site_local
// (the self coupling), while shell face sites (nout==1) take only the single
// fine hop that points back into block q via D_hop_local, instead of the full
// 8-direction stencil.  At a shell site exactly one neighbor lies in block q
// (the probe support) and the other 7 hops read the zeroed scratch, so this
// is bit-identical to build_coarse_op (in the double TU; the single TU's asm
// full kernel differs from the C masked kernel at the float-rounding level)
// while doing 1 hop instead of 8 at each shell site -- the bulk of the
// neighbor-coupling cost.  This is the directional face-hop assembly
// DDalphaAMG uses for its neighbor couplings, expressed in the existing
// probe/windowed framework.

void build_coarse_op_direct(coarse_op * Dc, REAL * nv,
                            REAL * U_kappa, REAL * F_csw, int clover) {
  if (Lxc < 3 || Lyc < 3 || Lzc < 3 || Ltc < 3) {
    comm_die("build_coarse_op_direct: coarse dims must be >= 3 (got %d %d %d %d)",
             Lxc, Lyc, Lzc, Ltc);
  }
  long matsz = (long) N_coarse * N_coarse * 2;
  Dc->self = calloc((long) Vc * matsz, sizeof(REAL));
  Dc->nbr  = calloc((long) Vc * 8 * matsz, sizeof(REAL));

  // no probe vector: prolong_unit_block writes P*(unit probe) directly
  REAL * fine  = malloc(sizeof(REAL) * (long) ferm_size * num_rhs);
  REAL * Dfine = malloc(sizeof(REAL) * (long) ferm_size * num_rhs);
  REAL * res   = malloc(sizeof(REAL) * coarse_ferm_size * num_rhs);
  memset(fine, 0, sizeof(REAL) * (long) ferm_size * num_rhs);

  // Under decomposition the shell hops below cross rank faces; the
  // ghost-aware D_site_local/D_hop_local read those off-rank neighbors from
  // wk_fghost, so it must hold zeros (matching the single-rank "shell
  // outside the block is 0").  Cross-rank neighbor couplings are filled by
  // the seam pass at the end.
  wilson_zero_fghost();

  int * qorder, * coff;
  int ncolor = coarse_color_order(&qorder, &coff);

  for (int c = 0; c < ncolor; c ++) {
    #pragma omp parallel for schedule(dynamic)
    for (int qi = coff[c]; qi < coff[c + 1]; qi ++) {
      int q = qorder[qi];
      int qx0, qy0, qz0, qt0;
      coarse_block_origin(q, &qx0, &qy0, &qz0, &qt0);
      for (int b0 = 0; b0 < N_coarse; b0 += num_rhs) {
        int nb = (N_coarse - b0 < num_rhs) ? (N_coarse - b0) : num_rhs;
  
        // P*(unit probe batch) directly = chiral-masked null-vector columns, no
        // probe vector and no num_null_vecs-deep interpolation (bit-identical).
        prolong_unit_block(fine, nv, q, b0, nb);
  
        // No Dfine zeroing needed: the restricts below read exactly the sites
        // the D loop overwrites this iteration -- block q (self, full) and the
        // 8 shell faces (neighbors, via apply_R_face).  D_site_local/D_hop_local
        // overwrite (not accumulate) every site they touch, so stale Dfine
        // outside that set is never read.  (build_coarse_op zeros +
        // full-restricts instead, and is the bit-identical reference this path
        // is validated against.)
        for (int dt = -1; dt <= BLOCK_AGG; dt ++)
          for (int dz = -1; dz <= BLOCK_AGG; dz ++)
            for (int dy = -1; dy <= BLOCK_AGG; dy ++)
              for (int dx = -1; dx <= BLOCK_AGG; dx ++) {
                int nout = (dx < 0) + (dx >= BLOCK_AGG)
                         + (dy < 0) + (dy >= BLOCK_AGG)
                         + (dz < 0) + (dz >= BLOCK_AGG)
                         + (dt < 0) + (dt >= BLOCK_AGG);
                if (nout >= 2) continue;
                int x = (qx0 + dx + Lx) % Lx;
                int y = (qy0 + dy + Ly) % Ly;
                int z = (qz0 + dz + Lz) % Lz;
                int t = (qt0 + dt + Lt) % Lt;
                if (nout == 0) {                // block interior: self, full D
                  D_site_local(Dfine, fine, U_kappa, F_csw, clover, x, y, z, t);
                } else {                        // shell face: single inward hop
                  int mask;
                  // +x face -> fine hop -x
                  if      (dx == BLOCK_AGG) mask = 1 << 4;
                  // -x face -> fine hop +x
                  else if (dx == -1)        mask = 1 << 0;
                  else if (dy == BLOCK_AGG) mask = 1 << 5;
                  else if (dy == -1)        mask = 1 << 1;
                  else if (dz == BLOCK_AGG) mask = 1 << 6;
                  else if (dz == -1)        mask = 1 << 2;
                  else if (dt == BLOCK_AGG) mask = 1 << 7;
                  else                      mask = 1 << 3;  // dt == -1
                  D_hop_local(Dfine, fine, U_kappa, x, y, z, t, mask);
                }
              }
  
        // Self: full block (all sites nonzero).  Neighbors: only the face
        // toward q (the only nonzero part of Dfine in the neighbor block);
        // dir^4 is the source direction from the neighbor's frame.  Cross-rank
        // neighbors wrap onto the wrong (off-rank) block here, so skip them --
        // their couplings are built by the seam pass below.  (No-op at
        // undecomposed grids.)
        apply_R_site(res, Dfine, nv, q, 2);
        for (int dir = 0; dir < 8; dir ++) {
#if FERM_GHOSTS
          if (coarse_dir_cross_rank(q, dir)) continue;
#endif
          apply_R_face(res, Dfine, nv, coarse_neighbor(q, dir), dir ^ 4);
        }
  
        store_coupling(Dc->self + (long) q * matsz, res, q, b0, nb);
  
        for (int dirp = 0; dirp < 8; dirp ++) {
#if FERM_GHOSTS
          // built by the seam pass
          if (coarse_dir_cross_rank(q, dirp)) continue;
#endif
          int p = coarse_neighbor(q, dirp);
          store_coupling(Dc->nbr + ((long) p * 8 + (dirp ^ 4)) * matsz,
                         res, p, b0, nb);
        }
        zero_fine_block(fine, q);
      }
    }
  }

#if FERM_GHOSTS
  coarse_seam_pass(Dc, nv, U_kappa, fine, Dfine, res, matsz);
#endif

  free(qorder); free(coff);
  free(fine); free(Dfine); free(res);
}

void free_coarse_op(coarse_op * Dc) {
  free(Dc->self); free(Dc->nbr);
  Dc->self = Dc->nbr = NULL;
}

// ===========================================================================
// Coarse-to-coarser transfer operators (hierarchy levels >= 1)
// ===========================================================================
// These mirror the fine (depth-0) apply_P / apply_R / local_orthonormalize
// above for the case where the "fine" side of the transfer is itself a coarse
// fermion -- lexicographic sites, N_coarse vars/site, chiral split by tau
// (vars [0,num_null_vecs) vs [num_null_vecs,N_coarse)) instead of by Dirac
// spin. Aggregation tiles each level's fine_lv->blk[] block of level-l sites
// into one level-(l+1) site, two aggregates per block (one per tau half),
// num_null_vecs test vectors each.  Block factors are per-level runtime
// (fine_lv->blk, possibly anisotropic); all per-site arithmetic dims
// (N_coarse, num_null_vecs, num_rhs) stay compile-time macros, so the inner
// loops vectorize the same as the fine ones.
// A level-l test-vector field nvc is packed one level down from `nv`:
//   nvc[(((s * N_coarse + v) * 2) + reim) * num_null_vecs + vec]
// (s a level-l lexicographic site, v in [0,N_coarse), reim 0/1, vec the
// column).

// level-l lexicographic site index (no even-odd reorder on coarse levels)
static inline long lvl_lex(const level_t * lv, int x, int y, int z, int t) {
  return (((long) t * lv->nz + z) * lv->ny + y) * lv->nx + x;
}

// Level-l block-origin coords of level-(l+1) site cl, given the level being
// coarsened (`fine_lv`).  The coarser lattice has fine_lv->n{x..t}/blk[d] sites
// per dimension; coarse lexicographic order matches the fine convention.
static void coarse_block_origin_lvl(const level_t * fine_lv, long cl,
                                    int * x0, int * y0, int * z0, int * t0) {
  int cnx = fine_lv->nx / fine_lv->blk[0];
  int cny = fine_lv->ny / fine_lv->blk[1];
  int cnz = fine_lv->nz / fine_lv->blk[2];
  int xb = cl % cnx; cl /= cnx;
  int yb = cl % cny; cl /= cny;
  int zb = cl % cnz; cl /= cnz;
  int tb = cl;
  *x0 = xb * fine_lv->blk[0]; *y0 = yb * fine_lv->blk[1];
  *z0 = zb * fine_lv->blk[2]; *t0 = tb * fine_lv->blk[3];
}

// real-part offset into nvc of (level-l site s, coarse var v, test vector vec);
// nvar_fine is the finer (level-l) variable count, ncol the number of columns
// (test vectors) for this transition -- the imaginary part sits ncol higher
static inline long nvc_off(long s, int v, int vec, int nvar_fine, int ncol) {
  return (((long) s * nvar_fine + v) * 2) * (long) ncol + vec;
}

// number of level-(l+1) sites in the hierarchy below `fine_lv`
long coarser_vol(const level_t * fine_lv) {
  return (long) (fine_lv->nx / fine_lv->blk[0])
               * (fine_lv->ny / fine_lv->blk[1])
               * (fine_lv->nz / fine_lv->blk[2])
               * (fine_lv->nt / fine_lv->blk[3]);
}

#if BOTTOM_SOLVE_DIAG
// -ffast-math makes isfinite()/x!=x unreliable (the compiler assumes
// finiteness), so test the IEEE exponent bits directly:
// all-ones exponent == Inf or NaN.  The sizeof branch is compile-time
// (REAL is float in the single-precision solve).
static long count_nonfinite(const REAL * a, long n) {
  long bad = 0;
  for (long i = 0; i < n; i ++) {
    if (sizeof(REAL) == 4) {
      unsigned int u; memcpy(&u, &a[i], 4);
      if ((u & 0x7F800000u) == 0x7F800000u) bad ++;
    } else {
      unsigned long long u; memcpy(&u, &a[i], 8);
      if ((u & 0x7FF0000000000000ull) == 0x7FF0000000000000ull) bad ++;
    }
  }
  return bad;
}
#endif

// Block-local orthonormalization of the fine_lv->nnv level-l test vectors
// within each aggregate (cl, tau), in place on nvc.  After this P_l^H P_l = I.
// Same per-aggregate CholeskyQR (+ MGS fallback) as the fine
// local_orthonormalize above -- the block geometry is per-level runtime
// (fine_lv->blk), so the component-offset list is built per aggregate and
// everything downstream is shared.
void local_orthonormalize_coarse(REAL * nvc, const level_t * fine_lv) {
  long cvol = coarser_vol(fine_lv);
#if BOTTOM_SOLVE_DIAG
  long nvlen = fine_lv->field_len * (long) fine_lv->nnv;
  long bad_in = count_nonfinite(nvc, nvlen);
  if (bad_in)
    print0("local_ortho_coarse[coarsen depth %d]: INPUT non-finite %ld / %ld\n",
           fine_lv->depth, bad_in, nvlen);
#endif
  int bx = fine_lv->blk[0], by = fine_lv->blk[1];
  int bz = fine_lv->blk[2], bt = fine_lv->blk[3];
  // The tau split point is the coarsened level's half-width (nvar/2), which
  // equals num_null_vecs only when coarsening a depth-1 level; deeper levels
  // are wider whenever a COARSE_NULL_VECS_LVL entry differs from the fine
  // count.  Enumerating [tau*nvh, (tau+1)*nvh) here must match the transfers'
  // tau = v / (nvar/2) mask, or part of each column is never orthonormalized
  // and P_l^H P_l != I.
  const int nvh = fine_lv->nvar / 2;
  // components per aggregate (one tau half)
  const int M = bx * by * bz * bt * nvh;
  double rmin = 2.0; long n2 = 0, nfb = 0;
  #pragma omp parallel reduction(min:rmin) reduction(+:n2,nfb)
  {
    long * base = malloc(sizeof(long) * M);  // re-part offset per component
    REAL * Rb = NULL, * Ib = NULL;  // MGS-fallback scratch, lazily allocated
    #pragma omp for collapse(2) schedule(dynamic)
    for (long cl = 0; cl < cvol; cl ++)
      for (int tau = 0; tau < 2; tau ++) {
        int x0, y0, z0, t0;
        coarse_block_origin_lvl(fine_lv, cl, &x0, &y0, &z0, &t0);
        // List the M component base offsets (block site x tau-half coarse
        // var), in the same (site, comp) order as the original scattered
        // reduction.
        int m = 0;
        for (int t = t0; t < t0 + bt; t ++)
          for (int z = z0; z < z0 + bz; z ++)
            for (int y = y0; y < y0 + by; y ++)
              for (int x = x0; x < x0 + bx; x ++) {
                long s = lvl_lex(fine_lv, x, y, z, t);
                for (int comp = 0; comp < nvh; comp ++)
                  base[m ++] = nvc_off(s, nvh * tau + comp, 0, fine_lv->nvar,
                                       fine_lv->nnv);
              }
        agg_orthonormalize(nvc, base, M, fine_lv->nnv, &Rb, &Ib, &rmin, &n2,
                           &nfb);
      }
    free(base); free(Rb); free(Ib);
  }
#if BOTTOM_SOLVE_DIAG
  long bad_out = count_nonfinite(nvc, nvlen);
  if (bad_out)
    print0("local_ortho_coarse[coarsen depth %d]: OUTPUT non-finite %ld / %ld "
           "(input had %ld)\n", fine_lv->depth, bad_out, nvlen, bad_in);
#endif
  agg_ortho_report("local_ortho_coarse", rmin, n2, nfb);
}

// Finer variables carried together through one contraction in the depth>=1
// transfers.  Each carries its own (re, im) accumulator pair, so a group
// gives 2*VAR_GROUP independent FMA chains -- a 2-port unit with 4-cycle FMA
// latency needs 8 in flight, and one variable alone would leave the
// ncol-deep contraction running at FMA latency rather than throughput.
// A group also shares every coarser-side operand: those depend on the
// aggregate and column, not the variable.
#define VAR_GROUP 4

// One group of group_vars (<= VAR_GROUP) finer variables of site s,
// chirality tau:
//   out[s,v] = sum_jj nvc[s,v,jj] * cin[cl, ncol*tau + jj].
// The contraction accumulates in registers rather than into out, which would
// otherwise be read-modify-written once per column (the compiler cannot keep
// it in a register across the loop without proof that out does not alias
// cin/nvc).  The finished accumulators are streamed out non-temporally:
// apply_P_coarse overwrites every level-l site, so the write-allocate read
// would be pure waste.  Column 0 SETs the accumulators, which also removes a
// separate zeroing pass.

static inline void apply_P_coarse_group(REAL * out, const REAL * cin,
                                        const REAL * nvc, long s, long cl,
                                        int v0, int group_vars, int tau,
                                        int nf, int ncol, int nc2) {
  REAL_VEC acc_re[VAR_GROUP][num_rhs_vec], acc_im[VAR_GROUP][num_rhs_vec];
  {
    const REAL_VEC * in_re = (const REAL_VEC *)
                             (cin + coarse_idx_lvl(cl, ncol * tau, 0, nc2));
    const REAL_VEC * in_im = in_re + num_rhs_vec;
    for (int i = 0; i < group_vars; i ++) {
      long vbase = nvc_off(s, v0 + i, 0, nf, ncol);
      REAL_VEC vr = SET1(nvc[vbase]);
      REAL_VEC vi = SET1(nvc[vbase + ncol]);
      for (int r = 0; r < num_rhs_vec; r ++) {
        acc_re[i][r] = in_re[r] * vr - in_im[r] * vi;
        acc_im[i][r] = in_im[r] * vr + in_re[r] * vi;
      }
    }
  }
  for (int jj = 1; jj < ncol; jj ++) {
    const REAL_VEC * in_re = (const REAL_VEC *) (cin
                               + coarse_idx_lvl(cl, ncol * tau + jj, 0, nc2));
    const REAL_VEC * in_im = in_re + num_rhs_vec;
    for (int i = 0; i < group_vars; i ++) {
      long vbase = nvc_off(s, v0 + i, 0, nf, ncol);
      REAL_VEC vr = SET1(nvc[vbase + jj]);
      REAL_VEC vi = SET1(nvc[vbase + ncol + jj]);
      for (int r = 0; r < num_rhs_vec; r ++) {
        acc_re[i][r] += in_re[r] * vr - in_im[r] * vi;
        acc_im[i][r] += in_im[r] * vr + in_re[r] * vi;
      }
    }
  }
  for (int i = 0; i < group_vars; i ++) {
    REAL_VEC * out_re = (REAL_VEC *) (out + coarse_idx_lvl(s, v0 + i, 0, nf));
    REAL_VEC * out_im = out_re + num_rhs_vec;
    for (int r = 0; r < num_rhs_vec; r ++) {
      NT_STORE(acc_re[i][r], &out_re[r]);
      NT_STORE(acc_im[i][r], &out_im[r]);
    }
  }
}

// Prolongate level-(l+1) site cl into its block of level-l sites:
// out = P_l cin.
static void apply_P_coarse_site(REAL * out, const REAL * cin, const REAL * nvc,
                                const level_t * fine_lv, long cl) {
  const int nf   = fine_lv->nvar;  // finer (level-l) variable count
  const int nfh  = nf / 2;    // finer half-width (chiral tau split point)
  const int ncol = fine_lv->nnv;   // columns = coarser (child) half-width
  const int nc2  = 2 * ncol;  // coarser variable count (child->nvar; set later)
  const int nfull = (nfh / VAR_GROUP) * VAR_GROUP;  // variables in whole groups
  int x0, y0, z0, t0;
  coarse_block_origin_lvl(fine_lv, cl, &x0, &y0, &z0, &t0);
  for (int t = t0; t < t0 + fine_lv->blk[3]; t ++)
    for (int z = z0; z < z0 + fine_lv->blk[2]; z ++)
      for (int y = y0; y < y0 + fine_lv->blk[1]; y ++)
        for (int x = x0; x < x0 + fine_lv->blk[0]; x ++) {
          long s = lvl_lex(fine_lv, x, y, z, t);
          // v ascending, split into VAR_GROUP-wide groups.  Grouping runs
          // inside one chirality (tau outer) because the coarser-side
          // operand a group shares is the tau half of cin.  The group width
          // reaches the helper as a literal so its per-variable loops unroll
          // and the accumulators stay in registers; a runtime width would
          // place them in a stack array instead.
          for (int tau = 0; tau < 2; tau ++) {
            for (int v0 = 0; v0 < nfull; v0 += VAR_GROUP)
              apply_P_coarse_group(out, cin, nvc, s, cl, tau * nfh + v0,
                                   VAR_GROUP, tau, nf, ncol, nc2);
            for (int v0 = nfull; v0 < nfh; v0 ++)
              apply_P_coarse_group(out, cin, nvc, s, cl, tau * nfh + v0, 1,
                                   tau, nf, ncol, nc2);
          }
        }
}

// One group of group_vars (<= VAR_GROUP) finer variables of site s,
// chirality tau:
//   cout[cl, ncol*tau + jj] += sum_i conj(nvc[s,v0+i,jj]) * fin[s,v0+i].
// The group's finer-side operands depend on the variable but not on the
// column, so they are loaded once and held; every variable in the group then
// accumulates into the same coarser output, so one register
// read-modify-write per column covers all of them (without that temp the
// compiler must reload the output around each store, since it cannot prove
// cout does not alias fin/nvc).  Each coarser variable still receives its
// contributions in ascending finer-variable order.
static inline void apply_R_coarse_group(REAL * cout, const REAL * fin,
                                        const REAL * nvc, long s, long cl,
                                        int v0, int group_vars, int tau,
                                        int nf, int ncol, int nc2) {
  REAL_VEC in_re[VAR_GROUP][num_rhs_vec], in_im[VAR_GROUP][num_rhs_vec];
  for (int i = 0; i < group_vars; i ++) {
    const REAL_VEC * f = (const REAL_VEC *)
                         (fin + coarse_idx_lvl(s, v0 + i, 0, nf));
    for (int r = 0; r < num_rhs_vec; r ++) {
      in_re[i][r] = f[r];
      in_im[i][r] = f[num_rhs_vec + r];
    }
  }
  for (int jj = 0; jj < ncol; jj ++) {
    REAL_VEC * out_re = (REAL_VEC *)
                        (cout + coarse_idx_lvl(cl, ncol * tau + jj, 0, nc2));
    REAL_VEC * out_im = out_re + num_rhs_vec;
    REAL_VEC o_re[num_rhs_vec], o_im[num_rhs_vec];
    for (int r = 0; r < num_rhs_vec; r ++) {
      o_re[r] = out_re[r];
      o_im[r] = out_im[r];
    }
    for (int i = 0; i < group_vars; i ++) {
      long vbase = nvc_off(s, v0 + i, 0, nf, ncol);
      REAL_VEC vr  = SET1( nvc[vbase + jj]);
      REAL_VEC nvi = SET1(-nvc[vbase + ncol + jj]);   // conj(v): -imag
      for (int r = 0; r < num_rhs_vec; r ++) {
        o_re[r] += in_re[i][r] * vr - in_im[i][r] * nvi;
        o_im[r] += in_im[i][r] * vr + in_re[i][r] * nvi;
      }
    }
    for (int r = 0; r < num_rhs_vec; r ++) {
      out_re[r] = o_re[r];
      out_im[r] = o_im[r];
    }
  }
}

// Restrict the level-l block of site cl onto level-(l+1):
// cout = R_l fin = P_l^H fin.  Shared by the full-block restrict and the face
// restrict: `face` pins the block's `fd` coordinate to its high (+ dir) or
// low (- dir) level-l layer, and everything else is common to both, so a face
// restrict visits its sites and accumulates into each coarser variable in
// exactly the order the full-block restrict does -- see apply_R_coarse_face
// for why that makes the two bit-identical on a face-supported input.
static void apply_R_coarse_impl(REAL * cout, const REAL * fin, const REAL * nvc,
                                const level_t * fine_lv, long cl,
                                int face, int fd) {
  const int nf   = fine_lv->nvar;  // finer (level-l) variable count
  const int nfh  = nf / 2;    // finer half-width (chiral tau split point)
  const int ncol = fine_lv->nnv;   // columns = coarser (child) half-width
  const int nc2  = 2 * ncol;  // coarser variable count (child->nvar; set later)
  const int nfull = (nfh / VAR_GROUP) * VAR_GROUP;  // variables in whole groups
  for (int v = 0; v < nc2; v ++)   // zero the coarser (child) output block
    for (int rhs = 0; rhs < num_rhs; rhs ++) {
      long ci = coarse_idx_lvl(cl, v, rhs, nc2);
      cout[ci] = 0; cout[ci + num_rhs] = 0;
    }
  int x0, y0, z0, t0;
  coarse_block_origin_lvl(fine_lv, cl, &x0, &y0, &z0, &t0);
  const int bx = fine_lv->blk[0], by = fine_lv->blk[1];
  const int bz = fine_lv->blk[2], bt = fine_lv->blk[3];
  // Full block, unless `face` pins one coordinate to a single level-l layer.
  int xlo = 0, xhi = bx, ylo = 0, yhi = by, zlo = 0, zhi = bz, tlo = 0, thi = bt;
  if (face)
    switch (fd) {
      case 0: xlo = bx - 1; break;   case 4: xhi = 1; break;   // +x / -x
      case 1: ylo = by - 1; break;   case 5: yhi = 1; break;   // +y / -y
      case 2: zlo = bz - 1; break;   case 6: zhi = 1; break;   // +z / -z
      case 3: tlo = bt - 1; break;   case 7: thi = 1; break;   // +t / -t
    }
  for (int t = t0 + tlo; t < t0 + thi; t ++)
    for (int z = z0 + zlo; z < z0 + zhi; z ++)
      for (int y = y0 + ylo; y < y0 + yhi; y ++)
        for (int x = x0 + xlo; x < x0 + xhi; x ++) {
          long s = lvl_lex(fine_lv, x, y, z, t);
          for (int tau = 0; tau < 2; tau ++) {
            for (int v0 = 0; v0 < nfull; v0 += VAR_GROUP)
              apply_R_coarse_group(cout, fin, nvc, s, cl, tau * nfh + v0,
                                   VAR_GROUP, tau, nf, ncol, nc2);
            for (int v0 = nfull; v0 < nfh; v0 ++)
              apply_R_coarse_group(cout, fin, nvc, s, cl, tau * nfh + v0, 1,
                                   tau, nf, ncol, nc2);
          }
        }
}

// Restrict the whole level-l block of site cl (face = 0 leaves the walk
// unwindowed -- see apply_R_coarse_impl).
static void apply_R_coarse_site(REAL * cout, const REAL * fin, const REAL * nvc,
                                const level_t * fine_lv, long cl) {
  apply_R_coarse_impl(cout, fin, nvc, fine_lv, cl, 0, 0);
}

// Prolong a batch of unit coarse probes onto block q of `fine_out`
// (the windowed build's prolong): probe s (s = 0..nb-1) excites coarser
// variable (b0 + s) in lane s.  A unit probe collapses apply_P_coarse_site's
// ncol-deep interpolation to one column -- P e is a single chiral-masked nvc
// column (variable b lives only in aggregate tau = b/ncol, at column b%ncol)
// -- so write it directly.  Bit-identical to apply_P_coarse_site(unit probe):
// its CSET(jj=0) + CAXPYs reduce to 0 + nvc[col] + 0 = nvc[col], every
// non-matching lane/aggregate exactly 0.  Overwrites all num_rhs lanes of
// block q (as apply_P_coarse_site CSETs them); sites outside q are untouched,
// preserving the "fine = 0 off-block" invariant the windowed D needs.
static void prolong_unit_coarse_block(REAL * fine_out, const REAL * nvc,
                                      const level_t * fine_lv, long q,
                                      int b0, int nb) {
  const int nf   = fine_lv->nvar;
  const int nfh  = nf / 2;
  const int ncol = fine_lv->nnv;
  int x0, y0, z0, t0;
  coarse_block_origin_lvl(fine_lv, q, &x0, &y0, &z0, &t0);
  for (int t = t0; t < t0 + fine_lv->blk[3]; t ++)
    for (int z = z0; z < z0 + fine_lv->blk[2]; z ++)
      for (int y = y0; y < y0 + fine_lv->blk[1]; y ++)
        for (int x = x0; x < x0 + fine_lv->blk[0]; x ++) {
          long s = lvl_lex(fine_lv, x, y, z, t);
          for (int v = 0; v < nf; v ++) {
            int tau = v / nfh;
            long vbase = nvc_off(s, v, 0, nf, ncol);
            REAL * o = fine_out + coarse_idx_lvl(s, v, 0, nf);
            for (int k = 0; k < num_rhs; k ++) {
              int b = b0 + k;
              if (k < nb && b / ncol == tau) {     // b in this spin aggregate
                int jj = b % ncol;                 // its nvc column
                o[k]           = nvc[vbase + jj];
                o[k + num_rhs] = nvc[vbase + ncol + jj];
              } else {
                o[k]           = 0;
                o[k + num_rhs] = 0;
              }
            }
          }
        }
}

// Restrict onto coarser site `cl` over only its one-layer level-l face in
// direction `fd` (0:+x..3:+t, 4:-x..7:-t) -- the neighbor-coupling restrict
// for the windowed build, where the windowed D*probe is supported only on
// the neighbor block's face toward the probed block and the rest of the
// block is exactly 0.0.  Summing over only the face is therefore bit-identical
// to the full-block apply_R_coarse_site (adding 0.0 changes nothing), at a
// fraction of the cost.  The bit-identity needs the two to agree on
// accumulation order as well as on the set of contributions, since
// floating-point summation is not associative.  That holds by construction:
// this is the same walk under face = 1, so the only difference is which sites
// it visits.
static void apply_R_coarse_face(REAL * cout, const REAL * fin, const REAL * nvc,
                                const level_t * fine_lv, long cl, int fd) {
  apply_R_coarse_impl(cout, fin, nvc, fine_lv, cl, 1, fd);
}

// out (level-l fermion) = P_l * cin (level-(l+1) fermion).  Disjoint blocks ->
// each level-l site written once, safe to parallelize over coarser sites.
void apply_P_coarse(REAL * out, REAL * cin, REAL * nvc,
                           const level_t * fine_lv) {
  long cvol = coarser_vol(fine_lv);
  #pragma omp parallel
  {
    #pragma omp for schedule(dynamic) nowait
    for (long cl = 0; cl < cvol; cl ++)
      apply_P_coarse_site(out, cin, nvc, fine_lv, cl);
    // apply_P_coarse_site streams `out` with non-temporal stores; fence each
    // thread before the region-end barrier so the consumer sees a drained
    // field.
    NT_FENCE();
  }
  // 8 flops (complex mul-add) per fine-side component x nullvec x rhs
  ctrans_flops[fine_lv->depth] += (double) cvol * fine_lv->blk[0]
                                 * fine_lv->blk[1] * fine_lv->blk[2]
                                 * fine_lv->blk[3] * fine_lv->nvar
                                 * fine_lv->nnv * 8.0 * num_rhs;
}

// cout (level-(l+1) fermion) = R_l * fin (level-l fermion).
void apply_R_coarse(REAL * cout, REAL * fin, REAL * nvc,
                           const level_t * fine_lv) {
  long cvol = coarser_vol(fine_lv);
  #pragma omp parallel for schedule(dynamic)
  for (long cl = 0; cl < cvol; cl ++)
    apply_R_coarse_site(cout, fin, nvc, fine_lv, cl);
  // same count as apply_P_coarse (R = P^H, identical contraction)
  ctrans_flops[fine_lv->depth] += (double) cvol * fine_lv->blk[0]
                                  * fine_lv->blk[1] * fine_lv->blk[2]
                                  * fine_lv->blk[3] * fine_lv->nvar
                                  * fine_lv->nnv * 8.0 * num_rhs;
}

// ===========================================================================
// Volume-aware coarse operator: neighbor, Dslash apply, and Galerkin build
// ===========================================================================
// These are the runtime-geometry analogues of coarse_neighbor / apply_Dc_ref /
// build_coarse_op.  They take the level's lattice from `lv` instead of the
// Lxc.. macros, so one set of code serves every coarse depth; the macro
// versions serve the level-1 path.

// Periodic nearest neighbor of lexicographic site cl in direction dir (0..7)
// on level lv's lattice.  Same direction table / decode order as
// coarse_neighbor.
long coarse_neighbor_lvl(const level_t * lv, long cl, int dir) {
  long x = cl % lv->nx; cl /= lv->nx;
  long y = cl % lv->ny; cl /= lv->ny;
  long z = cl % lv->nz; cl /= lv->nz;
  long t = cl;
  x = (x + coarse_off[dir][0] + lv->nx) % lv->nx;
  y = (y + coarse_off[dir][1] + lv->ny) % lv->ny;
  z = (z + coarse_off[dir][2] + lv->nz) % lv->nz;
  t = (t + coarse_off[dir][3] + lv->nt) % lv->nt;
  return (((long) t * lv->nz + z) * lv->ny + y) * lv->nx + x;
}

// Runtime-geometry analogue of coarse_parity: checkerboard parity of
// lexicographic coarse site `cl` on level `lv` (the depth-1 macro
// coarse_parity uses Lxc..Ltc).
static inline int coarse_parity_lvl(const level_t * lv, long cl) {
  long x = cl % lv->nx; cl /= lv->nx;
  long y = cl % lv->ny; cl /= lv->ny;
  long z = cl % lv->nz; cl /= lv->nz;
  long t = cl;
  return (int) ((x ^ y ^ z ^ t) & 1);
}

#if FERM_GHOSTS
// Allocate this coarse level's apply_Dc_lvl halo scratch, sized from its
// runtime extents and width (z face = nx*ny*nt sites, t face = nx*ny*nz,
// halo_coarse layout).  Called once at level creation; freed by
// coarse_sap_cache_free.  nvar and n{x..t} must already be set.  ghost_sites
// is > 0 whenever a dim is decomposed (guaranteed inside FERM_GHOSTS), but
// the guard keeps the shape identical to sap_dpsi_ghost.  External so a
// hand-built level_t (the selftests) sizes its scratch from this one formula:
// a second copy would silently overflow the day the layout changes.
void alloc_level_dc_ghost(level_t * lv) {
  long zface = (long) lv->nx * lv->ny * lv->nt;
  long tface = (long) lv->nx * lv->ny * lv->nz;
  long ghost_sites = 2 * zface * DECOMP_Z + 2 * tface * DECOMP_T;
  long blk = (long) lv->nvar * 2 * num_rhs;
  lv->dc_ghost = ghost_sites ? malloc(sizeof(REAL) * ghost_sites * blk) : NULL;
}
#endif


// out = Dc * in over lv->vol sites (runtime-geometry coarse fermions).
// Routes through the shared vectorized coarse_dslash_site; only the neighbor
// gather (coarse_neighbor_lvl) and loop bound (lv->vol) are level-specific.
// On level-1 geometry it reproduces apply_Dc.  Self-timed (dlvl_*) because
// at >= 3 levels this is the coarsest matvec that drives the bottom solve.
static double dlvl_secs[NUM_LEVELS], dlvl_flops[NUM_LEVELS];
static long   dlvl_calls[NUM_LEVELS];
void apply_Dc_lvl(const coarse_op * Dc, REAL * out, REAL * in,
                  const level_t * lv) {
  struct timeval _t0, _t1; gettimeofday(&_t0, NULL);
  const int  n_coarse = lv->nvar;           // coarse-site width at this level
  const long matsz = (long) n_coarse * n_coarse * 2;
  const long blk   = (long) n_coarse * 2 * num_rhs;   // reals per coarse site
  long vol = lv->vol;
#if FERM_GHOSTS
  // Refresh the z/t face ghosts of `in` for this apply (in changes per call).
  // The runtime analogue of apply_Dc's static wk_cghost: same halo_coarse
  // layout, face offsets from this level's extents, and the same dir->face
  // redirect below.  The coarse operator carries APBC in Dc->nbr, so the
  // exchange is a pure periodic fill.
  const int  cnx = lv->nx, cny = lv->ny, cnz = lv->nz, cnt = lv->nt;
  const long czface = (long) cnx * cny * cnt;   // z-face coarse sites (x,y,t)
  halo_coarse(in, lv->dc_ghost, blk * sizeof(REAL), cnx, cny, cnz, cnt);
#if DECOMP_Z
  const long off_zlo = 0;
  const long off_zhi = czface;
#endif
#if DECOMP_T
  const long ctface = (long) cnx * cny * cnz;   // t-face coarse sites (x,y,z)
  const long off_tlo = 2 * czface * DECOMP_Z;   // z faces precede t faces only if decomposed
  const long off_thi = off_tlo + ctface;
#endif
#endif
  #pragma omp parallel for schedule(static)
  for (long p = 0; p < vol; p ++) {
    REAL * cbases[10], * ibases[9];
    cbases[0] = Dc->self + p * matsz;
    ibases[0] = in + p * blk;
#if FERM_GHOSTS
    long rr = p;
    int xb = (int) (rr % cnx); rr /= cnx;
    int yb = (int) (rr % cny); rr /= cny;
    int zb = (int) (rr % cnz); rr /= cnz;
    int tb = (int) rr;
#endif
    for (int dir = 0; dir < 8; dir ++) {
      cbases[dir + 1] = Dc->nbr + ((long) p * 8 + dir) * matsz;
#if FERM_GHOSTS
      const REAL * nb;
#if DECOMP_Z
      if (dir == 2 && zb == cnz - 1)
        nb = lv->dc_ghost + (off_zhi + ((long) tb * cny + yb) * cnx + xb) * blk;
      else if (dir == 6 && zb == 0)
        nb = lv->dc_ghost + (off_zlo + ((long) tb * cny + yb) * cnx + xb) * blk;
      else
#endif
#if DECOMP_T
      if (dir == 3 && tb == cnt - 1)
        nb = lv->dc_ghost + (off_thi + ((long) zb * cny + yb) * cnx + xb) * blk;
      else if (dir == 7 && tb == 0)
        nb = lv->dc_ghost + (off_tlo + ((long) zb * cny + yb) * cnx + xb) * blk;
      else
#endif
        nb = in + coarse_neighbor_lvl(lv, p, dir) * blk;
      ibases[dir + 1] = (REAL *) nb;
#else
      ibases[dir + 1] = in + coarse_neighbor_lvl(lv, p, dir) * blk;
#endif
    }
    coarse_dslash_site(out + p * blk, cbases, ibases, 9, n_coarse);
  }
  gettimeofday(&_t1, NULL);
  dlvl_secs[lv->depth] += time_used(_t0, _t1); dlvl_calls[lv->depth] ++;
  // 9 couplings (self + 8 nbrs), nvar^2 complex MACs (8 flops) per site/rhs.
  dlvl_flops[lv->depth] += (double) vol * 9.0 * (double) n_coarse
                         * (double) n_coarse * 8.0 * (double) num_rhs;
}

// Windowed single-site coarse Dslash with a coupling mask: out at level-l
// lexicographic site p only, accumulating only the couplings flagged in `mask`
// (bit 0 = self, bit 1+dir = neighbor dir), in the same order as the full
// 9-coupling apply.  This lets the Galerkin build skip the couplings whose
// input site lies outside the probed block: the probe is nonzero only on
// block q, so a skipped coupling contributes an exact zero and masking is
// bit-identical to the full site apply (mask 0x1ff).  A shell site keeps 1 of
// 9 couplings, a block-interior site 1 + its in-block neighbors.  out's site
// block is overwritten.
static void apply_Dc_lvl_site_masked(const coarse_op * Dc, REAL * out,
                                     REAL * in, const level_t * lv, long p,
                                     int mask) {
  const int  n_coarse   = lv->nvar;         // coarse-site width at this level
  const long matsz = (long) n_coarse * n_coarse * 2;
  const long blk   = (long) n_coarse * 2 * num_rhs;
  REAL * cbases[10], * ibases[9];
  int nc = 0;
  if (mask & 1) {
    cbases[0] = Dc->self + p * matsz;
    ibases[0] = in + p * blk;
    nc = 1;
  }
  for (int dir = 0; dir < 8; dir ++)
    if (mask & (2 << dir)) {
      cbases[nc] = Dc->nbr + ((long) p * 8 + dir) * matsz;
      ibases[nc] = in + coarse_neighbor_lvl(lv, p, dir) * blk;
      nc ++;
    }
  coarse_dslash_site(out + p * blk, cbases, ibases, nc, n_coarse);
}

// Zero a level-l fermion `field` over all level-l sites of the coarser block
// `cl` (the sites apply_R_coarse_site(cl) reads and apply_P_coarse_site(cl)
// writes) -- the runtime-geometry analogue of zero_fine_block.
static void zero_lvl_block(REAL * field, const level_t * lv, long cl) {
  const long blk = (long) lv->nvar * 2 * num_rhs;
  int x0, y0, z0, t0;
  coarse_block_origin_lvl(lv, cl, &x0, &y0, &z0, &t0);
  for (int t = t0; t < t0 + lv->blk[3]; t ++)
    for (int z = z0; z < z0 + lv->blk[2]; z ++)
      for (int y = y0; y < y0 + lv->blk[1]; y ++)
        for (int x = x0; x < x0 + lv->blk[0]; x ++)
          memset(field + lvl_lex(lv, x, y, z, t) * blk, 0, blk * sizeof(REAL));
}

// Flat color index of coarser lexicographic site cl under per-dim strides;
// the runtime-geometry analogue of coarse_color (used to parallelize the
// windowed build below over disjoint-footprint probes).
static inline long coarse_color_lvl(const level_t * lv, long cl,
                                    int sx, int sy, int sz, int st) {
  long x = cl % lv->nx; cl /= lv->nx;
  long y = cl % lv->ny; cl /= lv->ny;
  long z = cl % lv->nz; cl /= lv->nz;
  long t = cl;
  return (((t % st) * sz + (z % sz)) * sy + (y % sy)) * sx + (x % sx);
}

// Full (non-windowed) Galerkin probe: for each coarser site q, a whole-field
// P / D_l / R apply, then extract the self + neighbor couplings with dedup
// over directions that coincide on a degenerate (extent <= 2) coarser
// dimension.  This is O(cvol^2), and serves the degenerate-lattice fallback.
// Caller owns the scratch (probe/fine/Dfine/res) and the cl_lvl geometry.
static void coarse_galerkin_full(coarse_op * Dc_out, const level_t * fine_lv,
                                 REAL * nvc, const level_t * clp, long matsz,
                                 REAL * probe, REAL * fine, REAL * Dfine,
                                 REAL * res) {
  long cvol = clp->vol;
  const int nc2 = 2 * fine_lv->nnv;          // coarser (child) variable count
  long cc_len = cvol * (long) nc2 * 2 * (long) num_rhs;
  // Probe by global child column.  Only z,t decompose, so the global child
  // lattice is (nx, ny, nz*Pz, nt*Pt) and this rank owns the contiguous slab
  // starting at corigin.  For each global probe site qg only its owner sets
  // the unit spike; all ranks then run P (aggregate-local) -> D_l
  // (halo-aware, so the single global spike reaches the neighbor rank
  // correctly) -> R, and each rank extracts the couplings for the sites it
  // owns.  At P=1 every site is local, gnz/gnt == local, corigin == 0,
  // and this is bit-identical to the serial probe over local sites.
  const int lnx = clp->nx, lny = clp->ny, lnz = clp->nz, lnt = clp->nt;
  const int gnz = lnz * qc_grid.P[2], gnt = lnt * qc_grid.P[3];
  const int oz = clp->corigin[2], ot = clp->corigin[3];
  const int cz = qc_grid.coords[2], ct = qc_grid.coords[3];
  const long gcvol = (long) lnx * lny * gnz * gnt;
  for (long qg = 0; qg < gcvol; qg ++) {
    int qx = (int) (qg % lnx);       long rr = qg / lnx;
    int qy = (int) (rr % lny);       rr /= lny;
    int qz = (int) (rr % gnz);       rr /= gnz;
    int qt = (int) rr;
    int own_q  = (qz / lnz == cz) && (qt / lnt == ct);
    long qloc  = own_q
          ? (((long) (qt - ot) * lnz + (qz - oz)) * lny + qy) * lnx + qx : -1;
    for (int b0 = 0; b0 < nc2; b0 += num_rhs) {
      int nb = (nc2 - b0 < num_rhs) ? (nc2 - b0) : num_rhs;
      memset(probe, 0, sizeof(REAL) * cc_len);
      if (own_q)
        for (int s = 0; s < nb; s ++)
          probe[coarse_idx_lvl(qloc, b0 + s, s, nc2)] = 1.0;

      apply_P_coarse(fine, probe, nvc, fine_lv);      // P probe (local)
      // D_l P probe (halo-aware)
      apply_Dc_lvl(fine_lv->Dc, Dfine, fine, fine_lv);
      apply_R_coarse(res, Dfine, nvc, fine_lv);       // R D_l P probe (local)

      // self coupling D_{l+1}[qg,qg] -- held only by qg's owner
      if (own_q) {
        REAL * selfM = Dc_out->self + qloc * matsz;
        for (int s = 0; s < nb; s ++) {
          int b = b0 + s;
          for (int a = 0; a < nc2; a ++) {
            long ri = coarse_idx_lvl(qloc, a, s, nc2);
            long m = (long) (a * nc2 + b) * 2;
            selfM[m]     = res[ri];
            selfM[m + 1] = res[ri + num_rhs];
          }
        }
      }
      // Neighbor couplings D_{l+1}[p,qg], dedup over coincident global
      // directions (extent<=2 dims fold several hops onto one neighbor;
      // R D P already summed them, so store once).  Each rank writes only
      // the neighbors p it owns.
      long png[8];
      for (int dir = 0; dir < 8; dir ++) {
        // x,y never decomposed
        int nxg = (qx + coarse_off[dir][0] + lnx) % lnx;
        int nyg = (qy + coarse_off[dir][1] + lny) % lny;
        int nzg = (qz + coarse_off[dir][2] + gnz) % gnz;
        int ntg = (qt + coarse_off[dir][3] + gnt) % gnt;
        png[dir] = (((long) ntg * gnz + nzg) * lny + nyg) * lnx + nxg;
      }
      for (int dirp = 0; dirp < 8; dirp ++) {
        long pg = png[dirp];
        if (pg == qg) continue;             // wraps to self (global extent 1)
        int dup = 0;
        for (int e = 0; e < dirp; e ++) if (png[e] == pg) { dup = 1; break; }
        if (dup) continue;                  // same neighbor already filled
        int pz = (int) ((pg / lnx / lny) % gnz),
            pt = (int) (pg / lnx / lny / gnz);
        if (pz / lnz != cz || pt / lnt != ct) continue;  // pg on another rank
        int px = (int) (pg % lnx), py = (int) ((pg / lnx) % lny);
        long ploc = (((long) (pt - ot) * lnz + (pz - oz)) * lny + py) * lnx
                    + px;
        REAL * C = Dc_out->nbr + ((long) ploc * 8 + (dirp ^ 4)) * matsz;
        for (int s = 0; s < nb; s ++) {
          int b = b0 + s;
          for (int a = 0; a < nc2; a ++) {
            long ri = coarse_idx_lvl(ploc, a, s, nc2);
            long m = (long) (a * nc2 + b) * 2;
            C[m]     = res[ri];
            C[m + 1] = res[ri + num_rhs];
          }
        }
      }
    }
  }
}

// Build D_{l+1} = R_l D_l P_l by probing each coarser site with the level-l
#if FERM_GHOSTS
// Runtime analogue of coarse_dir_cross_rank: does coarser direction `dir`
// (2:+z 6:-z 3:+t 7:-t) from coarser site `cl` on level `clv` cross a rank
// face?  The level being built has its z/t boundary aligned to the
// fine-lattice rank face (rank origins are aggregate-aligned), so a boundary
// coarser site's outward z/t hop is the cross-rank coupling -- filled by the
// seam pass.
static inline int coarse_dir_cross_rank_lvl(const level_t * clv, long cl,
                                            int dir) {
  long r = cl / clv->nx / clv->ny;
  long zb = r % clv->nz, tb = r / clv->nz;
  switch (dir) {
    case 2: return DECOMP_Z && zb == clv->nz - 1;   // +z
    case 6: return DECOMP_Z && zb == 0;             // -z
    case 3: return DECOMP_T && tb == clv->nt - 1;   // +t
    case 7: return DECOMP_T && tb == 0;             // -t
  }
  return 0;
}
#endif

// Coarse transfers + apply_Dc_lvl(fine_lv->Dc).  A single-block probe's
// footprint stays local (block q + 1-hop halo) whenever every coarser extent
// is >= 2, so D_{l+1} is built with a windowed parallel sweep -- O(cvol)
// work, not the O(cvol^2) full-field apply per probe.  Only a coarser extent
// of 1 falls back to the full apply below, where the block spans the
// dimension and its halo wraps onto itself.
//
// Degenerate fold: on a coarser dimension of extent <= 2, several +-mu
// directions land on the same neighbor (or wrap to q itself).  A single probe
// yields the combined matrix element res[p] = D_{l+1}[p,q] (R D P sums the
// coincident hops), which can't be split into separate stencil slots.  We
// store it in exactly one back-direction slot per distinct neighbor and leave
// the duplicates zero; wraps to q fold into `self` (which res[q] already
// captures).  apply_Dc_lvl then sums all 8 slots and recovers each coupling
// exactly once -- no change needed in the apply.  The windowed sweep carries
// the same fold; this one additionally handles the wrap-to-self that only an
// extent-1 dimension produces.
void build_coarse_op_lvl(coarse_op * Dc_out, const level_t * fine_lv,
                         REAL * nvc) {
  level_t cl_lvl;
  memset(&cl_lvl, 0, sizeof cl_lvl);
  cl_lvl.nx = fine_lv->nx / fine_lv->blk[0];
  cl_lvl.ny = fine_lv->ny / fine_lv->blk[1];
  cl_lvl.nz = fine_lv->nz / fine_lv->blk[2];
  cl_lvl.nt = fine_lv->nt / fine_lv->blk[3];
  long cvol = (long) cl_lvl.nx * cl_lvl.ny * cl_lvl.nz * cl_lvl.nt;
  cl_lvl.vol = cvol;
  // Child origin in child sites (parent origin / block); 0 at MPI=0.  The
  // degenerate build's global-column probe needs it to place each global
  // probe on its owner rank.
  for (int d = 0; d < 4; d ++) cl_lvl.corigin[d] = fine_lv->corigin[d] / fine_lv->blk[d];

  // Width mixing: the finer (level-l) field is nf wide (chiral split at
  // nf/2); the coarser (child) operator + fields are nc2 = 2*ncol wide,
  // ncol = per-transition columns.  These differ when
  // COARSE_NULL_VECS != num_null_vecs.  child->nvar is set only after this
  // returns, so derive nc2 from fine_lv->nnv (already set).
  const int nf   = fine_lv->nvar;
  const int ncol = fine_lv->nnv;
  const int nc2  = 2 * ncol;
  long matsz = (long) nc2 * nc2 * 2;                   // child coupling block
  Dc_out->self = calloc(cvol * matsz, sizeof(REAL));
  Dc_out->nbr  = calloc(cvol * 8 * matsz, sizeof(REAL));

  // finer (level-l) field
  long fl_len = fine_lv->vol * (long) nf * 2 * (long) num_rhs;
  // coarser (child) field
  long cc_len = cvol * (long) nc2 * 2 * (long) num_rhs;

  // --- windowed fast path --------------------------------------------------
  // Each probe touches only block q + its 8 face-neighbor blocks, so the
  // build is O(cvol) instead of a whole-field apply per probe.  Two ways to
  // run the probes concurrently over the scratch, chosen below: give each
  // worker its own copy, or color the sites so same-color footprints are
  // disjoint and share one copy.  Bit-identical to the serial full apply
  // either way.
  //
  // The one thing that must hold per coarser dim is that the probed block's
  // 1-fine-site halo lands on level-l sites the block does not already own:
  // the offsets [-1, blk] have to name blk + 2 distinct sites, so the local
  // coarser extent must be >= 3, or exactly 2 with a finer block of >= 2
  // (at local extent 1 the fine extent equals blk and both shells wrap
  // straight back onto the block, which only the full-field apply gets right).
  //
  // A coarser extent of 2 does not need excluding, even though it makes +mu
  // and -mu the same neighbor: R D P sums the two ring-hops into a single
  // matrix element that cannot be split back into two stencil slots, which
  // is a question of where the extraction files the element, not of whether
  //the windowed probe computes it.  Filing it once, in the first surviving
  // direction's back-slot, is the fold the loops below carry (apply_Dc_lvl
  // sums all 8 slots, so each coupling is recovered exactly once).  Under
  // decomposition the collision cannot arise in the first place: an extent-2
  // dim is never split, and on a decomposed axis at local extent 2 exactly
  // one of +-mu is the outward cross-rank face, which the local pass skips
  // and the seam pass fills.
  #define WIN_DIM(cext, b) ((cext) >= 3 || ((cext) == 2 && (b) >= 2))
  int windowed = WIN_DIM(cl_lvl.nx, fine_lv->blk[0])
              && WIN_DIM(cl_lvl.ny, fine_lv->blk[1])
              && WIN_DIM(cl_lvl.nz, fine_lv->blk[2])
              && WIN_DIM(cl_lvl.nt, fine_lv->blk[3]);
  #undef WIN_DIM

  // How wide the colored sweep can actually run is the size of one color
  // class, cvol / ncolor -- and the >= 3-apart rule pins each per-dim stride
  // to the smallest divisor >= 3 of that extent, so short, few-divisor child
  // extents starve it: an 8^3x12 child leaves 32 blocks per color, a 4^3x6
  // child only 2.  But the coloring exists solely so the probes can share
  // fine/Dfine/res.  A probe reads and writes nothing outside block q and
  // its 8 face neighbors, and no two probes reach the same D_{l+1} slot
  // (distinct q have distinct self, and nbr[p][dir^4] identifies its probe),
  // so once each worker owns a private set of scratch buffers the whole cvol
  // sweeps at once.  Take that path wherever it beats the color class and
  // the scratch fits the ceiling --  per worker that is two level-l fields
  // plus one child field, which prices the finest transitions out and leaves
  // them on the shared sweep.  Bit-identical either way: the probe bodies
  // are independent, only their order over the same disjoint writes changes.
  int nthr  = omp_get_max_threads();
  int nwork = 1;                                // scratch copies (1 = shared)
  int ncolor = 1, sx = 1, sy = 1, sz = 1, st = 1;
  if (windowed) {
    sx = coarse_color_stride(cl_lvl.nx); sy = coarse_color_stride(cl_lvl.ny);
    sz = coarse_color_stride(cl_lvl.nz); st = coarse_color_stride(cl_lvl.nt);
    ncolor = sx * sy * sz * st;
    long per_color = cvol / ncolor;
    long fits = ((long) COARSE_BUILD_SCRATCH_MB << 20)
              / ((long) sizeof(REAL) * (2 * fl_len + cc_len));
    int nmax = fits < nthr ? (int) fits : nthr;
    if (nmax > per_color) {
      nwork = nmax;
    }
  }

  // *_buf are the whole allocation; each probe below takes its worker's slice
  // (slice 0, i.e. the whole thing, when the scratch is shared).
  REAL * probe     = malloc(sizeof(REAL) * cc_len);
  REAL * fine_buf  = malloc(sizeof(REAL) * fl_len * nwork);
  REAL * Dfine_buf = malloc(sizeof(REAL) * fl_len * nwork);
  REAL * res_buf   = malloc(sizeof(REAL) * cc_len * nwork);

  if (windowed) {
    // One group per color when the scratch is shared; one group holding every
    // block when each worker has its own (nothing to serialize between them).
    int ngroup = nwork > 1 ? 1 : ncolor;
    long * qorder = malloc(sizeof(long) * cvol);
    long * coff   = calloc(ngroup + 1, sizeof(long));
    if (ngroup == 1) {
      for (long q = 0; q < cvol; q ++) qorder[q] = q;
      coff[1] = cvol;
    } else {
      for (long q = 0; q < cvol; q ++)
        coff[1 + coarse_color_lvl(&cl_lvl, q, sx, sy, sz, st)] ++;
      for (int c = 0; c < ncolor; c ++) coff[c + 1] += coff[c];
      long * fill = malloc(sizeof(long) * ncolor);
      memcpy(fill, coff, sizeof(long) * ncolor);
      for (long q = 0; q < cvol; q ++)
        qorder[fill[coarse_color_lvl(&cl_lvl, q, sx, sy, sz, st)] ++] = q;
      free(fill);
    }

    int bx = fine_lv->blk[0], by = fine_lv->blk[1];
    int bz = fine_lv->blk[2], bt = fine_lv->blk[3];
    int nrun = nwork > 1 ? nwork : nthr;         // workers in the probe sweep
    // `fine` stays zero outside the block being probed (each probe writes only
    // block q, restores it after); zero once up front, each copy on the worker
    // that will use it so it is first-touched where it is read.
    #pragma omp parallel for schedule(static) num_threads(nrun)
    for (int w = 0; w < nwork; w ++)
      memset(fine_buf + (long) w * fl_len, 0, sizeof(REAL) * fl_len);

    for (int c = 0; c < ngroup; c ++) {
      #pragma omp parallel for schedule(dynamic) num_threads(nrun)
      for (long qi = coff[c]; qi < coff[c + 1]; qi ++) {
        long q = qorder[qi];
        // slice 0 (the whole buffer) whenever the scratch is shared
        long _w = nwork > 1 ? omp_get_thread_num() : 0;
        REAL * fine  = fine_buf  + _w * fl_len;
        REAL * Dfine = Dfine_buf + _w * fl_len;
        REAL * res   = res_buf   + _w * cc_len;
        int qx0, qy0, qz0, qt0;
        coarse_block_origin_lvl(fine_lv, q, &qx0, &qy0, &qz0, &qt0);
        for (int b0 = 0; b0 < nc2; b0 += num_rhs) {
          int nb = (nc2 - b0 < num_rhs) ? (nc2 - b0) : num_rhs;
  
          // P*(unit probe batch) written directly = chiral-masked nvc columns;
          // no probe vector and no ncol-deep interpolation (bit-identical to
          // apply_P_coarse_site).
          // fine = P probe on block q
          prolong_unit_coarse_block(fine, nvc, fine_lv, q, b0, nb);
  
          // windowed D_l: zero block q + its 8 neighbor blocks, then evaluate D_l
          // only on block q and its 1-hop shell (sites >= 2 steps out are zero).
          zero_lvl_block(Dfine, fine_lv, q);
          for (int dir = 0; dir < 8; dir ++)
            zero_lvl_block(Dfine, fine_lv, coarse_neighbor_lvl(&cl_lvl, q, dir));
          for (int dt = -1; dt <= bt; dt ++)
            for (int dz = -1; dz <= bz; dz ++)
              for (int dy = -1; dy <= by; dy ++)
                for (int dx = -1; dx <= bx; dx ++) {
                  int nout = (dx < 0) + (dx >= bx) + (dy < 0) + (dy >= by)
                           + (dz < 0) + (dz >= bz) + (dt < 0) + (dt >= bt);
                  if (nout >= 2) continue;
                  int x = (qx0 + dx + fine_lv->nx) % fine_lv->nx;
                  int y = (qy0 + dy + fine_lv->ny) % fine_lv->ny;
                  int z = (qz0 + dz + fine_lv->nz) % fine_lv->nz;
                  int t = (qt0 + dt + fine_lv->nt) % fine_lv->nt;
                  // keep only the couplings whose input site lies inside block q
                  // (the probe's support; everything else reads exact zeros): a
                  // shell site keeps its single inward hop, an interior site self
                  // + in-block neighbors.  (dx..dt are block-relative in [-1,b].)
                  int cx = dx >= 0 && dx < bx, cy = dy >= 0 && dy < by,
                      cz = dz >= 0 && dz < bz, ct = dt >= 0 && dt < bt;
                  int mask = (cx && cy && cz && ct) ? 1 : 0;          // self
                  if (dx + 1 <  bx && cy && cz && ct) mask |= 2 << 0; // +x
                  if (dy + 1 <  by && cx && cz && ct) mask |= 2 << 1; // +y
                  if (dz + 1 <  bz && cx && cy && ct) mask |= 2 << 2; // +z
                  if (dt + 1 <  bt && cx && cy && cz) mask |= 2 << 3; // +t
                  if (dx - 1 >= 0  && cy && cz && ct) mask |= 2 << 4; // -x
                  if (dy - 1 >= 0  && cx && cz && ct) mask |= 2 << 5; // -y
                  if (dz - 1 >= 0  && cx && cy && ct) mask |= 2 << 6; // -z
                  if (dt - 1 >= 0  && cx && cy && cz) mask |= 2 << 7; // -t
                  apply_Dc_lvl_site_masked(fine_lv->Dc, Dfine, fine, fine_lv,
                                           lvl_lex(fine_lv, x, y, z, t), mask);
                }
  
          // Fold q's 8 directions onto the distinct neighbors the extraction
          // will read.  Cross-rank directions drop out first -- they wrap onto
          // the wrong (off-rank) block here, and the seam pass below fills
          // their couplings from the off-rank null vectors -- and only then
          // does a direction fold onto an earlier one, so a dropped direction
          // can never suppress a surviving twin.  keep[dir] marks the direction
          // that files the coupling; multi[dir] marks one that several
          // directions reach (only on a coarser extent of 2).
          long pnb[8]; int keep[8], multi[8];
          for (int dir = 0; dir < 8; dir ++) {
            pnb[dir] = coarse_neighbor_lvl(&cl_lvl, q, dir);
            multi[dir] = 0;
#if FERM_GHOSTS
            if (coarse_dir_cross_rank_lvl(&cl_lvl, q, dir)) {
              keep[dir] = 0;
              continue;
            }
#endif
            keep[dir] = 1;
            for (int e = 0; e < dir; e ++)
              if (keep[e] && pnb[e] == pnb[dir]) {
                keep[dir] = 0;
                multi[e] = 1;
                break;
              }
          }
  
          // restrict only the coarser sites the extraction reads.
          apply_R_coarse_site(res, Dfine, nvc, fine_lv, q);  // self: full block
          for (int dir = 0; dir < 8; dir ++) {
            if (!keep[dir]) continue;
            // The windowed D*probe reaches only neighbor p's face toward q, so
            // the face restrict alone is enough (the rest of p is exactly 0.0,
            // which contributes nothing to the same-order accumulation).
            // When several directions land on p, though, it holds one face per
            // direction, and a face restrict zeroes p's output before
            // accumulating -- so take the whole block and pick up every face in
            // one pass.
            if (multi[dir])
              apply_R_coarse_site(res, Dfine, nvc, fine_lv, pnb[dir]);
            else
              apply_R_coarse_face(res, Dfine, nvc, fine_lv, pnb[dir], dir ^ 4);
          }
  
          REAL * selfM = Dc_out->self + q * matsz;
          for (int s = 0; s < nb; s ++) {
            int b = b0 + s;
            for (int a = 0; a < nc2; a ++) {
              long ri = coarse_idx_lvl(q, a, s, nc2);
              long m = (long) (a * nc2 + b) * 2;
              selfM[m]     = res[ri];
              selfM[m + 1] = res[ri + num_rhs];
            }
          }
          // Neighbor couplings, one slot per surviving direction.  No two
          // probes collide: distinct q have distinct self, and (p, dirp^4)
          // names the probe that filed it, so the whole cvol can be swept in
          // any order.
          for (int dirp = 0; dirp < 8; dirp ++) {
            if (!keep[dirp]) continue;       // folded, or left to the seam pass
            long p = pnb[dirp];
            REAL * C = Dc_out->nbr + ((long) p * 8 + (dirp ^ 4)) * matsz;
            for (int s = 0; s < nb; s ++) {
              int b = b0 + s;
              for (int a = 0; a < nc2; a ++) {
                long ri = coarse_idx_lvl(p, a, s, nc2);
                long m = (long) (a * nc2 + b) * 2;
                C[m]     = res[ri];
                C[m + 1] = res[ri + num_rhs];
              }
            }
          }
        }
        zero_lvl_block(fine, fine_lv, q);  // restore "fine zero outside block q"
      }
    }
    free(qorder); free(coff);

#if FERM_GHOSTS
    // Cross-rank neighbor couplings (row-local pull; mirrors build_coarse_op)
    // D_{l+1}[p,q] for a local block p and a neighbor q one hop across a rank
    // face is a single depth-l hop: D_l's coupling fine_lv->Dc->nbr[s][fd]
    // (already cross-rank-correct from the depth-0->1 build) times P_l e_q
    // on block q's near-p depth-l layer.  Halo nvc once for that off-rank
    // layer, reconstruct P_l e_q on the coarse ghost (unit probe), apply the
    // single coupling via coarse_dslash_site (ncpl=1, the same kernel the
    // windowed apply uses -> bit-identical), and restrict over block p.
    // apply_Dc_lvl is not ghost-aware, so the input base points straight at
    // the reconstructed seam.
    if (DECOMP_Z || DECOMP_T) {
      int nx = fine_lv->nx, ny = fine_lv->ny,
          nz = fine_lv->nz, nt = fine_lv->nt;
      // finer half-width (chiral split)
      const int nfh = nf / 2;
      // fine_lv coupling block (nf x nf)
      const long nf_matsz = (long) nf * nf * 2;
      // depth-l (finer) coarse field/site
      long blk      = (long) nf * 2 * num_rhs;
      // nvc reals/site
      long nvc_site = (long) nf * 2 * ncol;
      int decz = qc_grid.P[2] > 1, dect = qc_grid.P[3] > 1;
      long zface = (long) nx * ny * nt, tface = (long) nx * ny * nz;
      long cg_zlo = 0, cg_zhi = zface;                   // halo_coarse layout
      long cg_tlo = 2 * zface * decz, cg_thi = cg_tlo + tface;
      long ghost_sites = 2 * zface * decz + 2 * tface * dect;

      REAL * nvc_ghost = malloc(sizeof(REAL) * ghost_sites * nvc_site);
      REAL * seam      = malloc(sizeof(REAL) * ghost_sites * blk);
      halo_coarse(nvc, nvc_ghost, nvc_site * sizeof(REAL), nx, ny, nz, nt);

      static const int crdirs[4] = {2, 6, 3, 7};
      long * cr_p     = malloc(sizeof(long) * cvol);
      int  * cr_fmask = malloc(sizeof(int)  * cvol);
      long ncr = 0;
      for (long p = 0; p < cvol; p ++) {
        int fm = 0;
        for (int k = 0; k < 4; k ++)
          if (coarse_dir_cross_rank_lvl(&cl_lvl, p, crdirs[k]))
            fm |= 1 << crdirs[k];
        if (fm) { cr_p[ncr] = p; cr_fmask[ncr] = fm; ncr ++; }
      }

      int bx = fine_lv->blk[0], by = fine_lv->blk[1];
      int bz = fine_lv->blk[2], bt = fine_lv->blk[3];
      for (int b0 = 0; b0 < nc2; b0 += num_rhs) {
        int nb = (nc2 - b0 < num_rhs) ? (nc2 - b0) : num_rhs;
        // reconstruct P_l e_q on the coarse ghost (unit probe: lane s carries
        // coarser var b0+s, nonzero on its aggregate tau = (b0+s)/ncol,
        // column j = (b0+s) % ncol).
        memset(seam, 0, sizeof(REAL) * ghost_sites * blk);
        #pragma omp parallel for schedule(static)
        for (long gi = 0; gi < ghost_sites; gi ++) {
          const REAL * ng = nvc_ghost + gi * nvc_site;
          REAL * sg = seam + gi * blk;
          for (int v = 0; v < nf; v ++) {
            int tau = v / nfh;
            for (int reim = 0; reim < 2; reim ++) {
              const REAL * ngv = ng + ((long) v * 2 + reim) * ncol;
              REAL * sgv = sg + ((long) v * 2 + reim) * num_rhs;
              for (int s = 0; s < nb; s ++) {
                int b = b0 + s;
                if (b / ncol == tau) sgv[s] = ngv[b % ncol];
              }
            }
          }
        }
        #pragma omp parallel for schedule(dynamic)
        for (long ci = 0; ci < ncr; ci ++) {
          long p = cr_p[ci];
          int x0, y0, z0, t0;
          coarse_block_origin_lvl(fine_lv, p, &x0, &y0, &z0, &t0);
          for (int k = 0; k < 4; k ++) {
            int fd = crdirs[k];
            if (!(cr_fmask[ci] & (1 << fd))) continue;
            int zlo = 0, zhi = bz, tlo = 0, thi = bt;
            switch (fd) {
              case 2: zlo = bz - 1; break;   // +z
              case 6: zhi = 1;      break;   // -z
              case 3: tlo = bt - 1; break;   // +t
              case 7: thi = 1;      break;   // -t
            }
            // zero the whole block in Dfine (apply_R_coarse_site reads it
            // all) then overwrite just the fd-face: D_l's single hop toward q
            // reads the off-rank P_l e_q from `seam` (couplings/input order
            // match the windowed apply_Dc_lvl_site_masked exactly ->
            // bit-identical)
            zero_lvl_block(Dfine_buf, fine_lv, p);
            for (int t = t0 + tlo; t < t0 + thi; t ++)
              for (int z = z0 + zlo; z < z0 + zhi; z ++)
                for (int y = y0; y < y0 + by; y ++)
                  for (int x = x0; x < x0 + bx; x ++) {
                    long s = lvl_lex(fine_lv, x, y, z, t);
                    long fidx;
                    switch (fd) {
                      case 2:
                        fidx = cg_zhi + ((long) t * ny + y) * nx + x; break;
                      case 6:
                        fidx = cg_zlo + ((long) t * ny + y) * nx + x; break;
                      case 3:
                        fidx = cg_thi + ((long) z * ny + y) * nx + x; break;
                      default:
                        fidx = cg_tlo + ((long) z * ny + y) * nx + x; break;
                    }
                    // fine_lv->Dc couplings are nf x nf (the finer width),
                    // not the child nc2 x nc2 that `matsz` sizes -- they
                    // differ under asymmetric per-level widths (nf != nc2),
                    // so index with nf_matsz
                    REAL * cb = fine_lv->Dc->nbr + ((long) s * 8 + fd)
                               * nf_matsz;
                    REAL * ib = seam + fidx * blk;
                    coarse_dslash_site(Dfine_buf + s * blk, &cb, &ib, 1,
                                       fine_lv->nvar);
                  }
            apply_R_coarse_site(res_buf, Dfine_buf, nvc, fine_lv, p);
            REAL * C = Dc_out->nbr + ((long) p * 8 + fd) * matsz;
            for (int s = 0; s < nb; s ++) {
              int b = b0 + s;
              for (int a = 0; a < nc2; a ++) {
                long ri = coarse_idx_lvl(p, a, s, nc2);
                long m = (long) (a * nc2 + b) * 2;
                C[m]     = res_buf[ri];
                C[m + 1] = res_buf[ri + num_rhs];
              }
            }
          }
        }
      }
      free(nvc_ghost); free(seam); free(cr_p); free(cr_fmask);
    }
#endif

    free(probe); free(fine_buf); free(Dfine_buf); free(res_buf);
    return;
  }

  // Fallback for a coarser extent of 1, where the probed block spans the whole
  // dimension and its halo wraps back onto itself so no window describes it.
  // coarse_galerkin_full probes by global column and applies the halo-aware
  // apply_Dc_lvl, so it is cross-rank correct at any decomposition (and
  // bit-identical to the serial build at P=1).
  coarse_galerkin_full(Dc_out, fine_lv, nvc, &cl_lvl, matsz,
                       probe, fine_buf, Dfine_buf, res_buf);
  free(probe); free(fine_buf); free(Dfine_buf); free(res_buf);
}

// Restarted GMRES on the coarse system, mirroring fgmres() in
// solver.c but with apply_Dc as the operator.  The coarse multi-rhs field has
// the same complex packing as a fine fermion (re at index, im at
// index+num_rhs, complex stride 2*num_rhs), so DOTC_KERNEL/CAXPY_KERNEL
// apply unchanged; only the vector length differs.  All num_rhs columns share
// one Krylov space (the operator is block-diagonal across rhs), as in the
// fine solver.  Wall time summed over every bottom solve billed to cg_bill()
// -- coarse_solve_full, coarse_solve_full_lvl and the eo Schur path -- split
// into matvec (apply_Dc) vs. orthogonalization (Arnoldi Gram-Schmidt: dotc +
// caxpy + normalize) vs. the rest (Givens, solve, x += V y, residual
// recompute).  The ortho and "other" halves are what mg_get_vcycle_profile
// exports as the profile table's bottom-solve rows.  cg_mv_secs is not
// exported: it times the same applies from the caller's side that
// apply_Dc / apply_coarse_schur already self-time into dc_secs / schur_secs,
// and the coarse-Dslash row uses those.  Adding it to the table would
// double-count.
static double cg_mv_secs = 0, cg_ortho_secs = 0, cg_other_secs = 0;
// GS dot/caxpy/norm flops (matches cg_ortho_secs)
static double cg_ortho_flops = 0;
static long   cg_calls = 0, cg_iters = 0;
void coarse_solve_stats_reset(void) {
  cg_mv_secs = cg_ortho_secs = cg_other_secs = 0; cg_ortho_flops = 0;
  cg_calls = cg_iters = 0;
}

// --- leaf vector/setup scopes ----------------------------------------------
// Both scopes exclude their matvecs, which are self-timed elsewhere, so these
// rows stay disjoint from the others.
//   kc_* : k-cycle (FGMRES) Gram-Schmidt and solution update.  Its
//          preconditioner apply is mg_cycle, billed to the V-cycle rows; its
//          matvec is assembled from that cycle's exit residual rather than
//          applied.
//   eo_* : even-odd Schur setup/recovery around the coarsest bottom solve in
//          coarse_solve_eo (gather/scatter + A_ee^-1 even solves + D_oe/D_eo)
//          -- everything outside the fgmres_coarse call, which cg_* times.
// kc_* is per level: a K-cycle wrapper runs on every level that has genuine
// recursion below it, so at four levels two of them are live at once.  eo_*
// and botd_* are not -- both exist only at the coarsest level.
static double kc_gs_secs[NUM_LEVELS], kc_gs_flops[NUM_LEVELS];
static long   kc_calls = 0;
static double eo_secs = 0, eo_flops = 0;       static long eo_calls = 0;
// botd_*: BOTTOM_DENSE_INV dense inverse applies (the GEMVs against the
// cached S^{-1} / D_c^{-1}), disjoint from eo_* (elimination/recovery) and
// cg_* (GMRES).
static double botd_secs = 0, botd_flops = 0;   static long botd_calls = 0;
void other_stats_reset(void) {
  for (int d = 0; d < NUM_LEVELS; d ++) {
    kc_gs_secs[d] = kc_gs_flops[d] = 0;
    dlvl_secs[d] = dlvl_flops[d] = 0; dlvl_calls[d] = 0;
  }
  kc_calls = 0;
  eo_secs = eo_flops = 0; eo_calls = 0;
  botd_secs = botd_flops = 0; botd_calls = 0;
}

// Generic restarted GMRES over an abstract operator op(ctx, out, in) acting
// on a split-complex multi-rhs field of cms REALs (complex stride
// 2*num_rhs).  Both the full coarse solve (op = apply_Dc,
// cms = coarse_ferm_size*num_rhs) and the even-odd Schur solve (op = Schur
// operator on odd sites, cms halved) share this core, so the instrumented
// Arnoldi loop lives in one place.
//
// A coarse operator carries near-null modes by construction, so GMRES
// routinely iterates into directions where the Arnoldi diagonal H[m,m]
// (a Rayleigh quotient) and the sub-diagonal H[m+1,m] collapse toward zero.
// Three guards keep the iteration finite there instead of dividing by a value
// the loose default tol simply stopped short of reaching: (1) a
// near-breakdown branch in the Givens rotation that never divides by the
// smaller-magnitude entry; (2) a lucky-breakdown exit when the new Krylov
// direction is (numerically) already in the span; (3) a relative pivot floor
// in the back-substitution.  All three are inert on a well-conditioned solve
// (they trip only below GMRES_BREAKDOWN of the local scale), so healthy runs
// are bit-for-bit unchanged.

// relative floor: Krylov invariance / negligible pivot
#define GMRES_BREAKDOWN   1e-11
// |H[m,m]| below this * |H[m+1,m]| -> overflow-free branch
#define GMRES_GIVENS_SAFE 1e-30

// --- flexible (right-preconditioned) extension: FGMRES ---------------------
// Passing a non-NULL `prec` turns fgmres_coarse into FGMRES.  It then keeps a
// second basis Z[j] = M^{-1} V[j] and builds the solution from Z rather than
// V, so the preconditioner may differ from step to step (as a multigrid cycle
// does).  Only V is orthonormalized -- the Arnoldi relation A Z = V H carries
// Z along, which is why FGMRES needs one projection sweep per step where an
// image-basis method (GCR) needs the sweep mirrored onto its direction basis
// as well.
//
// The hook fills both outputs for one basis vector: z = M^{-1} v and w = A z.
// They are produced together because a preconditioner that can hand back its
// own exit residual v - A z yields w as (v - r_exit) for free, deleting an
// operator apply per step; a caller without that shortcut simply applies the
// operator inside the hook.
//
// The flexible path runs exactly one cycle (asserted below) and exits on the
// Arnoldi residual estimate |g[m+1]|.  Both are deliberate.  It is used as a
// preconditioner at a loose relative tolerance, where the estimate is the
// right stopping test, and where the verifying `r = b - A x` apply that the
// unpreconditioned path does at each cycle boundary would cost more than the
// whole cycle's vector work -- on a coarse level an operator apply streams
// the entire 9-coupling matrix, tens of times the traffic of one basis sweep.
// A future caller that wants restarts here must first decide how to form that
// residual; the cheap route is the Arnoldi identity
// r = V_{m+1}(beta e_1 - H y), not a second operator apply.
// The unpreconditioned path keeps its explicit verify, unchanged.
typedef void (*fgmres_prec_fn)(void * pctx, REAL * z, REAL * w, REAL * v);

// Where a fgmres_coarse call bills its time and flops.  The coarsest bottom
// solve and the K-cycle wrapper are separate profile rows, so each caller
// names its own accumulators; any pointer may be NULL to leave that component
// unbilled.  Work done by a callee (apply_Dc, or the cycle behind `prec`) is
// timed where it happens and only passes through here.
typedef struct {
  double * mv_secs;      // operator applies fgmres_coarse itself issues
  double * ortho_secs;   // Arnoldi GS + Givens + solution update
  double * ortho_flops;  // ... and their flops
  double * other_secs;   // entry/exit bookkeeping, restart residual
  long   * calls;
  long   * iters;
} fgmres_bill;
#define FGBILL(p, v) do { if (p) *(p) += (v); } while (0)

static int fgmres_coarse(void (*op)(void *, REAL *, REAL *), void * ctx,
                         fgmres_prec_fn prec, void * pctx, fgmres_bill bill,
                         REAL * x, REAL * b, long cms,
                         int max_iter, REAL eps, int restart) {
  struct timeval ta, tb;
  gettimeofday(&ta, NULL);
  long rows = cms / (2 * num_rhs);           // complex elements per rhs block
  if (restart <= 0) restart = 30;
  if (restart > max_iter) restart = max_iter;
  if (restart <= 0) return 0;
  int ld = restart + 1;
  // Single cycle on the flexible path (see the note above): with
  // restart == max_iter the outer loop below cannot come round again, so no
  // restart residual is formed.
  assert(prec == NULL || restart >= max_iter);

  REAL * Vbasis = malloc(sizeof(REAL) * cms * ld);
  REAL * r = Vbasis;                           // first basis vector
  // Preconditioned basis: Z[j] = M^{-1} V[j], the solution basis on the
  // flexible path.  Only `restart` slots -- Z[m] is written for each Arnoldi
  // step taken.
  REAL * Zbasis = prec ? malloc(sizeof(REAL) * cms * restart) : NULL;
  REAL * Sbasis = prec ? Zbasis : Vbasis;  // basis the solution is built from
  // Only the unpreconditioned path recomputes r = b - A x at a cycle boundary.
  REAL * Ax = prec ? NULL : malloc(sizeof(REAL) * cms);
  // Every Krylov coefficient below carries a trailing rhs index: the rhs
  // share the basis and the operator, but each runs its own Arnoldi/Givens
  // process over that basis and gets its own solution polynomial.  Sharing
  // one set of coefficients would instead minimize the summed residual, which
  // is a different problem from any single rhs's -- and the shared basis is
  // orthonormal in the stacked inner product, not in any one rhs's, so a
  // shared Hessenberg never solved rhs k's minimization in the first place.
  // Matches QUDA (per-rhs alpha/beta/gamma in GCR, per-rhs Gram + Cholesky in
  // the CA-GCR coarsest solver).
  //
  // Lane-slowest: each rhs owns a contiguous slice, reached by a base pointer
  // at the top of every per-rhs loop, so the Givens and back-substitution
  // bodies stay the original single-Hessenberg code.  Readability, not speed
  // -- that work is O(restart^2) scalar per rhs.  The field passes want the
  // opposite order: hrow/yrow below.  Same convention as fgmres in solver.c.
  complex REAL * Hess = calloc((long) ld * restart * num_rhs, sizeof(complex REAL));
  REAL * cs = malloc(sizeof(REAL) * restart * num_rhs);
  complex REAL * sn = malloc(sizeof(complex REAL) * restart * num_rhs);
  complex REAL * g = calloc((long) (restart + 1) * num_rhs, sizeof(complex REAL));
  complex REAL * y = malloc(sizeof(complex REAL) * restart * num_rhs);
  // Split-complex, rhs-fastest scratch for the two hot field loops (the
  // Arnoldi reduction target and the solution-update coefficients).
  // Laid out the way the fields are -- num_rhs reals then num_rhs imags --
  // so the inner k loop reads a contiguous vector of coefficients where the
  // shared version broadcast a scalar.  The transposes between these and the
  // complex arrays above are O(m * num_rhs), against field loops of
  // O(rows * m * num_rhs).
  REAL * hrow = malloc(sizeof(REAL) * 2 * ld * num_rhs);
  REAL * yrow = malloc(sizeof(REAL) * 2 * restart * num_rhs);
  REAL beta[num_rhs], binv[num_rhs];

  // x = 0  =>  initial residual r = b.  Threaded, like every other full-field
  // pass: these run once per solve over a whole level field, so left serial
  // they are dead time on all but one core -- and they are the first touch of
  // x / r, so the parallel form also spreads those pages across the nodes
  // that go on to read them.
  #pragma omp parallel for
  for (long i = 0; i < cms; i ++) { x[i] = 0; r[i] = b[i]; }
  for (int k = 0; k < num_rhs; k ++) beta[k] = 0;
  #pragma omp parallel for reduction(+:beta[:num_rhs])
  for (long gi = 0; gi < cms / num_rhs; gi ++)
    for (int k = 0; k < num_rhs; k ++)
      beta[k] += r[gi * num_rhs + k] * r[gi * num_rhs + k];
  allreduce_sum_real(beta, num_rhs);
  REAL beta_f = 0;                        // Frobenius norm over the rhs block
  for (int k = 0; k < num_rhs; k ++) {
    beta_f += beta[k];
    beta[k] = sqrt(beta[k]);
  }
  beta_f = sqrt(beta_f);
  if (beta_f == 0.0) {
    free(Vbasis); free(Zbasis); free(Ax);
    free(Hess); free(cs); free(sn); free(g); free(y); free(hrow); free(yrow);
    return 0;
  }

  int total_iters = 0;
  while (total_iters < max_iter) {
    int m = 0;
    memset(Hess, 0, sizeof(complex REAL) * (long) ld * restart * num_rhs);
    // V0 = r / beta, per rhs.  A lane whose rhs is exactly zero has nothing
    // to solve and no direction to normalize: binv 0 zeroes it instead of
    // dividing, which retires it for the rest of the cycle (see the
    // lucky-breakdown note below -- a zero lane stays zero through every
    // operation that follows).
    for (int k = 0; k < num_rhs; k ++)
      binv[k] = (beta[k] > 0.0) ? (REAL) 1.0 / beta[k] : (REAL) 0.0;
    #pragma omp parallel for
    for (long gi = 0; gi < cms / num_rhs; gi ++)
      for (int k = 0; k < num_rhs; k ++) Vbasis[gi * num_rhs + k] *= binv[k];
    for (int k = 0; k < num_rhs; k ++) g[(long) k * (restart + 1)] = beta[k];

    for (m = 0; m < restart && total_iters + m < max_iter; m ++) {
      REAL * w = Vbasis + (long) (m + 1) * cms;
      gettimeofday(&tb, NULL);
      FGBILL(bill.other_secs, time_used(ta, tb));
      ta = tb;
      if (prec)   // z = M^{-1} V[m] and w = A z together (see fgmres_prec_fn)
        prec(pctx, Zbasis + (long) m * cms, w, Vbasis + (long) m * cms);
      else
        op(ctx, w, Vbasis + (long) m * cms);             // w = A V[m]
      gettimeofday(&tb, NULL);
      FGBILL(bill.mv_secs, time_used(ta, tb));
      ta = tb;

      // Arnoldi: orthogonalize w against V[0..m] (classical Gram-Schmidt).
      // One Hessenberg column per rhs, accumulated into the split-complex
      // scratch so the coefficients the projection below reads are
      // contiguous over k.  The reduction target must start at zero: OpenMP
      // zeroes the per-thread copies and adds them into this array, which
      // (unlike the Hess column the shared version wrote straight into) is
      // reused across Arnoldi steps.
      const int hlen = 2 * (m + 1) * num_rhs;
      for (int i = 0; i < hlen; i ++) hrow[i] = 0;
      #pragma omp parallel for reduction(+:hrow[:hlen])
      for (long i = 0; i < rows; i ++)
        for (int j = 0; j <= m; j ++)
          for (int k = 0; k < num_rhs; k ++)
            DOTC_KERNEL(hrow[(2 * j) * num_rhs + k],
                        hrow[(2 * j + 1) * num_rhs + k],
                        Vbasis + (long) j * cms + 2 * i * num_rhs,
                        w + 2 * i * num_rhs, k);
      allreduce_sum_real(hrow, hlen);
      REAL hmp1[num_rhs], hinv[num_rhs];
      int  rhs_converged[num_rhs];
      for (int k = 0; k < num_rhs; k ++) hmp1[k] = 0;
      #pragma omp parallel for reduction(+:hmp1[:num_rhs])
      for (long i = 0; i < rows; i ++) {
        REAL * shifted_w = w + 2 * i * num_rhs;
        for (int j = 0; j <= m; j ++)
          for (int k = 0; k < num_rhs; k ++)
            CAXPY_KERNEL(shifted_w,
                         Vbasis + (long) j * cms + 2 * i * num_rhs,
                         -hrow[(2 * j) * num_rhs + k],
                         -hrow[(2 * j + 1) * num_rhs + k], k);
        for (int k = 0; k < num_rhs; k ++)
          hmp1[k] += shifted_w[k] * shifted_w[k]
                   + shifted_w[k + num_rhs] * shifted_w[k + num_rhs];
      }
      allreduce_sum_real(hmp1, num_rhs);
      for (int k = 0; k < num_rhs; k ++) hmp1[k] = sqrt(hmp1[k]);
      // This rhs is solved.  Its new direction's component orthogonal to
      // V[0..m] has collapsed to the orthogonalization floor, so A V[m] is
      // (numerically) already in the span; an invariant Krylov subspace
      // makes H square and H y = beta e1 exactly solvable, so this rhs's
      // residual is zero.  This is the classical "lucky" (or happy) breakdown
      // of Arnoldi -- lucky because the division by zero it threatens arrives
      // exactly when it is no longer needed, unlike the serious breakdowns of
      // Lanczos/BiCG.  Continuing is harmful, not merely wasteful:
      // V[m+1] = w/hmp1 is roundoff over a denormal, which amplifies to O(1)
      // or overflows to Inf -- and -ffast-math flushes the denormal to zero
      // first, making it a deterministic 0/0.  ||A V[m]|| comes from the
      // reduced column by Pythagoras (sum|h_jm|^2 + hmp1^2), no extra global
      // sum.
      //
      // Per rhs this is a lane event, and the common case rather than an
      // exotic one: a lane's orthogonal component collapses exactly when its
      // rhs converges, which the shared-coefficient version never saw because
      // the stacked norm stays well scaled even when one lane does not.  So
      // the lane is retired, not the cycle.  hinv 0 does both jobs -- skips
      // the divide and zeroes the lane -- and a zero lane stays zero for the
      // rest of the solve: its later dots are 0, so its Hessenberg column is
      // 0, so its Givens rotation is the identity (rho == 0 below), its
      // residual estimate g stops moving, and the back-substitution's pivot
      // floor zeroes its y beyond this step.  The lane keeps the solution it
      // had converged to.  The cycle ends only when every lane is solved.
      int n_rhs_converged = 0;
      for (int k = 0; k < num_rhs; k ++) {
        complex REAL * H = Hess + (long) k * ld * restart;
        double wn0sq = (double) hmp1[k] * hmp1[k];
        for (int j = 0; j <= m; j ++)
          wn0sq += (double) hrow[(2 * j) * num_rhs + k] * hrow[(2 * j)
                    * num_rhs + k]
                  + (double) hrow[(2 * j + 1) * num_rhs + k]
                    * hrow[(2 * j + 1) * num_rhs + k];
        rhs_converged[k] = ((double) hmp1[k] <= GMRES_BREAKDOWN * sqrt(wn0sq));
        n_rhs_converged += rhs_converged[k];
        hinv[k] = (!rhs_converged[k] && hmp1[k] != 0.0)
                 ? (REAL) 1.0 / hmp1[k] : (REAL) 0.0;
        // Publish lane k's column: the split-complex reduction scratch
        // deinterleaves into the per-rhs Hessenberg the Givens sweep and
        // back-substitution walk.
        for (int j = 0; j <= m; j ++)
          H[j + (long) m * ld] = hrow[(2 * j) * num_rhs + k]
                           + I * hrow[(2 * j + 1) * num_rhs + k];
        H[(m + 1) + (long) m * ld] = hmp1[k];
      }
      #pragma omp parallel for
      for (long gi = 0; gi < cms / num_rhs; gi ++)
        for (int k = 0; k < num_rhs; k ++) w[gi * num_rhs + k] *= hinv[k];
      gettimeofday(&tb, NULL);
      FGBILL(bill.ortho_secs, time_used(ta, tb));
      ta = tb;
      // GS flops: (m+1) complex dots + (m+1) complex caxpys over cms/2
      // complex elements (8 flop each) + the hmp1 norm over cms reals, plus
      // one flop per real for the normalize -- which is now a multiply by a
      // per-lane reciprocal and so always runs, where the shared version
      // skipped it on breakdown.
      FGBILL(bill.ortho_flops, ((double) (m + 1) * 8.0 + 3.0) * cms);

      // Apply each rhs's own previous Givens rotations to its new column of H
      // One independent Givens sequence per rhs.  Each lane takes base
      // pointers into its own contiguous slice, so the body below is the
      // scalar single-Hessenberg code unchanged -- which is the point:
      // this is the numerically delicate part (the underflow-safe branch and
      // the rho == 0 identity a retired lane relies on), and it should stay
      // diffable against the form it was reviewed in.
      // New rotation to eliminate H[m+1,m].  The textbook
      // sn = cs*conj(H[m+1]/H[m]) divides by the diagonal H[m], which
      // collapses toward zero on a near-null Krylov direction; once
      // |H[m]| < |H[m+1]|*REAL_MIN the ratio overflows and cs*Inf
      // (with cs = |H[m]|/rho -> 0) is 0*Inf = NaN -- and under -ffast-math a
      // denormal H[m] is first flushed to exactly 0, making it deterministic.
      // When the diagonal underflows relative to the subdiagonal, use the
      // algebraically identical form that divides only by the larger
      // magnitude (|H[m]|/|H[m]| is a unit phase, so hd/hmm never overflows);
      // the common branch is the original expression, bit-for-bit.
      // A retired lane arrives here with an all-zero column, which lands in
      // the rho == 0 branch: the identity rotation, leaving its g -- and so
      // its residual estimate -- frozen where breakdown left it.
      for (int k = 0; k < num_rhs; k ++) {
        complex REAL * H  = Hess + (long) k * ld * restart;
        REAL         * ck = cs   + (long) k * restart;
        complex REAL * sk = sn   + (long) k * restart;
        complex REAL * gk = g    + (long) k * (restart + 1);

        for (int i = 0; i < m; i ++) {
          complex REAL temp = ck[i] * H[i + (long) m * ld]
                            + sk[i] * H[i + 1 + (long) m * ld];
          H[i + 1 + (long) m * ld] = -conj(sk[i]) * H[i + (long) m * ld]
                                   + ck[i] * H[i + 1 + (long) m * ld];
          H[i + (long) m * ld] = temp;
        }

        // Arnoldi diagonal
        complex REAL hd  = H[m + (long) m * ld];
        // subdiagonal (hmp1, real >= 0)
        complex REAL hsb = H[m + 1 + (long) m * ld];
        REAL hmm = cabs(hd);
        REAL hm1 = cabs(hsb);
        REAL rho = sqrt(hmm * hmm + hm1 * hm1);
        if (rho == 0.0) { ck[m] = 1.0; sk[m] = 0.0; }
        // diag underflows vs subdiag
        else if (hmm < (REAL) GMRES_GIVENS_SAFE * hm1) {
          ck[m] = hmm / rho;
          sk[m] = (hmm == 0.0) ? conj(hsb) / hm1  // exact swap limit
                 : (hd / hmm) * conj(hsb) / rho;  // |hd/hmm| == 1, no overflow
        } else {
          ck[m] = hmm / rho;
          sk[m] = ck[m] * conj(hsb / hd);
        }
        H[m + (long) m * ld] = ck[m] * H[m + (long) m * ld]
                             + sk[m] * H[m + 1 + (long) m * ld];
        H[m + 1 + (long) m * ld] = 0.0;
        complex REAL tempg = ck[m] * gk[m];
        gk[m + 1] = -conj(sk[m]) * gk[m];
        gk[m] = tempg;
      }

      // Cycle exit.  The residual estimate compared against eps is the
      // Frobenius norm over the rhs block -- exactly the quantity the
      // shared-coefficient version's scalar |g[m+1]| was, so the stopping
      // rule is unchanged by the move to per-rhs coefficients.  It must stay
      // a globally-reduced quantity: both callers' entry norms are allreduced
      // for the same reason, since a rank-local break point gives mismatched
      // collective counts and deadlocks.
      double gsq = 0;
      for (int k = 0; k < num_rhs; k ++) {
        double ga = cabs(g[(long) k * (restart + 1) + m + 1]);
        gsq += ga * ga;
      }
      if (n_rhs_converged == num_rhs || sqrt(gsq) < eps) { m ++; break; }
    }

    // Solve upper-triangular H y = g.  Drop any direction whose rotated pivot
    // is negligible vs the largest -- a near-null Krylov mode whose denormal
    // pivot would otherwise overflow y (and hence x) to Inf.  This is the
    // truncated pseudo-inverse the loose default coarse tol supplied
    // implicitly by stopping early; here it is explicit and bounded.
    // Each rhs scales its own floor to its own largest pivot, which is what
    // retires a broken lane cleanly: beyond its breakdown step that lane's
    // diagonal is zero, so every later y is dropped and it keeps exactly the
    // solution it had converged to.  The hmax scan is O(m) per rhs.
    for (int k = 0; k < num_rhs; k ++) {
      complex REAL * H  = Hess + (long) k * ld * restart;
      complex REAL * gk = g    + (long) k * (restart + 1);
      complex REAL * yk = y    + (long) k * restart;

      REAL hmax = 0.0;
      for (int i = 0; i < m; i ++) {
        REAL hii_abs = cabs(H[i + (long) i * ld]);
        if (hii_abs > hmax) hmax = hii_abs;
      }
      REAL hfloor = (REAL) GMRES_BREAKDOWN * hmax;
      for (int i = m - 1; i >= 0; i --) {
        complex REAL sum = 0.0;
        for (int j = i + 1; j < m; j ++) sum += H[i + (long) j * ld] * yk[j];
        complex REAL hii = H[i + (long) i * ld];
        yk[i] = (cabs(hii) <= hfloor) ? 0.0 : (gk[i] - sum) / hii;
      }
      // split-complex, rhs-fastest copy so the update loop's coefficient
      // reads are contiguous over k rather than a creal/cimag pair per
      // (basis vector, rhs)
      for (int i = 0; i < m; i ++) {
        yrow[(2 * i) * num_rhs + k]     = creal(yk[i]);
        yrow[(2 * i + 1) * num_rhs + k] = cimag(yk[i]);
      }
    }
    // x += S y, over the preconditioned basis Z when flexible, else V.
    // Blocked over the basis index so x is loaded once and every column
    // streams past it.  Billed to ortho: it is basis vector work, not
    // bookkeeping (the Givens/back-substitution scalars around it stay in
    // `other`, where they cost nothing to leave).
    gettimeofday(&tb, NULL);
    FGBILL(bill.other_secs, time_used(ta, tb));
    ta = tb;
    #pragma omp parallel for
    for (long i = 0; i < rows; i ++) {
      REAL * shifted_x = x + 2 * i * num_rhs;
      for (int j = 0; j < m; j ++)
        for (int k = 0; k < num_rhs; k ++)
          CAXPY_KERNEL(shifted_x, Sbasis + (long) j * cms + 2 * i * num_rhs,
                       yrow[(2 * j) * num_rhs + k],
                       yrow[(2 * j + 1) * num_rhs + k], k);
    }
    gettimeofday(&tb, NULL);
    FGBILL(bill.ortho_secs, time_used(ta, tb));
    ta = tb;
    FGBILL(bill.ortho_flops, 4.0 * (double) m * cms);   // m complex caxpys

    total_iters += m;
    // Flexible path: single cycle, and the exit was already decided by the
    // Arnoldi estimate -- nothing below would read r, so form no residual at
    // all.
    if (prec) break;

    // Recompute residual r = b - A x, per rhs: the next cycle normalizes each
    // lane by its own beta, and the exit test needs the same Frobenius norm
    // the in-cycle estimate above is compared against.
    op(ctx, Ax, x);
    for (int k = 0; k < num_rhs; k ++) beta[k] = 0;
    #pragma omp parallel for reduction(+:beta[:num_rhs])
    for (long gi = 0; gi < cms / num_rhs; gi ++)
      for (int k = 0; k < num_rhs; k ++) {
        REAL rk = b[gi * num_rhs + k] - Ax[gi * num_rhs + k];
        r[gi * num_rhs + k] = rk;
        beta[k] += rk * rk;
      }
    allreduce_sum_real(beta, num_rhs);
    beta_f = 0;
    for (int k = 0; k < num_rhs; k ++) {
      beta_f += beta[k];
      beta[k] = sqrt(beta[k]);
    }
    beta_f = sqrt(beta_f);
    if (beta_f < eps) break;
  }

  gettimeofday(&tb, NULL); FGBILL(bill.other_secs, time_used(ta, tb));
  if (bill.calls) (*bill.calls) ++;
  if (bill.iters) (*bill.iters) += total_iters;
  free(Vbasis); free(Zbasis); free(Ax);
  free(Hess); free(cs); free(sn); free(g); free(y); free(hrow); free(yrow);
  return total_iters;
}

// The coarsest bottom solve's profile row ("coarse bottom-solve GS" /
// "coarse GMRES Givens/upd"); the K-cycle wrapper names its own accumulators
// instead.
static fgmres_bill cg_bill(void) {
  fgmres_bill b = { &cg_mv_secs, &cg_ortho_secs, &cg_ortho_flops,
                    &cg_other_secs, &cg_calls, &cg_iters };
  return b;
}

// thin wrapper: full coarse system D_c x = b
static void op_apply_Dc(void * ctx, REAL * out, REAL * in) {
  apply_Dc((coarse_op *) ctx, out, in);
}
int coarse_solve_full(coarse_op * Dc, REAL * x, REAL * b,
                      int max_iter, REAL eps, int restart) {
  if (restart <= 0) restart = 30;
  if (restart > max_iter) restart = max_iter;
  if (restart <= 0) return 0;
  return fgmres_coarse(op_apply_Dc, Dc, NULL, NULL, cg_bill(),
                       x, b, coarse_ferm_size * num_rhs,
                       max_iter, eps, restart);
}

// Geometry-agnostic coarse GMRES: same restarted GMRES, but the operator is
// the runtime-volume apply_Dc_lvl and the field length comes from the level.
// It is the bottom solve on coarse levels whose geometry is not the level-1
// macro geometry.  ctx carries the operator and the level so fgmres_coarse's
// (void*) op stays a plain function.
typedef struct { const coarse_op * Dc; const level_t * lv; } dc_lvl_ctx;
static void op_apply_Dc_lvl(void * ctx, REAL * out, REAL * in) {
  const dc_lvl_ctx * c = (const dc_lvl_ctx *) ctx;
  apply_Dc_lvl(c->Dc, out, in, c->lv);
}
int coarse_solve_full_lvl(const level_t * lv, REAL * x, REAL * b,
                          int max_iter, REAL eps, int restart) {
  if (restart <= 0) restart = 30;
  if (restart > max_iter) restart = max_iter;
  if (restart <= 0) return 0;
  dc_lvl_ctx ctx = { lv->Dc, lv };
  return fgmres_coarse(op_apply_Dc_lvl, &ctx, NULL, NULL, cg_bill(),
                       x, b, lv->field_len * num_rhs,
                       max_iter, eps, restart);
}

// --- Multigrid cycle preconditioner ----------------------------------------
// MG_COARSE_EO selects the coarse solve at compile time, so this twice-compiled
// TU sees the same constant in both instantiations.  The cycle shape (cycle,
// w_mu, k_iters) is runtime: mg_params copies it onto every level_t, and the
// K-cycle tolerance is threaded in as mg_cycle's k_tol argument rather than
// carried in a global, so setup and solve can run the same cycle at different
// tolerances.

// The multigrid hierarchy (built by mg_setup, extended by mg_add_coarse_level).
// The cycle below recurses over `coarser`, so depth is purely a setup concern.
static level_t g_levels[NUM_LEVELS];
static int     g_num_levels = 0;
// Coarse operators the module owns and builds (levels >= 2; level 1's
// operator is caller-supplied to mg_setup).  Freed on teardown.
static coarse_op g_built_ops[NUM_LEVELS];

#if BOTTOM_DENSE_INV
// dense coarsest inverse (defined below)
static void coarse_direct_free(level_t * lv);
#endif

// Release everything mg_setup / mg_add_coarse_level allocated, then clear.
static void mg_teardown(void) {
  for (int i = 0; i < g_num_levels; i ++) {
    if (g_levels[i].eo_built) {
      free_coarse_eo_op(&g_levels[i].eo);
      g_levels[i].eo_built = 0;
    }
    coarse_sap_cache_free(&g_levels[i]);
#if BOTTOM_DENSE_INV
    coarse_direct_free(&g_levels[i]);
#endif
    if (i >= 2 && g_built_ops[i].self) free_coarse_op(&g_built_ops[i]);
  }
  memset(g_levels, 0, sizeof(g_levels));
  g_num_levels = 0;
}

// Partial teardown: release levels keep..g_num_levels-1 and make `keep` the
// new coarsest, preserving levels 0..keep-1 in full -- their operators
// (g_built_ops), geometry, and warm coarse-SAP caches.  The setup rebuild
// uses this after an inverse-iteration sweep at level `keep-1`: that sweep
// changes only the deeper levels, so re-deriving the upper ones (a full
// mg_teardown + rebuild) would just recompute identical bytes and re-invert
// an unchanged See_inv.  Mirrors the per-level frees in mg_teardown, over the
// trimmed suffix only.
void mg_trim_to(int keep) {
  if (keep < 0) keep = 0;
  if (keep >= g_num_levels) return;
  for (int i = keep; i < g_num_levels; i ++) {
    if (g_levels[i].eo_built) {
      free_coarse_eo_op(&g_levels[i].eo);
      g_levels[i].eo_built = 0;
    }
    coarse_sap_cache_free(&g_levels[i]);
#if BOTTOM_DENSE_INV
    coarse_direct_free(&g_levels[i]);
#endif
    if (i >= 2 && g_built_ops[i].self) free_coarse_op(&g_built_ops[i]);
  }
  memset(&g_levels[keep], 0, sizeof(g_levels[0]) * (g_num_levels - keep));
  g_num_levels = keep;
}

// out = D_l * in at level `lv`: fine Wilson-clover at depth 0, else the
// assembled coarse stencil.  Geometry dispatch (same test as bottom_solve):
// the level-1 macro geometry uses apply_Dc, any other coarse level the
// runtime-geometry apply_Dc_lvl.
static void apply_D_lvl(level_t * lv, REAL * out, REAL * in) {
  if (lv->depth == 0)
    D(out, in, lv->U_kappa, lv->F_csw, lv->clover);
  else if (lv->depth == 1 && lv->field_len == coarse_ferm_size)
    // macro depth-1 geometry (ghost-aware apply_Dc).  Both tests: depth alone
    // would route a hand-built depth-1 level of non-macro extents (synthetic
    // selftests) into the fixed-Vc kernel and overrun its fields; field_len
    // alone can alias a deeper level of coincidentally equal size.
    apply_Dc(lv->Dc, out, in);
  else
    // runtime geometry, ghost-aware at any depth
    apply_Dc_lvl(lv->Dc, out, in, lv);
}

// r = eta - D_l in, with the subtraction fused into the operator where cheap.
// At depth 0 the fine Dslash folds eta into its per-site negate pass (no
// extra streaming loop / fork-join).  At coarse levels there is no asm negate
// pass to ride on, so D_c is applied into r and subtracted in place.
// `r` must alias neither `eta` nor `in`.
//
// eo outer solve (g_eo_eta_in): the top-level eta is the even-site half-field
// of the eo outer FGMRES, odd part implicitly zero.  Only the depth-0 cycle
// ever sees that eta (deeper residuals act on coarse fields with no parity
// structure), and D_residual_eo skips the odd-site fold -- bit-identical to
// the zero-padded field, minus half the eta stream.
static int g_eo_eta_in = 0;  // set for the duration of one mg_vcycle(eo=1) call
static void apply_D_residual_lvl(level_t * lv, REAL * r, REAL * eta,
                                 REAL * in) {
  if (lv->depth == 0) {
    if (g_eo_eta_in)
      D_residual_eo(r, eta, in, lv->U_kappa, lv->F_csw, lv->clover);
    else
      D_residual(r, eta, in, lv->U_kappa, lv->F_csw, lv->clover);
  } else {
    apply_D_lvl(lv, r, in);                       // r = D_l in
    long n = lv->field_len * num_rhs;
    #pragma omp parallel for
    for (long i = 0; i < n; i ++) r[i] = eta[i] - r[i];
  }
}

// Inter-level transfers, dispatched by depth: the fine (depth-0) level uses
// the even-odd/spin-split apply_R/apply_P; coarse levels use the
// lexicographic/tau-split apply_R_coarse/apply_P_coarse.  lv is always the
// fine side of the transfer, and lv->nv its interpolation columns (fine null
// vecs at depth 0, nvc at depth >= 1).
#if BOTTOM_SOLVE_DIAG
static int g_restrict_diag_shown = 0;  // one-shot per solve (see restrict_lvl)
#endif
static void restrict_lvl(level_t * lv, REAL * cout, REAL * fin) {
  if (lv->depth == 0) {
    // eo outer solve: fin is the even half-field -> even-site-only restrict
    // (bit-identical to apply_R on the zero-padded field, half the work).
    if (g_eo_eta_in) apply_R_even(cout, fin, lv->nv);
    else             apply_R(cout, fin, lv->nv);
  }
  else                apply_R_coarse(cout, fin, lv->nv, lv);
#if BOTTOM_SOLVE_DIAG
  // one-shot per solve (g_restrict_diag_shown, cleared by
  // bottom_solve_diag_reset): the transfer feeding the coarsest bottom solve.
  // Separates the three ways cout can come out zero -- fin (residual) zero,
  // nv (transfer columns) zero, or a live restrict producing zero -- to
  // localize the coarsest zero-RHS bug.
  if (lv->coarser && lv->coarser->coarser == NULL) {
    if (!g_restrict_diag_shown) {
      g_restrict_diag_shown = 1;
      long nf = lv->field_len * (long) num_rhs;             // fin length
      long nc = lv->coarser->field_len * (long) num_rhs;    // cout length
      long nv = lv->field_len * (long) lv->nnv;             // nv length
      double fn = 0, cn = 0, vn = 0;
      for (long i = 0; i < nf; i ++) fn += (double) fin[i] * fin[i];
      for (long i = 0; i < nc; i ++) cn += (double) cout[i] * cout[i];
      for (long i = 0; i < nv; i ++) vn += (double) lv->nv[i] * lv->nv[i];
      print0("[restrict diag] depth %d -> coarsest (nnv=%d nvar=%d->%d): "
             "||fin||=%.3e ||nv||=%.3e ||cout||=%.3e\n",
             lv->depth, lv->nnv, lv->nvar, lv->coarser->nvar,
             sqrt(fn), sqrt(vn), sqrt(cn));
    }
  }
#endif
}
static void prolong_lvl(level_t * lv, REAL * fout, REAL * cin) {
  if (lv->depth == 0) {
    // Fused first-color prolongation: bulk-prolong only the SAP smoother's
    // second-processed block color; the resfold smoother that follows
    // immediately (depth-0 post-smoothing in mg_cycle) prolongs each
    // first-color block in-visit, cached, right before its fold seed reads
    // it -- so that half of the guess never makes a DRAM round trip.  Valid
    // only when the aggregation and SAP blockings coincide; the registration
    // is one-shot and consumed by that same smoother call.
    if (BLOCK_AGG == SCHWARZ_BLOCK_SIZE
        && sap_fuse_prolong_next(cin, lv->nv, TRANSFER_FLOPS_P / (double) Vc))
      apply_P_par(fout, cin, lv->nv, sap_bulk_blk_parity());
    else
      apply_P(fout, cin, lv->nv);
  }
  else                apply_P_coarse(fout, cin, lv->nv, lv);
}

// --- fused outer-normalize + depth-0 restrict (mixed outer fgmres) ---------
// The outer FGMRES normalize V[m+1] = w/hmp1 is the producer of the next
// V-cycle's input, whose first consumer is the depth-0 restrict R V[m+1] -- a
// full DRAM re-read of a basis vector that was in registers moments earlier.
// mg_restrict_fuse_normalize fuses the two: per aggregate, scale w and
// NT-publish it to the basis slot vnext (no write-allocate RFO; its next
// readers -- the GS dots and the SAP fold seeds -- are a coarse solve away
// and find it in DRAM under any store type), restricting the computed values
// while they are still register/L1-resident.  The coarse RHS lands in a
// private buffer registered one-shot against the vnext pointer; the next
// depth-0 cycle consumes it in mg_cycle_rexit and skips its restrict,
// deleting the restrict's fine-field DRAM read (the nv stream and the vnext
// store are compulsory either way).  Reached from the double fgmres through
// the mixed bridge (mg_normalize_restrict_single), so only the single twin's
// hierarchy ever registers; unregistered cycles (V0, any other caller)
// restrict as before. Restriction is aggregate-local, so this is MPI-clean;
// the per-rhs rinv is built from an allreduced hmp1, so every rank takes the
// same branch and applies the same scales.  Note: fused cycles skip
// restrict_lvl, so the BOTTOM_SOLVE_DIAG restrict probe (2-level runs)
// does not see them.

// registered coarse RHS (coarser-level field)
static REAL * fuse_r_buf = NULL;
// its allocated length (realloc on resize)
static long   fuse_r_len = 0;
// fine-input pointer the registration is for
static REAL * fuse_r_reg = NULL;
// parity mode it was built with (1 = even-only)
static int    fuse_r_eo  = 0;

// NT-publish per-aggregate body of the fused pass (see
// apply_R_site_impl): stages each site's scaled values through an L1 tile,
// streams them to vnext with NT stores, restricts from the tile
static void apply_R_site_nt(REAL * coarse_out, REAL * vnext, const REAL * w,
                            const double * rinv, REAL * nv, int cl,
                            int num_parities) {
  apply_R_site_impl(coarse_out, NULL, nv, cl, num_parities, 1, w, rinv, vnext,
                    0, 0);
}

int mg_restrict_fuse_normalize(REAL * vnext, REAL * w, const double * rinv,
                               int eo) {
  level_t * lv = mg_get_level(0);
  if (lv->coarser == NULL || lv->nv == NULL) return 0;   // no hierarchy yet

  struct timeval t0, t1; gettimeofday(&t0, NULL);
  long cn = (long) lv->coarser->field_len * num_rhs;
  if (fuse_r_len != cn) {
    free(fuse_r_buf);
    fuse_r_buf = malloc(sizeof(REAL) * cn);
    fuse_r_len = cn;
  }
  const int np = eo ? 1 : 2;
  REAL * nv = lv->nv;
  #pragma omp parallel
  {
    #pragma omp for schedule(dynamic) nowait
    for (int cl = 0; cl < Vc; cl ++)
      apply_R_site_nt(fuse_r_buf, vnext, w, rinv, nv, cl, np);
    // Drain each thread's weakly-ordered NT stores before the region-end
    // barrier: vnext is read later by other threads (GS dots, SAP fold
    // seeds).  Same pattern as apply_P.
    NT_FENCE();
  }
  gettimeofday(&t1, NULL);
  // Books exactly as a plain depth-0 restrict: the transfer row owns the pass
  // (call + eo half-call, so the GFlop/s rate stays honest) and level 0's vc_R
  // gets the same wall, which is where a restrict issued from inside the cycle
  // would have landed.  The scale flop is uncounted, as in the unfused
  // normalize.  The caller keeps this time OUT of its ortho bracket.
  transfer_R_secs += time_used(t0, t1);
  transfer_R_calls ++;
  if (eo) transfer_R_half_calls ++;
  vc_R[0] += time_used(t0, t1);

  fuse_r_reg = vnext;
  fuse_r_eo  = eo;
  return 1;
}

// One-shot consumption by the next depth-0 cycle: if the cycle input `eta` is
// the registered vector (same parity mode), hand back the prefused coarse RHS
// and skip the restrict.  Any other depth-0 cycle arriving with a live
// registration drops it (defensive -- registration and consumption are
// back-to-back in fgmres, so this should never fire) and restricts normally.
static int restrict_fused_consume(REAL * eta, REAL ** eta_c_out) {
  if (fuse_r_reg == NULL) return 0;
  int hit = (fuse_r_reg == eta) && (fuse_r_eo == g_eo_eta_in);
  fuse_r_reg = NULL;                    // one-shot either way
  if (!hit) {
    print0("[fuse-R] WARNING: dropped unconsumed registration\n");
    return 0;
  }
  *eta_c_out = fuse_r_buf;
  return 1;
}


// The post-smoother at level `lv`.
//
//   depth 0  : fine SAP with the residual fold (red_black_schwarz_resfold):
//              rhs is the cycle rhs eta, psi_inc is pre-loaded with P psi_c,
//              and the SAP seeds r = eta - D psi_inc block-locally in its
//              first sweep, accumulating psi_inc = P psi_c + M^nu r.  No
//              separate fine residual Dslash (or full-field r buffer) exists
//              at depth 0.
//   depth >=1: coarse SAP (coarse_sap) -- red-black multiplicative Schwarz on
//              the coarse stencil with a plain-MR block solve, matching
//              DDalphaAMG's coarse-level smoother; rhs is the pre-formed
//              residual r, psi_inc is overwritten with M_l^(nu) r.
//
// r_exit (NULL-able): receive the smoother's exit residual; only the
// coarse_sap path can provide it.  Returns 1 if r_exit was filled.
static int smooth_lvl(level_t * lv, REAL * psi_inc, REAL * rhs, REAL * r_exit) {
  struct timeval s0, s1; gettimeofday(&s0, NULL);
  if (lv->depth == 0) {
    // eo outer solve: the cycle's odd output is discarded at the bridge slice,
    // so the smoother's odd output writes are dead -- let SAP skip them
    // (terminal-write elimination, even output bit-identical).  eta is then
    // the even half-field, which the fold's seed passes handle directly.
    if (g_eo_eta_in) sap_discard_odd_output_next();
    red_black_schwarz_resfold(psi_inc, rhs, lv->U_kappa, lv->F_csw, lv->clover,
                              lv->nu, lv->blk_iter, 1e-12, g_eo_eta_in);
    gettimeofday(&s1, NULL); vc_smooth_fine += time_used(s0, s1);
    return 0;
  } else {
    // DDalphaAMG-style coarse red-black SAP
    coarse_sap(lv, psi_inc, rhs, r_exit);
    gettimeofday(&s1, NULL); vc_smooth_coarse[lv->depth] += time_used(s0, s1);
    return r_exit != NULL;              // coarse_sap always fills when asked
  }
}

// Coarsest-level bottom solve: x ~= D_l^{-1} b to the relative tolerance
// cg_tol * ||b||, with ||b|| accumulated in double.
//
// Dispatch on geometry: a level whose field_len equals the compile-time
// coarse_ferm_size is the level-1 macro-geometry coarse grid, so it takes the
// macro paths (even-odd Schur by default, full coarse_solve_full when
// MG_COARSE_EO == 0).  A deeper coarsest level has runtime geometry, and
// takes the eo-Schur solve when its op is built, else the geometry-agnostic
// coarse_solve_full_lvl (fgmres_coarse + apply_Dc_lvl).
#if BOTTOM_SOLVE_DIAG
// Diagnostic (BOTTOM_SOLVE_DIAG): tally the norm of the residual handed to
// the coarsest bottom solve.  ||b|| == 0 exactly => the restriction R = P^H
// annihilated the coarse-grid residual (a rank-deficient transfer);
// ||b|| > 0 with 0 iters => the eo-Schur reduction canceled it instead.
// bn is already globally reduced.
static long   g_bs_calls = 0, g_bs_zero = 0, g_bs_zeroiter = 0;
static double g_bs_min = 0, g_bs_max = 0;
static void bottom_solve_diag_tally(REAL bn, int its) {
  g_bs_calls ++;
  if (bn == 0.0) { g_bs_zero ++; return; }
  if (g_bs_max == 0.0 || bn < g_bs_min) g_bs_min = bn;
  if (bn > g_bs_max) g_bs_max = bn;
  if (its == 0) g_bs_zeroiter ++;
}
void bottom_solve_diag_reset(void) {
  g_bs_calls = g_bs_zero = g_bs_zeroiter = 0; g_bs_min = g_bs_max = 0;
  g_restrict_diag_shown = 0;
}
void bottom_solve_diag_report(void) {
  print0("[bottom-solve diag] %ld calls, %ld with ||b|| == 0 exactly, "
         "%ld nonzero-||b|| but 0 iters; ||b|| nonzero range [%.3e, %.3e]\n",
         g_bs_calls, g_bs_zero, g_bs_zeroiter, g_bs_min, g_bs_max);
}
#endif

// --- direct (dense LU) coarsest bottom solve (BOTTOM_DENSE_INV) ------------
#if BOTTOM_DENSE_INV
#include "dense.h"                        // LU factor/solve + the slab apply
// Solve the coarsest level exactly via a cached dense explicit inverse, in
// place of the iterative coarse_solve_full / coarse_solve_eo.  The inverse is
// computed in fp64 once per operator rebuild (probe -> zgetrf -> identity
// back-substitution) and stored fp64 or fp32 per BOTTOM_INV_FP32 (fp32 halves
// the footprint and the apply's stream traffic; its effective accuracy is
// ~1e-7 * kappa, printed by the build self-check).  Each bottom solve is then
// a slab GEMM against the cached inverse -- exact, and far cheaper than
// thousands of Krylov iterations.  The odd-Schur route
// (coarse_direct_build_eo) is the default; the full-operator route below
// serves a coarsest that cannot be eo-split.  Built only for the true bottom
// (depth == NUM_LEVELS-1) on an undecomposed lattice; a bottom too large to
// allocate simply falls back to the iterative solve.

// Inverse storage/apply precision (the factorization is always fp64).
#if BOTTOM_INV_FP32
typedef float complex  bot_cplx;
  #define BOT_APPLY dense_apply_f
#else
typedef double complex bot_cplx;
  #define BOT_APPLY dense_apply_d
#endif

#if BOTTOM_INV_FP32
  #define BOT_ALLREDUCE(p, n) allreduce_sum_f((float *) (p), (n))
#else
  #define BOT_ALLREDUCE(p, n) allreduce_sum_d((double *) (p), (n))
#endif

// One GEMM per row slab of the cached inverse: each thread computes its rows of
// X = INV * B for all num_rhs lanes at once, so the inverse streams through
// memory once per call (not once per lane) -- the multi-rhs reuse that matters
// when a large inverse is DRAM-resident.  The dense kernels are single-threaded
// by contract, so the slabs parallelize cleanly.  Rectangular by design: INV is
// N_out x N_in (col-major, leading dim N_out), B is N_in x num_rhs, X is
// N_out x num_rhs.  Single-rank N_out == N_in == N (the square full inverse);
// multi-rank N_out == this rank's local order and N_in == the global order
// (INV holds only this rank's rows).
static void bot_inv_apply(const bot_cplx * INV, long N_out, long N_in,
                          const bot_cplx * B, bot_cplx * X) {
  #pragma omp parallel
  {
    const int t = omp_get_thread_num(), T = omp_get_num_threads();
    const long r0 = N_out * (long) t / T, r1 = N_out * (long) (t + 1) / T;
    if (r1 > r0)
      BOT_APPLY(INV + r0, N_out, r1 - r0, N_in, num_rhs, B, X + r0);
  }
}

#if FERM_GHOSTS
static void * direct_explicit_inverse(double complex * A, long N, int depth);

// "Allgather" the local RHS block (N_loc x num_rhs col-major) into the global
// RHS (N_glob x num_rhs col-major, rank-concatenated:
// global row = rank * N_loc + local), via zero-fill + allreduce (comm.c has
// no allgather primitive; each rank owns a disjoint row range so the sum is a
// concatenation).  Bg must be caller-allocated, N_glob * num_rhs.  N_loc
// uniform across ranks (balanced decomposition).
static void bot_allgather_rhs(bot_cplx * Bg, const bot_cplx * Bloc,
                              long N_glob, long N_loc) {
  memset(Bg, 0, sizeof(bot_cplx) * N_glob * (long) num_rhs);
  const long r0 = (long) comm_rank() * N_loc;
  for (long l = 0; l < num_rhs; l ++)
    for (long k = 0; k < N_loc; k ++)
      Bg[r0 + k + l * N_glob] = Bloc[k + l * N_loc];
  BOT_ALLREDUCE(Bg, 2 * N_glob * (long) num_rhs);
}

// From the redundantly-factored global inverse (identical on every rank), keep
// only this rank's rows: loc[k + j*N_loc] = INV[(rank*N_loc + k) + j*N_glob],
// an N_loc x N_glob col-major slab.  Consumes (frees) the assembled global
// operator via direct_explicit_inverse; returns the local-row inverse
// (bot_cplx), or NULL on OOM/singular.  At the sizes this runs (a tiny bottom
// level) the transient full N_glob^2 inverse is small; the stored slab is
// N_glob^2 / nranks per rank.
static void * dist_inverse_local_rows(double complex * A_glob, long N_glob,
                                      long N_loc, int depth) {
  void * INVv = direct_explicit_inverse(A_glob, N_glob, depth);  // frees A_glob
  if (!INVv) return NULL;
  const bot_cplx * INV = (const bot_cplx *) INVv;
  bot_cplx * loc = alloc_try(sizeof(bot_cplx) * N_loc * N_glob);
  if (!loc) { free(INVv); return NULL; }
  const long r0 = (long) comm_rank() * N_loc;
  #pragma omp parallel for schedule(static)
  for (long j = 0; j < N_glob; j ++)
    for (long k = 0; k < N_loc; k ++)
      loc[k + j * N_loc] = INV[(r0 + k) + j * N_glob];
  free(INVv);
  return loc;
}
#endif

// LU-factor the assembled col-major operator A (order N) in fp64 and expand
// its explicit inverse by back-substituting identity columns, num_rhs at a
// time in parallel -- dense_getrs only reads the shared factors, so the
// chunks are independent.  Under BOTTOM_INV_FP32 the fp64 result is
// then rounded to fp32 for storage (the apply consumes bot_cplx).  Consumes
// (frees) A; returns the inverse, or NULL when singular / out of memory (the
// caller stays iterative).
static void * direct_explicit_inverse(double complex * A, long N, int depth) {
  int * piv = alloc_try(sizeof(int) * N);
  double complex * INV = alloc_try(sizeof(double complex) * N * N);
  if (!piv || !INV) { free(A); free(piv); free(INV); return NULL; }
  int info = dense_getrf(A, N, piv);
  if (info) {
    print0("coarse_direct[depth %d]: dense factor singular (info=%ld) -- "
           "iterative solve\n", depth, (long) info);
    free(A); free(piv); free(INV); return NULL;
  }
  memset(INV, 0, sizeof(double complex) * N * N);
  for (long j = 0; j < N; j ++) INV[j + j * N] = 1.0;
  const long nchunk = (N + num_rhs - 1) / num_rhs;
  #pragma omp parallel for schedule(dynamic)
  for (long c = 0; c < nchunk; c ++) {
    long j0 = c * num_rhs;
    long nc = N - j0 < num_rhs ? N - j0 : num_rhs;
    dense_getrs(A, N, piv, INV + j0 * N, nc);
  }
  free(A); free(piv);
#if BOTTOM_INV_FP32
  {
    float complex * S = alloc_try(sizeof(float complex) * N * N);
    if (!S) { free(INV); return NULL; }
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < N * N; i ++) S[i] = (float complex) INV[i];
    free(INV);
    return S;
  }
#else
  return INV;
#endif
}

// Assemble the odd-site Schur complement S = A_oo - D_oe A_ee^{-1} D_eo by
// probing apply_coarse_schur with unit odd-half columns -- a quarter of the
// full matrix and an eighth of the factor flops -- then invert.  At solve
// time coarse_solve_eo's existing elimination/recovery wraps the cached
// S^{-1} (its inner GMRES becomes one dense apply), so the eo machinery is
// reused end to end.  Requires the eo op, built just above in
// mg_add_coarse_level.
static int coarse_direct_build_eo(level_t * lv) {
  coarse_eo_op * eo = &lv->eo;
  const long volh = eo->volh, nvar = eo->nvar;
  const long Nloc = volh * nvar;         // this rank's odd-half complex DOF
  if (Nloc <= 0) return 0;
#if FERM_GHOSTS
  const long P = comm_size(), myrank = comm_rank();
#else
  const long P = 1, myrank = 0;
#endif
  const long Nglob = Nloc * P;           // global odd-half DOF
  const long r0 = myrank * Nloc;         // this rank's global row offset
  const long blk  = nvar * 2 * num_rhs;
  const long half = volh * blk;
  // col-major
  const size_t Asz = sizeof(double complex) * (size_t) Nglob * Nglob;
  double complex * A = alloc_try(Asz);   // as above: recoverable, so no abort
  REAL * in  = alloc_try(sizeof(REAL) * half);
  REAL * out = alloc_try(sizeof(REAL) * half);
  if (!A || !in || !out) { free(A); free(in); free(out); return 0; }
  memset(A, 0, Asz);

  for (long J0 = 0; J0 < Nglob; J0 += num_rhs) {
    long owner = J0 / Nloc;              // whose odd column this batch probes
    memset(in, 0, sizeof(REAL) * half);
    if (owner == myrank)
      for (int l = 0; l < num_rhs; l ++) {
        long J = J0 + l; if (J >= Nglob) break;
        long Jl = J - owner * Nloc, o = Jl / nvar, a = Jl % nvar;
        // unit odd column, owner only
        in[o * blk + (a * 2) * num_rhs + l] = (REAL) 1.0;
      }
    apply_coarse_schur(eo, out, in);             // collective (halo-aware)
    for (int l = 0; l < num_rhs; l ++) {
      long J = J0 + l; if (J >= Nglob) break;
      for (long q = 0; q < volh; q ++)           // this rank's local odd rows
        for (long b = 0; b < nvar; b ++) {
          long K = q * nvar + b;
          A[(r0 + K) + J * Nglob]
            = (double) out[q * blk + (b * 2) * num_rhs + l]
              + (double) out[q * blk + (b * 2 + 1) * num_rhs + l] * I;
        }
    }
  }
  free(in); free(out);

#if FERM_GHOSTS
  // assemble global S on every rank
  allreduce_sum_d((double *) A, 2 * Nglob * Nglob);
  void * inv = dist_inverse_local_rows(A, Nglob, Nloc, lv->depth);
#else
  void * inv = direct_explicit_inverse(A, Nglob, lv->depth);
#endif
  if (!inv) return 0;
  lv->bot_inv = inv; lv->bot_N = Nglob;
  // borrowed by coarse_solve_eo's dense core
  eo->dense_inv = inv; eo->dense_N = Nglob;
  return 1;
}

// Build the dense bottom inverse, then self-check through the actual solve
// route: x = solve(D r) must return r for a deterministic pseudo-random r
// (no Math.random in this TU).  Loud on failure, one quiet line the first
// time it activates.  Always the odd-Schur route: mg_add_coarse_level rejects
// a coarsest that cannot eo-split before this runs, so `eo_built` is an
// invariant here rather than a test.
static int coarse_direct_build(level_t * lv) {
  int ok = coarse_direct_build_eo(lv);
#if FERM_GHOSTS
  // The redundant factor is deterministic (every rank inverts the identical
  // global operator), so a singular bottom fails on all ranks together; only
  // an asymmetric malloc failure could diverge.  Reduce first so every rank
  // agrees on the outcome before any of them acts on it.
  {
    double okd = ok ? 1.0 : 0.0;
    allreduce_min_d(&okd, 1);
    if (ok && okd == 0.0)
      coarse_direct_free(lv);   // built here, but a peer did not
    ok = (okd != 0.0);
  }
#endif
  // A failed build stops the run, at one rank and at many alike.  Falling
  // back to the iterative solve would be slowest in exactly the case this
  // flag was set to avoid, and it would hide what is worth looking at: a
  // singular coarsest operator, or an inverse too large to allocate.  The
  // specific diagnosis is already printed above.
  if (!ok) {
    comm_die("coarse_direct[depth %d]: dense bottom build failed -- aborting.  "
             "Rebuild with -DBOTTOM_DENSE_INV=0 to take the iterative "
             "bottom solve on purpose.", lv->depth);
  }
  const long fld = lv->vol * (long) lv->nvar * 2 * num_rhs;
  REAL * r  = malloc(sizeof(REAL) * fld);
  REAL * Dr = malloc(sizeof(REAL) * fld);
  REAL * xr = malloc(sizeof(REAL) * fld);
  for (long i = 0; i < fld; i ++)
    r[i] = (REAL) (((i * 2654435761UL + 1013904223UL) >> 8) & 0xffff)
           / 32768.0f - 1.0f;
  apply_Dc_lvl(lv->Dc, Dr, r, lv);
  coarse_solve_eo(&lv->eo, xr, Dr, 1, (REAL) 0.0, 1);
  double num = 0, den = 0;
  for (long i = 0; i < fld; i ++) {
    double d = (double) xr[i] - (double) r[i];
    num += d * d; den += (double) r[i] * r[i];
  }
  double nd[2] = { num, den };
  allreduce_sum_d(nd, 2);        // global self-check residual (no-op @1 rank)
  double resid = nd[1] > 0 ? sqrt(nd[0] / nd[1]) : 0.0;
  free(r); free(Dr); free(xr);
  if (resid > 1e-4)
    print0("coarse_direct[depth %d]: WARNING self-check %.2e (assembly/inverse suspect)\n",
           lv->depth, resid);
  else {
    static int announced = 0;
    if (!announced) {
      print0("coarse_direct[depth %d]: exact dense inverse active (odd-Schur, "
             "N=%ld), self-check %.2e\n", lv->depth, lv->bot_N, resid);
      announced = 1;
    }
  }
  return 1;
}

static void coarse_direct_free(level_t * lv) {
  free(lv->bot_inv);
  lv->bot_inv = NULL;
  lv->bot_N = 0;
  lv->eo.dense_inv = NULL;  // borrowed reference into bot_inv
  lv->eo.dense_N = 0;
}
#endif  // BOTTOM_DENSE_INV

static int bottom_solve(level_t * lv, REAL * x, REAL * b) {
  long n = lv->field_len * num_rhs;
  double bn2 = 0;
  for (long i = 0; i < n; i ++) bn2 += (double) b[i] * b[i];
  // ||b|| must be global: tol feeds fgmres_coarse's eps, and its
  // happy-breakdown (cabs(g[m+1]) < eps) decides the inner-loop iteration
  // count.  A rank-local bn would give each rank a different eps ->
  // different break point -> mismatched collective counts -> multi-rank
  // deadlock.  No-op / byte-identical at 1 rank.
  allreduce_sum_d(&bn2, 1);
  REAL bn = sqrt(bn2);
  REAL tol = lv->cg_tol * bn;
  int its;
#if BOTTOM_DENSE_INV
  if (lv->bot_N)                            // exact dense coarsest (if built)
    // reuse coarse_solve_eo's gather/eliminate/recover shell; its inner GMRES
    // is replaced by one dense apply against the cached S^{-1} (eo.dense_inv)
    its = coarse_solve_eo(&lv->eo, x, b, 1, (REAL) 0.0, 1);
  else
#endif
  if (lv->depth == 1 && lv->field_len == coarse_ferm_size) {
    // level-1 macro geometry (same double test as apply_D_lvl: depth alone
    // would macro-route a synthetic non-macro depth-1 level)
    its = MG_COARSE_EO
        ? coarse_solve_eo(&lv->eo, x, b, lv->cg_max_iter, tol, lv->cg_restart)
        : coarse_solve_full(lv->Dc, x, b, lv->cg_max_iter, tol, lv->cg_restart);
  } else if (MG_COARSE_EO && lv->eo_built) {
    // Deeper coarsest (runtime geometry): prefer the even-odd Schur solve
    // (half the unknowns, better conditioned) when its op is built; a
    // degenerate-parity coarsest leaves eo_built == 0 and falls back to the
    // full op.
    its = coarse_solve_eo(&lv->eo, x, b, lv->cg_max_iter, tol, lv->cg_restart);
  } else {
    // Non-eo fallback (odd-extent coarsest): coarse_solve_full_lvl runs
    // apply_Dc_lvl, which is now ghost-aware (halo_coarse into lv->dc_ghost),
    // and fgmres_coarse's reductions are all allreduced -- so this path is
    // cross-rank correct.
    its = coarse_solve_full_lvl(lv, x, b, lv->cg_max_iter, tol, lv->cg_restart);
  }
#if BOTTOM_SOLVE_DIAG
  bottom_solve_diag_tally(bn, its);
#endif
  return its;
}

// The cycle helpers below recurse back into mg_cycle (declared in coarse.h),
// and mg_cycle's coarse-correction step dispatches through coarse_correction.
// mg_cycle_rexit is the extended entry that also returns the cycle's exit
// residual (consumed by k_cycle's assembled matvec, KCYCLE_ASSEMBLED_W).
static int mg_cycle_rexit(level_t * lv, REAL * psi, REAL * eta, REAL k_tol,
                          REAL * r_exit);

// W-cycle coarse correction: solve D_c psi_c = eta_c by w_mu stationary
// recursive cycles, recomputing the coarse residual between them (the
// mu-cycle of thesis Alg. 17 / Trottenberg et al.).  w_mu = 1 reduces exactly
// to the V-cycle's single recursive call (no extra D_c apply); w_mu = 2 is
// the W-cycle.  `cl` is the child (coarse) level; psi_c/eta_c live in its
// field space.  Returns inner iters summed.
static int w_cycle(level_t * cl, REAL * psi_c, REAL * eta_c, REAL k_tol) {
  long cn = cl->field_len * num_rhs;
  int mu = cl->w_mu < 1 ? 1 : cl->w_mu;

  REAL * dpsi = malloc(sizeof(REAL) * cn);
  REAL * r    = (mu > 1) ? malloc(sizeof(REAL) * cn) : NULL;
  REAL * Dp   = (mu > 1) ? malloc(sizeof(REAL) * cn) : NULL;

  #pragma omp parallel for
  for (long i = 0; i < cn; i ++) psi_c[i] = 0;
  REAL * res = eta_c;            // first sub-cycle: residual == eta_c (psi=0)
  int total = 0;
  for (int k = 0; k < mu; k ++) {
    total += mg_cycle(cl, dpsi, res, k_tol);     // dpsi ~= D_c^{-1} res
    #pragma omp parallel for
    for (long i = 0; i < cn; i ++) psi_c[i] += dpsi[i];
    if (k < mu - 1) {                            // recompute coarse residual
      apply_D_lvl(cl, Dp, psi_c);
      #pragma omp parallel for
      for (long i = 0; i < cn; i ++) r[i] = eta_c[i] - Dp[i];
      res = r;
    }
  }

  free(dpsi); free(r); free(Dp);
  return total;
}

// K-cycle coarse correction (Notay): FGMRES on D_c psi_c = eta_c, with the
// variable preconditioner M = one recursive mg_cycle on the child level `cl`.
// One cycle of up to cl->k_iters Arnoldi steps, exiting early once the
// estimated residual falls below k_tol*||eta_c||.  The Arnoldi / Givens /
// solution-update machinery is fgmres_coarse's, run in its flexible mode, so
// every level of the hierarchy -- outer solve, K-cycle wrapper, coarsest
// bottom solve -- uses the same Krylov method and inherits the same
// near-breakdown guards.  That matters here: a coarse operator carries
// near-null modes by construction, exactly the regime those guards were
// written for.
//
// The matvec is assembled, not applied (KCYCLE_ASSEMBLED_W): the recursive
// cycle hands back its exit residual rex = v - D_c z, so w = D_c z is
// (v - rex), replacing a full coarse Dslash per Arnoldi step with a vector
// subtraction.  Algebraically exact; rex carries the smoother's accumulated
// rounding, so w differs from an explicit apply at rounding level -- judge
// by the outer iteration count, not bitwise.
typedef struct {
  level_t * cl;
  REAL k_tol;
  REAL * rex;    // exit-residual scratch; NULL => apply the operator explicitly
  long cn;
  int iters;  // bottom-solve iterations accumulated by the recursive cycles
} kc_ctx;

// fgmres_prec_fn for the K-cycle: one recursive cycle plus the assembled
// matvec. Re-entrant -- all state is in the caller's kc_ctx, so a level-1
// wrapper may be running one of these while the level-2 wrapper inside it
// runs its own.
static void kc_precondition(void * pctx, REAL * z, REAL * w, REAL * v) {
  kc_ctx * kc = (kc_ctx *) pctx;
  // z ~= D_c^{-1} v
  kc->iters += mg_cycle_rexit(kc->cl, z, v, kc->k_tol, kc->rex);
  if (kc->rex) {
    struct timeval ka, kb;
    gettimeofday(&ka, NULL);
    #pragma omp parallel for
    for (long i = 0; i < kc->cn; i ++) w[i] = v[i] - kc->rex[i];  // w = D_c z
    gettimeofday(&kb, NULL);
    kc_gs_secs[kc->cl->depth]  += time_used(ka, kb);
    kc_gs_flops[kc->cl->depth] += (double) kc->cn;
  } else {
    apply_D_lvl(kc->cl, w, z);  // w = D_c z (timed by dc_secs)
  }
}

static int k_cycle(level_t * cl, REAL * psi_c, REAL * eta_c, REAL k_tol) {
  long cn = cl->field_len * num_rhs;
  int  m  = cl->k_iters < 1 ? 1 : cl->k_iters;

  // Entry norm, for the relative exit test.  cdot_ferm allreduces, so this is
  // the global norm -- required, not cosmetic: it sets the eps that
  // fgmres_coarse's happy-breakdown compares against, and a rank-local value
  // would give each rank a different break point, hence mismatched collective
  // counts and a multi-rank deadlock.  (Same reasoning as bottom_solve's
  // entry norm.)
  REAL bnorm = sqrt(creal(cdot_ferm(eta_c, eta_c, cn)));
  if (bnorm == 0) {
    #pragma omp parallel for
    for (long i = 0; i < cn; i ++) psi_c[i] = 0;
    kc_calls ++;
    return 0;
  }

  kc_ctx kc = { cl, k_tol,
                KCYCLE_ASSEMBLED_W ? malloc(sizeof(REAL) * cn) : NULL,
                cn, 0 };
  // This row is the Gram-Schmidt and solution-update work only.  The
  // recursive cycles behind the hook are timed by the V-cycle rows, and the
  // entry norm / basis setup stay unbilled and fall into the profile's
  // remainder.  The operator callback is unused: the flexible path never
  // applies it, and the unassembled matvec (when KCYCLE_ASSEMBLED_W is off)
  // is issued inside the hook.
  fgmres_bill bill = { NULL, &kc_gs_secs[cl->depth], &kc_gs_flops[cl->depth],
                       NULL, NULL, NULL };
  fgmres_coarse(NULL, NULL, kc_precondition, &kc, bill,
                psi_c, eta_c, cn, m, k_tol * bnorm, m);
  kc_calls ++;

  free(kc.rex);
  return kc.iters;
}

// Coarse-grid correction at level `lv`: produce psi_c ~= D_c^{-1} eta_c on
// the child level `cl`, using the active cycle type.  W/K acceleration
// applies only where there is genuine recursion to accelerate: if `cl` is
// itself the coarsest level its mg_cycle is the (already accurate) bottom
// solve, so call it directly.
static int coarse_correction(level_t * cl, REAL * psi_c, REAL * eta_c,
                             REAL k_tol) {
  if (cl->coarser == NULL || cl->cycle == MG_CYCLE_V)
    return mg_cycle(cl, psi_c, eta_c, k_tol);
  if (cl->cycle == MG_CYCLE_W)
    return w_cycle(cl, psi_c, eta_c, k_tol);
  return k_cycle(cl, psi_c, eta_c, k_tol);        // MG_CYCLE_K
}

// One multigrid cycle at level `lv`: approximately solve D_l psi = eta with a
// zero initial guess (so it serves as a preconditioner).  Post-smoothing only
// (thesis Algorithm 8 / Alg. 17 with mu = 0).  Returns the bottom-solve iters.
//
// The coarse-grid correction recurses through coarse_correction, which
// selects a V- (one recursive cycle), W- (w_mu stationary cycles), or
// K-cycle (flexible GMRES around the recursive cycle) per the child level's
// `cycle` field.  Extended entry: if r_exit != NULL it receives
// eta - D_l psi for the final psi (lv->field_len * num_rhs reals).  Nearly
// free on the coarse_sap smoother path (its incremental-residual state
// already holds it); a path that cannot provide it cheaply falls back to one
// explicit residual apply, so r_exit is always filled when requested.  The
// K-cycle uses it to assemble its Arnoldi matvec w = D_c z as (v - r_exit)
// without a coarse Dslash.
static int mg_cycle_rexit(level_t * lv, REAL * psi, REAL * eta, REAL k_tol,
                          REAL * r_exit) {
  // base case: coarsest grid -> bottom Krylov solve
  if (lv->coarser == NULL) {
    int cits = bottom_solve(lv, psi, eta);
    if (r_exit)
      apply_D_residual_lvl(lv, r_exit, eta, psi);  // backstop (unused)
    return cits;
  }

  level_t * cl = lv->coarser;
  long fn = lv->field_len * num_rhs;
  long cn = cl->field_len * num_rhs;

  // Fused-restrict consumption: the outer fgmres may have already produced
  // R eta while normalizing eta (mg_restrict_fuse_normalize); then eta_c is
  // that registered buffer -- not malloc'd, not freed here -- and the restrict
  // below is skipped (its wall was charged to vc_R by the fused pass itself).
  REAL * eta_c = NULL;
  int eta_c_fused = (lv->depth == 0) && restrict_fused_consume(eta, &eta_c);
  if (!eta_c_fused) eta_c = malloc(sizeof(REAL) * cn);   // R eta
  REAL * psi_c = malloc(sizeof(REAL) * cn);   // D_c^{-1} R eta
  // depth 0 folds both the residual formation and the post-smoothing add into
  // the fine SAP (it seeds r = eta - D psi block-locally and accumulates into
  // psi), so neither the full-field r buffer nor smth exists there; coarse
  // levels still form r explicitly and smooth-then-add.
  REAL * r     = (lv->depth == 0) ? NULL : malloc(sizeof(REAL) * fn);
  REAL * smth  = (lv->depth == 0) ? NULL : malloc(sizeof(REAL) * fn);

  struct timeval a, b;
  // Coarse-grid correction.  Initial guess psi = 0, so the residual is `eta`.
  gettimeofday(&a, NULL);
  if (!eta_c_fused)
    restrict_lvl(lv, eta_c, eta);               // eta_c = R eta
  gettimeofday(&b, NULL); vc_R[lv->depth] += time_used(a, b);

  gettimeofday(&a, NULL);
  // psi_c ~= D_c^{-1} eta_c (V/W/K)
  int cits = coarse_correction(cl, psi_c, eta_c, k_tol);
  gettimeofday(&b, NULL); vc_coarse += time_used(a, b);

  gettimeofday(&a, NULL);
  prolong_lvl(lv, psi, psi_c);                  // psi = P psi_c
  gettimeofday(&b, NULL); vc_P[lv->depth] += time_used(a, b);

  // Post-smoothing (multiplicative cycle).  At depth 0 the residual formation
  // is folded into the fine SAP: psi already holds P psi_c, and the smoother
  // seeds r = eta - D psi block-locally in its first sweep -- riding the
  // block-resident gauge/clover/guess streams its block solve reuses anyway --
  // then accumulates psi = P psi_c + M^nu r.  No separate fine residual
  // Dslash, no full-field r buffer, no `psi += smth` streaming loop.  Coarse
  // levels recompute r = eta - D psi explicitly (the subtraction fused into
  // the Dslash) and smooth-then-add.
  int rexit_filled = 0;
  if (lv->depth == 0) {
    gettimeofday(&a, NULL);
    smooth_lvl(lv, psi, eta, NULL);
    gettimeofday(&b, NULL); vc_smooth += time_used(a, b);
  } else {
    gettimeofday(&a, NULL);
    apply_D_residual_lvl(lv, r, eta, psi);
    gettimeofday(&b, NULL); vc_D += time_used(a, b);

    gettimeofday(&a, NULL);
    // The smoother solves D smth ~= r with r = eta - D(P psi_c), so its exit
    // residual r - D smth equals eta - D psi for the final psi below.
    rexit_filled = smooth_lvl(lv, smth, r, r_exit);
    gettimeofday(&b, NULL); vc_smooth += time_used(a, b);

    gettimeofday(&a, NULL);
    #pragma omp parallel for
    for (long i = 0; i < fn; i ++) psi[i] += smth[i];
    gettimeofday(&b, NULL); vc_vec[lv->depth] += time_used(a, b);
    vc_vec_flops[lv->depth] += (double) fn;   // fn additions (1 flop/real)
  }
  if (r_exit && !rexit_filled)                // smoother couldn't provide it:
    apply_D_residual_lvl(lv, r_exit, eta, psi);  // one explicit apply
                                                 // (correct, just no saving)

  vc_calls ++; vc_citers += cits;
  if (lv->depth == 0) vc_calls_fine ++;
  if (!eta_c_fused) free(eta_c);
  free(psi_c); free(r); free(smth);
  return cits;
}

int mg_cycle(level_t * lv, REAL * psi, REAL * eta, REAL k_tol) {
  return mg_cycle_rexit(lv, psi, eta, k_tol, NULL);
}

// Copy the cycle shape onto a level.  Uniform across the hierarchy: the
// recursion is handed a level rather than the parameters, so every level
// carries its own copy of what the cycle needs to read.
static void set_level_cycle(level_t * lv, const mg_params * p) {
  lv->cycle       = p->cycle;
  lv->w_mu        = p->w_mu;
  lv->k_iters     = p->k_iters;
  lv->solve_k_tol = p->solve_k_tol;
}

void mg_setup(coarse_op * Dc, REAL * nv, const mg_params * p) {
  // Tear down any previously-built hierarchy, then rebuild two levels deep.
  mg_teardown();
  g_num_levels = 2;

  // Level 0: the fine Wilson-clover grid.  Its operator handles
  // (U_kappa/F_csw/clover) are only known at apply time, so mg_vcycle patches
  // them in; geometry, the interpolation columns, and the smoother sweep
  // count are fixed here.
  level_t * f = &g_levels[0];
  f->depth     = 0;
  f->nx = Lx; f->ny = Ly; f->nz = Lz; f->nt = Lt;
  f->vol = V; f->volh = Vh;
  for (int d = 0; d < 4; d ++)
    f->corigin[d] = qc_grid.origin[d];  // fine origin
  f->nvar      = Nc * Ns;
  f->nnv       = num_null_vecs;   // L0->L1 coarsening column count
  f->field_len = (long) ferm_size;
  // fine aggregation
  f->blk[0] = f->blk[1] = f->blk[2] = f->blk[3] = BLOCK_AGG;
  f->nv        = nv;
  f->nu        = p->smoother_sweeps;
  f->blk_iter  = p->sap_block_iter;   // fine SAP's per-block MR count
  f->coarser   = &g_levels[1];
  set_level_cycle(f, p);

  // Level 1: the coarse grid; owns the operator that the bottom solve inverts.
  level_t * c = &g_levels[1];
  c->depth       = 1;
  c->nx = Lxc; c->ny = Lyc; c->nz = Lzc; c->nt = Ltc;
  c->vol = Vc; c->volh = Vhc;
  for (int d = 0; d < 4; d ++) c->corigin[d] = qc_grid.origin[d] / BLOCK_AGG;
  c->nvar        = N_coarse;      // = 2*num_null_vecs (L0->L1 width)
  // L1->L2 coarsening column count (first coarse transition);
  // bootstrap re-sets it
  {
    int cnv_lvl[] = COARSE_NULL_VECS_LVL;
    c->nnv = cnv_lvl[0];
  }
  c->field_len   = coarse_ferm_size;
#if FERM_GHOSTS
  // Depth 1 normally uses the macro apply_Dc (static wk_cghost), but a
  // degenerate depth-1->2 Galerkin build probes apply_Dc_lvl on this level,
  // so give it the per-level scratch too -- apply_Dc_lvl is then
  // ghost-correct at every depth.
  alloc_level_dc_ghost(c);
#endif
  c->Dc          = Dc;
  build_coarse_eo_op(&c->eo, Dc, NULL);   // NULL lv: depth-1 macro geometry
  c->eo_built    = 1;
  c->cg_max_iter = p->coarse_max_iter;
  c->cg_tol      = (REAL) p->coarse_tol;
  c->cg_restart  = p->coarse_restart;
  c->coarser     = NULL;
  set_level_cycle(c, p);
}

// Append one coarser level to the hierarchy: the current coarsest level
// becomes an intermediate level (gains the aggregation block `blk`, its
// interpolation columns `nvc`, and a `nu`-sweep smoother), and a
// freshly-built D_{l+1} = R_l D_l P_l grid is added below it as the new
// coarsest (with bottom-solve controls cg_*).  Must be called after mg_setup
// (which seeds levels 0,1): once for a 3-level cycle, repeatedly for deeper.
// `nvc` is owned by the caller (kept alive) and is locally orthonormalized
// in place here, matching the fine convention.
void mg_add_coarse_level(REAL * nvc, int blk[4], int n_null,
                         const mg_params * p) {
  int li = g_num_levels - 1;             // current coarsest -> intermediate
  level_t * up = &g_levels[li];
  level_t * lo = &g_levels[li + 1];      // new coarsest

  // up is no longer the bottom: drop its bottom-solve eo op, give it
  // transfer/smoother data, a child, and its coarsening count.  `n_null` is
  // the number of interpolation columns in nvc (= the test vectors coarsening
  // `up`); it becomes up->nnv, which local_orthonormalize_coarse /
  // build_coarse_op_lvl below and the solve's apply_R_coarse(fine_lv=up) all
  // read.  At depth 1 this equals the value mg_setup already stored; deeper
  // levels are created coarsest (nnv=0, memset below) and only get their
  // count here, when a further add promotes them -- so this assignment is
  // what makes hierarchies past 3 levels well-formed.
  if (up->eo_built) { free_coarse_eo_op(&up->eo); up->eo_built = 0; }
  up->blk[0] = blk[0]; up->blk[1] = blk[1];
  up->blk[2] = blk[2]; up->blk[3] = blk[3];
  up->nnv = n_null;
  up->nv = nvc;
  up->nu = p->coarse_smoother_sweeps;
  up->blk_iter = p->coarse_sap_block_iter;
  up->coarser = lo;
  // SAP smoother block = the aggregation block.  In DDalphaAMG a single
  // per-level "block_lattice" parameter does double duty: it defines the
  // Schwarz/SAP domains and the aggregation (the next coarser lattice is this
  // one divided by it -- see init.c,
  // global_lattice[i] = global_lattice[i-1]/block_lattice[i-1], and
  // schwarz_generic.c, block_size[mu] = block_lattice[mu]).  So the smoother
  // block is exactly `blk`, the factor tiling this level into its child.
  // (At depth 0 the fine SAP's SCHWARZ_BLOCK_SIZE likewise matches the fine
  // aggregation BLOCK_AGG.)
  up->sap_blk[0] = blk[0]; up->sap_blk[1] = blk[1];
  up->sap_blk[2] = blk[2]; up->sap_blk[3] = blk[3];

  local_orthonormalize_coarse(nvc, up);  // ensure P_l^H P_l = I (idempotent)

  // Build and store the new coarse operator.  (Named Dnew, not D: the global
  // symbol-mangling layer rewrites the bare identifier D to the Dslash D_f/D_d,
  // so a local called D here would shadow the operator within this function.)
  coarse_op * Dnew = &g_built_ops[li + 1];
  build_coarse_op_lvl(Dnew, up, nvc);

  memset(lo, 0, sizeof(*lo));
  lo->depth      = li + 1;
#if FERM_GHOSTS
  // The level->child transfers (apply_R/P_coarse) and the coarse-SAP blocking
  // are aggregate-local, so they stay comm-free only if rank faces align to
  // aggregate (= SAP block) boundaries: `up`'s local decomposed extent must
  // tile into whole aggregates and this rank's origin in `up`'s sites must
  // land on an aggregate boundary.  up->corigin is up's origin in its own
  // sites at any depth (fine origin / BLOCK_AGG at depth 1, recursively
  // divided deeper), so the same check serves every level -- no depth cap.
  if (DECOMP_Z || DECOMP_T) {
    int bad = (DECOMP_Z && (up->nz % blk[2] || up->corigin[2] % blk[2]))
            || (DECOMP_T && (up->nt % blk[3] || up->corigin[3] % blk[3]));
    if (bad) {
      comm_die("mg_add_coarse_level[depth %d]: rank face not aggregate-aligned "
               "(local nz,nt=%d,%d  origin z,t=%d,%d  blk=%d,%d) -- coarse "
               "transfers/SAP would straddle a rank boundary", up->depth,
               up->nz, up->nt, up->corigin[2], up->corigin[3], blk[2], blk[3]);
    }
  }
#endif
  lo->nx = up->nx / blk[0]; lo->ny = up->ny / blk[1];
  lo->nz = up->nz / blk[2]; lo->nt = up->nt / blk[3];
  // this level's coordinate origin in its own sites (parent origin / blk;
  // exact because the alignment guard above forces org % blk == 0 on
  // decomposed dims)
  for (int d = 0; d < 4; d ++) lo->corigin[d] = up->corigin[d] / blk[d];
  lo->vol        = (long) lo->nx * lo->ny * lo->nz * lo->nt;
  lo->volh       = lo->vol / 2;
  // The new level must be even-odd splittable: every global extent even or 1
  // (see build_coarse_eo_op -- an extent of 1 folds into `self` and is no
  // obstruction), and the volume even so the two parities are equal.  Only
  // an all-ones lattice meets the first and fails the second, and it earns
  // its own error: one coarse site means the parent is a single SAP block,
  // which the smoother already solves exactly, so this level would correct
  // nothing.  Checked here rather than in build_coarse_eo_op because that
  // runs after coarse_direct_build below, which has no non-eo route to fall
  // back to.
  {
    long g[4] = { (long) lo->nx * qc_grid.P[0], (long) lo->ny * qc_grid.P[1],
                  (long) lo->nz * qc_grid.P[2], (long) lo->nt * qc_grid.P[3] };
    for (int d = 0; d < 4; d ++)
      if (g[d] > 1 && (g[d] & 1))
        comm_die("mg_add_coarse_level[depth %d]: coarse extent (%ld %ld %ld %ld) is "
                 "odd and > 1 in dim %d -- an odd periodic extent cannot be "
                 "even-odd split.  Choose an aggregation block that divides "
                 "the parent extent into an even count (or into exactly 1).",
                 lo->depth, g[0], g[1], g[2], g[3], d);
    if (g[0] * g[1] * g[2] * g[3] == 1)
      comm_die("mg_add_coarse_level[depth %d]: coarse lattice is a single site -- "
               "its parent is then one SAP block, which the smoother solves "
               "exactly, so this level would correct nothing.  Use one level "
               "fewer or an aggregation block that leaves more than one site.",
               lo->depth);
  }
  // This level's width is 2 x the null-vector count used to coarsen its
  // parent (n_null == up->nnv, set above), which may differ from the L0->L1
  // width N_coarse when COARSE_NULL_VECS != num_null_vecs.  lo->nnv stays 0
  // (memset above): it is the coarsest until a further mg_add_coarse_level
  // makes it a parent and passes its own coarsening count.
  lo->nvar       = 2 * n_null;
  lo->field_len  = lo->vol * (long) lo->nvar * 2;
#if FERM_GHOSTS
  alloc_level_dc_ghost(lo);      // this level's runtime coarse-Dslash halo scratch
#endif
  lo->Dc         = Dnew;
  lo->cg_max_iter = p->coarse_max_iter;
  lo->cg_tol      = (REAL) p->coarse_tol;
  lo->cg_restart  = p->coarse_restart;
  // The memset above cleared these; a promoted level keeps what it already had.
  set_level_cycle(lo, p);
  lo->coarser    = NULL;

  // Build the even-odd Schur op for this new coarsest level so its bottom
  // solve runs on the half-size, better-conditioned odd system (matching
  // DDalphaAMG's odd-even-preconditioned coarsest GMRES).  Guarded by
  // balanced parity: a degenerate coarsest (odd extent) returns -1 and
  // bottom_solve falls back to the full coarse_solve_full_lvl.  Gated at
  // solve time by MG_COARSE_EO.
  lo->eo_built = (build_coarse_eo_op(&lo->eo, Dnew, lo) == 0) ? 1 : 0;

#if BOTTOM_DENSE_INV
  // Exact dense-LU bottom solve, built only for the true bottom (this level
  // is the coarsest of the fully-built hierarchy, not a transient
  // intermediate during recursive setup).  Under decomposition it is
  // row-distributed: each rank probes the global operator collectively,
  // factors it redundantly, and keeps its own rows of the inverse
  // (coarse_direct_build_eo/full).  Supersedes the iterative bottom solve.
  if (lo->depth == NUM_LEVELS - 1) coarse_direct_build(lo);
#endif

  g_num_levels = li + 2;

  // `up` just became a smoothed coarse level (Dc + sap_blk + child all set),
  // so build its coarse-SAP cache now (the one-time See_inv block inversion):
  // a setup cost, amortized across every solve on this hierarchy, instead of
  // leaking into the first solve's coarse-SAP timer.
  // phi == NULL = warm-only (build cache, then return).
  coarse_sap(up, NULL, NULL, NULL);
}

int mg_vcycle(REAL * output, REAL * input, REAL * U_kappa, REAL * F_csw,
              int clover, int max_iter, REAL eps, int eo_precondition,
              solver_info * preconditioner) {
  (void) max_iter; (void) eps; (void) preconditioner;
  // The fine operator handles are only known at apply time; patch them into
  // the hierarchy's fine level, then run one cycle from the top.
  g_levels[0].U_kappa = U_kappa;
  g_levels[0].F_csw   = F_csw;
  g_levels[0].clover  = clover;
  // eo_precondition = 1 (eo outer solve): `input` is the even-site half-field
  // of the outer Schur system, odd part implicitly zero -- no zero-padded
  // copy is ever materialized.  Its two depth-0 consumers handle the
  // half-field directly: restrict_lvl -> apply_R_even and
  // apply_D_residual_lvl -> D_residual_eo, both bit-identical to running the
  // full cycle on the zero-padded field.  `output` is still a full field,
  // so the caller must supply room for one and read back only its leading
  // even half.  Preconditioner calls are serial, so a static flag suffices.
  g_eo_eta_in = eo_precondition;
  int cits = mg_cycle(&g_levels[0], output, input,
                      (REAL) g_levels[0].solve_k_tol);
  g_eo_eta_in = 0;
  return cits;
}

// Accessor for a registered hierarchy level, so the adaptive setup
// (bootstrap.c) can drive a recursive cycle starting at a chosen depth via
// mg_cycle and patch the fine operator handles.  Valid only between mg_setup
// and mg_teardown; the &g_levels[depth] addresses are stable across rebuilds
// (a fixed static array).
level_t * mg_get_level(int depth) { return &g_levels[depth]; }

// --- coarse Dslash profiling -----------------------------------------------
// Flops are computed from coarse-level dimensions only (do not reuse the fine
// TRANSFER_FLOPS, which carries V / Ns / Nc / num_null_vecs).  Each apply_Dc:
//   Vc coarse sites x (1 self + 8 neighbor couplings)
//     x [N_coarse x N_coarse] complex block  *  [N_coarse x num_rhs] input
//     = N_coarse(out rows a) x N_coarse(in cols b) x num_rhs complex MACs
//   x 8 flops per complex multiply-add.
#define APPLY_DC_FLOPS \
  ((double) Vc * 9 * N_coarse * N_coarse * num_rhs * 8.0)
static double dc_secs = 0;
static long   dc_calls = 0;
void apply_dc_stats_reset(void) { dc_secs = 0; dc_calls = 0; }
void apply_dc_stats_report(const char * tag) {
  double gf = dc_secs > 0 ? 1e-9 * APPLY_DC_FLOPS * dc_calls / dc_secs : 0;
  printf("[apply_Dc %s] %ld calls  %.4f s  %.3f ms/call  %.2f GFlop/s\n",
         tag, dc_calls, dc_secs, dc_calls ? 1e3 * dc_secs / dc_calls : 0, gf);
  printf("[apply_Dc %s] flops/call = %.3e  (Vc=%d, 9 couplings, "
         "%dx%d blocks, num_rhs=%d)\n",
         tag, APPLY_DC_FLOPS, Vc, N_coarse, N_coarse, num_rhs);
}

// Schur-operator (even-odd coarse matvec) timing for the depth-1 macro level
// (the eo bottom solve of a 2-level run, where apply_Dc is dormant).  One
// such apply touches Vhc even + Vhc odd sites with 9 couplings each = Vc*9
// site-couplings, i.e. exactly APPLY_DC_FLOPS per call -- so the rate is
// directly comparable.  (apply_coarse_schur on a deeper 3-level+ coarsest is
// accounted in dlvl_* with that level's own volume, not here -- see the
// split at the end of that function.)
static double schur_secs = 0;
static long   schur_calls = 0;
void apply_schur_stats_reset(void) { schur_secs = 0; schur_calls = 0; }

// Fill a profiling snapshot from the V-cycle / transfer / coarse-Dslash timers.
// Placed here so transfer_*, vc_*, dc_*, schur_* statics are all in scope.
void mg_get_vcycle_profile(mg_vcycle_profile_t * p) {
  p->vc_coarse = vc_coarse; p->vc_D = vc_D;
  p->vc_smooth_fine = vc_smooth_fine;
  p->vc_calls = vc_calls; p->vc_citers = vc_citers;
  p->vc_calls_fine = vc_calls_fine;
  for (int d = 0; d < NUM_LEVELS; d ++) {
    mg_level_profile_t * L = &p->lvl[d];
    L->R_secs      = vc_R[d];
    L->P_secs      = vc_P[d];
    L->trans_flops = ctrans_flops[d];
    L->smooth_secs  = vc_smooth_coarse[d];
    L->smooth_flops = coarse_sap_flops(d);
    L->dslash_secs  = dlvl_secs[d];
    L->dslash_flops = dlvl_flops[d];
    L->dslash_calls = dlvl_calls[d];
    L->kc_secs   = kc_gs_secs[d];
    L->kc_flops  = kc_gs_flops[d];
    L->vec_secs  = vc_vec[d];
    L->vec_flops = vc_vec_flops[d];
  }
  // The two depth-0 accumulators that are not indexed, because there is only
  // one depth 0 to index: the fine transfers (apply_P / apply_R, whose exact
  // flop model differs from the runtime-geometry one) and, above, the fine
  // smoother.  Exact P+R flop total (P uses 8*nnv-2, R uses 8*nnv);
  // apply_R_even calls restrict only the even half, so they carry half the
  // restrict flops, and apply_P_par calls prolong only one block color,
  // likewise half.
  p->transfer_flops = TRANSFER_FLOPS_P * ((double) transfer_P_calls
                                          - 0.5 * transfer_P_half_calls)
                    + TRANSFER_FLOPS * ((double) transfer_R_calls
                                        - 0.5 * transfer_R_half_calls);
  p->lvl[0].trans_flops = p->transfer_flops;
  // The depth-1 matvec, which has two implementations selected by geometry
  // (apply_coarse_schur on the eo path, apply_Dc on the full path -- exactly
  // one is live) and a fixed per-call flop count, unlike the runtime-geometry
  // apply_Dc_lvl above.  Both land on level 1's Dslash entry.
  p->lvl[1].dslash_secs  += schur_secs + dc_secs;
  p->lvl[1].dslash_calls += schur_calls + dc_calls;
  p->lvl[1].dslash_flops += APPLY_DC_FLOPS * (double) (schur_calls + dc_calls);
  p->cg_ortho_secs    = cg_ortho_secs;
  p->cg_ortho_flops   = cg_ortho_flops;
  p->cg_other_secs    = cg_other_secs;
  p->kc_calls         = kc_calls;
  p->eo_secs          = eo_secs;
  p->eo_flops         = eo_flops;
  p->eo_calls         = eo_calls;
  p->botd_secs        = botd_secs;
  p->botd_flops       = botd_flops;
  p->botd_calls       = botd_calls;
}

// Reference (scalar) coarse Dslash.  Kept as the correctness oracle for the
// hand-vectorized apply_Dc below (test_coarse compares the two).  Not timed.
void apply_Dc_ref(coarse_op * Dc, REAL * out, REAL * in) {
  long matsz = (long) N_coarse * N_coarse * 2;
  #pragma omp parallel for schedule(static)
  for (int p = 0; p < Vc; p ++) {
    int qn[8];
    for (int dir = 0; dir < 8; dir ++) qn[dir] = coarse_neighbor(p, dir);
    REAL * selfM = Dc->self + (long) p * matsz;
    for (int rhs = 0; rhs < num_rhs; rhs ++) {
      for (int a = 0; a < N_coarse; a ++) {
        REAL accr = 0, acci = 0;
        // self coupling: out(p) += D_c[p,p] in(p)
        for (int b = 0; b < N_coarse; b ++) {
          long m = (long) (a * N_coarse + b) * 2;
          long ii = coarse_idx(p, b, rhs);
          REAL ir = in[ii], iim = in[ii + num_rhs];
          accr += selfM[m] * ir - selfM[m + 1] * iim;
          acci += selfM[m] * iim + selfM[m + 1] * ir;
        }
        // neighbor couplings: out(p) += D_c[p, p+off] in(p+off)
        for (int dir = 0; dir < 8; dir ++) {
          REAL * C = Dc->nbr + ((long) p * 8 + dir) * matsz;
          int qq = qn[dir];
          for (int b = 0; b < N_coarse; b ++) {
            long m = (long) (a * N_coarse + b) * 2;
            long ii = coarse_idx(qq, b, rhs);
            REAL ir = in[ii], iim = in[ii + num_rhs];
            accr += C[m] * ir - C[m + 1] * iim;
            acci += C[m] * iim + C[m + 1] * ir;
          }
        }
        long oi = coarse_idx(p, a, rhs);
        out[oi] = accr; out[oi + num_rhs] = acci;
      }
    }
  }
}

// Coarse Dslash out = D_c in on the level-1 macro geometry, vectorized over
// num_rhs through the shared coarse_dslash_site kernel (declared above with
// apply_Dc_lvl).  Each coupling-matrix element C[a,b] is a broadcast scalar
// FMA'd across the num_rhs SIMD lanes; the a,b coarse-variable loops stay
// scalar so any even num_null_vecs works with no SIMD remainder.  The
// accumulation order (self, then the 8 neighbors, b ascending) matches
// apply_Dc_ref, so the two agree to floating-point rounding.  Differs from
// apply_Dc_lvl only in the loop bound (Vc) and neighbor map
// (coarse_neighbor); the per-site math is identical.
#if FERM_GHOSTS
// Coarse halo ghost scratch for the macro depth-1 apply_Dc (one multi-rhs
// coarse field's z/t face slabs).  Static per-precision, like clover_ferm.
// c's wk_fghost; compiled out at MPI=0 (COARSE_GHOST_SITES == 0 there).
// Refreshed at the top of every apply_Dc since `in` changes per call.
// The coarse operator's APBC lives in Dc->nbr, so this exchange is a pure
// periodic index-space face fill (no phase).
#define COARSE_GHOST_REALS ((long) COARSE_GHOST_SITES * N_coarse * 2 * num_rhs)
static _Alignas(64) REAL wk_cghost[COARSE_GHOST_REALS];
#endif

void apply_Dc(coarse_op * Dc, REAL * out, REAL * in) {
  struct timeval _t0, _t1; gettimeofday(&_t0, NULL);
  const long matsz = (long) N_coarse * N_coarse * 2;
  const long blk   = (long) N_coarse * 2 * num_rhs;   // reals per coarse site
#if FERM_GHOSTS
  halo_coarse(in, wk_cghost, blk * sizeof(REAL), Lxc, Lyc, Lzc, Ltc);
#endif
  // cbases is sized 10 (ncpl+1): the asm kernel prefetches cbases[9],
  // one past end.
  #pragma omp parallel for schedule(static)
  for (int p = 0; p < Vc; p ++) {
    REAL * cbases[10], * ibases[9];
    cbases[0] = Dc->self + (long) p * matsz;
    ibases[0] = in + (long) p * blk;
#if FERM_GHOSTS
    // decode this site's coarse coords once; a z/t neighbor at a rank face is
    // redirected to the matching ghost-face slot (else the local periodic nbr).
    int xb = p % Lxc, rr = p / Lxc;
    int yb = rr % Lyc; rr /= Lyc;
    int zb = rr % Lzc; rr /= Lzc;
    int tb = rr;
#endif
    for (int dir = 0; dir < 8; dir ++) {
      cbases[dir + 1] = Dc->nbr + ((long) p * 8 + dir) * matsz;
#if FERM_GHOSTS
      const REAL * nb;
#if DECOMP_Z
      if (dir == 2 && zb == Lzc - 1)
        nb = wk_cghost + (CG_OFF_ZHI + (long) (tb * Lyc + yb) * Lxc + xb) * blk;
      else if (dir == 6 && zb == 0)
        nb = wk_cghost + (CG_OFF_ZLO + (long) (tb * Lyc + yb) * Lxc + xb) * blk;
      else
#endif
#if DECOMP_T
      if (dir == 3 && tb == Ltc - 1)
        nb = wk_cghost + (CG_OFF_THI + (long) (zb * Lyc + yb) * Lxc + xb) * blk;
      else if (dir == 7 && tb == 0)
        nb = wk_cghost + (CG_OFF_TLO + (long) (zb * Lyc + yb) * Lxc + xb) * blk;
      else
#endif
        nb = in + (long) coarse_neighbor(p, dir) * blk;
      ibases[dir + 1] = (REAL *) nb;
#else
      ibases[dir + 1] = in + (long) coarse_neighbor(p, dir) * blk;
#endif
    }
    coarse_dslash_site(out + (long) p * blk, cbases, ibases, 9, N_coarse);
  }
  gettimeofday(&_t1, NULL);
  dc_secs += time_used(_t0, _t1); dc_calls ++;
}

// --- Even-odd (Schur complement) coarse preconditioning --------------------

// Checkerboard parity of coarse lexicographic site cl (matches
// coarse_neighbor's coordinate decode).  All 8 nearest neighbors have the
// opposite parity.
static inline int coarse_parity(int cl) {
  int xb = cl % Lxc; int r = cl / Lxc;
  int yb = r % Lyc;  r /= Lyc;
  int zb = r % Lzc;  r /= Lzc;
  int tb = r;
  return (xb ^ yb ^ zb ^ tb) & 1;
}

// out_site = scale * sum_{k<ncpl} C[k] (.) in[k], with init=1 overwriting the
// output block and init=0 accumulating into it.  C[k] is an
// N_coarse x N_coarse interleaved-complex coupling (row a at
// C+a*N_coarse*2); in[k]/out_site are one coarse-site block
// (2*N_coarse*num_rhs REALs, split re/im over num_rhs).  Mirrors the
// vectorized apply_Dc inner kernel, generalized to a variable coupling count.
// Row-range form: computes only output variables [a0, a1) of the site block,
// so one site's matvec can be split across threads.  Matrix traffic splits
// with the rows, but every range reads the input vectors in full -- so a
// caller should split only as far as it needs to fill the thread pool,
// and no further.
static inline void coarse_block_gemv_rows(REAL * const * C, REAL * const * in,
                                          int ncpl, REAL * out_site,
                                          REAL scale, int init, int n_coarse,
                                          int a0, int a1) {
  for (int a = a0; a < a1; a ++) {
    REAL_VEC accr[num_rhs_vec], acci[num_rhs_vec];
    #pragma unroll
    for (int rv = 0; rv < num_rhs_vec; rv ++) {
      accr[rv] = ZERO_VEC;
      acci[rv] = ZERO_VEC;
    }
    for (int k = 0; k < ncpl; k ++) {
      const REAL * Crow = C[k] + (long) a * n_coarse * 2;
      const REAL * ib = in[k];
      for (int b = 0; b < n_coarse; b ++) {
        REAL_VEC cr = SET1(Crow[2 * b]);
        REAL_VEC ci = SET1(Crow[2 * b + 1]);
        const REAL * ibb = ib + (long) b * 2 * num_rhs;
        #pragma unroll
        for (int rv = 0; rv < num_rhs_vec; rv ++) {
          REAL_VEC ir  = *(const REAL_VEC *) (ibb + rv * veclen);
          REAL_VEC iim = *(const REAL_VEC *) (ibb + num_rhs + rv * veclen);
          accr[rv] += cr * ir - ci * iim;
          acci[rv] += cr * iim + ci * ir;
        }
      }
    }
    REAL * ob = out_site + (long) a * 2 * num_rhs;
    REAL_VEC s = SET1(scale);
    #pragma unroll
    for (int rv = 0; rv < num_rhs_vec; rv ++) {
      REAL_VEC vr = s * accr[rv], vi = s * acci[rv];
      if (!init) {
        vr += *(REAL_VEC *) (ob + rv * veclen);
        vi += *(REAL_VEC *) (ob + num_rhs + rv * veclen);
      }
      *(REAL_VEC *) (ob + rv * veclen)           = vr;
      *(REAL_VEC *) (ob + num_rhs + rv * veclen) = vi;
    }
  }
}

static inline void coarse_block_gemv(REAL * const * C, REAL * const * in,
                                     int ncpl, REAL * out_site,
                                     REAL scale, int init, int n_coarse) {
  coarse_block_gemv_rows(C, in, ncpl, out_site, scale, init, n_coarse,
                         0, n_coarse);
}

// One coarse-site block matvec, overwriting outb = sum_{k<ncpl} C[k] . in[k].
// Uses the hand-written asm microkernel when available (COARSE_USE_ASM,
// global_params.h), falling back to the generic coarse_block_gemv otherwise.
// cbases must hold ncpl+1 pointers (the asm kernel prefetches one past the
// end).  Unlike coarse_block_gemv this only overwrites (no scale /
// accumulate), which is all the Schur apply needs once the -A_ee^{-1}
// sign is folded into the block.

static inline void coarse_site_matvec(REAL ** cbases, REAL ** ibases,
                                      int ncpl, REAL * outb, int n_coarse) {
#ifdef DC_KERNEL
  if (COARSE_USE_ASM) {
    for (int a0 = 0; a0 < n_coarse; a0 += DC_MR)
      DC_KERNEL(outb + (long) a0 * 2 * num_rhs, cbases, ibases,
                (long) a0 * n_coarse * 2 * (long) sizeof(REAL), ncpl, n_coarse);
    return;
  }
#endif
  coarse_block_gemv(cbases, ibases, ncpl, outb, 1.0, 1, n_coarse);
}

#if FERM_GHOSTS
// Ghost-site offset of site (x,y,z,t)'s neighbor in direction dir, or -1 when
// the neighbor is local (not crossing a decomposed rank face).  Same
// direction codes (2/6 = +z/-z, 3/7 = +t/-t) and in-plane face index as
// apply_Dc's redirect.
static inline long coarse_eo_face_ghost(const coarse_eo_op * eo, int dir,
                                        int x, int y, int z, int t) {
  int decz = qc_grid.P[2] > 1, dect = qc_grid.P[3] > 1;
  if (decz && dir == 2 && z == eo->gnz - 1)
    return eo->cg_zhi + ((long) t * eo->gny + y) * eo->gnx + x;
  if (decz && dir == 6 && z == 0)
    return eo->cg_zlo + ((long) t * eo->gny + y) * eo->gnx + x;
  if (dect && dir == 3 && t == eo->gnt - 1)
    return eo->cg_thi + ((long) z * eo->gny + y) * eo->gnx + x;
  if (dect && dir == 7 && t == 0)
    return eo->cg_tlo + ((long) z * eo->gny + y) * eo->gnx + x;
  return -1;
}

// Scatter the parity-`par` half-field `half` (eo half-index i -> lex site)
// into the full-lex gfull scratch, then post the z/t face exchange into
// eo->ghost (non-blocking, for comm/compute overlap).  par: 0 = even half
// being sent, 1 = odd.  Only the sites on a sent face plane need scattering,
// and those are exactly this parity's boundary sites (a site is on the
// z=0/z-hi or t=0/t-hi plane iff it has a cross-rank neighbor iff it is in
// {even,odd}_bnd) -- so scatter that list, not all volh.  The pack reads
// only the face planes, which the boundary sites fully cover; untouched
// interior gfull slots stay stale but are never read, so this is
// bit-identical to a full scatter in every byte that is packed and read.
// No-op when not decomposed.  Paired with _wait; the caller must not mutate
// `half`/gfull until the wait returns.
static void coarse_eo_halo_start(coarse_eo_op * eo, int par,
                                 const REAL * half) {
  if (eo->ghost_sites == 0) return;
  // this level's width (gfull/ghost are sized from eo->nvar,
  // which differs from the macro N_coarse under per-level widths)
  long blk = (long) eo->nvar * 2 * num_rhs;
  const int * lex  = par ? eo->odd_lex  : eo->even_lex;
  const int * face = par ? eo->odd_bnd  : eo->even_bnd;
  long       nface = par ? eo->n_odd_bnd : eo->n_even_bnd;
  #pragma omp parallel for
  for (long k = 0; k < nface; k ++) {
    long i = face[k];
    memcpy(eo->gfull + (long) lex[i] * blk, half + i * blk, blk * sizeof(REAL));
  }
  halo_coarse_start(eo->gfull, eo->ghost, blk * sizeof(REAL),
                       eo->gnx, eo->gny, eo->gnz, eo->gnt);
}
static void coarse_eo_halo_wait(coarse_eo_op * eo) {
  if (eo->ghost_sites == 0) return;
  halo_coarse_wait();
}
// Blocking convenience wrapper (post + immediately wait) for the cold
// coarse_solve_eo RHS-formation / even-recovery exchanges, which run once
// per bottom solve and have no interior work to overlap -- only the
// per-Krylov matvec splits start/wait apart.
static void coarse_eo_halo(coarse_eo_op * eo, int par, const REAL * half) {
  coarse_eo_halo_start(eo, par, half);
  coarse_eo_halo_wait(eo);
}
#endif

int build_coarse_eo_op(coarse_eo_op * eo, coarse_op * Dc, const level_t * lv) {
  int  n_coarse = lv ? lv->nvar : N_coarse;  // coarsest width (macro L1 if
                                             // lv==NULL)
  eo->nvar   = n_coarse;
  eo->depth  = lv ? lv->depth : 1;       // NULL lv == the depth-1 macro level
  eo->dense_inv = NULL;  // attached later by coarse_direct_build_eo (if at all)
  long matsz = (long) n_coarse * n_coarse * 2;
  long blk   = (long) n_coarse * 2 * num_rhs;
  // geometry: NULL lv == depth-1 macro lattice; non-NULL == runtime coarsest
  long vol  = lv ? lv->vol  : (long) Vc;
  long volh = lv ? lv->volh : (long) Vhc;
  int  nx = lv ? lv->nx : Lxc, ny = lv ? lv->ny : Lyc;
  int  nz = lv ? lv->nz : Lzc, nt = lv ? lv->nt : Ltc;
  // Global parity offset: under z/t decomposition this rank's local block
  // starts at coordinate corigin, so the global checkerboard parity of a
  // local site is its local parity XOR (corigin_z + corigin_t) & 1.
  // Labeling even/odd by the global parity keeps the two sets consistent
  // across a rank face (a globally-odd site is odd on every rank), which is
  // what makes the cross-rank ghost reads and the bipartite hop tables agree.
  // 0 at MPI=0 (corigin {0,0,0,0}) -> unchanged.  lv == NULL is the depth-1
  // macro lattice, whose corigin is qc_grid.origin / BLOCK_AGG per component
  // (see mg_setup) -- fold that in too, or the macro-level eo op labels
  // parity locally and the cross-rank hop tables disagree wherever the
  // corigin sum is odd.
  int poff = 0;
#if FERM_GHOSTS
  poff = lv ? (lv->corigin[2] + lv->corigin[3]) & 1
            : ((qc_grid.origin[2] / BLOCK_AGG) + (qc_grid.origin[3] / BLOCK_AGG)) & 1;
#endif
  // Bipartiteness needs every global extent even or 1.  An odd extent >= 3
  // wraps a same-parity pair across the periodic boundary, and the balance
  // check below can still pass (3x6x6x12 splits 50/50), so the
  // all-neighbors-opposite-parity tables would be silently mis-indexed.
  // An extent of 1 is different: +mu and -mu reach the site itself, and the
  // Galerkin build has already folded that into `self` (it probes the
  // operator, so the contribution lands there and both nbr slots stay zero),
  // leaving the checkerboard of the lattice with that dimension removed.
  // Same predicate as the coarse-SAP block-count check.  Under decomposition
  // a local extent may be odd (6,6,3,3 @ Pz2Pt4) while the global one is
  // even, so test n*P, not n.  A runtime level returns -1, though
  // mg_add_coarse_level now rejects these before any level reaches here;
  // the macro level is a build error.
  {
    int g[4] = {nx * qc_grid.P[0], ny * qc_grid.P[1],
                nz * qc_grid.P[2], nt * qc_grid.P[3]};
    for (int d = 0; d < 4; d ++) if (g[d] > 1 && (g[d] & 1)) {
      if (lv) return -1;
      comm_die("build_coarse_eo_op: odd global coarse extent (%d %d %d %d) "
               "breaks the eo checkerboard", g[0], g[1], g[2], g[3]);
    }
  }
  eo->Dc = Dc;
  eo->vol = vol; eo->volh = volh;
  eo->gnx = nx; eo->gny = ny; eo->gnz = nz; eo->gnt = nt;
  eo->oe_ghost = eo->eo_ghost = NULL;
  eo->gfull = eo->ghost = NULL;
  eo->ghost_sites = 0;
  eo->cg_zlo = eo->cg_zhi = eo->cg_tlo = eo->cg_thi = 0;
  eo->even_int = eo->even_bnd = eo->odd_int = eo->odd_bnd = NULL;
  eo->n_even_int = eo->n_even_bnd = eo->n_odd_int = eo->n_odd_bnd = 0;
#if FERM_GHOSTS
  // z/t face-slab ghost geometry, in the same layout halo_coarse fills (the
  // CG_OFF_* rule, but with this level's runtime extents).  At MPI=0
  // decz==dect==0 -> ghost_sites 0 and all oe/eo_ghost entries stay -1,
  // so nothing is exchanged and the apply/solve are byte-identical.
  {
    int decz = qc_grid.P[2] > 1, dect = qc_grid.P[3] > 1;
    long zface = (long) nx * ny * nt, tface = (long) nx * ny * nz;
    eo->cg_zlo = 0;            eo->cg_zhi = zface;
    eo->cg_tlo = 2 * zface * decz;  eo->cg_thi = eo->cg_tlo + tface;
    eo->ghost_sites = 2 * zface * decz + 2 * tface * dect;
  }
#endif
  // Surviving hop directions (see hop_dir): a direction whose coupling is
  // zero at every site of this rank contributes nothing anywhere, so dropping
  // it from the stencil walk is exact, not an approximation.  Uniform across
  // sites in practice -- the case this catches is a whole degenerate
  // dimension -- so one list per level is enough.
  eo->n_hop = 0;
  for (int dir = 0; dir < 8; dir ++) {
    int nz = 0;
    for (long p = 0; p < vol && !nz; p ++) {
      const REAL * C = Dc->nbr + ((long) p * 8 + dir) * matsz;
      for (long k = 0; k < matsz; k ++)
        if (C[k] != (REAL) 0) { nz = 1; break; }
    }
    if (nz) eo->hop_dir[eo->n_hop ++] = dir;
  }
  // Split each site's output rows only as far as it takes to cover the threads.
  {
    int nthr = omp_get_max_threads();
    int want = (volh > 0 && volh < nthr) ? (int) ((nthr + volh - 1) / volh) : 1;
    int cap  = n_coarse / 32;        // keep panels wide enough to be worth it
    if (cap < 1) cap = 1;
    eo->npanel = want < 1 ? 1 : (want > cap ? cap : want);
  }
  // One line per distinct coarse volume, and again only when the structure
  // moves.  Setup rebuilds each level on every cycle, so most repeats are
  // identical -- but n_hop is counted from the couplings themselves,
  // so refining the null vectors can change it, and that change is the part
  // worth seeing.
  if (eo->n_hop < 8 || eo->npanel > 1) {
    static long seen_vol[NUM_LEVELS];
    static int  seen_hop[NUM_LEVELS], seen_pan[NUM_LEVELS], nseen = 0;
    int i = 0;
    while (i < nseen && seen_vol[i] != vol) i ++;
    if (i == nseen && nseen < NUM_LEVELS) {
      seen_vol[i] = vol; seen_hop[i] = -1; seen_pan[i] = -1; nseen ++;
    }
    if (i < nseen && (seen_hop[i] != eo->n_hop || seen_pan[i] != eo->npanel)) {
      seen_hop[i] = eo->n_hop; seen_pan[i] = eo->npanel;
      print0("coarse_eo[vol %ld]: %d/8 hop directions carry couplings, "
             "%d output-row panel(s) per site\n", vol, eo->n_hop, eo->npanel);
    }
  }

  eo->even_lex    = malloc(sizeof(int) * volh);
  eo->odd_lex     = malloc(sizeof(int) * volh);
  eo->odd_of_even = malloc(sizeof(int) * volh * 8);
  eo->even_of_odd = malloc(sizeof(int) * volh * 8);
  eo->Aee_inv = malloc(sizeof(REAL) * volh * matsz);
  eo->te = malloc(sizeof(REAL) * volh * blk);
  eo->ue = malloc(sizeof(REAL) * volh * blk);
#if FERM_GHOSTS
  eo->oe_ghost = malloc(sizeof(int) * volh * 8);
  eo->eo_ghost = malloc(sizeof(int) * volh * 8);
  if (eo->ghost_sites > 0) {
    eo->gfull = malloc(sizeof(REAL) * vol  * blk);
    eo->ghost = malloc(sizeof(REAL) * eo->ghost_sites * blk);
  }
#endif

  int * lex2eo = malloc(sizeof(int) * vol);
  long ne = 0, no = 0;
  for (long cl = 0; cl < vol; cl ++) {
    int par = (lv ? coarse_parity_lvl(lv, cl) : coarse_parity((int) cl)) ^ poff;
    if (par == 0) { eo->even_lex[ne] = (int) cl; lex2eo[cl] = (int) ne; ne ++; }
    else          { eo->odd_lex[no]  = (int) cl; lex2eo[cl] = (int) no; no ++; }
  }
  // Build expects a balanced checkerboard (vol even with an equal split).  A
  // degenerate runtime coarsest (odd extent) can't be eo-split -> tell the
  // caller to fall back; a macro-level imbalance is a build error.
  if (ne != volh || no != volh) {
    if (lv) { free(lex2eo); free_coarse_eo_op(eo); return -1; }
    comm_die("build_coarse_eo_op: unbalanced parity (%ld even, %ld odd, Vhc=%d)",
             ne, no, Vhc);
  }
  for (long e = 0; e < volh; e ++) {
    int cl = eo->even_lex[e];
    for (int dir = 0; dir < 8; dir ++) {
      long nb = lv ? coarse_neighbor_lvl(lv, cl, dir) : coarse_neighbor(cl, dir);
      eo->odd_of_even[e * 8 + dir] = lex2eo[nb];   // local eo-index (periodic wrap)
#if FERM_GHOSTS
      // cross-rank face neighbor: the periodic wrap above is wrong, redirect to
      // the ghost-face slot instead (oe_ghost >= 0 selects it in the apply)
      int x = cl % nx, r = cl / nx, y = r % ny; r /= ny;
      int z = r % nz; r /= nz; int t = r;
      eo->oe_ghost[e * 8 + dir]
        = (int) coarse_eo_face_ghost(eo, dir, x, y, z, t);
#endif
    }
  }
  for (long o = 0; o < volh; o ++) {
    int cl = eo->odd_lex[o];
    for (int dir = 0; dir < 8; dir ++) {
      long nb = lv ? coarse_neighbor_lvl(lv, cl, dir)
               : coarse_neighbor(cl, dir);
      eo->even_of_odd[o * 8 + dir] = lex2eo[nb];
#if FERM_GHOSTS
      int x = cl % nx, r = cl / nx, y = r % ny; r /= ny;
      int z = r % nz; r /= nz; int t = r;
      eo->eo_ghost[o * 8 + dir]
        = (int) coarse_eo_face_ghost(eo, dir, x, y, z, t);
#endif
    }
  }
  free(lex2eo);

#if FERM_GHOSTS
  // Partition each parity into interior (no cross-rank neighbor) and boundary
  // (>=1 ghost neighbor) half-sites, so apply_coarse_schur can compute the
  // interior while the face exchange is in flight.  Scanning in eo-index
  // order keeps the interior list ascending; with ghost_sites == 0 every
  // site is interior and the lists are exactly [0..volh) / empty
  // (apply still byte-identical).
  if (eo->ghost_sites > 0) {
    eo->even_int = malloc(sizeof(int) * volh);
    eo->even_bnd = malloc(sizeof(int) * volh);
    eo->odd_int  = malloc(sizeof(int) * volh);
    eo->odd_bnd  = malloc(sizeof(int) * volh);
    for (long e = 0; e < volh; e ++) {
      int bnd = 0;
      for (int dir = 0; dir < 8; dir ++)
        if (eo->oe_ghost[e * 8 + dir] >= 0)
          bnd = 1;
      if (bnd) eo->even_bnd[eo->n_even_bnd ++] = (int) e;
      else     eo->even_int[eo->n_even_int ++] = (int) e;
    }
    for (long o = 0; o < volh; o ++) {
      int bnd = 0;
      for (int dir = 0; dir < 8; dir ++)
        if (eo->eo_ghost[o * 8 + dir] >= 0)
          bnd = 1;
      if (bnd) eo->odd_bnd[eo->n_odd_bnd ++] = (int) o;
      else     eo->odd_int[eo->n_odd_int ++] = (int) o;
    }
  }
#endif

  // Store -A_ee^{-1} (not +A_ee^{-1}): the odd Schur phase is then a single
  // same-sign 9-coupling kernel call,
  // out = A_oo in + D_oe (-A_ee^{-1} D_eo in),
  // with no per-coupling sign flip.  The cold RHS/recovery paths compensate
  // with explicit scale signs (see coarse_solve_eo).
  #pragma omp parallel for schedule(dynamic)
  for (long e = 0; e < volh; e ++) {
    REAL * dst = eo->Aee_inv + (long) e * matsz;
    invert_dense_block(Dc->self + (long) eo->even_lex[e] * matsz, dst,
                       n_coarse);
    for (long i = 0; i < matsz; i ++) dst[i] = -dst[i];
  }
  return 0;
}

void free_coarse_eo_op(coarse_eo_op * eo) {
  free(eo->even_lex); free(eo->odd_lex);
  free(eo->odd_of_even); free(eo->even_of_odd);
  free(eo->Aee_inv); free(eo->te); free(eo->ue);
  eo->dense_inv = NULL;   // borrowed (owned by level_t.bot_inv) --
                          // never freed here
  eo->even_lex = eo->odd_lex = eo->odd_of_even = eo->even_of_odd = NULL;
  eo->Aee_inv = eo->te = eo->ue = NULL;
#if FERM_GHOSTS
  free(eo->oe_ghost); free(eo->eo_ghost); free(eo->gfull); free(eo->ghost);
  eo->oe_ghost = eo->eo_ghost = NULL; eo->gfull = eo->ghost = NULL;
  free(eo->even_int); free(eo->even_bnd); free(eo->odd_int); free(eo->odd_bnd);
  eo->even_int = eo->even_bnd = eo->odd_int = eo->odd_bnd = NULL;
  eo->n_even_int = eo->n_even_bnd = eo->n_odd_int = eo->n_odd_bnd = 0;
#endif
}

// One even half-site of the Schur apply: te = D_eo in_odd (8-coupling hop),
// then ue = (-A_ee^{-1}) te (1-coupling dense block; Aee_inv holds the
// negated inverse).  Cross-rank odd neighbors are redirected to the exchanged
// face slab (eo->ghost).  Each site writes only its own te/ue, so the
// iteration order is irrelevant.
static inline void schur_even_site(coarse_eo_op * eo, coarse_op * Dc, long e,
                                   REAL * in_odd, long blk, long matsz,
                                   int n_coarse) {
  int cl = eo->even_lex[e];
  REAL * te = eo->te + (long) e * blk;
  REAL * C[9]; REAL * in[8];               // C[8] holds the prefetch one-past
  for (int dir = 0; dir < 8; dir ++) {
    C[dir]  = Dc->nbr + ((long) cl * 8 + dir) * matsz;
#if FERM_GHOSTS
    long gh = eo->oe_ghost[e * 8 + dir];
    in[dir] = gh >= 0 ? eo->ghost  + gh * blk
                      : in_odd + (long) eo->odd_of_even[e * 8 + dir] * blk;
#else
    in[dir] = in_odd + (long) eo->odd_of_even[e * 8 + dir] * blk;
#endif
  }
  coarse_site_matvec(C, in, 8, te, n_coarse);            // te = D_eo in_odd
  REAL * Cb[2] = { eo->Aee_inv + (long) e * matsz };     // [1] = prefetch one-past
  REAL * ib[1] = { te };
  // ue = -Ainv te
  coarse_site_matvec(Cb, ib, 1, eo->ue + (long) e * blk, n_coarse);
}

// One odd half-site: out = A_oo in_odd + D_oe ue (single same-sign 9-coupling
// call; ue already carries the minus via the negated A_ee^{-1}, so this
// equals A_oo in_odd - D_oe A_ee^{-1} D_eo in_odd = S in_odd).  Cross-rank
// even neighbors read ue from the exchanged face slab.
static inline void schur_odd_site(coarse_eo_op * eo, coarse_op * Dc, long o,
                                  REAL * in_odd, REAL * out_odd,
                                  long blk, long matsz, int n_coarse) {
  int cl = eo->odd_lex[o];
  REAL * C[10]; REAL * in[9];              // C[9] holds the prefetch one-past
  C[0]  = Dc->self + (long) cl * matsz;
  in[0] = in_odd + (long) o * blk;
  for (int dir = 0; dir < 8; dir ++) {
    C[dir + 1]  = Dc->nbr + ((long) cl * 8 + dir) * matsz;
#if FERM_GHOSTS
    long gh = eo->eo_ghost[o * 8 + dir];
    in[dir + 1] = gh >= 0 ? eo->ghost + gh * blk
                          : eo->ue + (long) eo->even_of_odd[o * 8 + dir] * blk;
#else
    in[dir + 1] = eo->ue + (long) eo->even_of_odd[o * 8 + dir] * blk;
#endif
  }
  coarse_site_matvec(C, in, 9, out_odd + (long) o * blk, n_coarse);
}

void apply_coarse_schur(coarse_eo_op * eo, REAL * out_odd, REAL * in_odd) {
  struct timeval _t0, _t1; gettimeofday(&_t0, NULL);
  coarse_op * Dc = eo->Dc;
  int  n_coarse = eo->nvar;
  long matsz = (long) n_coarse * n_coarse * 2;
  long blk   = (long) n_coarse * 2 * num_rhs;

#if FERM_GHOSTS
  if (eo->ghost_sites > 0) {
    // overlap #1: post the in_odd face exchange (odd neighbors of even sites),
    // form the rank-interior even sites while it flies, then the boundary even
    // sites once the odd ghosts have landed
    coarse_eo_halo_start(eo, 1, in_odd);            // send the odd half
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < eo->n_even_int; i ++)
      schur_even_site(eo, Dc, eo->even_int[i], in_odd, blk, matsz, n_coarse);
    coarse_eo_halo_wait(eo);
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < eo->n_even_bnd; i ++)
      schur_even_site(eo, Dc, eo->even_bnd[i], in_odd, blk, matsz, n_coarse);

    // overlap #2: every even ue is now formed; post the ue exchange (even
    // neighbors of odd sites), compute interior odd sites, then boundary
    coarse_eo_halo_start(eo, 0, eo->ue);            // send the even half
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < eo->n_odd_int; i ++)
      schur_odd_site(eo, Dc, eo->odd_int[i], in_odd, out_odd, blk, matsz,
                     n_coarse);
    coarse_eo_halo_wait(eo);
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < eo->n_odd_bnd; i ++)
      schur_odd_site(eo, Dc, eo->odd_bnd[i], in_odd, out_odd, blk, matsz,
                     n_coarse);
  } else
#endif
  {
    // MPI=0 / undecomposed coarsest: straight per-site loops, byte-identical to
    // the pre-overlap apply (no exchange, all sites local)
    #pragma omp parallel for schedule(static)
    for (long e = 0; e < eo->volh; e ++)
      schur_even_site(eo, Dc, e, in_odd, blk, matsz, n_coarse);
    #pragma omp parallel for schedule(static)
    for (long o = 0; o < eo->volh; o ++)
      schur_odd_site(eo, Dc, o, in_odd, out_odd, blk, matsz, n_coarse);
  }
  gettimeofday(&_t1, NULL);
  double dt = time_used(_t0, _t1);
  // Flop model: eo->vol sites x 9 couplings x N_coarse^2 complex MACs
  // (8 flop) x num_rhs -- identical to apply_Dc_lvl.  At the depth-1 macro
  // level (eo->vol == Vc) one apply == APPLY_DC_FLOPS, so it feeds the
  // "coarse-grid Dslash" row (schur_*) at that fixed weight.  At a deeper
  // coarsest level the volume is smaller and APPLY_DC_FLOPS would massively
  // overcount, so route it into the dlvl_* (coarsest Dslash) accumulators
  // with its own volume: there it is the coarsest matvec, in place of the
  // dormant apply_Dc_lvl.
  if (eo->vol == (long) Vc) {
    schur_secs += dt; schur_calls ++;
  } else {
    dlvl_secs[eo->depth] += dt; dlvl_calls[eo->depth] ++;
    dlvl_flops[eo->depth] += (double) eo->vol * 9.0 * (double) n_coarse
                           * (double) n_coarse * 8.0 * (double) num_rhs;
  }
}

static void op_coarse_schur(void * ctx, REAL * out, REAL * in) {
  apply_coarse_schur((coarse_eo_op *) ctx, out, in);
}

int coarse_solve_eo(coarse_eo_op * eo, REAL * x, REAL * b,
                    int max_iter, REAL eps, int restart) {
  coarse_op * Dc = eo->Dc;
  int  n_coarse = eo->nvar;
  long matsz = (long) n_coarse * n_coarse * 2;
  long blk   = (long) n_coarse * 2 * num_rhs;
  long half  = (long) eo->volh * blk;   // REALs in a half (even or odd) field

  REAL * be  = malloc(sizeof(REAL) * half);
  REAL * bo  = malloc(sizeof(REAL) * half);
  REAL * xo  = malloc(sizeof(REAL) * half);
  REAL * rhs = malloc(sizeof(REAL) * half);
  REAL * ue  = malloc(sizeof(REAL) * half);

  struct timeval eo0, eo1; gettimeofday(&eo0, NULL);
  // gather b into even / odd packed half-fields (per-site blocks are
  // contiguous)
  #pragma omp parallel for
  for (long e = 0; e < eo->volh; e ++)
    memcpy(be + (long) e * blk, b + (long) eo->even_lex[e] * blk,
           blk * sizeof(REAL));
  #pragma omp parallel for
  for (long o = 0; o < eo->volh; o ++)
    memcpy(bo + (long) o * blk, b + (long) eo->odd_lex[o] * blk,
           blk * sizeof(REAL));

  // ue = A_ee^{-1} b_e   (even).  Aee_inv holds -A_ee^{-1}, so scale by -1.
  // Every loop below runs over (site x output-row panel): on a coarsest level
  // with fewer sites than threads the site index alone leaves most of the
  // pool idle.
  const int npan = eo->npanel;
  const int prow = (n_coarse + npan - 1) / npan;
  #pragma omp parallel for schedule(static) collapse(2)
  for (long e = 0; e < eo->volh; e ++)
    for (int pn = 0; pn < npan; pn ++) {
      int a0 = pn * prow, a1 = a0 + prow < n_coarse ? a0 + prow : n_coarse;
      if (a0 >= a1) continue;
      REAL * Cb[1] = { eo->Aee_inv + (long) e * matsz };
      REAL * ib[1] = { be + (long) e * blk };
      coarse_block_gemv_rows(Cb, ib, 1, ue + (long) e * blk, -1.0, 1, n_coarse,
                             a0, a1);
    }
#if FERM_GHOSTS
  coarse_eo_halo(eo, 0, ue);        // cross-rank even nbrs read ue from ghost
#endif
  // rhs_o = b_o - D_oe ue   (over the hop directions that carry a coupling)
  #pragma omp parallel for schedule(static) collapse(2)
  for (long o = 0; o < eo->volh; o ++)
    for (int pn = 0; pn < npan; pn ++) {
      int a0 = pn * prow, a1 = a0 + prow < n_coarse ? a0 + prow : n_coarse;
      if (a0 >= a1) continue;
      int cl = eo->odd_lex[o];
      REAL * out = rhs + (long) o * blk;
      long roff = (long) a0 * 2 * num_rhs;                // this panel's rows
      memcpy(out + roff, bo + (long) o * blk + roff,
             (size_t) (a1 - a0) * 2 * num_rhs * sizeof(REAL));   // out = b_o
      REAL * C[8]; REAL * in[8];
      for (int h = 0; h < eo->n_hop; h ++) {
        int dir = eo->hop_dir[h];
        C[h] = Dc->nbr + ((long) cl * 8 + dir) * matsz;
#if FERM_GHOSTS
        long gh = eo->eo_ghost[o * 8 + dir];
        in[h] = gh >= 0 ? eo->ghost + gh * blk
                        : ue + (long) eo->even_of_odd[o * 8 + dir] * blk;
#else
        in[h] = ue + (long) eo->even_of_odd[o * 8 + dir] * blk;
#endif
      }
      coarse_block_gemv_rows(C, in, eo->n_hop, out, -1.0, 0, n_coarse, a0, a1);
    }

  gettimeofday(&eo1, NULL); eo_secs += time_used(eo0, eo1);

  // solve S x_o = rhs_o on odd sites (half-size Krylov space) -- or, when the
  // dense odd-Schur inverse is cached (BOTTOM_DENSE_INV), exactly: a slab GEMM
  // against the read-only S^{-1} (bot_inv_apply streams the inverse once per
  // call for all num_rhs lanes)
  int iters;
#if BOTTOM_DENSE_INV
  if (eo->dense_inv) {
    struct timeval _b0, _b1; gettimeofday(&_b0, NULL);
    const long Nh   = (long) eo->volh * n_coarse;  // this rank's odd-Schur DOF
    const long Nhg  = eo->dense_N;     // global odd-Schur DOF (== Nh @1 rank)
    bot_cplx * B = malloc(sizeof(bot_cplx) * Nh * (long) num_rhs);
    bot_cplx * X = malloc(sizeof(bot_cplx) * Nh * (long) num_rhs);
    #pragma omp parallel for schedule(static)
    for (long o = 0; o < eo->volh; o ++)
      for (long a = 0; a < n_coarse; a ++) {
        long K = o * n_coarse + a;
        for (int l = 0; l < num_rhs; l ++)
          B[K + (long) l * Nh] = rhs[o * blk + (a * 2) * num_rhs + l]
                               + rhs[o * blk + (a * 2 + 1) * num_rhs + l] * I;
      }
#if FERM_GHOSTS
    // gather the local odd RHS into the global RHS,
    // GEMM this rank's inverse rows
    bot_cplx * Bg = malloc(sizeof(bot_cplx) * Nhg * (long) num_rhs);
    bot_allgather_rhs(Bg, B, Nhg, Nh);
    bot_inv_apply((const bot_cplx *) eo->dense_inv, Nh, Nhg, Bg, X);
    free(Bg);
#else
    bot_inv_apply((const bot_cplx *) eo->dense_inv, Nh, Nhg, B, X);
#endif
    #pragma omp parallel for schedule(static)
    for (long o = 0; o < eo->volh; o ++)
      for (long a = 0; a < n_coarse; a ++) {
        long K = o * n_coarse + a;
        for (int l = 0; l < num_rhs; l ++) {
          bot_cplx v = X[K + (long) l * Nh];
          xo[o * blk + (a * 2) * num_rhs + l]     = (REAL) creal(v);
          xo[o * blk + (a * 2 + 1) * num_rhs + l] = (REAL) cimag(v);
        }
      }
    free(B); free(X);
    gettimeofday(&_b1, NULL);
    botd_secs += time_used(_b0, _b1); botd_calls ++;
    botd_flops += 8.0 * (double) Nh * (double) Nhg * (double) num_rhs;
    iters = 1;                      // exact: reported as a single "iteration"
  } else
#endif
  iters = fgmres_coarse(op_coarse_schur, eo, NULL, NULL, cg_bill(),
                        xo, rhs, half, max_iter, eps, restart);

  gettimeofday(&eo0, NULL);
#if FERM_GHOSTS
  coarse_eo_halo(eo, 1, xo);         // cross-rank odd nbrs read xo from ghost
#endif
  // Recover even part: x_e = A_ee^{-1} (b_e - D_eo x_o), in two passes.  The
  // inverse consumes a whole site's t, which panels of that site now build
  // in parallel, so the hop has to complete first -- the loop barrier
  // between them is the ordering.
  #pragma omp parallel for schedule(static) collapse(2)
  for (long e = 0; e < eo->volh; e ++)
    for (int pn = 0; pn < npan; pn ++) {
      int a0 = pn * prow, a1 = a0 + prow < n_coarse ? a0 + prow : n_coarse;
      if (a0 >= a1) continue;
      int cl = eo->even_lex[e];
      REAL * C[8]; REAL * in[8];
      for (int h = 0; h < eo->n_hop; h ++) {
        int dir = eo->hop_dir[h];
        C[h] = Dc->nbr + ((long) cl * 8 + dir) * matsz;
#if FERM_GHOSTS
        long gh = eo->oe_ghost[e * 8 + dir];
        in[h] = gh >= 0 ? eo->ghost + gh * blk
                        : xo + (long) eo->odd_of_even[e * 8 + dir] * blk;
#else
        in[h] = xo + (long) eo->odd_of_even[e * 8 + dir] * blk;
#endif
      }
      REAL * t = ue + (long) e * blk;                  // reuse ue as scratch
      long roff = (long) a0 * 2 * num_rhs;
      memcpy(t + roff, be + (long) e * blk + roff,
             (size_t) (a1 - a0) * 2 * num_rhs * sizeof(REAL));   // t = b_e
      coarse_block_gemv_rows(C, in, eo->n_hop, t, -1.0, 0, n_coarse, a0, a1);
    }
  #pragma omp parallel for schedule(static) collapse(2)
  for (long e = 0; e < eo->volh; e ++)
    for (int pn = 0; pn < npan; pn ++) {
      int a0 = pn * prow, a1 = a0 + prow < n_coarse ? a0 + prow : n_coarse;
      if (a0 >= a1) continue;
      int cl = eo->even_lex[e];
      REAL * Cb[1] = { eo->Aee_inv + (long) e * matsz };
      REAL * ib[1] = { ue + (long) e * blk };
      // Aee_inv holds -A_ee^{-1}, so scale by -1 to apply the true inverse.
      coarse_block_gemv_rows(Cb, ib, 1, x + (long) cl * blk, -1.0, 1, n_coarse,
                             a0, a1);
    }
  // scatter x_o into the full solution
  #pragma omp parallel for
  for (long o = 0; o < eo->volh; o ++)
    memcpy(x + (long) eo->odd_lex[o] * blk, xo + (long) o * blk,
           blk * sizeof(REAL));
  gettimeofday(&eo1, NULL); eo_secs += time_used(eo0, eo1);
  // Flop model: 1 even A_ee^-1 (ncpl=1) + rhs_o D_oe (ncpl=n_hop) on setup,
  // plus D_eo (ncpl=n_hop) + even A_ee^-1 (ncpl=1) on recovery ->
  // 2 + 2*n_hop coarse block-gemv couplings per even/odd site.  n_hop is 8
  // on a non-degenerate level and smaller where a degenerate dimension left
  // a coupling slot empty -- those applies are skipped, so they must not be
  // billed either.  One gemv coupling = n_coarse^2 complex MACs (8 flops)
  // over num_rhs lanes; gather/scatter memcpy carries no flops.
  // n_coarse is the runtime width of the level this solve runs on, which is
  // the coarsest one -- not the N_coarse macro, which is the L0->L1 width
  // and only coincides with it in a hierarchy whose per-level null-vector
  // counts are all equal.  Every other flop model in this file already keys
  // off the runtime width.
  eo_flops += (double) (2 + 2 * eo->n_hop)
            * (double) eo->volh * (double) n_coarse * (double) n_coarse
            * 8.0 * (double) num_rhs;
  eo_calls ++;

  free(be); free(bo); free(xo); free(rhs); free(ue);
  return iters;
}

REAL nullvec_orthonormality_error(REAL * nv) {
  REAL maxerr = 0;
  #pragma omp parallel for collapse(2) schedule(dynamic) reduction(max:maxerr)
  for (int cl = 0; cl < Vc; cl ++)
    for (int tau = 0; tau < 2; tau ++) {
      long comp[agg_components];
      aggregate_components(cl, tau, comp);
      for (int i = 0; i < num_null_vecs; i ++)
        for (int j = 0; j < num_null_vecs; j ++) {
          REAL pr, pi;
          agg_dot(nv, comp, i, j, &pr, &pi);
          REAL d = (i == j) ? 1.0 : 0.0;
          REAL err = sqrt((pr - d) * (pr - d) + pi * pi);
          if (err > maxerr) maxerr = err;
        }
    }
  return maxerr;
}
