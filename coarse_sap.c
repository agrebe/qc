// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

// Coarse-level SAP smoother, split out of coarse.c.  Compiled twice
// (coarse_sap_f.o / coarse_sap_d.o) from this one body like the other
// templated units; MG_SINGLE selects the precision and mangle.h the symbol
// suffix.
//
// Its interface to the rest of the coarse layer is deliberately small: the
// shared site kernel and neighbor offsets come in through coarse_kernel.h,
// and only coarse_sap / coarse_csap_bench / coarse_csap_check /
// coarse_sap_cache_free / csap_profile_reset and the flop counter go back
// out (declared in coarse.h).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include "coarse.h"
#include "coarse_kernel.h"  // coarse_dslash_site, coarse_off, DC_KERNEL / DC_MR
#include "utils.h"          // complex.h
#include "comm.h"           // halo exchange for the cross-rank SAP sweep
#include "omp.h"            // omp_get_thread_num / omp_get_max_threads

// Modeled coarse-SAP flops per level, accumulated here and drained by
// coarse.c's mg_get_vcycle_profile (paired there with that level's smoother
// time).  Kept per level rather than as one total because the levels differ
// in both width and volume, so a single number could not be paired with any
// one time.
static double csap_flops[NUM_LEVELS];
double coarse_sap_flops(int depth)  { return csap_flops[depth]; }
void   coarse_sap_flops_reset(void) {
  for (int d = 0; d < NUM_LEVELS; d ++) csap_flops[d] = 0;
}

// ===========================================================================
// Coarse-level SAP smoother (coarse_sap): red-black multiplicative Schwarz
// on the coarse stencil operator D_l, the depth >= 1 analogue of the fine
// red_black_schwarz.  The block solve is vanilla MR (search direction
// = current residual) against the full coarse-block operator -- DDalphaAMG
// likewise uses plain "local minimal residual"
// (linsolve_generic.c:local_minres_PRECISION) on every coarse level, and not
// the even-odd block MR it reserves for depth 0.  Inner MR iterations per
// block solve: the level's blk_iter (mg_params).
//
// Block decomposition: the level lattice is tiled into lv->sap_blk[] blocks
// of coarse sites; blocks are red/black colored by the parity of their
// block-coordinate sum.  Same-color blocks are never face-adjacent, so a
// color sweep updates phi on disjoint sites while reading only opposite-color
// (this-sweep-frozen) neighbors -- safe to parallelize over blocks.
// Multiplicative ordering (red then black, shared phi) is preserved: black
// blocks see red's updates within the same sweep.  Coarse lattices are
// small, so each block's MR runs serially with L1-resident scratch and the
// parallelism comes from the many blocks.  The matvecs (block residual,
// per-MR-iter block apply) gather per-site coupling/input base pointers and
// route through the same coarse_dslash_site kernel apply_Dc_lvl uses; one
// coupling apply is nvar^2 complex MACs x num_rhs at this level's own width
// (= CFLOP in the flop model below).

// Block-local BLAS-1 for the SAP MR iteration (serial: parallelism is over
// blocks).  Data layout everywhere: groups of 2*num_rhs reals (num_rhs reals
// then num_rhs imags); all buffers 64-aligned with 64|2*num_rhs*sizeof(REAL),
// so aligned REAL_VEC derefs are safe.

// cre+i*cim = <x, y> (Hermitian, conj(x).y) and dd = <x, x>, fused in one
// pass with block-wide vector accumulators (4 independent chains each, so the
// 2-FMA-per-chain-per-group dependency never throttles) and a single
// horizontal reduction at the end.  Precondition: nreals % (8 * num_rhs) == 0,
// which always holds here (nreals = bsites * nvar * 2 * num_rhs and
// nvar % 4 == 0).  The block-wide accumulation reassociates the sum, so this
// is not bit-identical to a per-(site, coarse variable) reduction -- fine
// for the MR step length.

// MR-only helpers (dead when the block solve is dense)
#if !COARSE_SAP_DENSE_INV
static inline void blk_dot3(const REAL * x, const REAL * y, long nreals,
                            REAL * cre, REAL * cim, REAL * dd) {
  REAL_VEC acr[4], aci[4], add[4];
  #pragma unroll
  for (int q = 0; q < 4; q ++) {
    acr[q] = ZERO_VEC; aci[q] = ZERO_VEC; add[q] = ZERO_VEC;
  }
  for (long g0 = 0; g0 < nreals; g0 += 8 * num_rhs) {
    #pragma unroll
    for (int q = 0; q < 4; q ++) {
      const REAL_VEC * xr0 = (const REAL_VEC *) (x + g0 + 2 * q * num_rhs);
      const REAL_VEC * xi0
        = (const REAL_VEC *) (x + g0 + (2 * q + 1) * num_rhs);
      const REAL_VEC * yr0 = (const REAL_VEC *) (y + g0 + 2 * q * num_rhs);
      const REAL_VEC * yi0
        = (const REAL_VEC *) (y + g0 + (2 * q + 1) * num_rhs);
      #pragma unroll
      for (int rv = 0; rv < num_rhs_vec; rv ++) {
        REAL_VEC xr = xr0[rv], xi = xi0[rv], yr = yr0[rv], yi = yi0[rv];
        acr[q] += xr * yr + xi * yi;
        aci[q] += xr * yi - xi * yr;
        add[q] += xr * xr + xi * xi;
      }
    }
  }
  REAL buf[3][veclen] __attribute__((aligned(64)));
  *(REAL_VEC *) buf[0] = (acr[0] + acr[1]) + (acr[2] + acr[3]);
  *(REAL_VEC *) buf[1] = (aci[0] + aci[1]) + (aci[2] + aci[3]);
  *(REAL_VEC *) buf[2] = (add[0] + add[1]) + (add[2] + add[3]);
  REAL sr = 0, si = 0, sd = 0;
  for (int k = 0; k < veclen; k ++) {
    sr += buf[0][k]; si += buf[1][k]; sd += buf[2][k];
  }
  *cre = sr; *cim = si; *dd = sd;
}

// Fused MR update: dpsi += (ar+i*ai) * r, then r -= (ar+i*ai) * Dr -- one
// pass over the three buffers rather than two separate caxpy passes.  Both
// updates use r's pre-update value (held in registers before r is overwritten).
static inline void blk_caxpy_pair(REAL * dpsi, REAL * r, const REAL * Dr,
                                  REAL ar, REAL ai, long nreals) {
  const REAL_VEC var = SET1(ar), vai = SET1(ai);
  for (long g0 = 0; g0 < nreals; g0 += 2 * num_rhs) {
    REAL_VEC * pr = (REAL_VEC *) (dpsi + g0);
    REAL_VEC * pi = (REAL_VEC *) (dpsi + g0 + num_rhs);
    REAL_VEC * rr = (REAL_VEC *) (r + g0);
    REAL_VEC * ri = (REAL_VEC *) (r + g0 + num_rhs);
    const REAL_VEC * dr = (const REAL_VEC *) (Dr + g0);
    const REAL_VEC * di = (const REAL_VEC *) (Dr + g0 + num_rhs);
    #pragma unroll
    for (int rv = 0; rv < num_rhs_vec; rv ++) {
      REAL_VEC xr = rr[rv], xi = ri[rv];
      pr[rv] += var * xr - vai * xi;
      pi[rv] += var * xi + vai * xr;
      rr[rv] = xr - (var * dr[rv] - vai * di[rv]);
      ri[rv] = xi - (var * di[rv] + vai * dr[rv]);
    }
  }
}
#endif  // !COARSE_SAP_DENSE_INV

// Coarse SAP post-smoother: phi (zero initial guess) <- lv->nu sweeps of
// red-black multiplicative Schwarz approximating D_l phi = eta on level lv
// (depth >= 1).  If r_exit != NULL it receives the exit residual eta - D_l phi
// (site-major, lv->field_len * num_rhs reals) -- nearly free via the
// incremental-residual state (see the export pass at the end); the K-cycle
// uses it to assemble its Arnoldi matvec without a coarse Dslash
// (KCYCLE_ASSEMBLED_W).  CSAP_LAP is defined for all builds (no-op unless
// profiling) because it is used in the shared block-residual gather; the
// counters it references live in the dense block below (COARSE_SAP_PROFILE
// implies COARSE_SAP_DENSE_INV via the Makefile).
#if COARSE_SAP_PROFILE
  #define CSAP_LAP(k) do { double _n = omp_get_wtime(); \
                           csap_ps[tid][k] += _n - _t; _t = _n; } while (0)
#else
  #define CSAP_LAP(k)
#endif

#if COARSE_SAP_DENSE_INV
  #include "dense.h"       // Bunch-Kaufman factor/inverse + GEMM (setup only)

  #if COARSE_SAP_PROFILE
// Per-stage coarse_sap profiling: each thread accumulates wall-time per
// stage into its own row (no atomics); flops are deterministic (set in the
// flop model), so coarse_sap_profile_report multiplies by the total
// block-solve count.
    #define CSAP_NSTAGE 6
    #define CSAP_MAXTHR 256

// per-thread stage seconds
static double csap_ps[CSAP_MAXTHR][CSAP_NSTAGE];
// per-thread block-solve count
static long   csap_nbs[CSAP_MAXTHR];
// per-block-solve flops per stage (this call)
static double csap_sf[CSAP_NSTAGE];
// total flops per stage, accumulated over calls -- csap_sf is a per-call
// snapshot that differs by level (ncw), so a multi-level run must sum
// csap_sf*block-solves per call, not multiply the last snapshot by the
// global count
static double csap_fl[CSAP_NSTAGE];
// sweep-loop cross-rank halo wall (prod only)
static double csap_halo_s;
    #if FERM_GHOSTS
// halo split: pack | MPI | ghost scatter
static double csap_hpack_s, csap_hmpi_s, csap_hscat_s;  
    #endif
// block-loop parallel-region wall (fork+compute+barrier)
static double csap_par_s;
// exit-residual export region wall
static double csap_export_s;
// whole-call wall (internal, cross-checks summary)
static double csap_call_s;
// one-time lazy cache build (See_inv inversion)
static double csap_cache_s;
// Per-depth split of the block-loop wall.  In a 4+-level hierarchy coarse_sap
// runs at several intermediate depths that share the counters above; this
// pins down which one owns the fork/barrier gap (a deep level with fewer
// blocks than threads over-subscribes -- most threads idle at every color
// barrier -- and is called far more often than the shallow one).
    #define CSAP_MAXD 6
// per-depth block-loop region wall
static double csap_par_lvl[CSAP_MAXD];
// per-depth region count (2*nu per call)
static long   csap_reg_lvl[CSAP_MAXD];
// per-depth blocks per color (constant per level)
static int    csap_ncb_lvl[CSAP_MAXD][2];
// Per-depth per-thread stage-work seconds (snapshot-diffed around each
// block-loop region), to localize a slow rank's fork/barrier gap: a few slow
// threads (memory/placement asymmetry) vs all-threads-slower (kernel-rate
// asymmetry) print differently in the min/max spread.
static double csap_thr_lvl[CSAP_MAXD][CSAP_MAXTHR];
// region-entry per-thread work snapshot
static double csap_snap[CSAP_MAXTHR];
static const char * csap_sname[CSAP_NSTAGE]
  = { "gather (resid)", "Soo^-1 fwd", "Meo hop", "See_inv apply",
    "Moe hop+Soo^-1 back", "accumulate" };
void csap_profile_reset(void) {    // solve-only: called by vcycle_stats_reset
  csap_halo_s = csap_par_s = csap_export_s = csap_call_s = csap_cache_s = 0;
    #if FERM_GHOSTS
  csap_hpack_s = csap_hmpi_s = csap_hscat_s = 0;
    #endif
  for (int d = 0; d < CSAP_MAXD; d ++) {
    csap_par_lvl[d] = 0; csap_reg_lvl[d] = 0;
    csap_ncb_lvl[d][0] = csap_ncb_lvl[d][1] = 0;
    for (int q = 0; q < CSAP_MAXTHR; q ++) csap_thr_lvl[d][q] = 0;
  }
  for (int k = 0; k < CSAP_NSTAGE; k ++) csap_fl[k] = 0;
  for (int q = 0; q < CSAP_MAXTHR; q ++) {
    csap_nbs[q] = 0;
    for (int k = 0; k < CSAP_NSTAGE; k ++) csap_ps[q][k] = 0;
  }
}
  #endif  // COARSE_SAP_PROFILE
// --- dense per-block inverse (setup precompute + apply) --------------------
// inv = A^{-1} (both n x n row-major complex) in fp64.  A is gamma5c-Hermitian:
// Gamma5c*A is Hermitian, where Gamma5c = diag(+1 x chi/2, -1 x chi/2)
// repeats with period chi down the coarse-var index (the coarse chirality
// signature; the two chirality halves are contiguous, so the sign depends
// only on idx % chi).  Factor the Hermitian H = Gamma5c*A with Bunch-Kaufman
// (dense.h) -- ~half the flops of a general LU, touching only one triangle --
// then undo the signature: A^{-1} = (Gamma5c H)^{-1} = H^{-1} Gamma5c.
// Returns -1 if A is singular.  Rebuilt once per coarse-operator build and
// cached per level.
static int csap_cinv(const double complex * A, int n, int chi,
                     double complex * inv) {
  // inv <- Gamma5c A : sign-flip lower-chirality rows
  for (int i = 0; i < n; i ++) {
    double s = ((i % chi) < chi / 2) ? 1.0 : -1.0;
    for (int j = 0; j < n; j ++)
      inv[(size_t) i * n + j] = s * A[(size_t) i * n + j];
  }
  int * ipiv = malloc(sizeof(int) * n);
  int info = dense_hetrf(inv, n, ipiv);
  if (!info) info = dense_hetri(inv, n, ipiv);
  free(ipiv);
  if (info) return -1;
  for (int i = 0; i < n; i ++)  // only 'L' is promised; mirror the strict upper
    for (int j = i + 1; j < n; j ++)
      inv[(size_t) i * n + j] = conj(inv[(size_t) j * n + i]);
  for (int i = 0; i < n; i ++)  // undo the signature on columns: H^{-1} Gamma5c
    for (int j = 0; j < n; j ++)
      inv[(size_t) i * n + j] *= ((j % chi) < chi / 2) ? 1.0 : -1.0;
  return 0;
}

// dpsi = Minv . r for one block.  Minv is (bsites*nvar)^2 complex, row-major
// interleaved re/im.  r and dpsi are block fermions in coarse_sap's
// site-major layout: site lb, var v -> re at lb*S + (v*2)*num_rhs + rhs,
// im one num_rhs on.
//
// The apply's AI = num_rhs flop/byte (the matrix streams once and the input
// vectors stay cache-resident), comparable to the DRAM ridge on most machines
// -- so whether bandwidth or FMA throughput is the binding ceiling depends on
// the machine, and the untiled form sits below both.  A single output row
// carries only 2*num_rhs_vec independent accumulator chains (accr/acci per
// rhs-vec); at small num_rhs that is too few to hide the FMA latency and the
// pipe runs part-idle.  So tile ROWTILE = 4/num_rhs_vec output rows into one
// sweep over the input vectors: ~8 chains regardless of num_rhs, and each
// rre/rim load is reused ROWTILE times.  Bit-identical to the untiled form
// -- every row's reduction runs the same column order.
  #ifndef CSAP_ROWTILE
    #define CSAP_ROWTILE ((4 / num_rhs_vec) > 0 ? (4 / num_rhs_vec) : 1)
  #endif
// Callers pass whole sites of N = 2*nnv rows with nnv even (bootstrap.c), so
// N % 4 == 0 and a tile dividing 4 never leaves a partial tile.
_Static_assert(4 % CSAP_ROWTILE == 0, "CSAP_ROWTILE must be a factor of 4");
static void csap_dense_apply_rows(REAL * restrict dpsi,
                                  const REAL * restrict Minv,
                                  const REAL * restrict r,
                                  long bsites, int N, long row_lo,
                                  long row_hi) {
  const long dim = bsites * N;
  const long S   = (long) N * 2 * num_rhs;
  enum { T = CSAP_ROWTILE };
  for (long row = row_lo; row < row_hi; row += T) {
    REAL_VEC accr[T][num_rhs_vec], acci[T][num_rhs_vec];
    const REAL * Mrow[T];
    REAL_VEC   * ore[T], * oim[T];
    for (int t = 0; t < T; t ++) {
      long lo = (row + t) / N, a = (row + t) % N;
      ore[t]  = (REAL_VEC *) (dpsi + lo * S + (a * 2) * (long) num_rhs);
      oim[t]  = (REAL_VEC *) (dpsi + lo * S + (a * 2 + 1) * (long) num_rhs);
      Mrow[t] = Minv + (row + t) * dim * 2;
      for (int rv = 0; rv < num_rhs_vec; rv ++) {
        accr[t][rv] = ZERO_VEC;
        acci[t][rv] = ZERO_VEC;
      }
    }
    // Column walk as nested (site li, var b) with a running col index, so the
    // per-column site/var split needs no div/mod on the runtime width N
    // (col/N, col%N would otherwise be an O(dim^2) division in this hot loop).
    // Same column order (col = li*N + b) -> bit-identical to the flat form.
    long col = 0;
    for (long li = 0; li < bsites; li ++) {
      const REAL * rb = r + li * S;
      for (int b = 0; b < N; b ++, col ++) {
        const REAL_VEC * rre
          = (const REAL_VEC *) (rb + (b * 2) * (long) num_rhs);
        const REAL_VEC * rim
          = (const REAL_VEC *) (rb + (b * 2 + 1) * (long) num_rhs);
        for (int t = 0; t < T; t ++) {
          REAL_VEC mre = SET1(Mrow[t][col * 2]),
                   mim = SET1(Mrow[t][col * 2 + 1]);
          for (int rv = 0; rv < num_rhs_vec; rv ++) {
            accr[t][rv] += mre * rre[rv] - mim * rim[rv];
            acci[t][rv] += mre * rim[rv] + mim * rre[rv];
          }
        }
      }
    }
    for (int t = 0; t < T; t ++)
      for (int rv = 0; rv < num_rhs_vec; rv ++) {
        ore[t][rv] = accr[t][rv];
        oim[t][rv] = acci[t][rv];
      }
  }
}
// Full-matrix apply: rows [0, bsites*N).  Each output row's column reduction
// is self-contained, so a sub-range [row_lo,row_hi) applies independently and
// bit-identically -- the stage-major smoother uses that to split one block's
// See_inv apply across threads by output site.
static void csap_dense_apply(REAL * dpsi, const REAL * Minv, const REAL * r,
                             long bsites, int N) {
  csap_dense_apply_rows(dpsi, Minv, r, bsites, N, 0, bsites * (long) N);
}

// --- packed Hermitian-triangle storage + blocked-HEMM apply ----------------
// Both dense inverses are Gamma5c-Hermitian: csap_cinv returns
// A^-1 = H^-1 Gamma5c with H^-1 exactly Hermitian (zhetri fills 'L'; the
// upper triangle is its conj mirror), so the full stored matrix is redundant
// by half.  Storing only H^-1's lower triangle halves the matrix stream --
// the one the apply is bound by -- at identical flops: each stored element
// serves its direct (i,j) product and, conjugated, the mirrored (j,i)
// product.  The chirality signature moves to the input:
// y = A^-1 x = H^-1 (Gamma5c x), so callers pass xs = Gamma5c x (negate the
// lower-chirality half of each site's vars) and the apply is pure Hermitian.
//
// Layout: the lower tile-triangle is stored as square ts x ts tiles, row-major
// within a tile, tiles ordered (I, J<=I) row-panel-major: tile (I,J) at
// (I*(I+1)/2 + J) * ts*ts*2 REALs.  Off-diagonal tiles hold H^-1's lower-
// triangle blocks and get two passes -- dot y_I += T x_J, then mirror
// y_J += T^H x_I while the tile is L1-hot -- so each is read once for both
// uses.  Diagonal tiles are stored full (mirror-completed: the full inverse
// already holds the conjugate upper) and get a single dot pass: same storage
// as a padded triangle, but no wasted pad MACs and one fewer pass.  The tile
// edge is per-matrix, a pure function of dim (64 when it divides, else 32 --
// requires ncw % CSAP_TS == 0, checked at the format decision): the asm
// kernels take n_coarse at runtime, so both edges run the same code, and the
// bigger tile halves the per-call overhead and the y-panel re-loads wherever
// the dimension allows.  Both passes run the tuned asm coarse-Dslash
// microkernels where available (ncpl=1, runtime n_coarse = ts; base/_acc =
// dot, _tconj = mirror), which beat the best C codegen by ~1.4x; portable C
// tile kernels serve every other TU (double twin, AVX2, no-asm builds).
// Reduction order differs from csap_dense_apply: not bit-identical, same
// reassociation class as blk_dot3.
enum { CSAP_TS = 32 };          // fallback tile edge (divides every real ncw)
static inline long csap_ts(long dim) { return dim % 64 == 0 ? 64 : CSAP_TS; }
static inline long csap_tri_reals_ts(long dim, long ts) {
  long nt = dim / ts;
  return nt * (nt + 1) / 2 * ts * ts * 2;
}
static inline long csap_tri_reals(long dim) {   // REALs per packed matrix
  return csap_tri_reals_ts(dim, csap_ts(dim));
}

// Pack H^-1's tile-triangle out of the full A^-1 = H^-1 Gamma5c that
// csap_cinv produced: T[i][j] = Ainv[i][j] * gamma_j undoes the column
// signature -- a pure sign flip, so the packed values are bit-exactly
// zhetri's (diagonal tiles pick up the conjugate upper the same way; the full
// Ainv already holds it).
static void csap_pack_tri_ts(const double complex * Ainv, int n, int chi,
                             REAL * dst, long ts) {
  long nt = n / ts;
  for (long ti = 0; ti < nt; ti ++)        // ti/tj, not I: <complex.h> owns I
    for (long tj = 0; tj <= ti; tj ++) {
      REAL * t = dst + (ti * (ti + 1) / 2 + tj) * ts * ts * 2;
      for (long r = 0; r < ts; r ++)
        for (long c = 0; c < ts; c ++) {
          long i = ti * ts + r, j = tj * ts + c;
          long o = (r * ts + c) * 2;
          double s = ((j % chi) < chi / 2) ? 1.0 : -1.0;
          t[o]     = (REAL) (s * creal(Ainv[i * (long) n + j]));
          t[o + 1] = (REAL) (s * cimag(Ainv[i * (long) n + j]));
        }
    }
}
static void csap_pack_tri(const double complex * Ainv, int n, int chi,
                          REAL * dst) {
  csap_pack_tri_ts(Ainv, n, chi, dst, csap_ts(n));
}

  #if ASM && MG_SINGLE && AVX512
// The blocked-HEMM passes ride the asm microkernels
// (gen_coarse_dslash_avx512.py emits all three from one body): base = dot
// overwrite, _acc = dot accumulate, _tconj = transposed-conj accumulate.
void coarse_dslash_kernel_avx512_acc(REAL * out, REAL ** cbases,
                                     REAL ** ibases, long coff, int ncpl,
                                     long n_coarse);
void coarse_dslash_kernel_avx512_tconj(REAL * out, REAL ** cbases,
                                       REAL ** ibases, long coff, int ncpl,
                                       long n_coarse);
    #define CSAP_TILE_ASM 1
  #else
// Portable tile kernels (double twin, AVX2, no-asm builds).  Buffers must not
// alias.  csap_tile_dot: o (ow ? = : +=) T . x over one ts x ts tile;
// csap_tile_tconj: o += T^H . x (reads tile columns -- fine at L1-hot scale).
static void csap_tile_dot(REAL * restrict o, const REAL * restrict t,
                          const REAL * restrict x, long ts, int ow) {
  for (int a = 0; a < ts; a ++) {
    REAL_VEC * ore = (REAL_VEC *) (o + (long) (a * 2) * num_rhs);
    REAL_VEC * oim = (REAL_VEC *) (o + (long) (a * 2 + 1) * num_rhs);
    REAL_VEC accr[num_rhs_vec], acci[num_rhs_vec];
    for (int rv = 0; rv < num_rhs_vec; rv ++) {
      accr[rv] = ow ? ZERO_VEC : ore[rv];
      acci[rv] = ow ? ZERO_VEC : oim[rv];
    }
    const REAL * tr = t + (long) a * ts * 2;
    for (int b = 0; b < ts; b ++) {
      REAL_VEC cre = SET1(tr[b * 2]), cim = SET1(tr[b * 2 + 1]);
      const REAL_VEC * xre
        = (const REAL_VEC *) (x + (long) (b * 2) * num_rhs);
      const REAL_VEC * xim
        = (const REAL_VEC *) (x + (long) (b * 2 + 1) * num_rhs);
      for (int rv = 0; rv < num_rhs_vec; rv ++) {
        accr[rv] += cre * xre[rv] - cim * xim[rv];
        acci[rv] += cre * xim[rv] + cim * xre[rv];
      }
    }
    for (int rv = 0; rv < num_rhs_vec; rv ++) {
      ore[rv] = accr[rv];
      oim[rv] = acci[rv];
    }
  }
}
static void csap_tile_tconj(REAL * restrict o, const REAL * restrict t,
                            const REAL * restrict x, long ts) {
  for (int j = 0; j < ts; j ++) {                // output rows = tile columns
    REAL_VEC * ore = (REAL_VEC *) (o + (long) (j * 2) * num_rhs);
    REAL_VEC * oim = (REAL_VEC *) (o + (long) (j * 2 + 1) * num_rhs);
    REAL_VEC accr[num_rhs_vec], acci[num_rhs_vec];
    for (int rv = 0; rv < num_rhs_vec; rv ++) {
      accr[rv] = ore[rv];
      acci[rv] = oim[rv];
    }
    for (int r = 0; r < ts; r ++) {
      REAL_VEC cre = SET1(t[(long) r * ts * 2 + j * 2]),
               cim = SET1(t[(long) r * ts * 2 + j * 2 + 1]);
      const REAL_VEC * xre
        = (const REAL_VEC *) (x + (long) (r * 2) * num_rhs);
      const REAL_VEC * xim
        = (const REAL_VEC *) (x + (long) (r * 2 + 1) * num_rhs);
      for (int rv = 0; rv < num_rhs_vec; rv ++) {
        accr[rv] += cre * xre[rv] + cim * xim[rv];    // conj(t)
        acci[rv] += cre * xim[rv] - cim * xre[rv];
      }
    }
    for (int rv = 0; rv < num_rhs_vec; rv ++) {
      ore[rv] = accr[rv];
      oim[rv] = acci[rv];
    }
  }
}
  #endif // COARSE_SAP_DENSE_INV

// y = H^-1 . xs, blocked HEMM over the packed tiles: per stored tile (I, J<=I),
// a dot pass into y_I (overwrite at J==0, the row-panel's first touch; else
// accumulate); off-diagonal tiles are immediately followed by the mirrored
// transposed-conj pass into y_J while the tile is L1-hot (diagonal tiles are
// stored full and need no mirror).  Mirror contributions into a panel y_P only
// occur at I > P, after dot(P,0) initialized it, so ascending (I,J) order is
// the whole correctness story.  y must not alias xs (restrict).
static void csap_herm_apply_ts(REAL * restrict y, const REAL * restrict T,
                               const REAL * restrict xs, long bsites, int N,
                               long ts) {
  const long vb  = 2 * (long) num_rhs;     // REALs per coarse var
  const long dim = bsites * (long) N;
  const long nt  = dim / ts;               // tile panels
  const long tsz = ts * ts * 2;            // REALs per stored tile
  for (long ti = 0; ti < nt; ti ++) {      // ti/tj, not I: <complex.h> owns I
    const REAL * xI = xs + ti * ts * vb;
    for (long tj = 0; tj <= ti; tj ++) {
      const REAL * tile = T + (ti * (ti + 1) / 2 + tj) * tsz;
      const REAL * xJ   = xs + tj * ts * vb;
  #if CSAP_TILE_ASM
      REAL * cb[2], * ib[1];
      cb[0] = (REAL *) tile;
      // The dot kernels' one-past slot prefetches cbases[1]: point it at the
      // next stored tile (tiles are consecutive), so the dot calls' cursors
      // collectively pull the next tile into cache while this one computes --
      // the same next-matrix scheme the ncpl=9 stencil callers get.  After the
      // last tile this prefetches one-past the allocation: architecturally
      // safe (prefetch never faults) and the contents are irrelevant.
      cb[1] = (REAL *) (tile + tsz);
      ib[0] = (REAL *) xJ;
      for (long a0 = 0; a0 < ts; a0 += DC_MR) {
        REAL * o = y + (ti * ts + a0) * vb;
        if (tj == 0)
          DC_KERNEL(o, cb, ib, a0 * ts * 2 * (long) sizeof(REAL), 1, ts);
        else
          coarse_dslash_kernel_avx512_acc(o, cb, ib,
              a0 * ts * 2 * (long) sizeof(REAL), 1, ts);
      }
      if (tj < ti) {                              // diagonal tiles: dot only
        ib[0] = (REAL *) xI;
        for (long j0 = 0; j0 < ts; j0 += DC_MR)
          coarse_dslash_kernel_avx512_tconj(y + (tj * ts + j0) * vb, cb, ib,
              j0 * 2 * (long) sizeof(REAL), 1, ts);
      }
  #else
      csap_tile_dot(y + ti * ts * vb, tile, xJ, ts, tj == 0);
      if (tj < ti)                                // diagonal tiles: dot only
        csap_tile_tconj(y + tj * ts * vb, tile, xI, ts);
  #endif
    }
  }
}
static void csap_herm_apply(REAL * restrict y, const REAL * restrict T,
                            const REAL * restrict xs, long bsites, int N) {
  csap_herm_apply_ts(y, T, xs, bsites, N, csap_ts(bsites * (long) N));
}

// --- `./qc csapbench`: See_inv-apply kernel bench --------------------------
// Times csap_dense_apply (full matrix) and csap_herm_apply (packed triangle)
// at See-like dims, in two regimes: hot (one matrix per thread, L2-resident)
// and stream (per-thread pool far beyond the L3 share -- the production
// regime).  hot ~= stream pins the kernel's limiter in the core/L2 path;
// hot >> stream pins it on the DRAM stream.  Random data, perf-only
// (correctness is gated by the solver tests); no perf(1) needed.
static double csb_wall;
void coarse_csap_bench(void) {
  const int nthr = omp_get_max_threads();
  printf("[csap-bench] threads %d, num_rhs %d, REAL %zuB, GF/s/core (hot-L2 | stream-DRAM)\n",
         nthr, num_rhs, sizeof(REAL));
  static const int csbN[2] = { 64, 96 };
  for (int ni = 0; ni < 2; ni ++) {
    const int  N     = csbN[ni];
    const long bs    = 8;                       // nh of a 2^4 SAP block
    const long dim   = bs * N;
    const long S     = (long) N * 2 * num_rhs;
    const long vlen  = bs * S;                  // x / y reals
    const long fullr = dim * dim * 2;
    const long trir  = csap_tri_reals(dim);
    const long trir32 = csap_tri_reals_ts(dim, CSAP_TS);   // ts=32 A/B row
    // per-thread pool sized ~72MB per format:
    // far beyond any per-thread L3 share
    const int npf  = (int) (72e6 / (fullr  * sizeof(REAL))) + 1;
    const int npt  = (int) (72e6 / (trir   * sizeof(REAL))) + 1;
    const int npt3 = (int) (72e6 / (trir32 * sizeof(REAL))) + 1;
    const int KHOT = 400;
    const double flops1 = (double) dim * dim * num_rhs * 8.0;
    double w[3][2], fl[3][2];
    #pragma omp parallel
    {
      unsigned long sd = 12345u + 777u * (unsigned) omp_get_thread_num();
      // Knuth MMIX linear congruential generator; see test/kernel_tests.c
      #define CSB_R (sd = sd * 6364136223846793005UL + 1442695040888963407UL, \
                     (REAL) ((sd >> 40) & 0xffff) * (REAL) 1e-6)
      REAL * x  = malloc(sizeof(REAL) * vlen);
      REAL * yv = malloc(sizeof(REAL) * vlen);
      REAL * pf = malloc(sizeof(REAL) * fullr * npf);
      REAL * pt = malloc(sizeof(REAL) * trir * npt);
      REAL * p3 = malloc(sizeof(REAL) * trir32 * npt3);
      for (long i = 0; i < vlen; i ++) { x[i] = CSB_R; yv[i] = 0; }
      for (long i = 0; i < fullr * npf; i ++) pf[i] = CSB_R;
      for (long i = 0; i < trir * npt;  i ++) pt[i] = CSB_R;
      for (long i = 0; i < trir32 * npt3; i ++) p3[i] = CSB_R;
      for (int kern = 0; kern < 3; kern ++)
        for (int reg = 0; reg < 2; reg ++) {
          const int np   = kern == 0 ? npf : kern == 1 ? npt : npt3;
          const int reps = reg == 0 ? KHOT : 8 * np;
          #pragma omp barrier
          #pragma omp master
          csb_wall = omp_get_wtime();
          #pragma omp barrier
          for (int k = 0; k < reps; k ++) {
            long mi = reg == 0 ? 0 : k % np;  // hot: one matrix; stream: cycle
            if      (kern == 0) csap_dense_apply(yv, pf + mi * fullr, x, bs, N);
            else if (kern == 1) csap_herm_apply(yv, pt + mi * trir, x, bs, N);
            else csap_herm_apply_ts(yv, p3 + mi * trir32, x, bs, N, CSAP_TS);
          }
          #pragma omp barrier
          #pragma omp master
          { w[kern][reg]  = omp_get_wtime() - csb_wall;
            fl[kern][reg] = (double) nthr * reps * flops1; }
        }
      free(x); free(yv); free(pf); free(pt); free(p3);
      #undef CSB_R
    }
    printf("[csap-bench] N=%d dim=%ld ts=%ld (full %.1fMB tri %.1fMB tri32 %.1fMB)\n",
           N, dim, csap_ts(dim), fullr * sizeof(REAL) / 1048576.0,
           trir * sizeof(REAL) / 1048576.0, trir32 * sizeof(REAL) / 1048576.0);
    static const char * kn[3] = { "dense-full", "tri-fused", "tri-ts32" };
    for (int kern = 0; kern < 3; kern ++)
      printf("  %-12s %7.1f | %7.1f\n", kn[kern],
             fl[kern][0] / w[kern][0] / 1e9 / nthr,
             fl[kern][1] / w[kern][1] / 1e9 / nthr);
  }
}

// --- `./qc csapcheck`: packed-Hermitian apply vs the dense reference -------
// Builds a random Gamma5c-Hermitian matrix, stores it in both formats, and
// compares csap_herm_apply against csap_dense_apply on a random input.  This is
// the fast correctness gate for the blocked-HEMM tile kernels -- in particular
// it runs the asm variants in minutes under `sde64 -skx` on a non-AVX-512 host,
// where the full solver gates are impractically slow.
void coarse_csap_check(void) {
  // dims chosen to cover both tile edges: 512/768 -> ts=64, 2592 (the real
  // 3^3x2-block depth-2 shape) -> ts=32
  static const int cscN[3] = { 64, 96, 96 }, cscB[3] = { 8, 8, 27 };
  int fail = 0;
  for (int ni = 0; ni < 3; ni ++) {
    const int  N    = cscN[ni];
    const long bs   = cscB[ni], dim = bs * N;
    const long S    = (long) N * 2 * num_rhs;      // REALs per site
    const long vlen = bs * S;
    double complex * H = malloc(sizeof(double complex) * dim * dim);
    double complex * A = malloc(sizeof(double complex) * dim * dim);
    unsigned long sd = 99;
    // Knuth MMIX linear congruential generator; see test/kernel_tests.c
    #define CSC_R (sd = sd * 6364136223846793005UL + 1442695040888963407UL, \
                   (double) ((sd >> 40) & 0xffff) * 1e-5 - 0.32768)
    for (long i = 0; i < dim; i ++) {              // random Hermitian H
      for (long j = 0; j < i; j ++) {
        double re = CSC_R, im = CSC_R;
        H[i * dim + j] = re + I * im;
        H[j * dim + i] = re - I * im;
      }
      H[i * dim + i] = CSC_R;               // real diagonal
    }
    for (long i = 0; i < dim; i ++)         // A = H * Gamma5c (column signs)
      for (long j = 0; j < dim; j ++)
        A[i * dim + j] = H[i * dim + j] * (((j % N) < N / 2) ? 1.0 : -1.0);
    REAL * full = malloc(sizeof(REAL) * dim * dim * 2);
    REAL * tri  = malloc(sizeof(REAL) * csap_tri_reals(dim));
    for (long k = 0; k < dim * dim; k ++) {
      full[k * 2]     = (REAL) creal(A[k]);
      full[k * 2 + 1] = (REAL) cimag(A[k]);
    }
    csap_pack_tri(A, (int) dim, N, tri);
    REAL * x  = malloc(sizeof(REAL) * vlen);
    REAL * xg = malloc(sizeof(REAL) * vlen);
    REAL * y1 = malloc(sizeof(REAL) * vlen);
    REAL * y2 = malloc(sizeof(REAL) * vlen);
    for (long k = 0; k < vlen; k ++) x[k] = (REAL) CSC_R;
    for (long lo = 0; lo < bs; lo ++)              // xg = Gamma5c x
      for (long k = 0; k < S; k ++)
        xg[lo * S + k] = k < S / 2 ? x[lo * S + k] : -x[lo * S + k];
    csap_dense_apply(y1, full, x, bs, N);
    csap_herm_apply(y2, tri, xg, bs, N);
    double dmax = 0, nmax = 0;
    for (long k = 0; k < vlen; k ++) {
      double d = fabs((double) y1[k] - (double) y2[k]);
      if (d > dmax) dmax = d;
      if (fabs((double) y1[k]) > nmax) nmax = fabs((double) y1[k]);
    }
    double rel = dmax / (nmax > 0 ? nmax : 1);
    printf("[csap-check] N=%d dim=%ld: max diff %.3e (rel %.3e) => %s\n",
           N, dim, dmax, rel, rel < 1e-4 ? "PASS" : "FAIL");
    fail |= !(rel < 1e-4);
    free(H); free(A); free(full); free(tri);
    free(x); free(xg); free(y1); free(y2);
    #undef CSC_R
  }
  if (fail) exit(1);
}
#endif  // COARSE_SAP_DENSE_INV

#if COARSE_SAP_PROFILE
// Printed at program exit (registered via atexit from coarse_sap).  Stays in
// this TU so it reads the same twin's statics the smoother accumulated into
// (the mixed solve runs the single twin, whose file-static counters this
// reads).
static void coarse_sap_profile_report(void) {
  double secs[CSAP_NSTAGE] = {0}, tot_s = 0;
  long   nbs = 0;
  for (int q = 0; q < CSAP_MAXTHR; q ++) {
    for (int k = 0; k < CSAP_NSTAGE; k ++) secs[k] += csap_ps[q][k];
    nbs += csap_nbs[q];
  }
  for (int k = 0; k < CSAP_NSTAGE; k ++) tot_s += secs[k];
  if (nbs == 0 || tot_s == 0) return;
  printf("\n--- coarse_sap eo profile: gather + apply, solve-only "
         "(rank %d, %ld block-solves, %d threads) ---\n",
         comm_rank(), nbs, omp_get_max_threads());
  printf("  %-22s %10s %10s %8s %8s\n",
         "stage", "core-s", "GFlop/s", "%time", "Gflop");
  for (int k = 0; k < CSAP_NSTAGE; k ++) {
    double fl = csap_fl[k];           // total flops in this stage (all calls)
    if (fl > 0)
      printf("  %-22s %10.3f %10.1f %7.1f%% %8.2f\n", csap_sname[k], secs[k],
             secs[k] > 0 ? fl / secs[k] / 1e9 : 0.0, 100.0 * secs[k] / tot_s, fl / 1e9);
    else                              // memory-bound: time only
      printf("  %-22s %10.3f %10s %7.1f%% %8s\n", csap_sname[k], secs[k],
             "--", 100.0 * secs[k] / tot_s, "--");
  }
  double flt = 0; for (int k = 0; k < CSAP_NSTAGE; k ++) flt += csap_fl[k];
  printf("  %-22s %10.3f %10.1f %7.1f%% %8.2f  (per-core rates; x threads for aggregate)\n",
         "TOTAL", tot_s, flt / tot_s / 1e9, 100.0, flt / 1e9);
  // stage core-s spread over threads
  double busy_wall = tot_s / omp_get_max_threads();
  double serial = csap_call_s - csap_par_s - csap_halo_s - csap_export_s 
                 - csap_cache_s;
  printf("  wall decomposition (per rank): call %.3f s = parallel-regions %.3f s (busy %.3f s + "
         "%.3f s fork/barrier) + halo %.3f s + export %.3f s + cache-build %.3f s + serial %.3f s\n",
         csap_call_s, csap_par_s, busy_wall, csap_par_s - busy_wall,
         csap_halo_s, csap_export_s, csap_cache_s, serial);
#if FERM_GHOSTS
  // The halved-exchange internals (in-sweep + export exchanges together): the
  // MPI span is the single-threaded copy the color-halving attacks; pack/scatter
  // are the threaded passes it added.
  if (csap_hpack_s + csap_hmpi_s + csap_hscat_s > 0)
    printf("    halo split (color-halved dpsi): pack %.3f s | mpi %.3f s | scatter %.3f s\n",
           csap_hpack_s, csap_hmpi_s, csap_hscat_s);
#endif
  // Per-depth block-loop split: a level whose blocks/color is below the
  // thread count over-subscribes (idle threads at every color barrier), which
  // the busy = core-s / nthr model above cannot see and so charges to
  // "fork/barrier".  ms/region exposes the fixed per-region tax
  // (fork + barrier + any dispatch) at that level.
  for (int d = 0; d < CSAP_MAXD; d ++) {
    if (!csap_reg_lvl[d]) continue;
    // per-thread work spread inside this depth's regions: a wide max/min gap =
    // a few slow threads (memory/placement asymmetry); a uniform shift = a
    // kernel-rate difference.  Threads past the depth's useful count show ~0.
    int nthr = omp_get_max_threads();
    double tmin = 0, tmax = 0; int qmin = 0, qmax = 0;
    for (int q = 0; q < nthr && q < CSAP_MAXTHR; q ++) {
      double s = csap_thr_lvl[d][q];
      if (q == 0 || s < tmin) { tmin = s; qmin = q; }
      if (q == 0 || s > tmax) { tmax = s; qmax = q; }
    }
    printf("    depth %d: %ld regions, blocks/color %d/%d over %d threads, "
           "block-loop %.3f s (%.3f ms/region), thr work min %.3f s (t%d) / "
           "max %.3f s (t%d)\n",
           d, csap_reg_lvl[d], csap_ncb_lvl[d][0], csap_ncb_lvl[d][1],
           nthr, csap_par_lvl[d],
           1e3 * csap_par_lvl[d] / (double) csap_reg_lvl[d],
           tmin, qmin, tmax, qmax);
  }
}
#endif

#if FERM_GHOSTS
// Color-halved cross-rank dpsi exchange.  Ships only the boundary-layer sites
// whose owning SAP block color == parity -- the color the previous phase just
// solved, i.e. the only face entries whose dpsi changed and exactly the set
// the next phase's cross-rank gathers read (a face-adjacent neighbor block is
// always the opposite color; odd global block grids abort at the smoother
// entry, and decomposed levels are never self-wrapping).  The receiving ghost
// slots are the same physical sites seen from the other side: this rank's
// layer table at parity 1-parity, scattered in the sender's
// ascending-face-index pack order.  Every byte lands where the whole-face
// halo_coarse would have put it, so the smoother's reads are bit-identical
// -- only the dead half of the payload stops traveling.  Pack/scatter are
// threaded (the call sites are serial); halo_ferm_sap moves the bytes
// (asymmetric per-face sizes, distinct tags, safe at P=2 self-neighbor; each
// call fully drains, so sharing its tag block with the fine-SAP exchange
// cannot cross-match).
static REAL * csap_hbuf[8];             // 4 send + 4 recv faces, grown lazily
static long   csap_hbuf_reals = 0;
static void csap_halo_sap(level_t * lv, long S, int parity) {
  const long plane = (long) lv->nx * lv->ny;
  const long zface = plane * lv->nt, tface = plane * lv->nz;
  const long fmax  = (DECOMP_Z && zface > tface) ? zface : tface;
  if (fmax * S > csap_hbuf_reals) {
    for (int q = 0; q < 8; q ++) {
      free(csap_hbuf[q]);
      csap_hbuf[q] = malloc(sizeof(REAL) * fmax * S);
    }
    csap_hbuf_reals = fmax * S;
  }
  const REAL * f = lv->sap_dpsi_field;
  // ghost region bases: same layout rule the gathers' fidx uses
  const long goff[4] = { 0, zface, 2 * zface * DECOMP_Z,
                         2 * zface * DECOMP_Z + tface };
#if COARSE_SAP_PROFILE
  double _t0 = omp_get_wtime();
#endif
  const void * sb[4]; void * rb[4]; long sbytes[4], rbytes[4];
  for (int L = 0; L < 4; L ++) {
    const int  act = (L < 2) ? DECOMP_Z : DECOMP_T;
    const long ns  = act ? lv->sap_fcnt[L * 2 + parity]       : 0;
    const long nr  = act ? lv->sap_fcnt[L * 2 + (1 - parity)] : 0;
    sb[L] = csap_hbuf[L]; rb[L] = csap_hbuf[4 + L];
    sbytes[L] = ns * S * (long) sizeof(REAL);
    rbytes[L] = nr * S * (long) sizeof(REAL);
    if (!ns) continue;
    const long * tab = lv->sap_ftab[L * 2 + parity];
    REAL * dst = csap_hbuf[L];
    #pragma omp parallel for schedule(static)
    for (long k = 0; k < ns; k ++) {
      long i = tab[k], fld;
      if (L < 2) {                   // z layer: face (t,y,x) is field-strided
        long t = i / plane, r = i % plane;
        fld = (t * lv->nz + (L == 1 ? lv->nz - 1 : 0)) * plane + r;
      } else                         // t layer: contiguous slab
        fld = (L == 3 ? (long) (lv->nt - 1) * tface : 0) + i;
      memcpy(dst + k * S, f + fld * S, sizeof(REAL) * S);
    }
  }
#if COARSE_SAP_PROFILE
  double _t1 = omp_get_wtime();
#endif
  halo_ferm_sap(sb, rb, sbytes, rbytes);
#if COARSE_SAP_PROFILE
  double _t2 = omp_get_wtime();
#endif
  for (int L = 0; L < 4; L ++) {
    const int  act = (L < 2) ? DECOMP_Z : DECOMP_T;
    const long nr  = act ? lv->sap_fcnt[L * 2 + (1 - parity)] : 0;
    if (!nr) continue;
    const long * tab = lv->sap_ftab[L * 2 + (1 - parity)];
    const REAL * src = csap_hbuf[4 + L];
    REAL * gL = lv->sap_dpsi_ghost + goff[L] * S;
    #pragma omp parallel for schedule(static)
    for (long k = 0; k < nr; k ++)
      memcpy(gL + tab[k] * S, src + k * S, sizeof(REAL) * S);
  }
#if COARSE_SAP_PROFILE
  csap_hpack_s += _t1 - _t0;
  csap_hmpi_s  += _t2 - _t1;
  csap_hscat_s += omp_get_wtime() - _t2;
#endif
}

// Ghost-face geometry of one level, in halo_coarse's site-major
// [z-lo | z-hi | t-lo | t-hi] layout.  z faces precede t faces only when z is
// actually decomposed, so the t offsets shift accordingly.
typedef struct { long zlo, zhi, tlo, thi; int nx, ny; } coarse_ghost_faces;

static inline coarse_ghost_faces coarse_ghost_faces_of(const level_t * lv) {
  const long czface = (long) lv->nx * lv->ny * lv->nt;  // z-face sites (x,y,t)
  const long ctface = (long) lv->nx * lv->ny * lv->nz;  // t-face sites (x,y,z)
  coarse_ghost_faces f;
  f.zlo = 0;
  f.zhi = czface;
  f.tlo = 2 * czface * DECOMP_Z;
  f.thi = f.tlo + ctface;
  f.nx  = lv->nx;
  f.ny  = lv->ny;
  return f;
}

// Slot holding the off-rank neighbor in direction `dir` of the coarse site at
// (gx,gy,gz,gt).  Only z/t faces exist -- x and y are never decomposed -- so
// any other direction returns 0; callers reach this only where sap_nbr_rank
// says the neighbor really is cross-rank.
static inline long coarse_ghost_fidx(coarse_ghost_faces f, int dir,
                                     int gx, int gy, int gz, int gt) {
  switch (dir) {
    case 2: return f.zhi + ((long) gt * f.ny + gy) * f.nx + gx;
    case 6: return f.zlo + ((long) gt * f.ny + gy) * f.nx + gx;
    case 3: return f.thi + ((long) gz * f.ny + gy) * f.nx + gx;
    case 7: return f.tlo + ((long) gz * f.ny + gy) * f.nx + gx;
    default: return 0;
  }
}
#endif

int coarse_sap(level_t * lv, REAL * phi, REAL * eta, REAL * r_exit) {
  // Coarse variable count of this level.  N_coarse (== 2*num_null_vecs) is the
  // depth-1 value; deeper levels under an asymmetric hierarchy carry their own
  // width (level_t.nvar), so read it from lv and use ncw everywhere the block
  // stencil / dense inverse would otherwise bake N_coarse.  On L1 (and every
  // symmetric config) ncw == N_coarse, so this is bit-identical there.
  const int  ncw    = lv->nvar;
  const long S      = (long) ncw * 2 * num_rhs;   // reals per coarse site
  const long matsz  = (long) ncw * ncw * 2;       // reals per coupling block
#if COARSE_SAP_PROFILE
  {
    static int _reg = 0;
    if (!_reg) {
      atexit(coarse_sap_profile_report);
      _reg = 1;
    }
  }
#endif
  REAL * self = lv->Dc->self;
  REAL * nbr  = lv->Dc->nbr;
#if COARSE_SAP_PROFILE
  double _cv = omp_get_wtime();
#endif

  // SAP block = sap_blk, falling back to the aggregation block lv->blk when
  // unset (they are the same parameter in DDalphaAMG; lv->blk is always set
  // on a level that has a child, so synthetic/test levels that skip
  // mg_add_coarse_level still get a valid block here).
  int bl[4];
  for (int d = 0; d < 4; d ++)
    bl[d] = lv->sap_blk[d] > 0 ? lv->sap_blk[d] : lv->blk[d];
  int nb[4] = { lv->nx / bl[0], lv->ny / bl[1],
                lv->nz / bl[2], lv->nt / bl[3] };
  assert(lv->nx % bl[0] == 0 && lv->ny % bl[1] == 0 &&
         lv->nz % bl[2] == 0 && lv->nt % bl[3] == 0);
  long nblocks = (long) nb[0] * nb[1] * nb[2] * nb[3];
  long bsites  = (long) bl[0] * bl[1] * bl[2] * bl[3];
  long blk_reals = bsites * S;

  // --- per-level cache: per-thread scratch + block geometry, built on the
  // first call (everything below depends only on the level's extents, SAP
  // block, and thread count, all fixed after setup).  Freed by mg_teardown.
  int nthr = omp_get_max_threads();
#if COARSE_SAP_PROFILE
  double _kv = omp_get_wtime();
#endif
  if (lv->sap_nthr != nthr) {            // unbuilt (0) or thread-count change
    if (lv->sap_nthr) coarse_sap_cache_free(lv);

    // per-thread scratch (one block's worth each)
    lv->sap_rbuf  = malloc(sizeof(REAL *) * nthr);
    lv->sap_Drbuf = malloc(sizeof(REAL *) * nthr);
    lv->sap_dbuf  = malloc(sizeof(REAL *) * nthr);
    for (int q = 0; q < nthr; q ++) {
      lv->sap_rbuf[q]  = malloc(sizeof(REAL) * blk_reals);
      lv->sap_Drbuf[q] = malloc(sizeof(REAL) * blk_reals);
      lv->sap_dbuf[q]  = malloc(sizeof(REAL) * blk_reals);
    }
#if COARSE_SAP_DENSE_INV
    // eo-Schur scratch: y (both parities), the even sub-vector in/out of the
    // Schur solve, and a one-site buffer for the block-diagonal odd apply.
    lv->sap_ybuf  = malloc(sizeof(REAL *) * nthr);
    lv->sap_ebuf  = malloc(sizeof(REAL *) * nthr);
    lv->sap_ebuf2 = malloc(sizeof(REAL *) * nthr);
    lv->sap_sbuf  = malloc(sizeof(REAL *) * nthr);
    for (int q = 0; q < nthr; q ++) {
      lv->sap_ybuf[q]  = malloc(sizeof(REAL) * blk_reals);
      lv->sap_ebuf[q]  = malloc(sizeof(REAL) * (bsites / 2) * S);
      lv->sap_ebuf2[q] = malloc(sizeof(REAL) * (bsites / 2) * S);
      lv->sap_sbuf[q]  = malloc(sizeof(REAL) * S);
    }
#endif

    // In-block neighbor map [lb*8+dir] -> in-block local nbr (or -1 across a
    // face).  Depends only on the block's local topology, not its position, so
    // it is the same for every block -> shared read-only across blocks/threads.
    lv->sap_lnbr = malloc(sizeof(int) * bsites * 8);
    lv->sap_xnbr = malloc(sizeof(int) * bsites * 8);
    for (int lt = 0; lt < bl[3]; lt ++)
     for (int lz = 0; lz < bl[2]; lz ++)
      for (int ly = 0; ly < bl[1]; ly ++)
       for (int lx = 0; lx < bl[0]; lx ++) {
         long lb = (((long) lt * bl[2] + lz) * bl[1] + ly) * bl[0] + lx;
         int lc[4] = { lx, ly, lz, lt };
         for (int dir = 0; dir < 8; dir ++) {
           int nc[4] = { lc[0] + coarse_off[dir][0], lc[1] + coarse_off[dir][1],
                         lc[2] + coarse_off[dir][2], lc[3] + coarse_off[dir][3] };
           if (nc[0] >= 0 && nc[0] < bl[0] && nc[1] >= 0 && nc[1] < bl[1] &&
               nc[2] >= 0 && nc[2] < bl[2] && nc[3] >= 0 && nc[3] < bl[3])
             lv->sap_lnbr[lb * 8 + dir] =
                 (((nc[3] * bl[2] + nc[2]) * bl[1] + nc[1]) * bl[0] + nc[0]);
           else
             lv->sap_lnbr[lb * 8 + dir] = -1;  // crosses block face -> excluded
           // crossing site's local index inside the neighbor block (wrap the
           // out-of-range coordinate); meaningful only where lnbr == -1
           int wc[4];
           for (int i = 0; i < 4; i ++) wc[i] = (nc[i] + bl[i]) % bl[i];
           lv->sap_xnbr[lb * 8 + dir] =
               (((wc[3] * bl[2] + wc[2]) * bl[1] + wc[1]) * bl[0] + wc[0]);
         }
       }

    // Per-color block lists, so the red/black sweeps iterate exactly their own
    // blocks rather than walking all nblocks and skipping half.  Alongside: the
    // face-neighbor block map for the incremental residual.
    lv->sap_cblk[0] = malloc(sizeof(long) * nblocks);
    lv->sap_cblk[1] = malloc(sizeof(long) * nblocks);
    lv->sap_ncblk[0] = lv->sap_ncblk[1] = 0;
    lv->sap_nbrblk = malloc(sizeof(long) * nblocks * 8);
#if FERM_GHOSTS
    lv->sap_nbr_rank = malloc(sizeof(int) * nblocks * 8);
    // Global red-black block color: a block's rank-face neighbor lives on
    // another rank and must still be the opposite color, so fold in the
    // rank's SAP-block origin parity.  lv->corigin is this level's origin in
    // its own sites (set at every depth), and the SAP block is bl[] sites
    // wide here, so the origin's block count is corigin[d]/bl[d] -- valid at
    // any depth.  (At depth 1 this equals origin[d]/(BLOCK_AGG*bl[d]), the
    // old hard-coded form.)  Rank origins are coarse-block-aligned (checked
    // in comm_init) -> integer counts.  0 at MPI=0.
    const int rank_blk_color = ((lv->corigin[2] / bl[2])
                              + (lv->corigin[3] / bl[3])) & 0x1;
#else
    const int rank_blk_color = 0;
#endif
    for (long blk = 0; blk < nblocks; blk ++) {
      long bb = blk;
      int bc[4];
      bc[0] = bb % nb[0]; bb /= nb[0];
      bc[1] = bb % nb[1]; bb /= nb[1];
      bc[2] = bb % nb[2]; bb /= nb[2];
      bc[3] = (int) bb;
      int col = ((bc[0] + bc[1] + bc[2] + bc[3]) & 1) ^ rank_blk_color;
      lv->sap_cblk[col][lv->sap_ncblk[col] ++] = blk;
      for (int dir = 0; dir < 8; dir ++) {
        int nc[4];
        for (int i = 0; i < 4; i ++)
          nc[i] = (bc[i] + coarse_off[dir][i] + nb[i]) % nb[i];
        lv->sap_nbrblk[blk * 8 + dir] =
            (((long) nc[3] * nb[2] + nc[2]) * nb[1] + nc[1]) * nb[0] + nc[0];
#if FERM_GHOSTS
        // flag the face whose neighbor block crosses a rank boundary (its
        // periodic offset wraps along a decomposed dim) -- read from the dpsi
        // ghost there, not the local (self-wrapped) sap_dpsi block.
        int cr = 0;
        for (int i = 0; i < 4; i ++) {
          int raw = bc[i] + coarse_off[dir][i];
          if ((raw < 0 || raw >= nb[i]) &&
              ((i == 2 && DECOMP_Z) || (i == 3 && DECOMP_T)))
            cr = 1;
        }
        lv->sap_nbr_rank[blk * 8 + dir] = cr;
#endif
      }
    }

    // incremental-residual state: each block's exact residual as of its last
    // visit and its last block correction (written after every block solve,
    // read lazily -- sweep 0 reads eta, so no initialization is needed and
    // nothing leaks between smoother calls).
#if COARSE_SAP_DENSE_INV
    lv->sap_R = NULL;  // eo needs no residual store: post-solve residual is 0
#else
    lv->sap_R = malloc(sizeof(REAL) * nblocks * blk_reals);
#endif
    lv->sap_dpsi = malloc(sizeof(REAL) * nblocks * blk_reals);

#if FERM_GHOSTS
    // Cross-rank state: a site-major snapshot of the latest dpsi (for the
    // per-phase face exchange) and the ghost it lands in.  ghost_sites uses
    // the same z/t face layout halo_coarse computes (z faces present iff z
    // decomposed, then t).  Allocated unconditionally in the MPI build; the
    // face exchange below no-ops on whichever dim is not decomposed.
    {
      long zface = (long) lv->nx * lv->ny * lv->nt;
      long tface = (long) lv->nx * lv->ny * lv->nz;
      long ghost_sites = 2 * zface * DECOMP_Z + 2 * tface * DECOMP_T;
      lv->sap_dpsi_field = malloc(sizeof(REAL) * lv->field_len * num_rhs);
      lv->sap_dpsi_ghost = ghost_sites
                         ? malloc(sizeof(REAL) * ghost_sites * S) : NULL;
      // Face-color tables for the color-halved dpsi exchange (csap_halo_sap):
      // per boundary layer (z-lo, z-hi, t-lo, t-hi), the face-local site
      // indices split by owning-SAP-block color, ascending.  The ghost
      // adjacent to a layer holds the neighbor rank's facing layer, whose
      // blocks are always the opposite color (one block step flips the global
      // checkerboard, and odd global block counts abort at the smoother
      // entry), so one table set serves both the pack side (ships parity p)
      // and the scatter side (fills the ghost's parity-p slots = this rank's
      // layer table at 1-p).
      for (int q = 0; q < 8; q ++) {
        lv->sap_ftab[q] = NULL;
        lv->sap_fcnt[q] = 0;
      }
      for (int L = 0; L < 4; L ++) {
        if (L < 2 ? !DECOMP_Z : !DECOMP_T) continue;
        const long fs = L < 2 ? zface : tface;
        lv->sap_ftab[L * 2]     = malloc(sizeof(long) * fs);
        lv->sap_ftab[L * 2 + 1] = malloc(sizeof(long) * fs);
        for (long i = 0; i < fs; i ++) {
          // z faces are (t,y,x) lex, t faces (z,y,x) -- both match the ghost
          // fidx rule the gathers use and halo_coarse's pack order
          long x = i % lv->nx, y = (i / lv->nx) % lv->ny;
          long p = i / ((long) lv->nx * lv->ny);
          long zc = L < 2 ? (L == 1 ? lv->nz - 1 : 0) : p;
          long tc = L < 2 ? p : (L == 3 ? lv->nt - 1 : 0);
          int col = (int) ((x / bl[0] + y / bl[1] + zc / bl[2] + tc / bl[3])
                    & 1) ^ rank_blk_color;
          lv->sap_ftab[L * 2 + col][lv->sap_fcnt[L * 2 + col] ++] = i;
        }
      }
    }
#endif

    // Per-block global lexicographic site lists: sap_gsite[blk*bsites + lb].
    lv->sap_gsite = malloc(sizeof(long) * nblocks * bsites);
    #pragma omp parallel for schedule(static)
    for (long blk = 0; blk < nblocks; blk ++) {
      long bb = blk;
      int bx = bb % nb[0]; bb /= nb[0];
      int by = bb % nb[1]; bb /= nb[1];
      int bz = bb % nb[2]; bb /= nb[2];
      int bt = bb;
      int ox = bx * bl[0], oy = by * bl[1], oz = bz * bl[2], ot = bt * bl[3];
      long * gsite = lv->sap_gsite + blk * bsites;
      for (int lt = 0; lt < bl[3]; lt ++)
        for (int lz = 0; lz < bl[2]; lz ++)
          for (int ly = 0; ly < bl[1]; ly ++)
            for (int lx = 0; lx < bl[0]; lx ++) {
              long lb = (((long) lt * bl[2] + lz) * bl[1] + ly) * bl[0] + lx;
              gsite[lb] = ((((long) (ot + lt) * lv->nz + (oz + lz)) * lv->ny
                            + (oy + ly)) * lv->nx + (ox + lx));
         }
    }
#if COARSE_SAP_DENSE_INV
    // Stage-major (intra-block-parallel) path: chosen when a color has fewer
    // SAP blocks than threads, so the block-major sweep would strand most
    // threads at every barrier.  Then the eo-Schur block solve runs
    // stage-by-stage over (block x site), which needs the per-block
    // intermediates to persist -- allocate them here (sized to the larger
    // color).  Multi-rank levels take this path too -- deep levels of a
    // decomposed run are where the starvation bites hardest (few
    // blocks/rank, every rank's threads idle) -- so the stage gather and
    // accumulate carry the same cross-rank ghost read / dpsi_field snapshot
    // as the block-major loop; the halo exchange itself sits outside the
    // branch and is shared.  Decided before the inverse build below:
    // the storage format depends on it.
    lv->sap_max_ncb = lv->sap_ncblk[0] > lv->sap_ncblk[1]
                    ? lv->sap_ncblk[0] : lv->sap_ncblk[1];
    lv->sap_stage_major = (lv->sap_ncblk[0] < nthr || lv->sap_ncblk[1] < nthr);
    lv->sap_sr = lv->sap_sy = lv->sap_sse = lv->sap_seb2 = NULL;
    if (lv->sap_stage_major) {
      long nh = bsites / 2;
      lv->sap_sr   = malloc(sizeof(REAL) * lv->sap_max_ncb * blk_reals);
      lv->sap_sy   = malloc(sizeof(REAL) * lv->sap_max_ncb * blk_reals);
      lv->sap_sse  = malloc(sizeof(REAL) * lv->sap_max_ncb * nh * S);
      lv->sap_seb2 = malloc(sizeof(REAL) * lv->sap_max_ncb * nh * S);
    }
    // Packed-triangle Hermitian inverse storage (see csap_herm_apply):
    // block-major levels only -- the stage-major See_inv apply is row-split
    // across threads, which the blocked-HEMM mirror passes cannot honor.
    // ncw % CSAP_TS is the tile edge (holds at every real width 64/96/128;
    // guard anyway).
    lv->sap_herm_tri = !lv->sap_stage_major && ncw % CSAP_TS == 0;
    // eo-Schur exact block solve: assemble each block's local (Dirichlet)
    // operator from the D_c couplings, split it into even/odd (nn hopping
    // flips parity, so Mee/Moo are block-diagonal and Meo/Moe are the
    // hopping), then precompute the block-diagonal odd inverse Moo^-1 and the
    // dense even-Schur inverse (Mee - Meo Moo^-1 Moe)^-1.  fp64 throughout;
    // stored down-converted to REAL.
    {
      const long dimc = (long) bsites * ncw;      // full block order
      const long nh   = bsites / 2;               // even (== odd) site count
      const long ns   = nh * ncw;                 // even/odd sub-order
      const long Nc2  = (long) ncw * ncw;         // one site block
      // per-matrix REALs, format-dependent (packed triangle ~ half the
      // full size)
      const long see_sz = lv->sap_herm_tri ? csap_tri_reals(ns)  : ns * ns * 2;
      const long soo_sz = lv->sap_herm_tri ? csap_tri_reals(ncw) : Nc2 * 2;
      lv->sap_See_inv = malloc(sizeof(REAL) * nblocks * see_sz);
      lv->sap_Soo_inv = malloc(sizeof(REAL) * nblocks * nh * soo_sz);
      lv->sap_eoperm  = malloc(sizeof(int) * nblocks * bsites);
      // Folded odd<-even couplings (see sap_Moe_fold).  Every in-block edge
      // yields exactly one odd->even directed coupling whichever endpoint is
      // odd, so half the in-block coupling count sizes every block alike;
      // the connectivity is block-independent (sap_lnbr is keyed on the local
      // site alone).
      long moe_ncpl = 0;
      for (long lb = 0; lb < bsites; lb ++)
        for (int dir = 0; dir < 8; dir ++)
          if (lv->sap_lnbr[lb * 8 + dir] >= 0) moe_ncpl ++;
      moe_ncpl /= 2;
      lv->sap_Moe_fold = malloc(sizeof(REAL) * nblocks * moe_ncpl * matsz);
      lv->sap_moe_off  = malloc(sizeof(long) * nblocks * nh);
      // Which storage path this level took, once per depth.  Derived, not set:
      // stage-major levels cannot use the packed form, so a level can land on
      // full-matrix without anything having asked for it.
      static int reported[NUM_LEVELS];
      if (lv->depth >= 0 && lv->depth < NUM_LEVELS && !reported[lv->depth]) {
        reported[lv->depth] = 1;
        print0("coarse_sap[depth %d]: dense inverses %s\n", lv->depth,
               lv->sap_herm_tri ? "packed Hermitian-triangle" : "full-matrix");
      }
      // static, not dynamic: this loop first-touches sap_See_inv/sap_Soo_inv,
      // and the solve reads them under a static block partition -- matching
      // schedules fault each solve-thread's span as one contiguous ascending
      // run (placement-correct on any NUMA layout, and the friendliest
      // pattern for THP formation).  Per-block cost is uniform, so dynamic
      // bought no balance here anyway.
      #pragma omp parallel for schedule(static)
      for (long blk = 0; blk < nblocks; blk ++) {
        const long * gs = lv->sap_gsite + blk * bsites;
        double complex * M = calloc((size_t) dimc * dimc,
                                    sizeof(double complex));
        for (long lb = 0; lb < bsites; lb ++) {      // assemble M (Dirichlet)
          long p = gs[lb];
          const REAL * Sb = self + p * matsz;
          for (int a = 0; a < ncw; a ++)
            for (int b = 0; b < ncw; b ++)
              M[(size_t)(lb * ncw + a) * dimc + (lb * ncw + b)] =
                  Sb[(a * ncw + b) * 2] + I * Sb[(a * ncw + b) * 2 + 1];
          for (int dir = 0; dir < 8; dir ++) {
            int ln = lv->sap_lnbr[lb * 8 + dir];
            if (ln < 0) continue;                    // off-block hop dropped
            const REAL * C = nbr + ((long) p * 8 + dir) * matsz;
            for (int a = 0; a < ncw; a ++)
              for (int b = 0; b < ncw; b ++)
                M[(size_t)(lb * ncw + a) * dimc + (ln * ncw + b)] =
                    C[(a * ncw + b) * 2] + I * C[(a * ncw + b) * 2 + 1];
          }
        }
        // even/odd partition by coarse-site parity: even locals then odd locals
        int * eop = lv->sap_eoperm + blk * bsites;
        long ne = 0, no = 0;
        for (long lb = 0; lb < bsites; lb ++) {
          long g = gs[lb], gg = g;
          int gx = gg % lv->nx; gg /= lv->nx;
          int gy = gg % lv->ny; gg /= lv->ny;
          int gz = gg % lv->nz; gg /= lv->nz; int gt = (int) gg;
          if (((gx + gy + gz + gt) & 1) == 0) eop[ne ++]      = (int) lb;
          else                                eop[nh + no ++] = (int) lb;
        }
        // The halves-layout above requires a parity-balanced block (ne == no ==
        // bsites/2), which any block with at least one even extent satisfies at
        // every position.  An all-odd-extent block (e.g. 3^4) is unbalanced and
        // would silently scramble the permutation -- refuse it.
        if (ne != nh || no != nh) {
          comm_die("coarse_sap[depth %d]: parity-unbalanced SAP block "
                   "(%ld even / %ld odd of %ld sites) -- block extents must "
                   "include an even dimension", lv->depth, ne, no, (long) bsites);
        }
        // dense even/odd sub-blocks (ns x ns); t1 will hold Meo . Moo^-1, built
        // block-diagonally in the odd loop below (Moo^-1 is block-diagonal)
        double complex * Mee  = malloc(sizeof(double complex)
                                       * (size_t) ns * ns);
        double complex * Meo  = malloc(sizeof(double complex)
                                       * (size_t) ns * ns);
        double complex * Moe  = malloc(sizeof(double complex)
                                       * (size_t) ns * ns);
        double complex * t1   = malloc(sizeof(double complex)
                                       * (size_t) ns * ns);
        for (long i = 0; i < nh; i ++)
          for (long j = 0; j < nh; j ++) {
            long ei = eop[i], ej = eop[j], oi = eop[nh + i], oj = eop[nh + j];
            for (int a = 0; a < ncw; a ++)
              for (int b = 0; b < ncw; b ++) {
                Mee[(i*ncw+a)*ns + (j*ncw+b)]
                    = M[(size_t)(ei*ncw+a)*dimc + (ej*ncw+b)];
                Meo[(i*ncw+a)*ns + (j*ncw+b)]
                    = M[(size_t)(ei*ncw+a)*dimc + (oj*ncw+b)];
                Moe[(i*ncw+a)*ns + (j*ncw+b)]
                    = M[(size_t)(oi*ncw+a)*dimc + (ej*ncw+b)];
              }
          }
        // invert each odd site's self-block -> Soo^-1 storage + Moo^-1
        // block-diag
        REAL * soo = lv->sap_Soo_inv + blk * nh * soo_sz;
        double complex * Mjj  = malloc(sizeof(double complex) * Nc2);
        double complex * Mjji = malloc(sizeof(double complex) * Nc2);
        double complex * Mjc  = malloc(sizeof(double complex) * Nc2);
        double complex * Mjf  = malloc(sizeof(double complex) * Nc2);
        long fcpl = blk * moe_ncpl;           // running folded-coupling index
        for (long j = 0; j < nh; j ++) {
          long oj = eop[nh + j];
          for (int a = 0; a < ncw; a ++)
            for (int b = 0; b < ncw; b ++)
              Mjj[a*ncw+b] = M[(size_t)(oj*ncw+a)*dimc + (oj*ncw+b)];
          if (csap_cinv(Mjj, ncw, ncw, Mjji)) {
            comm_die("coarse_sap: block %ld odd self-block singular", blk);
          }
          if (lv->sap_herm_tri)
            csap_pack_tri(Mjji, ncw, ncw, soo + j * soo_sz);
          else for (long k = 0; k < Nc2; k ++) {
            soo[(j*Nc2 + k)*2]     = (REAL) creal(Mjji[k]);
            soo[(j*Nc2 + k)*2 + 1] = (REAL) cimag(Mjji[k]);
          }
          // t1[:, j-block] = Meo[:, j-block] . Mjji : the j-th column-block
          // of Meo . Moo^-1.  Moo^-1 is block-diagonal (odd sites in a 2^4
          // block are never neighbors), so this replaces a full ns^3 gemm
          // through 7/8 zeros with nh small (ns x nvar)(nvar x nvar) gemms.
          dense_gemm(ns, ncw, ncw, Meo + (size_t) j * ncw, ns, Mjji, ncw,
                     t1 + (size_t) j * ncw, ns);
          // Fold this odd site's inverse into each of its in-block couplings,
          // in the direction order the block solve's closing stage walks.
          // Taken from the fp64 Dirichlet operator, so the product rounds to
          // REAL once rather than twice as the two separate applies did.
          lv->sap_moe_off[blk * nh + j] = fcpl;
          for (int dir = 0; dir < 8; dir ++) {
            int ln = lv->sap_lnbr[oj * 8 + dir];
            if (ln < 0) continue;
            for (int a = 0; a < ncw; a ++)
              for (int b = 0; b < ncw; b ++)
                Mjc[a * ncw + b]
                  = M[(size_t)(oj * ncw + a) * dimc + (ln * ncw + b)];
            dense_gemm(ncw, ncw, ncw, Mjji, ncw, Mjc, ncw, Mjf, ncw);
            REAL * fdst = lv->sap_Moe_fold + fcpl * matsz;
            for (long k = 0; k < Nc2; k ++) {
              fdst[k * 2]     = (REAL) creal(Mjf[k]);
              fdst[k * 2 + 1] = (REAL) cimag(Mjf[k]);
            }
            fcpl ++;
          }
        }
        // S = Mee - (Meo Moo^-1) Moe = Mee - t1 Moe ; store
        // See_inv = S^-1.  t1 was built block-diagonally above; only this
        // second product is a dense ns^3 gemm.
        double complex * S  = malloc(sizeof(double complex) * (size_t) ns * ns);
        double complex * Si = malloc(sizeof(double complex) * (size_t) ns * ns);
        dense_gemm(ns, ns, ns, t1, ns, Moe, ns, S, ns);
        for (long k = 0; k < ns * ns; k ++) S[k] = Mee[k] - S[k];
        if (csap_cinv(S, (int) ns, ncw, Si)) {
          comm_die("coarse_sap: block %ld Schur complement singular", blk);
        }
        REAL * sdst = lv->sap_See_inv + blk * see_sz;
        if (lv->sap_herm_tri)
          csap_pack_tri(Si, (int) ns, ncw, sdst);
        else for (long k = 0; k < ns * ns; k ++) {
          sdst[k*2]     = (REAL) creal(Si[k]);
          sdst[k*2 + 1] = (REAL) cimag(Si[k]);
        }
        free(M); free(Mee); free(Meo); free(Moe); free(t1);
        free(Mjj); free(Mjji); free(Mjc); free(Mjf); free(S); free(Si);
      }
    }
#endif
    lv->sap_nthr = nthr;
  }
#if COARSE_SAP_PROFILE
  csap_cache_s += omp_get_wtime() - _kv;
#endif
  // Warm-only entry (phi == NULL): the See_inv cache is now built; return
  // before any smoothing.  Setup calls this once per smoothed level so the
  // one-time block inversion is charged to setup (amortized across solves)
  // instead of leaking into the first solve's coarse-SAP timer.
  if (!phi) return lv->nu;
  REAL ** rbuf  = lv->sap_rbuf;
  REAL ** Drbuf = lv->sap_Drbuf;
  REAL ** dbuf  = lv->sap_dbuf;
  const int  * lnbr      = lv->sap_lnbr;
  const int  * xnbr      = lv->sap_xnbr;
  const long * gsite_all = lv->sap_gsite;

  // The initial guess is zeroed lazily (smoother contract: psi_inc need not
  // be pre-zeroed).  The incremental path never reads phi during the sweeps
  // -- it is write-only there -- so its first touch of each site (sweep 0)
  // overwrites rather than accumulates, folding the zero into a write that
  // happens anyway.  That drops a full extra pass over the whole coarse field
  // (a serial memset + the sweep-0 read-back), which was the dominant
  // "serial" term and pure wasted traffic on a bandwidth-bound machine.
  // Only the full-recompute (!incr) path reads eta - D phi, so it still
  // needs a real pre-zero; that memset is below, once incr is known.

  // Invariant the sweep-0 shortcut and the incremental residual both rest on:
  // "cross a block face => land in an opposite-color block".  With a single
  // block along some dim a block wraps onto itself (its own dpsi feeds back
  // through the wrap couplings), so both must be disabled there.  The test
  // is on the global block count nb*P: a single local block (nb == 1) along
  // a decomposed dim still has its face neighbor on another rank, not itself,
  // so it keeps the incremental path (via the cross-rank dpsi exchange
  // below); only a genuinely undecomposed single block self-wraps.
  const int gnb[4] = { nb[0] * qc_grid.P[0], nb[1] * qc_grid.P[1],
                       nb[2] * qc_grid.P[2], nb[3] * qc_grid.P[3] };
  // An odd (> 1) global block count along any dim breaks the invariant
  // itself: an odd cycle cannot be two-colored, so some face-adjacent pair
  // shares a color and updates in the same phase.  Both residual paths are
  // then unsound -- the incremental bookkeeping drops the same-phase
  // neighbor's correction, and its dpsi/phi is read while unwritten
  // (uninitialized on the first phase) or mid-update (schedule-dependent).
  // A count of 1 is the supported self-wrap fallback below; anything else
  // must be even.
  for (int d = 0; d < 4; d ++)
    if (gnb[d] > 1 && (gnb[d] & 1)) {
      comm_die("coarse_sap[depth %d]: odd SAP block count (%dx%dx%dx%d blocks) "
               "-- an odd cycle cannot be red-black colored; use a blocking with "
               "an even (or single) block count per dim",
               lv->depth, gnb[0], gnb[1], gnb[2], gnb[3]);
    }
#if FERM_GHOSTS
  // The test above asks whether the block torus is two-colorable at all; this
  // one asks whether a rank labels its share correctly.  Colors are local
  // block coords xor rank_blk_color, which comes from corigin, so a level
  // built without its rank origin colors as if it were rank 0 -- a
  // face-adjacent pair then shares a color across the rank boundary,
  // degrading only the preconditioner (symptom: more iterations, never a
  // wrong answer).  corigin[d] = coord_d * (local extent >= 1), so a zero
  // corigin under a nonzero fine origin means the field was never set.
  // fprintf, not print0: rank 0 has origin 0 and never trips this.
  if ((DECOMP_Z && qc_grid.origin[2] && !lv->corigin[2]) ||
      (DECOMP_T && qc_grid.origin[3] && !lv->corigin[3])) {
    comm_die("coarse_sap[depth %d]: corigin (z,t) = (%d,%d) but fine origin is "
             "(%d,%d) -- level built without its rank origin, so its red-black "
             "coloring is not the global checkerboard.  Set corigin[d] = "
             "fine_origin[d] / accumulated block factor.",
             lv->depth, lv->corigin[2], lv->corigin[3],
             qc_grid.origin[2], qc_grid.origin[3]);
  }
#endif
  const int sweep0_ok = gnb[0] >= 2 && gnb[1] >= 2 && gnb[2] >= 2 && gnb[3] >= 2;
  // Incremental (gather-form) residual: sap_R[blk] holds the block's exact
  // residual as of its previous visit (the MR solve's final r); between two
  // visits exactly one opposite-color phase runs, so the only change is the
  // face-neighbors' dpsi entering through the cross-block couplings.
  // Self-wrapping block grids fall back to the full recompute.
  const int incr = sweep0_ok;
#if COARSE_SAP_DENSE_INV
  // Intra-block-parallel sweep for levels that would otherwise starve threads
  // (fewer blocks/color than threads).  Requires the incremental gather (the
  // stage path has no full-recompute branch); a degenerate !incr grid falls
  // back to block-major.  Storage stays consistent either way:
  // sap_stage_major forces sap_herm_tri = 0 at the cache build, so this path
  // always sees full-format inverses, and the !incr block-major fallback of
  // such a level reads them through tri == 0 too.
  const int stage_major = lv->sap_stage_major && incr;
#endif
  // Full-recompute (!incr, self-wrapping block grids) reads phi during the
  // residual, so it needs the field pre-zeroed; the incremental path folds
  // the zero into its sweep-0 first-touch write instead.  !incr is a
  // degenerate small-lattice fallback, so a plain serial memset is fine here.
  if (!incr) memset(phi, 0, sizeof(REAL) * lv->field_len * num_rhs);
#if FERM_GHOSTS
  // A decomposed level reads cross-rank face-neighbor dpsi from the ghost,
  // which only the incremental path maintains; the full-recompute fallback
  // would read phi at periodically-wrapped (wrong) cross-rank sites.  So a
  // decomposed level must be in the incremental regime (>=2 global SAP blocks
  // per dim); anything else is an unsupported geometry.
  if (!incr && (DECOMP_Z || DECOMP_T)) {
    comm_die("coarse_sap: decomposed level needs >=2 global SAP blocks/dim "
             "(gnb=%d,%d,%d,%d)", gnb[0], gnb[1], gnb[2], gnb[3]);
  }
  const int  cnx = lv->nx, cny = lv->ny, cnz = lv->nz;
  const coarse_ghost_faces gfaces = coarse_ghost_faces_of(lv);
#endif

  for (int sweep = 0; sweep < lv->nu; sweep ++) {
    for (int color = 0; color < 2; color ++) {
      const long * cblk = lv->sap_cblk[color];
      const long   ncb  = lv->sap_ncblk[color];
#if FERM_GHOSTS
      // Cross-rank dpsi exchange: the opposite-color face neighbors updated in
      // the previous phase are this phase's only changed inputs.  dpsi_field
      // holds the latest per-block dpsi site-major (written in the accumulate
      // loop); ship its z/t boundary layers so each rank gets its neighbors'
      // boundary dpsi.  Skip before the very first phase (nothing solved yet --
      // same condition the boundary reads use).  Reads consume only the
      // just-solved color's entries, so the exchange ships exactly that half
      // of each face (csap_halo_sap) -- bit-identical reads, half the payload.
      if (incr && !(sweep == 0 && color == 0)) {
  #if COARSE_SAP_PROFILE
        double _hv = omp_get_wtime();
  #endif
        csap_halo_sap(lv, S, 1 - color);
  #if COARSE_SAP_PROFILE
        csap_halo_s += omp_get_wtime() - _hv;
  #endif
      }
#endif
#if COARSE_SAP_PROFILE
      double _pv = omp_get_wtime();
      for (int q = 0; q < CSAP_MAXTHR; q ++) {   // region-entry per-thread work
        double s = 0;
        for (int k = 0; k < CSAP_NSTAGE; k ++) s += csap_ps[q][k];
        csap_snap[q] = s;
      }
#endif
#if COARSE_SAP_DENSE_INV
      if (stage_major) {
        // Intra-block-parallel eo-Schur block solve.  Each stage of the block
        // solve is a parallel loop over (color-block bi x site), so every
        // thread has work even when this level has fewer blocks than threads
        // (where the block-major else branch would strand most of them at
        // each color barrier).  The per-block intermediates persist in
        // sap_sr/sy/sse/seb2 (indexed by bi) across the stage barriers;
        // the solution is written straight into sap_dpsi[blk] (the
        // incremental state), which the accumulate stage folds into phi.
        // Every per-site / per-row op and its reduction order matches the
        // block-major branch below, so this is bit-identical -- see there
        // for the eo-Schur algebra.
        const long nh    = bsites / 2;
        const long Nc2   = (long) ncw * ncw;
        // nothing solved yet -> r = eta
        const int  first = (sweep == 0 && color == 0);
        REAL * sr = lv->sap_sr, * sy = lv->sap_sy, * sse = lv->sap_sse,
             * seb2 = lv->sap_seb2;
  #if COARSE_SAP_PROFILE
    #define SM_ACC(k) (csap_ps[tid][k] += omp_get_wtime() - _t0)
    #define SM_RESET  (_t0 = omp_get_wtime())
  #else
    #define SM_ACC(k) ((void) 0)
    #define SM_RESET  ((void) 0)
  #endif
        #pragma omp parallel
        {
          int tid = omp_get_thread_num();
  #if COARSE_SAP_PROFILE
          double _t0 = omp_get_wtime();
  #endif
          // Stage 0: block residual r (gather).  base = eta on sweep 0, else
          // 0; the cross-block face couplings subtract the neighbors' last
          // dpsi.
          #pragma omp for schedule(static) collapse(2) nowait
          for (long bi = 0; bi < ncb; bi ++)
            for (long lb = 0; lb < bsites; lb ++) {
              long blk = cblk[bi];
              const long * gsite = gsite_all + blk * bsites;
              const long * nbb   = lv->sap_nbrblk + blk * 8;
              long g = gsite[lb];
              const REAL * base = (sweep == 0) ? eta + g * S : NULL;
              REAL * acc = sr + bi * blk_reals + lb * S;
              REAL * cbases[10], * ibases[9];
              int nc = 0;
  #if FERM_GHOSTS
              // Decode this site's coarse coords once (for the cross-rank
              // ghost face index); only needed when a face neighbor is on
              // another rank.
              long gg = g;
              int gx = gg % cnx; gg /= cnx;
              int gy = gg % cny; gg /= cny;
              int gz = gg % cnz; gg /= cnz;
              int gt = (int) gg;
              const int * nrk = lv->sap_nbr_rank + blk * 8;
  #endif
              if (!first)
                for (int dir = 0; dir < 8; dir ++)
                  if (lnbr[lb * 8 + dir] < 0) {        // crosses a block face
                    cbases[nc] = nbr + ((long) g * 8 + dir) * matsz;
  #if FERM_GHOSTS
                    if (nrk[dir])       // neighbor block is on another rank
                      ibases[nc] = lv->sap_dpsi_ghost
                                   + coarse_ghost_fidx(gfaces, dir, gx, gy,
                                                       gz, gt) * S;
                    else
  #endif
                    ibases[nc] = lv->sap_dpsi + nbb[dir] * blk_reals
                                 + (long) xnbr[lb * 8 + dir] * S;
                    nc ++;
                  }
              if (nc == 0) {
                if (base) memcpy(acc, base, sizeof(REAL) * S);
                else      memset(acc, 0, sizeof(REAL) * S);
              } else {
                cbases[nc] = cbases[0];
                coarse_dslash_site(acc, cbases, ibases, nc, ncw);
                if (base)
                  for (long k = 0; k < S; k ++) acc[k] = base[k] - acc[k];
                else
                  for (long k = 0; k < S; k ++) acc[k] = -acc[k];
              }
            }
          SM_ACC(0);
          #pragma omp barrier
          SM_RESET;
          // Stage 1: y_o = Moo^-1 r_o  (block-diagonal, per odd site)
          #pragma omp for schedule(static) collapse(2) nowait
          for (long bi = 0; bi < ncb; bi ++)
            for (long j = 0; j < nh; j ++) {
              long blk = cblk[bi];
              const int  * eop = lv->sap_eoperm + blk * bsites;
              const REAL * Soo = lv->sap_Soo_inv + blk * nh * Nc2 * 2;
              long o = eop[nh + j];
              csap_dense_apply(sy + bi * blk_reals + o * S, Soo + j * Nc2 * 2,
                               sr + bi * blk_reals + o * S, 1, ncw);
            }
          SM_ACC(1);
          #pragma omp barrier
          SM_RESET;
          // Stage 2: s_e = r_e - Meo y_o  (even <- odd hopping) into sse
          #pragma omp for schedule(static) collapse(2) nowait
          for (long bi = 0; bi < ncb; bi ++)
            for (long i = 0; i < nh; i ++) {
              long blk = cblk[bi];
              const long * gsite = gsite_all + blk * bsites;
              const int  * eop   = lv->sap_eoperm + blk * bsites;
              REAL * y = sy + bi * blk_reals;
              long e = eop[i], g = gsite[e];
              REAL * cb[9], * ib[9]; int nc = 0;
              for (int dir = 0; dir < 8; dir ++) {
                int ln = lnbr[e * 8 + dir];
                if (ln >= 0) { cb[nc] = nbr + ((long) g * 8 + dir) * matsz;
                               ib[nc] = y + ln * S; nc ++; }
              }
              REAL * he = Drbuf[tid];           // per-thread one-site scratch
              if (nc) {
                cb[nc] = cb[0];
                coarse_dslash_site(he, cb, ib, nc, ncw);
              } else 
                memset(he, 0, sizeof(REAL) * S);
              REAL * ebi = sse + bi * nh * S + i * S;
              REAL * ri  = sr  + bi * blk_reals + e * S;
              for (long k = 0; k < S; k ++) ebi[k] = ri[k] - he[k];
            }
          SM_ACC(2);
          #pragma omp barrier
          SM_RESET;
          // Stage 3: x_e = See_inv . s_e, split by output even-site i (rows
          // [i*ncw,(i+1)*ncw)); bit-identical to the whole-block apply for
          // those rows.
          #pragma omp for schedule(static) collapse(2) nowait
          for (long bi = 0; bi < ncb; bi ++)
            for (long i = 0; i < nh; i ++) {
              long blk = cblk[bi];
              const REAL * See_inv = lv->sap_See_inv + blk * (nh * ncw)
                                    * (nh * ncw) * 2;
              csap_dense_apply_rows(seb2 + bi * nh * S, See_inv,
                                    sse + bi * nh * S, nh, ncw, i * ncw,
                                    (i + 1) * ncw);
            }
          SM_ACC(3);
          #pragma omp barrier
          SM_RESET;
          // Stage 3b: scatter x_e into sap_dpsi at the block-local even sites
          // eop[i] (timed with See_inv); the odd sites are filled by stage 4.
          #pragma omp for schedule(static) collapse(2) nowait
          for (long bi = 0; bi < ncb; bi ++)
            for (long i = 0; i < nh; i ++) {
              long blk = cblk[bi];
              const int * eop = lv->sap_eoperm + blk * bsites;
              memcpy(lv->sap_dpsi + blk * blk_reals + eop[i] * S,
                     seb2 + bi * nh * S + i * S, sizeof(REAL) * S);
            }
          SM_ACC(3);
          #pragma omp barrier
          SM_RESET;
          // Stage 4: x_o = y_o - Moe'' x_e  (odd <- even hopping; Moo^-1
          // folded in)
          #pragma omp for schedule(static) collapse(2) nowait
          for (long bi = 0; bi < ncb; bi ++)
            for (long j = 0; j < nh; j ++) {
              long blk = cblk[bi];
              const int  * eop   = lv->sap_eoperm + blk * bsites;
              REAL * dpsi = lv->sap_dpsi + blk * blk_reals;
              REAL * y    = sy + bi * blk_reals;
              long o = eop[nh + j];
              REAL * fb = lv->sap_Moe_fold
                        + lv->sap_moe_off[blk * nh + j] * matsz;
              REAL * cb[9], * ib[9]; int nc = 0;
              for (int dir = 0; dir < 8; dir ++) {
                int ln = lnbr[o * 8 + dir];
                if (ln >= 0) {
                  cb[nc] = fb + (long) nc * matsz;
                  ib[nc] = dpsi + ln * S;   // x_e at even nbrs
                  nc ++;
                }
              }
              REAL * ho = Drbuf[tid];
              if (nc) {
                cb[nc] = cb[0];
                coarse_dslash_site(ho, cb, ib, nc, ncw);
              }
              else
                memset(ho, 0, sizeof(REAL) * S);
              REAL * xo = dpsi + o * S, * yo = y + o * S;
              for (long k = 0; k < S; k ++) xo[k] = yo[k] - ho[k];
            }
          SM_ACC(4);
          #pragma omp barrier
          SM_RESET;
          // Stage 5: accumulate the block correction into phi (overwrite on
          // sweep 0's first touch, else add).  incr is required for
          // stage_major, so ow == sweep 0.
          #pragma omp for schedule(static) collapse(2) nowait
          for (long bi = 0; bi < ncb; bi ++)
            for (long lb = 0; lb < bsites; lb ++) {
              long blk = cblk[bi];
              const long * gsite = gsite_all + blk * bsites;
              long g = gsite[lb];
              REAL * pg = phi + g * S,
                   * dl = lv->sap_dpsi + blk * blk_reals + lb * S;
              if (sweep == 0) for (long k = 0; k < S; k ++) pg[k]  = dl[k];
              else            for (long k = 0; k < S; k ++) pg[k] += dl[k];
  #if FERM_GHOSTS
              // site-major snapshot of this block's correction for the next
              // phase's cross-rank dpsi exchange (incr is guaranteed here:
              // stage_major requires it, so no gate on the copy)
              memcpy(lv->sap_dpsi_field + g * S, dl, sizeof(REAL) * S);
  #endif
            }
          SM_ACC(5);
        }
  #undef SM_ACC
  #undef SM_RESET
  #if COARSE_SAP_PROFILE
        csap_nbs[0] += ncb;                        // blocks solved this phase
  #endif
      } else
#endif // COARSE_SAP_DENSE_INV
      {
        // Every block in a color phase does identical work -- same block
        // extents, same eo-Schur dense apply (fixed nh*ncw^2), same
        // face-crossing pattern (only which neighbor block it reads differs)
        // -- so the per-block cost is uniform and schedule(static) is both
        // perfectly balanced and free of the per-iteration atomic dispatch
        // that dynamic pays.  Static also pins each thread to a fixed block
        // subset across sweeps/calls, so a block's data keeps the same NUMA
        // home.
        #pragma omp parallel for schedule(static)
        for (long bi = 0; bi < ncb; bi ++) {
          long blk = cblk[bi];
          int tid = omp_get_thread_num();
          REAL * r = rbuf[tid], * Dr = Drbuf[tid], * dpsi = dbuf[tid];
          const long * gsite = gsite_all + blk * bsites;   // precomputed
#if COARSE_SAP_PROFILE
          double _t = omp_get_wtime();
#endif
          // Block residual r = (eta - D_l phi)|_block -- the DRAM-streamed
          // phase.  Incremental residual (gather form): base = eta on the
          // first visit (sweep 0; on the very first phase nothing has been
          // solved yet, so r = eta exactly), sap_R on later visits; then
          // subtract the face neighbors' last dpsi through the cross-block
          // couplings (the lnbr < 0 dirs -- a fraction of the 9/site the
          // full recompute applies).  Sweep 0 is bit-identical to the
          // recompute (dpsi == phi there, and the dropped products are
          // exact +-0); from sweep 1 on the residual is accumulated across
          // visits instead of re-derived from phi, so it differs at rounding
          // level.
          if (incr) {
#if !COARSE_SAP_DENSE_INV
            const REAL * Rb  = lv->sap_R + blk * blk_reals;
#endif
            const long * nbb = lv->sap_nbrblk + blk * 8;
            for (long lb = 0; lb < bsites; lb ++) {
              long g = gsite[lb];
              // eo-Schur: the block's post-solve residual is exactly 0, so
              // the base for sweep>0 is a zero (NULL) -- no sap_R array read;
              // MR reads its stored residual.
#if COARSE_SAP_DENSE_INV
              const REAL * base = sweep == 0 ? eta + g * S : NULL;
#else
              const REAL * base = sweep == 0 ? eta + g * S : Rb + lb * S;
#endif
              REAL * acc = r + lb * S;
              REAL * cbases[10], * ibases[9];
              int nc = 0;
#if FERM_GHOSTS
              // Decode this site's coarse coords once (for the cross-rank
              // ghost face index); only needed when a face neighbor is on
              // another rank.
              long gg = g;
              int gx = gg % cnx; gg /= cnx;
              int gy = gg % cny; gg /= cny;
              int gz = gg % cnz; gg /= cnz;
              int gt = (int) gg;
              const int * nrk = lv->sap_nbr_rank + blk * 8;
#endif
              if (!(sweep == 0 && color == 0))
                for (int dir = 0; dir < 8; dir ++)
                  if (lnbr[lb * 8 + dir] < 0) {      // crosses a block face
                    cbases[nc] = nbr + ((long) g * 8 + dir) * matsz;
#if FERM_GHOSTS
                    if (nrk[dir])         // neighbor block is on another rank
                      ibases[nc] = lv->sap_dpsi_ghost
                                   + coarse_ghost_fidx(gfaces, dir, gx, gy,
                                                       gz, gt) * S;
                    else
#endif
                    ibases[nc] = lv->sap_dpsi + nbb[dir] * blk_reals
                                 + (long) xnbr[lb * 8 + dir] * S;
                    nc ++;
                  }
              if (nc == 0) {                         // nothing changed here
                if (base) memcpy(acc, base, sizeof(REAL) * S);
                else      memset(acc, 0, sizeof(REAL) * S);
                continue;
              }
              cbases[nc] = cbases[0];                // prefetch one-past slot
              coarse_dslash_site(acc, cbases, ibases, nc, ncw);
              if (base)
                for (long k = 0; k < S; k ++)
                  acc[k] = base[k] - acc[k];
              else
                for (long k = 0; k < S; k ++)
                  acc[k] = -acc[k];
            }
          } else {
            for (long lb = 0; lb < bsites; lb ++) {
              long g = gsite[lb];
              REAL * acc = r + lb * S;
              REAL * cbases[10], * ibases[9];
              cbases[0] = self + g * matsz;  ibases[0] = phi + g * S;
              for (int dir = 0; dir < 8; dir ++) {
                cbases[dir + 1] = nbr + ((long) g * 8 + dir) * matsz;
                ibases[dir + 1] = phi + coarse_neighbor_lvl(lv, g, dir) * S;
              }
              // acc = D_l phi (overwrites)
              coarse_dslash_site(acc, cbases, ibases, 9, ncw);
              for (long k = 0; k < S; k ++) acc[k] = eta[g * S + k] - acc[k];
            }
          }
          CSAP_LAP(0);   // gather: the incremental block-residual
                         // (cross-block) work

          // Block solve D_block dpsi = r: eo-Schur exact dense inverse, or
          // (in the #else branch) the MR iteration.  MR's dpsi-zeroing and
          // stencil gather live in that branch -- the eo path writes every
          // dpsi site and needs no gather, so it skips both.
#if COARSE_SAP_DENSE_INV
          // eo-Schur exact block solve (dpsi = M_block^-1 r via the even
          // Schur complement).  Then bring the block residual to its exact
          // post-solve value r <- r - D_block dpsi, which the incremental
          // path persists as sap_R.
          {
            REAL * y   = lv->sap_ybuf[tid];   // both parities; y_o filled below
            REAL * eb  = lv->sap_ebuf[tid];   // even sub-vector s_e (nh sites)
            REAL * eb2 = lv->sap_ebuf2[tid];  // even sub-vector x_e
            REAL * sb  = lv->sap_sbuf[tid];   // one-site scratch
            const int  * eop = lv->sap_eoperm + blk * bsites;
            const long   nh  = bsites / 2;
            const long   Nc2 = (long) ncw * ncw;
            // Triangle format (sap_herm_tri): the stored matrices are the
            // Hermitian H^-1 (A^-1 = H^-1 Gamma5c), so each dense apply
            // becomes csap_herm_apply on a signature-scaled input
            // xs = Gamma5c x -- negate the lower-chirality half of each
            // site's vars, i.e. the k >= S/2 half of a site vector.
            const int    tri = lv->sap_herm_tri;
            const long   soo_str = tri ? csap_tri_reals(ncw) : Nc2 * 2;
            const long   see_sz  = tri ? csap_tri_reals(nh * ncw)
                                       : (nh * ncw) * (nh * ncw) * 2;
            const REAL * See_inv = lv->sap_See_inv + blk * see_sz;
            const REAL * Soo = lv->sap_Soo_inv + blk * nh * soo_str;
            // 1. y_o = Moo^-1 r_o  (block-diagonal, per odd site; r_o is
            //    still live for stage 2's r_e read, so the tri path scales
            //    into the sbuf scratch)
            for (long j = 0; j < nh; j ++) {
              long o = eop[nh + j];
              if (tri) {
                const REAL * ro = r + o * S;
                for (long k = 0;     k < S / 2; k ++) sb[k] = ro[k];
                for (long k = S / 2; k < S;     k ++) sb[k] = -ro[k];
                csap_herm_apply(y + o * S, Soo + j * soo_str, sb, 1, ncw);
              } else {
                csap_dense_apply(y + o * S, Soo + j * soo_str, r + o * S, 1,
                                 ncw);
              }
            }
            CSAP_LAP(1);
            // 2. s_e = r_e - Meo y_o  (even <- odd hopping), gathered into eb
            for (long i = 0; i < nh; i ++) {
              long e = eop[i], g = gsite[e];
              REAL * cb[9], * ib[9]; int nc = 0;
              for (int dir = 0; dir < 8; dir ++) {
                int ln = lnbr[e * 8 + dir];
                if (ln >= 0) {
                  cb[nc] = nbr + ((long) g * 8 + dir) * matsz;
                  ib[nc] = y + ln * S;
                  nc ++;
                }
              }
              REAL * he = Dr + e * S;             // Dr reused as site scratch
              if (nc) {
                cb[nc] = cb[0];
                coarse_dslash_site(he, cb, ib, nc, ncw);
              } else {
                memset(he, 0, sizeof(REAL) * S);
              }
              if (tri) {
                // eb = Gamma5c (r_e - Meo y_o): signature folded into the write
                for (long k = 0;     k < S / 2; k ++)
                  eb[i * S + k] = r[e * S + k] - he[k];
                for (long k = S / 2; k < S;     k ++)
                  eb[i * S + k] = he[k] - r[e * S + k];
              } else {
                for (long k = 0; k < S; k ++)
                  eb[i * S + k] = r[e * S + k] - he[k];
              }
            }
            CSAP_LAP(2);
            // 3. x_e = See_inv . s_e  (hand-vectorized dense apply; See_inv
            //    is the stored Schur inverse S^-1, so this is an apply, not
            //    a solve).  The hand kernel consumes the native split re/im
            //    layout -- no pack needed.
            if (tri) csap_herm_apply(eb2, See_inv, eb, nh, ncw);
            else     csap_dense_apply(eb2, See_inv, eb, nh, ncw);
            for (long i = 0; i < nh; i ++)
              memcpy(dpsi + eop[i] * S, eb2 + i * S, sizeof(REAL) * S);
            CSAP_LAP(3);
            // 4. x_o = y_o - Moe'' x_e  (odd <- even hopping; Moo^-1 is
            //    folded into the couplings at setup, so no dense apply closes
            //    the solve)
            REAL       * mf = lv->sap_Moe_fold;
            const long * mo = lv->sap_moe_off + blk * nh;
            for (long j = 0; j < nh; j ++) {
              long o = eop[nh + j];
              REAL * fb = mf + mo[j] * matsz;
              REAL * cb[9], * ib[9]; int nc = 0;
              for (int dir = 0; dir < 8; dir ++) {
                int ln = lnbr[o * 8 + dir];
                if (ln >= 0) {
                  cb[nc] = fb + (long) nc * matsz;
                  ib[nc] = dpsi + ln * S;   // x_e at even nbrs
                  nc ++;
                }
              }
              REAL * ho = Dr + o * S;
              if (nc) {
                cb[nc] = cb[0];
                coarse_dslash_site(ho, cb, ib, nc, ncw);
              } else {
                memset(ho, 0, sizeof(REAL) * S);
              }
              REAL * xo = dpsi + o * S, * yo = y + o * S;
              for (long k = 0; k < S; k ++) xo[k] = yo[k] - ho[k];
            }
          }
          CSAP_LAP(4);
          // The exact block solve makes the post-solve residual analytically
          // zero (D_block dpsi = M_block M_block^-1 r = r), so sap_R is set
          // to 0 below rather than computed by an in-block matvec --
          // dropping the largest apply stage and, being exact, avoiding its
          // fp32 rounding noise.
  #if COARSE_SAP_PROFILE
          if (tid < CSAP_MAXTHR) csap_nbs[tid] ++;
  #endif
#else  // !COARSE_SAP_DENSE_INV
          memset(dpsi, 0, sizeof(REAL) * blk_reals);
          // The in-block MR stencil (coupling pointers + r offsets) is
          // iteration-invariant, so gather it once per visit rather than
          // re-walking lnbr.
          REAL * mr_cb[bsites][10], * mr_ib[bsites][9];
          int mr_nc[bsites];
          for (long lb = 0; lb < bsites; lb ++) {
            long g = gsite[lb];
            int nc = 0;
            mr_cb[lb][nc] = self + g * matsz;
            mr_ib[lb][nc] = r + lb * S;
            nc ++;
            for (int dir = 0; dir < 8; dir ++) {
              int ln = lnbr[lb * 8 + dir];
              if (ln >= 0) {
                mr_cb[lb][nc] = nbr + ((long) g * 8 + dir) * matsz;
                mr_ib[lb][nc] = r + ln * S;
                nc ++;
              }
            }
            mr_cb[lb][nc] = mr_cb[lb][0];        // prefetch one-past slot
            mr_nc[lb] = nc;
          }
          for (int it = 0; it < lv->blk_iter; it ++) {
            // Dr = D_block * r  (self + in-block neighbors only; r is 0
            // off-block); the asm kernel accumulates the cached stencil.
            for (long lb = 0; lb < bsites; lb ++)
              coarse_dslash_site(Dr + lb * S, mr_cb[lb], mr_ib[lb], mr_nc[lb],
                                 ncw);
            // alpha = <Dr, r> / <Dr, Dr>
            // (Hermitian dot conj(Dr).r, over the block)
            REAL cre, cim, dd;
            blk_dot3(Dr, r, blk_reals, &cre, &cim, &dd);
            if (dd == 0) break;                       // converged / breakdown
            REAL ar = cre / dd, ai = cim / dd;
            // dpsi += a*r; r -= a*Dr
            blk_caxpy_pair(dpsi, r, Dr, ar, ai, blk_reals); 
          }
#endif // COARSE_SAP_DENSE_INV

          // accumulate the block correction into the global solution
          if (incr) {
            // Persist the incremental-residual state.  MR stores its final r
            // (the block's exact post-solve residual); eo needs no sap_R at
            // all -- that residual is analytically zero, so its gather uses
            // a zero base directly.
#if !COARSE_SAP_DENSE_INV
            memcpy(lv->sap_R + blk * blk_reals, r, sizeof(REAL) * blk_reals);
#endif
            memcpy(lv->sap_dpsi + blk * blk_reals, dpsi, sizeof(REAL) * blk_reals);
          }
          // First touch (incr sweep 0) overwrites -- folds the lazy zero into
          // a write that happens anyway; later sweeps accumulate.  !incr was
          // pre-zeroed above.
          const int ow = incr && sweep == 0;
          for (long lb = 0; lb < bsites; lb ++) {
            long g = gsite[lb];
            REAL * pg = phi + g * S, * dl = dpsi + lb * S;
            if (ow) for (long k = 0; k < S; k ++) pg[k]  = dl[k];
            else    for (long k = 0; k < S; k ++) pg[k] += dl[k];
#if FERM_GHOSTS
            // site-major snapshot of this block's correction for the next
            // phase's cross-rank dpsi exchange (boundary layers are what
            // gets shipped).
            if (incr) memcpy(lv->sap_dpsi_field + g * S, dl, sizeof(REAL) * S);
#endif
          }
          // accumulate: sap_dpsi store + phi += dpsi (memory-bound)
          CSAP_LAP(5);
        } // blocks
      }   // block-major branch (see stage_major above)
#if COARSE_SAP_PROFILE
      {
        double _pdt = omp_get_wtime() - _pv;
        csap_par_s += _pdt;
        int dd = lv->depth < CSAP_MAXD ? lv->depth : CSAP_MAXD - 1;
        csap_par_lvl[dd] += _pdt; csap_reg_lvl[dd] ++;
        csap_ncb_lvl[dd][color] = (int) ncb;
        for (int q = 0; q < CSAP_MAXTHR; q ++) {
          double s = 0;
          for (int k = 0; k < CSAP_NSTAGE; k ++)
            s += csap_ps[q][k];
          csap_thr_lvl[dd][q] += s - csap_snap[q];
        }
      }
#endif
    }   // color
  }     // sweep

  // --- exit-residual export (for the K-cycle assembled matvec) -------------
  // r_exit = eta - D_l phi at smoother exit.  With the incremental-residual
  // state this is nearly free: blocks of the last phase's color (1) just
  // wrote their exact residual to sap_R (copy it out), and color-0 blocks are
  // exactly one phase stale -- refresh them with the final color-1 dpsi
  // through the cross-block couplings, the same gather as the residual phase.
  // Degenerate block grids (no incremental state) fall back to a full
  // eta - D phi recompute: correct, just not free.
#if COARSE_SAP_PROFILE
  double _ev = omp_get_wtime();
#endif
  if (r_exit && incr) {
    const long * cb1 = lv->sap_cblk[1];
    #pragma omp parallel for schedule(dynamic)
    for (long bi = 0; bi < lv->sap_ncblk[1]; bi ++) {
      long blk = cb1[bi];
      const long * gsite = gsite_all + blk * bsites;
#if COARSE_SAP_DENSE_INV
      for (long lb = 0; lb < bsites; lb ++)   // eo: exact solve => residual 0
        memset(r_exit + gsite[lb] * S, 0, sizeof(REAL) * S);
#else
      const REAL * Rb = lv->sap_R + blk * blk_reals;
      for (long lb = 0; lb < bsites; lb ++)
        memcpy(r_exit + gsite[lb] * S, Rb + lb * S, sizeof(REAL) * S);
#endif
    }
#if FERM_GHOSTS
    // refresh the cross-rank ghost with the final color-1 dpsi (it holds the
    // last-phase corrections, written to dpsi_field in the accumulate loop)
    // before the color-0 blocks refresh their residual through it -- only the
    // color-1 half of each face is read, so ship exactly that (parity 1).
    csap_halo_sap(lv, S, 1);
#endif
    const long * cb0 = lv->sap_cblk[0];
    #pragma omp parallel for schedule(dynamic)
    for (long bi = 0; bi < lv->sap_ncblk[0]; bi ++) {
      long blk = cb0[bi];
      const long * gsite = gsite_all + blk * bsites;
#if !COARSE_SAP_DENSE_INV
      const REAL * Rb  = lv->sap_R + blk * blk_reals;
#endif
      const long * nbb = lv->sap_nbrblk + blk * 8;
#if FERM_GHOSTS
      const int * nrk = lv->sap_nbr_rank + blk * 8;
#endif
      for (long lb = 0; lb < bsites; lb ++) {
        long g = gsite[lb];
        REAL * out = r_exit + g * S;
#if COARSE_SAP_DENSE_INV
        const REAL * base = NULL;                   // eo: sap_R == 0
#else
        const REAL * base = Rb + lb * S;
#endif
        REAL * cbases[10], * ibases[9];
        int nc = 0;
#if FERM_GHOSTS
        long gg = g;
        int gx = gg % cnx; gg /= cnx;
        int gy = gg % cny; gg /= cny;
        int gz = gg % cnz; gg /= cnz;
        int gt = (int) gg;
#endif
        for (int dir = 0; dir < 8; dir ++)
          if (lnbr[lb * 8 + dir] < 0) {               // crosses a block face
            cbases[nc] = nbr + ((long) g * 8 + dir) * matsz;
#if FERM_GHOSTS
            if (nrk[dir])         // neighbor block is on another rank
              ibases[nc] = lv->sap_dpsi_ghost
                         + coarse_ghost_fidx(gfaces, dir, gx, gy, gz, gt) * S;
            else
#endif
            ibases[nc] = lv->sap_dpsi + nbb[dir] * blk_reals
                       + (long) xnbr[lb * 8 + dir] * S;
            nc ++;
          }
        if (nc == 0) {
          if (base) memcpy(out, base, sizeof(REAL) * S);
          else      memset(out, 0, sizeof(REAL) * S);
          continue;
        }
        cbases[nc] = cbases[0];                      // prefetch one-past slot
        coarse_dslash_site(out, cbases, ibases, nc, ncw);
        if (base) for (long k = 0; k < S; k ++) out[k] = base[k] - out[k];
        else      for (long k = 0; k < S; k ++) out[k] = -out[k];
      }
    }
  } else if (r_exit) {
    // full recompute fallback (degenerate / non-incremental geometries only)
    #pragma omp parallel for schedule(dynamic)
    for (long blk = 0; blk < nblocks; blk ++) {
      const long * gsite = gsite_all + blk * bsites;
      for (long lb = 0; lb < bsites; lb ++) {
        long g = gsite[lb];
        REAL * out = r_exit + g * S;
        REAL * cbases[10], * ibases[9];
        cbases[0] = self + g * matsz;  ibases[0] = phi + g * S;
        for (int dir = 0; dir < 8; dir ++) {
          cbases[dir + 1] = nbr + ((long) g * 8 + dir) * matsz;
          ibases[dir + 1] = phi + coarse_neighbor_lvl(lv, g, dir) * S;
        }
        coarse_dslash_site(out, cbases, ibases, 9, ncw);
        for (long k = 0; k < S; k ++) out[k] = eta[g * S + k] - out[k];
      }
    }
  }
#if COARSE_SAP_PROFILE
  csap_export_s += omp_get_wtime() - _ev;
#endif

  // --- flop model (paired with vc_smooth_coarse time) ----------------------
  // Per block-solve (every block is solved once per sweep; nu sweeps): the
  // block residual r = (eta - D_l phi)|_block is a full 9-coupling stencil
  // over bsites, then blk_iter MR iters, each a block matvec (self + in-block
  // neighbors only) plus the alpha dots and two block caxpys.  One coupling
  // apply is ncw^2 complex MACs x num_rhs = CFLOP.
  long CFLOP = (long) ncw * ncw * num_rhs * 8;
  long interior = 0;                   // directed in-block neighbor couplings
  for (int d = 0; d < 4; d ++)
    interior += 2 * (long) (bl[d] - 1) * (bsites / bl[d]);
  // Count only the residual couplings actually applied: the sweep-0 shortcuts
  // skip the identically-zero terms (first color applies none, second color
  // only the cross-block couplings), and counting them would inflate the
  // rate.  cross = directed couplings leaving a block.
  double cross = 0;
  for (int d = 0; d < 4; d ++) cross += 2.0 * ((double) bsites / bl[d]);
  double resid_cpl;                       // total residual couplings per call
  if (incr)        // cross couplings only, every phase but the very first
    resid_cpl = ((double) lv->sap_ncblk[1]
               + (double) nblocks * (lv->nu - 1)) * cross;
  else
    resid_cpl = (double) nblocks * 9.0 * bsites * lv->nu;
  if (r_exit)      // exit-residual export (color-0 refresh / full fallback)
    resid_cpl += incr ? (double) lv->sap_ncblk[0] * cross
                      : (double) nblocks * 9.0 * bsites;
  double nbnu = (double) nblocks * lv->nu;
  // eta-sub + accumulate (a negligible overcount: the sweep-0 memcpy blocks 
  // do only the accumulate)
  double f_vec   = 2.0 * bsites * S * nbnu;
#if COARSE_SAP_DENSE_INV
  // eo-Schur solve per block: Moo^-1 forward (nh site-blocks) + e<->o hopping
  // (interior couplings) + dense even-Schur solve (nh^2 site-blocks, ns=nh*N
  // wide).  The post-solve residual matvec is dropped: the exact solve makes
  // it 0, and the closing Moo^-1 lives inside the odd<-even couplings
  // (sap_Moe_fold).
  long   nh_       = bsites / 2;
  #if COARSE_SAP_PROFILE
  // per-block-solve flops per profiled stage (couplings * CFLOP): gather (the
  // cross-block boundary matvec -- off-block couplings x neighbor dpsi, a
  // real apply not zero), Soo^-1 fwd, Meo hop (interior/2 directed), See_inv
  // apply (nh^2 site-blocks), Moe hop.  (Residual matvec dropped = 0; the
  // closing Soo^-1 is folded into the Moe couplings, so no site-block term
  // rides here.)  Sweep-only couplings (matches LAP(0)'s scope; the exit
  // export is un-lapped).
  csap_sf[0]
    = (incr ? ((double) lv->sap_ncblk[1] + (double) nblocks * (lv->nu - 1))
        * cross : (double) nblocks * 9.0 * bsites * lv->nu) / nbnu * CFLOP;
  csap_sf[1] = (double) nh_ * CFLOP;
  csap_sf[2] = (interior / 2.0) * CFLOP;
  csap_sf[3] = (double) nh_ * nh_ * CFLOP;
  csap_sf[4] = (interior / 2.0) * CFLOP;
  csap_sf[5] = 0;                    // accumulate: memory-bound (phi += dpsi)
  // Accumulate this call's per-stage total (per-block-solve flops * this
  // call's block solves) so a multi-level run sums each level at its own
  // width, not the last snapshot times the global block-solve count.
  for (int k = 0; k < CSAP_NSTAGE; k ++) csap_fl[k] += csap_sf[k] * nbnu;
  #endif
  double eo_cmac   = (double) (nh_ + interior + nh_ * nh_)
                   * ncw * ncw * num_rhs;
  csap_flops[lv->depth] += resid_cpl * CFLOP + eo_cmac * 8.0 * nbnu + f_vec;
#else
  double mr_cpl  = (double) lv->blk_iter * (bsites + interior) * nbnu;
  double f_dot   = (double) lv->blk_iter * 12.0               // cre/cim/dd
                 * ncw * num_rhs * bsites * nbnu;
  double f_caxpy = (double) lv->blk_iter * 16.0               // dpsi,r caxpys
                 * ncw * num_rhs * bsites * nbnu;
  csap_flops[lv->depth] += (resid_cpl + mr_cpl) * CFLOP
                         + f_dot + f_caxpy + f_vec;
#endif
#if COARSE_SAP_PROFILE
  csap_call_s += omp_get_wtime() - _cv;
#endif
  return lv->nu;
}

// Free a level's coarse_sap cache (scratch + geometry); no-op if unbuilt.
// Owned here because everything it frees is allocated here; called by
// coarse.c's mg_teardown and by the rebuild path below on a thread-count
// change.
void coarse_sap_cache_free(level_t * lv) {
  for (int q = 0; q < lv->sap_nthr; q ++) {
    free(lv->sap_rbuf[q]); free(lv->sap_Drbuf[q]); free(lv->sap_dbuf[q]);
  }
  free(lv->sap_rbuf); free(lv->sap_Drbuf); free(lv->sap_dbuf);
  free(lv->sap_lnbr); free(lv->sap_gsite);
  free(lv->sap_cblk[0]); free(lv->sap_cblk[1]);
  free(lv->sap_R); free(lv->sap_dpsi);
#if COARSE_SAP_DENSE_INV
  for (int q = 0; q < lv->sap_nthr; q ++) {
    free(lv->sap_ybuf[q]); free(lv->sap_ebuf[q]);
    free(lv->sap_ebuf2[q]); free(lv->sap_sbuf[q]);
  }
  free(lv->sap_ybuf); free(lv->sap_ebuf);
  free(lv->sap_ebuf2); free(lv->sap_sbuf);
  free(lv->sap_See_inv); free(lv->sap_Soo_inv); free(lv->sap_eoperm);
  free(lv->sap_Moe_fold); free(lv->sap_moe_off);
  free(lv->sap_sr); free(lv->sap_sy); free(lv->sap_sse); free(lv->sap_seb2);
  lv->sap_See_inv = lv->sap_Soo_inv = NULL; lv->sap_eoperm = NULL;
  lv->sap_Moe_fold = NULL; lv->sap_moe_off = NULL;
  lv->sap_ybuf = lv->sap_ebuf = lv->sap_ebuf2 = lv->sap_sbuf = NULL;
  lv->sap_sr = lv->sap_sy = lv->sap_sse = lv->sap_seb2 = NULL;
  lv->sap_stage_major = 0; lv->sap_max_ncb = 0; lv->sap_herm_tri = 0;
#endif
  free(lv->sap_nbrblk); free(lv->sap_xnbr);
  free(lv->sap_dpsi_field); free(lv->sap_dpsi_ghost); free(lv->sap_nbr_rank);
  for (int q = 0; q < 8; q ++) {
    free(lv->sap_ftab[q]); lv->sap_ftab[q] = NULL; lv->sap_fcnt[q] = 0;
  }
  free(lv->dc_ghost); lv->dc_ghost = NULL;
  lv->sap_rbuf = lv->sap_Drbuf = lv->sap_dbuf = NULL;
  lv->sap_lnbr = NULL; lv->sap_gsite = NULL;
  lv->sap_cblk[0] = lv->sap_cblk[1] = NULL;
  lv->sap_ncblk[0] = lv->sap_ncblk[1] = 0;
  lv->sap_R = lv->sap_dpsi = NULL;
  lv->sap_nbrblk = NULL; lv->sap_xnbr = NULL;
  lv->sap_dpsi_field = lv->sap_dpsi_ghost = NULL; lv->sap_nbr_rank = NULL;
  lv->sap_nthr = 0;
}
