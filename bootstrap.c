// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

/* DDalphaAMG adaptive bootstrap setup for the fine near-null vectors (thesis
 * Algorithm 16): build the test vectors that define the prolongator P by
 * inverse iteration on the current multigrid method.  This is a
 * single-precision module (like mixed_bridge.c -- it owns part of the single
 * hierarchy build), so the plain operator / MG / utils names resolve to the
 * _f instantiation via mangle.h.
 *
 * Algorithm 16 (= Algorithm 14 with the line-8 initial-guess modification):
 *   1   v_1..v_N  <- random
 *   2-4 for tau=1..3:  v_j <- M_SAP v_j (initial smoothing, no Dc yet)
 *   5   for tau=1..ninv:
 *   6     rebuild P and D_c = P^H D P from current v_j
 *   7-8   for j:  v_j <- v_j + C^(nu)(v_j - D v_j)
 *                  (one V-cycle toward D^{-1}v_j)
 *   9     v_j <- v_j / ||v_j||
 *
 * All of the heavy lifting reuses existing primitives -- the SAP smoother
 * (red_black_schwarz), the V-cycle (mg_vcycle), the fused fine residual
 * (D_residual), the Galerkin coarse build (build_coarse_op_direct) and the
 * local orthonormalization (local_orthonormalize).  The only new code is
 * (a) the random seed and (b) the column-block gather/scatter between the
 * nv-layout array P consumes and the num_rhs-wide fields the smoother /
 * V-cycle operate on.
 *
 * Precision follows mixed_bridge.c, whose hierarchy this fills: single in a
 * mixed build (so the plain names resolve to the _f instantiation), the build's
 * own precision in a one-precision build (MIXED=0), where no _f twin exists.
 */
#ifdef MIXED
#define MG_SINGLE 1
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <omp.h>
#include <sys/time.h>
#include "global_params.h"
#include "coarse.h"          // local_orthonormalize, build_coarse_op_direct,
                             // free_coarse_op, mg_setup, mg_vcycle
#include "clover_ferm.h"     // D_residual
#include "solver.h"          // red_black_schwarz
#include "utils.h"           // caxpy_ferm
#include "comm.h"            // allreduce_sum_d (global GS / CholQR / norms)
#include "bootstrap.h"

// The num_null_vecs columns are processed NBATCH batches of num_rhs at a time:
// the smoother / V-cycle / Dslash all operate on num_rhs-wide multi-rhs fields.
#define NBATCH (num_null_vecs / num_rhs)
_Static_assert(num_null_vecs % num_rhs == 0,
               "bootstrap requires num_null_vecs to be a multiple of num_rhs");
_Static_assert(num_null_vecs % 2 == 0, "num_null_vecs must be even");
_Static_assert(COARSE_NULL_VECS % num_rhs == 0,
               "bootstrap requires COARSE_NULL_VECS to be a multiple of num_rhs");
_Static_assert(NUM_LEVELS >= 2, "NUM_LEVELS must be at least 2");
// Per-level column counts thread through the helpers as `ncol` (nv-layout
// column stride): num_null_vecs for the fine level, COARSE_NULL_VECS for the
// L1->L2 coarse level (in general lv->nnv).  The batch count is ncol/num_rhs.

// Number of REAL slots in one single-rhs field at the fine level
// (== ferm_size). Both the nv-layout (slot * num_null_vecs + vec) and a
// multi-rhs field (slot * num_rhs + rhs) index by this same slot in the same
// ordering, so the two differ only in how wide the minor axis is (ncol vs
// num_rhs): moving between them is a column-block slice, not a transpose --
// nothing is reordered, the batch is just narrower.  The coarse level has the
// identical structure with nslot = coarse_ferm_size, so every helper below is
// parameterized by `nslot` and serves both levels.
#define NSLOT ((long) ferm_size)

// --- setup profiling -------------------------------------------------------
// Self-contained timers for the one-time bootstrap setup, kept separate from
// the solver / V-cycle profiling (which times the steady-state solve).  The
// categories are the setup's distinct phases: the inverse-iteration cycle
// work (`cycle`), the per-sweep hierarchy rebuild it triggers
// (`build_dc`/`add_coarse`, `local_ortho`, `mg_setup`), and the bookkeeping
// (`ggs`, `batchio`, `init`, `restrict`).
static struct {
  double init;        // random seed + initial SAP smoothing (init_smooth_fine)
  double ggs;         // global Gram-Schmidt over all test vectors
  double cycle;       // mg_cycle inverse-iteration (recursive V-cycle solve)
  double batchio;     // gather/scatter/normalize batch bookkeeping
  double local_ortho; // per-aggregate local_orthonormalize (fine P columns)
  double build_dc;    // free+build_coarse_op_direct (fine Galerkin probe
                      //   build of D_1)
  double mg_setup;    // mg_setup hierarchy register/teardown
  double add_coarse;  // mg_add_coarse_level (coarse Galerkin + ortho + setup)
  double restrict_;   // apply_R coarse-seed restriction
  long   n_cycle;     // mg_cycle batch invocations
  long   n_build_dc;  // build_coarse_op_direct calls
  long   n_add_coarse;// mg_add_coarse_level calls
} g_bt;

static double bt_now(void) {
  struct timeval tv; gettimeofday(&tv, NULL);
  return (double) tv.tv_sec + 1e-6 * (double) tv.tv_usec;
}
static void bt_reset(void) {
  memset(&g_bt, 0, sizeof g_bt);
}

// Time a statement into accumulator field `acc`.
#define BT_TIME(acc, stmt) \
  do { \
    double _t0 = bt_now(); \
    stmt; \
    g_bt.acc += bt_now() - _t0; \
  } while (0)

static void bt_report(const char * tag) {
  double total = g_bt.init + g_bt.ggs + g_bt.cycle + g_bt.batchio
                + g_bt.local_ortho + g_bt.build_dc + g_bt.mg_setup
                + g_bt.add_coarse + g_bt.restrict_;
  double inv = total > 0 ? 100.0 / total : 0.0;
  print0("\n==== bootstrap setup profile (%s) ====\n", tag);
  print0("  %-22s %9.3f s  %5.1f%%  (%ld cycle-batches)\n",
         "inverse-iter cycle", g_bt.cycle, g_bt.cycle * inv, g_bt.n_cycle);
  print0("  %-22s %9.3f s  %5.1f%%  (%ld builds)\n",
         "Galerkin build D1", g_bt.build_dc, g_bt.build_dc * inv,
         g_bt.n_build_dc);
  print0("  %-22s %9.3f s  %5.1f%%  (%ld adds)\n",
         "mg_add_coarse_level", g_bt.add_coarse, g_bt.add_coarse * inv,
         g_bt.n_add_coarse);
  print0("  %-22s %9.3f s  %5.1f%%\n", "mg_setup register", g_bt.mg_setup,
         g_bt.mg_setup * inv);
  print0("  %-22s %9.3f s  %5.1f%%\n", "local_orthonormalize",
         g_bt.local_ortho, g_bt.local_ortho * inv);
  print0("  %-22s %9.3f s  %5.1f%%\n", "global Gram-Schmidt",
         g_bt.ggs, g_bt.ggs * inv);
  print0("  %-22s %9.3f s  %5.1f%%\n", "init smooth",
         g_bt.init,        g_bt.init * inv);
  print0("  %-22s %9.3f s  %5.1f%%\n", "restrict seed",
         g_bt.restrict_,   g_bt.restrict_ * inv);
  print0("  %-22s %9.3f s  %5.1f%%\n", "batching+normalize",
         g_bt.batchio,     g_bt.batchio * inv);
  print0("  %-22s %9.3f s\n", "TOTAL", total);
  print0("========================================\n\n");
}

// --- null-vector batching --------------------------------------------------
// The smoother, V-cycle and Dslash all work on num_rhs-wide fields, so the ncol
// null vectors are fed through them num_rhs at a time.  Gather batch b (columns
// [b*num_rhs, b*num_rhs+num_rhs)) of an nv-layout array into such a field, and
// scatter it back.  `nslot` = REAL slots per single-rhs field (ferm_size fine,
// coarse_ferm_size coarse).
static void gather_block(REAL * field, const REAL * nv, int b, long nslot,
                         int ncol) {
  int off = b * num_rhs;
  #pragma omp parallel for schedule(static)
  for (long s = 0; s < nslot; s ++)
    for (int r = 0; r < num_rhs; r ++)
      field[s * num_rhs + r] = nv[s * (long) ncol + off + r];
}
static void scatter_block(REAL * nv, const REAL * field, int b, long nslot,
                          int ncol) {
  int off = b * num_rhs;
  #pragma omp parallel for schedule(static)
  for (long s = 0; s < nslot; s ++)
    for (int r = 0; r < num_rhs; r ++)
      nv[s * (long) ncol + off + r] = field[s * num_rhs + r];
}

// Do the batches tile the columns exactly?  Gather every batch, scatter it 
// back into a zeroed array, and count the elements the round trip failed to
// reproduce.  Zeroed matters -- a batch that is never visited leaves zeros 
// behind, so this catches a skipped batch as well as a wrong offset or stride.
static long batching_roundtrip_errors(const REAL * nv, long nslot, int ncol) {
  long ntot = nslot * (long) ncol;
  REAL * field = malloc(sizeof(REAL) * nslot * num_rhs);
  REAL * back  = malloc(sizeof(REAL) * ntot);
  memset(back, 0, sizeof(REAL) * ntot);
  for (int b = 0; b < ncol / num_rhs; b ++) {
    gather_block(field, nv, b, nslot, ncol);
    scatter_block(back, field, b, nslot, ncol);
  }
  long bad = 0;
  for (long i = 0; i < ntot; i ++) if (back[i] != nv[i]) bad ++;
  free(field); free(back);
  return bad;
}

// Unit test for the batching, run by the `batching` arm rather than by every
// production setup.  gather_block / scatter_block branch on nothing, so their
// correctness is uniform in (nslot, ncol): a few small synthetic shapes prove
// what running it on the live hierarchy proved, without the wall time or the
// full-size scratch copy.  The other half of the precondition -- that each
// level's nnv really is a multiple of num_rhs -- is config-dependent and is
// enforced where the config is read, by the abort in bootstrap_null_vecs_nlvl.
//
// Values are the flat index itself, distinct across the whole array (and exact
// in float up to 2^24, well above the largest shape here), so a batch landing
// at the wrong offset cannot match by coincidence the way a repeating pattern 
// could.
int bootstrap_selfcheck_batching(void) {
  const long slots[] = { 1, 97, 4096 };    // 1 exercises the degenerate extent
  const int  ncols[] = { num_rhs, 2 * num_rhs, 3 * num_rhs };  // 1-3 batches
  long total_bad = 0;
  for (unsigned si = 0; si < sizeof slots / sizeof *slots; si ++)
    for (unsigned ci = 0; ci < sizeof ncols / sizeof *ncols; ci ++) {
      const long nslot = slots[si];
      const int  ncol  = ncols[ci];
      const long n     = nslot * (long) ncol;
      REAL * nv = malloc(sizeof(REAL) * n);
      for (long i = 0; i < n; i ++) nv[i] = (REAL) i;
      const long bad = batching_roundtrip_errors(nv, nslot, ncol);
      total_bad += bad;
      print0("  nslot %5ld  ncol %3d (%d batch(es)): %ld/%ld mismatches\n",
             nslot, ncol, ncol / num_rhs, bad, n);
      free(nv);
    }
  print0("test_batching: null-vector batching round-trip => %s\n",
         total_bad ? "FAIL" : "PASS");
  return total_bad != 0;
}

// --- random seed -----------------------------------------------------------
// Deterministic, parallel-safe random fill: hash the flat index (splitmix64)
// so the result is reproducible and independent of thread count -- no shared
// RNG state.  Values in [-0.5, 0.5) per real/imag component; the precise
// distribution is immaterial, the setup amplifies the near-kernel regardless
// of the seed's shape.
static inline double idx_uniform(unsigned long i, unsigned long seed) {
  unsigned long s = i + seed * 0x9E3779B97F4A7C15UL;
  s ^= s >> 30; s *= 0xBF58476D1CE4E5B9UL;
  s ^= s >> 27; s *= 0x94D049BB133111EBUL;
  s ^= s >> 31;
  return (double) (s >> 11) * (1.0 / 9007199254740992.0) - 0.5;
}
// Fine-lattice random fill, keyed on global position so that the starting
// field is the same whatever the process grid.  Keying on the flat array index
// would key on a local one: each rank would then emit the same value sequence
// over its own share, making the global field a periodic repeat of one rank's,
// and since the whole hierarchy is built from these vectors, every level would
// inherit the decomposition.  The key is the index the site would occupy on a
// single-rank lattice -- the same storage map at global extents, shifted by
// this rank's origin -- so at one rank it is just i.
static void random_init_fine(REAL * nv, unsigned long seed, int ncol) {
  const int  oZ = qc_grid.origin[2], oT = qc_grid.origin[3];
  // V/Vh/ferm_size are unparenthesized macros -- bind them once, as full
  // initializer expressions, rather than dividing by them inline.
  const long vh_loc = (long) Vh;
  const long vh_glb = (long) (Lx / 2) * Ly * Lz_glob * Lt_glob;
  const long n      = (long) ferm_size * ncol;
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < n; i ++) {
    const int col = (int) (i % ncol);
    long r = i / ncol;                       // real index within one column
    const int reim = (int) (r & 1);   r >>= 1;
    const int c    = (int) (r % Nc);  r /= Nc;
    const int d    = (int) (r % Ns);  r /= Ns;
    const int par  = (int) (r / vh_loc);     // parity is stored globally
    const long sh  = r % vh_loc;
    int xh, y, z, t;
    site_half_coords(sh, &xh, &y, &z, &t);
    const int gz = z + oZ, gt = t + oT;
    // x's low bit, halved away by the eo split, is not needed: parity is
    // stored globally, so (par, xh, y, gz, gt) already labels the site
    // uniquely -- the two x sharing an xh differ in parity, and so land in
    // different halves of the key.
    const long gloc = (long) par * vh_glb
                    + host_loc_of_dims(Lx / 2, Ly, Lz_glob, Lt_glob,
                                       2 * xh, y, gz, gt);
    nv[i] = (REAL) idx_uniform((unsigned long)
              ((((gloc * Ns + d) * Nc + c) * 2 + reim) * (long) ncol + col),
              seed);
  }
}

// The same for a coarse level's test vectors, which an increasing-width
// hierarchy (nnv[d] > nnv[d-1]) seeds with a random surplus on the child grid.
// Simpler than the fine case: a coarse fermion is stored in full
// coarse-lexicographic site order with no even-odd split (coarse.h), so the
// index decodes by plain division and no parity is involved, and the level's
// corigin is already its origin in that level's own sites.
static void random_init_coarse(REAL * nv, const level_t * lv,
                               unsigned long seed, int ncol) {
  const int nvar = lv->nvar;
  const int nx = lv->nx, ny = lv->ny, nz = lv->nz;
  const int gnz = nz * qc_grid.P[2];       // global z extent at this level
  const int oZ = lv->corigin[2], oT = lv->corigin[3];
  const long n = lv->field_len * (long) ncol;
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < n; i ++) {
    const int col = (int) (i % ncol);
    long r = i / ncol;                       // real index within one column
    const int reim = (int) (r & 1);   r >>= 1;
    const int v    = (int) (r % nvar); r /= nvar;
    const int x = (int) (r % nx);  r /= nx;
    const int y = (int) (r % ny);  r /= ny;
    const int z = (int) (r % nz);  r /= nz;
    const int t = (int) r;
    const long gcl = (((long) (t + oT) * gnz + (z + oZ)) * ny + y) * nx + x;
    nv[i] = (REAL) idx_uniform((unsigned long)
              (((gcl * nvar + v) * 2 + reim) * (long) ncol + col), seed);
  }
}

// Bit-level "finite and strictly positive" test.  -ffast-math
// (-ffinite-math-only) lets the compiler assume no NaN/Inf, so `x > 0` /
// isnan(x) are unreliable for catching a poisoned reduction -- it folds
// `x > 0` assuming x is finite.  Inspect the IEEE sign/exponent bits directly
// (memcpy is not an FP op, so the assumption cannot touch it): sign clear,
// exponent not all-ones, value nonzero.  This is what contains a non-finite
// norm or pivot: a near-singular coarse Dirichlet block can make the
// from-zero coarse smooth or a setup V-cycle overflow to Inf/NaN in fp32, and
// the check keeps that out of the test vectors.
static inline int finite_pos_d(double x) {
  unsigned long long u; memcpy(&u, &x, 8);
  return (u & 0x8000000000000000ull) == 0
      && (u & 0x7FF0000000000000ull) != 0x7FF0000000000000ull
      && (u & 0x7FFFFFFFFFFFFFFFull) != 0;
}

// --- per-column normalization ----------------------------------------------
// Normalize each of the num_rhs columns of a multi-rhs field to unit L2
// norm.  The global scale is irrelevant to P (the per-aggregate
// orthonormalize renormalizes), but it keeps magnitudes well-conditioned as
// inverse iteration amplifies the near-kernel, and mirrors the algorithm's
// per-vector normalization step.  A column whose norm comes back non-finite (a
// near-singular coarse operator overflowed the smooth / setup cycle to
// Inf/NaN) is zeroed rather than propagated; the per-aggregate ortho then
// refills it with an orthonormal complement (coarse.c), so the poison is
// contained here.
static void normalize_block(REAL * field, long nslot) {
  double nrm[num_rhs];
  for (int r = 0; r < num_rhs; r ++) nrm[r] = 0.0;
  #pragma omp parallel
  {
    double loc[num_rhs];
    for (int r = 0; r < num_rhs; r ++) loc[r] = 0.0;
    #pragma omp for schedule(static) nowait
    for (long s = 0; s < nslot; s ++)
      for (int r = 0; r < num_rhs; r ++) {
        double v = (double) field[s * num_rhs + r];
        loc[r] += v * v;
      }
    #pragma omp critical
    for (int r = 0; r < num_rhs; r ++) nrm[r] += loc[r];
  }
  allreduce_sum_d(nrm, num_rhs);
  REAL inv[num_rhs];
  int  bad[num_rhs];
  for (int r = 0; r < num_rhs; r ++) {
    bad[r] = !finite_pos_d(nrm[r]);
    inv[r] = bad[r] ? (REAL) 0.0 : (REAL) (1.0 / sqrt(nrm[r]));
  }
  #pragma omp parallel for schedule(static)
  for (long s = 0; s < nslot; s ++)
    for (int r = 0; r < num_rhs; r ++)
      // A non-finite norm means the column overflowed to Inf/NaN in fp32
      // (a near-singular coarse operator amplified the setup cycle past 
      // FLT_MAX).  Assign zero -- scaling by inv=0 would leave Inf*0 = NaN
      // (and -ffast-math keeps that NaN), which is the exact poison this
      // guard exists to stop.  The per-aggregate ortho then refills the
      // zeroed column with an orthonormal complement.
      field[s * num_rhs + r] = bad[r] ? (REAL) 0.0
                                      : field[s * num_rhs + r] * inv[r];
}

// --- global Gram-Schmidt over the test vectors -----------------------------
// Classical Gram-Schmidt over the num_null_vecs columns using whole-field
// complex inner products, in place on the nv-layout (slot 2c = Re,
// slot 2c+1 = Im of complex component c).  This is DDalphaAMG's
// gram_schmidt_PRECISION (the bootstrap step, setup_generic.c:376, via
// process_multi_inner_product_PRECISION): it keeps the test vectors a well-
// conditioned global basis between inverse-iteration sweeps, preventing their
// collapse toward the few smallest modes.  The per-aggregate
// orthonormalization at P-build time does not do this -- it only decouples
// vectors within an aggregate.
//
// Structure (classical, not modified GS): for each active column i, one fused
// streaming pass computes all projection coefficients p_j = <V[j], V[i]> (j<i)
// against the original V[i] -- the active column stays cache-resident while it
// is dotted against every earlier column -- then a single pass subtracts all
// projections (fusing the norm accumulation) and a final pass scales: 3 field
// passes + 3 fork/joins per column.
static void global_gram_schmidt(REAL * tv, long nslot, int ncol) {
  long ncpx = nslot / 2;               // complex components per vector
  const int N = ncol;                  // columns (test vectors) at this level
  // p_j = <V[j], V[i]>, double accumulation
  double pr [MAX_NULL_VECS], pi [MAX_NULL_VECS];

  for (int i = 0; i < N; i ++) {
    // --- fused multi-inner-product: all p_j = <V[j], V[i]> (j<i) in one pass
    for (int j = 0; j < i; j ++) { pr[j] = 0.0; pi[j] = 0.0; }
    #pragma omp parallel
    {
      double lr[MAX_NULL_VECS], li[MAX_NULL_VECS];
      for (int j = 0; j < i; j ++) { lr[j] = 0.0; li[j] = 0.0; }
      #pragma omp for schedule(static) nowait
      for (long c = 0; c < ncpx; c ++) {
        long re = (2 * c) * (long) N, im = re + N;
        // active column i, component c
        REAL air = tv[re + i], aii = tv[im + i];
        for (int j = 0; j < i; j ++) {
          REAL bjr = tv[re + j], bji = tv[im + j];
          lr[j] += (double) bjr * air + (double) bji * aii;  // Re <V[j],V[i]>
          li[j] += (double) bjr * aii - (double) bji * air;  // Im <V[j],V[i]>
        }
      }
      #pragma omp critical
      for (int j = 0; j < i; j ++) { pr[j] += lr[j]; pi[j] += li[j]; }
    }
    allreduce_sum_d(pr, i);
    allreduce_sum_d(pi, i);

    // --- V[i] -= sum_j (pr+i*pi)_j V[j], fusing ||V[i]||^2 accumulation ----
    double nrm = 0.0;
    #pragma omp parallel for reduction(+:nrm) schedule(static)
    for (long c = 0; c < ncpx; c ++) {
      long re = (2 * c) * (long) N, im = re + N;
      double air = tv[re + i], aii = tv[im + i];
      for (int j = 0; j < i; j ++) {
        REAL bjr = tv[re + j], bji = tv[im + j];
        air -= pr[j] * bjr - pi[j] * bji;
        aii -= pr[j] * bji + pi[j] * bjr;
      }
      tv[re + i] = (REAL) air; tv[im + i] = (REAL) aii;
      nrm += air * air + aii * aii;
    }
    allreduce_sum_d(&nrm, 1);

    // --- normalize V[i] ----------------------------------------------------
    // A non-finite norm means the column overflowed to Inf/NaN in fp32;
    // assign zero rather than scale -- Inf*0 = NaN would re-poison it (and 
    // -ffast-math keeps the NaN).  A zeroed column makes no contribution to
    // later projections and is refilled with an orthonormal complement by the
    // per-aggregate ortho downstream.
    int  bad = !finite_pos_d(nrm);
    REAL inv = bad ? (REAL) 0.0 : (REAL) (1.0 / sqrt(nrm));
    #pragma omp parallel for schedule(static)
    for (long c = 0; c < ncpx; c ++) {
      long re = (2 * c) * (long) N, im = re + N;
      tv[re + i] = bad ? (REAL) 0.0 : tv[re + i] * inv;
      tv[im + i] = bad ? (REAL) 0.0 : tv[im + i] * inv;
    }
  }
}

// --- global Cholesky-QR over the test vectors -----------------------------
// Same factorization as global_gram_schmidt above, computed blockwise: the
// Cholesky factor R of the Gram matrix G = V^H V is exactly the triangular
// factor classical GS produces (both have positive real diagonal), so
// V <- V R^{-1} yields the same orthonormal basis up to rounding.  The win is
// traffic and parallelism: one streaming pass forms G (double accumulation)
// and one pass applies the precomputed R^{-1}, vs ~3 passes per column for
// the per-column sweep -- and the nv layout (vec fastest, num_null_vecs
// contiguous REALs per slot) makes both passes unit-stride over the column
// index.  The O(N^3) Cholesky / triangular inversion on the 32x32 G is noise.

// Conditioning: forming G squares cond(V), so the double-precision
// factorization is sound up to cond(V) ~ 1e7 and the single-precision apply
// contributes ~eps_f*cond(V) orthogonality error.  min/max of R's diagonal is
// a cheap collapse proxy (R_ii = the norm of column i's component orthogonal
// to columns < i): when it dips below CHOLQR_RATIO_TOL a second pass
// (CholeskyQR2) restores orthogonality to working precision.  A non-positive
// pivot (cond beyond double) falls back to the classical-GS sweep,
// which never breaks down.
#define CHOLQR_RATIO_TOL 1e-3
static int global_cholqr_pass(REAL * tv, long nslot, int ncol, double * diag_ratio) {
  const int N = ncol;                   // columns (test vectors) at this level
  const long ncpx = nslot / 2;          // complex components per vector

  // --- pass 1: Gram matrix G = V^H V, upper triangle, double accumulation --
  // The N x N work arrays are on the heap: a few hundred columns would
  // overflow a thread's stack.
  double (*G)[N][N] = malloc(sizeof(double[4][N][N]));  // Gr, Gi, Rr, Ri
  double (*Gr)[N] = G[0], (*Gi)[N] = G[1];
  memset(G, 0, sizeof(double[2][N][N]));
  #pragma omp parallel
  {
    double (*l)[N][N] = malloc(sizeof(double[2][N][N]));
    double (*lr)[N] = l[0], (*li)[N] = l[1];
    memset(l, 0, sizeof(double[2][N][N]));
    #pragma omp for schedule(static) nowait
    for (long c = 0; c < ncpx; c ++) {
      const REAL * ar = tv + (2 * c) * (long) N, * ai = ar + N;
      for (int i = 0; i < N; i ++) {
        double xr = ar[i], xi = ai[i];
        for (int j = i; j < N; j ++) {
          // Re <V[i],V[j]>
          lr[i][j] += xr * (double) ar[j] + xi * (double) ai[j];
          // Im <V[i],V[j]>
          li[i][j] += xr * (double) ai[j] - xi * (double) ar[j];
        }
      }
    }
    #pragma omp critical
    for (int i = 0; i < N; i ++)
      for (int j = i; j < N; j ++) {
        Gr[i][j] += lr[i][j];
        Gi[i][j] += li[i][j];
      }
    free(l);
  }
  // global Gram matrix: lower triangles are zero on every rank, so summing the
  // full N x N arrays is exact; the Cholesky / R^{-1} below then runs
  // redundantly but deterministically on identical inputs on all ranks (same
  // double math => same W), so no broadcast is needed.
  allreduce_sum_d(&Gr[0][0], (long) N * N);
  allreduce_sum_d(&Gi[0][0], (long) N * N);

  // --- Cholesky G = R^H R (R upper triangular, real positive diagonal) -----
  double (*Rr)[N] = G[2], (*Ri)[N] = G[3];
  double dmin = 0.0, dmax = 0.0;
  for (int i = 0; i < N; i ++) {
    double d = Gr[i][i];
    for (int k = 0; k < i; k ++)
      d -= Rr[k][i] * Rr[k][i] + Ri[k][i] * Ri[k][i];
    // Breakdown -> classical-GS fallback.  Bit-level test so a non-finite
    // pivot is caught under -ffast-math (where !(d > 0) is folded to d <= 0,
    // false for NaN).
    if (!finite_pos_d(d)) {
      free(G);
      return 0;
    }
    double rii = sqrt(d);
    if (i == 0)
      dmin = dmax = rii;
    else {
      if (rii < dmin) dmin = rii;
      if (rii > dmax) dmax = rii;
    }
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

  // --- W = R^{-1} (upper triangular back-substitution, row-wise) -----------
  REAL (*W)[N][N] = malloc(sizeof(REAL[2][N][N]));
  REAL (*Wr)[N] = W[0], (*Wi)[N] = W[1];
  for (int i = 0; i < N; i ++) {
    double wr [MAX_NULL_VECS], wi [MAX_NULL_VECS];
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

  // --- pass 2: V <- V W, per component row (1x32 GEMV, unit-stride over j)
  #pragma omp parallel
  {
    REAL xr[MAX_NULL_VECS], xi[MAX_NULL_VECS];
    #pragma omp for schedule(static)
    for (long c = 0; c < ncpx; c ++) {
      REAL * ar = tv + (2 * c) * (long) N, * ai = ar + N;
      for (int i = 0; i < N; i ++) {
        xr[i] = ar[i];
        xi[i] = ai[i];
        ar[i] = 0;
        ai[i] = 0;
      }
      for (int k = 0; k < N; k ++) {
        REAL br = xr[k], bi = xi[k];
        for (int j = k; j < N; j ++) {
          ar[j] += br * Wr[k][j] - bi * Wi[k][j];
          ai[j] += br * Wi[k][j] + bi * Wr[k][j];
        }
      }
    }
  }
  free(G); free(W);
  return 1;
}

static void global_cholesky_qr(REAL * tv, long nslot, int ncol) {
  double ratio = 0.0;
  if (!global_cholqr_pass(tv, nslot, ncol, &ratio)) {
    print0("bootstrap: CholeskyQR breakdown -> classical-GS fallback\n");
    global_gram_schmidt(tv, nslot, ncol);
    return;
  }
  int pass2 = ratio < CHOLQR_RATIO_TOL;
  // Silent while healthy, one decade of warning before pass 2 -- same rule as
  // agg_ortho_report.  The breakdown paths print for themselves either way.
  if (ratio < 10 * CHOLQR_RATIO_TOL)
    print0("bootstrap: cholqr diag ratio %.2e%s\n", ratio,
           pass2 ? " -> pass 2" : "");
  if (pass2 && !global_cholqr_pass(tv, nslot, ncol, &ratio)) {
    print0("bootstrap: CholeskyQR pass-2 breakdown -> classical-GS fallback\n");
    global_gram_schmidt(tv, nslot, ncol);
  }
}

// One bootstrap sweep at a single level (fine or coarse): global Gram-Schmidt
// over all test vectors, then replace each with a zero-guess recursive cycle
// applied to it (v_j <- C v_j / ||C v_j||), batched num_rhs at a time.
// `lv` is the level the cycle starts on (mg_get_level(depth));
// mg_cycle recurses through all coarser levels with a zero initial guess,
// so x = C v_j exactly -- DDalphaAMG's vcycle(x, NULL, v_j, _NO_RES).
// At depth 0 the fine operator handles must be patched into the level (they
// are only known at apply time); deeper levels carry their assembled Dc.
static void bootstrap_sweep(level_t * lv, REAL * tv, long nslot, REAL * rhs,
                            REAL * x, REAL * Uk, REAL * Fcsw, int clover,
                            const mg_params * p) {
  if (lv->depth == 0) {
    lv->U_kappa = Uk;
    lv->F_csw = Fcsw;
    lv->clover = clover;
  }
  int ncol = lv->nnv;                     // test vectors coarsening this level
  BT_TIME(ggs, global_cholesky_qr(tv, nslot, ncol));
  for (int b = 0; b < ncol / num_rhs; b ++) {
    BT_TIME(batchio, gather_block(rhs, tv, b, nslot, ncol));   // rhs = v_j
    // x = C v_j (zero guess, recursive)
    BT_TIME(cycle, mg_cycle(lv, x, rhs, (REAL) p->setup_k_tol));
    g_bt.n_cycle ++;
    BT_TIME(batchio, normalize_block(x, nslot));
    BT_TIME(batchio, scatter_block(tv, x, b, nslot, ncol));    // v_j <- x
  }
}

// --- (re)build the fine level (P_0, D_1) and register the 2-level base -----
// Mirrors DDalphaAMG re_setup: the P columns (`nv`) are a copy of the test
// vectors (`tv`) that is aggregate-orthonormalized, so `tv` itself is never
// scrambled by the local GS and the next sweep's global GS / inverse
// iteration act on the raw test vectors.  build_coarse_op_direct callocs
// D_c->self/nbr fresh on every call, and mg_setup/teardown never free the
// caller-owned level-1 D_c, so we free the previous arrays first (guarded so
// the first call on a statically-zeroed D_c is a no-op).
// `rebuild_dc`: when 0, reuse the existing D_1 in Dc (skip the expensive
// Galerkin probe build) -- valid only when the fine test vectors are
// unchanged.
static void rebuild_fine(const REAL * tv, REAL * nv, coarse_op * Dc,
                         REAL * Uk, REAL * Fcsw, int clover, int rebuild_dc,
                         const mg_params * p) {
  if (rebuild_dc) {
    memcpy(nv, tv, sizeof(REAL) * NSLOT * (long) num_null_vecs);
    BT_TIME(local_ortho, local_orthonormalize(nv));
    if (Dc->self) free_coarse_op(Dc);
    BT_TIME(build_dc, build_coarse_op_direct(Dc, nv, Uk, Fcsw, clover));
    g_bt.n_build_dc ++;
  }
  BT_TIME(mg_setup, mg_setup(Dc, nv, p));
}

// Convenience wrapper for the two-level path: always rebuild D_1.
static void rebuild_hierarchy(const REAL * tv, REAL * nv, coarse_op * Dc,
                              REAL * Uk, REAL * Fcsw, int clover,
                              const mg_params * p) {
  rebuild_fine(tv, nv, Dc, Uk, Fcsw, clover, /*rebuild_dc*/ 1, p);
}

// Random seed + initial SAP smoothing of the fine test vectors -- DDalphaAMG's
// interpolation_PRECISION_define for the SAP method: init_sweeps chained
// single-sweep applications v <- M_SAP v (each re-smooths the previous
// result), then normalize.  rhs/x are num_rhs-wide fine scratch fields.
static void init_smooth_fine(REAL * tv, REAL * rhs, REAL * x, int init_sweeps,
                             REAL * Uk, REAL * Fcsw, int clover,
                             const mg_params * p) {
  random_init_fine(tv, 0x5DEECE66DUL, num_null_vecs);
  for (int b = 0; b < NBATCH; b ++) {
    gather_block(rhs, tv, b, NSLOT, num_null_vecs);
    for (int m = 0; m < init_sweeps; m ++) {
      // x = M_SAP rhs
      red_black_schwarz(x, rhs, Uk, Fcsw, clover, 1, p->sap_block_iter,
                        (REAL) 0, 0, NULL);
      // rhs <- result
      REAL * t = rhs;
      rhs = x;
      x = t;
    }
    normalize_block(rhs, NSLOT);
    scatter_block(tv, rhs, b, NSLOT, num_null_vecs);
  }
}

// Initial coarse smoothing of level `lv`'s test vectors -- the coarse-depth
// analog of init_smooth_fine, i.e. DDalphaAMG's interpolation_PRECISION_define
// at depth > 0 (setup_generic.c).  A few coarse-SAP smoother sweeps condition
// the vectors -- crucially the random-padded surplus columns present when a
// level carries more test vectors than its parent -- into locally near-null
// shape.  Without it a raw-random surplus poisons the first Galerkin coarse
// operator and never separates from the restricted columns, which is the
// rank-deficient transfer that appears when the null-vector count increases
// with depth.  Called after mg_add_coarse_level has promoted `lv`, so its
// coarse_sap smoother is live and its cache warm.  coarse_sap accumulates
// into its output, so x is zeroed each sweep to make the apply a from-zero
// smooth (x = M_SAP eta), matching init_smooth_fine's chained sweeps.
static void init_smooth_coarse(level_t * lv, REAL * tv, long nslot, int ncol,
                               REAL * rhs, REAL * x, int init_sweeps) {
  long fld = nslot * num_rhs;
  for (int b = 0; b < ncol / num_rhs; b ++) {
    gather_block(rhs, tv, b, nslot, ncol);
    for (int m = 0; m < init_sweeps; m ++) {
      #pragma omp parallel for schedule(static)
      for (long i = 0; i < fld; i ++) x[i] = 0;
      coarse_sap(lv, x, rhs, NULL);            // x = M_SAP rhs (from zero)
#if BOTTOM_SOLVE_DIAG
      {
        double mx = 0.0;
        for (long i = 0; i < fld; i ++) {
          double a = fabs((double) x[i]);
          if (a > mx) mx = a;
        }
        print0("[init_smooth_coarse] depth %d block %d sweep %d: max|x| %.3e\n",
               lv->depth, b, m, mx);
      }
#endif
      // Rescale each sweep (power iteration).  coarse_sap ~ D_block^-1
      // amplifies near-null modes; on a near-critical coarse operator a
      // near-singular Dirichlet block amplifies by orders of magnitude per
      // sweep, so three un-normalized chained sweeps overflow fp32 to Inf and
      // normalize_block below turns Inf*0 into NaN -- poisoning the
      // transfer.  Normalizing each step keeps the magnitude O(1) while the
      // smoothed direction (the low-mode alignment we want) is unchanged, so
      // this is the numerically-correct chained smooth.  (init_smooth_fine's
      // fine SAP cannot amplify like this, so it normalizes only once.)
      normalize_block(x, nslot);
      REAL * t = rhs; rhs = x; x = t;          // rhs <- smoothed (normalized)
    }
    normalize_block(rhs, nslot);
    scatter_block(tv, rhs, b, nslot, ncol);
  }
}

void bootstrap_null_vecs(REAL * nv, coarse_op * Dc,
                         REAL * Uk, REAL * Fcsw, int clover,
                         const mg_params * p) {
  const int ninv        = p->setup_sweeps;
  const int init_sweeps = p->setup_init_sweeps;
  long fld = NSLOT * num_rhs;
  // tv = the raw test vectors (DDalphaAMG's is.test_vector): global GS
  // + inverse iteration act on these.  nv (caller-owned) holds the P columns
  // (is.interpolation), an aggregate-orthonormalized copy rebuilt each
  // sweep -- it is what the registered hierarchy and the V-cycle consume.
  REAL * tv  = malloc(sizeof(REAL) * NSLOT * (long) num_null_vecs);
  REAL * rhs = malloc(sizeof(REAL) * fld);
  REAL * x   = malloc(sizeof(REAL) * fld);

  print0("bootstrap[2lvl]: %d init sweeps + %d bootstrap sweep(s), "
         "%d test vecs, %d batch(es) of %d\n",
         init_sweeps, ninv, num_null_vecs, NBATCH, num_rhs);

  bt_reset();
  BT_TIME(init, init_smooth_fine(tv, rhs, x, init_sweeps, Uk, Fcsw, clover, p));
  rebuild_hierarchy(tv, nv, Dc, Uk, Fcsw, clover, p);

  // Bootstrap sweeps (inv_iter_inv_fcycle): global GS + zero-guess V-cycle per
  // vector, then rebuild P, D_c from the improved vectors.
  for (int it = 0; it < ninv; it ++) {
    bootstrap_sweep(mg_get_level(0), tv, NSLOT, rhs, x, Uk, Fcsw, clover, p);
    rebuild_hierarchy(tv, nv, Dc, Uk, Fcsw, clover, p);
    print0("bootstrap[2lvl]: sweep %d/%d done\n", it + 1, ninv);
  }
  bt_report("2lvl");
  free(tv); free(rhs); free(x);
}

// --- multilevel (N-level) bootstrap ----------------------------------------
// Everything the N-level bootstrap threads through the recursion and the
// rebuild: per-level test vectors / scratch / schedule, plus the caller-owned
// P columns and operator handles.  Levels 0..T-1 (T = NUM_LEVELS-1) each
// carry test vectors and coarsen into the level below; level T is the
// coarsest (no test vectors).
typedef struct {
  int    T;                     // transition levels = NUM_LEVELS - 1
  REAL * tv   [NUM_LEVELS];     // raw test vectors at level d (nv-layout,
                                //   nnv[d] cols)
  REAL * rhs  [NUM_LEVELS];     // num_rhs-wide scratch (level-d field)
  REAL * x    [NUM_LEVELS];     // num_rhs-wide scratch (level-d field)
  long   nslot[NUM_LEVELS];     // level-d single-rhs field length (REALs)
  int    nnv  [NUM_LEVELS];     // level-d coarsening column count
  int    ninv [NUM_LEVELS];     // level-d setup sweeps (DDA per-level
                                //   setup_iter)
  int    blk  [NUM_LEVELS][4];  // blk[d] = level d -> d+1 aggregation (d >= 1)
  int    swept[NUM_LEVELS];     // sweeps finished at level d, for the
                                //   progress line
  int    nswp [NUM_LEVELS];     // sweeps the recursion will run at level d
                                //   in total
  REAL * nv0;                   // fine P columns (caller-owned, == level 0 nv)
  coarse_op * Dc0;              // fine coarse operator (caller-owned)
  REAL ** nvc;                  // nvc[d] = level-d coarse P columns
                                //  (caller-owned, d >= 1)
  REAL * Uk; REAL * Fcsw; int clover;
  const mg_params * p;          // caller-owned; outlives the setup
} boot_ctx;

// Rebuild the hierarchy after an inverse-iteration sweep at level
// `sweep_depth`.  A sweep changes only tv[sweep_depth] (the level's
// interpolation columns P) and hence only the Galerkin operators at levels
// sweep_depth+1..T; levels 0..sweep_depth keep their operators, geometry, and
// warm coarse-SAP caches.  mg_add orthonormalizes nvc in place, so tv[d]
// itself stays the raw vectors.
//
//   sweep_depth == 0 (fine): the fine sweep changes D_1, which propagates
//   through every coarse operator, so tear the whole hierarchy down
//   (mg_setup, inside rebuild_fine) and rebuild it.  rebuild_dc = 1 also
//   re-runs the fine Galerkin probe build -- the one expensive step incurred
//   only here.
//
//   sweep_depth > 0: preserve levels 0..sweep_depth (skip the teardown-all
//   entirely), trim off the changed levels, and rebuild only sweep_depth+1..T
//   from the preserved parent.  This drops the redundant re-derivation of the
//   unchanged upper levels and lets coarse_sap reuse the preserved level's
//   See_inv block inverse (its sap_nthr key still matches, so the warm-up is
//   a no-op).  Bit-identical: nothing the preserved levels depend on has
//   changed, and every level whose operator did change is freed here (so its
//   stale See_inv is never reused).
static void rebuild_nlvl(boot_ctx * c, int sweep_depth) {
  int d0;
  if (sweep_depth == 0) {
    rebuild_fine(c->tv[0], c->nv0, c->Dc0, c->Uk, c->Fcsw, c->clover,
                 /*rebuild_dc*/ 1, c->p);
    d0 = 1;
  } else {
    mg_trim_to(sweep_depth + 1);   // keep levels 0..sweep_depth, drop the rest
    d0 = sweep_depth;
  }
  for (int d = d0; d < c->T; d ++) {
    memcpy(c->nvc[d], c->tv[d], sizeof(REAL) * c->nslot[d] * (long) c->nnv[d]);
    BT_TIME(add_coarse,
            mg_add_coarse_level(c->nvc[d], c->blk[d], c->nnv[d], c->p));
    g_bt.n_add_coarse ++;
  }
}

// DDalphaAMG inv_iter_inv_fcycle (setup_generic.c): `iters` inverse-iteration
// sweeps at level d, each a global GS + zero-guess recursive cycle + hierarchy
// rebuild, interleaved with descents into level d+1.  The descent is
// asymmetric, exactly as DDalphaAMG: at the fine level (d == 0) it runs inside
// the sweep loop after each rebuild with a count that ramps with j (the coarse
// level is refined more as the fine vectors, hence D_1, settle); at
// intermediate levels (d > 0) it runs once after the loop.  At
// NUM_LEVELS == 3 this reduces byte-for-byte to the old two-deep fine/coarse 
// interleave.  Sweeps bootstrap_recurse will run at each depth, counted
// before the real pass so the progress line below can carry a denominator.
// A deeper level is entered several times with a different batch each time
// (the ramp), so its own `iters` is not its total -- only the sum over
// entries is.  This mirrors the control flow of bootstrap_recurse exactly;
// the two must be changed together.
static void boot_count_sweeps(boot_ctx * c, int d, int iters) {
  c->nswp[d] += iters;
  if (d == 0 && d + 1 < c->T)
    for (int j = 0; j < iters; j ++) {
      int m = (int) lround((double) (j + 1) * c->ninv[d + 1] / iters);
      if (m < 1) m = 1;
      boot_count_sweeps(c, d + 1, m);
    }
  else if (d > 0 && d + 1 < c->T) {
    int m = (int) lround((double) c->ninv[d + 1] * iters / c->ninv[d]);
    if (m < 1) m = 1;
    boot_count_sweeps(c, d + 1, m);
  }
}

static void bootstrap_recurse(boot_ctx * c, int d, int iters) {
  for (int j = 0; j < iters; j ++) {
    bootstrap_sweep(mg_get_level(d), c->tv[d], c->nslot[d], c->rhs[d], c->x[d],
                    c->Uk, c->Fcsw, c->clover, c->p);
    rebuild_nlvl(c, /*sweep_depth*/ d);
    // Progress, not diagnosis: the rebuild above is seconds or minutes,
    // and this is what says the setup is advancing rather than stuck.
    // Counted per level rather than per entry, so it rises monotonically to
    // the total.  Indented by depth so the nesting is visible at a glance;
    // the tag stays in column 0, as every other bracket tag in the tree does,
    // so the left edge and a ^\[ grep both survive.
    print0("[setup] %*slevel %d: sweep %d/%d\n",
           2 * d, "", d, ++ c->swept[d], c->nswp[d]);
    if (d == 0 && d + 1 < c->T) {
      int m = (int) lround((double) (j + 1) * c->ninv[d + 1] / iters);
      if (m < 1) m = 1;
      bootstrap_recurse(c, d + 1, m);
    }
  }
  if (d > 0 && d + 1 < c->T) {
    int m = (int) lround((double) c->ninv[d + 1] * iters / c->ninv[d]);
    if (m < 1) m = 1;
    bootstrap_recurse(c, d + 1, m);
  }
  if (d == 0)
    print0("bootstrap[%dlvl]: fine sweeps done\n", NUM_LEVELS);
}

void bootstrap_null_vecs_nlvl(REAL * nv0, coarse_op * Dc0, REAL ** nvc,
                              REAL * Uk, REAL * Fcsw, int clover,
                              const mg_params * p) {
  const int init_sweeps = p->setup_init_sweeps;
  boot_ctx c;
  memset(&c, 0, sizeof c);
  c.T = NUM_LEVELS - 1;
  c.nv0 = nv0; c.Dc0 = Dc0; c.nvc = nvc;
  c.Uk = Uk; c.Fcsw = Fcsw; c.clover = clover;
  c.p = p;

  // Per-level schedule.  The sweep counts come from mg_params: the fine level
  // takes setup_sweeps, every level below it coarse_setup_sweeps, so the
  // descent ratio is 1 below the fine level and each level inherits its
  // parent's count.  The geometry arrays stay compile-time and their lengths
  // are pinned to NUM_LEVELS: blkcfg has one entry per coarse transition below
  // L1 (L1->L2 .. L{T-1}->L{T}), so blkcfg[d-1] is level d's aggregation.
  int blkcfg[][4] = BOOTSTRAP_BLK_LVL;
  int cnv[]       = COARSE_NULL_VECS_LVL;
#if NUM_LEVELS >= 3
  _Static_assert(sizeof(blkcfg) / sizeof(blkcfg[0]) == NUM_LEVELS - 2,
                 "BOOTSTRAP_BLK_LVL must have NUM_LEVELS-2 entries");
  _Static_assert(sizeof(cnv) / sizeof(cnv[0]) == NUM_LEVELS - 2,
                 "COARSE_NULL_VECS_LVL must have NUM_LEVELS-2 entries");
#endif
  c.ninv[0] = p->setup_sweeps;
  for (int d = 1; d < c.T; d ++) c.ninv[d] = p->coarse_setup_sweeps;
  // widths: fine coarsens by num_null_vecs; each coarse transition by its own
  // COARSE_NULL_VECS_LVL entry (the runtime-width coarse operator / smoother
  // handle the per-level nvar).  Every entry must be even, a multiple of
  // num_rhs, and fit the MAX_NULL_VECS-sized stack arrays (COARSE_NULL_VECS
  // bounds it).
  c.nnv[0] = num_null_vecs;
  for (int d = 1; d < c.T; d ++) {
    c.nnv[d] = cnv[d - 1];
    if (c.nnv[d] % 2 != 0 || c.nnv[d] % num_rhs != 0
        || c.nnv[d] > COARSE_NULL_VECS) {
      comm_die("bootstrap: COARSE_NULL_VECS_LVL[%d]=%d must be even, a "
               "multiple of num_rhs=%d, and <= COARSE_NULL_VECS=%d",
               d - 1, c.nnv[d], num_rhs, COARSE_NULL_VECS);
    }
  }
  for (int d = 1; d < c.T; d ++)
    for (int k = 0; k < 4; k ++) c.blk[d][k] = blkcfg[d - 1][k];

  print0("bootstrap[%dlvl]: init %d sweeps; per-level ninv/nnv:",
         NUM_LEVELS, init_sweeps);
  for (int d = 0; d < c.T; d ++) print0(" L%d=%d/%d", d, c.ninv[d], c.nnv[d]);
  print0("\n");

  // --- fine level: allocate, random+SAP init, build the 2-level base --------
  c.nslot[0] = NSLOT;
  c.tv [0] = malloc(sizeof(REAL) * NSLOT * (long) c.nnv[0]);
  c.rhs[0] = malloc(sizeof(REAL) * NSLOT * num_rhs);
  c.x  [0] = malloc(sizeof(REAL) * NSLOT * num_rhs);

  bt_reset();
  BT_TIME(init,
          init_smooth_fine(c.tv[0], c.rhs[0], c.x[0], init_sweeps, Uk, Fcsw,
                           clover, p));
  rebuild_fine(c.tv[0], nv0, Dc0, Uk, Fcsw, clover, /*rebuild_dc*/ 1, p);
  print0("[setup] init: %d smoothing sweeps done, fine operator built\n",
         init_sweeps);

  // --- seed + build each coarser level in turn ------------------------------
  // Level d is seeded by restricting level d-1's test vectors through its (now
  // built and orthonormalized) transfer, then mg_add_coarse_level builds level
  // d+1.  Ordering matters: restricting into level d needs level d-1 to
  // already be a built intermediate -- which the previous iteration's
  // mg_add_coarse_level made it.  DDalphaAMG-style seed: the first
  // min(nnv[d-1],nnv[d]) columns are the restricted parent vectors;
  // any surplus (nnv[d] > nnv[d-1]) is random on the child grid.  When any
  // level carries such a random surplus, the coarse test vectors are given an
  // initial coarse-SAP smooth (init_smooth_coarse) before they define the next
  // operator -- DDalphaAMG's interpolation_PRECISION_define.  Gated on a
  // surplus so the uniform (non-increasing) hierarchies stay bit-identical.
  int has_surplus = 0;
  for (int d = 1; d < c.T; d ++) if (c.nnv[d] > c.nnv[d - 1]) has_surplus = 1;
  for (int d = 1; d < c.T; d ++) {
    level_t * lv = mg_get_level(d);            // built by the previous step
    c.nslot[d] = lv->field_len;
    c.tv [d] = malloc(sizeof(REAL) * c.nslot[d] * (long) c.nnv[d]);
    c.rhs[d] = malloc(sizeof(REAL) * c.nslot[d] * num_rhs);
    c.x  [d] = malloc(sizeof(REAL) * c.nslot[d] * num_rhs);

    int nseed = c.nnv[d - 1] < c.nnv[d] ? c.nnv[d - 1] : c.nnv[d];
    if (c.nnv[d] > nseed)     // per-level seed offset; d==1 reproduces the
                              //   3-level pad
      random_init_coarse(c.tv[d], lv, 0x1234ABCDUL + (unsigned long) (d - 1),
                         c.nnv[d]);
    for (int b = 0; b < nseed / num_rhs; b ++) {
      BT_TIME(batchio, gather_block(c.rhs[d - 1], c.tv[d - 1], b,
                                    c.nslot[d - 1], c.nnv[d - 1]));
      if (d == 1)                          // depth-0 restrict (fine P columns)
        BT_TIME(restrict_, apply_R(c.rhs[d], c.rhs[d - 1], nv0));
      else                                 // coarse->coarse restrict
        BT_TIME(restrict_, apply_R_coarse(c.rhs[d], c.rhs[d - 1],
                                          c.nvc[d - 1], mg_get_level(d - 1)));
      BT_TIME(batchio, scatter_block(c.tv[d], c.rhs[d], b, c.nslot[d], c.nnv[d]));
    }

    // The P columns must outlive the bootstrap (they are the registered
    // level->nv), and only here is level d's field size known, so the driver
    // allocates nvc[d] into the caller's array; the caller frees it at
    // teardown.
    c.nvc[d] = malloc(sizeof(REAL) * c.nslot[d] * (long) c.nnv[d]);
    memcpy(c.nvc[d], c.tv[d], sizeof(REAL) * c.nslot[d] * (long) c.nnv[d]);
    // Both call sites must time and count: bt_report's total is a sum of the
    // accumulators, not a measured wall time, so there is no residual row for
    // an untimed call to land in.  An untimed add would vanish from the
    // profile and inflate every other row's percentage.  The count is a
    // separate statement because BT_TIME takes no counter.
    BT_TIME(add_coarse, mg_add_coarse_level(c.nvc[d], c.blk[d], c.nnv[d], p));
    g_bt.n_add_coarse ++;
    print0("[setup] %*slevel %d seeded and built (%d of %d)\n",
           2 * d, "", d, d, c.T - 1);

    // Now that mg_add_coarse_level has promoted level d (its coarse_sap
    // smoother is live), condition the test vectors with an initial coarse
    // smooth; the smoothed tv[d] then seeds level d+1 and is rebuilt into the
    // operators by the first bootstrap sweep.  Skipped for uniform hierarchies
    // (has_surplus == 0) to keep them bit-identical.
    if (has_surplus)
      BT_TIME(init, init_smooth_coarse(lv, c.tv[d], c.nslot[d], c.nnv[d],
                                       c.rhs[d], c.x[d], init_sweeps));
  }

  // --- interleaved inverse-iteration bootstrap ------------------------------
  // boot_ctx is a plain stack struct with its fields assigned one by one,
  // so these two need clearing before the counting pass accumulates into them.
  for (int d = 0; d < NUM_LEVELS; d ++) { c.swept[d] = 0; c.nswp[d] = 0; }
  boot_count_sweeps(&c, 0, c.ninv[0]);
  print0("[setup] bootstrap:");
  for (int d = 0; d < c.T; d ++) print0(" L%d=%d sweeps", d, c.nswp[d]);
  print0("\n");
  bootstrap_recurse(&c, 0, c.ninv[0]);

  bt_report(NUM_LEVELS == 3 ? "3lvl" : "nlvl");

  for (int d = 0; d < c.T; d ++) {
    free(c.tv[d]);
    free(c.rhs[d]);
    free(c.x[d]);
  }
}
