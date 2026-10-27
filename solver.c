// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

//#include "solver.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>
#include "clover_ferm.h"
#include "global_params.h"
#include "utils.h"
#include "solver.h"
#include "comm.h"       // allreduce_sum_real: inline full-field reductions
                        // (Arnoldi GS, residual norms) become rank-global here
#include <complex.h>
#include <omp.h>

// FGMRES_SINGLE_KRYLOV: the outer FGMRES runs a single-precision Krylov ladder.
// The Arnoldi V basis, the unnormalized w, and the Z (preconditioned) basis are
// all stored float, and the Arnoldi matvec is fully single -- mg_D_single (or
// mg_D_hat_single for an eo outer solve), the bridge's stored single operator:
// asm wilson kernel, single gauge/clover streams.  Gram-Schmidt reads the basis
// with promote-on-load (the DOTC/CAXPY macros mix float operands into double
// accumulators), so dots and norms keep double accumulators and the caxpy
// w -= h_j V_j multiplies in double, rounding only on the float store.  Halves
// the V-basis stream, and the float V[m] slot doubles as the preconditioner
// input, so no separately down-converted copy exists.
//
// Double survives where it buys feasibility rather than speed: x, b, and the
// once-per-restart residual r = b - A x (computed in place in Ax, so no double
// scratch vector remains).  Those restart applies are the only double fine
// Dslash applies left in the solve and are the defect-correction anchor -- they
// decide convergence, so a float basis can only cost extra iterations, never
// precision.  This mirrors DDalphaAMG's fgmres_MP (their mixed_precision=2).
//
// The condition is structural, not a knob: exactly the double outer fgmres of a
// mixed build (solver_d.o -- MIXED set, MG_SINGLE unset) straddles the
// double/single seam.  The single twin (MG_SINGLE) and the one-precision builds
// (MIXED=0, whether single or double) have no seam, so they take the plain
// Krylov path at their own REAL and reach the preconditioner through the
// generic solver_info hook.  If a config ever needs fp64 throughout --
// bottom-quark masses where fp32 underflows, or a solve that will not
// converge otherwise -- the fallback is a whole all-double build (MIXED=0
// MG_SINGLE=0), not a partial one: intermediate rungs were removed because
// they multiplied build configurations without ever being the answer to a
// real failure.
#if MIXED && !MG_SINGLE
#define FGMRES_SINGLE_KRYLOV 1
#include "mixed.h"      // mg_precond_single, mg_D_single, the emit-w hooks
#endif

// helper method for timing
REAL time_used (struct timeval start, struct timeval end) {
  return ((REAL)(end.tv_usec-start.tv_usec)/1000000
          + (end.tv_sec - start.tv_sec));
}

// --- fine-grid SAP profiling (see solver.h) --------------------------------
static double g_sap_secs = 0, g_sap_flops = 0;
static long   g_sap_calls = 0;
// Serial MPI halo wall inside red_black_schwarz, split between the
// per-half-sweep exchanges and the emit-w tail's, with window counts.  Each
// window is a blocking exchange during which every core idles, so this is
// exactly the wall a halo byte reduction shrinks.
static double g_sap_halo_sweep_secs = 0, g_sap_halo_tail_secs = 0;
static long   g_sap_halo_sweep_win = 0,  g_sap_halo_tail_win = 0;
// Work that runs inside this call but belongs elsewhere in the solve (see
// sap_fusion_stats_get): modeled flops split back out of g_sap_flops, plus the
// wall of the emit-w tail, the one piece that is a pass of its own.
static double g_sap_flops_fold = 0, g_sap_flops_prolong = 0;
static double g_sap_flops_emitw_insweep = 0, g_sap_flops_emitw_tail = 0;
static double g_sap_flops_arndots = 0, g_sap_tail_secs = 0;
void sap_stats_reset(void) {
  g_sap_secs = 0; g_sap_flops = 0; g_sap_calls = 0;
  g_sap_halo_sweep_secs = 0; g_sap_halo_tail_secs = 0;
  g_sap_halo_sweep_win = 0;  g_sap_halo_tail_win = 0;
  g_sap_flops_fold = 0; g_sap_flops_prolong = 0;
  g_sap_flops_emitw_insweep = 0; g_sap_flops_emitw_tail = 0;
  g_sap_flops_arndots = 0; g_sap_tail_secs = 0;
  wilson_halo_sap_stats_reset();  // the window's pack/MPI/expand breakdown
}
void sap_stats_get(double * secs, long * calls, double * flops) {
  if (secs)  *secs  = g_sap_secs;
  if (calls) *calls = g_sap_calls;
  if (flops) *flops = g_sap_flops;
}
void sap_halo_stats_get(double * sweep_secs, long * sweep_win,
                        double * tail_secs, long * tail_win) {
  if (sweep_secs) *sweep_secs = g_sap_halo_sweep_secs;
  if (sweep_win)  *sweep_win  = g_sap_halo_sweep_win;
  if (tail_secs)  *tail_secs  = g_sap_halo_tail_secs;
  if (tail_win)   *tail_win   = g_sap_halo_tail_win;
}
void sap_fusion_stats_get(sap_fusion_t * p) {
  p->fold_flops          = g_sap_flops_fold;
  p->prolong_flops       = g_sap_flops_prolong;
  p->emitw_insweep_flops = g_sap_flops_emitw_insweep;
  p->emitw_tail_flops    = g_sap_flops_emitw_tail;
  p->dot_flops           = g_sap_flops_arndots;
  p->tail_secs           = g_sap_tail_secs;
}

// --- outer FGMRES profiling (see solver.h) ---------------------------------
static fgmres_profile_t g_fgmres_prof;
void fgmres_get_profile(fgmres_profile_t * p) { *p = g_fgmres_prof; }

// prepare gauge field with APBCs and kappa multiplication.
// U_kappa must be a gauge_alloc_size allocation: the tail gauge ghosts are
// exchanged here, after the APBC/kappa folding, so they carry the phases (the
// t-lo ghost of the coords[3]==0 rank comes from the global-Lt-1 owner).
void prepare_gauge(REAL * U_kappa, REAL * U, REAL kappa) {
  copy_gauge(U_kappa, U);
  // apply APBCs to copy -- on the global final t-slice only, which is the
  // local final slice of the ranks owning the top of the t grid
  if (qc_grid.coords[3] == qc_grid.P[3] - 1) {
    #pragma omp parallel for
    for (int z = 0; z < Lz; z ++)
      for (int y = 0; y < Ly; y ++)
        for (int x = 0; x < Lx; x ++) {
          int t = Lt - 1; // final link in time direction
          int dir = 3; // in time direction
          int gauge_loc = site_full(x, y, z, t);
          int U_index = (gauge_loc * 4 + dir) * Nc * Nc * 2;
          for (int i = 0; i < 2 * Nc * Nc; i ++)
            U_kappa[U_index + i] *= -1;
        }
  }
  // multiply in kappa to U links
  #pragma omp parallel for
  for (int i = 0; i < gauge_size; i ++)
    U_kappa[i] *= kappa;
  halo_gauge(U_kappa, sizeof(REAL) * 4 * Nc * Nc * 2);
}

// Form the inverse of one site's D_oo block, (1 - A)^{-1}, where A is the raw
// clover term stored (already scaled by kappa*csw/8) at site src_loc of src
// in the compressed Hermitian layout clover_kernel reads, and write it back
// in the same layout at site dst_loc of dst.  Used both by prepare_F (in
// place, for the global eo path) and by red_black_schwarz (out of place, to
// build a separate D_oo^{-1} field when eo_precondition=0).  src==dst /
// src_loc==dst_loc is safe: each 6x6 block is fully reconstructed into
// F_block before dst is written.
static void invert_clover_site(REAL * dst, long dst_loc,
                               REAL * src, long src_loc) {
  complex REAL F_block[6 * 6];
  for (int block = 0; block < 2; block ++) {
    // reconstruct block from compressed form
    for (int a = 0; a < 6; a ++) {
      for (int b = 0; b <= a; b ++) {
        F_block[6 * b + a]
          = src[((src_loc * 2 + block) * 6 + b) * 6 + a]
           + 1j * src[((src_loc * 2 + block) * 6 + a) * 6 + b];
        F_block[6 * b + a] *= -1; // D = 1 - k H
        F_block[6 * a + b] = conj(F_block[6 * b + a]);
        // diagonal should be purely real
        if (a == b)
          F_block[6 * a + b] = creal(F_block[6 * a + b]);
      }
    }
    // add 1 to diagonals
    for (int a = 0; a < 6; a ++)
      F_block[7 * a] += 1;
    // Invert the matrix.  F_block is a dense complex 6x6 in re/im-interleaved
    // (== complex REAL) layout, exactly what invert_dense_block reads/writes,
    // so we invert it in place via a cast -- no MKL/LAPACK dependency.
    invert_dense_block((const REAL *) F_block, (REAL *) F_block, 6);
    // store this back into dst
    for (int a = 0; a < 6; a ++) {
      for (int b = 0; b <= a; b ++) {
        // do imaginary first so it gets overridden on diagonal
        dst[((dst_loc * 2 + block) * 6 + a) * 6 + b]
          = cimag(F_block[6 * b + a]);
        dst[((dst_loc * 2 + block) * 6 + b) * 6 + a]
          = creal(F_block[6 * b + a]);
      }
    }
  }
}

// Build a standalone D_oo^{-1} field (size Vh*2*6*6, indexed by within-odd-half
// site) from an already-prepared F_csw (raw clover, eo layout: odd half starts
// at site Vh).  Lets red_black_schwarz do its block-internal odd-even MR even
// though solve() prepared F_csw with invert_odd=0.
static void build_Doo_inv(REAL * F_oo_inv, REAL * F_csw) {
  #pragma omp parallel for
  for (long wh = 0; wh < Vh; wh ++)
    invert_clover_site(F_oo_inv, wh, F_csw, (long) Vh + wh);
}

// Cache for the D_oo^{-1} field consumed by red_black_schwarz.  The SAP
// smoother needs it on every application, but F_csw is fixed for the lifetime
// of a solve (the mixed path's single g_Fcsw_f is stable from
// mg_hierarchy_build to teardown), so build it once, keyed on the F_csw
// pointer (+Vh), and rebuild only on a key change.  sap_free_doo_inv_cache()
// clears it when the owning F_csw is freed (solve() for the double path,
// mg_hierarchy_free() for the single path) so a reused malloc address can
// never produce a stale-content hit.
static REAL * sap_Foo_inv     = NULL;
static REAL * sap_Foo_inv_key = NULL;   // the F_csw the cache was built from
static long   sap_Foo_inv_Vh  = 0;

static REAL * get_Doo_inv(REAL * F_csw) {
  if (sap_Foo_inv == NULL || sap_Foo_inv_key != F_csw || sap_Foo_inv_Vh != Vh) {
    free(sap_Foo_inv);             // free(NULL) is a no-op on the first build
    sap_Foo_inv     = malloc(sizeof(REAL) * (long) Vh * 2 * 6 * 6);
    build_Doo_inv(sap_Foo_inv, F_csw);
    sap_Foo_inv_key = F_csw;
    sap_Foo_inv_Vh  = Vh;
  }
  return sap_Foo_inv;
}

// Cache for F_minus_I = A - 1 (the clover minus the identity, both parity
// halves in F_csw's eo layout: even at site 0, odd at site Vh).  The even half
// is consumed by red_black_schwarz's block MR step (2): with the overwrite
// clover kernel it forms -z_e = (A_ee - 1) r_e + wilson_eo(w_o) in one apply,
// with no "temp = -r_e" init copy.  The odd half serves the residual-fold seed
// the same way (r_o's diagonal term (A_oo - 1) x_o in one overwrite apply).
// Subtracting the identity only touches the 12 real diagonal entries per site
// (2 chiral blocks x 6).  Same key/lifetime discipline as the D_oo^{-1} cache
// above (F_csw is fixed per solve; freed together below).
static REAL * sap_Fmi     = NULL;
static REAL * sap_Fmi_key = NULL;   // the F_csw the cache was built from
static long   sap_Fmi_Vh  = 0;

static REAL * get_F_minus_I(REAL * F_csw) {
  if (sap_Fmi == NULL || sap_Fmi_key != F_csw || sap_Fmi_Vh != Vh) {
    free(sap_Fmi);                 // free(NULL) is a no-op on the first build
    long fsite = 2 * 6 * 6;
    sap_Fmi = malloc(sizeof(REAL) * (long) 2 * Vh * fsite);
    #pragma omp parallel for
    for (long wh = 0; wh < 2 * Vh; wh ++) {
      for (long i = 0; i < fsite; i ++)
        sap_Fmi[wh * fsite + i] = F_csw[wh * fsite + i];
      for (int blk = 0; blk < 2; blk ++)
        for (int a = 0; a < 6; a ++)
          sap_Fmi[wh * fsite + (blk * 6 + a) * 6 + a] -= 1.0;
    }
    sap_Fmi_key = F_csw;
    sap_Fmi_Vh  = Vh;
  }
  return sap_Fmi;
}

// Frees both per-solve SAP operator caches (D_oo^{-1} and F_minus_I).
void sap_free_doo_inv_cache(void) {
  free(sap_Foo_inv);
  sap_Foo_inv = NULL; sap_Foo_inv_key = NULL; sap_Foo_inv_Vh = 0;
  free(sap_Fmi);
  sap_Fmi = NULL; sap_Fmi_key = NULL; sap_Fmi_Vh = 0;
}

// This function multiplies F by kappa * csw / 8
// This also rearranges sites so all even sites are first
// invert_odd: controls whether we replace odd sites with inverse
// Note that this is inverse of D, not of H, so we must add identity
// matrix to odd sites (but not even ones) before inverting
void prepare_F(REAL * F_csw, REAL * F, REAL kappa, REAL csw, int invert_odd) {
  int site_size = 2 * 6 * 6;
  // multiply in kappa * csw to F
  for (int parity = 0; parity < 2; parity ++)
    #pragma omp parallel for collapse(2)
    for (int t = 0; t < Lt; t ++)
      for (int z = 0; z < Lz; z ++)
        for (int y = 0; y < Ly; y ++)
          for (int x = 0; x < Lxh; x ++) {
            long loc = (long) parity * Vh + site_half(x, y, z, t);
            int x_par = (parity ^ y ^ z ^ t) & 0x1;
            int full_x = 2 * x + x_par;
            int original_loc = site_full(full_x, y, z, t);
            for (int i = 0; i < site_size; i ++)
              F_csw[loc * site_size + i] 
                = F[original_loc * site_size + i] * kappa * csw / 8.0;
            if (parity && invert_odd)
              invert_clover_site(F_csw, loc, F_csw, loc);
          }
}

// code for various fermion solvers
// chroma_norm is a boolean to tell us if we should multiply results by 2*kappa
// This is needed if we want to match chroma's normalization
//
// solve() is the double-only orchestrator and is not symbol-mangled (see
// mangle.h).  When solver.c is compiled twice for the mixed build, the single
// twin's rule defines SOLVE_OMIT to suppress this definition, so the plain
// `solve` symbol is emitted exactly once -- by the double instantiation.  A
// one-precision build compiles solver.c once, without SOLVE_OMIT.
#ifndef SOLVE_OMIT
void solve(REAL * output, REAL * ferm, REAL * U, REAL * F,
           REAL kappa, REAL csw,
           int chroma_norm, int eo_precondition,
           solver_info solver) {
  // make a copy of the gauge field
  int clover = (csw != 0);
  long clover_size = Lt * Lz * Ly * Lx * 2 * 6 * 6;
  long multi_ferm_size = ((long) half_ferm_size) * num_rhs;
  REAL * U_kappa = malloc(sizeof(REAL) * gauge_alloc_size);
  REAL * F_csw = malloc(sizeof(REAL) * clover_size);
  prepare_gauge(U_kappa, U, kappa);
  if (clover)
    prepare_F(F_csw, F, kappa, csw, eo_precondition);
  REAL * ferm_eo;
  if (eo_precondition) {
    ferm_eo = malloc(sizeof(REAL) * multi_ferm_size);
    for (long i = 0; i < multi_ferm_size; i ++)
      output[i + multi_ferm_size] = 0;
    // setup for eo-preconditioning
    copy_multi_half_ferm(ferm_eo, ferm, num_rhs);
    if (clover) {
      D_oo_inv(output + (long) half_ferm_size * num_rhs, 
               ferm + (long) half_ferm_size * num_rhs, 
               F_csw + clover_size / 2);
      H_eo(ferm_eo, output + (long) half_ferm_size * num_rhs, U_kappa);
    } else {
      H_eo(ferm_eo, ferm + (long) half_ferm_size * num_rhs, U_kappa);
    }
  } else {
    ferm_eo = ferm;
  }
  // call actual solver

  solver.solver(output, ferm_eo, U_kappa, F_csw, clover,
                solver.max_iter, solver.eps, eo_precondition,
                solver.preconditioner);
  // reconstruct output from half fermion
  if (eo_precondition) {
    // output[odd] = ferm[odd] + kappa * H_oe(output[even])
    long oddh = (long) ferm_size * num_rhs / 2;
    #pragma omp parallel for
    for (long i = oddh; i < oddh * 2; i ++)
      output[i] = ferm[i];
    H_oe(output + (long) half_ferm_size * num_rhs, output, U_kappa);
    if (csw != 0)
      D_oo_inv(output + (long) half_ferm_size * num_rhs,
               output + (long) half_ferm_size * num_rhs,
               F_csw + clover_size / 2);
  }
  // rescale outputs by 2*kappa if we need to match chroma's conventions
  if (chroma_norm)
    rescale_multi_ferm(output, 2 * kappa, num_rhs);
  // free malloc'd memory
  free(U_kappa);
  free(F_csw);
  // F_csw just died; drop any SAP D_oo^{-1} built from it
  sap_free_doo_inv_cache();
  if (eo_precondition) free(ferm_eo);
}
#endif // SOLVE_OMIT

int richardson(REAL * output, REAL * ferm, REAL * U_kappa, REAL * F_csw,
              int clover, int max_iter, REAL eps,
              int eo_precondition, solver_info * preconditioner) {
  struct timeval start, end;
  struct timeval hop_start, hop_end;
  // Allocated and read only when a preconditioner is supplied.
  REAL * preconditioner_input = NULL, * preconditioner_output = NULL;
  REAL hop_time = 0;
  gettimeofday(&start, NULL);
  int num_half_ferm = num_rhs;
  if (!eo_precondition) num_half_ferm *= 2;
  long multi_ferm_size = ((long) half_ferm_size) * num_half_ferm;
  int site_size = Nc * Ns * num_rhs * 2;
  REAL * temp = malloc(sizeof(REAL) * multi_ferm_size);
  REAL * H_input, * H_output;
  H_input = output;
  if (preconditioner != NULL) {
    preconditioner_input  = malloc(sizeof(REAL) * multi_ferm_size);
    preconditioner_output = malloc(sizeof(REAL) * multi_ferm_size);
  }
  H_output = preconditioner ? preconditioner_input
                : (eo_precondition ? output : temp);
  // Per-rhs convergence, like fgmres below: each rhs is gated against its own
  // ||b_k||, so no lane can be masked by the others.  A stacked test can pass
  // while one hard rhs is still above eps, because the converged lanes dilute
  // the sum.  Lanes with ||b_k|| == 0 are skipped in the test -- a zero rhs is
  // solved at every step and its ratio is 0/0.
  REAL initial_norm[num_rhs];
  multi_half_ferm_norm_rhs(ferm, num_half_ferm, initial_norm);
  copy_multi_half_ferm(H_input, ferm, num_half_ferm);
  int n;
  int inner_iters = 0;
  int converged = 0;           // reached the tolerance, as opposed to the cap
  gettimeofday(&hop_start, NULL);
  for (n = 1; n <= max_iter; n ++) {
    REAL norm[num_rhs];
    for (int k = 0; k < num_rhs; k ++) norm[k] = 0;
    if (eo_precondition) {
      // temporary on odd sites
      REAL * odd_ferm = temp;
      wilson_halo_half(output, 0);   // odd results read the even iterate
      BLOCK_LOOPS_HALF {
        // zero odd sites
        long loc = site_half(x, y, z, t);
        for (int i = 0; i < Nc * Ns * num_rhs * 2; i ++)
          odd_ferm[loc * Nc * Ns * num_rhs * 2 + i] = 0;
        // H_oe
        wilson_kernel(odd_ferm, output, U_kappa, x, y, z, t, 1);
        // D_oo^{-1}
        if (clover)
          clover_kernel(odd_ferm, odd_ferm, F_csw + Vh * 2 * 6 * 6, 
                        x, y, z, t);
      }
      wilson_halo_half(odd_ferm, 1);   // even results read the odd temp
      BLOCK_LOOPS_HALF_REDUCTION_RHS {
        long loc = site_half(x, y, z, t);
        REAL * site_input = NULL;  // used only when there is no preconditioner
        if (preconditioner) {
          for (int i = 0; i < site_size; i ++)
            H_output[loc * site_size + i] = -output[loc * site_size + i];
        } else {
          site_input = malloc(sizeof(REAL) * site_size);
          for (int i = 0; i < site_size; i ++)
            site_input[i] = output[loc * site_size + i];
        }
        // H_ee
        if (clover)
          clover_kernel(H_output, output, F_csw, x, y, z, t);
        else if (!preconditioner) {
          for (int i = 0; i < site_size; i ++)
            H_output[loc * site_size + i] = 0;
        }
        // add ferm into output
        for (int i = 0; i < site_size; i ++) {
          long index = loc * site_size + i;
          H_output[index] += ferm[index];
        }
        // H_eo
        wilson_kernel(H_output, odd_ferm, U_kappa, x, y, z, t, 0);
        if (!preconditioner) {
          // compute norm.  rhs is the fastest axis, so a group of num_rhs
          // consecutive reals holds one value from each lane and the
          // accumulator stays lane-separated with no horizontal fold.
          for (int g = 0; g < site_size / num_rhs; g ++)
            for (int k = 0; k < num_rhs; k ++) {
              REAL r = H_output[loc * site_size + g * num_rhs + k]
                     - site_input[g * num_rhs + k];
              norm[k] += r * r;
            }
        }
        if (!preconditioner) free(site_input);
      }
      if (preconditioner) {
        // apply preconditioner
        inner_iters += preconditioner->solver(
                          preconditioner_output, 
                          preconditioner_input,
                          U_kappa, F_csw, clover, 
                          preconditioner->max_iter, 
                          preconditioner->eps,
                          eo_precondition, NULL);
        // compute norm of preconditioner output and add to output
        #pragma omp parallel for reduction(+:norm[:num_rhs])
        for (long g = 0; g < multi_ferm_size / num_rhs; g ++)
          for (int k = 0; k < num_rhs; k ++) {
            long i = g * num_rhs + k;
            norm[k] += preconditioner_output[i] * preconditioner_output[i];
            output[i] += preconditioner_output[i];
          }
      }
    } else {
      wilson_halo_half(H_input, 0);
      wilson_halo_half(H_input + (long) ferm_size * num_rhs / 2, 1);
      BLOCK_LOOPS_REDUCTION_RHS {
        int site_size = Nc * Ns * num_rhs * 2;
        // copy ferm into output
        int parity = (x ^ y ^ z ^ t) & 1;
        long loc = (long) parity * Vh + site_half(x / 2, y, z, t);
        if (preconditioner) {
          for (int i = 0; i < site_size; i ++)
            H_output[loc * site_size + i] = -H_input[loc * site_size + i];
        } else {
          for (int i = 0; i < site_size; i ++) {
            long index = loc * site_size + i;
            H_output[index] = ferm[index];
          }
        }
        wilson_kernel(H_output + (long) parity * ferm_size * num_rhs / 2, 
                      H_input + (long) (1 - parity) * ferm_size * num_rhs / 2, 
                      U_kappa, x/2, y, z, t, parity);
        if (clover)
          clover_kernel(H_output + (long) parity * ferm_size * num_rhs / 2, 
                        H_input + (long) parity * ferm_size * num_rhs / 2, 
                        F_csw + parity * Vh * 2 * 6 * 6,
                        x/2, y, z, t);
        if (!preconditioner) {
          for (int g = 0; g < site_size / num_rhs; g ++)
            for (int k = 0; k < num_rhs; k ++) {
              long index = loc * site_size + g * num_rhs + k;
              REAL r = H_output[index] - H_input[index];
              norm[k] += r * r;
            }
        } else {
          // add ferm into output
          for (int i = 0; i < site_size; i ++) {
            long index = loc * site_size + i;
            H_output[index] += ferm[index];
          }
        }
      }
      if (preconditioner) {
        // apply preconditioner
        inner_iters += preconditioner->solver(
                          preconditioner_output, 
                          preconditioner_input,
                          U_kappa, F_csw, clover, 
                          preconditioner->max_iter, 
                          preconditioner->eps,
                          eo_precondition, NULL);
        // compute norm of preconditioner output and add to output
        #pragma omp parallel for reduction(+:norm[:num_rhs])
        for (long g = 0; g < multi_ferm_size / num_rhs; g ++)
          for (int k = 0; k < num_rhs; k ++) {
            long i = g * num_rhs + k;
            norm[k] += preconditioner_output[i] * preconditioner_output[i];
            output[i] += preconditioner_output[i];
          }
      } else {
        // swap H_input and H_output
        REAL * temp_local = H_input;
        H_input = H_output;
        H_output = temp_local;
      }
    }
    allreduce_sum_real(norm, num_rhs);
    // Every active lane must pass.  Squared on both sides to avoid 2*num_rhs
    // sqrts.  Written as the negation of the converged condition so a
    // non-finite lane fails rather than passes; -ffast-math lets the compiler
    // assume that case away, so this is the safe form rather than a guarantee.
    int all_converged = 1;
    for (int k = 0; k < num_rhs; k ++)
      if (initial_norm[k] > 0.0 && !(norm[k] < eps * eps * initial_norm[k]))
        all_converged = 0;
    if (all_converged) { converged = 1; break; }
  }
  gettimeofday(&hop_end, NULL);
  hop_time += time_used(hop_start, hop_end);
  // copy H_output to output if needed
  if (!eo_precondition && (H_input != output))
    copy_multi_ferm(output, H_input, num_rhs);
  if (preconditioner != NULL) {
    free(preconditioner_input);
    free(preconditioner_output);
  }
  gettimeofday(&end, NULL);
  // The loop counter runs one past the cap when the cap is what stopped it,
  // so both the verdict and the iteration count come from the flag rather
  // than from n.
  const int rich_iters = converged ? n : max_iter;
  print0("Richardson expansion %s with %d outer iterations\n",
         converged ? "converged" : "stopped at its iteration cap", rich_iters);
  if (preconditioner)
    print0("Inner solver used %d iterations\n", inner_iters);
  print0("Time used: %f sec\n", time_used(start, end));
  print0("Time used per outer iteration: %f sec\n", hop_time / rich_iters);
  long flops_per_site = (clover) ? WILSON_CLOVER_FLOPS : WILSON_FLOPS;
  long inner_flops_per_site = flops_per_site;
  // add flops for taking norm
  flops_per_site += 3 * 12 * 2;
  print0("Speed of Richardson: %f GFlops/s\n",
         (flops_per_site * rich_iters + inner_flops_per_site * inner_iters)
          * V * num_rhs / hop_time / 1e9);
  free(temp);
  return n;
}

int bicgstab(REAL * output, REAL * ferm, REAL * U_kappa, REAL * F_csw,
             int clover, int max_iter, REAL eps,
             int eo_precondition, solver_info * preconditioner) {
  struct timeval start, end;
  struct timeval hop_start, hop_end;
  REAL hop_time = 0;
  gettimeofday(&start, NULL);
  int num_half_ferm = num_rhs;
  if (!eo_precondition) num_half_ferm *= 2;
  long multi_ferm_size = ((long) half_ferm_size) * num_half_ferm;
  // temporary for computing H_hat
  REAL * odd_ferm;
  if (eo_precondition)
    odd_ferm = malloc(sizeof(REAL) * multi_ferm_size);
  // match notation on 
  // https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
  REAL * x = output; // also h
  zero_multi_half_ferm(x, num_half_ferm);
  REAL * r = malloc(sizeof(REAL) * multi_ferm_size); // also s
  REAL * r0 = ferm;
  REAL * nu = malloc(sizeof(REAL) * multi_ferm_size);
  REAL * p = malloc(sizeof(REAL) * multi_ferm_size);
  REAL * t = malloc(sizeof(REAL) * multi_ferm_size);
  // Every rhs lane is its own BiCGStab: its own coefficients and its own
  // stopping test, sharing only the matvecs.  A lane stops updating once it
  // has converged, and an all-zero lane never starts.
  REAL bnorm[num_rhs];
  for (int k = 0; k < num_rhs; k ++) bnorm[k] = 0;
  #pragma omp parallel for reduction(+:bnorm[:num_rhs])
  for (long i = 0; i < multi_ferm_size; i ++) {
    r[i] = p[i] = ferm[i];
    bnorm[i % num_rhs] += ferm[i] * ferm[i];
  }
  allreduce_sum_real(bnorm, num_rhs);
  complex REAL rho[num_rhs], alpha[num_rhs], omega[num_rhs], beta[num_rhs];
  int done[num_rhs], stop[num_rhs];
  for (int k = 0; k < num_rhs; k ++) {
    rho[k] = bnorm[k];                 // (r0, r) with r = r0
    alpha[k] = omega[k] = beta[k] = 0;
    done[k] = !(bnorm[k] > 0);
  }
  const long ngroup = multi_ferm_size / (2 * num_rhs);
  int n = 0;
  int converged = 0;           // reached the tolerance, as opposed to the cap
  while (n < max_iter) {
    int all_done = 1;
    for (int k = 0; k < num_rhs; k ++) all_done &= done[k];
    if (all_done) { converged = 1; break; }
    n ++;
    gettimeofday(&hop_start, NULL);
    if (eo_precondition)
      D_hat(nu, p, U_kappa, F_csw, odd_ferm, clover);
    else
      D(nu, p, U_kappa, F_csw, clover);
    gettimeofday(&hop_end, NULL);
    hop_time += time_used(hop_start, hop_end);
    REAL dn[2 * num_rhs];              // (r0, nu): re lanes, then im lanes
    for (int k = 0; k < 2 * num_rhs; k ++) dn[k] = 0;
    #pragma omp parallel for reduction(+:dn[:2 * num_rhs])
    for (long i = 0; i < ngroup; i ++) {
      REAL * shifted_r0 = r0 + 2 * i * num_rhs;
      REAL * shifted_nu = nu + 2 * i * num_rhs;
      for (int j = 0; j < num_rhs; j ++)
        DOTC_KERNEL(dn[j], dn[num_rhs + j], shifted_r0, shifted_nu, j);
    }
    allreduce_sum_real(dn, 2 * num_rhs);
    for (int k = 0; k < num_rhs; k ++) {
      if (done[k]) { alpha[k] = 0; continue; }
      const complex REAL denom = dn[k] + 1j * dn[num_rhs + k];
      if (denom == 0.0) {
        print0("Error: BiCGStab has encountered zero norm in denominator "
               "(rhs %d)!\n", k);
        print0("This can happen for unpreconditioned point source solves "
               "with zero clover term.\n");
        print0("Please use a different initial guess.\n");
        assert(0);
      }
      alpha[k] = rho[k] / denom;
    }
    #pragma omp parallel for
    for (long i = 0; i < ngroup; i ++) {
      REAL * shifted_r  = r  + 2 * i * num_rhs;
      REAL * shifted_nu = nu + 2 * i * num_rhs;
      for (int j = 0; j < num_rhs; j ++)
        CAXPY_KERNEL(shifted_r, shifted_nu, -creal(alpha[j]),
                     -cimag(alpha[j]), j);
    }
    gettimeofday(&hop_start, NULL);
    if (eo_precondition)
      D_hat(t, r, U_kappa, F_csw, odd_ferm, clover);
    else
      D(t, r, U_kappa, F_csw, clover);
    gettimeofday(&hop_end, NULL);
    hop_time += time_used(hop_start, hop_end);
    // Per lane: |r|^2, |t|^2, then (r0, r), (r0, t), (t, r) as re/im pairs,
    // each quantity a row of num_rhs lanes, all in one packed allreduce.
    enum { RN, TN, R0R_RE, R0R_IM, R0T_RE, R0T_IM, TR_RE, TR_IM, NRED };
    REAL red[NRED * num_rhs];
    for (int k = 0; k < NRED * num_rhs; k ++) red[k] = 0;
    #pragma omp parallel for reduction(+:red[:NRED * num_rhs])
    for (long i = 0; i < ngroup; i ++) {
      REAL * shifted_r  = r  + 2 * i * num_rhs;
      REAL * shifted_r0 = r0 + 2 * i * num_rhs;
      REAL * shifted_t  = t  + 2 * i * num_rhs;
      for (int j = 0; j < num_rhs; j ++) {
        red[RN * num_rhs + j] += shifted_r[j] * shifted_r[j];
        red[RN * num_rhs + j] += shifted_r[j + num_rhs] * shifted_r[j + num_rhs];
        red[TN * num_rhs + j] += shifted_t[j] * shifted_t[j];
        red[TN * num_rhs + j] += shifted_t[j + num_rhs] * shifted_t[j + num_rhs];
        DOTC_KERNEL(red[R0R_RE * num_rhs + j], red[R0R_IM * num_rhs + j],
                    shifted_r0, shifted_r, j);
        DOTC_KERNEL(red[R0T_RE * num_rhs + j], red[R0T_IM * num_rhs + j],
                    shifted_r0, shifted_t, j);
        DOTC_KERNEL(red[TR_RE * num_rhs + j], red[TR_IM * num_rhs + j],
                    shifted_t, shifted_r, j);
      }
    }
    allreduce_sum_real(red, NRED * num_rhs);
    for (int k = 0; k < num_rhs; k ++) {
      stop[k] = 0;
      if (done[k]) { omega[k] = beta[k] = 0; continue; }
      const REAL r_norm = red[RN * num_rhs + k], t_norm = red[TN * num_rhs + k];
      const complex REAL t_dot_r  = red[TR_RE * num_rhs + k]
                                  + 1j * red[TR_IM * num_rhs + k];
      const complex REAL r0_dot_r = red[R0R_RE * num_rhs + k]
                                  + 1j * red[R0R_IM * num_rhs + k];
      const complex REAL r0_dot_t = red[R0T_RE * num_rhs + k]
                                  + 1j * red[R0T_IM * num_rhs + k];
      // t = A r vanishes only with r, which leaves nothing to minimize
      omega[k] = (t_norm > 0) ? t_dot_r / t_norm : 0;
      // new norm of r (after r updated as below):
      // |r - omega t|^2 = |r|^2 + |omega|^2 |t|^2 - 2 Re(conj(omega) (t, r))
      const REAL norm = r_norm + cabs(omega[k] * omega[k]) * t_norm
                        - 2 * creal(conj(omega[k]) * t_dot_r);
      stop[k] = sqrt(norm) < eps * sqrt(bnorm[k]);
      if (stop[k]) { beta[k] = 0; continue; }
      // new product of r and r0
      const complex REAL rho_new = r0_dot_r - omega[k] * r0_dot_t;
      beta[k] = (rho_new / rho[k]) * (alpha[k] / omega[k]);
      rho[k] = rho_new;
    }
    // update to x, r, and p
    #pragma omp parallel for
    for (long i = 0; i < ngroup; i ++) {
      REAL * shifted_x = x + 2 * i * num_rhs;
      REAL * shifted_p = p + 2 * i * num_rhs;
      REAL * shifted_r = r + 2 * i * num_rhs;
      REAL * shifted_t = t + 2 * i * num_rhs;
      REAL * shifted_nu = nu + 2 * i * num_rhs;
      REAL temp[2 * num_rhs];
      memcpy(temp, shifted_p, sizeof temp);   // p before this step
      for (int j = 0; j < num_rhs; j ++) {
        if (done[j]) continue;
        CAXPY_KERNEL(shifted_x, shifted_p, creal(alpha[j]), cimag(alpha[j]), j);
        CAXPY_KERNEL(shifted_x, shifted_r, creal(omega[j]), cimag(omega[j]), j);
        // a lane that has just converged keeps its last r and p
        if (stop[j]) continue;
        CAXPY_KERNEL(shifted_r, shifted_t, -creal(omega[j]), -cimag(omega[j]),
                     j);
        // p = r + beta (p - omega nu)
        shifted_p[j] = shifted_r[j];
        shifted_p[j + num_rhs] = shifted_r[j + num_rhs];
        CAXPY_KERNEL(shifted_p, temp, creal(beta[j]), cimag(beta[j]), j);
        CAXPY_KERNEL(shifted_p, shifted_nu, -creal(omega[j] * beta[j]),
                     -cimag(omega[j] * beta[j]), j);
      }
    }
    for (int k = 0; k < num_rhs; k ++) done[k] |= stop[k];
  }
  if (!converged) {
    int all_done = 1;
    for (int k = 0; k < num_rhs; k ++) all_done &= done[k];
    converged = all_done;
  }
  free(r);
  free(nu);
  free(p);
  free(t);
  if (eo_precondition) free(odd_ferm);
  gettimeofday(&end, NULL);
  // n stops at max_iter either way, so only the flag separates converged from
  // capped.
  print0("BiCGStab %s with %d terms\n",
         converged ? "converged" : "stopped at its iteration cap", n);
  print0("Time used: %f sec\n", time_used(start, end));
  print0("Time used for hopping: %f sec\n", hop_time);
  print0("Time used per hop: %f sec\n", hop_time / (2 * n));
  long flops_per_site = (clover) ? WILSON_CLOVER_FLOPS : WILSON_FLOPS;
  // 2 calls to D per iteration
  flops_per_site *= 2;
  // other flops used
  int inner_prods = 3; // 8 flops/complex
  int norms = 2;       // 4 flops/complex
  int saxpys = 6;      // 8 flops/complex
  // add flops for taking norm
  flops_per_site += 12 * (inner_prods * 8 + norms * 4 + saxpys * 8);
  print0("Speed of BiCGStab: %f GFlops/s\n",
         (flops_per_site * num_rhs * V * n) / time_used(start, end) / 1e9);
  return n;
}

#if defined(MIXED) && defined(MG_SINGLE)
// Single-precision BiCGStab behind a double-precision interface, so the mixed
// build's outer richardson can run its inner solve at single.  Only the single
// twin defines it, which is what makes the plain names in the body resolve to
// the _f instantiations; it is deliberately absent from mangle.h so the symbol
// keeps its name on both sides of the precision seam, exactly like the
// mixed_bridge entries.  The double twin sees only the prototype and calls it.
//
// prec is void * rather than solver_info *: this TU compiles at single, where
// solver_info's REAL eps field makes the struct a different type than the
// caller's.  bicgstab ignores its preconditioner field regardless.  That is
// also why this cannot sit in a solver_info.solver slot directly -- the driver
// wraps it (test.c).
int bicgstab_mixed(double * out_d, double * in_d, double * U_d, double * F_d,
                   int clover, int max_iter, double eps, int eo_precondition,
                   void * prec) {
  (void) prec;
  const long nferm  = (long) half_ferm_size * (eo_precondition ? num_rhs
                                                               : 2 * num_rhs);
  const long ngauge = gauge_alloc_size;
  const long nclov  = (long) Lt * Lz * Ly * Lx * 2 * 6 * 6;

  float * U_f   = malloc(sizeof(float) * ngauge);
  float * F_f   = malloc(sizeof(float) * nclov);
  float * in_f  = malloc(sizeof(float) * nferm);
  float * out_f = malloc(sizeof(float) * nferm);

  // The operator fields are re-converted on every call rather than cached.
  // The caller is an outer iteration -- a handful of calls, not one per inner
  // step -- so this is a few ms against a solve of minutes, and skipping the
  // cache also skips a teardown hook and the question of when the driver's
  // fields died.  gauge_alloc_size spans the tail ghosts, and prepare_gauge
  // already exchanged them on the double array, so they convert along and
  // need no second exchange.
  #pragma omp parallel for
  for (long i = 0; i < ngauge; i ++) U_f[i] = (float) U_d[i];
  if (clover) {
    #pragma omp parallel for
    for (long i = 0; i < nclov; i ++) F_f[i] = (float) F_d[i];
  }
  // Normalize the rhs on the way in, scale the answer back on the way out.
  // The field elements keep their exponent range fine; it is the solver's
  // scalars that do not.  BiCGStab's are inner products, so they go like
  // ||r||^2, and complex division then forms c^2 + d^2 internally -- an
  // effective floor of ||r||^4.  A caller doing defect correction hands over
  // a defect that shrinks every outer step, and ||r|| ~ 1e-6 already puts
  // that intermediate at ~1e-39, under float's smallest normal and flushed to
  // zero by -ffast-math, so alpha = rho/denom returns inf and the solve turns
  // to NaN.  Measured, not hypothetical.  bicgstab's own `denom == 0` guard
  // cannot see it: denom is a healthy 1e-19 and the underflow is in the
  // square inside the division.  Costs nothing in exactness -- BiCGStab is
  // scale invariant per lane and its stopping test is relative per lane, so
  // no iterate moves.  One scale per rhs (rhs is the fastest index), so a lane
  // far below the others is normalized too.
  double rhs_norm[num_rhs];
  for (int k = 0; k < num_rhs; k ++) rhs_norm[k] = 0;
  #pragma omp parallel for reduction(+:rhs_norm[:num_rhs])
  for (long i = 0; i < nferm; i ++)
    rhs_norm[i % num_rhs] += in_d[i] * in_d[i];
  allreduce_sum_d(rhs_norm, num_rhs);  // one scale per lane, not per rank
  // A zero lane has no direction to normalize; bicgstab leaves it at zero.
  double scale[num_rhs];
  for (int k = 0; k < num_rhs; k ++) {
    rhs_norm[k] = sqrt(rhs_norm[k]);
    scale[k] = (rhs_norm[k] > 0.0) ? 1.0 / rhs_norm[k] : 1.0;
  }

  #pragma omp parallel for
  for (long i = 0; i < nferm; i ++)
    in_f[i] = (float) (in_d[i] * scale[i % num_rhs]);

  int iters = bicgstab(out_f, in_f, U_f, F_f, clover, max_iter, (float) eps,
                       eo_precondition, NULL);

  #pragma omp parallel for
  for (long i = 0; i < nferm; i ++)
    out_d[i] = (double) out_f[i] * rhs_norm[i % num_rhs];

  free(U_f); free(F_f); free(in_f); free(out_f);
  return iters;
}
#endif // MIXED && MG_SINGLE

// Flexible (right-preconditioned) GMRES.  Each Arnoldi step applies the
// preconditioner z = M^{-1} V[m] (M may vary from step to step, e.g. an inexact
// MG V-cycle), forms w = A z, and builds the solution from the preconditioned
// basis Z rather than V:  x = x0 + sum_j z_j y_j,  z_j = M^{-1} v_j.  This is
// the standard outer solver for multigrid (cf. DDalphaAMG FGMRES + MG).  eps is
// relative to ||ferm||; the preconditioner is applied via
// preconditioner->solver(...), so it must be a full-system solver => call
// fgmres with eo_precondition = 0.
//
// preconditioner == NULL (runtime, not a build flag) is plain restarted
// GMRES(max_restart): M = I gives z_j == v_j, so Zbasis aliases Vbasis and
// the solution update reads V.  Kept for unpreconditioned reference runs and
// for heavy quarks, where MG buys little.  On the default single-Krylov build
// this mode still takes its Arnoldi matvec from the bridge's stored single
// operator (mg_D_single / mg_D_hat_single), so mg_hierarchy_build must have
// run to fill g_Uk_f / g_Fcsw_f -- dragging in an MG setup the run never
// uses.  An all-double build (MIXED=0 MG_SINGLE=0) avoids that: plain double
// D / D_hat, no bridge at all.
//
// A fixed preconditioner (e.g. richardson) needs no special case -- FGMRES
// with a constant M produces the same iterates as right-preconditioned
// GMRES, it just also stores Z -- but is honored only where
// FGMRES_SINGLE_KRYLOV is off (the single twin and the one-precision MIXED=0
// builds).  That is why the apply below is #if'd: the single-Krylov branch
// calls mg_precond_single(float *, float *, ...) directly, while .solver is a
// solver_function over REAL * (double in solver_d.o), so bridging them needs
// a pair of conversion scratch fields -- exactly the down-convert the fused
// path exists to delete.
int fgmres(REAL * output, REAL * ferm, REAL * U_kappa, REAL * F_csw,
           int clover, int max_iter, int max_restart, REAL eps,
           int eo_precondition, solver_info * preconditioner) {
  struct timeval start, end;
  struct timeval hop_start, hop_end;
  REAL hop_time = 0;
  // hop_time split: Arnoldi matvecs w = A z (precision set by the
  // FGMRES_SINGLE_KRYLOV) vs once-per-restart residual applies A x (always
  // double).  hop_time remains their sum.
  REAL hop_arn_time = 0, hop_res_time = 0;
  REAL ortho_time = 0;
  REAL prec_time = 0;
  double ortho_flops = 0;       // GS dot/caxpy/norm + solution update
  int num_restarts = 0;
  int inner_iters = 0;
  // Arnoldi matvecs emitted by the preconditioner's terminal fine SAP (the eo
  // free matvec) rather than applied explicitly here.  Excluded from the
  // profile's apply counts so the Arnoldi row's flop charge matches the (near
  // zero) time it measures -- the emitted work is charged in the SAP model.
  int arn_emitted = 0;
  // Arnoldi steps whose GS dots the emit tail computed in the same pass (the
  // fused-dot companion request); the explicit dot pass -- and its full-field
  // w read -- is skipped on those steps, and the dot work is timed in the SAP
  // row (its flops leave ortho_flops so the ortho GF/s stays honest).
  int arn_dots_fused = 0;
  // Outer-Krylov vector work outside the Arnoldi GS block (start-vector
  // scaling, x += Z*y solution update, explicit restart residual) is timed
  // into ortho_time and its flops into ortho_flops -- this is the
  // "Arnoldi / Givens / update" row.
  struct timeval ov0, ov1;
  gettimeofday(&start, NULL);
  int num_half_ferm = num_rhs;
  if (!eo_precondition) num_half_ferm *= 2;
  long multi_ferm_size = ((long) half_ferm_size) * num_half_ferm;

  int restart = (max_iter < max_restart) ? max_iter : max_restart;
  if (restart <= 0) return 0;

#if OUTER_EO && defined(FGMRES_SINGLE_KRYLOV)
  // Fused GS dot partials (rank-local sums), one (re, im) pair per rhs per
  // basis vector, rhs-fastest -- laid out to match hrow so the handoff is a
  // copy.  Declared here rather than at the top of the function because it
  // is sized by the restart length, which is a parameter now.
  double arn_dots[2 * restart * num_rhs];
#endif

  REAL * odd_ferm = NULL;
  if (eo_precondition)
    odd_ferm = malloc(sizeof(REAL) * multi_ferm_size);

  // x = 0  => initial residual r = b - A x = b
  REAL * x = output;
  zero_multi_half_ferm(x, num_half_ferm);

  int ld = restart + 1;
#ifdef FGMRES_SINGLE_KRYLOV
  // Arnoldi V basis in single (see header comment).  The GS loops below read it
  // through the type-generic DOTC/CAXPY macros (promote-on-load, double sums).
  float * Vbasis = malloc(sizeof(float) * multi_ferm_size * ld);
  // Z-basis single as in the fused path (written directly by the single
  // V-cycle).  An eo outer solve carries even-site half-field Krylov vectors,
  // but the V-cycle correction is a full field, so the cycle writing z_j in
  // place at slot j spills its discarded odd half into slot j+1 -- hence one
  // slack slot past the last.  That spill is always harmless: the slots are
  // filled in ascending order, so slot j+1 still holds nothing but a previous
  // spill when slot j is written and gets its own even half on the next
  // iteration, and the solution update below reads only j < m.  Writing in
  // place is what removes the even-half copy the bridge would otherwise make
  // out of a scratch full field.  Unpreconditioned: z_j == v_j, so alias
  // rather than allocate (see the header note).  ld = restart + 1 >= zslots,
  // so every slot the flexible path would write is backed; the spill slot is
  // moot with no V-cycle to spill.
  int zslots = restart + (eo_precondition ? 1 : 0);
  float * Zbasis = preconditioner
                 ? malloc(sizeof(float) * multi_ferm_size * zslots) : Vbasis;
  // The unnormalized Arnoldi vector w is float (mg_D_single output, GS target).
  // The restart residual lives in Ax (computed in place), so no double scratch
  // vector remains at all.
  float * wf = malloc(sizeof(float) * multi_ferm_size);
#else
  REAL * Vbasis = malloc(sizeof(REAL) * multi_ferm_size * ld);
  // Same eo slack slot as the single-Krylov arm above, and for the same
  // reason: with an eo outer solve the Krylov vectors are even-site
  // half-fields, so multi_ferm_size is a half field -- but the cycle's
  // correction is a full one, written in place at slot j, so it spills its
  // discarded odd half into slot j+1.  Without the extra slot the last spill
  // runs off the end of the allocation and corrupts the heap (it shows up as
  // a munmap_chunk() abort in the free below, well after the damage).
  // Harmless otherwise: slots fill in ascending order, so a spill is always
  // overwritten by the next call's own even half, and the solution update
  // reads only j < m.  Unpreconditioned: z_j == v_j, so alias (see the
  // header note).  r aliases Vbasis[0] on this path, but the solution update
  // reads V[0..m-1] before the cycle-end residual overwrites it, so the
  // aliasing is order-safe.
  int zslots = restart + (eo_precondition ? 1 : 0);
  REAL * Zbasis = preconditioner
                ? malloc(sizeof(REAL) * multi_ferm_size * zslots) : Vbasis;
#endif
  // Ax: A applied to the current solution, for the explicit restart residual.
  // Allocated before r because the single-Krylov path aliases r to it.
  REAL * Ax = malloc(sizeof(REAL) * multi_ferm_size);
#ifdef FGMRES_SINGLE_KRYLOV
  REAL * r = Ax;  // restart residual computed in place in Ax (r[i]=b[i]-Ax[i])
#else
  REAL * r = Vbasis;  // first vector in Vbasis
#endif
  copy_multi_half_ferm(r, ferm, num_half_ferm);
  // Stacked norm, used only to detect an all-zero block and bail.  Every
  // convergence test below is per rhs, so nothing else reads it.
  REAL beta = sqrt(multi_half_ferm_norm(r, num_half_ferm));
  if (beta == 0.0) {
    if (odd_ferm) free(odd_ferm);
    // Zbasis aliases Vbasis when unpreconditioned -- free it once.
    free(Vbasis); if (preconditioner) free(Zbasis); free(Ax);
#ifdef FGMRES_SINGLE_KRYLOV
    free(wf);
#endif
    gettimeofday(&end, NULL);
    print0("FGMRES converged with 0 terms\n");
    print0("Time used: %f sec\n", time_used(start, end));
    return 0;
  }

  // Convergence is per rhs: the test that decides the solve is
  // max_k ||r_k||/||b_k|| < eps.  Both places it is applied are per rhs, since
  // each lane carries its own Givens sequence and so its own residual estimate.
  // They differ in which residual they measure, not in how tightly:
  //   in-cycle (max_k |g_k[m+1]|/||b_k|| < eps): the Arnoldi estimate, which
  //     drifts from the true residual as the cycle runs.  It only ends the
  //     cycle.
  //   at the restart boundary: the same test on the true recomputed residual,
  //     using the per-rhs norms the residual apply returns alongside r.
  //     This is what stops the solve.
  // Nothing stops without the boundary test passing, so the guarantee is per
  // rhs and carries no slack.  When the estimate says converged and the true
  // residual disagrees, the cost is one extra residual apply -- one operator
  // apply, cheap beside an outer iteration, which also pays a preconditioner
  // V-cycle.  premature_restarts counts those.
  // Lanes with ||b_k|| == 0 are skipped in the max: a zero rhs is solved at
  // every step, and its ratio is 0/0.  beta > 0 here, so at least one lane
  // survives.
  REAL bnorm_rhs[num_rhs];   // ||b_k||^2 per rhs -- the exact test's
                             //   denominators
  multi_half_ferm_norm_rhs(r, num_half_ferm, bnorm_rhs);
  int nrhs_active = 0;
  for (int k = 0; k < num_rhs; k ++) if (bnorm_rhs[k] > 0.0) nrhs_active ++;
  int  premature_restarts = 0;   // estimate said converged, true residual
                                 //   said no
  REAL worst_rhs = 0;            // last boundary test's value, for the report
  REAL best_rhs  = 0;            // and its best lane, for the spread

  // Per-rhs Krylov.  The matvec is block-diagonal across rhs and every basis
  // operation is applied lane-wise -- only the scalars were ever shared -- so
  // lane k's basis already spans K_m(A, b_k) either way.  Giving each lane
  // its own Hessenberg/Givens/y therefore searches that same space with the
  // "all lanes use one y" constraint removed, so no lane's residual can get
  // worse.  It also makes each lane's least-squares exact: the shared basis
  // is orthonormal in the stacked inner product, not in any one lane's, so a
  // shared Hessenberg was never solving lane k's minimization at all.
  // Lane-slowest for the scalar state (Givens work is O(restart^2),
  // negligible), lane-fastest for hrow, which the field passes vector-load.
  int total_iters = 0;
  complex REAL * Hess = calloc((long) num_rhs * ld * restart,
                               sizeof(complex REAL));
  REAL * cs = malloc(sizeof(REAL) * num_rhs * restart);
  complex REAL * sn = malloc(sizeof(complex REAL) * num_rhs * restart);
  complex REAL * g = calloc((long) num_rhs * (restart + 1),
                            sizeof(complex REAL));
  complex REAL * y = malloc(sizeof(complex REAL) * num_rhs * restart);
  // GS coefficients for the current column: hrow[(2j)*num_rhs+k] = Re h_jm^(k),
  // hrow[(2j+1)*num_rhs+k] = Im.  k fastest, so the dot reduction and the caxpy
  // both address it as a contiguous num_rhs-wide vector.
  REAL * hrow = malloc(sizeof(REAL) * 2 * ld * num_rhs);
  REAL * yrow = malloc(sizeof(REAL) * 2 * ld * num_rhs);  // same layout, for y
  REAL beta_rhs[num_rhs];     // ||r_k|| per lane, refreshed at every boundary
  REAL hmp1[num_rhs], binv[num_rhs], rinv[num_rhs];
  int  frozen[num_rhs];          // lane retired on breakdown this cycle
  for (int k = 0; k < num_rhs; k ++) beta_rhs[k] = sqrt(bnorm_rhs[k]);
  // Relative floor below which V[m+1]'s lane is pure roundoff and normalizing
  // it would divide by a denormal.  Matches fgmres_coarse's GMRES_BREAKDOWN
  // (defined in coarse.c); promote to a shared header if this path ships.
  const REAL fg_breakdown = (REAL) 1e-11;
  int lanes_frozen = 0;          // lanes retired on breakdown, for the report

  // outer restart loop
  while (total_iters < max_iter) {
    num_restarts ++;
    int m = 0;
    int est_converged = 0;      // did the in-cycle estimate end this cycle?
    for (int k = 0; k < num_rhs; k ++) frozen[k] = 0;
    memset(Hess, 0, sizeof(complex REAL) * (long) num_rhs * ld * restart);

    // V0 = r / ||r_k||, per lane.  A lane whose residual has collapsed gets
    // binv = 0, which zeroes its basis lane: from there its dots, its caxpy
    // contributions and its next w are all identically zero, so it stays
    // frozen on its own with no flag to carry.  (The preconditioner couples
    // lanes only through shared scalars, and coefficient * 0 == 0, so a zero
    // lane in maps to a zero lane out -- freezing survives the V-cycle.)
    for (int k = 0; k < num_rhs; k ++)
      binv[k] = beta_rhs[k] > 0.0 ? (REAL) 1.0 / beta_rhs[k] : (REAL) 0.0;
    gettimeofday(&ov0, NULL);
#ifdef FGMRES_SINGLE_KRYLOV
    // r lives in Ax; the basis copy is float-only and gets plain (cached) stores
    // since the GS dots re-read it immediately.  Parallel: this was a serial
    // full-vector pass in the all-double path.
    #pragma omp parallel for
    for (long gi = 0; gi < multi_ferm_size / num_rhs; gi ++)
      for (int k = 0; k < num_rhs; k ++)
        Vbasis[gi * num_rhs + k] = (float) (r[gi * num_rhs + k] * binv[k]);
#else
    for (long gi = 0; gi < multi_ferm_size / num_rhs; gi ++)
      for (int k = 0; k < num_rhs; k ++)
        Vbasis[gi * num_rhs + k] *= binv[k];
#endif
    gettimeofday(&ov1, NULL); ortho_time += time_used(ov0, ov1);
    ortho_flops += (double) multi_ferm_size;        // start-vector scaling
    for (int k = 0; k < num_rhs; k ++) g[(long) k * (restart + 1)] = beta_rhs[k];

    // inner Arnoldi / FGMRES loop (up to restart)
    for (m = 0; m < restart && total_iters + m < max_iter; m ++) {
#ifdef FGMRES_SINGLE_KRYLOV
      // unnormalized Arnoldi vector in float (mg_D_single output); the GS
      // dots/norm still accumulate in double and the caxpy coefficients are
      // double, so float rounding enters only at the w stores.
      float * w = wf;
      float * zs = Zbasis + (long) m * multi_ferm_size;  // single Z slot
#else
      REAL * w = Vbasis + (long) (m + 1) * multi_ferm_size;
      REAL * z = Zbasis + (long) m * multi_ferm_size;
#endif

#if OUTER_EO && defined(FGMRES_SINGLE_KRYLOV)
      // Free Arnoldi matvec: ask the cycle's terminal fine SAP to emit
      // w_e = D_hat z_e from its exit residual (w = eta - r plus the Schur
      // boundary term, assembled on block-resident state), turning the
      // explicit mg_D_hat_single below into a fallback.
      // Unpreconditioned: no cycle runs, so nothing would consume the request --
      // skip it rather than leave a stale one armed for the next call.
      if (eo_precondition && preconditioner)
        mg_request_arnoldi_w(w);
      // Companion request: the emit tail also accumulates the GS dots
      // h_j = conj(V[j]) . w while each finalized w row is cache-hot, so the
      // explicit dot pass below (and its full-field w read) is skipped when
      // this is consumed.  Rank-local sums; the allreduce below still runs.
      // Needs the float basis: the single twin reads V[0..m] directly.
      // arn_dot_row keeps the rhs lanes apart, so the partials it hands back
      // are already a per-rhs Hessenberg column in hrow's layout -- the
      // fusion needed no more than widening its accumulator.  The sibling
      // request above (the free Arnoldi matvec) never had a shared scalar in
      // it at all: w = D_hat z is lane-independent.
      if (eo_precondition && preconditioner)
        mg_request_arnoldi_dots(Vbasis, multi_ferm_size, m + 1, arn_dots);
#endif
      // z = M^{-1} V[m]  (flexible preconditioner; M may vary each step).
      // Unpreconditioned: M = I and Z aliases V, so z_j is already V[m] in
      // place -- nothing to apply, nothing to write, and the matvec below
      // reads it through the same zs / z pointer.
      gettimeofday(&hop_start, NULL);
      if (preconditioner) {
#ifdef FGMRES_SINGLE_KRYLOV
        // The float basis slot V[m] is the single-precision preconditioner
        // input -- no separately down-converted copy exists.  Plain cached
        // stores wrote it (V0-scaling or normalize loop above), so no sfence
        // is needed: the OpenMP barrier at the end of that loop makes the
        // stores visible to the V-cycle's worker threads.  The V-cycle writes
        // its single correction straight into the Z-basis slot zs; the outer
        // matvec reads zs directly (promote-on-load) -- no up-convert.  In
        // the eo case that correction is a full field, so it also spills its
        // discarded odd half into slot m+1 -- see the Z-basis allocation for
        // why that is safe.
        inner_iters += mg_precond_single(
                          zs, Vbasis + (long) m * multi_ferm_size,
                          U_kappa, F_csw, clover,
                          preconditioner->max_iter, preconditioner->eps,
                          eo_precondition, NULL);
#else
        // Generic route: honors .solver, so a fixed preconditioner
        // (richardson, red_black_schwarz, ...) works here.  Only this path
        // has it -- see the type-wall note in the header comment.
        inner_iters += preconditioner->solver(
                          z, Vbasis + (long) m * multi_ferm_size,
                          U_kappa, F_csw, clover,
                          preconditioner->max_iter, preconditioner->eps,
                          eo_precondition, NULL);
#endif
      }
      gettimeofday(&hop_end, NULL);
      prec_time += time_used(hop_start, hop_end);

      // w = A z.  The timer goes inside the same condition as the matvec, not
      // around it: on the emitted path below there is no work to time, and a
      // bracket there would charge the Arnoldi-Dslash row its own gettimeofday
      // overhead -- enough to make the profile print a row for a matvec that
      // never ran, and to subtract that overhead from the "other" row.
      int arn_matvec = 1;
#if OUTER_EO && defined(FGMRES_SINGLE_KRYLOV)
      // w already emitted by the preconditioner's terminal fine SAP; the
      // explicit Schur matvec runs only if that request went unconsumed.
      if (eo_precondition && mg_arnoldi_w_filled()) {
        arn_emitted ++;
        arn_matvec = 0;
      }
#endif
      if (arn_matvec) {
        gettimeofday(&hop_start, NULL);
#ifdef FGMRES_SINGLE_KRYLOV
        // fully-single matvec: the bridge's stored single operator -- asm
        // wilson kernel + single gauge/clover streams.  With an eo outer
        // solve it is the Schur matvec on that same stored single operator.
        if (eo_precondition)
          mg_D_hat_single(w, zs);
        else
          mg_D_single(w, zs);
#else
        if (eo_precondition)
          D_hat(w, z, U_kappa, F_csw, odd_ferm, clover);
        else
          D(w, z, U_kappa, F_csw, clover);
#endif
        gettimeofday(&hop_end, NULL);
        hop_time += time_used(hop_start, hop_end);
        hop_arn_time += time_used(hop_start, hop_end);
      }

      // Arnoldi: orthogonalize w against V[0..m] (classical Gram-Schmidt)
      struct timeval ortho_start, ortho_end;
      gettimeofday(&ortho_start, NULL);
      // Coefficients are per lane.  Flop count and memory traffic are
      // identical to the shared-scalar form: the dot writes num_rhs distinct
      // accumulators instead of folding them into one, dropping a horizontal
      // reduction rather than adding work, and the caxpy vector-loads its
      // coefficient where it used to broadcast a scalar.  rhs is the fastest
      // index, so both are contiguous.
      const int hrn = 2 * (m + 1) * num_rhs;      // live entries of hrow
      int dots_fused = 0;
#if OUTER_EO && defined(FGMRES_SINGLE_KRYLOV)
      // The emit tail already holds the rank-local dot sums (double) in exactly
      // this layout, so a copy stands in for the reduction and the dot pass's
      // full-field w read disappears.  The allreduce is unchanged.
      if (mg_arnoldi_dots_filled()) {
        for (int j = 0; j < hrn; j ++) hrow[j] = arn_dots[j];
        dots_fused = 1;
        arn_dots_fused ++;
      }
#endif
      if (!dots_fused) {
        for (int j = 0; j < hrn; j ++) hrow[j] = 0;
        #pragma omp parallel for reduction(+:hrow[:hrn])
        for (long i = 0; i < multi_ferm_size / (2 * num_rhs); i ++)
          for (int j = 0; j <= m; j ++)
            for (int k = 0; k < num_rhs; k ++)
              DOTC_KERNEL(hrow[(2 * j) * num_rhs + k],
                          hrow[(2 * j + 1) * num_rhs + k],
                          Vbasis + j * multi_ferm_size + 2 * i * num_rhs,
                          w + 2 * i * num_rhs, k);
      }
      allreduce_sum_real(hrow, hrn);
      for (int k = 0; k < num_rhs; k ++) hmp1[k] = 0;
      #pragma omp parallel for reduction(+:hmp1[:num_rhs])
      for (long i = 0; i < multi_ferm_size / (2 * num_rhs); i ++) {
#ifdef FGMRES_SINGLE_KRYLOV
        float * shifted_w = w + 2 * i * num_rhs;
#else
        REAL * shifted_w = w + 2 * i * num_rhs;
#endif
        for (int j = 0; j <= m; j ++)
          for (int k = 0; k < num_rhs; k ++)
            CAXPY_KERNEL(shifted_w,
                         Vbasis + j * multi_ferm_size + 2 * i * num_rhs,
                         -hrow[(2 * j) * num_rhs + k],
                         -hrow[(2 * j + 1) * num_rhs + k], k);
        // (REAL) forces a double product when w is float (no-op when double)
        for (int k = 0; k < num_rhs; k ++)
          hmp1[k] += (REAL) shifted_w[k] * shifted_w[k]
                   + (REAL) shifted_w[num_rhs + k] * shifted_w[num_rhs + k];
      }
      allreduce_sum_real(hmp1, num_rhs);
      for (int k = 0; k < num_rhs; k ++) hmp1[k] = sqrt(hmp1[k]);

      // Apply previous Givens rotations to the new column of H.  (All the
      // Hess/g work through the residual estimate is scalar; it runs before
      // the V[m+1] normalize so the exit decision is known first -- see the
      // normalize below.)
      // One independent Givens sequence per lane.  O(restart^2) scalar work
      // each, so 16x of it is invisible beside the field passes above.
      REAL worst_est = 0;
      for (int k = 0; k < num_rhs; k ++) {
        complex REAL * H  = Hess + (long) k * ld * restart;
        REAL         * ck = cs   + (long) k * restart;
        complex REAL * sk = sn   + (long) k * restart;
        complex REAL * gk = g    + (long) k * (restart + 1);

        // Lane k's new Hessenberg column, gathered from the lane-fastest row.
        for (int i = 0; i <= m; i ++)
          H[i + (long) m * ld] = hrow[(2 * i) * num_rhs + k]
                           + I * hrow[(2 * i + 1) * num_rhs + k];

        // Per-lane breakdown.  This is the failure mode a shared normalization
        // never had: the stacked norm is always well scaled, but an individual
        // lane's orthogonal component can collapse -- exactly what happens once
        // that rhs converges -- and dividing by it fills the basis lane with
        // amplified roundoff.  Test it against the pre-orthogonalization column
        // norm via Pythagoras on the reduced column (no extra global sum), the
        // same criterion fgmres_coarse uses.  rinv = 0 both skips the normalize
        // and zeroes the lane, which retires it (see the V0 comment).
        double wn0sq = (double) hmp1[k] * hmp1[k];
        for (int i = 0; i <= m; i ++)
          wn0sq += (double) hrow[(2 * i) * num_rhs + k]
                          * hrow[(2 * i) * num_rhs + k]
                 + (double) hrow[(2 * i + 1) * num_rhs + k]
                          * hrow[(2 * i + 1) * num_rhs + k];
        const int rhs_converged
          = ((double) hmp1[k] <= fg_breakdown * sqrt(wn0sq));
        rinv[k] = (rhs_converged || hmp1[k] == 0.0) ? (REAL) 0.0
                                                    : (REAL) 1.0 / hmp1[k];
        if (rinv[k] == 0.0 && !frozen[k]) { frozen[k] = 1; lanes_frozen ++; }
        H[(m + 1) + (long) m * ld] = hmp1[k];

        for (int i = 0; i < m; i++) {
          complex REAL temp = ck[i] * H[i + (long) m * ld]
                              + sk[i] * H[i + 1 + (long) m * ld];
          H[i + 1 + (long) m * ld] = -conj(sk[i]) * H[i + (long) m * ld]
                                    + ck[i] * H[i + 1 + (long) m * ld];
          H[i + (long) m * ld] = temp;
        }

        // Generate m-th rotation to eliminate H[m+1,m]
        REAL rho = sqrt(cabs(H[m + (long) m * ld]) * cabs(H[m + (long) m * ld])
                      + cabs(H[m + 1 + (long) m * ld])
                      * cabs(H[m + 1 + (long) m * ld]));
        if (rho == 0.0) {
          ck[m] = 1.0; sk[m] = 0.0;
        } else {
          ck[m] = cabs(H[m + (long) m * ld]) / rho;
          sk[m] = ck[m] * conj(H[m + 1 + (long) m * ld] / H[m + (long) m * ld]);
        }
        H[m + (long) m * ld] = ck[m] * H[m + (long) m * ld]
                              + sk[m] * H[m + 1 + (long) m * ld];
        H[m + 1 + (long) m * ld] = 0.0;

        // Update g
        complex REAL tempg = ck[m] * gk[m];
        gk[m + 1] = -conj(sk[m]) * gk[m];
        gk[m] = tempg;

        // Lane k's own residual estimate against its own ||b_k||.  Splitting
        // the estimate per lane is what the per-rhs Givens buys here: a
        // stacked estimate is one number for the block, attributable to no
        // single rhs.
        if (bnorm_rhs[k] > 0.0) {
          const REAL rel = cabs(gk[m + 1]) / sqrt(bnorm_rhs[k]);
          if (rel > worst_est) worst_est = rel;
        }
      }
      // V[m+1] = w / hmp1 -- skipped when this is the cycle's last iteration
      // (convergence, restart boundary, or the max_iter cap): the inner loop
      // then ends before anything reads V[m+1] -- the GS reads only V[0..m],
      // the solution update reads Z and Hess, and the next cycle rebuilds V0
      // from the fresh restart residual -- so the pass would write a basis
      // vector with no reader (V[restart] in particular is written every full
      // cycle and is structurally unreadable).  Only the scalar hmp1 entry in
      // Hess is live.  Saves one full basis-vector pass per restart cycle.
      int last_iter = (worst_est < eps) || (m + 1 >= restart)
                      || (total_iters + m + 1 >= max_iter);
      // Close the ortho bracket before the normalize: the fused
      // normalize+restrict below times itself into the transfer/restrict rows
      // (it is mostly restriction work), so it must stay out of ortho_time --
      // only the unfused fallback pass is ortho work (bracketed separately).
      gettimeofday(&ortho_end, NULL);
      ortho_time += time_used(ortho_start, ortho_end);
      // GS flops for this Arnoldi step: (m+1) complex dots + (m+1) complex
      // caxpys over multi_ferm_size/2 complex elements (8 flop each) + the
      // hmp1 norm.  Fused-dot steps drop the dots' half (4 of the 8): that
      // work ran inside the emit tail, so the SAP flop model charges it there
      // instead.
      ortho_flops += ((double) (m + 1) * (dots_fused ? 4.0 : 8.0) + 2.0)
                   * multi_ferm_size;

#ifdef FGMRES_SINGLE_KRYLOV
      if (!last_iter) {
        float * vnext = Vbasis + (long) (m + 1) * multi_ferm_size;
        // Fused normalize+restrict: w is float, so the single twin can both
        // emit V[m+1] = w/hmp1 into the basis slot and restrict it to the
        // coarse grid in one cache-resident pass, pre-computing the next
        // V-cycle's depth-0 restriction (consumed one-shot there) -- the
        // restrict's DRAM re-read of V[m+1] disappears.
        // Unpreconditioned: no V-cycle follows, so the pre-computed restriction
        // would never be consumed -- a wasted restrict per Arnoldi step, and a
        // one-shot buffer left armed.  Take the plain normalize below instead.
        // The kernel takes the whole num_rhs block of reciprocals and applies
        // lane k to lane k, so a retired lane's 0 zeroes it there exactly as it
        // does in the plain path below.  Passed as double independent of this
        // twin's REAL: the bridge crosses into the single hierarchy.
        int fused = 0;
        double fuse_rinv[num_rhs];
        for (int k = 0; k < num_rhs; k ++) fuse_rinv[k] = (double) rinv[k];
        if (preconditioner)
          fused = mg_normalize_restrict_single(vnext, w, fuse_rinv,
                                               eo_precondition);
        if (!fused) {
          // Written straight into the float basis slot: w is never written
          // back at all (one read, one half-width write -- the all-double
          // path read-modify-writes w in place).  Cached stores: the GS dots
          // re-read V[m+1] next iteration.  rinv is 0 for a retired lane,
          // which zeroes it.
          gettimeofday(&ov0, NULL);
          #pragma omp parallel for
          for (long gi = 0; gi < multi_ferm_size / num_rhs; gi ++)
            for (int k = 0; k < num_rhs; k ++)
              vnext[gi * num_rhs + k] = (float) (w[gi * num_rhs + k] * rinv[k]);
          gettimeofday(&ov1, NULL); ortho_time += time_used(ov0, ov1);
        }
      }
#else
      if (!last_iter) {
        gettimeofday(&ov0, NULL);
        for (long gi = 0; gi < multi_ferm_size / num_rhs; gi ++)
          for (int k = 0; k < num_rhs; k ++)
            w[gi * num_rhs + k] *= rinv[k];
        gettimeofday(&ov1, NULL); ortho_time += time_used(ov0, ov1);
      }
#endif

      // Convergence history, one line per Arnoldi step.  worst_est is the
      // Givens estimate the exit test below already computed, so the line
      // costs only the write.  It is the estimate and not the true residual:
      // the restart boundary recomputes that one and the two can disagree,
      // so the label says so rather than leaving a reader to wonder why it
      // misses the final report's number.
      print0("  iter %3d  est resid %.3e\n", total_iters + m + 1,
             (double) worst_est);

      // Per rhs: each lane is measured against its own ||b_k||.  The boundary
      // below still re-tests on the true recomputed residual, because the
      // Arnoldi estimate can drift from it.
      if (worst_est < eps) {
        est_converged = 1;
        m ++; // number of Krylov vectors used in this cycle
        break;
      }
    }

    // Solve each lane's own upper triangular system H^(k) y^(k) = g^(k).  A
    // retired lane has a zero diagonal from the row it broke down on, and the
    // existing hii == 0 guard already yields y = 0 there.
    for (int k = 0; k < num_rhs; k ++) {
      complex REAL * H  = Hess + (long) k * ld * restart;
      complex REAL * gk = g    + (long) k * (restart + 1);
      complex REAL * yk = y    + (long) k * restart;
      for (int i = m - 1; i >= 0; i--) {
        complex REAL sum = 0.0;
        for (int j = i + 1; j < m; j++)
          sum += H[i + (long) j * ld] * yk[j];
        complex REAL hii = H[i + (long) i * ld];
        yk[i] = (hii == 0.0) ? 0.0 : (gk[i] - sum) / hii;
      }
    }
    // Lane-fastest copy for the field pass below, so the caxpy vector-loads
    // its coefficient exactly as the GS one does.
    for (int j = 0; j < m; j ++)
      for (int k = 0; k < num_rhs; k ++) {
        yrow[(2 * j) * num_rhs + k]     = creal(y[(long) k * restart + j]);
        yrow[(2 * j + 1) * num_rhs + k] = cimag(y[(long) k * restart + j]);
      }

    // x += Z * y   (preconditioned basis: FGMRES update uses Z, not V)
    gettimeofday(&ov0, NULL);
    #pragma omp parallel for
    for (long i = 0; i < multi_ferm_size / (2 * num_rhs); i ++) {
      REAL * shifted_x = x + 2 * i * num_rhs;
      for (int j = 0; j < m; j ++)
        for (int k = 0; k < num_rhs; k ++)
          CAXPY_KERNEL(shifted_x,
                       Zbasis + j * multi_ferm_size + 2 * i * num_rhs,
                       yrow[(2 * j) * num_rhs + k],
                       yrow[(2 * j + 1) * num_rhs + k], k);
    }
    gettimeofday(&ov1, NULL); ortho_time += time_used(ov0, ov1);
    // m complex caxpys (8 flop/cmpx)
    ortho_flops += 4.0 * (double) m * multi_ferm_size;

    // Update residual r = b - A x and its norm beta.  On the eo path the
    // subtract and the norm ride inside the operator's closing per-site pass
    // (D_hat_residual), which deletes the Ax round trip -- the read-back and
    // the second writeback of a full even half-field.  The eta read is not
    // saved, only relocated; it is the Ax traffic that was pure overhead.
    // Both the time and the corresponding flops therefore move from the ortho
    // row to the restart-D row (res_folded tells the profile's flop model to
    // charge them there).  The norms come back per rhs, which is what both
    // consumers need now: the exact test below, and the next cycle's per-lane
    // V0 normalize and g[0].  Nothing wants the stacked norm any more.
    REAL rn[num_rhs];
    for (int k = 0; k < num_rhs; k ++) rn[k] = 0;
    gettimeofday(&hop_start, NULL);
    if (eo_precondition)
      D_hat_residual(r, ferm, x, U_kappa, F_csw, odd_ferm, clover, rn);
    else
      D(Ax, x, U_kappa, F_csw, clover);
    gettimeofday(&hop_end, NULL);
    hop_time += time_used(hop_start, hop_end);
    hop_res_time += time_used(hop_start, hop_end);
    if (!eo_precondition) {
      gettimeofday(&ov0, NULL);
      // rhs is the fastest index, so the lane is the inner loop and the
      // accumulator is one vector register wide -- no horizontal fold.
      #pragma omp parallel for reduction(+:rn[:num_rhs])
      for (long gi = 0; gi < multi_ferm_size / num_rhs; gi ++)
        for (int k = 0; k < num_rhs; k ++) {
          const long i = gi * num_rhs + k;
          r[i] = ferm[i] - Ax[i];
          rn[k] += r[i] * r[i];
        }
      gettimeofday(&ov1, NULL); ortho_time += time_used(ov0, ov1);
      ortho_flops += 3.0 * (double) multi_ferm_size;  // r=b-Ax (1) + norm (2)
    }
    allreduce_sum_real(rn, num_rhs);
    // Per-lane restart norms for the next cycle's V0 and g[0].
    for (int k = 0; k < num_rhs; k ++) beta_rhs[k] = sqrt(rn[k]);

    total_iters += m;

    // The test that decides the solve: per rhs, on the true residual.  Every
    // active rhs must clear its own eps * ||b_k||; the worst one decides.
    // The best lane comes along for free and is reported as a spread: with
    // each lane on its own polynomial it measures how unequal the sources
    // themselves are, which is exactly how much slack a block-average test
    // would have hidden.
    worst_rhs = 0;
    best_rhs  = 0;
    for (int k = 0; k < num_rhs; k ++) {
      if (bnorm_rhs[k] == 0.0) continue;         // zero rhs: solved, ratio is 0/0
      const REAL rk = sqrt(rn[k] / bnorm_rhs[k]);
      if (rk > worst_rhs) worst_rhs = rk;
      if (best_rhs == 0.0 || rk < best_rhs) best_rhs = rk;
    }

    if (worst_rhs < eps)
      break;
    // Still going: report the true residual the test above used.  Printed
    // only on the cycles that continue -- on the converged one the report
    // below restates the same number a line later.
    print0("  restart %d: true resid %.3e\n", num_restarts, (double) worst_rhs);
    if (est_converged)
      premature_restarts ++;       // estimate overruled by the true residual
    // otherwise continue with next restart cycle (V0 will be built from r)
  }

  if (odd_ferm) free(odd_ferm);
  // Zbasis aliases Vbasis when unpreconditioned -- free it once.
  free(Vbasis); if (preconditioner) free(Zbasis);
#ifdef FGMRES_SINGLE_KRYLOV
  free(wf);
#endif
  free(Hess); free(cs); free(sn); free(g); free(y); free(Ax);
  free(hrow); free(yrow);
  gettimeofday(&end, NULL);

  // worst_rhs is the value the boundary test actually accepted, so it is the
  // honest statement of what was delivered -- not a block average that could
  // hide an under-solved rhs.  premature_restarts is the estimate's whole
  // cost: cycles it ended early that the true residual then overruled.
  // Convergence, not performance: where the time went belongs to
  // mg_profile_report, which breaks the same wall down by kernel.  A caller
  // with no profiler (it is not in qc.h, so the library path has none) can
  // time its own solve call, but cannot reconstruct the residual from outside.
  // print0 throughout: every rank reaches this, only rank 0 should say so.
  print0("%s converged with %d outer iterations "
         "(worst rhs %.3e vs per-rhs target %g, over %d active rhs, "
         "spread %.1fx)\n",
         preconditioner ? "FGMRES" : "GMRES (unpreconditioned)",
         total_iters, (double) worst_rhs, eps, nrhs_active,
         best_rhs > 0 ? (double) (worst_rhs / best_rhs) : 1.0);
  if (premature_restarts)
    print0("  %d restart cycle(s) where the Arnoldi estimate said converged and "
           "the true residual disagreed (one extra residual apply each)\n",
           premature_restarts);
  if (lanes_frozen)
    print0("  %d lane-freeze event(s): a rhs hit the orthogonalization floor and "
           "was retired for the rest of its cycle\n", lanes_frozen);
  if (preconditioner)
    print0("Inner preconditioner used %d total iterations\n", inner_iters);
  if (arn_dots_fused)
    print0("Fused GS dots: %d of %d Arnoldi steps\n",
           arn_dots_fused, total_iters);

  // snapshot for the profiling summary.  One fine A = D apply runs per Arnoldi
  // step (total_iters) plus one residual A apply per restart cycle -- minus
  // the Arnoldi matvecs the terminal fine SAP emitted (arn_emitted), whose
  // work is timed and flop-charged in the SAP row instead.  Fused-dot steps
  // (arn_dots_fused) likewise moved their GS dot time into the SAP row; their
  // dot flops left ortho_flops and are untallied there.
  g_fgmres_prof.total_secs   = time_used(start, end);
  g_fgmres_prof.prec_secs    = prec_time;
  g_fgmres_prof.hop_secs     = hop_time;
  g_fgmres_prof.hop_arn_secs = hop_arn_time;
  g_fgmres_prof.hop_res_secs = hop_res_time;
  g_fgmres_prof.ortho_secs   = ortho_time;
  g_fgmres_prof.ortho_flops  = ortho_flops;
  g_fgmres_prof.d_applies    = (long) total_iters - arn_emitted + num_restarts;
  g_fgmres_prof.arn_applies  = total_iters - arn_emitted;
  g_fgmres_prof.res_applies  = num_restarts;
  g_fgmres_prof.res_folded   = eo_precondition ? 1 : 0;
  // which matvec variant produced the Arnoldi w (for the report's row label)
#ifdef FGMRES_SINGLE_KRYLOV
  g_fgmres_prof.arn_prec     = 1;   // mg_D_single: fully single, asm kernel
#else
  g_fgmres_prof.arn_prec     = 0;   // D / D_hat: double output
#endif
  g_fgmres_prof.outer_iters  = total_iters;
  g_fgmres_prof.inner_iters  = inner_iters;
  return total_iters;
}

// ---- explicit-vector helpers for the block-MR BLAS-1 tail -----------------
// The alpha reduction and the phi/r update are hand-vectorized over the
// num_rhs lanes (REAL_VEC, the transfer-kernel idiom) with
// restrict-qualified site pointers: the compiler cannot prove
// phi/r/temp/output distinct (so it guards its own vectorization with
// runtime aliasing checks), and its reduction leaves the FMA chains too deep
// to reach the throughput floor.  Unaligned loads/stores: r/phi/temp are
// plain malloc (16B-aligned only) and `output` belongs to the caller;
// vmovups on an aligned address costs nothing.
#if AVX512
  #if MG_SINGLE
    #define SAP_LD(p)     _mm512_loadu_ps(p)
    #define SAP_ST(p, v)  _mm512_storeu_ps(p, v)
  #else
    #define SAP_LD(p)     _mm512_loadu_pd(p)
    #define SAP_ST(p, v)  _mm512_storeu_pd(p, v)
  #endif
#elif AVX2
  #if MG_SINGLE
    #define SAP_LD(p)     _mm256_loadu_ps(p)
    #define SAP_ST(p, v)  _mm256_storeu_ps(p, v)
  #else
    #define SAP_LD(p)     _mm256_loadu_pd(p)
    #define SAP_ST(p, v)  _mm256_storeu_pd(p, v)
  #endif
#else
  #define SAP_LD(p)     (*(p))
  #define SAP_ST(p, v)  (*(p) = (v))
#endif

// fused 3-way reduction over one site, per rhs: den += |t|^2, n_re + i n_im
// += <t, r> (t holds -z, which is where the conjugation signs below come
// from).  Each rhs owns its accumulator triple -- they are the numerator and
// denominator of that rhs's own MR step length -- so the lanes are never
// summed together: the caller's REAL_VEC triples are the running totals over
// the block's even sites, one lane per rhs.
// 12 independent accumulator chains (2 banks x {den,nre,nim} x {re,im part})
// keep the FMA chain depth at/below the throughput floor on both AVX2 and
// AVX-512.  The rhs chunk is the outer loop so those chains stay live in
// registers across the whole site; the site is re-read per chunk out of L1.
static inline void sap_blk_dot3(const REAL * restrict t,
                                const REAL * restrict r,
                                REAL_VEC * restrict nre,
                                REAL_VEC * restrict nim,
                                REAL_VEC * restrict den) {
  // fast-math reassociation would collapse the accumulator banks below back
  // into one serial FMA chain per output; pin the source ILP shape.  FMA
  // contraction within each statement is unaffected.
  #pragma clang fp reassociate(off)
  for (int v = 0; v < num_rhs_vec; v ++) {
    // named accumulators: an acc[2][6] array defeats register promotion
    // (every FMA turns into a stack load-FMA-store roundtrip)
    REAL_VEC da0 = ZERO_VEC, da1 = ZERO_VEC, ra0 = ZERO_VEC, ra1 = ZERO_VEC,
             ia0 = ZERO_VEC, ia1 = ZERO_VEC,
             db0 = ZERO_VEC, db1 = ZERO_VEC, rb0 = ZERO_VEC, rb1 = ZERO_VEC,
             ib0 = ZERO_VEC, ib1 = ZERO_VEC;
    for (int j = 0; j < Nc * Ns; j += 2) {
      const REAL * tp = t + (long) j * 2 * num_rhs + v * veclen;
      const REAL * rp = r + (long) j * 2 * num_rhs + v * veclen;
      REAL_VEC tre = SAP_LD(tp), tim = SAP_LD(tp + num_rhs);
      REAL_VEC rre = SAP_LD(rp), rim = SAP_LD(rp + num_rhs);
      da0 += tre * tre;  da1 += tim * tim;            // denom
      ra0 += tre * rre;  ra1 += tim * rim;            // num_re
      ia0 += tre * rim;  ia1 -= tim * rre;            // num_im
      tre = SAP_LD(tp + 2 * num_rhs); tim = SAP_LD(tp + 3 * num_rhs);
      rre = SAP_LD(rp + 2 * num_rhs); rim = SAP_LD(rp + 3 * num_rhs);
      db0 += tre * tre;  db1 += tim * tim;
      rb0 += tre * rre;  rb1 += tim * rim;
      ib0 += tre * rim;  ib1 -= tim * rre;
    }
    den[v] += (da0 + db0) + (da1 + db1);
    nre[v] += (ra0 + rb0) + (ra1 + rb1);
    nim[v] += (ia0 + ib0) + (ia1 + ib1);
  }
}

// one site of MR step (3): phi (+)= a*r  (CSET when first), r += a*t, and
// out += phi on the last inner iteration.  a = (a_re, a_im) is the caller's
// negated alpha, one entry per rhs laid out like a fermion component's rhs
// run -- so lane v*veclen+l carries the step length of the rhs whose fields
// that same lane holds, and the coefficient loads need no shuffling.  Single
// pass, no aliasing checks: every stream is restrict-qualified.
static inline void sap_blk_update(REAL * restrict phi, REAL * restrict r,
                                  const REAL * restrict t,
                                  REAL * restrict out,
                                  const REAL * restrict a_re,
                                  const REAL * restrict a_im,
                                  int first, int last) {
  for (int j = 0; j < Nc * Ns; j ++)
    for (int v = 0; v < num_rhs_vec; v ++) {
      REAL_VEC va_re = SAP_LD(a_re + v * veclen);
      REAL_VEC va_im = SAP_LD(a_im + v * veclen);
      long o = (long) j * 2 * num_rhs + v * veclen;
      REAL_VEC rre = SAP_LD(r + o), rim = SAP_LD(r + o + num_rhs);
      REAL_VEC tre = SAP_LD(t + o), tim = SAP_LD(t + o + num_rhs);
      REAL_VEC pre, pim;
      if (first) {
        pre = rre * va_re - rim * va_im;
        pim = rim * va_re + rre * va_im;
      } else {
        pre = SAP_LD(phi + o)           + rre * va_re - rim * va_im;
        pim = SAP_LD(phi + o + num_rhs) + rim * va_re + rre * va_im;
      }
      SAP_ST(phi + o, pre);
      SAP_ST(phi + o + num_rhs, pim);
      SAP_ST(r + o,           rre + tre * va_re - tim * va_im);
      SAP_ST(r + o + num_rhs, rim + tim * va_re + tre * va_im);
      if (last) {
        SAP_ST(out + o,           SAP_LD(out + o) + pre);
        SAP_ST(out + o + num_rhs, SAP_LD(out + o + num_rhs) + pim);
      }
    }
}

// red-black Schwarz alternating procedure
// One-shot flag: when set, the next red_black_schwarz call accumulates into
// `output` (output += M^nu ferm) instead of overwriting it
// (output := M^nu ferm).  Set only via red_black_schwarz_resfold (below) and
// consumed on entry, so the default overwrite contract -- relied on by the
// solver_info fn-pointer, the standalone test, and the bootstrap M_SAP
// apply -- is never disturbed.  Safe as a static because the only accumulate
// caller (mg_cycle -> smooth_lvl) dispatches single-threaded, and the flag
// is read+reset before any parallel region.  Accumulate is correct because
// `output` is a pure accumulator across visits: it is only memset (skipped
// when accumulating) and += into by the visit updates -- the per-block
// correction depends solely on r/phi/temp.  (Under the residual fold below,
// the sweep-0 seed passes additionally read output as the initial guess --
// always before the visit's own updates, so the accumulate contract is
// unchanged.)  The fine V-cycle uses it to fuse `psi += smth` into the
// smoother: psi already holds the prolonged coarse correction P psi_c, so
// the smoother turns it directly into psi = P psi_c + M^nu r with no extra
// streaming loop or smth buffer.
static int rbs_accumulate_next = 0;

// One-shot flag (same idiom as rbs_accumulate_next): when set, the next
// red_black_schwarz call treats the odd half of `output` as write-only dead
// data -- the eo outer solve discards the cycle's odd output at the bridge
// slice, so:
//   (a) the per-visit `output_o += phi_o` accumulation (a pure-streaming RMW
//       of the odd half-field, not in the flop model) is skipped in every
//       visit, and
//   (b) the recover pass (psi_o = D_oo^{-1}(eta_o + wilson_oe(psi_e))) is
//       skipped for blocks of the last color of the last sweep, whose phi_o
//       has no remaining reader (same-color blocks are never face-adjacent,
//       there is no later half-sweep, and phi is freed on exit).
// Both are terminal-write eliminations: no even-site value depends on them,
// so the even output is bit-identical.  Set only via
// sap_discard_odd_output_next() (smooth_lvl, when the eo outer flag is
// active); bootstrap / standalone / full-system callers never set it, and
// their odd output is untouched.
static int rbs_discard_odd_next = 0;
void sap_discard_odd_output_next(void) { rbs_discard_odd_next = 1; }

// One-shot out-param (same idiom): when set, the next red_black_schwarz call
// also emits the eo outer solve's Arnoldi matvec w_e = D_hat psi_e into it,
// assembled from the smoother's exit residual instead of a separate explicit
// Schur matvec:
//   w_e = eta_e - r_e + D_eo D_oo^{-1} r_o.
// The identity is exact for any consistent (r_e, r_o) pair, and the exit state
// is one boundary layer short of consistent: r_e is current on last-color
// blocks but misses the final half-sweep's D_eo hops on first-color blocks,
// and r_o -- zero on last-color blocks, because recover satisfies the odd
// block rows exactly at every visit -- equals the exterior D_oe hops of that
// same half-sweep's increments on first-color faces.  Both gaps are closed by
// a face-only tail over the phi those increments still sit in (see the emit_w
// passes at the end of red_black_schwarz), so the whole D_hat apply rides
// block-resident SAP state instead of a full gather-bound Dslash pass.
// Requires the resfold + even-half-eta entry (the eo outer smoother shape).
// Consumed on entry; sap_arnoldi_w_filled() reports whether the last call
// filled the buffer (fgmres falls back to the explicit matvec when not).
static REAL * rbs_emit_w_next = NULL;
static int rbs_emit_w_filled = 0;
// One site's Arnoldi rows, w = eta - r, streamed past the cache (see the
// emit_w pass).  Explicit intrinsics rather than __builtin_nontemporal_store:
// that hint is silently dropped unless the vectorizer both fires and can
// prove alignment, and a dropped hint leaves correct code that simply does
// not stream.  The destination's 64 B alignment is the caller's guarantee;
// sources load unaligned.  site_size = Nc*Ns*num_rhs*2 is a multiple of 16
// floats.  Skipping the write-allocate is an ISA-independent trick; only the
// store width changes.  Both widths need the same alignment the caller
// already tests for (64 B is stronger than the 32 B a ymm stream needs), and
// site_size divides by both 16 and 8.
#define EMITW_NT_ISA ((AVX512 || AVX2) && MG_SINGLE)
static inline void emitw_site_nt(REAL * o, const REAL * a, const REAL * b,
                                 int n) {
#if AVX512 && MG_SINGLE
  for (int k = 0; k < n; k += 16)
    _mm512_stream_ps(o + k, _mm512_sub_ps(_mm512_loadu_ps(a + k),
                                          _mm512_loadu_ps(b + k)));
#elif AVX2 && MG_SINGLE
  for (int k = 0; k < n; k += 8)
    _mm256_stream_ps(o + k, _mm256_sub_ps(_mm256_loadu_ps(a + k),
                                          _mm256_loadu_ps(b + k)));
#else
  for (int k = 0; k < n; k ++) o[k] = a[k] - b[k];
#endif
}
// A fermion site record is a multiple of 64 B, so every site base shares w's
// alignment and one test here settles it for the whole sweep.  Allocations
// are 64 B-aligned, but w is the caller's pointer and need not start one, so
// an unaligned w must fall back rather than fault -- and must say so, since a
// silent fallback reads exactly like "the change did nothing."
static int rbs_emit_w_nt = 0;
#if EMITW_NT_ISA
static int rbs_emit_w_warned = 0;
#endif
void sap_emit_arnoldi_w_next(REAL * w) {
  rbs_emit_w_next = w;
  rbs_emit_w_filled = 0;
#if EMITW_NT_ISA
  rbs_emit_w_nt = (((unsigned long) w) & 63) == 0;
  if (!rbs_emit_w_nt && !rbs_emit_w_warned) {   // report, never fault
    rbs_emit_w_warned = 1;
    print0("[emit-w] w %p not 64B-aligned: non-temporal store disabled\n",
           (void *) w);
  }
#else
  rbs_emit_w_nt = 0;                       // scalar build: no streaming path
#endif
}
int sap_arnoldi_w_filled(void) { return rbs_emit_w_filled; }

// Companion one-shot: fuse the outer GS dots into the emit tail.  The tail
// finalizes nearly every w row while it is cache-hot (pass 1 accumulates the
// missing hops into every first-color row, pass 2 the cross-face Schur term
// into last-color face rows), so dotting each row against the basis right
// there deletes the explicit GS dot pass's full-field w read; only last-color
// interior rows ((B-2)^4 of B^4 sites) were final at the in-sweep emit and
// come back from DRAM.  The basis reads themselves just move here from the
// explicit pass -- the deleted byte is the w stream.
static const REAL * rbs_arn_V = NULL;
static long rbs_arn_stride = 0;
static int rbs_arn_nvec = 0;
static double * rbs_arn_dots = NULL;
static int rbs_arn_dots_filled = 0;
void sap_emit_arnoldi_dots_next(const REAL * Vb, long stride, int nvec,
                                double * partials) {
  // No capacity check: the reduction below is sized from nvec, and `partials`
  // and nvec reach this from the same caller in the same call, so there is no
  // fixed buffer to overflow.
  rbs_arn_dots_filled = 0;
  rbs_arn_V = Vb;
  rbs_arn_stride = stride;
  rbs_arn_nvec = nvec;
  rbs_arn_dots = partials;
}
int sap_arnoldi_dots_filled(void) { return rbs_arn_dots_filled; }

// One row of the fused GS dots: h_j^(k) += conj(V[j] row) . (w row) for each
// rhs k separately (row layout per complex element is num_rhs reals then
// num_rhs imags).  Each rhs drives its own Hessenberg, so the lanes are kept
// apart rather than folded -- which costs nothing here: the k loop was
// already the inner loop, and this drops the horizontal fold at the end
// instead of adding work.  It only widens the accumulator from 2 doubles to
// 2*num_rhs, still a handful of vector registers across the site_size sweep.
// Products and sums are double regardless of REAL: this runs in the single
// twin, where REAL arithmetic would drift the Hessenberg off the outer pass's
// promote-to-double DOTC_KERNEL.  Order still differs from the linear explicit
// pass, so the fused solve matches at roundoff, not bitwise.
static inline void arn_dot_row(const REAL * __restrict v,
                               const REAL * __restrict w, int site_size,
                               double * __restrict re_acc,
                               double * __restrict im_acc) {
  double re[num_rhs], im[num_rhs];
  for (int k = 0; k < num_rhs; k ++) { re[k] = 0.0; im[k] = 0.0; }
  for (int e = 0; e < site_size; e += 2 * num_rhs)
    for (int k = 0; k < num_rhs; k ++) {
      double vre = v[e + k], vim = v[e + num_rhs + k];
      double wre = w[e + k], wim = w[e + num_rhs + k];
      re[k] += vre * wre + vim * wim;
      im[k] += vre * wim - vim * wre;
    }
  for (int k = 0; k < num_rhs; k ++) {
    re_acc[k] += re[k];
    im_acc[k] += im[k];
  }
}

// Tile-major visit order for the half-sweep block loops.
// Same-color blocks write disjoint sites, so the field output is bit-identical
// under any visit order -- the order is free to spend on cache reuse.  The
// fused Arnoldi dots are the exception: they sum across blocks into arn_red, so
// the visit order reassociates them and they move at rounding level (the same
// way the fused path already differs from the explicit GS pass).  The boundary
// passes re-read shared lines: an edge/corner site of a block lies on 2-4 of
// its faces, so of the 512 neighbor face-site reads a block visit issues,
// only 240 are distinct -- 53% duplicate a read issued by a diagonal
// same-color partner (+/-1 in two block coordinates).  Under a lex
// collapse(4)/guided schedule those partners run whole block-rows or -planes
// apart, far outside the L3 lifetime at the DRAM rate the boundary passes
// sustain, so every duplicate is a fresh DRAM read.  The list below is
// tile-major over compact 4D tiles of ~nthreads same-color blocks, snaked so
// consecutive tiles share a face, and the block loops issue it round-robin
// (schedule(static, 1)): the ~nthreads visits in flight at any instant sit
// in one tile-sized region and read their shared lines within the cache
// window instead of a half-sweep apart.
// tile extents in blocks; product/2 same-color blocks ~ one concurrent window
#ifndef SAP_VISIT_TILE_X
#define SAP_VISIT_TILE_X 2
#define SAP_VISIT_TILE_Y 2
#define SAP_VISIT_TILE_Z 2
#define SAP_VISIT_TILE_T 4
#endif
static int sap_visit_n[2];        // list lengths (blocks of each color)
static int * sap_visit_order[2];  // block coords packed 8 bits per dimension
static void sap_build_visit_order(int bs, int rank_blk_color) {
  if (sap_visit_order[0]) return;      // extents and bs are fixed for the run
  const int nbx = Lx / bs, nby = Ly / bs, nbz = Lz / bs, nbt = Lt / bs;
  const int nb = nbx * nby * nbz * nbt;
  const int ntx = (nbx + SAP_VISIT_TILE_X - 1) / SAP_VISIT_TILE_X;
  const int nty = (nby + SAP_VISIT_TILE_Y - 1) / SAP_VISIT_TILE_Y;
  const int ntz = (nbz + SAP_VISIT_TILE_Z - 1) / SAP_VISIT_TILE_Z;
  const int ntt = (nbt + SAP_VISIT_TILE_T - 1) / SAP_VISIT_TILE_T;
  sap_visit_order[0] = malloc(sizeof(int) * nb);
  sap_visit_order[1] = malloc(sizeof(int) * nb);
  int cnt[2] = {0, 0};
  // boustrophedon tile walk: each dimension's direction flips with the count
  // of completed lower-dimension sweeps, so successive tiles share a face
  // even across row and plane ends
  for (int wt = 0; wt < ntt; wt ++)
    for (int wz = 0; wz < ntz; wz ++)
      for (int wy = 0; wy < nty; wy ++)
        for (int wx = 0; wx < ntx; wx ++) {
          const int tt = wt;
          const int tz = (wt & 1) ? ntz - 1 - wz : wz;
          const int ty = ((wz + wt * ntz) & 1) ? nty - 1 - wy : wy;
          const int tx = ((wy + (wz + wt * ntz) * nty) & 1) ? ntx - 1 - wx
                                                            : wx;
          for (int bt = tt * SAP_VISIT_TILE_T;
               bt < nbt && bt < (tt + 1) * SAP_VISIT_TILE_T; bt ++)
            for (int bz = tz * SAP_VISIT_TILE_Z;
                 bz < nbz && bz < (tz + 1) * SAP_VISIT_TILE_Z; bz ++)
              for (int by = ty * SAP_VISIT_TILE_Y;
                   by < nby && by < (ty + 1) * SAP_VISIT_TILE_Y; by ++)
                for (int bx = tx * SAP_VISIT_TILE_X;
                     bx < nbx && bx < (tx + 1) * SAP_VISIT_TILE_X; bx ++) {
                  const int c = ((bx ^ by ^ bz ^ bt) & 0x1) ^ rank_blk_color;
                  sap_visit_order[c][cnt[c] ++]
                    = bx | (by << 8) | (bz << 16) | (bt << 24);
                }
        }
  sap_visit_n[0] = cnt[0];
  sap_visit_n[1] = cnt[1];
  // one-shot list check: a wrong list is silently wrong numerics, so verify
  // every block appears exactly once and in the list of its own color
  char * seen = calloc(nb, 1);
  int bad = (cnt[0] + cnt[1] != nb);
  for (int c = 0; c < 2; c ++)
    for (int i = 0; i < sap_visit_n[c]; i ++) {
      const int v = sap_visit_order[c][i];
      const int idx = (((v >> 24) * nbz + ((v >> 16) & 0xff)) * nby
                      + ((v >> 8) & 0xff)) * nbx + (v & 0xff);
      seen[idx] ++;
      bad |= ((((v ^ (v >> 8) ^ (v >> 16) ^ (v >> 24)) & 0x1)
               ^ rank_blk_color) != c);
    }
  for (int i = 0; i < nb; i ++) bad |= (seen[i] != 1);
  free(seen);
  if (bad) {
    comm_die("[sap-visit] bad visit list (%d + %d of %d blocks)",
             cnt[0], cnt[1], nb);
  }
}

// One-shot pair for the residual fold (set only via red_black_schwarz_resfold):
// the next call treats `ferm` as the original rhs eta (not a pre-formed
// residual) and `output` as pre-loaded with an initial guess x, and seeds the
// block residuals r = eta - D x block-locally during sweep 0 -- DDalphaAMG's
// res == _RES smoother contract.  rbs_fold_eta_even: eta is the eo outer
// solve's even half-field (odd part implicitly zero), so the odd seed skips
// its eta term.  Fold implies accumulate (output must keep holding x).
static int rbs_fold_next = 0;
static int rbs_fold_eta_even_next = 0;

// --- fused first-color prolongation (see sap_fuse_prolong_next in solver.h)
// One-block cached-store prolongation, defined in coarse.c (solver.c does not
// include coarse.h; mangle.h keeps the twin symbols consistent).
void apply_P_block_cached(REAL * fine_out, REAL * coarse_in, REAL * nv,
                          int bx, int by, int bz, int bt);
// The red-black block coloring must be global: a block's rank-face neighbors
// live on another rank and must still be the opposite color.  Rank origins are
// SAP-block-aligned (checked in comm_init), so the global block index is the
// local one plus origin/bs and the color just picks up the rank origin's block
// parity.  0 at an undecomposed grid.
static int sap_rank_blk_color(void) {
  return ((qc_grid.origin[2] / SCHWARZ_BLOCK_SIZE)
        + (qc_grid.origin[3] / SCHWARZ_BLOCK_SIZE)) & 0x1;
}
// Visit color c = local block parity ^ rank_blk_color, so the second-processed
// color (c == 1) has local parity 1 ^ rank_blk_color.
int sap_bulk_blk_parity(void) { return 1 ^ sap_rank_blk_color(); }
static REAL * rbs_fusep_cin_next = NULL;
static REAL * rbs_fusep_nv_next  = NULL;
static double rbs_fusep_blk_flops = 0;
int sap_fuse_prolong_next(REAL * coarse_in, REAL * nv, double blk_flops) {
  rbs_fusep_cin_next   = coarse_in;
  rbs_fusep_nv_next    = nv;
  rbs_fusep_blk_flops  = blk_flops;
  return 1;
}

// Block-sweep helpers shared by every pass of red_black_schwarz below.  They
// read the enclosing scope's block origin (x_outer..t_outer),
// schwarz_block_size and blk_sites, so they are only usable inside that
// function's block loops; they are macros rather than inlines so the loop
// nest fuses into each pass's body.

// Visit every site of the current block, x fastest -- the order the interior
// neighbor tables are rowed in
#define SCHWARZ_BLOCK_SITES \
  for (int t = t_outer; t < t_outer + schwarz_block_size; t ++) \
    for (int z = z_outer; z < z_outer + schwarz_block_size; z ++) \
      for (int y = y_outer; y < y_outer + schwarz_block_size; y ++) \
        for (int x = x_outer; x < x_outer + schwarz_block_size; x ++)

// r_o (the odd residual) is pure per-block scratch in the odd-even MR: the
// boundary pass writes it from neighbor phi_e, setup and recover read it, and
// it carries no state across sweeps (the source enters once, on sweep 0).
// So, like 'temp' for z, it is block-local rather than a global odd half of r
// streamed through DRAM each sweep: it lives in temp's odd half (unused by
// the even-only z = D_hat*r MR).  Bit-identical to a global r_o -- same ops,
// different address.  RO_LOC_IDX indexes that half.
#define RO_LOC_IDX \
  ((blk_sites / 2) + \
   ((((long)(t - t_outer) * schwarz_block_size + (z - z_outer)) \
     * schwarz_block_size + (y - y_outer)) * (schwarz_block_size / 2) \
    + (x - x_outer) / 2))

// Only sites on a block face have cross-block (boundary) directions; everything
// else gets a no-op masked call, so the boundary passes skip non-face sites
// outright.
#define SCHWARZ_ON_FACE \
  (x == x_outer || x == x_outer + schwarz_block_size - 1 || \
   y == y_outer || y == y_outer + schwarz_block_size - 1 || \
   z == z_outer || z == z_outer + schwarz_block_size - 1 || \
   t == t_outer || t == t_outer + schwarz_block_size - 1)

int red_black_schwarz(REAL * output, REAL * ferm, REAL * U_kappa, REAL * F_csw,
                      int clover, int max_iter, int sap_block_iter, REAL eps,
                      int eo_precondition, solver_info * preconditioner) {
  int zero_output = !rbs_accumulate_next;
  rbs_accumulate_next = 0;              // one-shot: consume immediately
  int discard_odd = rbs_discard_odd_next;
  rbs_discard_odd_next = 0;             // one-shot: consume immediately
  int fold = rbs_fold_next;
  rbs_fold_next = 0;                    // one-shot: consume immediately
  int fold_eta_even = rbs_fold_eta_even_next;
  rbs_fold_eta_even_next = 0;           // one-shot: consume immediately
  REAL * emit_w = rbs_emit_w_next;
  const int emit_w_nt = rbs_emit_w_nt;  // 64B-aligned: stream w, skip the RFO
  rbs_emit_w_next = NULL;               // one-shot: consume immediately
  // w emission is defined only for the eo outer smoother shape: resfold entry
  // (ferm is the original eta), even-half eta, discarded odd output (the
  // terminal recover stores its increment negated -- see the emit tail), and
  // at least one sweep.
  assert(!emit_w || (fold && fold_eta_even && discard_odd && max_iter > 0));
  const REAL * arn_V = rbs_arn_V;       // fused-dot companion (one-shot)
  const long arn_stride = rbs_arn_stride;
  double * arn_dots = rbs_arn_dots;
  const int arn_n = (arn_V && emit_w) ? rbs_arn_nvec : 0;
  rbs_arn_V = NULL;                     // one-shot: consume immediately
  rbs_arn_dots = NULL;
  // the dots are defined only as a companion to a consumed w request
  assert(!arn_V || emit_w);
  REAL * fusep_cin = rbs_fusep_cin_next;
  REAL * fusep_nv  = rbs_fusep_nv_next;
  rbs_fusep_cin_next = NULL;            // one-shot: consume immediately
  rbs_fusep_nv_next  = NULL;
  // Fused prolongation is defined only for the resfold entry (the guess is
  // what the sweep-0 seeds read) with at least one sweep (the in-visit
  // prolong must actually run to complete the guess).
  assert(!fusep_cin || (fold && max_iter > 0));
  // This solver runs on the full both-parity system: even-odd preconditioning
  // is applied *inside* each Schwarz block's MR, not as a global Schur
  // reduction.  So it does not consume solve()'s global eo_precondition
  // reduction -- use it with eo_precondition = 0.
  assert(!eo_precondition);
  struct timeval start, end;
  gettimeofday(&start, NULL);
  // The block edge is compile-time (it divides the lattice extents and is
  // static-asserted in coarse.h); the MR count arrives as an argument.
  int schwarz_block_size = SCHWARZ_BLOCK_SIZE;
  int num_half_ferm = num_rhs;
  int site_size = Nc * Ns * num_rhs * 2;
  if (!eo_precondition) num_half_ferm *= 2;
  long multi_ferm_size = (long) half_ferm_size * num_half_ferm;
  // r holds only the even residual: r_o is localized to per-block scratch
  // (temp's odd half), so nothing per-sweep touches r's odd half.  Allocate
  // and first-touch only the even half (oddh elements) -- avoids a dead
  // odd-half-field allocation + parallel copy at startup, which matters when
  // only a few sweeps run (preconditioner use).  phi/output stay full (both
  // parities, read by neighbors' boundary updates / hold the odd solution).
  long oddh = (long) ferm_size * num_rhs / 2;   // odd-half offset into
                                                //   phi/output
  REAL * r = malloc(sizeof(REAL) * oddh);
  REAL * phi = malloc(sizeof(REAL) * multi_ferm_size);
  // 'temp' (z) is pure per-block scratch: it is written by the dslash/clover,
  // read to form alpha, and read once more in the caxpy update, all within a
  // single block's inner-iteration loop -- it never crosses a block or sweep
  // boundary (unlike phi, which the boundary update reads from neighboring
  // blocks).  It is dead the instant a block finishes, so rather than a
  // full-lattice array streaming through DRAM each sweep, each thread gets a
  // small block-sized buffer that stays L2-resident and is reused for every
  // block it owns.  blk_sites counts both parities; the per-parity half
  // lives at offset (blk_sites/2)*site_size within the buffer.
  int nthr_temp = omp_get_max_threads();
  long blk_sites = (long) schwarz_block_size * schwarz_block_size
                 * schwarz_block_size * schwarz_block_size;
  long blk_buf   = blk_sites * site_size;
  REAL ** temp_th = malloc(sizeof(REAL *) * nthr_temp);
  for (int q = 0; q < nthr_temp; q ++)
    temp_th[q] = malloc(sizeof(REAL) * blk_buf);
  // For the block-internal odd-even MR we need D_oo^{-1} (the inverse of the
  // odd clover block 1 - A_oo).  solve() prepared F_csw with invert_odd=0
  // (this solver runs on the full both-parity system, eo_precondition=0), so
  // build a separate inverted-odd-clover field here.  Without clover
  // D_oo = I, so none is needed.
  // D_oo^{-1} is cached across SAP calls (F_csw is fixed per solve); see
  // get_Doo_inv / sap_free_doo_inv_cache above.  NULL with no clover
  // (D_oo = I).
  REAL * F_oo_inv = clover ? get_Doo_inv(F_csw) : NULL;
  // (A - 1) for the MR's fused -z_e formation and the fold seeds' diagonal
  // terms; even half at F_mi, odd half at F_mi_o.  See get_F_minus_I.
  REAL * F_mi     = clover ? get_F_minus_I(F_csw) : NULL;
  REAL * F_mi_o   = clover ? F_mi + (long) Vh * 2 * 6 * 6 : NULL;
  // shared bs^4 offset table for the interior block calls (idempotent)
  wilson_block_itab_init(schwarz_block_size);
  // Fold entry: r_e is seeded block-locally during sweep 0 (from eta and the
  // initial guess in output), so the global even-half source copy is skipped
  // and ferm is read directly by the seed passes instead.
  if (!fold)
    copy_multi_half_ferm(r, ferm, num_rhs); // even half only (oddh elements)
  if (zero_output) memset(output, 0, sizeof(REAL) * multi_ferm_size);
  // Global red-black block color of the rank origin (see sap_rank_blk_color).
  const int rank_blk_color = sap_rank_blk_color();
  sap_build_visit_order(schwarz_block_size, rank_blk_color);
  int n;
  for (n = 0; n < max_iter; n ++) {
    for (int color = 0; color < 2; color ++) {
      // Cross-rank boundary reads (both 1 site deep): the odd pre-pass hop
      // reads neighbor-block even sites (ghost block 0), the merged even
      // Schur-RHS full kernel reads neighbor-block odd sites (ghost block 1).
      // Face neighbors are opposite-color blocks, untouched during this
      // half-sweep, so exchanging once at the top is exactly the values a
      // single rank would read.  During a fold sweep 0 both hops read the
      // initial guess `output` (whose color-0 rows color 1 sees with that
      // half-sweep's visit updates already accumulated), so output is
      // exchanged instead of phi; phi carries no cross-block state until
      // n = 1.  Without the fold the very first half-sweep skips all boundary
      // reads (phi uninitialized) -- same condition as the kernels.
      // Timed halo window (constant-folds away undecomposed): the blocking
      // exchange wall, during which every core idles.  Compact exchange:
      // this half-sweep's readers are the color-`color` blocks, so only the
      // adjacent (color-halved) face rows travel, spin-projected -- see
      // wilson_halo_sap.  Window counts stay 2 per both-parity exchange for
      // continuity of the [sap-halo] ms/win metric across the change.
      struct timeval h0, h1;
      if (FERM_GHOSTS) gettimeofday(&h0, NULL);
      if (fold && n == 0) {
        wilson_halo_sap(output, output + oddh, color);
        g_sap_halo_sweep_win += 2;
      } else if (n > 0 || color > 0) {
        wilson_halo_sap(phi, phi + oddh, color);
        g_sap_halo_sweep_win += 2;
      }
      if (FERM_GHOSTS) {
        gettimeofday(&h1, NULL);
        g_sap_halo_sweep_secs += time_used(h0, h1);
      }
      // Same-color Schwarz blocks are mutually independent (every face
      // neighbor of a block is the opposite color, untouched during this
      // sweep), so we parallelize over blocks: each thread owns a whole block
      // and runs its boundary-update / MR-solve / accumulate serially.  This
      // keeps a block's working set L2/L3-resident and costs one fork/join per
      // sweep rather than several per block.
      #pragma omp parallel
      {
        // round-robin over the tile-major list: the ~nthreads visits in flight
        // at any instant sit in one compact tile (see sap_build_visit_order)
        #pragma omp for schedule(static, 1)
        for (int visit_i = 0; visit_i < sap_visit_n[color]; visit_i ++) {
          const int vcode = sap_visit_order[color][visit_i];
          const int x_outer = (vcode & 0xff)         * schwarz_block_size;
          const int y_outer = ((vcode >> 8) & 0xff)  * schwarz_block_size;
          const int z_outer = ((vcode >> 16) & 0xff) * schwarz_block_size;
          const int t_outer = ((vcode >> 24) & 0xff) * schwarz_block_size;
          // Fused first-color prolongation: this block's guess was skipped
          // by the bulk apply_P, so produce it here (cached stores) and
          // let the fold seeds below read it from cache -- the raw guess
          // never round-trips DRAM; only its updated value (mr3/recover)
          // does.  First color only: its face neighbors (the second color)
          // were bulk-prolonged, and second-color seeds read neighbor
          // output already carrying first-color updates, exactly as
          // without the fusion.  Values are bit-identical to apply_P's.
          if (fusep_cin && n == 0 && color == 0)
            apply_P_block_cached(output, fusep_cin, fusep_nv,
                                 vcode & 0xff, (vcode >> 8) & 0xff,
                                 (vcode >> 16) & 0xff,
                                 (vcode >> 24) & 0xff);
          // block-local scratch for z = M*r (reused across all blocks this
          // thread owns, so it stays resident in L2 instead of DRAM)
          REAL * temp = temp_th[omp_get_thread_num()];
          // block-origin site indices (half-/full-lattice) for the
          // interior-table calls; wk_lidx counts SCHWARZ_BLOCK_SITES
          // visits (x fastest), matching the table's row order
          long blk_half = site_half(x_outer / 2, y_outer, z_outer,
                                    t_outer);
          long blk_full = site_full(x_outer, y_outer, z_outer, t_outer);
          int wk_lidx;
          // ---- fused odd pre-pass: one scan over the odd sites does
          //   (a) seed block-local r_o: without the fold, eta_o on sweep 0
          //       and 0 afterwards; on a fold sweep 0, the full initial
          //       residual r_o = eta_o - (D x)_o for the guess x = output,
          //       i.e. (A_oo - 1) x_o + eta_o + the unmasked 8-direction
          //       hop of x_e (interior directions read this block's guess,
          //       exterior directions the neighbors' current output -- for
          //       color 1 that already carries color 0's accumulated
          //       corrections, so no separate increment hop is needed);
          //   (b) the cross-block boundary hops into r_o (skipped on the
          //       very first half-sweep, when phi is uninitialized, and on
          //       a fold sweep 0, when the seed's hop absorbs them);
          //   (c) the eo-setup psi_o^0 = D_oo^{-1} r_o into phi_o.
          // (a)-(c) at a site touch only that site's r_o/phi_o (the hops
          // read neighbor-block phi/output only), so fusing the former
          // separate scans is bit-identical and saves passes over the odd
          // half.
          SCHWARZ_BLOCK_SITES {
            if (((x ^ y ^ z ^ t) & 0x1) != 1) continue;       // odd sites
            long rloc = site_half(x / 2, y, z, t);
            long ro = RO_LOC_IDX;
            if (fold && n == 0) {
              // eo outer solve: eta is the even half-field, eta_o = 0
              if (clover)
                clover_kernel_ow(temp + (ro - rloc) * site_size,
                                 output + oddh, F_mi_o, x/2, y, z, t);
              else
                for (int k = 0; k < site_size; k ++)
                  temp[ro * site_size + k]
                    = -output[oddh + rloc * site_size + k];
              if (!fold_eta_even)
                for (int k = 0; k < site_size; k ++)
                  temp[ro * site_size + k]
                    += ferm[oddh + rloc * site_size + k];
              wilson_kernel(temp + (ro - rloc) * site_size, output,
                            U_kappa, x/2, y, z, t, 1);
            } else {
              if (n == 0)
                for (int k = 0; k < site_size; k ++)
                  temp[ro * site_size + k]
                    = ferm[oddh + rloc * site_size + k];
              else
                for (int k = 0; k < site_size; k ++)
                  temp[ro * site_size + k] = 0;
              if ((n > 0 || color > 0) && SCHWARZ_ON_FACE)
                wilson_kernel_block(temp + (ro - rloc) * site_size, phi,
                                    U_kappa, x/2, y, z, t, 1,
                                    schwarz_block_size, 1);
            }
            // psi_o^0 = D_oo^{-1} eta_o in one overwrite clover apply
            // (clover_kernel_ow: no zero pass, phi_o written once).
            // Nothing reads this block's phi_o before the Schur-RHS pass
            // below, so setting it here instead of in a later scan is
            // bit-identical.
            if (clover)
              clover_kernel_ow(phi + oddh, temp + (ro - rloc) * site_size,
                               F_oo_inv, x/2, y, z, t);
            else
              for (int k = 0; k < site_size; k ++)
                phi[oddh + rloc * site_size + k]
                  = temp[ro * site_size + k];
          }
          // (even-site boundary hops are folded into the merged even pass)
          // ---- block-internal odd-even preconditioned MR -------------
          // Solve the block system D_i psi = eta (eta = r|_block) by
          // eliminating the odd sites: with psi_o = D_oo^{-1}(eta_o - D_oe
          // psi_e) the odd block residual vanishes identically, so MR
          // iterates only on the even sites of the local Schur complement
          // D_hat = D_ee - D_eo D_oo^{-1} D_oe.  The even residual it
          // produces is the true block residual there, so the full-block
          // residual (read by neighbors' boundary updates) is recovered
          // exactly and phi/output/r keep their both-parity meaning.
          //
          // Hopping sign convention: D's off-diagonal is -wilson, so
          //   D_oe v = -wilson_oe(v),  D_eo v = -wilson_eo(v),
          //   D_ee v =  v - clover_ee(v).
          // phi_e accumulates psi_e (CSET on the first iter); phi_o is set
          // to psi_o^0 by the fused odd pre-pass above, used as the w_o
          // target during MR, then set to the final psi_o by recover.
          // D_oo^{-1} is clover_kernel with F_oo_inv (the identity when
          // there is no clover).  eta_o is the block-local r_o (temp odd
          // half, via RO_LOC_IDX).
          // --- even pass: boundary update + Schur RHS in one hop -------
          //   r_e <- r_e + wilson_ext(phi_o^nbr) + wilson_int(psi_o^0)
          // The exterior and interior direction masks are complements, so
          // one unmasked 8-direction kernel computes their sum: exterior
          // dirs read neighbor-block phi_o (the previous half-sweep's final
          // psi_o -- neighbors are the opposite color, untouched now),
          // interior dirs read this block's psi_o^0 from the odd pre-pass.
          // This requires every face neighbor to be the opposite color,
          // i.e. an even number of blocks per dimension.  Direction
          // summation order differs from the two-call scheme, so the merged
          // pass is algebraically exact but not bit-identical.  On the very
          // first half-sweep neighbor phi is uninitialized, so fall back to
          // the interior-only masked call.
          // Fold sweep 0 replaces the boundary-update role of the merged
          // pass: r_e is seeded in place as eta_e - (D x)_e for the guess
          // x = output ((A_ee - 1) x_e + eta_e + the unmasked hop of x_o,
          // whose exterior directions read neighbor output_o -- for
          // color 1 already carrying color 0's recover updates), then the
          // usual interior Schur-RHS hop of this block's psi_o^0 is added
          // on top.
          wk_lidx = -1;
          SCHWARZ_BLOCK_SITES {
            wk_lidx ++;
            if (((x ^ y ^ z ^ t) & 0x1) != 0) continue;      // even sites
            if (fold && n == 0) {
              long loc = site_half(x / 2, y, z, t);
              if (clover)
                clover_kernel_ow(r, output, F_mi, x/2, y, z, t);
              else
                for (int k = 0; k < site_size; k ++)
                  r[loc * site_size + k] = -output[loc * site_size + k];
              for (int k = 0; k < site_size; k ++)
                r[loc * site_size + k] += ferm[loc * site_size + k];
              wilson_kernel(r, output + oddh, U_kappa, x/2, y, z, t, 0);
              wilson_kernel_block_itab(r, phi + oddh, U_kappa,
                                  x/2, y, z, t, 0, schwarz_block_size,
                                  blk_half, blk_full, wk_lidx);
            } else if (n > 0 || color > 0)
              wilson_kernel(r, phi + oddh, U_kappa, x/2, y, z, t, 0);
            else
              wilson_kernel_block_itab(r, phi + oddh, U_kappa,
                                  x/2, y, z, t, 0, schwarz_block_size,
                                  blk_half, blk_full, wk_lidx);
          }
          // --- MR on D_hat over the even sites only -------------------
          for (int it = 0; it < sap_block_iter; it ++) {
            // (1) phi_o = D_oo^{-1} wilson_oe(r_e) = w_o.  The hop is
            //     built in block-local scratch (temp's even half: its
            //     last -z values are dead, and step (2) below rewrites it
            //     only after this pass) so the clover apply is not
            //     in-place -- avoiding clover_kernel's in-place stack
            //     zero + copy-back -- and phi_o is written once.
            wk_lidx = -1;
            SCHWARZ_BLOCK_SITES {
              wk_lidx ++;
              if (((x ^ y ^ z ^ t) & 0x1) != 1) continue;     // odd sites
              long rloc = site_half(x / 2, y, z, t);
              // even-half scratch slot
              long lo = RO_LOC_IDX - blk_sites / 2;
              for (int k = 0; k < site_size; k ++)
                temp[lo * site_size + k] = 0;
              wilson_kernel_block_itab(temp + (lo - rloc) * site_size, r,
                                       U_kappa, x/2, y, z, t, 1,
                                       schwarz_block_size,
                                       blk_half, blk_full, wk_lidx);
              if (clover)
                clover_kernel_ow(phi + oddh,
                                 temp + (lo - rloc) * site_size,
                                 F_oo_inv, x/2, y, z, t);
              else
                for (int k = 0; k < site_size; k ++)
                  phi[oddh + rloc * site_size + k]
                    = temp[lo * site_size + k];
            }
            // (2) temp_e = -z_e where z_e = D_hat r_e
            //     -z_e = -r_e + clover_ee(r_e) + wilson_eo(w_o)
            //     and accumulate the per-rhs alpha = <z_e,r_e>/<z_e,z_e>
            //     (temp = -z_e, so stored alpha = -alpha_true; both updates
            //     use -coeff)
            // One (num_re, num_im, denom) triple per rhs, held in the rhs
            // lanes for the whole block reduction.
            REAL_VEC anum_re[num_rhs_vec], anum_im[num_rhs_vec],
                     aden[num_rhs_vec];
            for (int v = 0; v < num_rhs_vec; v ++)
              anum_re[v] = anum_im[v] = aden[v] = ZERO_VEC;
            wk_lidx = -1;
            SCHWARZ_BLOCK_SITES {
              wk_lidx ++;
              if (((x ^ y ^ z ^ t) & 0x1) != 0) continue;    // even sites
              long loc = site_half(x / 2, y, z, t);
              long local_loc
                  = (((long)(t - t_outer) * schwarz_block_size
                      + (z - z_outer)) * schwarz_block_size
                     + (y - y_outer)) * (schwarz_block_size / 2)
                    + (x - x_outer) / 2;
              REAL * temp_out = temp + (local_loc - loc) * site_size;
              // -r_e + clover_ee(r_e) = (A_ee - 1) r_e in one overwrite
              // apply of the cached F_mi: no -r init copy.  Exact but not
              // bit-identical to init-then-accumulate (the diagonal term
              // rounds as (F_aa-1)*r instead of -r + F_aa*r).
              if (clover)
                clover_kernel_ow(temp_out, r, F_mi, x/2, y, z, t);
              else
                for (int k = 0; k < site_size; k ++)
                  temp[local_loc * site_size + k]
                    = -r[loc * site_size + k];
              wilson_kernel_block_itab(temp_out, phi + oddh, U_kappa,
                                  x/2, y, z, t, 0, schwarz_block_size,
                                  blk_half, blk_full, wk_lidx);
              sap_blk_dot3(temp + local_loc * site_size,
                           r + loc * site_size,
                           anum_re, anum_im, aden);
            }
            // Per-rhs step length.  The rhs share only the block
            // operator, and alpha_i = <z_i,r_i>/<z_i,z_i> is the exact
            // minimizer of ||r_i - alpha z_i|| for rhs i, so collapsing
            // the lanes into one alpha would instead minimize the summed
            // residual and hand every rhs a compromise step: harmless
            // when the rhs are the spin-colors of one point source (same
            // support, similar local norms and residual directions), a
            // bounded efficiency loss when they are sources at different
            // positions, where a block sits in the near field of some rhs
            // and the far field of others.  Negated here, for
            // sap_blk_update's -coeff convention.
            REAL nalpha_re[num_rhs], nalpha_im[num_rhs];
            for (int v = 0; v < num_rhs_vec; v ++) {
              const REAL * nr = (const REAL *) &anum_re[v];
              const REAL * ni = (const REAL *) &anum_im[v];
              const REAL * dn = (const REAL *) &aden[v];
              for (int l = 0; l < veclen; l ++) {
                REAL d = dn[l];
                nalpha_re[v * veclen + l] = d != 0 ? -nr[l] / d : 0;
                nalpha_im[v * veclen + l] = d != 0 ? -ni[l] / d : 0;
              }
            }
            int first = (it == 0);
            int last  = (it == sap_block_iter - 1);
            // (3) phi_e += alpha r_e (CSET first) ; r_e -= alpha z_e ;
            //     output_e += phi_e on the last iteration
            SCHWARZ_BLOCK_SITES {
              if (((x ^ y ^ z ^ t) & 0x1) != 0) continue;    // even sites
              long loc = site_half(x / 2, y, z, t);
              long local_loc
                  = (((long)(t - t_outer) * schwarz_block_size
                      + (z - z_outer)) * schwarz_block_size
                     + (y - y_outer)) * (schwarz_block_size / 2)
                    + (x - x_outer) / 2;
              sap_blk_update(phi + loc * site_size, r + loc * site_size,
                             temp + local_loc * site_size,
                             output + loc * site_size,
                             nalpha_re, nalpha_im, first, last);
            }
          }
          // --- recover psi_o = D_oo^{-1}(eta_o - D_oe psi_e)
          //              = D_oo^{-1}(eta_o + wilson_oe(psi_e)) ;
          //     finalize phi_o = psi_o, output_o += psi_o, r_o = 0.
          //     eta_o is still in r_o (MR touched only even sites);
          //     psi_e is in phi_e. --------------------------------------
          // eta_o + wilson_oe(psi_e) is built in block-local scratch
          // (temp's even half, dead now that the MR is finished) so the
          // D_oo^{-1} apply is not in-place (no stack zero + copy-back)
          // and the per-direction hop accumulation stays in L1/L2 scratch
          // instead of RMW-ing phi_o.
          //
          // discard_odd (eo outer solve): phi_o's only readers are later
          // half-sweeps' boundary/Schur passes, and output_o's only reader
          // is the caller (who discards it).  So skip the whole recover on
          // the terminal half-sweep (no later reader of phi_o), and skip
          // the output_o accumulation always.  Even output bit-identical.
          // emit_w gives the terminal phi_o a reader again -- the tail's
          // first-color r_e fix-up hops -- so its recover must run (the
          // output_o accumulation stays skipped: the residual identity
          // holds for the virtual psi_o these increments imply, which is
          // all the emitted w depends on).
          if (!(discard_odd && n == max_iter - 1 && color == 1
                && !emit_w)) {
            wk_lidx = -1;
            SCHWARZ_BLOCK_SITES {
              wk_lidx ++;
              if (((x ^ y ^ z ^ t) & 0x1) != 1) continue;       // odd sites
              long rloc = site_half(x / 2, y, z, t);
              long ro = RO_LOC_IDX;
              long lo = ro - blk_sites / 2;        // even-half scratch slot
              for (int k = 0; k < site_size; k ++)
                temp[lo * site_size + k] = temp[ro * site_size + k];
              wilson_kernel_block_itab(temp + (lo - rloc) * site_size, phi,
                                       U_kappa, x/2, y, z, t, 1,
                                       schwarz_block_size,
                                       blk_half, blk_full, wk_lidx);
              // emit_w: store the terminal increment negated
              // (phi_o = -psi_o).  Its only reader is the tail's unmasked
              // w hop, whose exterior directions need
              // -wilson_ext(psi_o^nbr) while its interior directions need
              // +wilson_int(t): with both odd fields stored negated,
              // one accumulating kernel serves both corrections.
              // The odd output accumulation below is dead under discard_odd
              // (asserted with emit_w), so nothing else sees the sign.
              if (emit_w && n == max_iter - 1 && color == 1)
                for (int k = 0; k < site_size; k ++)
                  temp[lo * site_size + k] = -temp[lo * site_size + k];
              if (clover)
                clover_kernel_ow(phi + oddh, temp + (lo - rloc) * site_size,
                                 F_oo_inv, x/2, y, z, t);
              else
                for (int k = 0; k < site_size; k ++)
                  phi[oddh + rloc * site_size + k]
                    = temp[lo * site_size + k];
              // no r_o zeroing: r_o is block-local and re-seeded next sweep
              // fold: color 1's sweep-0 seeds read neighbor output_o, so
              // even when the eo outer discards the odd output, color 0's
              // first-visit corrections must still be accumulated into it.
              if (!discard_odd || (fold && n == 0 && color == 0))
                for (int k = 0; k < site_size; k ++)
                  output[oddh + rloc * site_size + k]
                    += phi[oddh + rloc * site_size + k];
            }
          }
          // emit_w: a block's own r_e is final right here at its terminal
          // visit (its MR just ended, later half-sweeps write only their
          // own blocks' r rows, recover never writes r_e), so write its
          // Arnoldi rows while r_e is still block-resident: w_e = eta_e -
          // r_e.  Last-color rows are complete up to the cross-face Schur
          // term (r_o = 0 by recover), added by the tail's last pass;
          // first-color rows still miss the final half-sweep's cross-block
          // corrections, accumulated in place by the tail's unmasked hop.
          if (emit_w && n == max_iter - 1)
            SCHWARZ_BLOCK_SITES {
              if (((x ^ y ^ z ^ t) & 0x1) != 0) continue;    // even sites
              long loc = site_half(x / 2, y, z, t);
              const REAL * a = ferm + loc * site_size;
              const REAL * b = r + loc * site_size;
              REAL * o = emit_w + loc * site_size;
              // w is write-only in the sweep and its next reader is the
              // tail's own parallel region, by which point this full-field
              // array is long evicted and comes from DRAM either way.  So
              // the write-allocate line fill a normal store pays buys
              // nothing; skipping it removes a third of the pass's traffic.
              if (emit_w_nt) emitw_site_nt(o, a, b, site_size);
              else for (int k = 0; k < site_size; k ++) o[k] = a[k] - b[k];
            }
        } // end loop over blocks
        // One drain per thread per half-sweep.  The streamed w's only reader is
        // the tail's own parallel region, so per-visit ordering was never needed
        // -- and a fence inside the visit turns background writeback into
        // synchronous DRAM latency once per block.
        if (emit_w_nt) NT_FENCE();
      }   // end omp parallel
    }
  }
  // ---- emit_w tail: close the first-color gaps and add the Schur term -----
  // After the final color-1 half-sweep, phi still holds exactly its increments
  // (each visit overwrites its block's phi region), which are everything the
  // exit residual is missing: on first-color blocks r_e lacks their D_eo hops
  // and the true r_o equals their exterior D_oe hops; last-color blocks are
  // already consistent (r_e current, r_o = 0).  Both colors emitted their
  // w_e = eta_e - r_e rows in-sweep at their terminal visits (r_e block-hot),
  // so the tail never touches eta or r: it only accumulates the first-color
  // corrections into w.
  // Pass 1 (first-color blocks): pull the exterior odd hops -> build r_o in
  //   block scratch and store t = -D_oo^{-1} r_o into this block's phi_o
  //   (dead data), then one plain 8-direction hop per even row adds both
  //   remaining terms to w at once: exterior directions read the neighbors'
  //   negated terminal increments -psi_o (undoing the D_eo hops r_e never
  //   saw), interior directions read t (the block-internal Schur term
  //   D_eo D_oo^{-1} r_o = +wilson(t), since D_eo v = -wilson_eo(v)).  The
  //   masks are complements, so the unmasked kernel sums exactly the two
  //   corrections -- the merged even pass's trick; storing both odd fields
  //   negated is what gives them one accumulation sign.
  // Pass 2 (last-color blocks, face rows only): w_e += wilson_ext(t^nbr) --
  //   their cross-face odd neighbors are first-color face sites.  A separate
  //   pass because it needs every neighbor's t final, and pushing from pass 1
  //   would race where two first-color blocks meet the same edge.
  // Fused GS dots (arn_n > 0): each pass also dots the rows it finalizes
  //   against the outer basis while they are cache-hot from the accumulate.
  //   Row ownership is exact -- pass 1 owns every first-color row, pass 2
  //   every last-color row (face rows hot; interior rows, (B-2)^4 of B^4,
  //   were final at the in-sweep emit and are re-read) -- so each row is
  //   dotted exactly once and the explicit dot pass's w read disappears.
  if (emit_w) {
    // The tail is a pass of its own, so unlike the riders interleaved into the
    // block visits it can be bracketed: one pair of calls per SAP call.  What
    // it measures is the Arnoldi matvec's close-out together with the fused GS
    // dots, which run in the same two passes and cannot be split off them.
    struct timeval tw0, tw1;
    gettimeofday(&tw0, NULL);
    // Fused-dot partials: double, reduced across both tail passes.
    // One (re, im) pair per (basis vector, rhs), rhs-fastest -- the same
    // split-complex layout the outer solve's hrow uses, so the handoff below
    // is a straight copy.  Sized by the basis vectors actually registered
    // rather than by a restart ceiling; the array section in the reduction
    // clauses below is what carries that length, exactly as the outer solve's
    // hrow[:hrn] does.  One element when nothing is registered, since a
    // zero-length section is not allowed -- the passes then never touch it
    // (every use is under `if (arn_n)`).  A few doubles per thread either
    // way: ~1.3 KB at the tuned restart, against block working sets of
    // hundreds of KB.
    const int arn_len = arn_n > 0 ? 2 * arn_n * num_rhs : 1;
    double arn_red[arn_len];
    for (int j = 0; j < arn_len; j ++) arn_red[j] = 0.0;
    struct timeval th0, th1;
    if (FERM_GHOSTS) gettimeofday(&th0, NULL);
    // the last in-sweep exchange predates the final half-sweep's visits;
    // pass 1's readers are the first-color blocks (compact reader_color 0)
    wilson_halo_sap(phi, phi + oddh, 0);
    g_sap_halo_tail_win += 2;
    if (FERM_GHOSTS) {
      gettimeofday(&th1, NULL);
      g_sap_halo_tail_secs += time_used(th0, th1);
    }
    #pragma omp parallel for schedule(static, 1) reduction(+:arn_red[:arn_len])
    for (int visit_i = 0; visit_i < sap_visit_n[0]; visit_i ++) {
      const int vcode = sap_visit_order[0][visit_i];  // first color
      const int x_outer = (vcode & 0xff)         * schwarz_block_size;
      const int y_outer = ((vcode >> 8) & 0xff)  * schwarz_block_size;
      const int z_outer = ((vcode >> 16) & 0xff) * schwarz_block_size;
      const int t_outer = ((vcode >> 24) & 0xff) * schwarz_block_size;
      REAL * temp = temp_th[omp_get_thread_num()];
      // odd sites: r_o = exterior hops of neighbor phi_e (the residual
      // seed the next sweep would have built), t = -D_oo^{-1} r_o into
      // phi_o.  Interior odd sites have no cross-block direction, so
      // r_o = 0 there -- write t = 0 so the w hop's interior directions
      // read clean data over the whole odd half.
      SCHWARZ_BLOCK_SITES {
        if (((x ^ y ^ z ^ t) & 0x1) != 1) continue;         // odd sites
        long rloc = site_half(x / 2, y, z, t);
        if (SCHWARZ_ON_FACE) {
          long ro = RO_LOC_IDX;
          for (int k = 0; k < site_size; k ++)
            temp[ro * site_size + k] = 0;
          wilson_kernel_block(temp + (ro - rloc) * site_size, phi,
                              U_kappa, x/2, y, z, t, 1,
                              schwarz_block_size, 1);
          for (int k = 0; k < site_size; k ++)
            temp[ro * site_size + k] = -temp[ro * site_size + k];
          if (clover)
            clover_kernel_ow(phi + oddh,
                             temp + (ro - rloc) * site_size,
                             F_oo_inv, x/2, y, z, t);
          else
            for (int k = 0; k < site_size; k ++)
              phi[oddh + rloc * site_size + k]
                = temp[ro * site_size + k];
        } else
          for (int k = 0; k < site_size; k ++)
            phi[oddh + rloc * site_size + k] = 0;
      }
      // even sites: one unmasked hop per row accumulates both
      // corrections into the in-sweep w_e = eta_e - r_e rows (exterior
      // directions: neighbor -psi_o, interior directions: this block's
      // t) -- see the tail comment above.
      SCHWARZ_BLOCK_SITES {
        if (((x ^ y ^ z ^ t) & 0x1) != 0) continue;        // even sites
        wilson_kernel(emit_w, phi + oddh, U_kappa, x/2, y, z, t, 0);
      }
      // fused GS dots, second phase of the visit: every even row of
      // this block is final once its hops are done (pass 2 touches only
      // last-color blocks), so dot the whole block as one burst.  The
      // block-major layout makes the span contiguous and this loop
      // order linear over it -- the same pure-stream shape (w row hot
      // across j, V slots linear streams) the explicit GS pass runs at
      // full bandwidth.  Dotting per site inside the hop loop instead
      // duty-cycles each core's streams behind the hop and flushes the
      // block's phi_o mid-loop (~20% slower on the moved bytes).
      if (arn_n)
        SCHWARZ_BLOCK_SITES {
          if (((x ^ y ^ z ^ t) & 0x1) != 0) continue;      // even sites
          long loc = site_half(x / 2, y, z, t);
          for (int j = 0; j < arn_n; j ++)
            arn_dot_row(arn_V + j * arn_stride + loc * site_size,
                        emit_w + loc * site_size, site_size,
                        arn_red + (2 * j) * num_rhs,
                        arn_red + (2 * j + 1) * num_rhs);
        }
    }
    // pass 2: last-color face rows pick up the cross-face Schur term.
    // Re-exchange the odd half first: rank-face ghosts must now carry t, not
    // the increments pass 1 read.  Readers are the last-color blocks
    // (compact reader_color 1), odd half only.
    if (FERM_GHOSTS) gettimeofday(&th0, NULL);
    wilson_halo_sap(NULL, phi + oddh, 1);
    g_sap_halo_tail_win += 1;
    if (FERM_GHOSTS) {
      gettimeofday(&th1, NULL);
      g_sap_halo_tail_secs += time_used(th0, th1);
    }
    #pragma omp parallel for schedule(static, 1) reduction(+:arn_red[:arn_len])
    for (int visit_i = 0; visit_i < sap_visit_n[1]; visit_i ++) {
      const int vcode = sap_visit_order[1][visit_i];   // last color
      const int x_outer = (vcode & 0xff)         * schwarz_block_size;
      const int y_outer = ((vcode >> 8) & 0xff)  * schwarz_block_size;
      const int z_outer = ((vcode >> 16) & 0xff) * schwarz_block_size;
      const int t_outer = ((vcode >> 24) & 0xff) * schwarz_block_size;
      SCHWARZ_BLOCK_SITES {
        if (((x ^ y ^ z ^ t) & 0x1) != 0) continue;        // even sites
        if (SCHWARZ_ON_FACE)
          wilson_kernel_block(emit_w, phi + oddh, U_kappa,
                              x/2, y, z, t, 0, schwarz_block_size, 1);
      }
      // fused GS dots, second phase (see pass 1): the block is final
      // once its face rows have the cross-face Schur term -- one linear
      // burst over the contiguous span covers face rows (cache-hot) and
      // in-sweep-final interior rows (streamed with the span) alike.
      if (arn_n)
        SCHWARZ_BLOCK_SITES {
          if (((x ^ y ^ z ^ t) & 0x1) != 0) continue;      // even sites
          long loc = site_half(x / 2, y, z, t);
          for (int j = 0; j < arn_n; j ++)
            arn_dot_row(arn_V + j * arn_stride + loc * site_size,
                        emit_w + loc * site_size, site_size,
                        arn_red + (2 * j) * num_rhs,
                        arn_red + (2 * j + 1) * num_rhs);
        }
    }
    if (arn_n) {
      for (int j = 0; j < 2 * arn_n * num_rhs; j ++) arn_dots[j] = arn_red[j];
      rbs_arn_dots_filled = 1;
    }
    rbs_emit_w_filled = 1;
    gettimeofday(&tw1, NULL);
    g_sap_tail_secs += time_used(tw0, tw1);
  }
  gettimeofday(&end, NULL);
  free(r);
  free(phi);
  // F_oo_inv is owned by the get_Doo_inv cache -- do not free it here.
  for (int q = 0; q < nthr_temp; q ++) free(temp_th[q]);
  free(temp_th);
  // Flop model for the block-internal odd-even MR (per lattice site, per rhs,
  // summed over one sweep -- the whole-solve total is this x n x num_rhs x V).
  // A sweep visits every block once; per block:
  //   per-sweep fixed cost (does not scale with sap_block_iter):
  //     - boundary residual update: the hopping the block solve excludes,
  //       1368/schwarz_block_size per site (each of 4 dims drops the +/- face
  //       coupling, 8/B directions x 171 flops = 1368/B);
  //     - eo setup (psi_o = D_oo^-1 eta_o; r_e -= D_eo psi_o) and recover
  //       (psi_o from the final psi_e): wilson_eo over half + wilson_oe over
  //       half = one full interior hopping (1368 - 1368/B); plus, with clover,
  //       two D_oo^-1 applies over half the sites each = one full clover (552).
  //     boundary + setup/recover hopping = 1368/B + (1368 - 1368/B) = 1368.
  //   per inner iteration (x sap_block_iter): the D_hat apply -- interior hopping
  //     (1368 - 1368/B) + clover (D_oo^-1 on odd + D_ee on even = one full
  //     clover) + the MR vector ops (sap_blk_dot3, 144 flops, plus
  //     sap_blk_update, 192 -- 336 per even site; even sites only, so
  //     336/2 = 168 per full site).
  long clover_flops  = clover ? CLOVER_FLOPS : 0;
  long interior_hop  = WILSON_FLOPS - WILSON_FLOPS / schwarz_block_size;
  long fixed_per_sweep = WILSON_FLOPS + clover_flops;  // boundary+setup+recover
  long per_inner       = interior_hop + clover_flops + 336 / 2;
  long flops_per_site  = fixed_per_sweep + (long) sap_block_iter * per_inner;
  double sap_secs  = time_used(start, end);
  double sap_flops = (double) flops_per_site * n * num_rhs * (double) V;
  // discard_odd: the terminal half-sweep skipped its recover -- wilson_oe +
  // D_oo^{-1} over the odd half (interior_hop/2 + clover_flops/2 per site) for
  // half the blocks, once per call.  Subtract so the GFlop/s stays honest.
  // (The skipped output_o accumulation is not flop-modeled -- pure streaming.)
  // Not under emit_w: the tail reads the terminal phi_o, so that recover ran.
  if (discard_odd && n > 0 && !emit_w)
    sap_flops -= 0.25 * (interior_hop + clover_flops) * num_rhs * (double) V;
  // Fold seeds (sweep 0): the block-local r = eta - D x formation adds one
  // conventional residual apply -- a full unmasked hop on both parity halves
  // (1368), both (A - 1) diagonal applies (one full clover; the fused-operator
  // convention, like CLOVER_FLOPS itself, regardless of which micro-op the
  // identity rides), and the eta combine (2*Nc*Ns reals per site; even half
  // only when eta is the eo outer's even half-field) -- and absorbs that
  // sweep's separate boundary hops (-1368/B).  Once per call, not per sweep.
  // The three bundled riders are kept in named locals as well as summed in, so
  // the profile can report what share of this row is not SAP
  // (sap_fusion_stats_get).
  double f_fold = 0, f_prolong = 0, f_arndots = 0;
  double f_emitw_tail = 0, f_emitw_insweep = 0;
  if (fold)
    f_fold = (double) (WILSON_FLOPS + clover_flops
                       + (fold_eta_even ? Nc * Ns : 2 * Nc * Ns)
                       - WILSON_FLOPS / schwarz_block_size)
           * num_rhs * (double) V;
  // Without the fold, the very first half-sweep (sweep 0, color 0) skips all
  // boundary hops (phi is uninitialized), but the per-sweep model above
  // charges them at every site -- subtract the skipped half-lattice worth,
  // once per call.  Largest relative effect on 1-sweep applies (bootstrap's
  // M_SAP), where it would otherwise inflate the rate by ~4%.
  else
    sap_flops -= 0.5 * (WILSON_FLOPS / schwarz_block_size)
               * num_rhs * (double) V;
  // emit_w (the free eo Arnoldi matvec), split at the sweep/tail boundary so
  // the tail's flops can pair with the tail's wall.  Tail: exterior seed hops
  // on the first-color blocks (half the full boundary set, both parities -- the
  // even half rides the unmasked w hop's exterior directions), D_oo^{-1} on
  // their odd sites (a quarter of a full clover), the block-internal t hop into
  // w on their even sites (a quarter of the interior hopping, the same hop's
  // interior directions), and the exterior t hops into last-color face rows
  // (another quarter of the boundary set).  In-sweep: the w = eta - r combine
  // on every even site (Nc*Ns per full site), done at each block's terminal
  // visit.  The t / -psi_o negations are sign bookkeeping, not counted.
  if (emit_w) {
    f_emitw_tail = (0.75 * (WILSON_FLOPS / schwarz_block_size)
                    + 0.25 * interior_hop + 0.25 * clover_flops)
                 * num_rhs * (double) V;
    f_emitw_insweep = (double) (Nc * Ns) * num_rhs * (double) V;
  }
  // Fused first-color prolongation: those blocks' P flops ran inside sweep 0
  // (the bulk apply_P was counted as a half call in the transfer stats).
  if (fusep_cin)
    f_prolong = rbs_fusep_blk_flops * (double) sap_visit_n[0];
  // Fused GS dots: arn_n basis vectors dotted against w, 8 flops per complex
  // MAC over the Nc*Ns complex components of each even site.  This is the outer
  // solve's dot pass, so fgmres charges only the caxpy half of its GS block on
  // the steps this runs (see the ortho_flops note there).
  if (arn_n)
    f_arndots = (double) arn_n * 8.0 * (Nc * Ns) * 0.5 * num_rhs * (double) V;
  sap_flops += f_fold + f_emitw_tail + f_emitw_insweep + f_prolong + f_arndots;
  // accumulate into the SAP profiling totals (consumed by the mixed summary,
  // which sum-reduces across ranks itself -- keep g_sap_flops rank-local here)
  g_sap_secs += sap_secs; g_sap_flops += sap_flops; g_sap_calls ++;
  g_sap_flops_fold += f_fold; g_sap_flops_prolong += f_prolong;
  g_sap_flops_emitw_tail += f_emitw_tail;
  g_sap_flops_emitw_insweep += f_emitw_insweep;
  g_sap_flops_arndots += f_arndots;
  return n;
}

#undef SCHWARZ_ON_FACE
#undef RO_LOC_IDX
#undef SCHWARZ_BLOCK_SITES

// Residual-fold SAP smoother: eta is the original rhs (not a pre-formed
// residual) and output is pre-loaded with an initial guess x (the fine
// V-cycle's prolonged coarse correction P psi_c).  Sweep 0 seeds the block
// residuals r = eta - D x block-locally -- DDalphaAMG's res == _RES smoother
// contract -- and output accumulates to x + M^nu (eta - D x).  This replaces
// the V-cycle's separate global fine-Dslash residual pass: the seed rides the
// block-resident gauge/clover/guess streams the block solve reuses anyway.
// eta_even_half: eta is the eo outer solve's even half-field (odd part
// implicitly zero), so the odd seed skips its eta term.
int red_black_schwarz_resfold(REAL * output, REAL * eta, REAL * U_kappa,
                              REAL * F_csw, int clover, int max_iter,
                              int sap_block_iter, REAL eps, int eta_even_half) {
  rbs_accumulate_next = 1;
  rbs_fold_next = 1;
  rbs_fold_eta_even_next = eta_even_half;
  return red_black_schwarz(output, eta, U_kappa, F_csw, clover, max_iter,
                           sap_block_iter, eps, 0, NULL);
}
