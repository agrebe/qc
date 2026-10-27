// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

// MG hierarchy driver.  Owns the multigrid hierarchy (gauge/clover copies,
// test vectors, coarse operators) and exposes the driver-facing interface in
// mixed.h.
//
// In a mixed build this TU is the bridge across the precision seam: it compiles
// at single precision, so the hierarchy it owns is float while the driver above
// it is double, and the plain operator/MG/utils names resolve to the _f
// instantiation via mangle.h.  In a one-precision build (MIXED=0) there is no
// seam -- the TU takes the build's own precision, the plain names resolve to
// that instantiation, and the float-typed solve-path entries below drop out
// (the outer solver reaches the cycle through the ordinary mangled mg_vcycle).
#ifdef MIXED
#define MG_SINGLE 1
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <complex.h>
#include <omp.h>
#include "global_params.h"   // REAL == float here; pulls in
                             //   precision.h + mangle.h
#include "coarse.h"          // mg_setup, mg_vcycle, build_coarse_op,
                             //   coarse_op, ...
#include "clover_ferm.h"
#include "solver.h"
#include "utils.h"
#include "bootstrap.h"    // bootstrap_null_vecs (adaptive setup)
#include "mixed.h"        // the double-typed bridge interface (literal double)
#include "comm.h"         // print0 + allreduce_sum_d for the MPI profile
                          // aggregation (rank-0 print, machine-wide flop totals)

// --- hierarchy state (owned by this module) ---------------------------------
// The _f suffixes name the role, not a C type: these are REAL, i.e. float in a
// mixed build (where they are the single-precision copies the seam exists for)
// and the build's own precision otherwise.
static REAL * g_Uk_f   = NULL;   // prepared gauge   (gauge_alloc_size REALs)
static REAL * g_Fcsw_f = NULL;   // prepared clover  (clover_size REALs)
static REAL * g_nv_f   = NULL;   // fine test vectors / P columns
// coarse P columns per level: entry d (d >= 1) holds level d's interpolation
// columns (entry 0 unused -- the fine columns are g_nv_f).  Allocated by the
// bootstrap driver (which alone knows each level's field size) or, on the
// file-read path, here; all freed in mg_hierarchy_free.
static REAL * g_nvc_f[NUM_LEVELS] = {0};
static coarse_op g_Dc_f;         // single fine->coarse operator
static int    g_Dc_built = 0;
static int    g_clover   = 0;

// The parameters this hierarchy was built with, kept so the profile report can
// name them.  A copy rather than the caller's pointer: the report outlives the
// solve, and the levels it would otherwise be read from are torn down by then.
static mg_params g_mgp;

// eo outer solve, single Schur matvec (mg_D_hat_single): prepared clover with
// the odd half inverted (D_hat applies D_oo^{-1} via the odd clover kernel) --
// a separate field because the cycle's smoother needs g_Fcsw_f's raw odd half
// -- plus D_hat's half-field odd scratch.  Both owned here, built by
// mg_setup_outer_eo_mixed, freed in mg_hierarchy_free.
static REAL * g_Fcsw_hat_f = NULL;
static REAL * g_eo_odd_f   = NULL;

void mg_hierarchy_build(mg_driver_real * U_kappa_d, mg_driver_real * F_csw_d,
                        int clover, const mg_params * p) {
  g_clover = clover;
  g_mgp    = *p;
  long csz = clover_size_full();

  // Store the hierarchy's own copy of the prepared fine operator fields,
  // down-converting on the way in (a mixed build's REAL is float; in a
  // one-precision build the cast is a no-op and this is a plain copy -- setup
  // cost only, and it keeps teardown's ownership rule uniform: the hierarchy
  // frees what it allocated, never the driver's arrays).
  // gauge_alloc_size: the tail gauge ghosts (already exchanged by prepare_gauge
  // on the double array) are converted along, so no second exchange is needed.
  g_Uk_f   = malloc(sizeof(REAL) * gauge_alloc_size);
  g_Fcsw_f = malloc(sizeof(REAL) * csz);
  #pragma omp parallel for
  for (long i = 0; i < gauge_alloc_size; i ++) g_Uk_f[i] = (REAL) U_kappa_d[i];
  if (clover) {
    #pragma omp parallel for
    for (long i = 0; i < csz; i ++) g_Fcsw_f[i] = (REAL) F_csw_d[i];
  }

  // fine test vectors -> orthonormalized P columns (single), generated in-code
  // by the adaptive bootstrap setup.
  g_nv_f = malloc(sizeof(REAL) * (long) ferm_size * num_null_vecs);

  if (MULTILEVEL) {
    // bootstrap_null_vecs_nlvl builds g_Dc_f, allocates every coarse level's P
    // columns into g_nvc_f[1..NUM_LEVELS-2], and registers the full NUM_LEVELS
    // hierarchy (fine + coarse test vectors generated in-code).
    bootstrap_null_vecs_nlvl(g_nv_f, &g_Dc_f, g_nvc_f, g_Uk_f, g_Fcsw_f, clover, p);
  } else {
    // bootstrap_null_vecs leaves g_Dc_f built and the two-level hierarchy
    // registered
    bootstrap_null_vecs(g_nv_f, &g_Dc_f, g_Uk_f, g_Fcsw_f, clover, p);
  }
  g_Dc_built = 1;
}

#ifdef MIXED
// Fused-input preconditioner: in_f already holds the single copy of the Krylov
// vector (the double fgmres emitted it in-place during normalization), so
// there is no down-convert pass here, and the V-cycle writes its single
// correction straight into the caller's single Z slot out_s -- so there is no
// up-convert either.  The outer matvec (D_si) and the FGMRES update read
// out_s as single directly.  In this TU REAL == float, so in_f / out_s slot
// straight into mg_vcycle.
//
// eo outer solve (eo_precondition = 1): in_f is an even-site half-field,
// consumed by the cycle in place (mg_vcycle(eo=1) routes it to apply_R_even /
// D_residual_eo, so no zero-padded copy exists on the input side).  The
// correction, though, is a full field, and it is written directly into out_s
// -- the caller must therefore give out_s room for a full field even though
// only its leading even half is the answer.  The discarded odd half spills
// into the next Z slot, which the caller has reserved for exactly that: the
// slots are filled in ascending order, so the spill lands in a slot that the
// next call overwrites with its own even half (or, for the last one, in slack
// the update loop never reads).  Writing in place is what keeps the eo path
// copy-free on both sides.
int mg_precond_single(float * out_s, float * in_f,
                      double * U_kappa_d, double * F_csw_d, int clover,
                      int max_iter, double eps, int eo_precondition,
                      void * prec) {
  (void) U_kappa_d; (void) F_csw_d; (void) clover;
  (void) max_iter;  (void) eps;     (void) prec;
  return mg_vcycle(out_s, in_f, g_Uk_f, g_Fcsw_f, g_clover,
                   /*max_iter*/ 0, /*eps*/ (REAL) 0, eo_precondition, NULL);
}

// Arnoldi matvec for the eo outer solve: w_e = D_hat z_e on the bridge's stored
// single operator (single gauge stream + the odd-inverted single clover), the
// Schur-system counterpart of mg_D_single below.  Same float-perturbed-operator
// caveat as mg_D_single; the double restart residual (D_hat on the driver's
// double F_hat) anchors final accuracy.  in/out are even-site half-fields.
void mg_D_hat_single(float * out_e, float * in_e) {
  D_hat(out_e, in_e, g_Uk_f, g_Fcsw_hat_f, g_eo_odd_f, g_clover);
}

// Free-matvec plumbing for the eo outer solve: forward the outer's w buffer to
// the single fine SAP's one-shot (plain names resolve to the _f instantiations
// in this TU).  The next red_black_schwarz call -- the depth-0 terminal
// smoother of the requested preconditioner apply -- assembles w_e = D_hat z_e
// from its exit residual; the filled query gates fgmres's explicit fallback.
void mg_request_arnoldi_w(float * w) { sap_emit_arnoldi_w_next(w); }
int mg_arnoldi_w_filled(void) { return sap_arnoldi_w_filled(); }
void mg_request_arnoldi_dots(const float * Vb, long stride, int nvec,
                             double * partials)
{ sap_emit_arnoldi_dots_next(Vb, stride, nvec, partials); }
int mg_arnoldi_dots_filled(void) { return sap_arnoldi_dots_filled(); }

// Arnoldi matvec on the bridge's stored single operator: w = D_f z, float
// in/out.  This is the fully-single path DDalphaAMG's mixed_precision=2 uses
// (apply_operator_float): single gauge/clover streams and the asm wilson
// kernel (the double TU's D / D_si run the C fallback -- the asm kernel exists
// only in single precision).  The operator itself is rounded to float
// (gauge/clover down-converted at setup), so the Arnoldi recurrence runs on a
// float-perturbed A~; the double restart residual anchors final accuracy
// regardless.  In this TU REAL == float and D resolves to the single
// instantiation, so the args slot straight in.
void mg_D_single(float * out_s, float * in_s) {
  D(out_s, in_s, g_Uk_f, g_Fcsw_f, g_clover);
}

// Fused normalize+restrict forward (see mixed.h): the plain name resolves to
// the single twin in this TU, where the live hierarchy and the one-shot
// registration state live.  In this TU REAL == float, so the args slot in.
int mg_normalize_restrict_single(float * vnext, float * w, const double * rinv,
                                 int eo) {
  return mg_restrict_fuse_normalize(vnext, w, rinv, eo);
}
#else  // !MIXED

// One-precision preconditioner entry: apply one MG cycle as a solver_function,
// for an outer solver that dispatches through solver_info.solver.
//
// This is not a precision shim -- in this build REAL already matches the
// hierarchy.  It exists because the operator fields the outer solver hands its
// preconditioner are the outer operator's, which are not the ones the cycle
// must run on.  Under an eo outer solve the driver gives fgmres F_hat, the
// clover with the odd block inverted (D_hat needs it); the cycle's smoother
// needs the raw clover -- the same distinction that makes the mixed build keep
// g_Fcsw_hat_f separate from g_Fcsw_f.  So U/F/clover are ignored here exactly
// as they are in mg_precond_single above, and the cycle runs on the
// hierarchy's own stored fields.  Passing mg_vcycle directly as .solver
// instead would silently smooth with the wrong operator and the solve would
// not converge.
int mg_precond(REAL * out, REAL * in, REAL * U_kappa, REAL * F_csw, int clover,
               int max_iter, REAL eps, int eo_precondition,
               solver_info * prec) {
  (void) U_kappa; (void) F_csw; (void) clover;
  (void) max_iter; (void) eps;  (void) prec;
  return mg_vcycle(out, in, g_Uk_f, g_Fcsw_f, g_clover,
                   /*max_iter*/ 0, /*eps*/ (REAL) 0, eo_precondition, NULL);
}

#endif // MIXED -- end of the precision-dependent solve-path entries

// eo outer solve setup: store the hierarchy's copy of the odd-inverted
// prepared clover (built by the driver as prepare_F(..., invert_odd = 1)) for
// the Schur matvec, and reserve D_hat's odd half-field scratch.  Call once
// after mg_hierarchy_build when the outer solve runs eo (OUTER_EO); harmless
// when it does not (the buffers just sit unused).
//
// One-precision build: there is no separate stored operator for the outer
// solve to apply -- fgmres calls D_hat on the driver's own U_kappa / F_csw --
// so there is nothing to store and this is a no-op.  Kept callable at both
// precisions so the driver's eo setup sequence reads the same either way.
void mg_setup_outer_eo_mixed(mg_driver_real * F_hat_d) {
#ifdef MIXED
  long csz = clover_size_full();
  if (g_eo_odd_f == NULL)
    g_eo_odd_f = malloc(sizeof(REAL) * ((long) ferm_size * num_rhs / 2));
  if (F_hat_d != NULL) {
    if (g_Fcsw_hat_f == NULL)
      g_Fcsw_hat_f = malloc(sizeof(REAL) * csz);
    #pragma omp parallel for
    for (long i = 0; i < csz; i ++) g_Fcsw_hat_f[i] = (REAL) F_hat_d[i];
  }
#else
  (void) F_hat_d;
#endif
}

// --- solve profiling --------------------------------------------------------
// Reset the single-precision (_f) V-cycle / transfer / coarse-Dslash / SAP
// timers.  These plain names resolve to the _f instantiations inside this TU,
// so this is the only place the double driver can clear them.  Call right
// before the outer fgmres solve.
void mg_profile_reset(void) {
  vcycle_stats_reset();
  transfer_stats_reset();
  apply_dc_stats_reset();
  apply_schur_stats_reset();
  coarse_solve_stats_reset();
  sap_stats_reset();
  other_stats_reset();
#if BOTTOM_SOLVE_DIAG
  bottom_solve_diag_reset();
#endif
}

// Print a DDalphaAMG-style profile of the mixed-precision solve: a
// per-category time breakdown (fractions of the total wall time) grouped by
// the level of the hierarchy that ran the work, plus total flops,
// flop/lattice-site and aggregate flop rate.  fg carries the double outer
// fgmres breakdown; the single inner V-cycle timers are read here from the _f
// statics.
//
// Every row belongs to exactly one level, and the levels partition the wall
// clock together with "other" -- so the level subtotals answer the question
// the flat table could not: which grid the solve is actually spending its time
// on.  A row that appears at several levels (Dslash, smoother, transfers) is
// the same code at a different width; a row that appears at only one is work
// that exists only there, and is named for what it does rather than for its
// level.
void mg_profile_report(const fgmres_profile_t * fg, int clover) {
  mg_vcycle_profile_t vp;
  mg_get_vcycle_profile(&vp);
  double sap_secs = 0, sap_flops = 0; long sap_calls = 0;
  sap_stats_get(&sap_secs, &sap_calls, &sap_flops);
  // Work inside that call belonging elsewhere in the solve (see
  // sap_fusion_stats_get).  The flop counts join the reduction below, so the
  // shares printed under the table divide the same machine-wide totals.
  sap_fusion_t sf;
  sap_fusion_stats_get(&sf);

  double vol = (double) V;    // (parenthesized: V is an unparenthesized macro)
  double T   = fg->total_secs;
  if (T <= 0) T = 1e-30;

  // --- per-category times ---------------------------------------------------
  // Each row pairs a wall-time leaf with the flops done in that same scope, so
  // its GFlop/s is a true efficiency.  The V-cycle residual Dslash has no row:
  // at depth 0 it is folded into the fine SAP (its time and flops ride that
  // row), and the depth>=1 residual is a coarse matvec already counted in the
  // coarse-Dslash row.
  double t_fineDa  = fg->hop_arn_secs;      // fine Dslash: Arnoldi w = A z
  double t_fineDr  = fg->hop_res_secs;      // fine Dslash: restart resid (dbl)
  double t_fgmres  = fg->ortho_secs;        // Arnoldi Gram-Schmidt
  // The emitted Arnoldi matvec's tail is a pass of its own inside the SAP call,
  // so it gets its own row and comes back out of the SAP row -- otherwise the
  // table would count the same seconds twice.  What stays in the SAP row is the
  // sweeps, still carrying the riders that are interleaved into them.
  double t_emitw   = sf.tail_secs;        // emitted matvec tail + GS dots
  double t_sapf    = sap_secs - t_emitw;  // fine-grid SAP sweeps
  double t_cgGS    = vp.cg_ortho_secs;    // coarse bottom-solve GS
  double t_eo      = vp.eo_secs;          // coarse e/o Schur setup/recovery
  double t_botd    = vp.botd_secs;        // dense bottom inverse apply (S^-1)
  double t_cgOth   = vp.cg_other_secs;    // fgmres_coarse Givens/back-subst/
                                          //   update
  // The transfer between level d and d+1 is charged to d, the finer side: it
  // is the finer field that both passes stream, so it is the finer level's
  // traffic.
  double t_tr[NUM_LEVELS];
  for (int d = 0; d < NUM_LEVELS; d ++)
    t_tr[d] = vp.lvl[d].R_secs + vp.lvl[d].P_secs;
  double t_named   = t_fineDa + t_fineDr + t_fgmres + t_sapf + t_emitw
                   + t_cgGS + t_eo + t_botd + t_cgOth;
  for (int d = 0; d < NUM_LEVELS; d ++)
    t_named += t_tr[d] + vp.lvl[d].dslash_secs + vp.lvl[d].smooth_secs
             + vp.lvl[d].kc_secs + vp.lvl[d].vec_secs;
  // True remainder: malloc/free in the recursive cycles, coarse_sap/MR loop
  // overhead outside the modeled scopes, OpenMP fork/join, timer skew.
  // By diff.
  double t_other   = T - t_named;
  if (t_other < 0) t_other = 0;

  // --- flops, paired with the matching time scope ---------------------------
  // fine Dslash flops/site/rhs
  long   fpsd      = clover ? WILSON_CLOVER_FLOPS : WILSON_FLOPS;
  double f_fineDa  = (double) fg->arn_applies * fpsd * vol * num_rhs;
  // the eo restart residual folds r = b - Ax and its norm into the D apply
  // (D_hat_residual), so those 3 flops per real element are charged here rather
  // than in the ortho row: 3 * Nc*Ns*2 per even site = half that per lattice
  // site.  Same total flops either way -- only the row they land in changes.
  long   fpsd_res  = fpsd + (fg->res_folded ? 3 * Nc * Ns * 2 / 2 : 0);
  double f_fineDr  = (double) fg->res_applies * fpsd_res * vol * num_rhs;
  double f_fgmres  = fg->ortho_flops;
  double f_sapf    = sap_flops;
  double f_cgGS    = vp.cg_ortho_flops;
  double f_eo      = vp.eo_flops;          // coarse e/o Schur block-gemv flops
  double f_botd    = vp.botd_flops;        // dense bottom inverse GEMV flops
#if FERM_GHOSTS
  // Aggregate the per-rank flop counts into machine-wide totals so the
  // GFlop/s columns reflect the whole job.  The time columns stay
  // rank-0-local: the ranks are SPMD-balanced, so rank 0 is representative and
  // the %total column stays coherent against its wall T.  Per-row rate is then
  // (all-rank flops) / (rank-0 time) = the aggregate machine throughput.  The
  // row counts (arn_applies, dslash_calls, ...) are iteration counts identical
  // on every rank, so each f_* = count * per-site-flops * local vol; summing
  // over ranks reconstructs count * per-site-flops * global vol exactly.
  //
  // Packed by a running index rather than by hand-numbered slots: the
  // per-level entries make the count depend on NUM_LEVELS, and a miscounted
  // slot here would silently move flops between rows.
  {
    // 7 solve-wide rows, 2 fusion rows, 5 per level, 3 more fusion rows.
    double red[12 + 5 * NUM_LEVELS];
    int n = 0;
    red[n ++] = f_fineDa; red[n ++] = f_fineDr; red[n ++] = f_fgmres;
    red[n ++] = f_sapf;   red[n ++] = f_cgGS;   red[n ++] = f_eo;
    red[n ++] = f_botd;
    red[n ++] = sf.fold_flops;        red[n ++] = sf.prolong_flops;
    for (int d = 0; d < NUM_LEVELS; d ++) {
      red[n ++] = vp.lvl[d].trans_flops;  red[n ++] = vp.lvl[d].dslash_flops;
      red[n ++] = vp.lvl[d].smooth_flops; red[n ++] = vp.lvl[d].kc_flops;
      red[n ++] = vp.lvl[d].vec_flops;
    }
    red[n ++] = sf.emitw_tail_flops;  red[n ++] = sf.emitw_insweep_flops;
    red[n ++] = sf.dot_flops;
    // The size above and the count here are edited in different places, so
    // check rather than trust: a row added below without widening the array
    // reduces past its end, which corrupts the stack instead of a printed
    // number.
    if (n != (int) (sizeof red / sizeof red[0]))
      comm_die("profile reduction packed %d values into %d slots",
               n, (int) (sizeof red / sizeof red[0]));
    allreduce_sum_d(red, n);
    n = 0;
    f_fineDa = red[n ++]; f_fineDr = red[n ++]; f_fgmres = red[n ++];
    f_sapf   = red[n ++]; f_cgGS   = red[n ++]; f_eo     = red[n ++];
    f_botd   = red[n ++];
    sf.fold_flops = red[n ++];        sf.prolong_flops = red[n ++];
    for (int d = 0; d < NUM_LEVELS; d ++) {
      vp.lvl[d].trans_flops = red[n ++];  vp.lvl[d].dslash_flops = red[n ++];
      vp.lvl[d].smooth_flops = red[n ++]; vp.lvl[d].kc_flops     = red[n ++];
      vp.lvl[d].vec_flops   = red[n ++];
    }
    sf.emitw_tail_flops = red[n ++];  sf.emitw_insweep_flops = red[n ++];
    sf.dot_flops = red[n ++];
  }
#endif
  // The emitted matvec's tail row takes its flops out of the SAP row, matching
  // the wall that moved with it.  Both pieces run in the tail's two passes.
  double f_emitw = sf.emitw_tail_flops + sf.dot_flops;
  f_sapf -= f_emitw;
  double F = f_fineDa + f_fineDr + f_fgmres + f_sapf + f_emitw
           + f_cgGS + f_eo + f_botd;    // Givens: no flops
  for (int d = 0; d < NUM_LEVELS; d ++)
    F += vp.lvl[d].trans_flops + vp.lvl[d].dslash_flops + vp.lvl[d].smooth_flops
       + vp.lvl[d].kc_flops + vp.lvl[d].vec_flops;

  // the preconditioner cycle type comes from the parameters this hierarchy was
  // built with; mg_vcycle is the single entry point for all three shapes.
  const char * cyc = g_mgp.cycle == MG_CYCLE_W ? "W"
                   : g_mgp.cycle == MG_CYCLE_K ? "K" : "V";

  #define GFR(flops, secs) ((secs) > 0 ? 1e-9 * (flops) / (secs) : 0.0)
  print0("\n");
  print0("+--------------------------------------------------------------------+\n");
  print0("| mixed-precision solve profiling (double FGMRES + single %s-cycle)   |\n",
         cyc);
  print0("+--------------------------------------------------------------------+\n");
  print0("| FGMRES: %3d outer iters, %5d coarse bottom-solve iters           |\n",
         fg->outer_iters, fg->inner_iters);
  print0("| precond %s-cycle applies: %4ld   mg_cycle calls (all levels): %4ld  |\n",
         cyc, vp.vc_calls_fine, vp.vc_calls);
  print0("| fine A=D applies: %3ld+%-3ld total solve wall time: %10.4f s      |\n",
         fg->arn_applies, fg->res_applies, T);
  print0("+--------------------------------------------------------------------+\n");
  print0("| category                          time(s)   %%total    GFlop/s      |\n");
  // A row prints only if it carries time or flops.  Most of this table is
  // config-conditional: the Arnoldi-matvec row is empty whenever the terminal
  // fine SAP emits w for free (OUTER_EO), the two bottom-GMRES rows whenever
  // the coarsest solve is the dense inverse (BOTTOM_DENSE_INV), and a level's
  // rows wherever the hierarchy is too shallow to reach it.  Those paths are
  // all still live in other builds, so the rows stay -- they just don't print
  // a line of zeros in the builds that skip them.
  // Suppressing a zero row cannot hide time: it contributes zero to t_named,
  // and anything genuinely untimed still lands in "other" (= T - t_named).
  #define ROW(label, t, f)                                                    \
    do { if ((t) > 0 || (f) > 0)                                              \
           print0("| %-33s%8.4f   %5.1f%%   %8.2f      |\n",                  \
                  (label), (t), 100 * (t) / T, GFR(f, t)); } while (0)
  #define ROW_NOFLOP(label, t)                                                \
    do { if ((t) > 0)                                                         \
           print0("| %-33s%8.4f   %5.1f%%   %8s      |\n",                    \
                  (label), (t), 100 * (t) / T, "--"); } while (0)
  // Arnoldi-matvec row label reflects the variant fgmres was built with
  const char * arnL = fg->arn_prec == 1 ? "  Dslash Arnoldi w = A z (single)"
                    :                     "  Dslash Arnoldi w = A z (double)";
  for (int d = 0; d < NUM_LEVELS; d ++) {
    const mg_level_profile_t * L = &vp.lvl[d];
    const level_t * lev = mg_get_level(d);
    // Banner carries the level's extents and site width, the two numbers that
    // set what its rows can possibly achieve.  Extents are global, matching the
    // machine-wide flop columns: level_t holds this rank's local block, and
    // every level is decomposed by the same z/t rank grid, so scaling those two
    // by the grid recovers the whole lattice at any depth.  Padded out to the
    // table rule so the block reads as a section and not as a row.
    char bar[96];
    int n = snprintf(bar, sizeof bar, "+-- level %d: %dx%dx%dx%d x %d vars %s",
                     d, lev->nx, lev->ny, lev->nz * MPI_GRID_PZ,
                     lev->nt * MPI_GRID_PT, lev->nvar,
                     d == 0 ? "(fine) "
                      : d == NUM_LEVELS - 1 ? "(coarsest) " : "");
    while (n < 69) bar[n ++] = '-';
    bar[n ++] = '+'; bar[n] = 0;
    print0("%s\n", bar);
    double t_sub = t_tr[d] + L->dslash_secs + L->smooth_secs + L->kc_secs
                 + L->vec_secs;
    double f_sub = L->trans_flops + L->dslash_flops + L->smooth_flops
                 + L->kc_flops + L->vec_flops;
    if (d == 0) {
      // Level 0's operator work is the outer solve's, so it comes from fgmres
      // and the fine SAP rather than from the cycle's own brackets -- those see
      // finer scopes (the emitted matvec's tail, the restart residual's fold).
      ROW(arnL,                              t_fineDa, f_fineDa);
      // The same matvec on the emitted path, where the terminal fine SAP
      // produces w instead of an explicit apply.  Only one of the two rows is
      // ever live: the emit is requested exactly when a preconditioner runs on
      // the eo system, and the cycle's one depth-0 smoother always consumes
      // it.  Named apart because the scopes differ -- this is a close-out pass
      // and it carries the fused GS dots, so its rate is not comparable with
      // the explicit apply's.
      ROW("  Dslash Arnoldi emit + GS dots", t_emitw,  f_emitw);
      ROW("  Dslash restart residual (dbl)", t_fineDr, f_fineDr);
      ROW("  outer FGMRES (Arnoldi/GS)",     t_fgmres, f_fgmres);
      ROW("  SAP smoother (single)",         t_sapf,   f_sapf);
      t_sub += t_fineDa + t_emitw + t_fineDr + t_fgmres + t_sapf;
      f_sub += f_fineDa + f_emitw + f_fineDr + f_fgmres + f_sapf;
    } else {
      ROW("  Dslash",                        L->dslash_secs,  L->dslash_flops);
      ROW("  SAP smoother",                  L->smooth_secs,  L->smooth_flops);
      ROW("  K-cycle accel (FGMRES GS)",     L->kc_secs,      L->kc_flops);
      ROW("  vector ops (psi += smooth)",    L->vec_secs,     L->vec_flops);
    }
    if (d == NUM_LEVELS - 1) {
      // Bottom-solve rows.  Each exists only here, so none of them is per
      // level.
      ROW("  e/o Schur setup",               t_eo,     f_eo);
      ROW("  bottom dense inverse apply",    t_botd,   f_botd);
      ROW("  bottom-solve GS",               t_cgGS,   f_cgGS);
      ROW_NOFLOP("  bottom GMRES Givens/upd", t_cgOth);
      t_sub += t_eo + t_botd + t_cgGS + t_cgOth;
      f_sub += f_eo + f_botd + f_cgGS;
    }
    // The transfer sits at the foot of the finer level's block because that is
    // the level whose field it streams, and reads as the seam into the next.
    char seam[40];
    snprintf(seam, sizeof seam, "  restrict/prolong %d <-> %d", d, d + 1);
    ROW(seam,                                t_tr[d],  L->trans_flops);
    // Subtotal, marked with a leading "=" so it does not read as another row to
    // be summed.  Its rate is a blend across the rows above, useful only for
    // comparing one level's aggregate efficiency against another's.
    char sub[40];
    snprintf(sub, sizeof sub, "= level %d total", d);
    ROW(sub,                                 t_sub,    f_sub);
  }
  print0("+--------------------------------------------------------------------+\n");
  // "other" always prints: it is the leftover T - t_named, so a zero there is
  // itself the result (the named rows accounted for the whole wall).
  print0("| other (malloc/fork/skew)         %8.4f   %5.1f%%   %8s      |\n",
         t_other, 100 * t_other / T, "--");
  #undef ROW
  #undef ROW_NOFLOP
  print0("+--------------------------------------------------------------------+\n");
  print0("| total                            %8.4f   100.0%%                 |\n", T);
  print0("+--------------------------------------------------------------------+\n");
  print0("| instrumented flops:        %10.3e  (all flop-bearing rows)     |\n", F);
  // F is the machine-wide (all-rank) flop sum; vol is this rank's local volume,
  // so the global site count is vol * nranks (1 at MPI=0 -> byte-identical).
  print0("|   flop / lattice site:     %10.3e                              |\n",
         F / (vol * qc_grid.nranks));
  print0("|   aggregate flop rate:     %10.2f GFlop/s                      |\n",
         GFR(F, T));
  print0("+--------------------------------------------------------------------+\n");
  #undef GFR

  // What is left in the SAP row that is not SAP.  The tail came out above into
  // its own row, but these are interleaved into the block visits site by site,
  // sharing the loads the block solve is about to reuse -- so they can be
  // attributed in flops and not in time, and no further row can be split off.
  double sf_bundled = sf.fold_flops + sf.prolong_flops
                    + sf.emitw_insweep_flops;
  if (f_sapf > 0 && sf_bundled > 0)
    print0("[sap-row] bundled into the SAP row: residual fold %.1f%%, "
           "prolongation %.1f%%, emitted w rows %.1f%% of its flops "
           "(SAP proper %.1f%%)\n",
           100 * sf.fold_flops / f_sapf, 100 * sf.prolong_flops / f_sapf,
           100 * sf.emitw_insweep_flops / f_sapf,
           100 * (f_sapf - sf_bundled) / f_sapf);

#if BOTTOM_SOLVE_DIAG
  bottom_solve_diag_report();
#endif

#if FERM_GHOSTS
  // Serial halo wall inside the fine SAP (blocking exchanges; every core
  // idles for the duration), split main-sweep vs emit-tail.  Rank-0 times,
  // like every row above.
  {
    double hs = 0, ht = 0; long ws = 0, wt = 0;
    sap_halo_stats_get(&hs, &ws, &ht, &wt);
    // Spans both rows now -- the sweep windows are in the SAP row, the tail
    // windows in the emitted-matvec row -- so the share is of the whole call.
    if (ws + wt > 0)
      print0("[sap-halo] sweep %.4f s (%ld win, %.2f ms/win) | tail %.4f s "
             "(%ld win, %.2f ms/win) | %.1f%% of the fine SAP call\n",
             hs, ws, ws > 0 ? 1e3 * hs / ws : 0.0,
             ht, wt, wt > 0 ? 1e3 * ht / wt : 0.0,
             sap_secs > 0 ? 100.0 * (hs + ht) / sap_secs : 0.0);
    // Window decomposition of the compact exchange: threaded pack + expand
    // vs the blocking MPI copy of the quarter-size payload.  A pack share
    // worth seconds is the trigger for fusing the emit into the block visits.
    double cp = 0, cm = 0, cu = 0;
    wilson_halo_sap_stats_get(&cp, &cm, &cu);
    if (cp + cm + cu > 0)
      print0("[sap-cmp] pack %.4f s | mpi %.4f s | expand %.4f s\n",
             cp, cm, cu);
  }
#endif
}

void mg_hierarchy_free(void) {
  if (g_Dc_built) { free_coarse_op(&g_Dc_f); g_Dc_built = 0; }
  free(g_Uk_f);   g_Uk_f   = NULL;
  free(g_Fcsw_f); g_Fcsw_f = NULL;
  free(g_Fcsw_hat_f); g_Fcsw_hat_f = NULL;
  free(g_eo_odd_f);   g_eo_odd_f   = NULL;
  // drop the single SAP D_oo^{-1} built from g_Fcsw_f
  sap_free_doo_inv_cache();
  free(g_nv_f);   g_nv_f   = NULL;
  for (int d = 0; d < NUM_LEVELS; d ++) { free(g_nvc_f[d]); g_nvc_f[d] = NULL; }
}
