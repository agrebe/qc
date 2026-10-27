// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef SOLVER_H
#define SOLVER_H
#include <sys/time.h>   // struct timeval, used by time_used() below
#include "global_params.h"
typedef struct solver_info solver_info;
typedef int (* solver_function)(
    REAL *, REAL *, REAL *, REAL *,
    int, int, REAL, int, solver_info *
);
// struct to store info about solvers
struct solver_info {
    solver_function solver;
    solver_info * preconditioner;
    int max_iter;
    REAL eps;
};

REAL time_used(struct timeval start, struct timeval end);
void prepare_gauge(REAL * U_kappa, REAL * U, REAL kappa);
void prepare_F(REAL * F_csw, REAL * F, REAL kappa, REAL csw, int invert_odd);
void solve(REAL * output, REAL * ferm, REAL * U, REAL * F,
           REAL kappa, REAL csw,
           int chroma_norm, int eo_precondition,
           solver_info solver);
int richardson(REAL * output, REAL * ferm, REAL * U_kappa, REAL * F_csw,
              int clover, int max_iter, REAL eps,
              int eo_precondition, solver_info * preconditioner);
int bicgstab(REAL * output, REAL * ferm, REAL * U_kappa, REAL * F_csw,
             int clover, int max_iter, REAL eps,
             int eo_precondition, solver_info * preconditioner);
#ifdef MIXED
// bicgstab at single, called from double: takes the driver's double fields,
// down-converts them and the rhs, solves, and converts the answer back.
// Defined only by the single twin (see solver.c) and not mangled, so it is
// one shared symbol across the precision seam.  Intended as richardson's
// inner solver in a mixed build, where the outer defect and convergence test
// stay double.  Not solver_function-shaped -- the trailing argument is void *
// because solver_info is not the same type on the two sides of the seam -- so
// a driver installing it in a solver_info.solver slot wraps it.
int bicgstab_mixed(double * out_d, double * in_d, double * U_d, double * F_d,
                   int clover, int max_iter, double eps, int eo_precondition,
                   void * prec);
#endif
// Flexible (right-preconditioned) GMRES.  Applies a variable preconditioner M
// (e.g. mg_vcycle) to each Krylov vector and builds the solution from the
// preconditioned basis Z = M^{-1} V.  Intended as the outer solver with the MG
// V-cycle as preconditioner.  eps is relative to ||ferm||.  Assumes a
// full-system preconditioner => call with eo_precondition = 0.
// preconditioner == NULL runs plain restarted GMRES (M = I; the Z basis aliases
// V) -- see the header comment on fgmres() in solver.c for that mode's
// build-config caveats.  A fixed preconditioner (e.g. richardson) needs no
// special mode, but is honored only where FGMRES_SINGLE_KRYLOV is off (the
// single twin and the one-precision MIXED=0 builds); same comment for why.
//
// max_restart is the restart length: Krylov vectors per cycle, lowered to
// max_iter if that is smaller.  FGMRES stores both a V-basis and a
// preconditioned Z-basis (each ~ferm_size * num_rhs reals per vector), so it is
// the solver's memory high-water mark and is normally kept short.  Like
// red_black_schwarz below, this takes one more argument than solver_function
// and is never installed in a solver_info.solver slot.
int fgmres(REAL * output, REAL * ferm, REAL * U_kappa, REAL * F_csw,
           int clover, int max_iter, int max_restart, REAL eps,
           int eo_precondition, solver_info * preconditioner);
// Schwarz alternating-procedure smoother.  max_iter is the number of sweeps
// over the block coloring; sap_block_iter is the minimal-residual iteration
// count inside each block, per visit.  Both are runtime (mg_params), which is
// why this and its resfold variant take one more argument than the
// solver_function family above -- neither is ever installed in a
// solver_info.solver slot.
int red_black_schwarz(REAL * output, REAL * ferm, REAL * U_kappa, REAL * F_csw,
                      int clover, int max_iter, int sap_block_iter, REAL eps,
                      int eo_precondition, solver_info * preconditioner);
// As red_black_schwarz but with the fine V-cycle's residual fold: eta is the
// original rhs and output is pre-loaded with an initial guess x (P psi_c);
// sweep 0 seeds the block residuals r = eta - D x block-locally and output
// accumulates to x + M^nu (eta - D x), replacing the separate global fine
// residual Dslash.  eta_even_half: eta is the eo outer solve's even half-field
// (odd part implicitly zero).
int red_black_schwarz_resfold(REAL * output, REAL * eta, REAL * U_kappa,
                              REAL * F_csw, int clover, int max_iter,
                              int sap_block_iter, REAL eps, int eta_even_half);

// One-shot (consumed by the next red_black_schwarz call): treat the odd half
// of the output as dead -- skip the per-visit output_o accumulation and the
// terminal half-sweep's odd recover.  For the eo outer solve, where the
// cycle's odd output is discarded at the bridge slice; even-site output is
// bit-identical.  See the rbs_discard_odd_next comment in solver.c for the
// liveness argument.
void sap_discard_odd_output_next(void);

// One-shot (consumed by the next red_black_schwarz call; resfold entry only):
// the initial guess in `output` was bulk-prolonged only for the
// second-processed block color -- each first-color sweep-0 visit prolongs its
// own block on the fly (apply_P_block_cached) right before its fold seed
// reads it, so that half of the guess never makes a DRAM round trip.  Sound
// because the raw first-color guess has exactly one reader (its own fold
// seed): second-color seeds read neighbor output that already carries the
// first color's recover updates.  coarse_in/nv are the prolongation operands;
// blk_flops the P flops per fine block (added to the smoother's tally -- the
// bulk call is counted as a half apply_P).  Returns 1 (registration made); a
// caller that cannot use the fused path (aggregation and SAP blockings differ)
// must not call this and must bulk-prolong all blocks instead.
int sap_fuse_prolong_next(REAL * coarse_in, REAL * nv, double blk_flops);
// Local block parity ((bx^by^bz^bt) & 1) of the smoother's second-processed
// color, i.e. the color the fused-prolong caller must bulk-prolong.
int sap_bulk_blk_parity(void);

// One-shot out-param (consumed by the next red_black_schwarz call): also emit
// the eo outer solve's Arnoldi matvec w_e = D_hat psi_e into w, assembled from
// the smoother's exit residual (w_e = eta_e - r_e + D_eo D_oo^{-1} r_o) plus a
// face-only tail, instead of a separate explicit Schur matvec.  Requires the
// resfold entry with an even-half eta (the eo outer smoother shape).
// sap_arnoldi_w_filled() reports whether the last SAP call consumed a pending
// request; fgmres applies the explicit D_hat when it did not.
void sap_emit_arnoldi_w_next(REAL * w);
int sap_arnoldi_w_filled(void);

// Optional companion to the w request: the emit tail also accumulates the outer
// GS dot products h_j = conj(V[j]) . w while it holds each finalized w row
// cache-hot (pass 1 owns every first-color row, pass 2 every last-color row),
// deleting the explicit dot pass's full-field w read.  Vb is the basis (nvec
// slots, stride REALs apart); partials receives 2*nvec (re,im) local sums --
// the caller allreduces.  Accumulation is double regardless of REAL, matching
// the outer twin's promote-to-double products.  Consumed by the same call that
// consumes the w request; never filled without one, or if nvec exceeds
// FGMRES_RESTART.  sap_arnoldi_dots_filled() reports whether the last SAP call
// filled partials; the caller runs the explicit dot pass when it did not.
void sap_emit_arnoldi_dots_next(const REAL * Vb, long stride, int nvec,
                                double * partials);
int sap_arnoldi_dots_filled(void);

// Free the cached SAP D_oo^{-1} field (built lazily per F_csw by
// red_black_schwarz).  Call when the F_csw it was built from is freed, so a
// reused malloc address cannot alias a stale inverse: solve() for the double
// path, mg_hierarchy_free() for the single hierarchy.
void sap_free_doo_inv_cache(void);

// --- fine-grid SAP (red_black_schwarz) profiling ----------------------------
// Wall time, call count, and modeled flops for the fine-grid Schwarz smoother,
// summed over all invocations between reset() and the getter call.  Consumed
// by the mixed-precision profiling summary; the flop model is the one
// red_black_schwarz prints as "Speed of Schwarz".
void sap_stats_reset(void);
void sap_stats_get(double * secs, long * calls, double * flops);
// Serial MPI halo wall accumulated inside red_black_schwarz (reset by
// sap_stats_reset), split between the per-half-sweep exchanges (sweep) and the
// emit-w tail's (tail), with window counts.  Each window is a blocking
// exchange during which every core idles.
void sap_halo_stats_get(double * sweep_secs, long * sweep_win,
                        double * tail_secs, long * tail_win);
// Work that runs inside red_black_schwarz but belongs to another part of the
// solve: the V-cycle residual fold, the fused first-color prolongation, the
// outer Arnoldi matvec (emit-w) and the outer Gram-Schmidt dots.  Their flops
// are a subset of what sap_stats_get reports, split back out here so the
// profile can attribute them.
//
// Only the tail is separable in time -- it is a pass of its own, after the
// sweeps -- and it carries the matvec's close-out and the dots together, which
// is why they share one wall.  Everything else is interleaved into the block
// visits site by site, sharing the loads the block solve is about to reuse;
// that is the point of fusing it, and also why there is no time to attribute
// to it separately.  So the flop split is exact, and only the tail's part of
// it carries over to a time.
typedef struct {
  double fold_flops;           // block-local r = eta - D x seeds (sweep 0)
  double prolong_flops;        // first-color prolongation, done in-visit
  double emitw_insweep_flops;  // emitted Arnoldi matvec: the w = eta - r rows
  double emitw_tail_flops;     //   ... and its two close-out passes
  double dot_flops;            // outer GS dots, inside the tail passes
  double tail_secs;            // wall of the tail (emitw_tail + dot work)
} sap_fusion_t;
void sap_fusion_stats_get(sap_fusion_t * p);

// --- outer FGMRES profiling -------------------------------------------------
// One-shot snapshot of the last fgmres() call's time breakdown.  hop = fine
// operator A applies, split into the per-iteration Arnoldi matvecs w = A z
// (whose precision depends on FGMRES_SINGLE_KRYLOV; see arn_prec) and
// the once-per-restart residual applies A x (always double -- the defect-
// correction anchor).  prec = preconditioner (the V-cycle), ortho = Arnoldi
// Gram-Schmidt + Givens + solution update (the outer Krylov vector work).
typedef struct {
  double total_secs;   // whole fgmres() call
  double prec_secs;    // preconditioner->solver (V-cycle)
  double hop_secs;     // fine operator A = D applies (arn + res, total)
  double hop_arn_secs; // ... Arnoldi matvec w = A z subset
  double hop_res_secs; // ... restart residual A x subset (double D)
  double ortho_secs;   // Arnoldi orthogonalization / Givens / update
  double ortho_flops;  // flops in the orthogonalization / solution-update work
  long   d_applies;    // number of fine A = D applies (total)
  long   arn_applies;  // ... Arnoldi matvecs (= outer iterations)
  long   res_applies;  // ... restart residuals (= restart cycles)
  long   res_folded;   // 1 = r = b - Ax and its norm are folded into the
                       // restart D apply (eo path, D_hat_residual), so their
                       // flops belong to the restart-D row, not to ortho_flops
  long   arn_prec;     // Arnoldi matvec kind: 0 = double output (D / D_si),
                       // 1 = mg_D_single (fully single, asm kernel)
  int    outer_iters;  // FGMRES iterations
  int    inner_iters;  // total preconditioner iterations
} fgmres_profile_t;
void fgmres_get_profile(fgmres_profile_t * p);
#endif
