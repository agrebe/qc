// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MIXED_H
#define MIXED_H
#include "solver.h"   // fgmres_profile_t (layout-identical across the
                      //   precision seam)
#include "mg_params.h"
// MG hierarchy driver, and -- in a mixed build -- the bridge across the
// precision seam.
//
// mixed_bridge.c owns the entire multigrid hierarchy (test vectors, coarse
// operator, V-cycle scratch) and exposes the driver-facing interface below.
// In the mixed build (MIXED=1) it is compiled at single precision, so the
// hierarchy is float while the driver above it is double: the setup/profiling
// entry points take literal `double` and the solve-path entry points take
// literal `float`, and all of these names are deliberately absent from
// mangle.h so they form one shared interface across the seam (declared here,
// defined at the TU's own precision there).  In a one-precision build
// (MIXED=0, single or double) it compiles at that same precision, there is no
// seam, and the float-typed solve-path entries below are replaced by the
// single REAL-typed mg_precond, which the outer solver dispatches through
// solver_info.solver.

// Field type on the driver side of the interface.  In a mixed build the driver
// is always double no matter which TU includes this header, so the
// seam-crossing entries are literally double.  In a one-precision build there
// is one REAL in the whole binary and the driver shares it -- which is what
// lets MIXED=0 MG_SINGLE=1 (the all-single asm build) hand its float fields
// to the same setup entry point.
#ifdef MIXED
typedef double mg_driver_real;
#else
typedef REAL   mg_driver_real;
#endif

// Build the MG hierarchy from the driver's prepared fine operator fields
// U_kappa_d / F_csw_d (as the driver builds them at eo_precondition = 0),
// down-converting them into the hierarchy's own precision on the way in.  The
// adaptive bootstrap setup generates and orthonormalizes the test vectors,
// builds the coarse operators, and registers every level -- two below
// MULTILEVEL, NUM_LEVELS at or above it.  Call once.
//
// The 2-vs-multilevel choice is read from MULTILEVEL rather than passed in:
// it is derived from NUM_LEVELS, which sizes the level array this builds into,
// so a caller asking for a depth the build is not sized for is not a request
// worth being able to express.
void mg_hierarchy_build(mg_driver_real * U_kappa_d, mg_driver_real * F_csw_d,
                        int clover, const mg_params * p);

// --- solve-path entries, split by build -------------------------------------
// The MIXED entries take literal `float` because they exist to be called from
// the double side of the seam; declaring them at MIXED=0 would mean float
// prototypes over a hierarchy that is not float.  A one-precision build has no
// seam -- its outer solver's REAL already matches the hierarchy's -- so it
// gets one REAL-typed preconditioner entry instead, and reaches the fine
// operator through the ordinary D / D_hat with no bridging at all.
#ifdef MIXED

// Fused-input preconditioner for the mixed outer fgmres.  The caller (double
// fgmres_d) already emitted the single-precision copy of the Arnoldi vector
// in-place while scaling/normalizing it, so this skips the double->single
// down-convert and runs the V-cycle directly on the supplied single field
// in_f, writing the single correction straight into out_s (the permanent
// single Z-basis slot).  No conversion at all: the outer matvec reads out_s
// as float (promote-on-load) and the FGMRES solution update reads it via a
// float->double CAXPY.  in_f / out_s are full multi-rhs single fine fields
// (ferm_size*num_rhs floats).  The U_kappa_d / F_csw_d args from fgmres_d are
// ignored -- the bridge uses its own stored single fields.
//
// The cycle itself is always full-system.  With eo_precondition = 1 (eo outer
// solve, OUTER_EO) in_f is an even-site half-field and the bridge zero-pads /
// even-slices around the unchanged cycle -- S D^{-1} E = D_hat^{-1} makes the
// full-system cycle an exact-form preconditioner for the outer Schur operator.
// out_s is not a half-field, though: the correction is a full field written in
// place, of which only the leading even half is the answer.  The caller must
// give out_s room for a full field -- in fgmres that is the slack slot
// appended to the Z basis, which absorbs the discarded odd half of the last
// vector.
int mg_precond_single(float * out_s, float * in_f,
                      double * U_kappa_d, double * F_csw_d, int clover,
                      int max_iter, double eps, int eo_precondition,
                      void * prec);

// Arnoldi matvec via the bridge's stored single-precision fine operator (asm
// wilson kernel, single gauge/clover streams).  Used by the outer fgmres on its
// FGMRES_SINGLE_KRYLOV path; requires mg_hierarchy_build to have been called.
void mg_D_single(float * out_s, float * in_s);

// Fused normalize+restrict for the outer Arnoldi step (FGMRES_SINGLE_KRYLOV,
// where w is float): scale w by 1/hmp1 into the float basis slot vnext
// (= V[m+1]) and restrict the result to the coarse grid in the same pass,
// pre-computing the next V-cycle's depth-0 restriction (registered one-shot
// inside the single hierarchy; that cycle skips its restrict).  eo = eo outer
// solve (even-site half-fields).  Returns 1 if fused (vnext filled; the pass
// timed itself into the restrict rows); 0 = caller normalizes itself and the
// cycle restricts as usual (no hierarchy yet).
int mg_normalize_restrict_single(float * vnext, float * w, const double * rinv,
                                 int eo);

// eo outer solve (OUTER_EO) support.  mg_setup_outer_eo_mixed stores the single
// copy of the odd-inverted prepared clover (driver-built: prepare_F with
// invert_odd = 1) and reserves D_hat's odd scratch -- call once after
// mg_hierarchy_build.  mg_D_hat_single is then the Schur-system Arnoldi matvec
// w_e = D_hat z_e on even-site half-fields (the eo counterpart of mg_D_single).
// mg_precond_single needs no extra setup for eo: it zero-pads /
// even-slices around the unchanged full-system cycle internally.
void mg_D_hat_single(float * out_e, float * in_e);

// Free eo Arnoldi matvec: ask the next depth-0 fine SAP call -- the terminal
// smoother of the upcoming preconditioner apply -- to emit w_e = D_hat z_e
// into w from its exit residual, making mg_D_hat_single a fallback.
// mg_arnoldi_w_filled() reports whether that request was consumed and filled.
void mg_request_arnoldi_w(float * w);
int mg_arnoldi_w_filled(void);

// Companion request: the same emit tail also accumulates the outer GS dot
// products h_j = conj(V[j]) . w row-by-row while each finalized w row is
// cache-hot (see sap_emit_arnoldi_dots_next), so the explicit dot pass and its
// full-field w read are skipped when this is consumed.  partials receives
// 2*nvec double (re,im) rank-local sums; the caller allreduces.
void mg_request_arnoldi_dots(const float * Vb, long stride, int nvec,
                             double * partials);
int mg_arnoldi_dots_filled(void);

#else  // !MIXED

// One-precision preconditioner entry, for an outer solver that dispatches
// through solver_info.solver.  Signature is exactly solver_function.  Not a
// precision shim: it exists because the operator fields the outer solver passes
// its preconditioner are the outer operator's (under an eo outer solve, the
// odd-inverted F_hat that D_hat needs), while the cycle's smoother must run on
// the raw operator the hierarchy was built from.  So it ignores U/F/clover and
// uses the stored fields, exactly as mg_precond_single does in a mixed
// build.  Do not substitute mg_vcycle here -- see the definition.
int mg_precond(REAL * out, REAL * in, REAL * U_kappa, REAL * F_csw, int clover,
               int max_iter, REAL eps, int eo_precondition,
               solver_info * prec);

#endif // MIXED -- end of the precision-dependent solve-path entries

// eo outer solve (OUTER_EO): store the odd-inverted prepared clover
// (driver-built by prepare_F with invert_odd = 1) for the Schur Arnoldi
// matvec, and reserve D_hat's odd half-field scratch.  Call once after
// mg_hierarchy_build when the outer solve runs eo.  In a one-precision build
// the outer fgmres applies D_hat on the driver's own fields, so nothing here
// is consulted and this is a no-op kept for call-site uniformity.
void mg_setup_outer_eo_mixed(mg_driver_real * F_hat_d);

// Release everything mg_hierarchy_build allocated.
void mg_hierarchy_free(void);

// Elements in the full (both-parity) prepared clover field: V sites x 2 chiral
// blocks x a packed 6x6 hermitian block.  Precision-independent, so it is
// shared by the single-precision TUs that size clover buffers (mixed_bridge.c).
static inline long clover_size_full(void) {
  return (long) Lt * Lz * Ly * Lx * 2 * 6 * 6;
}


// --- mixed-precision profiling ----------------------------------------------
// The V-cycle runs in the single-precision translation unit, so its timers
// (the _f instantiations of vcycle/transfer/apply_dc/apply_schur/SAP stats)
// are only reachable from inside this bridge.  These two entry points let the
// double driver reset them before a solve and print a unified
// DDalphaAMG-style summary afterwards, combining the double outer fgmres
// breakdown with the single inner V-cycle breakdown.  fgmres_profile_t is
// layout identical across the seam (all double/long/int), so it is passed by
// pointer.
void mg_profile_reset(void);
void mg_profile_report(const fgmres_profile_t * fg, int clover);
#endif // MIXED_H
