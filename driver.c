// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

#include <stdlib.h>
#include "global_params.h"
#include "utils.h"
#include "clover_ferm.h"
#include "solver.h"
#include "mixed.h"
#include "driver.h"

// Solver driver.  See driver.h for the interface and for what is deliberately
// compile-time rather than a parameter.
//
// Compiled once, not twinned: this is driver-layer code, so it
// builds with MG_SINGLE unset and its calls resolve to the _d twins in a mixed
// build (and to _f under PRECISION=single, where there is one precision).  Its
// own externals must therefore stay out of mangle.h.

// State the solve needs between drv_setup and drv_solve.  The hierarchy keeps
// its own down-converted copies of these fields (mixed_bridge.c), but the
// outer solver applies the double operator on the arrays here, so they must
// outlive every solve.
struct drv_handle {
  // prepared gauge; gauge_alloc_size, backward-link ghosts included
  REAL * U_kappa;
  // prepared clover, odd half raw -- what the hierarchy was built on
  REAL * F_csw;
#if OUTER_EO
  // prepared clover, odd half inverted -- the outer Schur operator
  REAL * F_hat;
#endif
  long   clover_size;
  REAL   kappa;
  int    clover;
};

drv_handle * drv_setup(REAL * U, const drv_setup_params * p) {
  drv_handle * h = malloc(sizeof(* h));
  h->clover_size = clover_size_full();
  h->kappa       = p->kappa;
  h->clover      = (p->csw != 0);

  // F_munu is consumed by prepare_F alone, so it is a setup temporary; it is
  // held until after the hierarchy build only to keep the allocation order
  // the callers had, since the SAP D_oo^-1 cache is keyed on a field's
  // address.
  REAL * F = calloc(h->clover_size, sizeof(REAL));
  if (h->clover) compute_Fmunu(U, F);

  h->U_kappa = malloc(sizeof(REAL) * gauge_alloc_size);
  h->F_csw   = malloc(sizeof(REAL) * h->clover_size);
  prepare_gauge(h->U_kappa, U, p->kappa);
  prepare_F(h->F_csw, F, p->kappa, p->csw, /*invert_odd*/ 0);
#if OUTER_EO
  // Second prepared clover with the odd half inverted, for the outer Schur
  // operator D_hat.  The hierarchy and any true-residual check keep the raw-odd
  // F_csw above, so both are built here while the raw F is still alive.
  h->F_hat = malloc(sizeof(REAL) * h->clover_size);
  prepare_F(h->F_hat, F, p->kappa, p->csw, /*invert_odd*/ 1);
#endif

  mg_hierarchy_build(h->U_kappa, h->F_csw, h->clover, &p->mg);
  free(F);
#if OUTER_EO
  // Hand the bridge the odd-inverted clover for the single Schur Arnoldi
  // matvec.  Idempotent, and a no-op in a one-precision build, where fgmres
  // applies D_hat on the fields above directly.
  mg_setup_outer_eo_mixed(h->F_hat);
#endif
  return h;
}

int drv_solve(drv_handle * h, REAL * x, REAL * b, const drv_solve_params * p) {
  // The preconditioner apply.  In a mixed build fgmres hard-routes to the
  // bridge's float entry point and never reads .solver -- only the non-NULL
  // solver_info itself matters -- so leaving it NULL is the honest statement.
  // In a one-precision build fgmres dispatches .solver, and mg_precond is the
  // entry that runs the cycle on the hierarchy's own stored operator (not
  // mg_vcycle directly -- fgmres would hand it the outer operator's fields).
#ifdef MIXED
  solver_info mgp = { .solver = NULL,
                      .preconditioner = NULL, .max_iter = 0, .eps = 0 };
#else
  solver_info mgp = { .solver = mg_precond,
                      .preconditioner = NULL, .max_iter = 0, .eps = 0 };
#endif
  mg_profile_reset();     // clear the V-cycle / SAP / transfer timers

  int it;
#if OUTER_EO
  // Even-site Schur solve: D_hat x_e = b_e + kappa H_eo D_oo^-1 b_o, with the
  // MG preconditioner bridged by zero-pad / even-slice inside the cycle.
  // x's odd half is the D_oo^-1 b_o scratch until the back-substitution
  // overwrites it.
  const long n  = (long) ferm_size * num_rhs;
  const long mh = n / 2;        // even-half multi-rhs length (even sites lead)
  REAL * b_eo = malloc(sizeof(REAL) * mh);
  copy_multi_half_ferm(b_eo, b, num_rhs);
  for (long i = mh; i < n; i ++) x[i] = 0;
  if (h->clover) {
    D_oo_inv(x + mh, b + mh, h->F_hat + h->clover_size / 2);
    H_eo(b_eo, x + mh, h->U_kappa);
  } else {
    H_eo(b_eo, b + mh, h->U_kappa);
  }
  it = fgmres(x, b_eo, h->U_kappa, h->F_hat, h->clover,
              p->max_iter, p->restart, (REAL) p->tol,/*eo*/ 1, &mgp);
  // back-substitute the odd sites: x_o = D_oo^-1 (b_o + kappa H_oe x_e)
  #pragma omp parallel for
  for (long i = mh; i < n; i ++) x[i] = b[i];
  H_oe(x + mh, x, h->U_kappa);
  if (h->clover)
    D_oo_inv(x + mh, x + mh, h->F_hat + h->clover_size / 2);
  free(b_eo);
#else
  it = fgmres(x, b, h->U_kappa, h->F_csw, h->clover,
              p->max_iter, p->restart, (REAL) p->tol,/*eo*/ 0, &mgp);
#endif

  // match DDalphaAMG's / chroma's solution convention, 2*kappa * D_kappa^-1 b
  if (p->chroma_norm) rescale_multi_ferm(x, 2 * h->kappa, num_rhs);
  return it;
}

void drv_apply_D(drv_handle * h, REAL * out, REAL * in) {
  // the raw-odd clover, not F_hat: this is the full operator the hierarchy was
  // built on, which is what a true-residual check has to be measured against.
  D(out, in, h->U_kappa, h->F_csw, h->clover);
}

// The counters this reads are cleared by mg_profile_reset at the top of
// drv_solve, so what it prints is the solve that just ran, never a running
// total and never the setup.  clover comes from the handle because the flop
// model charges the clover term per site.
void drv_report_profile(drv_handle * h) {
  fgmres_profile_t fgp;
  fgmres_get_profile(&fgp);
  mg_profile_report(&fgp, h->clover);
}

void drv_teardown(drv_handle * h) {
  if (h == NULL) return;
  mg_hierarchy_free();
  free(h->U_kappa);
  free(h->F_csw);
#if OUTER_EO
  free(h->F_hat);
#endif
  // The double-side SAP D_oo^-1 cache is keyed on the F_csw address, so drop
  // it now that field is gone and a recycled address could alias a stale
  // inverse.  mg_hierarchy_free reaches only the single twin.
  sap_free_doo_inv_cache();
  free(h);
}
