// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef DRIVER_H
#define DRIVER_H
#include "global_params.h"
#include "mg_params.h"

// The production Wilson-clover multigrid solve, as a reusable setup/solve
// pair.
//
// The whole sequence in one place: prepare the gauge and clover fields, build
// the MG hierarchy, then run the outer FGMRES -- even-odd (Schur)
// preconditioned when OUTER_EO is set, which means forming the even-site
// right-hand side and back-substituting the odd sites around the solve.
// Splitting setup from solve is the point: the hierarchy build is the
// expensive part and is reused across every solve on one gauge configuration.
//
// Internal, and free to change: qc.h is the public interface, and qc.c wraps
// everything here.  The two differ in what they may assume.  This layer speaks
// REAL and includes global_params.h, so it is usable only from inside the
// build; qc.h speaks void * plus an explicit precision tag and includes
// nothing.  The drv_ prefix keeps the qc_ namespace for the names that are
// published.
//
// Fields are handed over already packed in the internal layout (parity-major
// halves, rhs the fastest index).  qc.c is where a caller's own layout is
// converted, so nothing below has to know about more than one.

// Opaque; the hierarchy and the prepared operator fields live behind it.
typedef struct drv_handle drv_handle;

// Setup inputs, split by what they shape: kappa and csw define the operator,
// mg tunes the preconditioner built from it (see mg_params.h for the fields).
// The clover flag is not among them: it is derived as (csw != 0), as solve()
// already does, so the two cannot disagree.  Everything else that shapes the
// hierarchy -- NUM_LEVELS and the per-level schedules it is static-asserted
// against, OUTER_EO, the block sizes -- is compile-time and is read from
// global_params.h inside drv_setup rather than passed here.
typedef struct {
  REAL      kappa, csw;
  mg_params mg;
} drv_setup_params;

typedef struct {
  double tol;            // relative-residual target.  double, not REAL: it is
                         // a threshold rather than field data, and carrying it
                         // at REAL would round-trip a single-precision build's
                         // value through float and hand the caller back
                         // something other than what it asked for.
  int  max_iter;         // outer FGMRES cap
  int  restart;          // outer FGMRES restart length: Krylov vectors per
                         // cycle, and the solve's memory high-water mark
                         // (a V- and a Z-basis vector each per unit)
  int  chroma_norm;      // 1 = rescale the solution by 2*kappa
} drv_solve_params;

// The tuned values from global_params.h, for the same reason mg_params_default
// exists: a designated initializer zero-fills what it omits, and a zeroed tol
// or restart is a legal-looking value that silently changes the solve.
// Override after calling this, never instead of it.
static inline void drv_solve_params_default(drv_solve_params * p) {
  p->tol         = OUTER_TOL;
  p->max_iter    = OUTER_MAX;
  p->restart     = FGMRES_RESTART;
  p->chroma_norm = 0;        // caller policy, not a tuned value
}

// Prepare the operator fields from the raw gauge configuration U and build the
// hierarchy.  U is caller-owned and is only read: it may be freed as soon as
// this returns.  Returns a handle to be passed to drv_solve.
drv_handle * drv_setup(REAL * U, const drv_setup_params * p);

// Solve D x = b for all num_rhs right-hand sides.  Returns the outer FGMRES
// iteration count.
//
// Both x and b are full multi-rhs fields of ferm_size * num_rhs REALs.  x must
// be full-length even under OUTER_EO, where the solve itself runs on the even
// half: FGMRES spills the discarded odd half of its last preconditioned basis
// vector there, and the odd half doubles as the D_oo^-1 b_o scratch until the
// back-substitution overwrites it.  x need not be initialized.
int drv_solve(drv_handle * h, REAL * x, REAL * b, const drv_solve_params * p);

// Apply the full fine operator: out = D in, on the handle's prepared fields.
// Exposed so a caller can form its own true residual b - D x without the
// prepared gauge and clover leaking out of the handle.  Both fields are full
// multi-rhs (ferm_size * num_rhs).  Note this is the full operator even under
// OUTER_EO, where the solve itself runs on the even-site Schur system.
void drv_apply_D(drv_handle * h, REAL * out, REAL * in);

// Print where the last drv_solve's wall time went, broken down by kernel, with
// each row's flop rate.  Valid only after a solve, and it describes that one
// solve alone -- drv_solve clears the counters on entry.  Rank 0 only.
void drv_report_profile(drv_handle * h);

// Release the hierarchy and everything drv_setup allocated.
void drv_teardown(drv_handle * h);

#endif
