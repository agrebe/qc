// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef BOOTSTRAP_H
#define BOOTSTRAP_H
#include "global_params.h"   // REAL
#include "coarse.h"          // coarse_op
#include "mg_params.h"

// DDalphaAMG adaptive bootstrap setup for the fine near-null (test) vectors --
// thesis (Rottmann) Algorithm 16, the inexact-deflation setup with the
// initial-guess modification.  Generates the test vectors that define the
// prolongator P by inverse iteration on the current multigrid method.
//
// On return:
//   * nv  holds num_null_vecs locally-orthonormalized columns (nv-layout),
//   * Dc  holds the freshly-built D_c = P^H D P for those columns,
//   * the two-level MG hierarchy is registered (mg_setup) ready for mg_vcycle.
//
// Single-precision only (the MG hierarchy is single): call from the MG_SINGLE
// side (mixed_bridge.c).  Two-level; bootstrap_null_vecs_nlvl below is the
// multilevel recursion (Algorithm 20/21).
//
// p supplies the schedule -- setup_sweeps inverse-iteration refinement cycles
// after setup_init_sweeps SAP sweeps on the initial random vectors -- and is
// forwarded verbatim to mg_setup on every rebuild.
void bootstrap_null_vecs(REAL * nv, coarse_op * Dc,
                         REAL * U_kappa, REAL * F_csw, int clover,
                         const mg_params * p);

// Multilevel bootstrap: builds the fine (nv0 / Dc0) and every coarse (nvc[d])
// set of test vectors and registers the full NUM_LEVELS-deep hierarchy.  Each
// level's vectors are seeded by restricting its parent's, then all levels are
// bootstrapped in the interleaved order of DDalphaAMG's recursive
// inv_iter_inv_fcycle.  nv0 and nvc[1..NUM_LEVELS-2] are caller-owned and kept
// alive (they are the registered P columns on each level); the sweep schedule
// comes from p (setup_sweeps / coarse_setup_sweeps) and the aggregation blocks
// from BOOTSTRAP_BLK_LVL.  nvc is an array of NUM_LEVELS pointers; entry d
// (d >= 1) holds level d's coarse P columns (entry 0 is unused -- the fine
// columns are nv0).
void bootstrap_null_vecs_nlvl(REAL * nv0, coarse_op * Dc0, REAL ** nvc,
                              REAL * U_kappa, REAL * F_csw, int clover,
                              const mg_params * p);

// Unit test for the null-vector batching (the gather/scatter that feeds the
// ncol test vectors through the num_rhs-wide smoother).  Runs on synthetic
// shapes and touches no hierarchy, so it needs neither a gauge field nor a
// setup.  Returns nonzero on failure.  Takes no REAL argument on purpose:
// this file's REAL is the single-precision one, so a double caller can
// declare it without the seam.
int bootstrap_selfcheck_batching(void);
#endif
