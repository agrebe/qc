// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef MG_PARAMS_H
#define MG_PARAMS_H
// Deliberately includes nothing.  The tuned defaults are literals in
// mg_params_default below rather than macros read from global_params.h, so
// this header does not drag that one -- which #defines malloc, and through
// mangle.h the bare names D, H, V and Nc -- into everything that wants a
// parameter struct.

// Tuning parameters of the multigrid preconditioner, carried in one struct
// from the driver down to the individual hierarchy levels.
//
// One struct rather than a longer argument list at each layer.  The same
// values travel through drv_setup -> mg_hierarchy_build -> the adaptive
// bootstrap -> mg_setup / mg_add_coarse_level, and a positional chain that
// long is one transposed pair of ints away from a misconfiguration no
// compiler can see.  Adding a knob then touches the struct and the two ends,
// not five signatures.
//
// Nothing here is a REAL.  This struct crosses the precision seam -- the
// driver above it is double while the hierarchy below it is single in a mixed
// build -- so every field is a plain C type, and the conversion to REAL
// happens where the value is stored, once per field.
//
// Always initialize with mg_params_default and then override, never by listing
// fields.  A designated initializer zero-fills whatever it omits, and zero is
// a legal-looking, wrong value for most of these: nothing here can tell a
// caller who wanted the default from one who asked for no iterations at all.
//
// What is deliberately absent: anything the build is sized or _Static_asserted
// against.  NUM_LEVELS, the aggregation and SAP block edges, num_null_vecs,
// the per-level coarsening widths and the outer solve's eo split stay
// compile-time in global_params.h, because loop bounds, array extents and
// generated assembly are derived from them.  The setup sweep counts below are
// not in that class -- they only drive loop trip counts -- so they live here.
// The test for whether a knob belongs here is whether a second caller could
// reasonably disagree about it: a library caller tuning a solve is that second
// caller, even where every caller in this repo happens to run the defaults.

// --- Multigrid cycle type ---------------------------------------------------
// Selects how each intermediate level computes its coarse-grid correction (the
// recursion into the coarser level).  A coarse correction whose child is the
// coarsest level calls the bottom solve directly, so a 2-level hierarchy is
// unaffected by this setting.
//
//   MG_CYCLE_V : one recursive cycle (the classic V-cycle).
//   MG_CYCLE_W : w_mu stationary recursive cycles, recomputing the coarse
//                residual between them (the mu-cycle; w_mu = 2 is the W-cycle).
//   MG_CYCLE_K : a flexible-GMRES accelerator (Notay K-cycle) around the
//                recursive cycle -- k_iters Arnoldi steps on D_c
//                preconditioned by one recursive cycle, stopped early at
//                relative residual k_tol.
typedef enum { MG_CYCLE_V = 0, MG_CYCLE_W = 1, MG_CYCLE_K = 2 } mg_cycle_t;

typedef struct {
  // --- smoothers ---
  // fine-level SAP post-smoothing sweeps
  int    smoother_sweeps;
  // minimal-residual iterations inside each fine SAP block, per sweep
  int    sap_block_iter;
  // SAP sweeps on every level below the fine one.  Tuned separately from the
  // fine count: the coarse block solve is a different, stronger process, so
  // its optimum differs.
  int    coarse_smoother_sweeps;
  // MR iterations inside each coarse SAP block.  Read only where the coarse
  // block solve is iterative (COARSE_SAP_DENSE_INV=0); the dense-inverse build solves
  // each block exactly instead.
  int    coarse_sap_block_iter;

  // --- cycle shape ---
  mg_cycle_t cycle;               // V / W / K (see above)
  int    w_mu;                    // W-cycle: recursive cycles per level
  int    k_iters;                 // K-cycle: max Arnoldi steps per level
  double setup_k_tol;             // K-cycle relative residual during setup, and
  double solve_k_tol;             //   during the solve -- looser here, since
                                  //   the cycle is only a preconditioner
                                  //    by then

  // --- coarsest-level solve ---
  // Classic cap + tolerance semantics (fgmres_coarse): iterate until coarse_tol
  // or coarse_max_iter, restarting every coarse_restart Arnoldi steps.  The
  // K-cycle above stays adaptive (solve_k_tol), which bounds the damage if the
  // cap binds first.  All three are bypassed by an exact bottom solve
  // (BOTTOM_DENSE_INV, once its inverse is built) and are its fallback.
  // iteration cap
  int    coarse_max_iter;
  // relative-residual target
  double coarse_tol;
  // Restart length.  Sizes the coarse Krylov basis fgmres_coarse allocates up
  // front (restart+1 coarse fields), so it is a memory cost whether or not it
  // ever binds.
  int    coarse_restart;

  // --- adaptive setup schedule (DDalphaAMG's setup_iter) ---
  // SAP sweeps smoothing the initial random vectors
  int    setup_init_sweeps;       
  // inverse-iteration sweeps at the fine level
  int    setup_sweeps;            
  // What the descent ramps toward below the fine level, not the sweeps that
  // level receives: its ratio to setup_sweeps sets the shape, so at (4, 3)
  // the fine level takes 4 sweeps and each level below takes 1+2+2+3 = 8.
  int    coarse_setup_sweeps;
} mg_params;

// The tuned method: these literals are the definition, not a copy of one.
// This is what the repo is regression-tested at and what a library caller
// inherits for every field it does not override, so retuning the method means
// editing here.  Everything below the driver reads the struct, so an override
// reaches all the way down and a field left alone behaves exactly as the qc
// binary does.
static inline void mg_params_default(mg_params * p) {
  p->smoother_sweeps        = 2;
  p->sap_block_iter         = 4;
  p->coarse_smoother_sweeps = 2;
  p->coarse_sap_block_iter  = 4;
  p->cycle                  = MG_CYCLE_K;
  p->w_mu                   = 2;
  p->k_iters                = 8;
  p->setup_k_tol            = 5e-2;
  p->solve_k_tol            = 0.15;
  p->coarse_max_iter        = 100;
  p->coarse_tol             = 5e-2;
  p->coarse_restart         = 12;
  p->setup_init_sweeps      = 3;
  p->setup_sweeps           = 4;
  p->coarse_setup_sweeps    = 3;
}

#endif
