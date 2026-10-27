// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

#include <stdlib.h>
#include <assert.h>
#include "global_params.h"
#include "utils.h"
#include "comm.h"
#include "driver.h"
#include "qc.h"

// Implementation of the public interface.  This is the only file that sees both
// sides: qc.h, which may assume nothing about the build, and global_params.h,
// which defines everything.  Two things cross here -- the field layout (a
// caller's lexicographic array against qc's own order) and the precision (the
// caller's choice against this build's REAL).
//
// Compiled once and not twinned, like driver.c and correlator.c: it builds with
// MG_SINGLE unset, so REAL is double and its calls resolve to the _d twins in a
// mixed build, and to the single ones under PRECISION=single, where there is
// only one precision.  Its own names must therefore stay out of mangle.h.

struct qc_handle {
  drv_handle * d;
};

// ---------------------------------------------------------------------------
// Precision and geometry
// ---------------------------------------------------------------------------

qc_precision qc_native_precision(void) {
  return sizeof(REAL) == 8 ? QC_DOUBLE : QC_SINGLE;
}

void qc_global_dims(int dims[4]) {
  dims[0] = Lx; dims[1] = Ly; dims[2] = Lz_glob; dims[3] = Lt_glob;
}

void qc_local_dims(int dims[4]) {
  dims[0] = Lx; dims[1] = Ly; dims[2] = Lz; dims[3] = Lt;
}

void qc_local_origin(int org[4]) {
  for (int d = 0; d < 4; d ++) org[d] = qc_grid.origin[d];
}

int qc_num_rhs(void)    { return num_rhs; }
int qc_num_colors(void) { return Nc; }
int qc_num_spins(void)  { return Ns; }

long qc_field_elements(void) { return (long) ferm_size * num_rhs; }
long qc_lane_elements(void)  { return (long) V * Ns * Nc * 2; }
long qc_gauge_elements(void) { return (long) gauge_size; }

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

void * qc_alloc(long elements) {
  return malloc(sizeof(REAL) * elements);   // the macro form, 64-byte aligned
}

void qc_free(void * p) { free(p); }

void qc_field_zero(void * field) {
  REAL * f = (REAL *) field;
  const long n = qc_field_elements();
  // A parallel loop rather than memset so a fresh field is first-touched by
  // the threads that will own it.
  #pragma omp parallel for
  for (long i = 0; i < n; i ++) f[i] = 0;
}

// ---------------------------------------------------------------------------
// The published field order
// ---------------------------------------------------------------------------

long qc_site_index(int x, int y, int z, int t) {
  const long vh = Vh;      // bind first: Vh expands to a bare product
  // Parity is a property of the global site, not the local one: the
  // coordinates are local, so the rank origin has to come back in before
  // taking it.  The two agree today only because comm_init forces every rank
  // origin onto a Schwarz block boundary, hence onto an even coordinate.
  const int par = (x ^ y ^ (z + qc_grid.origin[2]) ^ (t + qc_grid.origin[3]))
                  & 1;
  return par * vh + site_half(x / 2, y, z, t);
}

void qc_site_coords(long site, int * x, int * y, int * z, int * t) {
  const long vh = Vh;
  const int par = (int) (site / vh);
  int xh;
  site_half_coords(site - (long) par * vh, &xh, y, z, t);
  // Recover x's low bit from the parity: the half-lattice index dropped it.
  *x = 2 * xh + ((par ^ *y ^ (*z + qc_grid.origin[2])
                      ^ (*t + qc_grid.origin[3])) & 1);
}

long qc_component_index(long site, int s, int c, int reim, int lane) {
  return (((site * Ns + s) * Nc + c) * 2 + reim) * (long) num_rhs + lane;
}

// ---------------------------------------------------------------------------
// Conversion to and from the lexicographic order
// ---------------------------------------------------------------------------
// Generated per caller-side type so the precision test stays out of the loop.
// The real and imaginary parts of one component sit num_rhs apart in a solver
// field, which is why only the real offset is computed.

// The two directions are separate macros rather than one with a direction
// argument: the assignment is the only line that differs, and parameterizing
// an assignment reads worse than repeating the loop.

#define QC_DEFINE_IMPORT(NAME, TYPE)                                          \
  static void NAME(REAL * f, const TYPE * u, int lane) {                      \
    _Pragma("omp parallel for")                                               \
    for (int t = 0; t < Lt; t ++)                                             \
      for (int z = 0; z < Lz; z ++)                                           \
        for (int y = 0; y < Ly; y ++)                                         \
          for (int x = 0; x < Lx; x ++) {                                     \
            const long site = qc_site_index(x, y, z, t);                      \
            const long ub = ((((long) t * Lz + z) * Ly + y) * Lx + x)         \
                            * (long) (Ns * Nc * 2);                           \
            for (int s = 0; s < Ns; s ++)                                     \
              for (int c = 0; c < Nc; c ++) {                                 \
                const long uo = ub + ((long) s * Nc + c) * 2;                 \
                const long fo = qc_component_index(site, s, c, 0, lane);      \
                f[fo]           = (REAL) u[uo];                               \
                f[fo + num_rhs] = (REAL) u[uo + 1];                           \
              }                                                               \
          }                                                                   \
  }

#define QC_DEFINE_EXPORT(NAME, TYPE)                                          \
  static void NAME(const REAL * f, TYPE * u, int lane) {                      \
    _Pragma("omp parallel for")                                               \
    for (int t = 0; t < Lt; t ++)                                             \
      for (int z = 0; z < Lz; z ++)                                           \
        for (int y = 0; y < Ly; y ++)                                         \
          for (int x = 0; x < Lx; x ++) {                                     \
            const long site = qc_site_index(x, y, z, t);                      \
            const long ub = ((((long) t * Lz + z) * Ly + y) * Lx + x)         \
                            * (long) (Ns * Nc * 2);                           \
            for (int s = 0; s < Ns; s ++)                                     \
              for (int c = 0; c < Nc; c ++) {                                 \
                const long uo = ub + ((long) s * Nc + c) * 2;                 \
                const long fo = qc_component_index(site, s, c, 0, lane);      \
                u[uo]     = (TYPE) f[fo];                                     \
                u[uo + 1] = (TYPE) f[fo + num_rhs];                           \
              }                                                               \
          }                                                                   \
  }

QC_DEFINE_IMPORT(import_double, double)
QC_DEFINE_IMPORT(import_float,  float)
QC_DEFINE_EXPORT(export_double, double)
QC_DEFINE_EXPORT(export_float,  float)

void qc_import(void * field, const void * user, qc_precision user_prec,
               int lane) {
  assert(lane >= 0 && lane < num_rhs);
  if (user_prec == QC_DOUBLE)
    import_double((REAL *) field, (const double *) user, lane);
  else
    import_float ((REAL *) field, (const float  *) user, lane);
}

void qc_export(const void * field, void * user, qc_precision user_prec,
               int lane) {
  assert(lane >= 0 && lane < num_rhs);
  if (user_prec == QC_DOUBLE)
    export_double((const REAL *) field, (double *) user, lane);
  else
    export_float ((const REAL *) field, (float  *) user, lane);
}

// ---------------------------------------------------------------------------
// Startup and configuration input
// ---------------------------------------------------------------------------

void qc_init(int * argc, char *** argv) { comm_init(argc, argv); }
void qc_finalize(void)                  { comm_finalize(); }
int  qc_rank(void)                      { return comm_rank(); }
int  qc_num_ranks(void)                 { return comm_size(); }

void qc_read_gauge_file(void * U, const char * path) {
  read_gauge((REAL *) U, (char *) path);
}

// ---------------------------------------------------------------------------
// Setup and solve
// ---------------------------------------------------------------------------

// The public enumerators are the internal ones; qc.h cannot include mg_params.h
// (it may assume nothing about the build), so the two are declared separately
// and pinned to each other here.
_Static_assert((int) QC_CYCLE_V == (int) MG_CYCLE_V
            && (int) QC_CYCLE_W == (int) MG_CYCLE_W
            && (int) QC_CYCLE_K == (int) MG_CYCLE_K,
               "qc_cycle and mg_cycle_t must agree");

void qc_default_setup_params(qc_setup_params * p) {
  mg_params m;
  mg_params_default(&m);
  p->kappa                  = 0.0;    // no default; qc_setup rejects it
  p->csw                    = 0.0;    // plain Wilson unless the caller says
                                      //   otherwise
  p->smoother_sweeps        = m.smoother_sweeps;
  p->sap_block_iter         = m.sap_block_iter;
  p->coarse_smoother_sweeps = m.coarse_smoother_sweeps;
  p->coarse_sap_block_iter  = m.coarse_sap_block_iter;
  p->cycle                  = (qc_cycle) m.cycle;
  p->w_mu                   = m.w_mu;
  p->k_iters                = m.k_iters;
  p->setup_k_tol            = m.setup_k_tol;
  p->solve_k_tol            = m.solve_k_tol;
  p->coarse_max_iter        = m.coarse_max_iter;
  p->coarse_tol             = m.coarse_tol;
  p->coarse_restart         = m.coarse_restart;
  p->setup_init_sweeps      = m.setup_init_sweeps;
  p->setup_sweeps           = m.setup_sweeps;
  p->coarse_setup_sweeps    = m.coarse_setup_sweeps;
}

// The inward copy, split out of qc_setup and left non-static so ./qc params
// can drive it: qc_setup itself builds a hierarchy, so no unit test can call
// it, and these fifteen lines would otherwise be checked only by a solve
// coming out wrong.
//
// Defaults first, so that a field added to mg_params and forgotten here
// arrives as the tuned value rather than as whatever was on the caller's stack
void qc_setup_params_to_mg(const qc_setup_params * p, mg_params * m) {
  mg_params_default(m);
  m->smoother_sweeps        = p->smoother_sweeps;
  m->sap_block_iter         = p->sap_block_iter;
  m->coarse_smoother_sweeps = p->coarse_smoother_sweeps;
  m->coarse_sap_block_iter  = p->coarse_sap_block_iter;
  m->cycle                  = (mg_cycle_t) p->cycle;
  m->w_mu                   = p->w_mu;
  m->k_iters                = p->k_iters;
  m->setup_k_tol            = p->setup_k_tol;
  m->solve_k_tol            = p->solve_k_tol;
  m->coarse_max_iter        = p->coarse_max_iter;
  m->coarse_tol             = p->coarse_tol;
  m->coarse_restart         = p->coarse_restart;
  m->setup_init_sweeps      = p->setup_init_sweeps;
  m->setup_sweeps           = p->setup_sweeps;
  m->coarse_setup_sweeps    = p->coarse_setup_sweeps;
}

void qc_default_solve_params(qc_solve_params * p) {
  drv_solve_params v;
  drv_solve_params_default(&v);
  p->tol         = v.tol;
  p->max_iter    = v.max_iter;
  p->restart     = v.restart;
  p->chroma_norm = v.chroma_norm;
}

// Why these return a message instead of aborting on the spot: a checker that
// aborts cannot be tested.  ./qc params drives every branch below and confirms
// each one fires, which is the only way to know the guard rejects what it
// claims to.  qc_setup / qc_solve turn a non-NULL return into the abort.
const char * qc_setup_params_problem(const qc_setup_params * p) {
  if (p->kappa == 0)                return "kappa is 0 (unset?)";
  if (p->smoother_sweeps < 0)       return "smoother_sweeps is negative";
  if (p->sap_block_iter < 1)        return "sap_block_iter is below 1";
  if (p->coarse_smoother_sweeps < 0) return "coarse_smoother_sweeps is negative";
  if (p->coarse_sap_block_iter < 1) return "coarse_sap_block_iter is below 1";
  if (p->cycle != QC_CYCLE_V && p->cycle != QC_CYCLE_W
      && p->cycle != QC_CYCLE_K)    return "cycle is not one of QC_CYCLE_V/W/K";
  if (p->cycle == QC_CYCLE_W && p->w_mu < 1) return "w_mu is below 1";
  if (p->cycle == QC_CYCLE_K && p->k_iters < 1) return "k_iters is below 1";
  if (!(p->setup_k_tol > 0))        return "setup_k_tol is not positive";
  if (!(p->solve_k_tol > 0))        return "solve_k_tol is not positive";
  if (p->coarse_max_iter < 1)       return "coarse_max_iter is below 1";
  if (!(p->coarse_tol > 0))         return "coarse_tol is not positive";
  if (p->coarse_restart < 1)        return "coarse_restart is below 1";
  if (p->setup_init_sweeps < 0)     return "setup_init_sweeps is negative";
  if (p->setup_sweeps < 0)          return "setup_sweeps is negative";
  // Below 1 rather than below 0: bootstrap_recurse divides by this to size each
  // level's descent, so zero is a division by zero at four levels and deeper.
  if (p->coarse_setup_sweeps < 1)   return "coarse_setup_sweeps is below 1";
  return NULL;
}

const char * qc_solve_params_problem(const qc_solve_params * p) {
  if (!(p->tol > 0))    return "tol is not positive";
  if (p->max_iter < 1)  return "max_iter is below 1";
  if (p->restart < 1)   return "restart is below 1";
  return NULL;
}

// Zero sweeps are allowed above and a zero count is not, deliberately: a level
// with no smoothing is a real (if poor) method, whereas a block solve that runs
// no iterations is a smoother that returns its input -- the failure that made
// test_mixed stop converging while every count still looked plausible.
static void reject(const char * what, const char * why) {
  comm_die("qc_%s: %s. Fill the struct with qc_default_%s_params first, then "
           "override -- see qc.h.", what, why, what);
}

qc_handle * qc_setup(const void * U, qc_precision U_prec, const qc_setup_params * p) {
  const char * bad = qc_setup_params_problem(p);
  if (bad) reject("setup", bad);

  drv_setup_params sp;
  sp.kappa = (REAL) p->kappa;
  sp.csw   = (REAL) p->csw;
  qc_setup_params_to_mg(p, &sp.mg);

  // drv_setup only reads U (compute_Fmunu and prepare_gauge both take it as an
  // input), so the const cast is safe on the pass-through path.
  REAL * Ur = (REAL *) U;
  REAL * converted = NULL;
  if (U_prec != qc_native_precision()) {
    // Costs a transient copy of the whole configuration; the native path does
    // not, which is why qc_read_gauge_file writes native precision.
    const long n = qc_gauge_elements();
    converted = malloc(sizeof(REAL) * n);
    if (U_prec == QC_DOUBLE) {
      const double * s = (const double *) U;
      #pragma omp parallel for
      for (long i = 0; i < n; i ++) converted[i] = (REAL) s[i];
    } else {
      const float * s = (const float *) U;
      #pragma omp parallel for
      for (long i = 0; i < n; i ++) converted[i] = (REAL) s[i];
    }
    Ur = converted;
  }

  qc_handle * h = malloc(sizeof(* h));
  h->d = drv_setup(Ur, &sp);
  free(converted);
  return h;
}

int qc_solve(qc_handle * h, void * x, const void * b,
             const qc_solve_params * p) {
  const char * bad = qc_solve_params_problem(p);
  if (bad) reject("solve", bad);

  drv_solve_params vp;
  vp.tol         = p->tol;
  vp.max_iter    = p->max_iter;
  vp.restart     = p->restart;
  vp.chroma_norm = p->chroma_norm;
  // b is read-only through the solve (it is copied into the Schur right-hand
  // side and read again by the odd-site reconstruction), hence the const cast.
  return drv_solve(h->d, (REAL *) x, (REAL *) b, &vp);
}

void qc_apply_D(qc_handle * h, void * out, const void * in) {
  drv_apply_D(h->d, (REAL *) out, (REAL *) in);
}

void qc_report_profile(qc_handle * h) {
  drv_report_profile(h->d);
}

void qc_teardown(qc_handle * h) {
  if (h == NULL) return;
  drv_teardown(h->d);
  free(h);
}
