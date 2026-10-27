// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef QC_H
#define QC_H

// qc -- Wilson-clover multigrid solver.  Public interface.
//
// This header deliberately includes nothing from the qc build.  The internal
// headers define a macro named D (the Dirac operator) and replace malloc, so a
// caller that pulled them in would find its own code rewritten underneath it.
// Nothing here is a macro, and every name is prefixed qc_ or QC_.
//
// A caller does: qc_init, read or build a gauge configuration, qc_setup once
// per configuration, then any number of qc_solve calls, then qc_teardown.  The
// hierarchy built by qc_setup is the expensive part and is what gets reused.
//
// --- What is fixed at compile time -----------------------------------------
// The lattice extents, the process grid, the number of right-hand sides solved
// per call and the arithmetic precision are all compile-time constants of the
// library, not arguments.  A given build of qc is valid for exactly one such
// combination: the geometry is baked into loop bounds and into generated
// assembly.  Query it with the functions below and check it against what your
// program expects -- there is no runtime negotiation, and no error will be
// raised if you assume something else.
//
// --- Conventions you cannot infer from an index formula ---------------------
// Gamma basis is DeGrand-Rossi, gamma5 = diag(1, 1, -1, -1).
// Boundary conditions are periodic in space and anti-periodic in time; the
// time-direction twist is applied internally, so hand over an untwisted
// configuration.
// The operator is D = 1 - kappa*H (clover included when csw is nonzero).  Set
// chroma_norm to rescale the solution by 2*kappa, which is the convention
// chroma and DDalphaAMG report.
//
// --- Limitations of this release --------------------------------------------
// One handle may be live at a time: the hierarchy lives in file-scope state.
// Calls must be made from serial context -- qc parallelizes internally with
// OpenMP and does not nest.
// A fatal configuration error inside qc calls MPI_Abort on MPI_COMM_WORLD, so
// it takes the whole job down rather than just qc's ranks.  Running out of
// memory is one of these: qc checks every allocation and aborts naming the
// request and the machine's memory state, rather than returning an error for
// the caller to recover from.  Note that on a machine that overcommits (the
// default on Linux) a request beyond physical memory succeeds here and the
// kernel's OOM killer ends the process later, which no library-side check can
// intercept.
//
// MPI is adopted, not owned: qc_init initializes MPI only if nothing else has,
// and qc_finalize finalizes only what qc_init started, so a caller that runs
// its own MPI_Init keeps its own MPI_Finalize.  Whoever initializes must ask
// for at least MPI_THREAD_FUNNELED.  To run on a communicator other than
// MPI_COMM_WORLD, call qc_set_comm before qc_init -- that one function lives
// in qc_mpi.h, so that this header stays free of <mpi.h> and a serial consumer
// needs no MPI at all.
//
// --- Floating-point environment ---------------------------------------------
// qc inherits the process floating-point environment and never modifies it.
//
// This is worth knowing because the standalone qc binary does not run in the
// same one.  It links with -ffast-math, which adds a startup constructor that
// sets FTZ/DAZ -- denormals flushed to zero.  That is a property of the link
// line, not of the library: it cannot attach to a shared object at all, and it
// does not come along inside libqc.a either.  So an application that also links
// with -ffast-math runs qc exactly as the binary does, and one that does not
// gets IEEE gradual underflow instead.
//
// Gradual underflow is the more accurate of the two, and the only cost is
// speed -- a denormal result takes a slow path in the hardware.  Whether that
// ever happens here is a question about your quark mass and volume, not about
// the library: fp32 denormals begin below 1e-38, and an ordinary solve stays
// many decades above that.
//
// Note it is the link line that decides.  -ffast-math in CFLAGS alone does not
// do it.  To get the behavior without letting -ffast-math near your own
// arithmetic, set the two bits yourself at the top of main:
//
//   #include <xmmintrin.h>
//   #include <pmmintrin.h>
//   _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
//   _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

#include <stddef.h>

// The library is C, and most of what links against it is C++.  This is the one
// header in the tree with the guard, for that reason: the internal headers
// deliberately do not have it, because they are not consumable from C++ at all.
#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Precision
// ---------------------------------------------------------------------------

// Bytes per real, so the enumerator doubles as an element size.
typedef enum { QC_SINGLE = 4, QC_DOUBLE = 8 } qc_precision;

// The precision of solver fields, of qc_alloc's memory, and of the gauge array
// qc_read_gauge_file fills.  Fixed by how the library was built.
//
// Solver fields are passed as void * so that one header serves either build.
// That is the same trade QUDA makes, and it has the same cost: handing a solve
// an array of the wrong precision compiles cleanly and reads garbage.  Nothing
// in the library can detect it.  Assert this against your own type, once, at
// startup.
qc_precision qc_native_precision(void);

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------
// All of these may be called before qc_setup.  Local extents and the origin
// describe this rank's slab of the lattice; they equal the global extents and
// zero on an undecomposed build.  Only z and t are ever decomposed.

void qc_global_dims (int dims[4]);  // {x, y, z, t} extents of the whole lattice
void qc_local_dims  (int dims[4]);  // extents of this rank's slab
void qc_local_origin(int org[4]);   // global coordinate of this rank's origin

int qc_num_rhs(void);               // right-hand sides solved per qc_solve call
int qc_num_colors(void);
int qc_num_spins(void);

long qc_field_elements(void);       // reals in a solver field (all lanes)
long qc_lane_elements(void);        // reals in one lane in the layout
                                    //   qc_import reads
long qc_gauge_elements(void);       // reals in a gauge configuration

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

// Allocate `elements` reals of the native precision, aligned for the vector
// kernels.  Solver fields must satisfy that alignment; use this, or match it.
void * qc_alloc(long elements);
void   qc_free(void * p);

// Zero a whole solver field, every lane.  Worth stating because qc_import fills
// one lane: a lane you neither import nor write by hand holds whatever the
// allocator returned.
void qc_field_zero(void * field);

// ---------------------------------------------------------------------------
// Solver field layout, in qc's own order
// ---------------------------------------------------------------------------
// A solver field holds all qc_num_rhs() right-hand sides interleaved.  Its
// storage order is published so that a caller can build a source directly in
// it and read a solution directly out of it, with no intermediate array -- for
// a point source that is easier than the lexicographic route below, not harder:
// twelve stores instead of a full-volume buffer.
//
//   long site = qc_site_index(x, y, z, t);  // local coordinates
//   long i    = qc_component_index(site, s, c, reim, lane);
//   ((double *) field)[i] = ...;            // if native precision is QC_DOUBLE
//
// The order these two express is even sites before odd; within a parity, sites
// grouped into 4^4 blocks (which is what keeps a smoother sweep's working set
// contiguous); then spin, then color, then real/imaginary, with the right-hand
// side lane fastest.  A field is exactly qc_field_elements() reals with no
// padding, so every element is addressable this way.
//
// Treat these two functions as the contract, not the description above.  The
// blocking exists for cache reasons and may be retuned; code written against
// qc_site_index survives that, code that reimplements the block arithmetic
// does not.

// Index of local site (x, y, z, t) within a solver field, parity included.
long qc_site_index(int x, int y, int z, int t);

// Inverse of qc_site_index.
void qc_site_coords(long site, int * x, int * y, int * z, int * t);

// Element offset of one component: spin s, color c, reim 0 (real) or 1
// (imaginary), of right-hand side `lane`, at the site qc_site_index returned.
long qc_component_index(long site, int s, int c, int reim, int lane);

// ---------------------------------------------------------------------------
// Solver field layout, converted
// ---------------------------------------------------------------------------
// For callers whose fields already exist in the conventional order.  One array
// per lane, holding this rank's slab, of qc_lane_elements() reals:
//
//   off = ((((t*Lz + z)*Ly + y)*Lx + x)*Ns + s)*Nc + c
//   real = a[2*off], imaginary = a[2*off + 1]
//
// t slowest and x fastest, spin outside color, complex interleaved -- the ILDG
// and chroma convention.  Extents are the local ones from qc_local_dims.
//
// user_prec describes the caller's array only; the field is always native
// precision.  The two are converted if they differ.

void qc_import(void * field, const void * user, qc_precision user_prec,
               int lane);
void qc_export(const void * field, void * user, qc_precision user_prec,
               int lane);

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

// Call once before anything else, and once at the end.  See the note on MPI
// ownership at the top of this header.
void qc_init(int * argc, char *** argv);
void qc_finalize(void);

int qc_rank(void);
int qc_num_ranks(void);

// ---------------------------------------------------------------------------
// Gauge configuration
// ---------------------------------------------------------------------------
// A gauge array holds this rank's slab in lexicographic order, four links per
// site, each link a row-major Nc x Nc complex matrix:
//
//   off = ((((t*Lz + z)*Ly + y)*Lx + x)*4 + mu)*Nc*Nc + row*Nc + col
//   real = U[2*off], imaginary = U[2*off + 1]
//
// mu ordering is x, y, z, t.  This is the ILDG record order unchanged, so a
// configuration read from file needs no permutation.

// Read a single-file ILDG or SciDAC configuration (or a payload extracted from
// one) into a native-precision array of qc_gauge_elements() reals.  Handles the
// byte order, verifies the file's own checksum and geometry, and reads only
// this rank's slab.  Aborts with a message on any mismatch.
void qc_read_gauge_file(void * U, const char * path);

// ---------------------------------------------------------------------------
// Setup and solve
// ---------------------------------------------------------------------------

typedef struct qc_handle qc_handle;

// How each intermediate level computes its coarse-grid correction.  A
// two-level hierarchy is unaffected: with no level between the fine grid and
// the coarsest, there is no recursion to accelerate.
typedef enum {
  QC_CYCLE_V = 0,  // one recursive cycle
  QC_CYCLE_W = 1,  // w_mu stationary cycles, coarse residual recomputed between
  QC_CYCLE_K = 2   // a flexible-GMRES accelerator around the recursive cycle
} qc_cycle;

// --- Filling these in -------------------------------------------------------
// Call qc_default_setup_params / qc_default_solve_params first, set what you
// care about, and pass the result.  Never build one with a designated
// initializer:
//
//   qc_setup_params p = { .kappa = 0.1372, .csw = 1.9 };   // wrong
//
// C zero-fills every field you leave out, and zero is a legal-looking, wrong
// value for most of these -- zero smoothing sweeps, a zero iteration cap, a
// zero tolerance.  Nothing in qc can tell that apart from a deliberate
// request.  The defaults are the values this build is tuned and
// regression-tested at, so a field you never touch behaves exactly as the qc
// binary does.

typedef struct {
  // hopping parameter; no default, user must set it
  double kappa;
  // clover coefficient; 0 selects plain Wilson
  double csw;

  // Smoothers.  Sweeps are applications of the Schwarz smoother; the inner
  // counts are minimal-residual iterations solving each Schwarz block.
  // fine level
  int    smoother_sweeps;
  // fine-level per-block count
  int    sap_block_iter;
  // every level below the fine one
  int    coarse_smoother_sweeps;
  // Per-block count below fine level.  Read only when this build solves
  // coarse blocks iteratively; the default build inverts them exactly and
  // ignores this.
  int    coarse_sap_block_iter;

  // Cycle shape
  qc_cycle cycle;
  // QC_CYCLE_W: recursive cycles per level
  int    w_mu;                 
  // QC_CYCLE_K: max Arnoldi steps per level
  int    k_iters;              
  // cycle tolerance while building the hierarchy,
  double setup_k_tol;          
  // cycle tolerance while solving -- typically looser, since by then the 
  // cycle is only a preconditioner
  double solve_k_tol;

  // Coarsest-level solve
  // iteration cap
  int    coarse_max_iter;      
  // relative-residual target
  double coarse_tol;           
  // restart length
  int    coarse_restart;       

  // Adaptive setup.  These set how long qc_setup spends improving the vectors
  // that define the hierarchy, so they trade setup time against solve time.
  // smoothing sweeps on the initial random vectors
  int    setup_init_sweeps;    
  // inverse-iteration sweeps at the fine level
  int    setup_sweeps;         
  // What the descent ramps toward below the fine level, not the sweeps that 
  // level receives: its ratio to setup_sweeps sets the shape, so at (4, 3) the
  // fine level takes 4 sweeps and each level below takes 1+2+2+3 = 8.
  int    coarse_setup_sweeps;
} qc_setup_params;

typedef struct {
  // relative-residual target
  double tol;
  // cap on outer iterations
  int    max_iter;
  // Outer Krylov vectors per cycle.  This is the solve's memory high-water
  // mark: it holds this many search directions, twice over.
  int    restart;
  // 1 rescales the solution by 2*kappa
  int    chroma_norm;        
} qc_solve_params;

// Fill p with this build's tuned defaults.  kappa is left at 0, which qc_setup
// rejects -- it is the one value with no sensible default.
void qc_default_setup_params(qc_setup_params * p);
void qc_default_solve_params(qc_solve_params * p);

// Build the operator and the multigrid hierarchy from configuration U.  U is
// only read and may be freed as soon as this returns; qc keeps its own prepared
// copy.  U_prec describes U, which need not be the native precision (it is
// converted if it differs, which costs a transient copy of the configuration).
//
// Every field of p is checked first.  A value qc cannot act on -- a zero
// tolerance, a negative sweep count, an unset kappa -- names itself and aborts
// rather than solving something you did not ask for.
qc_handle * qc_setup(const void * U, qc_precision U_prec, const qc_setup_params * p);

// Solve D x = b for all qc_num_rhs() right-hand sides at once.  Both fields
// are native precision and qc_field_elements() long.  x need not be
// initialized.  Returns the number of outer iterations taken.
int qc_solve(qc_handle * h, void * x, const void * b,
             const qc_solve_params * p);

// Apply the operator: out = D in.  Provided so a caller can form its own
// residual b - D x without the prepared fields leaking out of the handle.
void qc_apply_D(qc_handle * h, void * out, const void * in);

// Print where the last qc_solve's time went: one line per kernel with its
// share of the wall and its flop rate, then the totals.  Meant for tuning --
// it is what tells you whether a parameter change moved the smoother, the
// coarse grid or the transfers.  Call it after qc_solve; it describes that
// solve alone, since qc_solve clears the counters on entry.  Prints on rank 0
// only.
void qc_report_profile(qc_handle * h);

void qc_teardown(qc_handle * h);

#ifdef __cplusplus
}
#endif

#endif
