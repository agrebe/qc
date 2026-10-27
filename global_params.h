// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

// Build-time parameters.  The user parameters are hard-coded with no -D or
// environment override: recompilation is cheap, so editing a value and
// rebuilding is the way to change a run.  The exceptions are the switches in
// sections 2 and 3 wrapped in #ifndef: the value here is the default, and a
// -D flag (e.g. make XCFLAGS=-DBOTTOM_DENSE_INV=0) overrides it for an A/B
// build; run make clean first, since make cannot see a -D change.  The file
// is ordered by who should edit what:
//   1. user parameters        -- lattice, solver tolerance, multigrid hierarchy
//   2. machine tuning         -- cache-blocking tiles
//   3. algorithm switches     -- developer A/B toggles; the defaults are tuned
//   4. fixed / derived values -- do not edit
// The multigrid method knobs that do not size any array (smoother iterations,
// setup sweeps, cycle type, ...) are runtime settings, defaulted in
// mg_params_default (mg_params.h).  Physics inputs (kappa, csw, the gauge
// file) are supplied by the driver.  Build-structural switches (MIXED, ASM,
// AVX2/AVX512, PREFETCH) are in the Makefile because they choose which objects
// compile and the -march flags.

// ===========================================================================
// 1. User parameters
// ===========================================================================

// --- Lattice and problem size -----------------------------------------------
// Global lattice extents.  Lz_glob/Lt_glob are the global z/t extents; the
// per-rank local extents Lz/Lt used by all layout and kernel code are derived
// below from the MPI process grid.  x and y are never decomposed.
#define Lx 16
#define Ly Lx
#define Lz_glob Lx
#define Lt_glob 32

// MPI process grid: ranks per dimension.  Only z and t may exceed 1 --
// decomposing only the two slowest indices keeps every face a regular strided
// slab and keeps whole x-lines (and their Lxh even/odd packing) on one rank.
// Each must divide its global extent (checked at startup); a 1x1 grid compiles
// to zero ghost overhead.
#define MPI_GRID_PZ       1
#define MPI_GRID_PT       1

// number of right-hand sides solved simultaneously
#define num_rhs 16

// --- Outer solver (FGMRES) --------------------------------------------------
#define OUTER_TOL         1e-10        // relative-residual convergence target
#define OUTER_MAX         100          // max outer iterations
// Restart length (Krylov vectors per cycle).  FGMRES stores both a V-basis and
// a preconditioned Z-basis (each ~ferm_size*num_rhs reals), so keep this
// short: the basis is the solver's memory high-water mark.
#define FGMRES_RESTART 5

// --- Multigrid hierarchy ----------------------------------------------------
// These are tuned together.  Level 0 is the fine lattice; each transition
// aggregates a block of sites on one level into one site on the next.  The
// divisibility constraints relating the block edges to the lattice extents are
// _Static_asserted in coarse.h and bootstrap.c, so a bad combination fails at
// compile time.

// Total hierarchy depth (>= 2).  Raising it means adding one entry to each of
// BOOTSTRAP_BLK_LVL and COARSE_NULL_VECS_LVL below.
#define NUM_LEVELS        3

// Fine SAP smoother (red_black_schwarz) block edge.  Must be even and must
// divide every local extent.
#define SCHWARZ_BLOCK_SIZE 4

// Level 0 -> 1 aggregation block edge: the fine lattice is tiled into
// BLOCK_AGG^4 blocks, each becoming one coarse site.  Need not equal
// SCHWARZ_BLOCK_SIZE, but the fused first-color prolongation in coarse.c only
// fires when the two are equal.
#define BLOCK_AGG 4

// Aggregation blocks for the coarser transitions (1 -> 2, 2 -> 3, ...), one
// {x, y, z, t} entry each: NUM_LEVELS-2 entries.  Each edge must divide the
// extent of the level it aggregates.
#define BOOTSTRAP_BLK_LVL    { {2, 2, 2, 2} }

// Level 0 -> 1 near-null (test) vector count used to build the interpolation.
// These are a separate set from the num_rhs vectors being inverted, so this
// need not equal num_rhs, but it must be even and a multiple of num_rhs.  A
// near-null field occupies ferm_size * num_null_vecs reals.
#define num_null_vecs 32

// Near-null vector counts for the coarser transitions (1 -> 2, 2 -> 3, ...):
// NUM_LEVELS-2 entries, each even and a multiple of num_rhs.  More coarse
// variables on an inner level trades work against preconditioner quality.
#define COARSE_NULL_VECS_LVL { 32 }
// The largest entry of COARSE_NULL_VECS_LVL: it sizes the coarse path's stack
// arrays and field paddings.  Keep it equal to that maximum (a value below it
// aborts at setup; a value above it only wastes memory).
#define COARSE_NULL_VECS 32

// ===========================================================================
// 2. Machine tuning
// ===========================================================================

// eo half-sweep tiling step (clover_ferm.h OUTER_LOOPS_HALF); increase on
// machines with larger caches.  Under MPI every local extent must be a
// multiple of it (checked in comm.c).
#define block_size_half 8

// Wilson-kernel cache-blocking tiles (the DRAM-BW-bound hopping kernel keeps
// each thread's neighbor-reuse working set in cache).  Every BLK_* must
// divide its lattice dim, and BLK_X must stay = Lx (x is swept innermost /
// prefetched).  See the detailed retuning note in clover_ferm.h for how to
// pick these per machine.
#define BLK_X  Lx
#define BLK_Y  4
#define BLK_Z  4
#define BLK_T  Lt
#define BLK2_X BLK_X
#define BLK2_Y BLK_Y
#define BLK2_Z BLK_Z
#define BLK2_T BLK_T

// Ceiling on the private per-worker scratch the windowed Galerkin build may
// allocate to escape its site-coloring parallelism limit (see
// build_coarse_op_lvl).  A worker needs two level-l fields plus one child
// field, so the affordable worker count falls as the level gets finer: this
// bounds the transient footprint rather than selecting an algorithm.
// 0 disables the private path entirely, leaving every build on the
// shared-scratch colored sweep.
#ifndef COARSE_BUILD_SCRATCH_MB
#define COARSE_BUILD_SCRATCH_MB 4096
#endif

// ===========================================================================
// 3. Algorithm switches (developer A/B toggles)
// ===========================================================================

// Even-odd (Schur) outer solve, QUDA --solve-type direct-pc style: FGMRES runs
// on the even-site Schur system D_hat x_e = b_e - D_eo D_oo^{-1} b_o (the
// driver forms the RHS and back-substitutes x_o); the MG preconditioner stays
// on the full operator, bridged by zero-pad / even-slice (mg_precond_single).
// Outer Krylov vectors halve, so the GS/BLAS-1 rows and the V/Z-basis memory
// halve; the outer iteration count is unchanged (cycle quality, not kappa(D),
// sets it).  Valid on both outer-Krylov precisions: the mixed build's
// single-Krylov path (the Arnoldi matvec becomes the bridge's single Schur
// operator, mg_D_hat_single) and a one-precision MIXED=0 build (plain D_hat via
// the generic solver_info hook).  The driver must form
// the Schur RHS, back-substitute the odd sites, and hand the bridge the
// odd-inverted clover (mg_setup_outer_eo_mixed) -- see test.c's correlator test
// or solve()'s eo path.
#define OUTER_EO          1            // 1 = eo/Schur outer solve

// Coarsest (bottom) solve: 1 = exact dense-LU inverse, built once at setup
// (coarse.c coarse_direct_*); 0 = iterative coarse_gmres/coarse_solve_eo.
// An algorithm A/B, not a size heuristic: on a tiny bottom the inverse is
// strictly better, but on a larger bottom (e.g. 4^4) dense-vs-iterative is
// worth measuring directly.
#ifndef BOTTOM_DENSE_INV
#define BOTTOM_DENSE_INV  1
#endif
// Coarse solve via even-odd Schur (0 = full operator).  Read only where the
// coarsest solve is iterative: with BOTTOM_DENSE_INV the exact bottom takes
// its own route and this has no effect.
#define MG_COARSE_EO      1
// BOTTOM_DENSE_INV inverse storage/apply: 1 = fp32 (half the inverse footprint
// and stream traffic, which is what the apply is bound by), 0 = fp64 (exact to
// the fp32 hierarchy floor).  The factorization stays fp64 either way; only
// the stored copy narrows, so effective bottom accuracy becomes
// ~1e-7 * kappa(S) -- the build self-check prints the realized value.
#define BOTTOM_INV_FP32   1
// Schur block matvec: asm kernel (0 = generic)
#define COARSE_USE_ASM    1
// 1 = tally ||b|| at the coarsest bottom solve
// (diagnostic: is the restricted residual zero?)
#define BOTTOM_SOLVE_DIAG 0

// Coarse-level smoother:
// 1 = replace the per-block MR with one GEMM against each block's exact
//     inverse, precomputed (fp64) at setup from the block-local Dirichlet
//     operator.  The inverse is exact, so the block solve matches
//     coarse_sap_block_iter -> infinity; the lever is per-apply efficiency
//     (dense GEMM regime), not iteration count.
// 0 = the iterative per-block MR smoother, which converges differently.
#ifndef COARSE_SAP_DENSE_INV
#define COARSE_SAP_DENSE_INV 1
#endif
// 1 = time the five eo-apply stages and the halo pack/MPI/scatter split, and
//     print a per-stage GFlop/s report at exit
//     (make XCFLAGS=-DCOARSE_SAP_PROFILE=1).  Requires the dense inverse
//     above, whose counters the stage timers reference.
#ifndef COARSE_SAP_PROFILE
#define COARSE_SAP_PROFILE 0
#endif
#if COARSE_SAP_PROFILE && !COARSE_SAP_DENSE_INV
#error "COARSE_SAP_PROFILE requires COARSE_SAP_DENSE_INV=1"
#endif

// K-cycle assembled matvec:
// 1 = the K-cycle FGMRES forms c_j = D_c z_j as (r - r_exit), where r_exit =
//     r - D_c z_j is the post-smoother exit residual that coarse_sap's
//     incremental-residual state provides almost for free (the last-phase
//     color is current in sap_R; the other color needs one cross-coupling
//     refresh ~ 1/4 of a residual recompute).  Deletes one full coarse
//     Dslash per Arnoldi step.  Algebraically exact; it differs from the
//     explicit apply only at rounding level -- judge it by the outer iteration
//     count.
// 0 = explicit apply_D_lvl, the reference the fused form is judged against
//     (make XCFLAGS=-DKCYCLE_ASSEMBLED_W=0).
#ifndef KCYCLE_ASSEMBLED_W
#define KCYCLE_ASSEMBLED_W 1
#endif

// ===========================================================================
// 4. Fixed and derived values -- do not edit
// ===========================================================================

// colors and spins
#define Nc 3
#define Ns 4

// --- Fine Wilson/clover operator flop counts (per site, per rhs) ------------
// Single-sourced so the reported GFlop/s (compared against processor
// rooflines) uses one convention everywhere -- every solver and profiler.
// Convention matches QUDA and DDalphaAMG:
//   WILSON_FLOPS = 1320 hopping (8 dirs; SU(3) matvec counted as the efficient
//     8*Nc-2 per row, i.e. the leading term is a multiply, not an accumulate)
//     + 48 mass/diagonal (the x - kappa*D xpay) = 1368, the no-clover fine D.
//   CLOVER_FLOPS = 552 = 2 chiral blocks x (8n^2 - 2n), n = (Ns/2)*Nc = 6, i.e.
//     each 6x6 chiral block as a full complex matrix-vector.  (The CPU kernel
//     exploits the real Hermitian diagonal and executes 528; QUDA's standalone
//     clover uses 504; 552 is the fused-operator convention both references
//     publish -- change here to recount under a different convention.)
// Change these in one place to switch conventions; every counter follows.
#define WILSON_FLOPS         1368
#define CLOVER_FLOPS         552
#define WILSON_CLOVER_FLOPS  (WILSON_FLOPS + CLOVER_FLOPS)

// Local extents -- what every field layout, loop bound, and kernel uses.
#define Lxh (Lx/2)
#define Lz (Lz_glob / MPI_GRID_PZ)
#define Lt (Lt_glob / MPI_GRID_PT)

// derived parameters (all per-rank at a decomposed grid)
//
// Deliberately unparenthesized, and it has to stay that way.  The codebase
// writes `(long) ferm_size * num_rhs` in dozens of places; with the product
// open, the cast lands on the first factor and the whole multiplication is
// done in long, which is what keeps a production volume from overflowing int.
// Parenthesized, the same expression would compute the product in int first
// and convert the overflowed result.  The cost is that these must never be
// used as a divisor or after % without parentheses of their own.
#define V Lx * Ly * Lz * Lt
#define Vh (Lx/2) * Ly * Lz * Lt
// number of doubles in lattice objects
#define gauge_size V * 4 * Nc * Nc * 2
#define ferm_size V * Nc * Ns * 2
#define half_ferm_size Vh * Nc * Ns * 2

// Whole-lattice gauge element count -- what a configuration file holds, which
// is independent of the rank decomposition (gauge_size above is per-rank).
// Used by read_gauge to check the ILDG binary record's length against this
// build.  Parenthesized and long-typed on purpose: it gets multiplied by 8
// for a byte count, which overflows int at production volumes.
#define gauge_size_glob ((long) Lx * Ly * Lz_glob * Lt_glob * 4 * Nc * Nc * 2)

// --- Halo (ghost-slice) geometry for the fine-lattice decomposition ---------
// Wilson D is nearest-neighbor, so a rank only ever reads one site deep into
// its z/t face neighbors (x,y wrap locally).  Fermion ghosts are not stored
// with the fields: the dispatch layer (clover_ferm.c) exchanges the input
// field's faces into a shared per-precision scratch block before each stencil
// sweep, and the kernels read boundary neighbors from there.  Gauge ghosts
// (backward links only) are appended past the local volume of the few
// long-lived gauge arrays (gauge_alloc_size), filled once by prepare_gauge.
// Everything here is 0 / empty at the default 1x1 grid.
#define DECOMP_Z (MPI_GRID_PZ > 1)
#define DECOMP_T (MPI_GRID_PT > 1)
#define FERM_GHOSTS (DECOMP_Z || DECOMP_T)
// half-lattice sites in one z / t ghost face of one parity half
#define GHOST_FACE_Z (Lxh * Ly * Lt)
#define GHOST_FACE_T (Lxh * Ly * Lz)
// Face offsets (in half-lattice sites) inside one parity's ghost block; faces
// of undecomposed dims are absent.  Order: [z-lo | z-hi | t-lo | t-hi].
#define G_OFF_ZLO 0
#define G_OFF_ZHI GHOST_FACE_Z
#define G_OFF_TLO (2 * GHOST_FACE_Z * DECOMP_Z)
#define G_OFF_THI (G_OFF_TLO + GHOST_FACE_T)
#define GHOST_SITES (2 * GHOST_FACE_Z * DECOMP_Z + 2 * GHOST_FACE_T * DECOMP_T)
// Gauge ghosts: only the lo faces are read (the backward links U_mu(x-mu));
// appended at full-lattice site index V, order [z-lo | t-lo].
#define GG_OFF_ZLO 0
#define GG_OFF_TLO (Lx * Ly * Lt * DECOMP_Z)
#define GAUGE_GHOST_SITES (Lx * Ly * Lt * DECOMP_Z + Lx * Ly * Lz * DECOMP_T)
#define gauge_alloc_size (gauge_size + GAUGE_GHOST_SITES * 4 * Nc * Nc * 2)

// 2-vs-multilevel gate handed to mg_hierarchy_build: 0 selects the two-level
// setup, 1 the recursive one that builds all NUM_LEVELS.
#define MULTILEVEL        (NUM_LEVELS >= 3)

// Largest per-transition null-vector count anywhere, for the few stack arrays
// / field paddings in the coarse path that must accommodate the widest level.
#define MAX_NULL_VECS ((num_null_vecs) > (COARSE_NULL_VECS) \
                       ? (num_null_vecs) : (COARSE_NULL_VECS))

// ===========================================================================

// Everything above is integer #defines safe for both C and the assembler.
// Everything below is C-only (it pulls in <tgmath.h>, SIMD intrinsic types, and
// the malloc macro), so it is hidden from the .S kernels, which include this
// header purely to pick up num_rhs/Nc/Ns.  cpp auto-defines __ASSEMBLER__ when
// preprocessing a .S file.
#ifndef __ASSEMBLER__

// Precision, SIMD vector type, and num_rhs_vec are selected per translation
// unit (keyed on MG_SINGLE) in precision.h -- included here, at the end, so
// num_rhs above is already defined.  A TU that does not set MG_SINGLE (the
// double-only driver layer) gets double by default.
#include "precision.h"

// Symbol-mangling layer: rewrites the operator / MG / inner-solver / utils
// externals to precision-suffixed names (foo -> foo_f / foo_d) keyed on this
// TU's MG_SINGLE, so the single and double instantiations coexist in one binary
// and plain-name references in every TU resolve to the matching precision.
// Included here (after precision.h, which defines PFX) so it applies uniformly.
#include "mangle.h"


// Aligned, checked memory allocation.
//
// Every field a kernel reads or writes must come from here.  The generated
// assembly and the SIMD intrinsic paths use aligned moves throughout -- there
// is not one unaligned load in any of the kernels -- while the C library's
// malloc and calloc promise only 16 bytes.  Gauge, clover and coarse coupling
// matrices are read by broadcast and so do not care; fermion-shaped fields
// do.  Routing calloc here too means no field can reach a kernel
// under-aligned whichever call produced it.
//
// These macros are the library's one allocation choke point, so the failure
// check lives there rather than at ~250 call sites: alloc_checked aborts with
// the requesting file:line and the machine's memory state (comm.c), which is
// what turns an exhausted node into a diagnosis instead of a NULL dereference.
// A size that overflowed on the way in arrives here as an absurd size_t and is
// reported as such, naming the site that computed it.
//
// alloc_try is the opt-out for the handful of callers that treat "no memory" as
// a condition to handle rather than a fatal error -- the dense coarsest-level
// paths in coarse.c, which fall back to an iterative solve.  Everything else
// wants the abort.
#include <stddef.h>
void * alloc_checked (size_t bytes, const char * file, int line);
void * calloc_checked(size_t nmemb, size_t size, const char * file, int line);
void * alloc_try     (size_t bytes);

#define malloc(x)    alloc_checked ((size_t) (x), __FILE__, __LINE__)
#define calloc(n, s) calloc_checked((size_t) (n), (size_t) (s), __FILE__, __LINE__)

// --- Fine-lattice site-storage order (the funnel) ---------------------------
// Every fine-level site access goes through these maps; no other code may
// assume a storage order.  The map itself lives in the top-level,
// self-contained site_layout.h, where LOC_HOST_BLOCK selects the array order
// (host_loc_*).
//
// site_half: index of half-lattice site (xh, y, z, t) within one parity half
//   -- fermion fields and the eo clover (F_csw / F_oo_inv / sap Fmi).  The
//   caller owns the parity * Vh offset (parity-major halves).
// site_full: index of full-lattice site (x, y, z, t) -- the gauge field and
//   the pre-eo clover intermediate.  Stays lexicographic independently of the
//   fermion order (the gauge array's order is its own decision; the ghost
//   tails at index V and the GG_OFF_* enumeration depend on it).
// site_half_coords: inverse of site_half, for consumers that must decode a
//   stored index (per-timeslice residual breakdowns).
// (own guard: global_params.h has no file-wide include guard -- its macros
// tolerate re-inclusion, function definitions do not)
#ifndef QC_SITE_FUNNEL_DEFINED
#define QC_SITE_FUNNEL_DEFINED
#include "site_layout.h"
#if LOC_HOST_BLOCK
// the block-major map tiles 4^4 blocks (2 in xh): every local extent must
// divide into whole blocks
_Static_assert(Lxh % 2 == 0 && Ly % 4 == 0 && Lz % 4 == 0 && Lt % 4 == 0,
               "block-major fine layout needs Lx%4==0 (x pairs), "
               "Ly/Lz/Lt%4==0 (whole 4^4 blocks per rank)");
#endif
static inline long site_half(int xh, int y, int z, int t) {
  return host_loc_of_dims(Lxh, Ly, Lz, Lt, 2 * xh, y, z, t);
}
static inline long site_full(int x, int y, int z, int t) {
  return (((long) t * Lz + z) * Ly + y) * Lx + x;
}
static inline void site_half_coords(long s, int * xh, int * y, int * z,
                                    int * t) {
  host_loc_inv_of_dims(Lxh, Ly, Lz, Lt, s, xh, y, z, t);
}
#endif // QC_SITE_FUNNEL_DEFINED

#endif // __ASSEMBLER__
