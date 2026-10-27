// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef COARSE_H
#define COARSE_H
#include "global_params.h"
#include "solver.h"          // solver_info / solver_function for the V-cycle
#include "mg_params.h"       // mg_params, the hierarchy's tuning inputs

// --- Algebraic multigrid: coarse grid geometry ------------------------------
// Aggregation-based, Gamma5-compatible standard aggregation (thesis Def. 4.7).
// The fine lattice is tiled into blocks of size BLOCK_AGG^4; each block
// becomes one coarse lattice site.  Each block contributes two aggregates
// (the thesis' A_{p,0} and A_{p,1}): one for fine spins {0,1} (tau=0) and one
// for spins {2,3} (tau=1) -- this is the Gamma5 = diag(1,1,-1,-1) split in the
// DeGrand-Rossi basis this codebase uses.  Every aggregate spans all 3 colors
// and all sites of its block.  Each aggregate induces N = num_null_vecs coarse
// variables, so a coarse site carries N_coarse = 2*num_null_vecs variables.

// BLOCK_AGG (the block edge) is a per-run tuning knob and lives beside
// SCHWARZ_BLOCK_SIZE in global_params.h's knob block, reached through the
// include above.

// --- Geometry consistency ---------------------------------------------------
// Both block edges, the lattice extents and the MPI process grid are
// compile-time constants, so every constraint between them is decidable at
// build time.  That is worth doing because each of these fails *silently* at
// runtime: none corrupts the answer (outer FGMRES still converges on the true
// operator), they only degrade the preconditioner, so the symptom is a
// mysterious iteration-count regression that reads like a bad tuning choice.
//
// They live here rather than beside the knobs in global_params.h because this
// is where the terms they relate come together: Lxc..Ltc and the coarse block
// geometry are derived just below, and coarse.h is compiled in every build.
// This is the single place these invariants are stated, and they are
// compile-time: a runtime-configurable process grid, rather than the
// compile-time MPI_GRID_*, would need checks of its own.
_Static_assert(Lz_glob % MPI_GRID_PZ == 0 && Lt_glob % MPI_GRID_PT == 0,
               "the MPI process grid must divide the global z/t extents: "
               "Lz/Lt are the per-rank quotients, and an inexact division "
               "truncates them");
_Static_assert(SCHWARZ_BLOCK_SIZE % 2 == 0,
               "SCHWARZ_BLOCK_SIZE must be even: the block-internal even-odd "
               "MR strides rows by bs/2, so an odd edge aliases distinct even "
               "sites onto one scratch slot and the block correction is "
               "partly garbage");
// Dividing the *local* extent also makes rank faces coincide with block faces,
// so the Schwarz boundary update only ever reads one ghost site of an
// opposite-color neighbor block and no aggregate straddles a rank.
// (Coarse-level sap_blk / BOOTSTRAP_BLK_LVL divisibility is checked where that
// geometry is built.)
_Static_assert(Lx % SCHWARZ_BLOCK_SIZE == 0 && Ly % SCHWARZ_BLOCK_SIZE == 0
            && Lz % SCHWARZ_BLOCK_SIZE == 0 && Lt % SCHWARZ_BLOCK_SIZE == 0,
               "SCHWARZ_BLOCK_SIZE must divide every local extent: the SAP "
               "sweep walks whole blocks with no tail clamp, so a remainder "
               "slab is never smoothed (and under MPI a block would "
               "straddle a rank)");
// The eo half-sweep tiling (clover_ferm.h OUTER_LOOPS_HALF / INNER_LOOPS_HALF)
// steps whole tiles with no boundary clamp -- x at half stride, on the
// half-lattice.
_Static_assert(block_size_half % 2 == 0
            && Lxh % (block_size_half / 2) == 0
            && Ly % block_size_half == 0
            && Lz % block_size_half == 0
            && Lt % block_size_half == 0,
               "block_size_half must tile every local extent (x at half "
               "stride on the half-lattice): the eo half-sweep loops step "
               "whole tiles with no boundary clamp");
// The color the smoother uses is the *global* checkerboard parity: solver.c
// folds the rank's block origin in via sap_rank_blk_color, so the torus that
// has to 2-color is the global block lattice, not the per-rank one.  An odd
// block count in any dimension makes the first and last block neighbors of
// the same color.
_Static_assert((Lx / SCHWARZ_BLOCK_SIZE) % 2 == 0
            && (Ly / SCHWARZ_BLOCK_SIZE) % 2 == 0
            && (Lz_glob / SCHWARZ_BLOCK_SIZE) % 2 == 0
            && (Lt_glob / SCHWARZ_BLOCK_SIZE) % 2 == 0,
               "each global extent must hold an even number of SAP blocks, or "
               "the red-black coloring collides across the periodic wrap and "
               "two adjacent blocks are updated in the same half-sweep");
_Static_assert(Lx % BLOCK_AGG == 0 && Ly % BLOCK_AGG == 0
            && Lz % BLOCK_AGG == 0 && Lt % BLOCK_AGG == 0,
               "BLOCK_AGG must divide every local extent: the coarse extents "
               "below are integer divisions, so a remainder silently truncates "
               "the coarse lattice and the aggregation drops the tail sites");

// coarse lattice dimensions (BLOCK_AGG divides the fine ones exactly --
// asserted just above, so these are exact)
#define Lxc (Lx / BLOCK_AGG)
#define Lyc (Ly / BLOCK_AGG)
#define Lzc (Lz / BLOCK_AGG)
#define Ltc (Lt / BLOCK_AGG)

// The Galerkin build separates the +mu and -mu couplings by probing one coarse
// block at a time, which needs the two directions to reach different
// neighbors; at an extent of 2 they reach the same one.  These are *local*
// extents over BLOCK_AGG, so it is usually the rank grid that runs into it --
// adding ranks in z or t shrinks them.  build_coarse_op_direct checks this
// too, but every term here is compile-time, so a bad grid is worth rejecting
// at build time rather than part-way through a setup.
_Static_assert(Lxc >= 3 && Lyc >= 3 && Lzc >= 3 && Ltc >= 3,
               "every local coarse extent (local extent / BLOCK_AGG) must be "
               "at least 3 -- use fewer ranks in z/t, a smaller BLOCK_AGG, or "
               "a larger local volume");
#define Vc  (Lxc * Lyc * Lzc * Ltc)
#define Vhc (Vc / 2)

// --- Coarse-lattice halo ghosts ---------------------------------------------
// z/t face slabs of a full coarse field (lexicographic, no even-odd split),
// mirroring the fine GHOST_* geometry on the macro depth-1 coarse lattice.  The
// coarse operator's APBC is baked into Dc->nbr at build time, so the halo is a
// pure periodic index-space face exchange (no boundary phase).  Ghost layout
// [z-lo | z-hi | t-lo | t-hi]; the face of an undecomposed dim is absent.
// A z face holds Lxc*Lyc*Ltc coarse sites, a t face Lxc*Lyc*Lzc.
#define COARSE_FACE_Z (Lxc * Lyc * Ltc)
#define COARSE_FACE_T (Lxc * Lyc * Lzc)
#define CG_OFF_ZLO 0
#define CG_OFF_ZHI COARSE_FACE_Z
#define CG_OFF_TLO (2 * COARSE_FACE_Z * DECOMP_Z)
#define CG_OFF_THI (CG_OFF_TLO + COARSE_FACE_T)
#define COARSE_GHOST_SITES \
  (2 * COARSE_FACE_Z * DECOMP_Z + 2 * COARSE_FACE_T * DECOMP_T)

// number of fine sites in one aggregation block
#define block_sites (BLOCK_AGG * BLOCK_AGG * BLOCK_AGG * BLOCK_AGG)

// coarse variables per coarse site (2 spin-aggregates x N test vectors).
// This is the L0->L1 width (num_null_vecs); the depth->=1 coarse path reads
// its per-level width from level_t.nvar, since the L1->L2 count
// (COARSE_NULL_VECS) may differ.
#define N_coarse (2 * num_null_vecs)
// number of REALs in a single-rhs coarse fermion: Vc sites x N_coarse complex
#define coarse_ferm_size ((long) Vc * N_coarse * 2)

// Map a full (non-eo) lexicographic coarse index to its block-origin fine
// coordinates.  Coarse lexicographic order matches the fine convention
// (t slowest, x fastest): cl = ((tb*Lzc + zb)*Lyc + yb)*Lxc + xb.
static inline void coarse_block_origin(int cl, int * x0, int * y0,
                                       int * z0, int * t0) {
  int xb = cl % Lxc; cl /= Lxc;
  int yb = cl % Lyc; cl /= Lyc;
  int zb = cl % Lzc; cl /= Lzc;
  int tb = cl;
  *x0 = xb * BLOCK_AGG;
  *y0 = yb * BLOCK_AGG;
  *z0 = zb * BLOCK_AGG;
  *t0 = tb * BLOCK_AGG;
}

// --- Block-local orthonormalization of the test vectors ---------------------
// Orthonormalize the N = num_null_vecs test vectors within each aggregate, in
// place on a field laid out with the null-vector index running fastest:
//   nv[(((loc * Ns + d) * Nc + c) * 2 + reim) * num_null_vecs + vec]
// After this, P^H P = I (P built from these vectors).
void local_orthonormalize(REAL * nv);

// Largest deviation of any aggregate's N x N Gram matrix from the identity
// (max over aggregates and i,j of |<v_i,v_j>_agg - delta_ij|); ~0 after
// local_orthonormalize.
REAL nullvec_orthonormality_error(REAL * nv);

// --- Prolongation P and restriction R = P^H ---------------------------------
// A coarse fermion is stored, in full (non-eo) coarse-lexicographic site order
// matching coarse_block_origin, as
//   coarse[((cl * N_coarse + v) * 2 + reim) * num_rhs + rhs]
// where the coarse variable v in [0, N_coarse) splits as v = num_null_vecs*tau
// + j: tau in {0,1} selects the spin aggregate (fine spins {0,1} or {2,3}) and
// j in [0, num_null_vecs) selects the test vector.  Size: coarse_ferm_size
// REALs per rhs, i.e. coarse_ferm_size * num_rhs for the multi-rhs field.
//
// Both transfers carry all num_rhs right-hand sides and need the locally
// orthonormalized test-vector field `nv` (the P columns).

// fine_out = P * coarse_in   (GAXPY, thesis sec 4.4.1).  fine_out is a standard
// fine fermion (even-odd, multi-rhs, ferm_size * num_rhs REALs).
void apply_P(REAL * fine_out, REAL * coarse_in, REAL * nv);

// One aggregation block of fine_out = P * coarse_in, with cached (not NT)
// stores -- for the fused first-color SAP prolong, whose fold seeds read the
// block straight back.  Bit-identical values to apply_P on that block.  Block
// coords in aggregation units (the fuse registration gate guarantees
// BLOCK_AGG == SCHWARZ_BLOCK_SIZE, so SAP visit block coords slot in).
void apply_P_block_cached(REAL * fine_out, REAL * coarse_in, REAL * nv,
                          int bx, int by, int bz, int bt);

// coarse_out = P^H * fine_in = R * fine_in.
void apply_R(REAL * coarse_out, REAL * fine_in, REAL * nv);

// apply_R for the eo outer solve: fine_in is the even-site half-field (odd part
// implicitly zero), so only even sites are gathered -- bit-identical to apply_R
// on the zero-padded field at half the cost (counted as a half call in the
// transfer flop stats).
void apply_R_even(REAL * coarse_out, REAL * fine_in, REAL * nv);

// Fused outer-normalize + depth-0 restrict (producer side, called through the
// mixed bridge's mg_normalize_restrict_single): scale w by the per-rhs rinv
// (num_rhs reciprocals, lane k applied to lane k -- a retired lane's 0 zeroes
// it) into the Krylov basis slot vnext (the V[m+1] normalize) and restrict
// the result to the coarse grid in the same cache-resident pass.  The coarse
// RHS is registered one-shot against the vnext pointer and consumed by the
// next depth-0 mg_cycle, which then skips its restrict.  eo = even-half-field
// mode (eo outer solve).  Returns 1 if the fused pass ran (vnext filled,
// registration made); 0 if the caller must normalize -- and the cycle
// restrict -- itself (no hierarchy yet).
int mg_restrict_fuse_normalize(REAL * vnext, REAL * w, const double * rinv,
                               int eo);

// Clears the apply_P / apply_R wall-time and call accumulators, including the
// half-call counts (apply_R_even, apply_P_par) that keep the GFlop/s honest.
// These have no standalone printer: they surface through mg_get_vcycle_profile
// as the profile table's transfer row.
void transfer_stats_reset(void);

// Clears the per-phase wall-time accumulators of mg_vcycle (restriction,
// coarse solve, prolongation, fine residual D, SAP smoother, vector ops).
// No standalone printer: they surface through mg_get_vcycle_profile as the
// profile table's rows.
void vcycle_stats_reset(void);

// Reset the k-cycle FGMRES Gram-Schmidt / coarse e-o Schur setup timers.
// Cleared together with the V-cycle timers before a profiled solve.
void other_stats_reset(void);

// Timing + (coarse-level) flop rate for apply_Dc.  Flops use Vc / N_coarse /
// num_rhs, not fine dims.
void apply_dc_stats_reset(void);
void apply_dc_stats_report(const char * tag);

// Clears the apply_coarse_schur timer -- the coarse matvec of the even-odd
// (MG_COARSE_EO) path, where apply_Dc is dormant.  No standalone printer: it
// is summed with the apply_Dc timer into mg_get_vcycle_profile's
// coarse-Dslash row, which is why both carry the same per-call flop model.
void apply_schur_stats_reset(void);

// Clears the coarsest bottom solve's fgmres_coarse timers: matvec (apply_Dc)
// vs. orthogonalization (Arnoldi Gram-Schmidt) vs. other (Givens/solve/update/
// residual).  No standalone printer: the orthogonalization and "other" halves
// surface through mg_get_vcycle_profile as the profile table's "coarse
// bottom-solve GS" and "coarse GMRES Givens/upd" rows.
void coarse_solve_stats_reset(void);

// Machine-readable snapshot of the V-cycle / transfer / coarse-Dslash timers,
// for the mixed-precision profiling summary.  All times are wall seconds summed
// since the last reset, and everything the recursion can do at more than one
// depth is reported per level in lvl[].  The Dslash entry collects all three
// implementations of the same matvec -- apply_Dc and apply_coarse_schur at the
// depth-1 macro geometry, apply_Dc_lvl below it -- since which one runs is a
// geometry dispatch, not a difference in the work.  The exact depth-0 transfer
// flop total (P and R carry different counts) is pre-summed into
// transfer_flops.  One level's slice.  Indexed by level_t.depth, so a
// hierarchy deeper than three reports each intermediate level separately
// instead of summing them.  A level leaves at zero whatever its role has no
// work for -- the coarsest has no smoother and nothing below it to transfer
// to.  Level 0's Dslash and smoother stay zero too, but for a different
// reason: the fine grid has finer-grained timers of its own
// (fgmres_profile_t, sap_stats_get), and the report prefers those to the
// cycle's outer brackets.
typedef struct {
  double R_secs, P_secs;       // restrict depth->depth+1, prolong back
  double trans_flops;          //   ... and the flops those two passes move
  double smooth_secs;          // post-smoother (coarse SAP at depth >= 1)
  double smooth_flops;
  double dslash_secs;          // D_l applies: assembled coarse stencil
  double dslash_flops;         long dslash_calls;
  double kc_secs, kc_flops;    // K-cycle FGMRES Gram-Schmidt run on this level
  double vec_secs, vec_flops;  // psi += smth
} mg_level_profile_t;

typedef struct {
  mg_level_profile_t lvl[NUM_LEVELS];
  double vc_coarse, vc_D, vc_smooth_fine;
  long   vc_calls, vc_citers, vc_calls_fine;
  double transfer_flops;       // exact depth-0 P+R flops (lvl[0].trans_flops)
  // The coarsest level's own rows.  Not per-level: each of these exists only at
  // the bottom, so its depth is NUM_LEVELS-1 by construction.
  // bottom-solve GS
  double cg_ortho_secs;        double cg_ortho_flops;
  // fgmres_coarse Givens/back-subst/update
  double cg_other_secs;
  long   kc_calls;
  // e/o Schur setup
  double eo_secs;              double eo_flops;     long eo_calls;
  // dense inverse apply
  double botd_secs;            double botd_flops;   long botd_calls;
} mg_vcycle_profile_t;
void mg_get_vcycle_profile(mg_vcycle_profile_t * p);
#if BOTTOM_SOLVE_DIAG
void bottom_solve_diag_reset(void);   // per-solve ||b|| tally at the
                                      // coarsest solve
void bottom_solve_diag_report(void);
#endif

// --- Coarse grid operator D_c = P^H D P -------------------------------------
// D_c is nearest-neighbor on the coarse lattice (Wilson has no diagonal
// hopping, so only the 8 face-neighbor coarse sites couple).  Each coupling is
// a dense N_coarse x N_coarse complex matrix, stored row-major as
// M[(a*N_coarse + b)*2 + reim] where a is the output (row) variable and b the
// source (column). Coarse neighbor directions dir = 0..7 map to
// +x,+y,+z,+t,-x,-y,-z,-t; the opposite of dir is dir ^ 4.
typedef struct {
  REAL * self;   // [Vc] blocks: self[p] = D_c[p, p]
  REAL * nbr;    // [Vc][8] blocks: nbr[p][dir] = D_c[p, p + offset[dir]]
} coarse_op;

// periodic nearest neighbor of coarse site `cl` in direction `dir` (0..7)
int coarse_neighbor(int cl, int dir);

// Build D_c = P^H D P (self + neighbor couplings) by probing each coarse site
// with unit coarse vectors, prolonging, applying the fine operator D, and
// restricting (thesis sec 4.4.2, "simple" O(n_c) construction).  Shell face
// sites take only the single inward fine hop (D_hop_local) rather than the
// full 8-direction stencil -- 1 hop vs 8 at each shell site, which is the
// bulk of the neighbor-coupling cost.  U_kappa/F_csw/clover are the prepared
// fine operator fields (as solve() builds them at eo_precondition=0).
void build_coarse_op_direct(coarse_op * Dc, REAL * nv,
                            REAL * U_kappa, REAL * F_csw, int clover);
void free_coarse_op(coarse_op * Dc);

// The same D_c, built the plain way: the full 8-direction fine stencil at
// every shell site.  This is the definition build_coarse_op_direct optimizes,
// kept as its correctness oracle (test_coarse_direct); the two agree
// bit-for-bit in the double TU (the single TU's asm full kernel differs from
// the C masked kernel at the float-rounding level).  Not used in production
// -- a wrong D_c would only cost preconditioner iterations, never corrupt the
// solve, so the direct build needs an independent reference to be checked
// against.
void build_coarse_op(coarse_op * Dc, REAL * nv,
                     REAL * U_kappa, REAL * F_csw, int clover);

// Coarse Dslash: out = D_c * in  (multi-rhs coarse fermions).  apply_Dc is the
// production kernel (hand-vectorized over num_rhs); apply_Dc_ref is the scalar
// reference kept as a correctness oracle.
void apply_Dc(coarse_op * Dc, REAL * out, REAL * in);
void apply_Dc_ref(coarse_op * Dc, REAL * out, REAL * in);

#if COARSE_SAP_DENSE_INV
// `./qc csapbench`: See_inv-apply kernel bench (hot-L2 vs DRAM-stream regimes;
// see the definition in coarse.c).  Perf diagnostic only, no correctness role.
void coarse_csap_bench(void);
// `./qc csapcheck`: packed-Hermitian apply vs the dense reference on a random
// Gamma5c-Hermitian matrix -- the fast (SDE-friendly) gate for the tile
// kernels.
void coarse_csap_check(void);
#endif

// --- Even-odd (Schur complement) preconditioning of the coarse system -------
// The coarse lattice is bipartite under the checkerboard (every
// nearest-neighbor hop flips parity), so D_c = [[A_ee, D_eo],[D_oe, A_oo]]
// with the site-local `self` blocks on the diagonal and the `nbr` hops
// off-diagonal.  The Schur complement S = A_oo - D_oe A_ee^{-1} D_eo acts on
// odd coarse sites only, so the coarse GMRES runs on a half-size,
// better-conditioned Krylov space.  coarse_eo_op borrows an existing coarse_op
// (does not own it); A_ee^{-1} is precomputed per even site.
struct level;            // forward decl (full definition below)

typedef struct {
  coarse_op * Dc;    // borrowed full coarse operator
  int  depth;        // level this was built for; the Schur apply charges its
                     //   time to that level's profile row
  int  nvar;         // coarse-site width of Dc (2*num_null_vecs at the depth-1
                     //   macro level, this level's 2*nnv_parent at the
                     //   coarsest); set from lv at build so the apply/solve
                     //   are level-correct
  long vol, volh;    // geometry: total / half coarse sites.  Vc/Vhc at the
                     //   depth-1 macro level; the runtime level's vol/volh at
                     //   the 3-level+ coarsest.  The apply/solve read volh as
                     //   their only geometry input (neighbors are baked into
                     //   the index tables below at build time).
  REAL * Aee_inv;    // [volh] negated inverse even self-blocks (-A_ee^{-1},
                     //   interleaved complex; sign folded for the Schur apply)
  REAL * te;         // [Vhc] even scratch (D_eo result) for apply_coarse_schur
  REAL * ue;         // [Vhc] even scratch (A_ee^{-1} result)
  // Hop directions that carry a coupling at all.  On a level whose extent in
  // some dimension is 2, +mu and -mu name the same neighbor, and the Galerkin
  // build already sums both ring-hops into one stencil slot and leaves the
  // other slot identically zero -- so the eo hops would otherwise run a full
  // dense matvec, and pull its matrix off DRAM, against a zero.  Detected by
  // scanning the stored couplings rather than from the geometry, which makes
  // skipping them exact whatever the reason a slot is empty.
  int    hop_dir[8];     // the surviving direction indices, ascending
  int    n_hop;          // how many of them (8 on a non-degenerate level)
  // Output-row panels per site for the eo solve's parallel loops.  The loops
  // run over volh sites, which on a small coarsest level is fewer than the
  // thread pool; splitting each site's output rows fills the pool.
  // 1 whenever volh already covers the threads, so a large level pays nothing
  // for this.
  int    npanel;
  int * even_lex;     // [Vhc] eo-even index -> coarse lexicographic index
  int * odd_lex;      // [Vhc] eo-odd  index -> coarse lexicographic index
  int * odd_of_even;  // [Vhc*8] odd-set eo-index of each even site's neighbors
  int * even_of_odd;  // [Vhc*8] even-set eo-index of each odd site's neighbors

  // BOTTOM_DENSE_INV: borrowed pointer to the level's cached explicit
  // odd-Schur inverse S^{-1} (level_t.bot_inv; col-major complex, fp64 or
  // fp32 per BOTTOM_INV_FP32).  Single-rank: the full inverse, order
  // volh*nvar.  Multi-rank: only this rank's rows, shape
  // (volh*nvar) x dense_N, where dense_N is the global odd-Schur order
  // (volh*nvar summed over ranks) -- the apply allgathers the RHS to length
  // dense_N and GEMMs the local rows against it.  When set, coarse_solve_eo
  // replaces its inner GMRES with one dense apply.  Owned and freed by the
  // level (coarse_direct_free), never by free_coarse_eo_op.
  void * dense_inv;
  long   dense_N;            // global odd-Schur order (== volh*nvar at 1 rank)

  // Cross-rank coarsest eo-Schur halo state (allocated only when this level is
  // decomposed; ghost_sites == 0 / NULL pointers at MPI=0, so the Schur apply/
  // solve stay byte-identical there).  The Schur half-fields are eo-packed,
  // but halo_coarse exchanges a full lexicographic face slab, so each
  // cross-rank hop scatters the packed half into `gfull` (full-lex),
  // exchanges into `ghost`, and reads the neighbor block back from `ghost` by
  // its in-plane face index.  Parity is global (corigin-offset, set at
  // build), so a globally-odd site is odd on every rank and the scatter/read
  // agree across the face.
  int    gnx, gny, gnz, gnt;     // this level's extents (ghost face indexing)
  long   ghost_sites;            // total ghost face sites (0 = not decomposed)
  long   cg_zlo, cg_zhi, cg_tlo, cg_thi;   // ghost-site base offsets
                                           //   (halo_coarse)
  int  * oe_ghost;  // [Vhc*8] even site nbr: ghost-site offset if cross-rank,
                    //   else -1
  int  * eo_ghost;  // [Vhc*8] odd  site nbr: ghost-site offset if cross-rank,
                    //   else -1
  REAL * gfull;     // [vol  * blk] full-lex scatter scratch for the exchange
  REAL * ghost;     // [ghost_sites * blk] exchanged z/t face slabs

  // Comm/compute overlap partition for the decomposed coarsest Schur matvec.
  // Each parity's half-sites are split into interior (no cross-rank neighbor ->
  // computable while the face exchange is in flight) and boundary (>=1 ghost
  // neighbor -> needs the landed halo).  Built only when decomposed; at MPI=0
  // the apply takes the straight 0..volh loops and these stay NULL/0.
  int  * even_int; long n_even_int;  // interior even half-sites (eo-index list)
  int  * even_bnd; long n_even_bnd;  // boundary even half-sites
  int  * odd_int;  long n_odd_int;   // interior odd half-sites
  int  * odd_bnd;  long n_odd_bnd;   // boundary odd half-sites
} coarse_eo_op;

// Build the even-odd Schur operator from `Dc`.  `lv` selects the geometry:
// NULL uses the depth-1 macro lattice (Vc/Vhc,
// coarse_neighbor/coarse_parity); a non-NULL level uses its runtime vol/volh
// and coarse_neighbor_lvl (the 3-level+coarsest).  Returns 0 on success, or
// -1 when a runtime level's parity is unbalanced (the caller then falls back
// to the full coarse_solve_full_lvl).
int build_coarse_eo_op(coarse_eo_op * eo, coarse_op * Dc,
                       const struct level * lv);
void free_coarse_eo_op(coarse_eo_op * eo);

// Schur operator on odd sites: out_odd = (A_oo - D_oe A_ee^{-1} D_eo) in_odd.
// in_odd / out_odd are half-size coarse fields (Vhc sites, multi-rhs).
void apply_coarse_schur(coarse_eo_op * eo, REAL * out_odd, REAL * in_odd);

// Solve D_c x = b via the odd-site Schur complement (restarted GMRES on S),
// recovering the even part directly.  x, b are full-size coarse fields
// (coarse_ferm_size * num_rhs REALs).  Returns the inner Schur GMRES
// iterations.
int coarse_solve_eo(coarse_eo_op * eo, REAL * x, REAL * b,
                    int max_iter, REAL eps, int restart);

// Restarted GMRES on the coarse system D_c x = b (multi-rhs coarse fermions,
// x and b sized coarse_ferm_size * num_rhs).  x is overwritten (zero initial
// guess).  `restart` is the cycle length (<=0 picks a default).  Returns the
// total number of inner iterations.  Quiet: the V-cycle calls it repeatedly.
int coarse_solve_full(coarse_op * Dc, REAL * x, REAL * b,
                      int max_iter, REAL eps, int restart);

// --- Multilevel hierarchy descriptor ----------------------------------------
// One grid in the V/W/K-cycle recursion.  mg_setup seeds levels 0 and 1;
// mg_add_coarse_level appends further levels, and the cycle /
// operator-dispatch bottom-solve code in coarse.c recurses over `coarser`,
// so depth is purely a setup concern.
//
// Performance boundary: only per-level *geometry* lives here as runtime fields,
// and it is used solely in outer-loop bounds and field sizing.  The per-site
// arithmetic dimensions (Nc, Ns, num_rhs, num_null_vecs, N_coarse) stay
// compile-time macros, so apply_Dc / D / the coarse asm kernel compile to
// byte-identical code regardless of depth.
typedef struct level level_t;
struct level {
  int   depth;                 // 0 = finest, deepest = coarsest

  // Geometry of this level (runtime, per level).  Note: these are named to
  // avoid the global geometry macros (Lx/Ly/Lz/Lt/V/Vh/ferm_size in
  // global_params.h), which would otherwise textually replace any
  // identically-named field.
  int   nx, ny, nz, nt;        // lattice extents at this level
  long  vol, volh;             // volume and half-volume
  // Global coordinate origin of this rank's local block, in this level's sites
  // (fine origin at depth 0; parent_origin / blk per level).  {0,0,0,0} at
  // MPI=0. Used by the coarsest eo-Schur build to label parity globally, so
  // the even/odd sets stay consistent across a rank face under decomposition.
  int   corigin[4];
  int   nvar;          // variables per site: Nc*Ns at depth 0, else 2*nnv of
                       // the coarsening that built this level (2*num_null_vecs
                       // at depth 1, 2*COARSE_NULL_VECS at depth 2, ...)
  int   nnv;           // null vectors used to coarsen this level into its
                       // child (= child->nvar/2); 0 on the coarsest level.
                       // Distinct from nvar: nvar is this level's own width,
                       // nnv is the next transition's column count.
  long  field_len;     // single-rhs field length in REALs: vol * nvar * 2

  // Aggregation factors tiling this level into its child (per dimension, so
  // the block can differ by direction and by level).  Child extent
  // = n{x..t}/blk[d].  Depth 0 mirrors the fine BLOCK_AGG macro (the fine
  // transfers use the macro directly); coarse levels read blk[] at runtime.
  int   blk[4];                // {x, y, z, t} block factors

  // SAP (red-black multiplicative Schwarz) smoother block factors for this
  // level, used only at depth >= 1 by coarse_sap: the smoother's
  // domain-decomposition block, distinct from the aggregation factor blk[].
  // Set on intermediate levels by mg_add_coarse_level; unused at depth 0
  // (which uses the fine red_black_schwarz with SCHWARZ_BLOCK_SIZE) and on
  // the coarsest level.
  int   sap_blk[4];

  // operator D at this level (exactly one branch is active)
  REAL * U_kappa, * F_csw;  // depth 0: fine Wilson-clover handles
                            //  (patched per apply)
  int    clover;
  coarse_op * Dc;           // depth >= 1: assembled nearest-neighbor
                            //   coarse stencil

  // post-smoother on this level (unused on the coarsest level)
  int    nu;                // post-smoothing sweeps
  int    blk_iter;          // MR iterations inside each smoother block: the
                            // fine SAP's sap_block_iter at depth 0, the coarse
                            // SAP's block iterations below it

  // Cycle shape, copied onto every level from mg_params so that the recursion
  // (which is handed a level, not the parameters) can read it wherever it is.
  // Uniform across the hierarchy -- a level is where they are reachable, not a
  // per-level setting.
  mg_cycle_t cycle;
  int    w_mu;              // W-cycle: recursive cycles per level
  int    k_iters;           // K-cycle: max Arnoldi steps per level
  double solve_k_tol;       // K-cycle target for a solve-time cycle; read off
                            // level 0 by mg_vcycle, which has no other route
                            // to the parameters

  // interpolation from level depth+1 into this level (NULL on the coarsest
  // level)
  REAL * nv;                // P columns (locally orthonormalized test vectors)

  // bottom-solve controls (used when this level is the coarsest)
  coarse_eo_op eo; int eo_built;
  int    cg_max_iter, cg_restart;
  REAL   cg_tol;

  // BOTTOM_DENSE_INV exact bottom solve: the cached col-major complex explicit
  // inverse (computed in fp64 once per operator rebuild; stored fp64 or fp32
  // per BOTTOM_INV_FP32; applied by slab GEMM at each solve).  bot_inv is the
  // odd-Schur inverse S^{-1} (order volh*nvar), applied inside coarse_solve_eo
  // via eo.dense_inv -- mg_add_coarse_level rejects a coarsest that cannot
  // eo-split, so there is no full-operator variant.  bot_N == 0 unless built.
  void   * bot_inv;            // [N*N] col-major complex explicit inverse
  long     bot_N;              // inverse order (0 = unbuilt)

  // coarse_sap per-level cache: per-thread block scratch + block geometry
  // (in-block neighbor map, per-block global site lists).  All of it depends
  // only on the level's extents, SAP block, and thread count -- fixed after
  // setup -- so it is built lazily on the first coarse_sap call at this level
  // (sap_nthr == 0 means unbuilt) and freed by mg_teardown.
  int     sap_nthr;         // threads the scratch was sized for (0 = unbuilt)
  REAL ** sap_rbuf, ** sap_Drbuf, ** sap_dbuf; // [sap_nthr] block-sized scratch
  int   * sap_lnbr;         // [bsites*8] in-block local nbr (-1 across a face)
  long  * sap_gsite;        // [nblocks*bsites] global lexicographic site lists
  long  * sap_cblk[2];      // per-color block-index lists (red/black sweep
  long    sap_ncblk[2];     //   iterates these directly, no skip pass)
  // incremental-residual state + cross-block geometry (gather form):
  REAL  * sap_R;            // [nblocks*blk_reals] each block's exact residual
                            //   as of its last visit (the MR solve's final r)
  REAL  * sap_dpsi;         // [nblocks*blk_reals] each block's last correction
  // COARSE_SAP_DENSE_INV (eo-Schur exact block solve): the dense even-Schur
  // inverse See_inv = (Mee - Meo Moo^-1 Moe)^-1 and the block-diagonal
  // odd-site inverse Soo_inv = Moo^-1, per block; the e<->o hopping is applied
  // on the fly from Dc->nbr.  sap_eoperm lists the even block-local sites
  // (first bsites/2) then the odd ones.  All three are NULL unless the dense
  // path is compiled in.  Storage format (sap_herm_tri): both inverses are
  // Gamma5c-Hermitian, so on block-major levels only the lower triangle of
  // the Hermitian factor H^-1 (A^-1 = H^-1 Gamma5c) is kept -- half the
  // matrix bytes, the stream the apply is bandwidth-bound by.  Stage-major
  // levels keep the full mirrored matrix: their See_inv apply is row-split
  // across threads, which the fused triangle kernel's mirrored updates cannot
  // honor.
  REAL  * sap_See_inv;      // [nblocks] full (bsites/2*nvar)^2*2 or packed tri
  REAL  * sap_Soo_inv;      // [nblocks*(bsites/2)] full nvar^2*2 or packed tri
  int   * sap_eoperm;       // [nblocks * bsites] even locals then odd locals
  int     sap_herm_tri;     // 1 = packed-triangle Hermitian inverse storage
  // Odd-site self-inverse folded into the odd<-even couplings: Moe''[o][dir] =
  // Moo_o^-1 . Moe[o][dir].  The block solve's closing stage is
  // x_o = y_o - Moo^-1 (Moe x_e), and its Moo^-1 acts on a vector that same
  // stage just formed, so pushing the inverse into the coupling at setup
  // turns the stage into a plain hop plus a subtraction and removes one
  // per-odd-site dense apply.  (The opening stage's y_o = Moo^-1 r_o stays:
  // the subtraction still needs it.)  Couplings are packed per block in
  // (odd index, direction) ascending order -- the order the stage's own
  // neighbor walk visits them -- so the stage indexes them with the counter
  // it already keeps.  sap_moe_off holds the absolute index of each
  // (block, odd index) run because blocks at even and odd origins have
  // complementary parity maps, hence different per-site coupling counts.
  REAL  * sap_Moe_fold;    // [nblocks * odd->even in-block couplings] * matsz
  long  * sap_moe_off;     // [nblocks * (bsites/2)] first coupling of each run
  REAL ** sap_ybuf, ** sap_ebuf, ** sap_ebuf2, ** sap_sbuf;  // eo-Schur scratch
  // Stage-major (intra-block-parallel) smoother state.  A deep level with
  // fewer SAP blocks per color than threads starves the block-major sweep
  // (most threads idle at every color barrier), so its eo-Schur block solve
  // is instead run stage-by-stage, each stage a parallel loop over
  // (block x site).  That needs the per-block intermediates to persist across
  // stages, so they move from per-thread scratch to these per-color-block
  // arrays (sized to the larger color's block count).  Enabled (and
  // allocated) only when sap_stage_major; NULL/0 otherwise.  Multi-rank levels
  // are included rather than excluded: a decomposed run has the fewest blocks
  // per rank per color, so its deep levels are the likeliest to need this path.
  int     sap_stage_major;  // 1 = intra-block-parallel eo-Schur solve
  long    sap_max_ncb;      // max(sap_ncblk[0], sap_ncblk[1]) (buffer sizing)
  REAL  * sap_sr;      // [max_ncb*blk_reals]    block residual r (gather->Meo)
  REAL  * sap_sy;      // [max_ncb*blk_reals]    y_o = Moo^-1 r_o (Soo->Meo,Moe)
  REAL  * sap_sse;     // [max_ncb*(bsites/2)*S] s_e = r_e - Meo y_o
                       //   (Meo->See_inv)
  REAL  * sap_seb2;    // [max_ncb*(bsites/2)*S] x_e = See_inv s_e
                       //   (See_inv->scatter)
  long  * sap_nbrblk;  // [nblocks*8] face-neighbor block ids
  int   * sap_xnbr;    // [bsites*8] crossing site's local index in the
                       //   neighbor block (valid where sap_lnbr < 0)
  // Cross-rank coarse-SAP halo state (allocated only when this level is
  // decomposed; NULL/unused at MPI=0, so the smoother stays byte-identical
  // there).  dpsi_field is a site-major (local lex) snapshot of the latest
  // per-block dpsi, exchanged per color phase into dpsi_ghost (z/t face slabs,
  // halo_coarse layout).  sap_nbr_rank flags the block faces whose neighbor
  // block lives on another rank -- those reads come from the ghost, not the
  // local sap_dpsi.
  REAL  * sap_dpsi_field;  // [field_len * num_rhs] site-major latest dpsi
  REAL  * sap_dpsi_ghost;  // [ghost_sites * N_coarse*2*num_rhs] boundary dpsi
  int   * sap_nbr_rank;   // [nblocks*8] 1 = this face neighbor is cross-rank
  long  * sap_ftab[8];    // [layer*2+color] color-halved halo tables: ascending
                          //   face-local site indices of boundary layer
                          //   (0=z-lo,1=z-hi,2=t-lo,3=t-hi) owned by that color
  long    sap_fcnt[8];    //   matching counts (0 on undecomposed dims)

  // Runtime-geometry coarse Dslash halo scratch (z/t face slabs of one
  // multi-rhs coarse field, sized from this level's extents + width).
  // apply_Dc_lvl refreshes it each apply, so a decomposed coarse level's
  // stencil reads the neighbor rank's boundary sites instead of the local
  // periodic wrap.  The depth-1 macro apply_Dc uses its own static scratch
  // instead; this is the per-depth analogue for every deeper level.
  // NULL / unused at MPI=0.  Allocated at level creation, freed by
  // coarse_sap_cache_free.
  REAL  * dc_ghost;

  struct level * coarser;      // level depth+1, or NULL at the coarsest
};

// --- Volume-aware (per-level) coarse operator -------------------------------
// Runtime-geometry analogues that read the lattice from a level_t instead of
// the Lxc.. macros, so one code path serves any coarse depth.
// coarse_neighbor_lvl is the lexicographic nearest neighbor; apply_Dc_lvl the
// Dslash; and build_coarse_op_lvl assembles D_{l+1} = R_l D_l P_l from
// fine_lv->Dc and the level-l test vectors nvc (handles degenerate small
// lattices, extent <= 2).
long coarse_neighbor_lvl(const level_t * lv, long cl, int dir);
void apply_Dc_lvl(const coarse_op * Dc, REAL * out, REAL * in,
                  const level_t * lv);
void build_coarse_op_lvl(coarse_op * Dc_out, const level_t * fine_lv,
                         REAL * nvc);
#if FERM_GHOSTS
// Size and allocate lv->dc_ghost, the halo scratch apply_Dc_lvl refreshes each
// apply.  mg_add_coarse_level does this for hierarchy levels; a hand-built
// level_t must call it itself, after n{x..t} and nvar are set, or the first
// decomposed apply exchanges into a NULL buffer.  Declared only where ghosts
// exist, so a misplaced call is a compile error rather than a link error.
void alloc_level_dc_ghost(level_t * lv);
#endif

// Geometry-agnostic coarse GMRES (fgmres_coarse + apply_Dc_lvl), for the
// bottom solve on coarse levels whose geometry is not the level-1 macro
// geometry.  x and b are sized lv->field_len * num_rhs.  Returns the total
// inner iterations.
int coarse_solve_full_lvl(const level_t * lv, REAL * x, REAL * b,
                          int max_iter, REAL eps, int restart);

// --- Internal building blocks exposed for the test harness ------------------
// coarse.c internals: production code calls them only from within coarse.c,
// but the selftests in test.c exercise them directly.  coarser_vol returns the
// number of level-(l+1) sites tiled from fine_lv; local_orthonormalize_coarse
// makes the per-aggregate test vectors orthonormal; apply_P_coarse /
// apply_R_coarse are the level-l prolongation / restriction; mg_cycle runs
// one recursive V/W/K cycle.
long coarser_vol(const level_t * fine_lv);
void local_orthonormalize_coarse(REAL * nvc, const level_t * fine_lv);
void apply_P_coarse(REAL * out, REAL * cin, REAL * nvc,
                    const level_t * fine_lv);
void apply_R_coarse(REAL * cout, REAL * fin, REAL * nvc,
                    const level_t * fine_lv);
// k_tol is the K-cycle early-exit relative residual for this cycle invocation
// (mg_params setup_k_tol during bootstrap, solve_k_tol during the solve); it
// threads down the recursion and is read only by the K-cycle (coarse.c
// k_cycle).
int  mg_cycle(level_t * lv, REAL * psi, REAL * eta, REAL k_tol);

// Coarse-level (depth >= 1) SAP post-smoother: phi <- nu red-black
// multiplicative Schwarz sweeps approximating D_l phi = eta (zero initial
// guess).  If r_exit is non-NULL it receives the exit residual eta - D_l phi.
// The solve drives it via smooth_lvl; the multi-rank halo test harness calls
// it directly.
int  coarse_sap(level_t * lv, REAL * phi, REAL * eta, REAL * r_exit);

// The rest of coarse_sap.c's interface to coarse.c.  coarse_sap_cache_free
// releases a level's lazily-built SAP cache (mg_teardown / mg_trim_to call
// it); coarse_sap_flops reports the modeled smoother flops for one level,
// pairing with that level's smoother time in the profile; csap_profile_reset
// clears the per-stage timers alongside the other solve-scoped counters.
void   coarse_sap_cache_free(level_t * lv);
double coarse_sap_flops(int depth);
void   coarse_sap_flops_reset(void);
#if COARSE_SAP_PROFILE
void   csap_profile_reset(void);
#endif

// --- Multigrid V-cycle preconditioner ---------------------------------------
// Register the multigrid context (coarse operator, orthonormalized test
// vectors, and solve parameters) used by mg_vcycle.  Must be called before
// using mg_vcycle as a preconditioner.  coarse_tol is relative (to ||R r||).
// When MG_COARSE_EO is nonzero (default), mg_vcycle solves the coarse system
// via the even-odd Schur complement (coarse_solve_eo); when zero, via the
// full-system coarse_solve_full.
//
// MG_COARSE_EO and COARSE_USE_ASM are hard-coded #defines in global_params.h
// -- compile-time constants, so the single and double instantiations of
// coarse.c agree on them with no shared object needed.  The cycle type and
// its counts are runtime instead: they arrive in mg_params (mg_params.h) and
// are stored on each level here, so both instantiations read the same struct
// rather than the same macro.  The K-cycle tolerance is passed per-invocation
// as mg_cycle's k_tol argument, since it differs between setup and solve.

void mg_setup(coarse_op * Dc, REAL * nv, const mg_params * p);

// Accessor for a registered hierarchy level (valid between mg_setup and
// teardown), used by the adaptive setup to start a recursive mg_cycle at a
// chosen depth
level_t * mg_get_level(int depth);

// Partial teardown: release levels keep..NUM-1, preserving 0..keep-1
// (operators, geometry, warm sap caches).  Used by the setup rebuild to keep
// the levels an inverse-iteration sweep did not change instead of rebuilding
// them from scratch.
void mg_trim_to(int keep);

// Append one coarser level (call after mg_setup; repeat for deeper
// hierarchies).  The current coarsest level becomes intermediate -- gaining
// aggregation block factors `blk[4]`, interpolation columns `nvc` (owned by
// caller, orthonormalized in place here), the coarse smoother from `p`, and
// its coarsening count `n_null` (the number of columns in nvc) -- and a
// freshly-built Galerkin grid D_{l+1} = R_l D_l P_l is added below it with
// bottom-solve controls cg_*.  n_null must be supplied explicitly (not
// derived from a prior nnv) so hierarchies deeper than 3 levels, whose
// intermediate levels are created coarsest, are sized correctly when promoted.
void mg_add_coarse_level(REAL * nvc, int blk[4], int n_null,
                         const mg_params * p);

// One multigrid cycle with post-smoothing (thesis Algorithm 8), packaged with
// the solver_function signature so it drops into a solver_info.preconditioner
// slot.  Computes output ~= D^{-1} input (zero initial guess):
//   output  = P D_c^{-1} R input                      (coarse grid correction)
//   r       = input - D output
//   output += M_SAP^(nu) r                            (SAP post-smoothing)
// It is the preconditioner of the flexible outer solver (FGMRES); at
// eo_precondition = 0 that runs on the full operator.
//
// eo_precondition = 1 (eo outer solve): `input` is the even-site half-field of
// the outer Schur system, odd part implicitly zero.  The cycle itself is
// unchanged (full operator); its two depth-0 consumers of `input` read the
// half-field directly (apply_R_even / D_residual_eo), bit-identical to running
// on the zero-padded field at lower cost.  `output` is still a full field --
// the caller takes the even half (S D^{-1} E = D_hat^{-1} makes that the exact
// Schur preconditioner form).
int mg_vcycle(REAL * output, REAL * input, REAL * U_kappa, REAL * F_csw,
              int clover, int max_iter, REAL eps, int eo_precondition,
              solver_info * preconditioner);

#endif // COARSE_H
