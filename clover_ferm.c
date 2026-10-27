// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <complex.h>
#include "global_params.h"
#include "utils.h"
#include "clover_ferm.h"
#include "comm.h"
#include "assert.h"
#include <string.h>

#define mult_U_dag() \
  prod_re = ZERO_VEC; \
  prod_im = ZERO_VEC; \
  for (int c2 = 0; c2 < Nc; c2 ++) { \
    int U_index = ((gauge_loc * 4 + dir) * Nc * Nc + c2 * Nc + c1) * 2; \
    REAL_VEC U_re = SET1(U[U_index]); \
    REAL_VEC U_im = SET1(U[U_index + 1]); \
    prod_re += U_re * temp_re[c2] + U_im * temp_im[c2]; \
    prod_im += U_re * temp_im[c2] - U_im * temp_re[c2]; \
  }

#define mult_U() \
  prod_re = ZERO_VEC; \
  prod_im = ZERO_VEC; \
  for (int c2 = 0; c2 < Nc; c2 ++) { \
    int U_index = ((gauge_loc * 4 + dir) * Nc * Nc + c1 * Nc + c2) * 2; \
    REAL_VEC U_re = SET1(U[U_index]); \
    REAL_VEC U_im = SET1(U[U_index + 1]); \
    prod_re += U_re * temp_re[c2] - U_im * temp_im[c2]; \
    prod_im += U_re * temp_im[c2] + U_im * temp_re[c2]; \
  }

// Fermion-load templating for a single-input/double-output kernel variant.
// FERM_VEC_T is the SIMD type of one rhs-block of the input fermion;
// LOAD_FERM_VEC fetches one block as a REAL_VEC (double).  The defaults are
// input == REAL == output precision; the mixed-precision D_si overrides them
// (input = float, promote-on-load via vcvtps2pd) -- see the gated
// wilson_kernel_c_si below. num_rhs_vec is unchanged: a __m128 holds 4
// floats, matching __m256d's 4 doubles, so the block count is identical.
#ifndef FERM_VEC_T
  #define FERM_VEC_T REAL_VEC
#endif
#ifndef LOAD_FERM_VEC
  #define LOAD_FERM_VEC(fv, idx) ((fv)[idx])
#endif

// Generalized version to track all phases.  Neighbor loads go through
// ferm_in; the per-direction input base the kernel preamble maintains the
// local field for x/y (and interior z/t) and the ghost scratch for off-rank
// z/t neighbors. At an undecomposed grid ferm_in is a const alias of ferm.
#define A_PHASE_B(a_re, a_im, b_re, b_im, OP1, OP2, U_OP) \
  /* temp = ferm[a] + ferm[b] */ \
  for (int c2 = 0; c2 < Nc; c2 ++) { \
    long ferm_index = (ferm_loc * Nc * Ns + c2) * 2; \
    FERM_VEC_T * ferm_vec = ((FERM_VEC_T*) (ferm_in + ferm_index * num_rhs)); \
    temp_re[c2] = LOAD_FERM_VEC(ferm_vec, a_re * num_rhs_vec + r) \
                  OP1 LOAD_FERM_VEC(ferm_vec, b_re * num_rhs_vec + r); \
    temp_im[c2] = LOAD_FERM_VEC(ferm_vec, a_im * num_rhs_vec + r) \
                  OP2 LOAD_FERM_VEC(ferm_vec, b_im * num_rhs_vec + r); \
  } \
  for (int c1 = 0; c1 < Nc; c1 ++) { \
    U_OP(); \
    /* output[a] += temp */ \
    /* output[b] += temp */ \
    long output_index = (result_loc * Nc * Ns + c1) * 2; \
    REAL_VEC * output_vec = ((REAL_VEC*) (output + output_index * num_rhs)); \
    output_vec[a_re * num_rhs_vec + r] += prod_re; \
    output_vec[a_im * num_rhs_vec + r] += prod_im; \
    output_vec[b_re * num_rhs_vec + r] OP1##= prod_re; \
    output_vec[b_im * num_rhs_vec + r] OP2##= prod_im; \
  }

// macro to compute (a +/- b) and multiply by U
#define A_OP_B(a, b, OP) \
  A_PHASE_B((2 * Nc * a), (2 * Nc * a + 1), \
            (2 * Nc * b), (2 * Nc * b + 1), OP, OP, mult_U)
// compute (a +/- bi)
#define A_MINUS_BI(a, b) \
  A_PHASE_B((2 * Nc * a), (2 * Nc * a + 1), \
            (2 * Nc * b + 1), (2 * Nc * b), +, -, mult_U)
#define A_PLUS_BI(a, b) \
  A_PHASE_B((2 * Nc * a), (2 * Nc * a + 1), \
            (2 * Nc * b + 1), (2 * Nc * b), -, +, mult_U)

// daggered versions
#define A_OP_B_DAG(a, b, OP) \
  A_PHASE_B((2 * Nc * a), (2 * Nc * a + 1), \
            (2 * Nc * b), (2 * Nc * b + 1), OP, OP, mult_U_dag)
// compute (a +/- bi)
#define A_MINUS_BI_DAG(a, b) \
  A_PHASE_B((2 * Nc * a), (2 * Nc * a + 1), \
            (2 * Nc * b + 1), (2 * Nc * b), +, -, mult_U_dag)
#define A_PLUS_BI_DAG(a, b) \
  A_PHASE_B((2 * Nc * a), (2 * Nc * a + 1), \
            (2 * Nc * b + 1), (2 * Nc * b), -, +, mult_U_dag)

// Cross-site (dispatcher-level) prefetch toggles used by H().
// PF_NEXT_OUTPUT: prefetch the next result site's output (default on).
// PF_NEXT_INPUT_X: also prefetch the next site's +x input fermion + dir-0 gauge
// link (the one neighbor that is cold at every kernel entry).
#ifndef PF_NEXT_OUTPUT
  #define PF_NEXT_OUTPUT 1
#endif
#ifndef PF_NEXT_INPUT_X
  #define PF_NEXT_INPUT_X 0
#endif

#if PREFETCH
  #define prefetch_ferm(ferm_loc) \
    for (int i = 0; i < Nc * Ns * num_rhs_vec * 2; i ++) { \
      long ferm_index = ferm_loc * Nc * Ns * 2; \
      REAL_VEC * ferm_vec = ((REAL_VEC*) (ferm + ferm_index * num_rhs)); \
      __builtin_prefetch(ferm_vec + i, 0, 3); \
    }

  #define prefetch_gauge(gauge_loc, dir) \
    for (int i = 0; i < Nc * Nc * 2; i += 8) { \
      int U_index = (gauge_loc * 4 + dir) * Nc * Nc * 2; \
      __builtin_prefetch(U + U_index + i, 0, 3); \
    }
#else
  #define prefetch_ferm(ferm_loc)
  #define prefetch_gauge(gauge_loc, dir)
#endif

/*
 * Apply the hopping operator H given in equation (5.56)
 * of p. 114 in Gattringer and Lang
 * Also multiply the term added by kappa (the hopping parameter)
 * Dimensions of ferm: Lt * Lz * Ly * Lxh * Ns * Nc * 2 * num_rhs
 * Dimensions of U:    Lt * Lz * Ly * Lx * 4 * Nc * Nc * 2
 * Note that ferm is half the sites (one parity), but U is everything
 * U is not ordered by parity (full lex ordering)
 * To match chroma/ILDG conventions, t should run slowest
 * (e.g. we should store the entire xyz-plane before changing to a new t)
 */

// --- Fine-stencil ghost scratch (decomposed MPI grids only) ----------------
// Off-rank z/t neighbors are not stored with the fermion fields: before each
// stencil sweep the input field's faces are exchanged into this shared scratch
// (one block per input parity, [z-lo | z-hi | t-lo | t-hi] G_OFF_* layout) and
// the kernels read boundary neighbors from it via the per-direction ferm_in
// base.  Keeps every field allocation and Krylov-basis stride untouched.
// Per-precision automatically via the TU twinning.
#if FERM_GHOSTS
#define GHOST_BLOCK_REALS ((long) GHOST_SITES * Nc * Ns * 2 * num_rhs)
// Heap, not static: with both dims decomposed at production volume this scratch
// is GBs per TU twin (2 * GHOST_SITES * 768 * num_rhs bytes at REAL=double),
// and as .bss it overflows the small-code-model 2 GB relocation window at link
// time.  Allocated zeroed on first use through the writer entry points below
// (every use sequence starts with a halo fill, a seam install, or an explicit
// zero), matching the zero-initialized start state kernels saw as a static.
static REAL * wk_fghost = NULL;
static REAL * wk_fghost_buf(void) {
  if (!wk_fghost) {
    wk_fghost = malloc(2 * GHOST_BLOCK_REALS * sizeof(REAL));
    memset(wk_fghost, 0, 2 * GHOST_BLOCK_REALS * sizeof(REAL));
  }
  return wk_fghost;
}
#endif

#if FERM_GHOSTS
// Disarm the SAP compact-ghost read path (defined with the compact-halo
// machinery below): after any full-ghost write, wk_fghost is authoritative.
static void wilson_saph_disarm(void);
#endif

// Exchange one parity-half's z/t faces into its ghost scratch block
// (input_parity selects the block).  No-op when nothing is decomposed, so
// callers need not gate.  Single thread, outside parallel regions (funneled).
// The D-family wrappers below call it internally; only code driving
// wilson_kernel* sweeps directly (richardson's inline eo sweeps, the SAP
// boundary update, setup's per-site applies) needs explicit calls.
void wilson_halo_half(const REAL * half, int input_parity) {
#if FERM_GHOSTS
  halo_ferm_half(half,
                 wk_fghost_buf() + (long) input_parity * GHOST_BLOCK_REALS,
                 (long) Nc * Ns * 2 * num_rhs * sizeof(REAL));
  wilson_saph_disarm();   // full rows are authoritative again
#else
  (void) half; (void) input_parity;
#endif
}

// For setup's cross-rank coarse-op build: install a locally assembled seam
// block straight into the fermion ghost scratch -- no MPI.  build_coarse_op's
// cross-rank pass reconstructs the prolonged probe P*e_q on the off-rank seam
// layer (from the halo'd null-vector boundary) and hands it here so
// wilson_kernel_dir_c reads it as the cross-face neighbor, exactly as a
// solve-time halo would.  `seam_block` is GHOST_SITES sites in the G_OFF_*
// face layout, per-site Nc*Ns*2*num_rhs REALs (== one wk_fghost parity block).
void wilson_set_fghost(const REAL * seam_block, int input_parity) {
#if FERM_GHOSTS
  memcpy(wk_fghost_buf() + (long) input_parity * GHOST_BLOCK_REALS, seam_block,
         GHOST_BLOCK_REALS * sizeof(REAL));
  wilson_saph_disarm();
#else
  (void) seam_block; (void) input_parity;
#endif
}

// Zero both parity blocks of the fermion ghost scratch.  build_coarse_op calls
// this so the windowed local pass reads zeros at rank-face shells (matching
// the single-rank "shell is off-block zero"), and again to leave it clean
// afterward.
void wilson_zero_fghost(void) {
#if FERM_GHOSTS
  memset(wk_fghost_buf(), 0, 2 * GHOST_BLOCK_REALS * sizeof(REAL));
  wilson_saph_disarm();
#endif
}

// --- fine-SAP compact halo (color-halved + spin-projected, bit-exact) --------
// red_black_schwarz's cross-rank reads are special in two ways a generic halo
// cannot exploit:
//   (1) Color-halving.  A half-sweep visits only blocks of one color c, and a
//       visited block's cross-cut neighbors are all color 1-c (global
//       red-black block coloring + nearest-neighbor stencil + SAP-block-
//       aligned rank origins).  So of each face slab only the rows lying in
//       color-(1-c) blocks can be read before the next exchange refreshes the
//       ghost -- the same-color half is dead freight, and its ghost rows may
//       legally hold stale data.
//   (2) Spin projection.  A ghost face is read by exactly one hop direction
//       (the one crossing that cut), whose first act is the projection
//       temp = f[a] OP f[b] pairing spin rows (0,2) and (1,3) -- see
//       A_OP_B / A_MINUS_BI / A_PLUS_BI in wilson_kernel_body.inc.  Shipping
//       u = (f[a] OP f[b]) / 2 (6 of 12 complex) and expanding on receive so
//       that the untouched kernel's own add computes u/2 + u/2 recovers the
//       direct result bit for bit: the pack's add is the same IEEE add the
//       kernel would have done, * 0.5 is exact (barring subnormals, absent in
//       this data), x + x and x - (-x) are always exact, and the signed re/im
//       permutation is sign-exact including -0 (which the tail's negated
//       fields do contain).
// Together: exactly 4x fewer exchanged bytes.  The pack pass is threaded and
// runs inside the otherwise-idle blocking window; halo_ferm_half (the
// full-operator halos, which read every site of every face) is untouched.
//
// The receive side is not bulk-expanded into wk_fghost: the compact recv
// sections themselves are the read source, and the kernel dispatchers/preambles
// expand a ghost row on demand into a per-call L1 stack tile (saph_ghost_tile)
// and read the hop from it.  Each ghost row is read by exactly one adjacent
// site per windowed pass, so the expansion work is the same either way --
// but on demand it rides inside the parallel compute instead of a serial
// window, and the expanded form never touches DRAM.  Routing: g_saph_active
// arms the compact path at the compact exchange and any full-ghost writer
// (wilson_halo_half / wilson_set_fghost / wilson_zero_fghost) disarms it,
// so non-SAP consumers always read wk_fghost exactly as before.
#if FERM_GHOSTS

// Ghost-face read patterns, indexed in the G_OFF_* face order {z-lo, z-hi,
// t-lo, t-hi}: the projection the single hop direction reading that face
// applies to a neighbor site, per spin pair h = 0:(a=0,b=2), 1:(a=1,b=3):
//   temp_re = f[a].re + s1[h] * (swap ? f[b].im : f[b].re)
//   temp_im = f[a].im + s2[h] * (swap ? f[b].re : f[b].im)
// (adding s*x is bit-identical to the kernel's +/- x).  z-lo is read by -z
// hops (A_PLUS_BI_DAG(0,2), A_MINUS_BI_DAG(1,3)), z-hi by +z (A_MINUS_BI,
// A_PLUS_BI), t-lo by -t (A_OP_B_DAG(.,.,+)), t-hi by +t (A_OP_B(.,.,-));
// the DAG variants differ only in the gauge factor, not the projection.
// A face slab lands in the opposite ghost (my z-hi face is the +z neighbor's
// z-lo ghost), so packing face f uses pattern f^1; expanding ghost f uses
// pattern f.
typedef struct { int swap; REAL s1[2], s2[2]; } saph_proj_t;
static const saph_proj_t saph_proj[4] = {
  { 1, { -1,  1 }, {  1, -1 } },   // z-lo ghost: -z hop
  { 1, {  1, -1 }, { -1,  1 } },   // z-hi ghost: +z hop
  { 0, {  1,  1 }, {  1,  1 } },   // t-lo ghost: -t hop
  { 0, { -1, -1 }, { -1, -1 } },   // t-hi ghost: +t hop
};

// Live-row tables: per face, the face-row indices (in ghost enumeration
// order -- z faces [t][y][xh], t faces [z][y][xh]) whose rank-face-adjacent
// block has color k.  A reader-color-c exchange sends the tab[1-c] rows of
// each of its own faces and expands the tab[c] rows of each ghost; across a
// cut those are the same physical block rows in the same face order, so the
// counts match row for row.  The adjacent block is parity-independent (full
// x = 2*xh and 2*xh + 1 share a block for even block size), so one table
// serves both parity halves.
enum { SAPH_ZLO = 0, SAPH_ZHI = 1, SAPH_TLO = 2, SAPH_THI = 3 };
static int  * saph_tab[4][2] = {{0}};
static long   saph_n[4][2] = {{0}};
// row -> dense position within color k's live section (-1 = dead row); the
// compact-ghost kernel reads index the compact recv sections through this
static int  * saph_pos[4][2] = {{0}};
static REAL * saph_send[4] = {0}, * saph_recv[4] = {0};
static int    saph_ready = 0;

// Compact-ghost kernel-read state: g_saph_active = the current ghost content
// is a SAP compact exchange (read via saph_ghost_tile); cleared by every
// full-ghost writer.  saph_cur_* give, per face and input parity, the compact
// section of the last exchange that carried that parity and its row->pos
// table -- an unexchanged parity legally keeps its prior state (its live rows
// are exactly the ones the next reader color can reach, unchanged since they
// were exchanged).
static int          g_saph_active = 0;
static const REAL * saph_cur_u[4][2];
static const int  * saph_cur_pos[4][2];
static void wilson_saph_disarm(void) { g_saph_active = 0; }

// compact payload per site: the two projected spin rows, laid out like the
// first half of a full site: [h][color][re/im][rhs]
#define SAPH_CSITE ((long) Nc * Ns * num_rhs)

static void saph_init(void) {
  if (saph_ready) return;
  const int bs = SCHWARZ_BLOCK_SIZE;
  // rank contribution to the global block color -- same rule as
  // red_black_schwarz (rank origins are SAP-block-aligned, comm_init checks)
  const int rbc = ((qc_grid.origin[2] / bs) + (qc_grid.origin[3] / bs)) & 0x1;
  if (DECOMP_Z)
    for (int f = 0; f < 2; f ++) {
      saph_tab[f][0] = malloc((long) GHOST_FACE_Z * sizeof(int));
      saph_tab[f][1] = malloc((long) GHOST_FACE_Z * sizeof(int));
      saph_pos[f][0] = malloc((long) GHOST_FACE_Z * sizeof(int));
      saph_pos[f][1] = malloc((long) GHOST_FACE_Z * sizeof(int));
      const int zblk = (f == 0) ? 0 : (Lz - 1) / bs;
      for (int t = 0; t < Lt; t ++)
        for (int y = 0; y < Ly; y ++)
          for (int x = 0; x < Lxh; x ++) {
            int k = (((2 * x / bs) ^ (y / bs) ^ zblk ^ (t / bs)) & 0x1) ^ rbc;
            int row = (t * Ly + y) * Lxh + x;
            saph_pos[f][k][row] = (int) saph_n[f][k];
            saph_pos[f][1 - k][row] = -1;
            saph_tab[f][k][saph_n[f][k] ++] = row;
          }
    }
  if (DECOMP_T)
    for (int f = 2; f < 4; f ++) {
      saph_tab[f][0] = malloc((long) GHOST_FACE_T * sizeof(int));
      saph_tab[f][1] = malloc((long) GHOST_FACE_T * sizeof(int));
      saph_pos[f][0] = malloc((long) GHOST_FACE_T * sizeof(int));
      saph_pos[f][1] = malloc((long) GHOST_FACE_T * sizeof(int));
      const int tblk = (f == 2) ? 0 : (Lt - 1) / bs;
      for (int z = 0; z < Lz; z ++)
        for (int y = 0; y < Ly; y ++)
          for (int x = 0; x < Lxh; x ++) {
            int k = (((2 * x / bs) ^ (y / bs) ^ (z / bs) ^ tblk) & 0x1) ^ rbc;
            int row = (z * Ly + y) * Lxh + x;
            saph_pos[f][k][row] = (int) saph_n[f][k];
            saph_pos[f][1 - k][row] = -1;
            saph_tab[f][k][saph_n[f][k] ++] = row;
          }
    }
  for (int f = 0; f < 4; f ++) {
    long nmax = saph_n[f][0] > saph_n[f][1] ? saph_n[f][0] : saph_n[f][1];
    if (nmax > 0) {
      saph_send[f] = malloc(sizeof(REAL) * 2 * nmax * SAPH_CSITE);
      saph_recv[f] = malloc(sizeof(REAL) * 2 * nmax * SAPH_CSITE);
    }
  }
  saph_ready = 1;
}

// Pack the live rows of face f from one parity half into dst0 (called inside
// a parallel region; orphaned worksharing loop).  sc = live send color.
static void saph_pack_face(int f, const REAL * half, REAL * dst0, int sc) {
  const int  * tab  = saph_tab[f][sc];
  const long   n    = saph_n[f][sc];
  const long   fss  = (long) Nc * Ns * 2 * num_rhs;
  const long   spin = (long) Nc * 2 * num_rhs;   // one spin row, in REALs
  const long   fchunk = (long) Ly * Lxh;
  const saph_proj_t * pr = &saph_proj[f ^ 1];
  #pragma omp for nowait
  for (long i = 0; i < n; i ++) {
    const long row = tab[i];
    // decode the face-row enumeration ((t or z)*Ly + y)*Lxh + x back to
    // coordinates; the interior site comes from the funnel
    const int rx = (int)(row % Lxh), ry = (int)(row / Lxh % Ly);
    const int rw = (int)(row / fchunk);
    const long site = (f < 2)
        ? site_half(rx, ry, f == 0 ? 0 : Lz - 1, rw)
        : site_half(rx, ry, rw, f == 2 ? 0 : Lt - 1);
    const REAL * src = half + site * fss;
    REAL * dst = dst0 + i * SAPH_CSITE;
    for (int h = 0; h < 2; h ++) {
      const REAL * a = src + h * spin, * b = src + (h + 2) * spin;
      REAL * u = dst + h * spin;
      for (int c = 0; c < Nc; c ++) {
        const REAL * are = a + c * 2 * num_rhs, * aim = are + num_rhs;
        const REAL * bre = b + c * 2 * num_rhs, * bim = bre + num_rhs;
        const REAL * bx = pr->swap ? bim : bre, * by = pr->swap ? bre : bim;
        REAL * ure = u + c * 2 * num_rhs, * uim = ure + num_rhs;
        for (int r = 0; r < num_rhs; r ++) {
          ure[r] = (are[r] + pr->s1[h] * bx[r]) * (REAL) 0.5;
          uim[r] = (aim[r] + pr->s2[h] * by[r]) * (REAL) 0.5;
        }
      }
    }
  }
}

// Expand one compact ghost row into a full-format site (dst): spin a gets u,
// spin b the signed permutation that makes the kernel's own projection add
// compute u/2 + u/2 = u, so the ghost content the kernel sees is bit-identical
// to a directly exchanged full-format face.
static inline void saph_expand_site(REAL * dst, const REAL * u, int f) {
  const long spin = (long) Nc * 2 * num_rhs;
  const saph_proj_t * pr = &saph_proj[f];
  for (int h = 0; h < 2; h ++) {
    REAL * ga = dst + h * spin, * gb = dst + (h + 2) * spin;
    const REAL * uh = u + h * spin;
    for (int c = 0; c < Nc; c ++) {
      const REAL * ure = uh + c * 2 * num_rhs, * uim = ure + num_rhs;
      REAL * gare = ga + c * 2 * num_rhs, * gaim = gare + num_rhs;
      REAL * gbre = gb + c * 2 * num_rhs, * gbim = gbre + num_rhs;
      REAL * gbx = pr->swap ? gbim : gbre, * gby = pr->swap ? gbre : gbim;
      for (int r = 0; r < num_rhs; r ++) {
        gare[r] = ure[r];
        gaim[r] = uim[r];
        gbx[r]  = pr->s1[h] * ure[r];
        gby[r]  = pr->s2[h] * uim[r];
      }
    }
  }
}

// Ghost read for the kernel dispatchers/preambles: expand face f's
// compact row (ghost enumeration index `row`, input parity ip) into the
// caller's stack tile and return it.  Call only when g_saph_active.  The
// assert is the color-halving liveness invariant: a visited block can only
// reach rows adjacent to it, which are live by construction.
static inline REAL * saph_ghost_tile(REAL * tile, int f, int ip, long row) {
  const int pos = saph_cur_pos[f][ip][row];
  assert(pos >= 0);
  saph_expand_site(tile, saph_cur_u[f][ip] + (long) pos * SAPH_CSITE, f);
  return tile;
}

// Serial-window decomposition of the compact exchange, accumulated across
// calls; reset via wilson_halo_sap_stats_reset (chained from
// sap_stats_reset).  Gates the fused-emit follow-up: only if the pack span is
// a real fraction of the window is emitting at block-write time (cache-hot)
// worth its invasiveness.  The unpack accumulator stays 0: the expansion rides
// inside the compute passes, off the serial window.
static double g_saph_pack_secs = 0, g_saph_mpi_secs = 0, g_saph_unpack_secs = 0;
static double saph_dt(struct timeval a, struct timeval b) {
  return (double) (b.tv_sec - a.tv_sec)
          + 1e-6 * (double) (b.tv_usec - a.tv_usec);
}
#endif  // FERM_GHOSTS

void wilson_halo_sap_stats_get(double * pack, double * mpi, double * unpack) {
#if FERM_GHOSTS
  *pack = g_saph_pack_secs;
  *mpi = g_saph_mpi_secs;
  *unpack = g_saph_unpack_secs;
#else
  *pack = *mpi = *unpack = 0;
#endif
}
void wilson_halo_sap_stats_reset(void) {
#if FERM_GHOSTS
  g_saph_pack_secs = g_saph_mpi_secs = g_saph_unpack_secs = 0;
#endif
}

// SAP compact exchange of one or both parity halves (NULL = skip that half)
// for the half-sweep whose visited blocks have color reader_color.  Drop-in
// for the wilson_halo_half pair at red_black_schwarz's exchange points; the
// ghost rows it does not refresh are exactly the rows no color-reader_color
// block can read.  Single thread, outside parallel regions (funneled).
void wilson_halo_sap(const REAL * even, const REAL * odd, int reader_color) {
#if FERM_GHOSTS
  saph_init();
  const int sc = reader_color ^ 1;   // send rows: color-(1-c) blocks
  const int rc = reader_color;       // ghost rows: adjacent to color-c blocks
  const REAL * halves[2] = { even, odd };
  struct timeval t0, t1, t2;
  gettimeofday(&t0, NULL);
  #pragma omp parallel
  for (int f = 0; f < 4; f ++) {
    if (!(f < 2 ? DECOMP_Z : DECOMP_T)) continue;
    long sec = 0;
    for (int p = 0; p < 2; p ++)
      if (halves[p])
        saph_pack_face(f, halves[p],
                       saph_send[f] + (sec ++) * saph_n[f][sc] * SAPH_CSITE,
                       sc);
  }
  gettimeofday(&t1, NULL);
  const long npar = (even != NULL) + (odd != NULL);
  const void * sb[4]; void * rb[4]; long sbytes[4], rbytes[4];
  for (int f = 0; f < 4; f ++) {
    sb[f] = saph_send[f]; rb[f] = saph_recv[f];
    sbytes[f] = npar * saph_n[f][sc] * SAPH_CSITE * sizeof(REAL);
    rbytes[f] = npar * saph_n[f][rc] * SAPH_CSITE * sizeof(REAL);
  }
  halo_ferm_sap(sb, rb, sbytes, rbytes);
  gettimeofday(&t2, NULL);
  // No bulk expand: install the kernel-read state for each exchanged parity,
  // and the boundary kernels expand rows on demand from the compact recv
  // sections (saph_ghost_tile) during the compute passes.
  {
    long sec = 0;
    for (int p = 0; p < 2; p ++) {
      if (!halves[p]) continue;
      for (int f = 0; f < 4; f ++) {
        if (!(f < 2 ? DECOMP_Z : DECOMP_T)) continue;
        saph_cur_u[f][p]   = saph_recv[f] + sec * saph_n[f][rc] * SAPH_CSITE;
        saph_cur_pos[f][p] = saph_pos[f][rc];
      }
      sec ++;
    }
    g_saph_active = 1;
  }
  g_saph_pack_secs   += saph_dt(t0, t1);
  g_saph_mpi_secs    += saph_dt(t1, t2);
#else
  (void) even; (void) odd; (void) reader_color;
#endif
}

// Pure-C reference implementation of the Wilson hopping kernel.
// Always compiled; used directly when ASM=0 and as the correctness
// oracle for the hand-written assembly kernels.
void wilson_kernel_c(REAL * output, REAL * ferm, REAL * U,
                   int result_x, int result_y, int result_z, int result_t,
                   int parity) {
#define WK_ACTIVE(n) 1            // all 8 directions; constant-folds, no overhead
#define WK_GHOST_IN wk_fghost
#define WK_SAPH 1                 // may read the SAP compact ghost when armed
#include "wilson_kernel_body.inc"
#undef WK_SAPH
#undef WK_GHOST_IN
#undef WK_ACTIVE
}

// Masked single-/multi-direction variant of wilson_kernel_c (double input):
// the same shared body, but only the directions whose mask bit is set are
// summed (bit order 0:+x 1:+y 2:+z 3:+t 4:-x 5:-y 6:-z 7:-t).  Accumulates
// into output (caller zeros it) exactly like wilson_kernel_c, so for a mask
// reaching only a region where the other directions read zeros it is
// bit-identical to the full kernel (x + 0 == x).  Used by
// build_coarse_op_direct to take each coarse-block face hop alone instead of
// the full stencil.  C only -- there is no asm twin.
void wilson_kernel_dir_c(REAL * output, REAL * ferm, REAL * U,
                   int result_x, int result_y, int result_z, int result_t,
                   int parity, int wk_mask) {
#define WK_ACTIVE(n) (wk_mask & (1 << (n)))
#define WK_GHOST_IN wk_fghost
#define WK_SAPH 1   // its callers always run disarmed (seam installs / zero
                    // fills clear the flag), so this is the wk_fghost path;
                    // kept consistent with wilson_kernel_c
#include "wilson_kernel_body.inc"
#undef WK_SAPH
#undef WK_GHOST_IN
#undef WK_ACTIVE
}

// Dispatcher: applies the Wilson hopping kernel for one result site.
// When ASM=1 it computes the neighbor pointers and calls the hand-written
// vector kernel; otherwise it falls back to the pure-C reference.
void wilson_kernel(REAL * output, REAL * ferm, REAL * U,
                   int result_x, int result_y, int result_z, int result_t,
                   int parity) {
#if WILSON_ASM_LIVE
  long result_loc = site_half(result_x, result_y, result_z, result_t);
  int x_par = (parity ^ result_y ^ result_z ^ result_t) & 0x1;
  long plus_x = site_half((result_x + x_par) % Lxh,
                          result_y, result_z, result_t);
  long plus_y = site_half(result_x, (result_y + 1) % Ly, result_z, result_t);
  long minus_x = site_half((result_x + Lxh - 1 + x_par) % Lxh,
                           result_y, result_z, result_t);
  long minus_y = site_half(result_x, (result_y + Ly - 1) % Ly,
                           result_z, result_t);

  long ferm_site_size = Nc * Ns * 2 * num_rhs;
  long gauge_site_size = Nc * Nc * 2 * 4;
  int full_x = result_x * 2 + x_par;
  long gauge_loc = site_full(full_x, result_y, result_z, result_t);
  long gauge_minus_x = site_full((full_x + Lx - 1) % Lx,
                                 result_y, result_z, result_t);
  long gauge_minus_y = site_full(full_x, (result_y + Ly - 1) % Ly,
                                 result_z, result_t);

  // z/t neighbors as pointers: across a rank face they sit in the ghost
  // scratch (fermions) / U's tail ghosts (gauge), not in the field -- the asm
  // kernel takes explicit neighbor pointers, so it needs no changes.  At an
  // undecomposed grid these are exactly the original wrapped offsets.
#if FERM_GHOSTS
  REAL * const ghost_in = wk_fghost + (long) (1 - parity) * GHOST_BLOCK_REALS;
#endif
#if DECOMP_Z
  long ghz = ((long) result_t * Ly + result_y) * Lxh + result_x;
  // compact ghost (g_saph_active): expand the row into an L1 stack
  // tile and hand the asm kernel the tile pointer -- no asm changes
  REAL wk_ztile[(long) Nc * Ns * 2 * num_rhs] __attribute__((aligned(64)));
  REAL * f_pz, * f_mz;
  if (result_z + 1 < Lz)
    f_pz = ferm + site_half(result_x, result_y, result_z + 1, result_t)
                  * ferm_site_size;
  else if (g_saph_active)
    f_pz = saph_ghost_tile(wk_ztile, SAPH_ZHI, 1 - parity, ghz);
  else
    f_pz = ghost_in + (G_OFF_ZHI + ghz) * ferm_site_size;
  if (result_z > 0)
    f_mz = ferm + site_half(result_x, result_y, result_z - 1, result_t)
                  * ferm_site_size;
  else if (g_saph_active)
    f_mz = saph_ghost_tile(wk_ztile, SAPH_ZLO, 1 - parity, ghz);
  else
    f_mz = ghost_in + (G_OFF_ZLO + ghz) * ferm_site_size;
  long gauge_minus_z = result_z > 0
      ? site_full(full_x, result_y, result_z - 1, result_t)
      : V + GG_OFF_ZLO + ((long) result_t * Ly + result_y) * Lx + full_x;
#else
  REAL * f_pz = ferm + site_half(result_x, result_y, (result_z + 1) % Lz,
                                 result_t) * ferm_site_size;
  REAL * f_mz = ferm + site_half(result_x, result_y, (result_z + Lz - 1) % Lz,
                                 result_t) * ferm_site_size;
  long gauge_minus_z = site_full(full_x, result_y, (result_z + Lz - 1) % Lz,
                                 result_t);
#endif
#if DECOMP_T
  long ght = ((long) result_z * Ly + result_y) * Lxh + result_x;
  REAL wk_ttile[(long) Nc * Ns * 2 * num_rhs] __attribute__((aligned(64)));
  REAL * f_pt, * f_mt;
  if (result_t + 1 < Lt)
    f_pt = ferm + site_half(result_x, result_y, result_z, result_t + 1)
                  * ferm_site_size;
  else if (g_saph_active)
    f_pt = saph_ghost_tile(wk_ttile, SAPH_THI, 1 - parity, ght);
  else
    f_pt = ghost_in + (G_OFF_THI + ght) * ferm_site_size;
  if (result_t > 0)
    f_mt = ferm + site_half(result_x, result_y, result_z, result_t - 1)
                  * ferm_site_size;
  else if (g_saph_active)
    f_mt = saph_ghost_tile(wk_ttile, SAPH_TLO, 1 - parity, ght);
  else
    f_mt = ghost_in + (G_OFF_TLO + ght) * ferm_site_size;
  long gauge_minus_t = result_t > 0
      ? site_full(full_x, result_y, result_z, result_t - 1)
      : V + GG_OFF_TLO + ((long) result_z * Ly + result_y) * Lx + full_x;
#else
  REAL * f_pt = ferm + site_half(result_x, result_y, result_z,
                                 (result_t + 1) % Lt) * ferm_site_size;
  REAL * f_mt = ferm + site_half(result_x, result_y, result_z,
                                 (result_t + Lt - 1) % Lt) * ferm_site_size;
  long gauge_minus_t = site_full(full_x, result_y, result_z,
                                 (result_t + Lt - 1) % Lt);
#endif

  // The asm kernel processes WILSON_ASM_TILE columns per call: the generated
  // AVX2 kernel takes all num_rhs at once (one tile, loops over rhs half-blocks
  // internally); the AVX-512 kernel takes 16, so it is invoked num_rhs/16
  // times with the fermion/output pointers stepped by 16 columns.  Gauge
  // links are rhs-independent, so they are passed unshifted on every tile.
  for (int tile = 0; tile < num_rhs / WILSON_ASM_TILE; tile ++) {
    long co = (long) tile * WILSON_ASM_TILE;   // column offset (in REALs)
    wilson_kernel_asm(output + result_loc * ferm_site_size + co,
                      ferm + plus_x * ferm_site_size + co,
                      ferm + plus_y * ferm_site_size + co,
                      f_pz + co,
                      f_pt + co,
                      ferm + minus_x * ferm_site_size + co,
                      ferm + minus_y * ferm_site_size + co,
                      f_mz + co,
                      f_mt + co,
                      U + gauge_loc * gauge_site_size,
                      U + gauge_minus_x * gauge_site_size,
                      U + gauge_minus_y * gauge_site_size,
                      U + gauge_minus_z * gauge_site_size,
                      U + gauge_minus_t * gauge_site_size);
  }
#else
  wilson_kernel_c(output, ferm, U, result_x, result_y, result_z, result_t,
                  parity);
#endif
}

// Wilson kernel for Schwarz blocks.  When an asm kernel is available we
// dispatch to wilson_kernel_masked_asm (the AVX2 or AVX-512 masked kernel,
// picked by the build flags): it does the same per-direction arithmetic as
// the full-lattice kernel but takes an 8-bit mask selecting which
// of the 8 directions to perform, so each excluded direction (the Dirichlet
// block face for the block solve, update_boundary=0; the interior couplings
// for the boundary residual update, update_boundary=1) costs only a
// (well-predicted) branch and no flops -- no wasted FMAs and no per-direction
// function calls.  The kernel is sequential per direction with a
// memory-resident accumulator, so skipping a direction is just a jump.
// Mask bit order matches the asm DIRS:
//   bit 0:+x 1:+y 2:+z 3:+t 4:-x 5:-y 6:-z 7:-t.
// See wilson_kernel_block_c for the reference C implementation and the
// boundary semantics.
void wilson_kernel_block(REAL * output, REAL * ferm, REAL * U,
                         int result_x, int result_y, int result_z,
                         int result_t, int parity, int schwarz_block_size,
                         int update_boundary) {
#if WILSON_ASM_LIVE
  long result_loc = site_half(result_x, result_y, result_z, result_t);
  int x_par = (parity ^ result_y ^ result_z ^ result_t) & 0x1;
  long plus_x  = site_half((result_x + x_par) % Lxh,
                           result_y, result_z, result_t);
  long plus_y  = site_half(result_x, (result_y + 1) % Ly, result_z, result_t);
  long minus_x = site_half((result_x + Lxh - 1 + x_par) % Lxh,
                           result_y, result_z, result_t);
  long minus_y = site_half(result_x, (result_y + Ly - 1) % Ly,
                           result_z, result_t);
  int full_x = result_x * 2 + x_par;
  long gauge_loc = site_full(full_x, result_y, result_z, result_t);
  long gauge_minus_x = site_full((full_x + Lx - 1) % Lx,
                                 result_y, result_z, result_t);
  long gauge_minus_y = site_full(full_x, (result_y + Ly - 1) % Ly,
                                 result_z, result_t);
  long ferm_site_size_b = Nc * Ns * 2 * num_rhs;
  // z/t neighbors as pointers (ghost scratch / gauge tail across a rank face;
  // the interior mask never reaches them, the boundary-update mask -- the SAP
  // cross-block residual refresh -- does).  Identical wrapped offsets at an
  // undecomposed grid.
#if FERM_GHOSTS
  REAL * const ghost_in = wk_fghost + (long) (1 - parity) * GHOST_BLOCK_REALS;
#endif
#if DECOMP_Z
  long ghz_b = ((long) result_t * Ly + result_y) * Lxh + result_x;
  // Compact ghost: on-demand L1 tile, exactly as in wilson_kernel.
  // Only the boundary-update mask ever reaches these pointers.
  REAL wk_ztile[(long) Nc * Ns * 2 * num_rhs] __attribute__((aligned(64)));
  REAL * f_pz, * f_mz;
  if (result_z + 1 < Lz)
    f_pz = ferm + site_half(result_x, result_y, result_z + 1, result_t)
                  * ferm_site_size_b;
  else if (g_saph_active)
    f_pz = saph_ghost_tile(wk_ztile, SAPH_ZHI, 1 - parity, ghz_b);
  else
    f_pz = ghost_in + (G_OFF_ZHI + ghz_b) * ferm_site_size_b;
  if (result_z > 0)
    f_mz = ferm + site_half(result_x, result_y, result_z - 1, result_t)
                  * ferm_site_size_b;
  else if (g_saph_active)
    f_mz = saph_ghost_tile(wk_ztile, SAPH_ZLO, 1 - parity, ghz_b);
  else
    f_mz = ghost_in + (G_OFF_ZLO + ghz_b) * ferm_site_size_b;
  long gauge_minus_z = result_z > 0
      ? site_full(full_x, result_y, result_z - 1, result_t)
      : V + GG_OFF_ZLO + ((long) result_t * Ly + result_y) * Lx + full_x;
#else
  REAL * f_pz = ferm + site_half(result_x, result_y, (result_z + 1) % Lz,
                                 result_t) * ferm_site_size_b;
  REAL * f_mz = ferm + site_half(result_x, result_y, (result_z + Lz - 1) % Lz,
                                 result_t) * ferm_site_size_b;
  long gauge_minus_z = site_full(full_x, result_y, (result_z + Lz - 1) % Lz,
                                 result_t);
#endif
#if DECOMP_T
  long ght_b = ((long) result_z * Ly + result_y) * Lxh + result_x;
  REAL wk_ttile[(long) Nc * Ns * 2 * num_rhs] __attribute__((aligned(64)));
  REAL * f_pt, * f_mt;
  if (result_t + 1 < Lt)
    f_pt = ferm + site_half(result_x, result_y, result_z, result_t + 1)
                  * ferm_site_size_b;
  else if (g_saph_active)
    f_pt = saph_ghost_tile(wk_ttile, SAPH_THI, 1 - parity, ght_b);
  else
    f_pt = ghost_in + (G_OFF_THI + ght_b) * ferm_site_size_b;
  if (result_t > 0)
    f_mt = ferm + site_half(result_x, result_y, result_z, result_t - 1)
                  * ferm_site_size_b;
  else if (g_saph_active)
    f_mt = saph_ghost_tile(wk_ttile, SAPH_TLO, 1 - parity, ght_b);
  else
    f_mt = ghost_in + (G_OFF_TLO + ght_b) * ferm_site_size_b;
  long gauge_minus_t = result_t > 0
      ? site_full(full_x, result_y, result_z, result_t - 1)
      : V + GG_OFF_TLO + ((long) result_z * Ly + result_y) * Lx + full_x;
#else
  REAL * f_pt = ferm + site_half(result_x, result_y, result_z,
                                 (result_t + 1) % Lt) * ferm_site_size_b;
  REAL * f_mt = ferm + site_half(result_x, result_y, result_z,
                                 (result_t + Lt - 1) % Lt) * ferm_site_size_b;
  long gauge_minus_t = site_full(full_x, result_y, result_z,
                                 (result_t + Lt - 1) % Lt);
#endif
  int bs = schwarz_block_size;
  // direction active <=> neighbor inside the block (update_boundary=0) or
  // outside it (update_boundary=1) -- the exact wilson_kernel_block_c ifs.
  int mask = ((((full_x   + 1) % bs != 0) ^ update_boundary) << 0)
            | ((((result_y + 1) % bs != 0) ^ update_boundary) << 1)
            | ((((result_z + 1) % bs != 0) ^ update_boundary) << 2)
            | ((((result_t + 1) % bs != 0) ^ update_boundary) << 3)
            | (((full_x   % bs != 0) ^ update_boundary) << 4)
            | (((result_y % bs != 0) ^ update_boundary) << 5)
            | (((result_z % bs != 0) ^ update_boundary) << 6)
            | (((result_t % bs != 0) ^ update_boundary) << 7);
  long ferm_site_size = ferm_site_size_b;
  long gauge_site_size = Nc * Nc * 2 * 4;
  // Tile over columns exactly as wilson_kernel does (WILSON_ASM_TILE per call);
  // the mask and gauge links are rhs-independent, so they repeat each tile.
  for (int tile = 0; tile < num_rhs / WILSON_ASM_TILE; tile ++) {
    long co = (long) tile * WILSON_ASM_TILE;   // column offset (in REALs)
    wilson_kernel_masked_asm(
        output + result_loc * ferm_site_size + co,
        ferm + plus_x  * ferm_site_size + co,
        ferm + plus_y  * ferm_site_size + co,
        f_pz + co, f_pt + co,
        ferm + minus_x * ferm_site_size + co,
        ferm + minus_y * ferm_site_size + co,
        f_mz + co, f_mt + co,
        U + gauge_loc     * gauge_site_size,
        U + gauge_minus_x * gauge_site_size,
        U + gauge_minus_y * gauge_site_size,
        U + gauge_minus_z * gauge_site_size,
        U + gauge_minus_t * gauge_site_size, mask);
  }
#else
  wilson_kernel_block_c(output, ferm, U, result_x, result_y, result_z,
                        result_t, parity, schwarz_block_size, update_boundary);
#endif
}

// ---- interior-call offset table -------------------------------------------
// The interior masked calls (update_boundary=0: the Schwarz block solves) only
// ever couple sites within one bs^4 block, so their neighbor offsets never
// wrap the lattice and are a pure function of the local block coordinates:
// block origins are bs-aligned (bs even), so global coordinate parity equals
// local parity and every local site has a fixed kernel parity.  One shared
// bs^4-row table therefore replaces the dispatcher's per-site modulo/address
// math for MR step (1)/(2), recover, and the first-half-sweep Schur RHS.
// Offsets are stored relative to the block origin, already scaled to REAL
// units; callers pass the origin's half-lattice / full-lattice site index
// once per block.  Because the offsets are local-only, the table is
// independent of the global lattice and of the MPI decomposition: it is built
// from the local dims, and only the modulo path above ever touches the halo.
//
// Inactive directions (the Dirichlet faces) are never read by the masked asm
// kernel (the mask jump skips the whole direction, loads included), but their
// table slots are clamped to the result site / local gauge link anyway so all
// 14 pointers stay valid no matter what.
#if WILSON_ASM_LIVE
static long * wk_itab = NULL;       // [bs^4][14] offsets in REALs from origin
static int  * wk_itab_mask = NULL;  // [bs^4] interior direction masks
static int    wk_itab_bs = 0;
#endif

// Build (or rebuild) the interior offset table for block size bs.  Idempotent
// and cheap; call once before a Schwarz sweep (not thread-safe -- call outside
// the parallel region).  Runs the exact wilson_kernel_block address math for
// a block at the origin, so active directions get bit-identical pointers.
void wilson_block_itab_init(int bs) {
#if WILSON_ASM_LIVE
  if (wk_itab != NULL && wk_itab_bs == bs) return;
  free(wk_itab); free(wk_itab_mask);
  long nsites = (long) bs * bs * bs * bs;
  wk_itab      = malloc(sizeof(long) * nsites * 14);
  wk_itab_mask = malloc(sizeof(int) * nsites);
  long fss = Nc * Ns * 2 * num_rhs;
  long gss = Nc * Nc * 2 * 4;
  for (int lt = 0; lt < bs; lt ++)
    for (int lz = 0; lz < bs; lz ++)
      for (int ly = 0; ly < bs; ly ++)
        for (int lx = 0; lx < bs; lx ++) {
          int lidx = ((lt * bs + lz) * bs + ly) * bs + lx;
          // the fixed parity of this local site
          // (origins are even in all coords)
          int parity = (lx ^ ly ^ lz ^ lt) & 0x1;
          int result_x = lx / 2, result_y = ly, result_z = lz, result_t = lt;
          // --- the wilson_kernel_block math, block at the origin ---
          long result_loc = site_half(result_x, result_y, result_z, result_t);
          int x_par = (parity ^ result_y ^ result_z ^ result_t) & 0x1;
          long plus_x  = site_half((result_x + x_par) % Lxh,
                                   result_y, result_z, result_t);
          long plus_y  = site_half(result_x, (result_y + 1) % Ly,
                                   result_z, result_t);
          long plus_z  = site_half(result_x, result_y, (result_z + 1) % Lz,
                                   result_t);
          long plus_t  = site_half(result_x, result_y, result_z,
                                   (result_t + 1) % Lt);
          long minus_x = site_half((result_x + Lxh - 1 + x_par) % Lxh,
                                   result_y, result_z, result_t);
          long minus_y = site_half(result_x, (result_y + Ly - 1) % Ly,
                                   result_z, result_t);
          long minus_z = site_half(result_x, result_y,
                                   (result_z + Lz - 1) % Lz, result_t);
          long minus_t = site_half(result_x, result_y, result_z,
                                   (result_t + Lt - 1) % Lt);
          int full_x = result_x * 2 + x_par;
          long gauge_loc = site_full(full_x, result_y, result_z, result_t);
          long gauge_minus_x = site_full((full_x + Lx - 1) % Lx,
                                         result_y, result_z, result_t);
          long gauge_minus_y = site_full(full_x, (result_y + Ly - 1) % Ly,
                                         result_z, result_t);
          long gauge_minus_z = site_full(full_x, result_y,
                                         (result_z + Lz - 1) % Lz, result_t);
          long gauge_minus_t = site_full(full_x, result_y, result_z,
                                         (result_t + Lt - 1) % Lt);
          // interior mask (update_boundary = 0)
          int mask = (((full_x   + 1) % bs != 0) << 0)
                    | (((result_y + 1) % bs != 0) << 1)
                    | (((result_z + 1) % bs != 0) << 2)
                    | (((result_t + 1) % bs != 0) << 3)
                    | ((full_x   % bs != 0) << 4)
                    | ((result_y % bs != 0) << 5)
                    | ((result_z % bs != 0) << 6)
                    | ((result_t % bs != 0) << 7);
          // clamp inactive directions (their wrapped values are meaningless
          // at a general block position; the kernel skips them,
          // but keep pointers valid)
          if (!(mask & 0x01)) plus_x  = result_loc;
          if (!(mask & 0x02)) plus_y  = result_loc;
          if (!(mask & 0x04)) plus_z  = result_loc;
          if (!(mask & 0x08)) plus_t  = result_loc;
          if (!(mask & 0x10)) {
            minus_x = result_loc;
            gauge_minus_x = gauge_loc;
          }
          if (!(mask & 0x20)) {
            minus_y = result_loc;
            gauge_minus_y = gauge_loc;
          }
          if (!(mask & 0x40)) {
            minus_z = result_loc;
            gauge_minus_z = gauge_loc;
          }
          if (!(mask & 0x80)) {
            minus_t = result_loc;
            gauge_minus_t = gauge_loc;
          }
          long * row = wk_itab + (long) lidx * 14;
          row[0]  = result_loc * fss;
          row[1]  = plus_x  * fss;
          row[2]  = plus_y  * fss;
          row[3]  = plus_z  * fss;
          row[4]  = plus_t  * fss;
          row[5]  = minus_x * fss;
          row[6]  = minus_y * fss;
          row[7]  = minus_z * fss;
          row[8]  = minus_t * fss;
          row[9]  = gauge_loc     * gss;
          row[10] = gauge_minus_x * gss;
          row[11] = gauge_minus_y * gss;
          row[12] = gauge_minus_z * gss;
          row[13] = gauge_minus_t * gss;
          wk_itab_mask[lidx] = mask;
        }
  wk_itab_bs = bs;
#else
  (void) bs;
#endif
}

// Interior (update_boundary=0) Schwarz-block Wilson call via the offset table:
// identical pointers and mask to wilson_kernel_block(..., 0), but the per-site
// modulo/address math is replaced by one table-row load.  org_half/org_full
// are the block origin's half-lattice / full-lattice site indices; local_idx
// is the site's local linear index (x fastest), i.e. the SCHWARZ_BLOCK_SITES
// visit order.  The coordinate/parity arguments are only used by the non-ASM
// fallback.  wilson_block_itab_init(bs) must have been called.
void wilson_kernel_block_itab(REAL * output, REAL * ferm, REAL * U,
                              int result_x, int result_y, int result_z,
                              int result_t, int parity, int schwarz_block_size,
                              long org_half, long org_full, int local_idx) {
#if WILSON_ASM_LIVE
  (void) result_x; (void) result_y; (void) result_z; (void) result_t;
  (void) parity; (void) schwarz_block_size;
  const long * row = wk_itab + (long) local_idx * 14;
  int mask = wk_itab_mask[local_idx];
  long fo = org_half * (long) (Nc * Ns * 2 * num_rhs);
  long go = org_full * (long) (Nc * Nc * 2 * 4);
  for (int tile = 0; tile < num_rhs / WILSON_ASM_TILE; tile ++) {
    long co = (long) tile * WILSON_ASM_TILE;   // column offset (in REALs)
    wilson_kernel_masked_asm(
        output + fo + row[0] + co,
        ferm + fo + row[1] + co, ferm + fo + row[2] + co,
        ferm + fo + row[3] + co, ferm + fo + row[4] + co,
        ferm + fo + row[5] + co, ferm + fo + row[6] + co,
        ferm + fo + row[7] + co, ferm + fo + row[8] + co,
        U + go + row[9],  U + go + row[10],
        U + go + row[11], U + go + row[12],
        U + go + row[13], mask);
  }
#else
  (void) org_half; (void) org_full; (void) local_idx;
  wilson_kernel_block_c(output, ferm, U, result_x, result_y, result_z,
                        result_t, parity, schwarz_block_size, 0);
#endif
}

// Reference C implementation of the Schwarz-block Wilson kernel (also the
// non-ASM fallback).  Drops all contributions from outside the block; the
// "block" here is the Schwarz domain (changes the answer, not just speed), not
// the cache-blocking over loops.  update_boundary selects boundary vs interior
// couplings (see wilson_kernel_block).
void wilson_kernel_block_c(REAL * output, REAL * ferm, REAL * U,
                         int result_x, int result_y, int result_z,
                         int result_t, int parity, int schwarz_block_size,
                         int update_boundary) {
  int result_loc = site_half(result_x, result_y, result_z, result_t);
  REAL_VEC temp_re [3], temp_im [3], prod_re, prod_im;
  int ferm_loc, dir;

  // various shifts of result_loc (same ghost-aware z/t selection as the shared
  // kernel body: across a rank face the boundary-update couplings read the
  // ghost scratch / gauge tail; identical wrapped offsets at a 1x1 grid)
  int x_par = (parity ^ result_y ^ result_z ^ result_t) & 0x1;
  int plus_x = site_half((result_x + x_par) % Lxh, result_y, result_z,
                         result_t);
  int plus_y = site_half(result_x, (result_y + 1) % Ly, result_z, result_t);
  int minus_x = site_half((result_x + Lxh - 1 + x_par) % Lxh, result_y,
                          result_z, result_t);
  int minus_y = site_half(result_x, (result_y + Ly - 1) % Ly, result_z,
                          result_t);
#if FERM_GHOSTS
  __typeof__(ferm) const ghost_in = wk_fghost
                                  + (long) (1 - parity) * GHOST_BLOCK_REALS;
  __typeof__(ferm) ferm_in = ferm;
#else
  __typeof__(ferm) const ferm_in = ferm;   // never reassigned
#endif
#if DECOMP_Z
  int ghz = (result_t * Ly + result_y) * Lxh + result_x;
  // compact ghost: on-demand L1 tile, as in the shared kernel body
  REAL wk_ztile[(long) Nc * Ns * 2 * num_rhs] __attribute__((aligned(64)));
  __typeof__(ferm) fb_pz, fb_mz;
  int plus_z, minus_z;
  if (result_z + 1 < Lz) {
    fb_pz = ferm;
    plus_z = site_half(result_x, result_y, result_z + 1, result_t);
  } else if (g_saph_active) {
    fb_pz = saph_ghost_tile(wk_ztile, SAPH_ZHI, 1 - parity, ghz); plus_z = 0;
  } else               { fb_pz = ghost_in; plus_z = G_OFF_ZHI + ghz; }
  if (result_z > 0) {
    fb_mz = ferm;
    minus_z = site_half(result_x, result_y, result_z - 1, result_t);
  } else if (g_saph_active) {
    fb_mz = saph_ghost_tile(wk_ztile, SAPH_ZLO, 1 - parity, ghz); minus_z = 0;
  } else               { fb_mz = ghost_in; minus_z = G_OFF_ZLO + ghz; }
#else
  int plus_z = site_half(result_x, result_y, (result_z + 1) % Lz, result_t);
  int minus_z = site_half(result_x, result_y, (result_z + Lz - 1) % Lz,
                          result_t);
#if FERM_GHOSTS
  // z undecomposed but t is: the z direction blocks must still reset ferm_in
  // (a preceding t hop may have left it on the ghost scratch), so give them
  // local-field aliases.  Mirrors the WK_FB_* convention of the shared body.
  __typeof__(ferm) const fb_pz = ferm, fb_mz = ferm;
#endif
#endif
#if DECOMP_T
  int ght = (result_z * Ly + result_y) * Lxh + result_x;
  REAL wk_ttile[(long) Nc * Ns * 2 * num_rhs] __attribute__((aligned(64)));
  __typeof__(ferm) fb_pt, fb_mt;
  int plus_t, minus_t;
  if (result_t + 1 < Lt) {
    fb_pt = ferm;
    plus_t = site_half(result_x, result_y, result_z, result_t + 1);
  } else if (g_saph_active) {
    fb_pt = saph_ghost_tile(wk_ttile, SAPH_THI, 1 - parity, ght); plus_t = 0;
  } else               { fb_pt = ghost_in; plus_t = G_OFF_THI + ght; }
  if (result_t > 0) {
    fb_mt = ferm;
    minus_t = site_half(result_x, result_y, result_z, result_t - 1);
  } else if (g_saph_active) {
    fb_mt = saph_ghost_tile(wk_ttile, SAPH_TLO, 1 - parity, ght); minus_t = 0;
  } else               { fb_mt = ghost_in; minus_t = G_OFF_TLO + ght; }
#else
  int plus_t = site_half(result_x, result_y, result_z, (result_t + 1) % Lt);
  int minus_t = site_half(result_x, result_y, result_z,
                          (result_t + Lt - 1) % Lt);
#if FERM_GHOSTS
  // t undecomposed but z is: same reset-alias for the t direction blocks
  // (a preceding z hop may have pointed ferm_in at the ghost scratch)
  __typeof__(ferm) const fb_pt = ferm, fb_mt = ferm;
#endif
#endif

  int full_x = result_x * 2 + x_par;
  int gauge_loc = site_full(full_x, result_y, result_z, result_t);
  int gauge_minus_x = site_full((full_x + Lx - 1) % Lx, result_y, result_z,
                                result_t);
  int gauge_minus_y = site_full(full_x, (result_y + Ly - 1) % Ly, result_z,
                                result_t);
#if DECOMP_Z
  int gauge_minus_z = result_z > 0
      ? site_full(full_x, result_y, result_z - 1, result_t)
      : V + GG_OFF_ZLO + (result_t * Ly + result_y) * Lx + full_x;
#else
  int gauge_minus_z = site_full(full_x, result_y, (result_z + Lz - 1) % Lz,
                                result_t);
#endif
#if DECOMP_T
  int gauge_minus_t = result_t > 0
      ? site_full(full_x, result_y, result_z, result_t - 1)
      : V + GG_OFF_TLO + (result_z * Ly + result_y) * Lx + full_x;
#else
  int gauge_minus_t = site_full(full_x, result_y, result_z,
                                (result_t + Lt - 1) % Lt);
#endif

  // positive shifts
  // multiplication is
  // output[c1][d] += U[c1][c2] * ferm[c2][d] * (1 - gamma_mu)

  // x direction
  if (((full_x + 1) % schwarz_block_size != 0) ^ update_boundary) {
    ferm_loc = plus_x;
#if FERM_GHOSTS
    ferm_in = ferm;
#endif
    dir = 0; // for gauge link
    #pragma unroll
    for (int r = 0; r < num_rhs_vec; r ++) {
      A_MINUS_BI(0, 3);
      A_MINUS_BI(1, 2);
    }
  }
  // y direction
  if (((result_y + 1) % schwarz_block_size != 0) ^ update_boundary) {
    ferm_loc = plus_y;
#if FERM_GHOSTS
    ferm_in = ferm;
#endif
    dir = 1;
    #pragma unroll
    for (int r = 0; r < num_rhs_vec; r ++) {
      A_OP_B(0, 3, +);
      A_OP_B(1, 2, -);
    }
  }
  // z direction
  if (((result_z + 1) % schwarz_block_size != 0) ^ update_boundary) {
    ferm_loc = plus_z;
#if FERM_GHOSTS
    ferm_in = fb_pz;
#endif
    dir = 2;
    #pragma unroll
    for (int r = 0; r < num_rhs_vec; r ++) {
      A_MINUS_BI(0, 2);
      A_PLUS_BI(1, 3);
    }
  }
  // t direction
  if (((result_t + 1) % schwarz_block_size != 0) ^ update_boundary) {
    ferm_loc = plus_t;
#if FERM_GHOSTS
    ferm_in = fb_pt;
#endif
    dir = 3;
    #pragma unroll
    for (int r = 0; r < num_rhs_vec; r ++) {
      A_OP_B(0, 2, -);
      A_OP_B(1, 3, -);
    }
  }

  // negative shifts
  // x direction
  if ((full_x % schwarz_block_size != 0) ^ update_boundary) {
    ferm_loc = minus_x;
#if FERM_GHOSTS
    ferm_in = ferm;
#endif
    gauge_loc = gauge_minus_x;
    dir = 0; // for gauge link
    #pragma unroll
    for (int r = 0; r < num_rhs_vec; r ++) {
      A_PLUS_BI_DAG(0, 3);
      A_PLUS_BI_DAG(1, 2);
    }
  }
  // y direction
  if ((result_y % schwarz_block_size != 0) ^ update_boundary) {
    ferm_loc = minus_y;
#if FERM_GHOSTS
    ferm_in = ferm;
#endif
    gauge_loc = gauge_minus_y;
    dir = 1;
    #pragma unroll
    for (int r = 0; r < num_rhs_vec; r ++) {
      A_OP_B_DAG(0, 3, -);
      A_OP_B_DAG(1, 2, +);
    }
  }
  // z direction
  if ((result_z % schwarz_block_size != 0) ^ update_boundary) {
    ferm_loc = minus_z;
#if FERM_GHOSTS
    ferm_in = fb_mz;
#endif
    gauge_loc = gauge_minus_z;
    dir = 2;
    #pragma unroll
    for (int r = 0; r < num_rhs_vec; r ++) {
      A_PLUS_BI_DAG(0, 2);
      A_MINUS_BI_DAG(1, 3);
    }
  }
  // t direction
  if ((result_t % schwarz_block_size != 0) ^ update_boundary) {
    ferm_loc = minus_t;
#if FERM_GHOSTS
    ferm_in = fb_mt;
#endif
    gauge_loc = gauge_minus_t;
    dir = 3;
    #pragma unroll
    for (int r = 0; r < num_rhs_vec; r ++) {
      A_OP_B_DAG(0, 2, +);
      A_OP_B_DAG(1, 3, +);
    }
  }
}

// Pure-C reference implementation of the clover term: output += F * ferm at one
// site (in place when ferm == output).  Always compiled; it is the live path
// whenever the asm dispatch in clover_kernel below does not apply -- ASM=0, a
// double-precision TU, or a num_rhs the tile kernel cannot cover -- and it is
// the correctness oracle for the hand-written assembly kernels, mirroring
// wilson_kernel_c's role for the hopping term.
// in_place controls whether we are applying clover term to ferm
// or whether we are applying to output (= ferm)
void clover_kernel_c(REAL * output, REAL * ferm, REAL * F,
                     int result_x, int result_y, int result_z, int result_t) {
  int in_place = (ferm == output);
  long result_loc = site_half(result_x, result_y, result_z, result_t);
  // if ferm and output are the same, create a temporary
  REAL_VEC * temp_output;
  if (in_place) {
    temp_output = malloc(sizeof(complex REAL) * num_rhs * Nc * Ns);
    for (int i = 0; i < Nc * Ns * num_rhs_vec * 2; i ++)
      temp_output[i] = ZERO_VEC;
  }
  for (int block = 0; block < 2; block ++) {
    REAL_VEC * shifted_ferm = ((REAL_VEC *) (ferm)
               + (result_loc * Ns * Nc  + 6 * block) * 2 * num_rhs_vec);
    for (int a = 0; a < 6; a ++) {
      long output_index = (result_loc * Ns * Nc + 6 * block + a) * 2;
      REAL_VEC * ferm_vec, * output_vec;
      if (in_place)
        output_vec = temp_output + (6 * block + a) * 2 * num_rhs_vec;
      else
        output_vec = ((REAL_VEC *) (output + output_index * num_rhs));
      #pragma unroll
      for (int b = 0; b < 6; b ++) {
        ferm_vec = shifted_ferm + 2 * b * num_rhs_vec;

        int F_index_1 = (((result_loc * 2 + block) * 6 + a) * 6 + b);
        int F_index_2 = (((result_loc * 2 + block) * 6 + b) * 6 + a);
        REAL_VEC F_re, F_im;
        if (a <= b) {
          F_re = SET1(F[F_index_1]);
          F_im = SET1(F[F_index_2]);
        } else {
          F_re = SET1(F[F_index_2]);
          F_im = SET1(-F[F_index_1]);
        }
        #pragma unroll
        for (int c = 0; c < num_rhs_vec; c ++) {
          output_vec[c]               += F_re * ferm_vec[c];
          output_vec[num_rhs_vec + c] += F_re * ferm_vec[num_rhs_vec + c];
          if (a != b) {
            output_vec[num_rhs_vec + c] += F_im * ferm_vec[c];
            output_vec[c]               -= F_im * ferm_vec[num_rhs_vec + c];
          }
        }
      }
    }
  }
  // copy output_vec to ferm for in-place transfers
  if (in_place) {
    memcpy(output + result_loc * Ns * Nc * num_rhs * 2,
           temp_output, sizeof(REAL) * Ns * Nc * num_rhs * 2);
    free(temp_output);
  }
}

// clover piece of H.  Routes to the asm tile kernel where this build has one,
// otherwise to the portable C reference above.
void clover_kernel(REAL * output, REAL * ferm, REAL * F,
                   int result_x, int result_y, int result_z, int result_t) {
  // clover_kernel_asm (clover_ferm.h: clover_kernel_avx512 when AVX512, else
  // clover_kernel_avx2) is a cpp/GAS template that tiles its hand-tuned
  // num_rhs=16 schedule in 16-column steps, so it is correct for any num_rhs
  // that is a multiple of 16.  For other num_rhs (rare) fall through to the
  // portable C path, which scales via num_rhs_vec.
  #if CLOVER_ASM_LIVE
    int in_place = (ferm == output);
    long result_loc = site_half(result_x, result_y, result_z, result_t);
    long ferm_offset = result_loc * Ns * Nc * 2 * num_rhs;
    long F_offset    = result_loc * 2 * 6 * 6;
    if (!in_place) {
      clover_kernel_asm(output + ferm_offset, ferm + ferm_offset,
                        F + F_offset);
    } else {
      // in-place version requires temporary for output.  The asm kernel
      // accumulates into its output arg, so zero the temp then apply F to the
      // (read-only) input ferm; copy the result back over output last.
      // The temp is one site (Nc*Ns*num_rhs*2 REALs, a compile-time constant),
      // so it is a 64-byte-aligned stack buffer rather than a heap allocation:
      // red_black_schwarz's block-eo MR calls this once per odd site per inner
      // iteration, and a malloc/free per call would dominate.
      _Alignas(64) REAL temp_output[Nc * Ns * num_rhs * 2];
      for (int i = 0; i < Nc * Ns * num_rhs_vec * 2; i ++)
        ((REAL_VEC *) temp_output)[i] = ZERO_VEC;
      clover_kernel_asm(temp_output, ferm + ferm_offset, F + F_offset);
      memcpy(output + ferm_offset,
             temp_output, sizeof(REAL) * Ns * Nc * num_rhs * 2);
    }
  #else
    clover_kernel_c(output, ferm, F, result_x, result_y, result_z, result_t);
  #endif
}

// Overwrite variant: output = F * ferm (no accumulate, output must not alias
// ferm).  The asm path is clover_kernel_ow_asm, the same hand-tuned schedule
// with each output accumulator's first load replaced by a register zero -- so
// it is bit-identical to "zero the output site, then clover_kernel" while
// skipping the caller's zeroing pass and the first-touch output reads.  The
// portable fallback realizes exactly that zero-then-accumulate pattern.
void clover_kernel_ow(REAL * output, REAL * ferm, REAL * F,
                      int result_x, int result_y, int result_z, int result_t) {
  long result_loc = site_half(result_x, result_y, result_z, result_t);
  #if CLOVER_ASM_LIVE
    long ferm_offset = result_loc * Ns * Nc * 2 * num_rhs;
    long F_offset    = result_loc * 2 * 6 * 6;
    clover_kernel_ow_asm(output + ferm_offset, ferm + ferm_offset,
                         F + F_offset);
  #else
    REAL_VEC * out_vec = (REAL_VEC *) (output
                         + result_loc * Ns * Nc * num_rhs * 2);
    for (int i = 0; i < Nc * Ns * num_rhs_vec * 2; i ++)
      out_vec[i] = ZERO_VEC;
    clover_kernel(output, ferm, F, result_x, result_y, result_z, result_t);
  #endif
}

void H(REAL * output, REAL * ferm, REAL * U, REAL * F,
       int clover) {
  wilson_halo_half(ferm, 0);
  wilson_halo_half(ferm + (long) ferm_size * num_rhs / 2, 1);
  BLOCK_LOOPS {
    int parity = (x ^ y ^ z ^ t) & 1;
    #if PREFETCH && PF_NEXT_OUTPUT
      // prefetch next location's output (read-modify-written by the kernel)
      if (x < x_mid + BLK2_X) {
        int next_x = x + 1;
        int next_parity = (next_x ^ y ^ z ^ t) & 1;
        int next_loc = site_half(next_x / 2, y, z, t);
        REAL * next_output = output
                            + (long) next_parity * ferm_size * num_rhs / 2;
        for (int i = 0; i < Nc * Ns * num_rhs_vec * 2; i ++) {
          long next_index = next_loc * Nc * Ns * 2;
          REAL_VEC * next_vec
            = ((REAL_VEC*) (next_output + next_index * num_rhs));
          __builtin_prefetch(next_vec + i, 0, 3);
        }
      }
    #endif
    #if PREFETCH && PF_NEXT_INPUT_X
      // Prefetch the next site's +x input fermion + its dir-0 gauge link --
      // the only neighbor data that is cold when the next kernel starts.
      // (A prefetch never faults, so an out-of-block/edge address is safe.)
      {
        int nx = x + 1;
        int nparity = (nx ^ y ^ z ^ t) & 1;
        int nrx = nx / 2;
        int nx_par = nx & 1;                       // = (nparity ^ y^z^t)&1
        int nplus_x = site_half((nrx + nx_par) % Lxh, y, z, t);
        REAL * nferm = ferm + (long) (1 - nparity) * ferm_size * num_rhs / 2;
        REAL_VEC * v = (REAL_VEC*) (nferm
                                    + (long) nplus_x * Nc * Ns * 2 * num_rhs);
        for (int i = 0; i < Nc * Ns * num_rhs_vec * 2; i ++)
          __builtin_prefetch(v + i, 0, 3);
        int nfull_x = nrx * 2 + nx_par;
        long ngauge = site_full(nfull_x, y, z, t);
        for (int i = 0; i < Nc * Nc * 2; i += 8)
          __builtin_prefetch(U + ngauge * 4 * Nc * Nc * 2 + i, 0, 3);
      }
    #endif
    wilson_kernel(output + (long) parity * ferm_size * num_rhs / 2,
                  ferm + (long) (1 - parity) * ferm_size * num_rhs / 2,
                  U, x/2, y, z, t, parity);
    if (clover)
      clover_kernel(output + (long) parity * ferm_size * num_rhs / 2,
                    ferm + (long) parity * ferm_size * num_rhs / 2,
                    F + parity * Vh * 2 * 6 * 6,
                    x/2, y, z, t);
  }
}

// compute H on one set of blocks
void H_half(REAL * output, REAL * ferm, REAL * U, int parity) {
  wilson_halo_half(ferm, 1 - parity);   // ferm is the input (1-parity) half
  BLOCK_LOOPS_HALF {
    wilson_kernel(output, ferm, U, x, y, z, t, parity);
  }
}

// neighbor couplings
void H_oe(REAL * output, REAL * ferm, REAL * U) {
  H_half(output, ferm, U, 1);
}
void H_eo(REAL * output, REAL * ferm, REAL * U) {
  H_half(output, ferm, U, 0);
}

// These are the self couplings.  H_ee has no standalone function: the even
// clover term is always applied in the same block visit as H_eo (see H_hat
// below), so a separate pass would only re-read the even sites.
void D_oo_inv(REAL * output, REAL * ferm, REAL * F) {
  BLOCK_LOOPS_HALF {
    clover_kernel(output, ferm, F, x, y, z, t);
  }
}

// preconditioned version of H (= H_ee + H_eo D_oo^{-1} H_oe)
// This returns even sites stored in output
// odd_ferm is temporary needed for this function
void H_hat(REAL * output, REAL * ferm, REAL * U, REAL * F,
           REAL * odd_ferm, int clover) {
  wilson_halo_half(ferm, 0);
  BLOCK_LOOPS_HALF {
    // zero odd sites
    long loc = site_half(x, y, z, t);
    for (int i = 0; i < Nc * Ns * num_rhs * 2; i ++)
      odd_ferm[loc * Nc * Ns * num_rhs * 2 + i] = 0;
    // H_oe
    wilson_kernel(odd_ferm, ferm, U, x, y, z, t, 1);
    // D_oo^{-1}
    if (clover)
      clover_kernel(odd_ferm, odd_ferm, F + Vh * 2 * 6 * 6,
                    x, y, z, t);
  }
  wilson_halo_half(odd_ferm, 1);   // just computed; sweep 2 reads it
  BLOCK_LOOPS_HALF {
    // H_ee
    if (clover)
      clover_kernel(output, ferm, F, x, y, z, t);
    else {
      long loc = site_half(x, y, z, t);
      for (int i = 0; i < Nc * Ns * num_rhs * 2; i ++)
        output[loc * Nc * Ns * num_rhs * 2 + i] = 0;
    }
    // H_eo
    wilson_kernel(output, odd_ferm, U, x, y, z, t, 0);
  }
}

#if MIXED && !MG_SINGLE
// ---------------------------------------------------------------------------
// Single-input / double-output operator variant for the mixed outer FGMRES.
// The preconditioned vectors z = M^{-1} V come straight out of the single
// V-cycle, so they are single-valued; D_si reads them as float and promotes
// each load to double (vcvtps2pd).  Gauge/clover/F, all arithmetic, and the
// output w stay double, so this is bit-identical to D applied to (double) z,
// at half the input-fermion DRAM read.  Used only for the Arnoldi matvec
// w = A z; the restart residual r = b - A x keeps the plain double D
// (x is a genuine double field).

// Single-input load templating: FERM_VEC_T is the float SIMD block whose lane
// count matches REAL_VEC (the double output vector); LOAD_FERM_VEC promotes
// one block to REAL_VEC on the fly (vcvtps2pd).  Mirrors precision.h's
// AVX512/AVX2/scalar split so D_si works on every target the double D does.
// Active for both _si kernels below; the default (double-input) templating
// is restored after them.
#undef FERM_VEC_T
#undef LOAD_FERM_VEC
#if AVX512
  #define FERM_VEC_T __m256  // 8 floats -> __m512d
  #define LOAD_FERM_VEC(fv, idx) _mm512_cvtps_pd((fv)[idx])
#elif AVX2
  #define FERM_VEC_T __m128  // 4 floats -> __m256d
  #define LOAD_FERM_VEC(fv, idx) _mm256_cvtps_pd((fv)[idx])
#else
  #define FERM_VEC_T float   // scalar promote
  #define LOAD_FERM_VEC(fv, idx) ((REAL_VEC) ((fv)[idx]))
#endif

// Ghost scratch for the float input of D_si (this TU's wk_fghost is double):
// same geometry, filled by wilson_halo_half_si below.  Heap for the same
// .bss-overflow reason as wk_fghost; the halo fill below is its only writer.
#if FERM_GHOSTS
static float * wk_fghost_si = NULL;
static float * wk_fghost_si_buf(void) {
  if (!wk_fghost_si) {
    wk_fghost_si = malloc(2 * GHOST_BLOCK_REALS * sizeof(float));
    memset(wk_fghost_si, 0, 2 * GHOST_BLOCK_REALS * sizeof(float));
  }
  return wk_fghost_si;
}
#endif
static void wilson_halo_half_si(const float * half, int input_parity) {
#if FERM_GHOSTS
  halo_ferm_half(half,
                 wk_fghost_si_buf() + (long) input_parity * GHOST_BLOCK_REALS,
                 (long) Nc * Ns * 2 * num_rhs * sizeof(float));
#else
  (void) half; (void) input_parity;
#endif
}

// single-input Wilson hop: same body as wilson_kernel_c (shared .inc), but
// the input fermion is float and each rhs-block load promotes float->double
static void wilson_kernel_c_si(REAL * output, float * ferm, REAL * U,
                   int result_x, int result_y, int result_z, int result_t,
                   int parity) {
#define WK_ACTIVE(n) 1
#define WK_GHOST_IN wk_fghost_si
#define WK_SAPH 0   // own ghost scratch (wilson_halo_half_si), never compact
#include "wilson_kernel_body.inc"
#undef WK_SAPH
#undef WK_GHOST_IN
#undef WK_ACTIVE
}

// Single-input clover (self-coupling): not-in-place only (D never calls it in
// place), float input promoted on load.  Mirrors the C path of clover_kernel.
static void clover_kernel_si(REAL * output, float * ferm, REAL * F,
                             int result_x, int result_y, int result_z,
                             int result_t) {
  long result_loc = site_half(result_x, result_y, result_z, result_t);
  for (int block = 0; block < 2; block ++) {
    FERM_VEC_T * shifted_ferm = ((FERM_VEC_T *) (ferm)
               + (result_loc * Ns * Nc + 6 * block) * 2 * num_rhs_vec);
    for (int a = 0; a < 6; a ++) {
      long output_index = (result_loc * Ns * Nc + 6 * block + a) * 2;
      REAL_VEC * output_vec = ((REAL_VEC *) (output + output_index * num_rhs));
      #pragma unroll
      for (int b = 0; b < 6; b ++) {
        FERM_VEC_T * ferm_vec = shifted_ferm + 2 * b * num_rhs_vec;
        int F_index_1 = (((result_loc * 2 + block) * 6 + a) * 6 + b);
        int F_index_2 = (((result_loc * 2 + block) * 6 + b) * 6 + a);
        REAL_VEC F_re, F_im;
        if (a <= b) {
          F_re = SET1(F[F_index_1]);
          F_im = SET1(F[F_index_2]);
        } else {
          F_re = SET1(F[F_index_2]);
          F_im = SET1(-F[F_index_1]);
        }
        #pragma unroll
        for (int c = 0; c < num_rhs_vec; c ++) {
          output_vec[c] += F_re * LOAD_FERM_VEC(ferm_vec, c);
          output_vec[num_rhs_vec + c]
            += F_re * LOAD_FERM_VEC(ferm_vec, num_rhs_vec + c);
          if (a != b) {
            output_vec[num_rhs_vec + c] += F_im * LOAD_FERM_VEC(ferm_vec, c);
            output_vec[c] -= F_im * LOAD_FERM_VEC(ferm_vec, num_rhs_vec + c);
          }
        }
      }
    }
  }
}
// restore the default (double-input) load templating for any later expansion
#undef FERM_VEC_T
#undef LOAD_FERM_VEC
#define FERM_VEC_T REAL_VEC
#define LOAD_FERM_VEC(fv, idx) ((fv)[idx])

// Single-input D: w = D z, z single (promote-on-load), w double.  Mirrors D().
void D_si(REAL * output, float * ferm, REAL * U, REAL * F, int clover) {
  wilson_halo_half_si(ferm, 0);
  wilson_halo_half_si(ferm + (long) ferm_size * num_rhs / 2, 1);
  BLOCK_LOOPS {
    int site_size = Nc * Ns * num_rhs * 2;
    int parity = (x ^ y ^ z ^ t) & 1;
    long loc = (long) parity * Vh + site_half(x/2, y, z, t);
    for (int i = 0; i < site_size; i ++) {
      long index = loc * site_size + i;
      output[index] = -(REAL) ferm[index];
    }
    wilson_kernel_c_si(output + (long) parity * ferm_size * num_rhs / 2,
                  ferm + (long) (1 - parity) * ferm_size * num_rhs / 2,
                  U, x/2, y, z, t, parity);
    if (clover)
      clover_kernel_si(output + (long) parity * ferm_size * num_rhs / 2,
                    ferm + (long) parity * ferm_size * num_rhs / 2,
                    F + parity * Vh * 2 * 6 * 6,
                    x/2, y, z, t);
    for (int i = 0; i < site_size; i ++)
      output[loc * site_size + i] *= -1;
  }
}
#endif  // MIXED && !MG_SINGLE

// apply the Wilson operator D to a fermion ferm, storing the result in output
// Note: By default, we use the mass convention in equation (5.57) of
// Gattringer and Lang.  This convention differs from chroma.
void D(REAL * output, REAL * ferm, REAL * U, REAL * F, int clover) {
  wilson_halo_half(ferm, 0);
  wilson_halo_half(ferm + (long) ferm_size * num_rhs / 2, 1);
  BLOCK_LOOPS {
    int site_size = Nc * Ns * num_rhs * 2;
    // copy ferm into output
    int parity = (x ^ y ^ z ^ t) & 1;
    long loc = (long) parity * Vh + site_half(x/2, y, z, t);
    for (int i = 0; i < site_size; i ++) {
      long index = loc * site_size + i;
      output[index] = -ferm[index];
    }
    wilson_kernel(output + (long) parity * ferm_size * num_rhs / 2,
                  ferm + (long) (1 - parity) * ferm_size * num_rhs / 2,
                  U, x/2, y, z, t, parity);
    if (clover)
      clover_kernel(output + (long) parity * ferm_size * num_rhs / 2,
                    ferm + (long) parity * ferm_size * num_rhs / 2,
                    F + parity * Vh * 2 * 6 * 6,
                    x/2, y, z, t);
    for (int i = 0; i < site_size; i ++)
      output[loc * site_size + i] *= -1;
  }
}

// Residual variant of D: returns r = eta - D ferm.  Identical to D() except
// the scratch buffer is `r` and the closing per-site pass (which in D()
// flips the sign to turn -(D ferm) into D ferm) instead folds in eta:
// at that point r holds -(D ferm), so r = eta + r == eta - D ferm.
// Bit-identical to computing D() then a separate `r = eta - Dpsi` loop,
// but with no extra memory pass or parallel region (the fold rides on the
// negate pass that was already there).
//
// eta_even_only (D_residual_eo): eta is an even-site half-field whose odd
// part is identically zero (the eo outer solve's Krylov vector).  Even sites
// lead the eo-split layout, so even-site indexing into the half-field is
// unchanged; on odd sites the fold is skipped because r = -(D ferm) already
// equals eta - D ferm there.  Bit-identical to folding an explicitly
// zero-padded eta (x + 0 == x).
static void D_residual_impl(REAL * r, REAL * eta, REAL * ferm, REAL * U,
                            REAL * F, int clover, int eta_even_only) {
  wilson_halo_half(ferm, 0);
  wilson_halo_half(ferm + (long) ferm_size * num_rhs / 2, 1);
  BLOCK_LOOPS {
    int site_size = Nc * Ns * num_rhs * 2;
    int parity = (x ^ y ^ z ^ t) & 1;
    long loc = (long) parity * Vh + site_half(x/2, y, z, t);
    for (int i = 0; i < site_size; i ++) {
      long index = loc * site_size + i;
      r[index] = -ferm[index];
    }
    wilson_kernel(r + (long) parity * ferm_size * num_rhs / 2,
                  ferm + (long) (1 - parity) * ferm_size * num_rhs / 2,
                  U, x/2, y, z, t, parity);
    if (clover)
      clover_kernel(r + (long) parity * ferm_size * num_rhs / 2,
                    ferm + (long) parity * ferm_size * num_rhs / 2,
                    F + parity * Vh * 2 * 6 * 6,
                    x/2, y, z, t);
    // r currently holds -(D ferm); fold eta in to get eta - D ferm directly.
    if (!eta_even_only || parity == 0)
      for (int i = 0; i < site_size; i ++) {
        long index = loc * site_size + i;
        r[index] = eta[index] + r[index];
      }
  }
}

void D_residual(REAL * r, REAL * eta, REAL * ferm, REAL * U, REAL * F,
                int clover) {
  D_residual_impl(r, eta, ferm, U, F, clover, 0);
}

// eta is the even-site half-field of the eo outer solve
// (odd part implicitly 0)
void D_residual_eo(REAL * r, REAL * eta, REAL * ferm, REAL * U, REAL * F,
                   int clover) {
  D_residual_impl(r, eta, ferm, U, F, clover, 1);
}

// Shared implementation of D_hat and D_hat_residual.  eta == NULL is the plain
// Schur apply; eta != NULL folds r = eta - D_hat ferm and its norm into sweep
// 2's closing per-site pass.  The branch is loop-invariant (a parameter), so
// it costs one predictable test per site against a wilson_kernel call.
static void D_hat_impl(REAL * output, REAL * eta, REAL * ferm, REAL * U, REAL * F,
                       REAL * odd_ferm, int clover, REAL * norm_out) {
  // BLOCK_LOOPS_HALF_REDUCTION_RHS reduces on this name.  Per rhs rather than
  // one scalar: the outer solve gates every rhs against its own ||b_k||,
  // so the restart residual has to come back with the lanes still apart.
  // The plain Schur apply (eta == NULL) never touches it and pays only the
  // empty reduction, once per call -- negligible against a full sweep of the
  // local volume.
  REAL norm[num_rhs];
  for (int k = 0; k < num_rhs; k ++) norm[k] = 0;
  wilson_halo_half(ferm, 0);
  BLOCK_LOOPS_HALF {
    // zero odd sites
    long loc = site_half(x, y, z, t);
    for (int i = 0; i < Nc * Ns * num_rhs * 2; i ++)
      odd_ferm[loc * Nc * Ns * num_rhs * 2 + i] = 0;
    // H_oe
    wilson_kernel(odd_ferm, ferm, U, x, y, z, t, 1);
    // D_oo^{-1}
    if (clover)
      clover_kernel(odd_ferm, odd_ferm, F + Vh * 2 * 6 * 6,
                    x, y, z, t);
  }
  wilson_halo_half(odd_ferm, 1);   // just computed; sweep 2 reads it
  BLOCK_LOOPS_HALF_REDUCTION_RHS {
    long loc = site_half(x, y, z, t);
    int site_size = Nc * Ns * num_rhs * 2;
    for (int i = 0; i < site_size; i ++) {
      long index = loc * site_size + i;
      output[index] = -ferm[index];
    }
    // H_ee
    if (clover)
      clover_kernel(output, ferm, F, x, y, z, t);
    // H_eo
    wilson_kernel(output, odd_ferm, U, x, y, z, t, 0);
    if (eta == NULL) {
      for (int i = 0; i < site_size; i ++)
        output[loc * site_size + i] *= -1;
    } else {
      // output currently holds -(D_hat ferm), so the residual is just
      // r = eta - D_hat ferm = eta + output: the sign flip this pass was
      // already doing becomes the eta combine, on rows still in registers.
      // The norm rides here too -- deferring it to a separate pass would put
      // the full-field read of r straight back.
      // rhs is the fastest index, so the accumulator is one vector register
      // wide and the lanes never mix -- same flop count as the scalar
      // version it replaced, minus that version's horizontal fold.
      // Kept site-local (rather than accumulating straight into the
      // reduction array) so it stays in registers across the whole site,
      // exactly as the scalar `local` did.
      REAL local[num_rhs];
      for (int k = 0; k < num_rhs; k ++) local[k] = 0;
      for (int i = 0; i < site_size; i += num_rhs)
        for (int k = 0; k < num_rhs; k ++) {
          long index = loc * site_size + i + k;
          REAL ri = eta[index] + output[index];
          output[index] = ri;
          local[k] += ri * ri;
        }
      for (int k = 0; k < num_rhs; k ++) norm[k] += local[k];
    }
  }
  if (norm_out) for (int k = 0; k < num_rhs; k ++) norm_out[k] = norm[k];
}

// This is the analog of H_hat but with the extra contribution of -ferm
void D_hat(REAL * output, REAL * ferm, REAL * U, REAL * F,
           REAL * odd_ferm, int clover) {
  D_hat_impl(output, NULL, ferm, U, F, odd_ferm, clover, NULL);
}

// D_hat with the residual folded in: r = eta - D_hat ferm, plus the per-rhs
// ||r_k||^2 into norm_out[0..num_rhs-1] (rank-local -- the caller allreduces).
// Per rhs because the outer solve's convergence test is per rhs; sum the
// entries for the stacked ||r||^2.  eta and r are even-site half-fields;
// r is the Dslash scratch, so it must alias neither eta nor ferm.
void D_hat_residual(REAL * r, REAL * eta, REAL * ferm, REAL * U, REAL * F,
                    REAL * odd_ferm, int clover, REAL * norm_out) {
  D_hat_impl(r, eta, ferm, U, F, odd_ferm, clover, norm_out);
}
