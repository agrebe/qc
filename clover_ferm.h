// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef CLOVER_FERM_H
#define CLOVER_FERM_H

#include "global_params.h"

void H(REAL * output, REAL * ferm, REAL * U, REAL * F,
       int clover);
void H_oe(REAL * output, REAL * ferm, REAL * U);
void H_eo(REAL * output, REAL * ferm, REAL * U);
void D_oo_inv(REAL * output, REAL * ferm, REAL * F);
void H_hat(REAL * output, REAL * ferm, REAL * U, REAL * F,
           REAL * odd_ferm, int clover);
void wilson_kernel(REAL * output, REAL * ferm, REAL * U,
                   int result_x, int result_y, int result_z, int result_t,
                   int parity);
void wilson_kernel_c(REAL * output, REAL * ferm, REAL * U,
                     int result_x, int result_y, int result_z, int result_t,
                     int parity);
// Masked variant of wilson_kernel_c: 8-bit wk_mask selects directions
// (0:+x 1:+y 2:+z 3:+t 4:-x 5:-y 6:-z 7:-t); excluded directions are skipped.
// Accumulates into output like wilson_kernel_c.
void wilson_kernel_dir_c(REAL * output, REAL * ferm, REAL * U,
                     int result_x, int result_y, int result_z, int result_t,
                     int parity, int wk_mask);
void wilson_kernel_block(REAL * output, REAL * ferm, REAL * U,
                         int result_x, int result_y, int result_z,
                         int result_t, int parity, int schwarz_block_size,
                         int update_boundary);
void wilson_kernel_block_c(REAL * output, REAL * ferm, REAL * U,
                         int result_x, int result_y, int result_z,
                         int result_t, int parity, int schwarz_block_size,
                         int update_boundary);
// Interior (update_boundary=0) block call via the shared bs^4 offset table:
// same result as wilson_kernel_block(..., 0) without the per-site address
// math.  org_half/org_full = block origin's half-/full-lattice site index;
// local_idx = local site index in SCHWARZ_BLOCK_SITES visit order (x fastest).
// Call wilson_block_itab_init(bs) once (outside parallel regions) first.
void wilson_block_itab_init(int schwarz_block_size);
// Exchange one parity-half fermion block's z/t rank faces into the shared
// ghost scratch the stencil kernels read (input_parity selects the scratch
// block; no-op when no dimension is decomposed).  The D-family wrappers call
// it internally; only code that drives wilson_kernel* sweeps directly
// (richardson's inline eo sweeps, the SAP boundary update, setup's per-site
// applies) needs explicit calls.
void wilson_halo_half(const REAL * half, int input_parity);
// Setup's cross-rank coarse-op build: install a locally-assembled seam block
// into the fermion ghost scratch, and zero the scratch.  No-ops at MPI=0.
void wilson_set_fghost(const REAL * seam_block, int input_parity);
void wilson_zero_fghost(void);
// SAP compact halo: exchange one or both parity halves (NULL = skip) for the
// half-sweep whose visited blocks have color reader_color, shipping only the
// live quarter of each face (color-halved rows, spin-projected payload) --
// bit-identical to the wilson_halo_half pair for every read a color-
// reader_color block can make.  See the design comment in clover_ferm.c.
// Master thread, outside parallel regions; no-op when nothing is decomposed.
void wilson_halo_sap(const REAL * even, const REAL * odd, int reader_color);
// Serial-window breakdown of the compact exchange (pack / MPI / expand secs
// since the last reset; zeros undecomposed).  Reset rides sap_stats_reset.
void wilson_halo_sap_stats_get(double * pack, double * mpi, double * unpack);
void wilson_halo_sap_stats_reset(void);
void wilson_kernel_block_itab(REAL * output, REAL * ferm, REAL * U,
                         int result_x, int result_y, int result_z,
                         int result_t, int parity, int schwarz_block_size,
                         long org_half, long org_full, int local_idx);
// Pure-C reference for the clover term (clover_ferm.c).  Always compiled: the
// live path wherever the asm tile kernels do not apply, and their correctness
// oracle -- the clover counterpart of wilson_kernel_c.
void clover_kernel_c(REAL * output, REAL * ferm, REAL * F,
                     int result_x, int result_y, int result_z, int result_t);
void clover_kernel(REAL * output, REAL * ferm, REAL * F,
                   int result_x, int result_y, int result_z, int result_t);
// Overwrite variant: output = F * ferm at the site (no accumulate).  output
// must not alias ferm (no in-place support -- that's the pattern this variant
// exists to eliminate).  Bit-identical to zeroing the output site and calling
// clover_kernel.
void clover_kernel_ow(REAL * output, REAL * ferm, REAL * F,
                      int result_x, int result_y, int result_z, int result_t);
void D(REAL * output, REAL * ferm, REAL * U, REAL * F,
       int clover);
#if MIXED && !MG_SINGLE
// Single-input / double-output D for the mixed outer FGMRES (Arnoldi matvec
// on the single-valued preconditioned vector z): reads a float input fermion,
// promotes on load, double output -- bit-identical to D applied to
// (double) z, half the input BW.
void D_si(REAL * output, float * ferm, REAL * U, REAL * F, int clover);
#endif
// Fused residual: r = eta - D ferm, computed in D's existing final per-site
// pass (no extra streaming loop / fork-join).  r is used as the Dslash
// scratch, so it must alias neither eta nor ferm.
void D_residual(REAL * r, REAL * eta, REAL * ferm, REAL * U, REAL * F,
                int clover);
// D_residual with eta an even-site half-field whose odd part is implicitly
// zero (eo outer solve): the fold is skipped on odd sites -- bit-identical to
// D_residual on the zero-padded eta, minus half the eta stream.
void D_residual_eo(REAL * r, REAL * eta, REAL * ferm, REAL * U, REAL * F,
                   int clover);
void D_hat(REAL * output, REAL * ferm, REAL * U, REAL * F,
           REAL * odd_ferm, int clover);
// D_hat with the residual folded in, the Schur counterpart of D_residual:
// r = eta - D_hat ferm and the per-rhs ||r_k||^2 (rank-local, into
// norm_out[0..num_rhs-1]) are computed in the sign-flip pass that closed
// D_hat's second sweep, while those rows are still register-resident.
// Deletes the separate pass's Ax round trip -- a full even-half-field
// read-back plus a second writeback.  Per rhs because the outer convergence
// test is per rhs; sum the entries for the stacked norm.  eta and r are
// even-site half-fields; r must alias neither eta nor ferm.
void D_hat_residual(REAL * r, REAL * eta, REAL * ferm, REAL * U, REAL * F,
                    REAL * odd_ferm, int clover, REAL * norm_out);

// True exactly when clover_kernel / clover_kernel_ow route to the hand-written
// asm tile kernels rather than to clover_kernel_c.  The tile kernels cover any
// num_rhs that is a multiple of 16 (see the dispatch comment in clover_kernel);
// other values fall through to the portable C path.  Named rather than spelled
// out at each use so the condition has one definition: clover_ferm.c branches
// on it, and test/kernel_tests.c uses it to report whether its asm-vs-C
// comparison has any content in the twin being run.
#if ASM && MG_SINGLE && (num_rhs % 16 == 0)
  #define CLOVER_ASM_LIVE 1
#else
  #define CLOVER_ASM_LIVE 0
#endif

// The same for wilson_kernel.  There is no num_rhs condition here, unlike the
// clover pair: the AVX-512 kernel is tiled 16 columns at a time and the
// generated AVX2 kernel takes all num_rhs in one call, so between them every
// supported num_rhs is covered and a build with neither ISA is an #error below.
// Covers the whole Wilson family: the full-lattice wilson_kernel, the masked
// Schwarz-block wilson_kernel_block, and the offset-table
// wilson_kernel_block_ itab with its table.  One condition serves all of them
// because the ISA choice below picks a kernel per num_rhs rather than falling
// back to C, so unlike the clover pair there is nothing left to constrain.
// Split this only when a kernel actually acquires a different condition.
#if ASM && MG_SINGLE
  #define WILSON_ASM_LIVE 1
#else
  #define WILSON_ASM_LIVE 0
#endif

#if ASM && MG_SINGLE
  // The AVX2 kernels are generated from global_params.h and rescale to any
  // num_rhs (multiple of 8).  The AVX512 kernels (wilson_kernel_avx512.S) are
  // a cpp/GAS template that processes one 16-column tile; the C caller tiles
  // them num_rhs/16 times, so they are valid for any num_rhs that is a
  // multiple of 16. Use AVX512 when it covers this num_rhs; otherwise fall
  // back to the generated AVX2 kernels, which are always assembled/linked and
  // run fine on AVX512 hardware (-march=skylake-avx512 includes AVX2).
  // The #error only fires for a build with no SIMD asm ISA at all.
  #if AVX512 && (num_rhs % 16 == 0)
    #define wilson_kernel_asm wilson_kernel_avx512
    #define wilson_kernel_masked_asm wilson_kernel_avx512_masked
    // The AVX512 kernel processes one 16-column tile per call, so the
    // dispatcher calls it num_rhs/16 times with the fermion/output pointers
    // stepped by 16 columns (gauge unchanged).
    #define WILSON_ASM_TILE 16
    // The AVX512 clover kernel tiles internally (like the AVX2 one), so the
    // clover dispatch is just a symbol swap.
    #define clover_kernel_asm    clover_kernel_avx512
    #define clover_kernel_ow_asm clover_kernel_avx512_ow
  #elif AVX2 || AVX512
    #define wilson_kernel_asm wilson_kernel_avx2
    #define wilson_kernel_masked_asm wilson_kernel_avx2_masked
    // The generated AVX2 kernel handles all num_rhs columns internally (NHALF
    // loop), so it is called once: a single tile of width num_rhs.
    #define WILSON_ASM_TILE num_rhs
    #define clover_kernel_asm    clover_kernel_avx2
    #define clover_kernel_ow_asm clover_kernel_avx2_ow
  #else
    #error "no asm Wilson kernel: build with AVX2 or AVX512 (ASM requires a SIMD ISA)"
  #endif
#endif


void clover_kernel_avx2(float * output, float * ferm, float * F);
// Overwrite twin of clover_kernel_avx2 (same schedule, first output loads
// zeroed; built from the same .S with CLOVER_OW=1): output = F * ferm.
void clover_kernel_avx2_ow(float * output, float * ferm, float * F);
// AVX-512 twins.  The FMAs are grouped for accumulator ILP (see
// clover_kernel_avx512.S), so each accumulator receives the same products as
// the AVX2 pair but in a different order: results agree to roundoff, not bits.
void clover_kernel_avx512(float * output, float * ferm, float * F);
void clover_kernel_avx512_ow(float * output, float * ferm, float * F);
void wilson_kernel_avx2(float * output,
                        float * ferm_plus_x, float * ferm_plus_y,
                        float * ferm_plus_z, float * ferm_plus_t,
                        float * ferm_minus_x, float * ferm_minus_y,
                        float * ferm_minus_z, float * ferm_minus_t,
                        float * U, float * U_minus_x, float * U_minus_y,
                        float * U_minus_z, float * U_minus_t);
// Schwarz-block variant: 15th arg is an 8-bit mask, bit i = perform direction
// i (0:+x 1:+y 2:+z 3:+t 4:-x 5:-y 6:-z 7:-t); excluded directions are
// branched over.  No software prefetch (block data is L2-resident).
void wilson_kernel_avx2_masked(float * output,
                        float * ferm_plus_x, float * ferm_plus_y,
                        float * ferm_plus_z, float * ferm_plus_t,
                        float * ferm_minus_x, float * ferm_minus_y,
                        float * ferm_minus_z, float * ferm_minus_t,
                        float * U, float * U_minus_x, float * U_minus_y,
                        float * U_minus_z, float * U_minus_t, int mask);
void wilson_kernel_avx512(float * output,
                          float * ferm_plus_x, float * ferm_plus_y,
                          float * ferm_plus_z, float * ferm_plus_t,
                          float * ferm_minus_x, float * ferm_minus_y,
                          float * ferm_minus_z, float * ferm_minus_t,
                          float * U, float * U_minus_x, float * U_minus_y,
                          float * U_minus_z, float * U_minus_t);
// Schwarz-block variant: 15th arg is an 8-bit mask, bit i = perform direction
// i (0:+x 1:+y 2:+z 3:+t 4:-x 5:-y 6:-z 7:-t); excluded directions are
// branched over.  No software prefetch (block data is L2-resident).
void wilson_kernel_avx512_masked(float * output,
                          float * ferm_plus_x, float * ferm_plus_y,
                          float * ferm_plus_z, float * ferm_plus_t,
                          float * ferm_minus_x, float * ferm_minus_y,
                          float * ferm_minus_z, float * ferm_minus_t,
                          float * U, float * U_minus_x, float * U_minus_y,
                          float * U_minus_z, float * U_minus_t, int mask);

#define OUTER_LOOPS_HALF \
  for (int t_outer = 0; t_outer < Lt; t_outer += block_size_half) \
    for (int z_outer = 0; z_outer < Lz; z_outer += block_size_half) \
      for (int y_outer = 0; y_outer < Ly; y_outer += block_size_half) \
        for (int x_outer = 0; x_outer < Lxh; x_outer += block_size_half / 2)

// --- Cache blocking --------------------------------------------------------
// OUTER tile (coarse) with an optional MIDDLE sub-tile (BLK2_*, ~L1; defaults
// to the OUTER size so the middle loop is a single trip => no-op).
// The BLK_* / BLK2_* values live in global_params.h with the other tuning
// parameters.
//
// The kernel is DRAM-bandwidth bound; the tiles exist to keep each thread's
// neighbor-reuse working set in cache so the field streams from DRAM ~once.
// Retuning for another machine / lattice / thread count:
//   1. BLK_X = Lx always.  x is swept innermost and the asm prefetches the +x
//      neighbor; blocking x breaks that contiguous stream.
//   2. Enough parallel tiles: (Lt/BLK_T)*(Lz/BLK_Z)*(Ly/BLK_Y) >= nthreads
//      (a small multiple gives load-balance slack).  This caps block size.
//   3. Among the choices left take the largest BLK_T -- t is the outermost
//      tile dim, so deepening it costs no working set and each loaded
//      cross-section is reused BLK_T times before eviction -- and size
//      BLK_Y,BLK_Z so the reuse cross-section fits private cache:
//        nthreads * (BLK_X*BLK_Y*BLK_Z * Nc*Ns*2*num_rhs*sizeof(REAL))
//          < L3_shared
//      and ideally BLK_X*BLK_Y*BLK_Z*site_bytes <~ L2_private.  num_rhs scales
//      site_bytes, so larger num_rhs -> smaller BLK_Y*BLK_Z.
// Every BLK_* must divide its lattice dim (the loops do not clamp at
// boundaries).

#define OUTER_LOOPS \
  for (int t_outer = 0; t_outer < Lt; t_outer += BLK_T) \
    for (int z_outer = 0; z_outer < Lz; z_outer += BLK_Z) \
      for (int y_outer = 0; y_outer < Ly; y_outer += BLK_Y) \
        for (int x_outer = 0; x_outer < Lx; x_outer += BLK_X)

#define MIDDLE_LOOPS \
  for (int t_mid = t_outer; t_mid < t_outer + BLK_T; t_mid += BLK2_T) \
    for (int z_mid = z_outer; z_mid < z_outer + BLK_Z; z_mid += BLK2_Z) \
      for (int y_mid = y_outer; y_mid < y_outer + BLK_Y; y_mid += BLK2_Y) \
        for (int x_mid = x_outer; x_mid < x_outer + BLK_X; x_mid += BLK2_X)

#define INNER_LOOPS_HALF \
  for (int t = t_outer; t < t_outer + block_size_half; t ++) \
    for (int z = z_outer; z < z_outer + block_size_half; z ++) \
      for (int y = y_outer; y < y_outer + block_size_half; y ++) \
        for (int x = x_outer; x < x_outer + block_size_half / 2; x ++)

#define INNER_LOOPS \
  for (int t = t_mid; t < t_mid + BLK2_T; t ++) \
    for (int z = z_mid; z < z_mid + BLK2_Z; z ++) \
      for (int y = y_mid; y < y_mid + BLK2_Y; y ++) \
        for (int x = x_mid; x < x_mid + BLK2_X; x ++)

#define BLOCK_LOOPS_HALF \
  _Pragma("omp parallel for collapse(3)") \
  OUTER_LOOPS_HALF \
    INNER_LOOPS_HALF

// Reduces a num_rhs-wide accumulator named `norm`.  Every caller gates each rhs
// against its own ||b_k||, so the lanes have to come back separated; there is no
// scalar variant.
#define BLOCK_LOOPS_HALF_REDUCTION_RHS \
  _Pragma("omp parallel for collapse(3) reduction(+:norm[:num_rhs])") \
  OUTER_LOOPS_HALF \
    INNER_LOOPS_HALF

// Tiles are processed serially with the omp pragma on the INNER (site) loops,
// so all threads cooperate on one tile at a time, fork/join per tile.
// Parallelism never starves (good for small local volume / many threads), only
// one tile is resident, and intra-tile halos are shared -- at the cost of more
// fork/join.  (The alternative, parallelizing the OUTER tile loops so each
// thread owns whole tiles, was measured slower on every volume.)
#define BLOCK_LOOPS \
  OUTER_LOOPS \
    MIDDLE_LOOPS \
      _Pragma("omp parallel for collapse(3)") \
      INNER_LOOPS
#define BLOCK_LOOPS_REDUCTION_RHS \
  OUTER_LOOPS \
    MIDDLE_LOOPS \
      _Pragma("omp parallel for collapse(3) reduction(+:norm[:num_rhs])") \
      INNER_LOOPS

#endif // CLOVER_FERM_H
