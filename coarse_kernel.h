// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef COARSE_KERNEL_H
#define COARSE_KERNEL_H
#include "global_params.h"   // REAL / REAL_VEC / num_rhs (via precision.h)

// Coarse-stencil pieces shared by coarse.c (apply_Dc, apply_Dc_lvl, the
// Galerkin builds) and coarse_sap.c (the block matvec and the blocked HEMM).
// Kept in a header, and kept `static inline`, so both translation units inline
// the site kernel exactly as they did when it was file-local -- an exported
// call here would cost a call per site in the hottest loop of each.

// coarse nearest-neighbor offsets per direction: +x,+y,+z,+t,-x,-y,-z,-t
static const int coarse_off[8][4] = {
  { 1, 0, 0, 0}, { 0, 1, 0, 0}, { 0, 0, 1, 0}, { 0, 0, 0, 1},
  {-1, 0, 0, 0}, { 0,-1, 0, 0}, { 0, 0,-1, 0}, { 0, 0, 0,-1},
};

// --- Shared vectorized per-site coarse Dslash kernel ------------------------
// The same kernel serves the macro-geometry level-1 apply_Dc and the
// runtime-geometry apply_Dc_lvl below; only the site loop bound and the
// neighbor gather are per-level, and the callers do those.  num_rhs and MR are
// compile-time (the asm bakes num_rhs); N_coarse is passed to the asm
// microkernel as a runtime argument (gen_coarse_dslash_avx{2,512}.py),
// so one kernel serves any coarse depth's null-vector count.  Without the asm
// (and in every double-precision TU) the portable num_rhs-vectorized loop
// runs, reading N_coarse directly.
#if ASM && MG_SINGLE && AVX512
void coarse_dslash_kernel_avx512(REAL * out, REAL ** cbases, REAL ** ibases,
                                 long coff, int ncpl, long n_coarse);
#define DC_KERNEL      coarse_dslash_kernel_avx512
#define DC_MR 4
#elif ASM && MG_SINGLE && AVX2
void coarse_dslash_kernel_avx2(REAL * out, REAL ** cbases, REAL ** ibases,
                               long coff, int ncpl, long n_coarse);
#define DC_KERNEL      coarse_dslash_kernel_avx2
#define DC_MR 4       // matches gen_coarse_dslash_avx2.py default spec "4422"
#endif

// One coarse site: outb = sum_{k<ncpl} C[k] (.) in[k], from the gathered
// coupling-matrix bases cbases[k] and input-block bases ibases[k].  ncpl = 9
// is the full self + 8-neighbor stencil; the coarse SAP block matvec passes
// fewer when block faces are excluded.  outb (this site's 2*N_coarse*num_rhs
// block) is overwritten with the sum -- the accumulators start at zero, so
// callers need not pre-zero it.  The split-complex layout (re lanes then im
// lanes, num_rhs wide, per coarse var) is exactly coarse_idx's, so callers
// using either addressing convention agree.  Precondition for the asm path:
// cbases must hold ncpl+1 slots, since the kernel prefetches cbases[ncpl] one
// past the end (contents irrelevant -- prefetch ignores bad addresses);
// a 10-slot array covers every ncpl up to 9.
static inline void coarse_dslash_site(REAL * outb, REAL ** cbases,
                                      REAL ** ibases, int ncpl, int n_coarse) {
#ifdef DC_KERNEL
  for (int a0 = 0; a0 < n_coarse; a0 += DC_MR)    // DC_MR output rows per call
    DC_KERNEL(outb + (long) a0 * 2 * num_rhs, cbases, ibases,
              (long) a0 * n_coarse * 2 * (long) sizeof(REAL), ncpl, n_coarse);
#else
  for (int a = 0; a < n_coarse; a ++) {
    REAL_VEC accr[num_rhs_vec], acci[num_rhs_vec];
    #pragma unroll
    for (int rv = 0; rv < num_rhs_vec; rv ++) {
      accr[rv] = ZERO_VEC;
      acci[rv] = ZERO_VEC;
    }
    for (int k = 0; k < ncpl; k ++) {
      // row a of coupling k
      const REAL * C  = cbases[k] + (long) a * n_coarse * 2;
      const REAL * ib = ibases[k];
      for (int b = 0; b < n_coarse; b ++) {
        REAL_VEC cr = SET1(C[2 * b]);
        REAL_VEC ci = SET1(C[2 * b + 1]);
        const REAL * ibb = ib + (long) b * 2 * num_rhs;
        #pragma unroll
        for (int rv = 0; rv < num_rhs_vec; rv ++) {
          REAL_VEC ir  = *(const REAL_VEC *) (ibb + rv * veclen);
          REAL_VEC iim = *(const REAL_VEC *) (ibb + num_rhs + rv * veclen);
          accr[rv] += cr * ir - ci * iim;
          acci[rv] += cr * iim + ci * ir;
        }
      }
    }
    REAL * ob = outb + (long) a * 2 * num_rhs;
    #pragma unroll
    for (int rv = 0; rv < num_rhs_vec; rv ++) {
      *(REAL_VEC *) (ob + rv * veclen)            = accr[rv];
      *(REAL_VEC *) (ob + num_rhs + rv * veclen)  = acci[rv];
    }
  }
#endif
}

#endif // COARSE_KERNEL_H
