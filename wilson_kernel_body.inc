// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

// Shared Wilson hopping kernel body (one result site, all num_rhs).
// #included by wilson_kernel_c (double input) and, in the mixed build, by
// wilson_kernel_c_si (single input, promote-on-load) -- the only difference
// between them is FERM_VEC_T / LOAD_FERM_VEC (see clover_ferm.c).  Relies on
// the enclosing function's params: output, ferm, U, result_x/y/z/t, parity.
// Under a decomposed MPI grid (DECOMP_Z/T) the off-rank z/t neighbors are
// read from the WK_GHOST_IN scratch (filled by wilson_halo_half before the
// sweep) and the boundary gauge links from U's tail ghosts; the default 1x1
// grid compiles to exactly the original periodic indexing.
  int result_loc = site_half(result_x, result_y, result_z, result_t);
  REAL_VEC temp_re [3], temp_im [3], prod_re, prod_im;
  int ferm_loc, dir;

  // various shifts of result_loc
  int x_par = (parity ^ result_y ^ result_z ^ result_t) & 0x1;
  int plus_x = site_half((result_x + x_par) % Lxh, result_y, result_z,
                         result_t);
  int plus_y = site_half(result_x, (result_y + 1) % Ly, result_z, result_t);
  int minus_x = site_half((result_x + Lxh - 1 + x_par) % Lxh, result_y,
                          result_z, result_t);
  int minus_y = site_half(result_x, (result_y + Ly - 1) % Ly, result_z,
                          result_t);
#if FERM_GHOSTS
  // per-direction input base: x/y neighbors always read the field; z/t
  // neighbors across a rank face read the input half's ghost block instead.
  __typeof__(ferm) const ghost_in = WK_GHOST_IN
                                  + (long) (1 - parity) * GHOST_BLOCK_REALS;
  __typeof__(ferm) ferm_in = ferm;
  #define WK_SET_IN(b) ferm_in = (b);
#else
  __typeof__(ferm) const ferm_in = ferm;   // never reassigned
  #define WK_SET_IN(b)
#endif
#if DECOMP_Z
  int ghz = (result_t * Ly + result_y) * Lxh + result_x;  // z-face ghost site
#if WK_SAPH
  // stage-2 SAP compact ghost (g_saph_active): expand the row into an L1
  // stack tile on demand and read the hop from it (index 0 in the tile)
  REAL wk_ztile[(long) Nc * Ns * 2 * num_rhs] __attribute__((aligned(64)));
#endif
  __typeof__(ferm) fb_pz, fb_mz;
  int plus_z, minus_z;
  if (result_z + 1 < Lz) {
    fb_pz = ferm;
    plus_z = site_half(result_x, result_y, result_z + 1, result_t);
  }
#if WK_SAPH
  else if (g_saph_active) {
    fb_pz = saph_ghost_tile(wk_ztile, SAPH_ZHI, 1 - parity, ghz); plus_z = 0;
  }
#endif
  else                   { fb_pz = ghost_in; plus_z = G_OFF_ZHI + ghz; }
  if (result_z > 0) {
    fb_mz = ferm;
    minus_z = site_half(result_x, result_y, result_z - 1, result_t);
  }
#if WK_SAPH
  else if (g_saph_active) {
    fb_mz = saph_ghost_tile(wk_ztile, SAPH_ZLO, 1 - parity, ghz); minus_z = 0;
  }
#endif
  else                   { fb_mz = ghost_in; minus_z = G_OFF_ZLO + ghz; }
  #define WK_FB_PZ fb_pz
  #define WK_FB_MZ fb_mz
#else
  int plus_z = site_half(result_x, result_y, (result_z + 1) % Lz, result_t);
  int minus_z = site_half(result_x, result_y, (result_z + Lz - 1) % Lz,
                          result_t);
  #define WK_FB_PZ ferm
  #define WK_FB_MZ ferm
#endif
#if DECOMP_T
  int ght = (result_z * Ly + result_y) * Lxh + result_x;  // t-face ghost site
#if WK_SAPH
  REAL wk_ttile[(long) Nc * Ns * 2 * num_rhs] __attribute__((aligned(64)));
#endif
  __typeof__(ferm) fb_pt, fb_mt;
  int plus_t, minus_t;
  if (result_t + 1 < Lt) {
    fb_pt = ferm;
    plus_t = site_half(result_x, result_y, result_z, result_t + 1);
  }
#if WK_SAPH
  else if (g_saph_active) {
    fb_pt = saph_ghost_tile(wk_ttile, SAPH_THI, 1 - parity, ght); plus_t = 0;
  }
#endif
  else                   { fb_pt = ghost_in; plus_t = G_OFF_THI + ght; }
  if (result_t > 0) {
    fb_mt = ferm;
    minus_t = site_half(result_x, result_y, result_z, result_t - 1);
  }
#if WK_SAPH
  else if (g_saph_active) {
    fb_mt = saph_ghost_tile(wk_ttile, SAPH_TLO, 1 - parity, ght); minus_t = 0;
  }
#endif
  else                   { fb_mt = ghost_in; minus_t = G_OFF_TLO + ght; }
  #define WK_FB_PT fb_pt
  #define WK_FB_MT fb_mt
#else
  int plus_t = site_half(result_x, result_y, result_z, (result_t + 1) % Lt);
  int minus_t = site_half(result_x, result_y, result_z,
                          (result_t + Lt - 1) % Lt);
  #define WK_FB_PT ferm
  #define WK_FB_MT ferm
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

  // prefetch y direction
  prefetch_ferm(plus_y);
  prefetch_gauge(gauge_loc, 1);
  // prefetch z direction
  prefetch_ferm(plus_z);
  prefetch_gauge(gauge_loc, 2);
  // x direction  (mask bit 0: +x)
  if (WK_ACTIVE(0)) {
    ferm_loc = plus_x;
    WK_SET_IN(ferm)
    dir = 0; // for gauge link
    #pragma unroll
    for (int r = 0; r < num_rhs_vec; r ++) {
      A_MINUS_BI(0, 3);
      A_MINUS_BI(1, 2);
    }
  }
  // prefetch t direction
  prefetch_ferm(plus_t);
  prefetch_gauge(gauge_loc, 3);
  // y direction  (mask bit 1: +y)
  if (WK_ACTIVE(1)) {
    ferm_loc = plus_y;
    WK_SET_IN(ferm)
    dir = 1;
    #pragma unroll
    for (int r = 0; r < num_rhs_vec; r ++) {
      A_OP_B(0, 3, +);
      A_OP_B(1, 2, -);
    }
  }
  // prefetch -x direction
  prefetch_ferm(minus_x);
  prefetch_gauge(gauge_minus_x, 0);
  // z direction  (mask bit 2: +z)
  if (WK_ACTIVE(2)) {
    ferm_loc = plus_z;
    WK_SET_IN(WK_FB_PZ)
    dir = 2;
    #pragma unroll
    for (int r = 0; r < num_rhs_vec; r ++) {
      A_MINUS_BI(0, 2);
      A_PLUS_BI(1, 3);
    }
  }
  // prefetch y direction
  prefetch_ferm(minus_y);
  prefetch_gauge(gauge_minus_y, 1);
  // t direction  (mask bit 3: +t)
  if (WK_ACTIVE(3)) {
    ferm_loc = plus_t;
    WK_SET_IN(WK_FB_PT)
    dir = 3;
    #pragma unroll
    for (int r = 0; r < num_rhs_vec; r ++) {
      A_OP_B(0, 2, -);
      A_OP_B(1, 3, -);
    }
  }

  // negative shifts
  // prefetch z direction
  prefetch_ferm(minus_z);
  prefetch_gauge(gauge_minus_z, 2);
  // x direction  (mask bit 4: -x)
  if (WK_ACTIVE(4)) {
    ferm_loc = minus_x;
    WK_SET_IN(ferm)
    gauge_loc = gauge_minus_x;
    dir = 0; // for gauge link
    #pragma unroll
    for (int r = 0; r < num_rhs_vec; r ++) {
      A_PLUS_BI_DAG(0, 3);
      A_PLUS_BI_DAG(1, 2);
    }
  }
  // prefetch t direction
  prefetch_ferm(minus_t);
  prefetch_gauge(gauge_minus_t, 3);
  // y direction  (mask bit 5: -y)
  if (WK_ACTIVE(5)) {
    ferm_loc = minus_y;
    WK_SET_IN(ferm)
    gauge_loc = gauge_minus_y;
    dir = 1;
    #pragma unroll
    for (int r = 0; r < num_rhs_vec; r ++) {
      A_OP_B_DAG(0, 3, -);
      A_OP_B_DAG(1, 2, +);
    }
  }
  // z direction  (mask bit 6: -z)
  if (WK_ACTIVE(6)) {
    ferm_loc = minus_z;
    WK_SET_IN(WK_FB_MZ)
    gauge_loc = gauge_minus_z;
    dir = 2;
    #pragma unroll
    for (int r = 0; r < num_rhs_vec; r ++) {
      A_PLUS_BI_DAG(0, 2);
      A_MINUS_BI_DAG(1, 3);
    }
  }
  // t direction  (mask bit 7: -t)
  if (WK_ACTIVE(7)) {
    ferm_loc = minus_t;
    WK_SET_IN(WK_FB_MT)
    gauge_loc = gauge_minus_t;
    dir = 3;
    #pragma unroll
    for (int r = 0; r < num_rhs_vec; r ++) {
      A_OP_B_DAG(0, 2, +);
      A_OP_B_DAG(1, 3, +);
    }
  }
#undef WK_SET_IN
#undef WK_FB_PZ
#undef WK_FB_MZ
#undef WK_FB_PT
#undef WK_FB_MT
