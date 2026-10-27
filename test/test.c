// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>  // mkdir (halo dump directory)
#include <errno.h>
#include "global_params.h"
#include "utils.h"
#include "clover_ferm.h"
#include "test.h"
#include "solver.h"
#include "coarse.h"
#include "mixed.h"
#include "driver.h"
#include "qc.h"     // test_qc_layout checks the published layout
                    // against this build
#include "omp.h"
#include "comm.h"   // qc_grid / print0 / allreduce_sum_d (used throughout)

// Precision banner for the solve tests.  "mixed" names the build, not the test:
// at MIXED=0 the outer solver and the whole hierarchy share one REAL, so the
// banner has to say so rather than claim a single-precision V-cycle.
#ifdef MIXED
#define PREC_HIER  "single-precision"
#define PREC_SOLVE "DOUBLE FGMRES + SINGLE V-cycle"
#else
#define PREC_HIER  (sizeof(REAL) == 8 ? "double-precision" \
                                      : "single-precision")
#define PREC_SOLVE (sizeof(REAL) == 8 ? "ALL-DOUBLE FGMRES + V-cycle"        \
                                      : "ALL-SINGLE FGMRES + V-cycle")
#endif

// Regression test for even-odd preconditioning with the clover term.  The
// even-odd-preconditioned solve and the unpreconditioned solve solve the same
// linear system D x = b, so for any csw their solutions must agree to the
// solver tolerance -- in particular this catches an eo D_oo^{-1} step that
// silently drops its result.  Uses a real gauge config and a point source
// like the correlator.
void test_eo_clover() {
  REAL kappa = 0.125;
  REAL csw   = 1.0;
  int x = 0, y = 0, z = 0, t = 0;
  int parity = (x ^ y ^ z ^ t) & 0x1;

  REAL * U = calloc(Lx * Ly * Lz * Lt * 4 * Nc * Nc * 2, sizeof(REAL));
  read_gauge(U, GAUGE_PATH);
  REAL * F = calloc(Lt * Lz * Ly * Lx * 2 * 6 * 6, sizeof(REAL));
  compute_Fmunu(U, F);

  REAL * src     = malloc(sizeof(REAL) * ferm_size * num_rhs);
  REAL * prop_eo = malloc(sizeof(REAL) * ferm_size * num_rhs);
  REAL * prop_no = malloc(sizeof(REAL) * ferm_size * num_rhs);
  for (long i = 0; i < (long) ferm_size * num_rhs; i ++) src[i] = 0;
  // point source: one entry per (spin,color) right-hand side, at the origin
  for (int c = 0; c < Nc; c ++)
    for (int d = 0; d < Ns; d ++) {
      int index = d * Nc + c;
      long loc = (long) parity * Vh + site_half(x / 2, y, z, t);
      src[(((loc * Ns + d) * Nc + c) * 2) * num_rhs + index] = 1;
    }

  solver_info solver = { .solver = richardson, .preconditioner = NULL,
                         .max_iter = 1000, .eps = 1e-6 };
  int chroma_norm = 1;
  solve(prop_eo, src, U, F, kappa, csw, chroma_norm, /*eo*/ 1, solver);
  solve(prop_no, src, U, F, kappa, csw, chroma_norm, /*eo*/ 0, solver);

  double max_abs = 0, max_mag = 0;
  for (long i = 0; i < (long) ferm_size * num_rhs; i ++) {
    double d  = fabs((double) prop_eo[i] - (double) prop_no[i]);
    double m  = fabs((double) prop_no[i]);
    if (d > max_abs) max_abs = d;
    if (m > max_mag) max_mag = m;
  }
  double rel = (max_mag > 0) ? max_abs / max_mag : 0;
  printf("test_eo_clover (csw=%.1f): max|eo-noeo| = %.3e, max|noeo| = %.3e, "
         "rel = %.3e => %s\n", csw, max_abs, max_mag, rel,
         rel < 1e-3 ? "PASS" : "FAIL");

  free(U); free(F); free(src); free(prop_eo); free(prop_no);
}

// Standalone convergence/correctness test for red_black_schwarz on the full
// (both-parity) Wilson[-clover] system D x = b (eo_precondition = 0).  Runs a
// fixed number of Schwarz sweeps and reports the relative residual
// ||b - D x|| / ||b||.  As a multiplicative Schwarz iteration it must drive the
// residual down monotonically, so this gates both the block solve's correctness
// and its convergence rate.  Set CSW below to test the clover operator.
#ifndef TEST_SCHWARZ_CSW
#define TEST_SCHWARZ_CSW 0.0
#endif
#ifndef TEST_SCHWARZ_SWEEPS
#define TEST_SCHWARZ_SWEEPS 50
#endif
void test_schwarz() {
  REAL kappa = 0.125;
  REAL csw   = TEST_SCHWARZ_CSW;
  int  clover = (csw != 0);
  int  x = 0, y = 0, z = 0, t = 0;
  int  parity = (x ^ y ^ z ^ t) & 0x1;

  REAL * U = calloc(Lx * Ly * Lz * Lt * 4 * Nc * Nc * 2, sizeof(REAL));
  read_gauge(U, GAUGE_PATH);
  long clover_size = (long) Lt * Lz * Ly * Lx * 2 * 6 * 6;
  REAL * F = calloc(clover_size, sizeof(REAL));
  if (clover) compute_Fmunu(U, F);

  // Prepared operator fields, exactly as solve() would build them at eo=0.
  // gauge_alloc_size carries the backward-link tail ghosts the block kernel
  // reads at a decomposed grid; equal to gauge_size at the 1x1 grid.
  REAL * U_kappa = malloc(sizeof(REAL) * gauge_alloc_size);
  REAL * F_csw   = malloc(sizeof(REAL) * clover_size);
  prepare_gauge(U_kappa, U, kappa);
  if (clover) prepare_F(F_csw, F, kappa, csw, /*invert_odd*/ 0);

  long n = (long) ferm_size * num_rhs;
  REAL * src = malloc(sizeof(REAL) * n);
  REAL * out = malloc(sizeof(REAL) * n);
  REAL * Dx  = malloc(sizeof(REAL) * n);
  for (long i = 0; i < n; i ++) src[i] = 0;
  // x,y,z,t above are global coordinates; under z/t decomposition only the
  // owning rank places the source, at local z,t (x,y are never decomposed).
  // Parity stays a global property, so it is computed from the global site.
  const int oZ = qc_grid.origin[2], oT = qc_grid.origin[3];
  if (z >= oZ && z < oZ + Lz && t >= oT && t < oT + Lt)
    for (int c = 0; c < Nc; c ++)
      for (int d = 0; d < Ns; d ++) {
        int index = d * Nc + c;
        long loc = (long) parity * Vh + site_half(x / 2, y, z - oZ, t - oT);
        src[(((loc * Ns + d) * Nc + c) * 2) * num_rhs + index] = 1;
      }

  int sweeps = TEST_SCHWARZ_SWEEPS;
  mg_params tp; mg_params_default(&tp);
  red_black_schwarz(out, src, U_kappa, F_csw, clover, sweeps, tp.sap_block_iter,
                    1e-12, 0, NULL);

  // relative residual ||src - D out|| / ||src||
  D(Dx, out, U_kappa, F_csw, clover);
  double rnorm = 0, bnorm = 0;
  for (long i = 0; i < n; i ++) {
    double ri = (double) src[i] - (double) Dx[i];
    rnorm += ri * ri;
    bnorm += (double) src[i] * (double) src[i];
  }
  allreduce_sum_d(&rnorm, 1);        // the loop above covers local sites only
  allreduce_sum_d(&bnorm, 1);
  double rel = sqrt(rnorm / bnorm);
  print0("test_schwarz [%s, csw=%.1f]: %d sweeps, ||b-Dx||/||b|| = %.3e => %s\n",
         clover ? "wilson+clover" : "wilson", csw, sweeps, rel,
         rel < 1e-4 ? "PASS" : "FAIL");

  free(U); free(F); free(U_kappa); free(F_csw);
  free(src); free(out); free(Dx);
}

// Bit-identicality check: build_coarse_op_direct, which evaluates only the
// single inward fine hop at each shell site, must reproduce the full-stencil
// reference build_coarse_op exactly -- self and neighbor couplings alike.
// This is the only thing standing between a mis-assembled D_c and a silent
// iteration-count regression: D_c is a preconditioner, so an error in it
// degrades convergence without ever corrupting the solve.  Inputs are
// random -- the claim holds for any nv/U/F since both builds consume the
// same arrays -- so this is self-contained (no gauge config file) and cheap
// to run anywhere.  Runs in double (test.c is the double TU in the mixed
// build); in the single TU the asm full kernel and the C masked kernel differ
// at the float-rounding level, so the bitwise claim is a double-precision one.
void test_coarse_direct() {
  printf("coarse direct build: fine %dx%dx%dx%d, block %d -> %d coarse sites, "
         "N=%d, %d vars/site\n", Lx, Ly, Lz, Lt, BLOCK_AGG, Vc,
         num_null_vecs, N_coarse);

  long nvn = (long) ferm_size * num_null_vecs;
  long clover_size = (long) Lt * Lz * Ly * Lx * 2 * 6 * 6;
  REAL * nv      = malloc(sizeof(REAL) * nvn);
  // +tail ghost for the seam pass
  REAL * U_kappa = malloc(sizeof(REAL) * gauge_alloc_size);
  REAL * F_csw   = malloc(sizeof(REAL) * clover_size);
  int clover = 1;

  // per-rank data => the seam exchanges real cross-rank values
  srand(2024 + comm_rank());
  for (long i = 0; i < nvn; i ++)
    nv[i]      = 2.0 * rand() / RAND_MAX - 1.0;
  for (long i = 0; i < gauge_alloc_size; i ++)
    U_kappa[i] = 2.0 * rand() / RAND_MAX - 1.0;
  for (long i = 0; i < clover_size; i ++)
    F_csw[i]   = 2.0 * rand() / RAND_MAX - 1.0;
  // tail backward-link ghosts (no-op at MPI=0)
  halo_gauge(U_kappa, sizeof(REAL) * 4 * Nc * Nc * 2);

  coarse_op ref = {0}, td = {0};
  build_coarse_op(&ref, nv, U_kappa, F_csw, clover);  // full-stencil reference
  build_coarse_op_direct(&td, nv, U_kappa, F_csw, clover);  // production build

  long ns = (long) Vc * N_coarse * N_coarse * 2;    // REALs / self block * Vc
  long nn = (long) Vc * 8 * N_coarse * N_coarse * 2;  // REALs / all nbr blocks

  double ms = 0, mn = 0, refs = 0, refn = 0; long ds = 0, dn = 0;
  for (long i = 0; i < ns; i ++) {
    double d = fabs((double) td.self[i] - (double) ref.self[i]);
    if (d > ms) ms = d;
    if (fabs((double) ref.self[i]) > refs) refs = fabs((double) ref.self[i]);
    if (td.self[i] != ref.self[i]) ds ++;
  }
  for (long i = 0; i < nn; i ++) {
    double d = fabs((double) td.nbr[i] - (double) ref.nbr[i]);
    if (d > mn) mn = d;
    if (fabs((double) ref.nbr[i]) > refn) refn = fabs((double) ref.nbr[i]);
    if (td.nbr[i] != ref.nbr[i]) dn ++;
  }
  printf("[self]      max|diff|=%.3e ref_max=%.3e bitdiffs=%ld/%ld => %s\n",
         ms, refs, ds, ns, ds == 0 ? "PASS" : "FAIL");
  printf("[neighbor]  max|diff|=%.3e ref_max=%.3e bitdiffs=%ld/%ld => %s\n",
         mn, refn, dn, nn, dn == 0 ? "PASS" : "FAIL");
  printf("test_coarse_direct: %s\n",
         (ds == 0 && dn == 0) ? "PASS (bit-identical)" : "FAIL");

  free_coarse_op(&ref); free_coarse_op(&td);
  free(nv); free(U_kappa); free(F_csw);
}

// Validate local_orthonormalize: after it runs, every aggregate's P columns
// must be orthonormal (P^H P = I per aggregate).  That defining property
// catches any gather/scatter indexing bug independently of the SIMD summation
// order.
void test_local_ortho() {
  printf("local_ortho: fine %dx%dx%dx%d, block %d -> %d coarse sites, N=%d\n",
         Lx, Ly, Lz, Lt, BLOCK_AGG, Vc, num_null_vecs);
  long nvn = (long) ferm_size * num_null_vecs;
  REAL * nv = malloc(sizeof(REAL) * nvn);
  srand(2024);
  for (long i = 0; i < nvn; i ++) nv[i] = 2.0 * rand() / RAND_MAX - 1.0;

  REAL err_before = nullvec_orthonormality_error(nv);
  local_orthonormalize(nv);
  REAL err_after = nullvec_orthonormality_error(nv);
  printf("test_local_ortho: ||P^H P - I||_max  %.3e -> %.3e  => %s\n",
         (double) err_before, (double) err_after,
         err_after < 1e-4 ? "PASS" : "FAIL");
  free(nv);
}

// Coarse-grid machinery on random inputs: the transfers (R = P^H, R P = I),
// the Galerkin identity D_c = R D P, the vectorized apply_Dc against its
// scalar reference, and the two depth-1 bottom solves.  Everything but the
// solves is an identity that holds for any nv/U/F -- the same argument
// test_coarse_direct makes -- so this needs no gauge config.
void test_coarse() {
  printf("coarse geometry: fine %dx%dx%dx%d, block %d -> coarse %dx%dx%dx%d "
         "(%d sites), N=%d test vecs, %d vars/coarse site\n",
         Lx, Ly, Lz, Lt, BLOCK_AGG, Lxc, Lyc, Lzc, Ltc, Vc,
         num_null_vecs, N_coarse);

  long nvn = (long) ferm_size * num_null_vecs;
  REAL * nv = malloc(sizeof(REAL) * nvn);
  srand(2024);
  for (long i = 0; i < nvn; i ++) nv[i] = 2.0 * rand() / RAND_MAX - 1.0;
  local_orthonormalize(nv);   // test_local_ortho checks this step on its own

  // --- transfers: R is the adjoint of P, and R P = I -----------------------
  long fn = (long) ferm_size * num_rhs;
  long cn = coarse_ferm_size * num_rhs;
  REAL * cin  = malloc(sizeof(REAL) * cn);   // random coarse vector c
  REAL * fw   = malloc(sizeof(REAL) * fn);   // random fine vector w
  REAL * Pc   = malloc(sizeof(REAL) * fn);   // P c
  REAL * Rw   = malloc(sizeof(REAL) * cn);   // R w
  REAL * RPc  = malloc(sizeof(REAL) * cn);   // R P c
  srand(12345);
  for (long i = 0; i < cn; i ++) cin[i] = 2.0 * rand() / RAND_MAX - 1.0;
  for (long i = 0; i < fn; i ++) fw[i]  = 2.0 * rand() / RAND_MAX - 1.0;

  apply_P(Pc, cin, nv);
  apply_R(Rw, fw, nv);
  apply_R(RPc, Pc, nv);

  // adjoint identity: real_dot(w, P c) == real_dot(R w, c)
  double lhs = 0, rhs_ = 0;
  for (long i = 0; i < fn; i ++) lhs  += (double) fw[i] * Pc[i];
  for (long i = 0; i < cn; i ++) rhs_ += (double) Rw[i] * cin[i];
  double adj_rel = fabs(lhs - rhs_) / (fabs(lhs) + fabs(rhs_) + 1e-30);
  printf("test_coarse [M2 adjoint]: <w,Pc>=%.6e <Rw,c>=%.6e rel=%.3e => %s\n",
         lhs, rhs_, adj_rel, adj_rel < 1e-4 ? "PASS" : "FAIL");

  // R P = I on the coarse space: ||R P c - c|| / ||c||
  double dn = 0, nn = 0;
  for (long i = 0; i < cn; i ++) {
    double di = (double) RPc[i] - cin[i];
    dn += di * di; nn += (double) cin[i] * cin[i];
  }
  double rp_rel = sqrt(dn / nn);
  printf("test_coarse [M2 R*P=I]: ||RPc - c||/||c|| = %.3e => %s\n",
         rp_rel, rp_rel < 1e-3 ? "PASS" : "FAIL");

  // --- coarse operator D_c = P^H D P ---------------------------------------
  // Off-diagonality of the random operator: D = I - hop - F_csw, so a small
  // scale keeps D_c conditioned.  Raising it toward 1 stalls the GMRES(30)
  // below.
  const REAL hop_scale = 0.125;   // free-field critical kappa
  int clover = 1;
  long clover_size = (long) Lt * Lz * Ly * Lx * 2 * 6 * 6;
  REAL * U_kappa = malloc(sizeof(REAL) * gauge_alloc_size);
  REAL * F_csw   = malloc(sizeof(REAL) * clover_size);
  for (long i = 0; i < gauge_alloc_size; i ++)
    U_kappa[i] = hop_scale * (2.0 * rand() / RAND_MAX - 1.0);
  for (long i = 0; i < clover_size; i ++)
    F_csw[i]   = hop_scale * (2.0 * rand() / RAND_MAX - 1.0);

  struct timeval bt0, bt1;
  gettimeofday(&bt0, NULL);
  coarse_op Dc;
  build_coarse_op(&Dc, nv, U_kappa, F_csw, clover);
  gettimeofday(&bt1, NULL);
  printf("test_coarse [M3 build]: D_c built in %.2f s (%d coarse sites, %d^2 blocks)\n",
         (bt1.tv_sec - bt0.tv_sec) + 1e-6 * (bt1.tv_usec - bt0.tv_usec),
         Vc, N_coarse);

  // verify D_c c == R (D (P c)) on a random coarse vector
  REAL * Dcc  = malloc(sizeof(REAL) * cn);   // D_c c (stencil)
  REAL * RDPc = malloc(sizeof(REAL) * cn);   // R D P c (reference)
  apply_Dc(&Dc, Dcc, cin);
  apply_P(Pc, cin, nv);
  REAL * DPc = malloc(sizeof(REAL) * fn);
  D(DPc, Pc, U_kappa, F_csw, clover);
  apply_R(RDPc, DPc, nv);
  double en = 0, rn = 0;
  for (long i = 0; i < cn; i ++) {
    double di = (double) Dcc[i] - RDPc[i];
    en += di * di; rn += (double) RDPc[i] * RDPc[i];
  }
  double dc_rel = sqrt(en / rn);
  printf("test_coarse [M3 D_c=RDP]: ||D_c c - R D P c||/||R D P c|| = %.3e => %s\n",
         dc_rel, dc_rel < 1e-4 ? "PASS" : "FAIL");

  // hand-vectorized apply_Dc (in Dcc) vs scalar apply_Dc_ref, same input
  REAL * Dcref = malloc(sizeof(REAL) * cn);
  apply_Dc_ref(&Dc, Dcref, cin);
  double ev = 0, rv2 = 0, maxabs = 0;
  for (long i = 0; i < cn; i ++) {
    double di = (double) Dcc[i] - Dcref[i];
    ev += di * di; rv2 += (double) Dcref[i] * Dcref[i];
    if (fabs(di) > maxabs) maxabs = fabs(di);
  }
  double v_rel = sqrt(ev / rv2);
  printf("test_coarse [M3b vec apply_Dc]: ||Dc_vec - Dc_ref||/||Dc_ref|| = %.3e, "
         "max|diff| = %.3e => %s\n", v_rel, maxabs,
         v_rel < 1e-5 ? "PASS" : "FAIL");
  free(Dcref);

  // --- coarse GMRES solves D_c x = b ---------------------------------------
  REAL * cx  = malloc(sizeof(REAL) * cn);    // solution
  REAL * cDx = malloc(sizeof(REAL) * cn);    // D_c x
  double bnorm = 0;
  for (long i = 0; i < cn; i ++) bnorm += (double) cin[i] * cin[i];
  bnorm = sqrt(bnorm);
  struct timeval ct0, ct1;
  gettimeofday(&ct0, NULL);
  int cits = coarse_solve_full(&Dc, cx, cin, 1000, 1e-6 * bnorm, 30);
  gettimeofday(&ct1, NULL);
  apply_Dc(&Dc, cDx, cx);
  double crr = 0;
  for (long i = 0; i < cn; i ++) {
    double di = (double) cin[i] - cDx[i];
    crr += di * di;
  }
  double c_rel = sqrt(crr) / bnorm;
  printf("test_coarse [M4 coarse GMRES]: %d iters in %.2f s, "
         "||b - D_c x||/||b|| = %.3e => %s\n",
         cits, (ct1.tv_sec - ct0.tv_sec) + 1e-6 * (ct1.tv_usec - ct0.tv_usec),
         c_rel, c_rel < 1e-5 ? "PASS" : "FAIL");

  // --- even-odd (Schur complement) coarse solve ----------------------------
  // Same system, solved through the odd-site Schur complement.  The recovered
  // full solution must satisfy D_c x = b to the same tolerance, and the inner
  // (odd-only) GMRES converges in fewer iterations than the full solve above.
  coarse_eo_op eo;
  build_coarse_eo_op(&eo, &Dc, NULL);   // NULL lv: depth-1 macro geometry
  REAL * cxe  = malloc(sizeof(REAL) * cn);
  REAL * cDxe = malloc(sizeof(REAL) * cn);
  gettimeofday(&ct0, NULL);
  int cits_eo = coarse_solve_eo(&eo, cxe, cin, 1000, 1e-6 * bnorm, 30);
  gettimeofday(&ct1, NULL);
  apply_Dc(&Dc, cDxe, cxe);
  double crre = 0;
  for (long i = 0; i < cn; i ++) {
    double di = (double) cin[i] - cDxe[i];
    crre += di * di;
  }
  double c_rele = sqrt(crre) / bnorm;
  printf("test_coarse [M4b eo Schur GMRES]: %d iters in %.2f s, "
         "||b - D_c x||/||b|| = %.3e => %s\n", cits_eo,
         (ct1.tv_sec - ct0.tv_sec) + 1e-6 * (ct1.tv_usec - ct0.tv_usec),
         c_rele, c_rele < 1e-5 ? "PASS" : "FAIL");
  free_coarse_eo_op(&eo);
  free(cxe); free(cDxe);

  free(cx); free(cDx);
  free(Dcc); free(RDPc); free(DPc);
  free_coarse_op(&Dc);
  free(U_kappa); free(F_csw);
  free(cin); free(fw); free(Pc); free(Rw); free(RPc);
  free(nv);
}


// Isolated apply_Dc throughput benchmark: a tight loop of the coarse Dslash
// on a fixed coarse vector, so a whole-program profiler sees ~100% kernel.
// Reports GFlop/s via apply_dc_stats.  (apply_Dc dominates coarse_solve_full,
// which dominates the V-cycle solve.)
void test_apply_dc_bench() {
  // Throughput only -- no solve, so the operator need not be physical or even
  // well conditioned; D_c just has to have the right shape and be nonzero.
  long nvn = (long) ferm_size * num_null_vecs;
  REAL * nv = malloc(sizeof(REAL) * nvn);
  srand(2024);
  for (long i = 0; i < nvn; i ++) nv[i] = 2.0 * rand() / RAND_MAX - 1.0;
  local_orthonormalize(nv);

  coarse_op Dc;
  {
    long clover_size = (long) Lt * Lz * Ly * Lx * 2 * 6 * 6;
    REAL * U_kappa = malloc(sizeof(REAL) * gauge_alloc_size);
    REAL * F_csw   = malloc(sizeof(REAL) * clover_size);
    for (long i = 0; i < gauge_alloc_size; i ++)
      U_kappa[i] = 2.0 * rand() / RAND_MAX - 1.0;
    for (long i = 0; i < clover_size; i ++)
      F_csw[i]   = 2.0 * rand() / RAND_MAX - 1.0;
    build_coarse_op(&Dc, nv, U_kappa, F_csw, /*clover*/ 1);
    free(U_kappa); free(F_csw);
  }

  long cn = coarse_ferm_size * num_rhs;
  REAL * x = malloc(sizeof(REAL) * cn);
  REAL * y = malloc(sizeof(REAL) * cn);
  srand(11);
  for (long i = 0; i < cn; i ++) x[i] = 2.0 * rand() / RAND_MAX - 1.0;

  const int NREP = 1000;
  apply_Dc(&Dc, y, x);                 // warm up
  apply_dc_stats_reset();
  for (int r = 0; r < NREP; r ++) apply_Dc(&Dc, y, x);
  printf("--- isolated apply_Dc throughput (%d reps) ---\n", NREP);
  apply_dc_stats_report("bench");

  free(x); free(y); free(nv); free_coarse_op(&Dc);
}

// Mixed-precision end-to-end: a double outer FGMRES preconditioned by a
// single-precision MG V-cycle (via mixed_bridge.c).  This is the deliverable
// configuration -- it converges to double tolerance (which a pure-single solve
// cannot) while paying single's cheaper preconditioner cost.  Compare the final
// true residual ||b - D x|| / ||b|| against the requested (tight) tolerance.
void test_mixed() {
  REAL kappa = 0.134, csw = 1.87567;   // REAL == double in this driver TU
  int clover = 1;

  // Coarse-solve config (MG_COARSE_EO, COARSE_USE_ASM) is hard-coded in
  // global_params.h.

  // gauge + prepared double fine operator fields (the outer operator D)
  REAL * U = calloc((long) Lx * Ly * Lz * Lt * 4 * Nc * Nc * 2, sizeof(REAL));
  read_gauge(U, GAUGE_PATH);
  long clover_size = (long) Lt * Lz * Ly * Lx * 2 * 6 * 6;
  REAL * F = calloc(clover_size, sizeof(REAL));
  compute_Fmunu(U, F);
  // gauge_alloc_size, not gauge_size: U_kappa carries the backward-link tail
  // ghosts a decomposed solve reads, filled by prepare_gauge.  Equal at the 1x1
  // grid.  The raw U above needs none -- the ghosts arrive with the exchange.
  REAL * U_kappa = malloc(sizeof(REAL) * gauge_alloc_size);
  REAL * F_csw   = malloc(sizeof(REAL) * clover_size);
  prepare_gauge(U_kappa, U, kappa);
  prepare_F(F_csw, F, kappa, csw, /*invert_odd*/ 0);

  // Build the MG hierarchy from the driver's prepared operator fields.
  print0("test_mixed: building %s MG hierarchy (multilevel=%d)\n", PREC_HIER,
         MULTILEVEL);
  // The tuned defaults, unmodified.  Not a designated initializer: that would
  // zero every knob not named, and a zeroed sap_block_iter leaves the smoother
  // doing nothing at all.
  mg_params mgp_setup; mg_params_default(&mgp_setup);
  mg_hierarchy_build(U_kappa, F_csw, clover, &mgp_setup);

  // random multi-rhs right-hand side (double)
  long n = (long) ferm_size * num_rhs;
  REAL * b  = malloc(sizeof(REAL) * n);
  REAL * xf = malloc(sizeof(REAL) * n);
  REAL * Dx = malloc(sizeof(REAL) * n);
  // per-rank data, so a decomposed b is not a periodic repeat
  srand(7 + comm_rank());
  for (long i = 0; i < n; i ++) b[i] = 2.0 * rand() / RAND_MAX - 1.0;

  // outer FGMRES preconditioned by one MG V-cycle.
  // The preconditioner apply.  In a mixed build fgmres hard-routes to the
  // bridge's float entry point and never reads .solver -- only the non-NULL
  // solver_info itself matters -- so leaving it NULL is the honest statement.
  // In a one-precision build fgmres dispatches .solver, and mg_precond is the
  // entry that runs the cycle on the hierarchy's own stored operator (not
  // mg_vcycle directly -- fgmres would hand it the outer operator's fields).
#ifdef MIXED
  solver_info mgp = { .solver = NULL,
                      .preconditioner = NULL, .max_iter = 0, .eps = 0 };
#else
  solver_info mgp = { .solver = mg_precond,
                      .preconditioner = NULL, .max_iter = 0, .eps = 0 };
#endif
  drv_solve_params vp; drv_solve_params_default(&vp);
  REAL outer_tol = vp.tol;
  print0("\n--- test_mixed: %s "
         "(nu=%d, restart=%d, tol=%.1e) ---\n",
         PREC_SOLVE, mgp_setup.smoother_sweeps, vp.restart, (double) outer_tol);
  int it = fgmres(xf, b, U_kappa, F_csw, clover, vp.max_iter, vp.restart,
                  vp.tol, 0, &mgp);

  D(Dx, xf, U_kappa, F_csw, clover);
  // Worst-rhs relative residual.  fgmres gates per rhs, so the check has to
  // as well: an aggregate norm here passes on the block average and cannot
  // see a single under-solved rhs, which is precisely the failure the gate
  // prevents.  rhs is the fastest index, so i % num_rhs picks the lane in
  // both the real and the imaginary half of each complex element.
  double rn[num_rhs], bn[num_rhs];
  for (int k = 0; k < num_rhs; k ++) { rn[k] = 0; bn[k] = 0; }
  for (long i = 0; i < n; i ++) {
    const int k = i % num_rhs;
    const double d = (double) b[i] - Dx[i];
    rn[k] += d * d;
    bn[k] += (double) b[i] * b[i];
  }
  allreduce_sum_d(rn, num_rhs);      // the loop above covers local sites only
  allreduce_sum_d(bn, num_rhs);
  double r = 0;
  for (int k = 0; k < num_rhs; k ++)
    if (bn[k] > 0 && sqrt(rn[k] / bn[k]) > r) r = sqrt(rn[k] / bn[k]);
  print0("\ntest_mixed [MIXED]: %d outer iters, worst-rhs rel res %.3e (tol %.1e) "
         "=> %s\n", it, r, (double) outer_tol,
         (r < outer_tol * 10) ? "PASS (mixed reaches double tol)" : "CHECK");

  mg_hierarchy_free();
  free(U); free(F); free(U_kappa); free(F_csw);
  free(b); free(xf); free(Dx);
}

// Project a propagator onto zero momentum, with gamma_5 at both source and
// sink.  In that channel gamma_5-hermiticity collapses the whole contraction
// to the propagator norm on each timeslice -- no gamma matrix appears below,
// and the result is real, so the correlator carries one entry per timeslice
// rather than a complex pair.  The pion is the one correlator this works for.
//
// The local loop covers this rank's z/t slab; results are written at the
// global timeslice tg = t + origin_t (so each rank fills only the entries it
// owns) and the spatial sum is partial in z (other z-ranks hold the rest).
// A single sum-allreduce over the full Lt_glob array below closes both
// decompositions at once: z-rank partials add, and a timeslice not owned by
// a t-rank stays 0.
static void project_pion_zero_momentum(REAL * prop, REAL * correlator) {
  const int oT = qc_grid.origin[3];
  #pragma omp parallel for
  for (int t = 0; t < Lt; t ++) {
    int tg = t + oT;
    for (int parity = 0; parity < 2; parity ++)
      for (int z = 0; z < Lz; z ++)
        for (int y = 0; y < Ly; y ++)
          for (int x = 0; x < Lxh; x ++) {
            // x is the half-lattice index; the physical coordinate is 2*x
            // plus a parity-dependent offset, which is what a momentum phase
            // would need
            long loc = (long) parity * Vh + site_half(x, y, z, t);
            REAL sum = 0;
            // Spin outside color, matching every other consumer and the order
            // qc.h publishes.  The two orders happen to give the same answer
            // here -- the sum runs over all twelve, so permuting them cancels
            // -- but this was the one place in the tree spelling it the other
            // way.
            for (int s = 0; s < Ns; s ++) {
              for (int c = 0; c < Nc; c ++) {
                long index = ((loc * Ns + s) * Nc + c) * 2;
                for (int r = 0; r < num_rhs; r ++) {
                  REAL re = prop[index * num_rhs + r];
                  REAL im = prop[(index + 1) * num_rhs + r];
                  sum += re * re + im * im;
                }
              }
            }
            correlator[tg] += sum;
          }
  }
  // gather: spatial-z partials add, unowned timeslices were left 0;
  // no-op @1 rank
  allreduce_sum_real(correlator, Lt_glob);
}

// The operator both correlator tests solve: the same kappa, csw, APBC in time
// and gauge configuration the reference C_ref was produced with.
#define CORR_KAPPA 0.134
#define CORR_CSW   1.87567

// Reference correlator (sum over all 12 spin-color sources), indexed by
// global timeslice.  It is specific to one gauge configuration on one volume,
// so the guard below must name the whole geometry and not just its length: a
// build of a different volume gets the zeroed table (=> FAIL) rather than a
// comparison against numbers that never described it.  Add a reference when
// you retune.  It is an independent reference, so any change that perturbs
// the propagator -- decomposition, overlap, precision -- shows up here.
#if Lx == 16 && Ly == 16 && Lz_glob == 16 && Lt_glob == 32
// Computed with QUDA at solve tolerance 1e-15.
static const double C_ref[Lt_glob] = {
    1.453629985773243e+00, 2.417741838781490e-01, 9.984487101659226e-02,
    5.451182075463618e-02, 3.293214703960619e-02, 2.092870012086846e-02,
    1.266552019785261e-02, 8.605454382994690e-03, 6.297090025416110e-03,
    4.975062495433238e-03, 3.990113467302426e-03, 3.189032960915923e-03,
    2.380503765487864e-03, 1.856954704885003e-03, 1.373547258902175e-03,
    1.148418569314329e-03, 1.134295239714202e-03, 1.220288851742905e-03,
    1.450699366717844e-03, 1.930406228893200e-03, 2.561580252430510e-03,
    3.447946242854714e-03, 4.922270774380474e-03, 6.909103747518729e-03,
    9.172240624883249e-03, 1.175895031427815e-02, 1.595731046054714e-02,
    2.172894017012313e-02, 2.916418710448759e-02, 4.626394427927510e-02,
    8.308295882753299e-02, 2.352742250755595e-01 };
#else
static const double C_ref[Lt_glob] = {0};   // no reference for this volume
#endif

// Timeslice-by-timeslice comparison against C_ref, shared by both correlator
// tests.  The source sits at t=0 with p=0, so the correlator is purely real
// and directly indexed by timeslice -- no shift or APBC sign flip.  bar is
// the caller's gate; both callers set it to 1000x the tolerance they asked
// their solver for.  True for NaN and for both infinities.  Two layers, and
// both are needed here.
//
// The test is written against the bits because the usual spellings are unusable
// in this build: -ffast-math implies -ffinite-math-only, under which the
// compiler may assume no NaN or Inf exists and folds isnan(v), v != v and
// isfinite(v) to a constant.
//
// The volatile round-trip is what makes the bit test survive.  Reading the
// exponent is not on its own enough: a value arriving straight from an FP
// operation carries that same no-NaN assumption on the operation itself, and it
// propagates through the bit reinterpretation, so the check folds away just as
// the arithmetic ones do.  Forcing the value out to memory and back leaves the
// optimizer with an ordinary load it knows nothing about.
static int nonfinite(double v) {
  volatile double vv = v;
  double w = vv;
  uint64_t u;
  memcpy(&u, &w, sizeof u);
  return ((u >> 52) & 0x7ff) == 0x7ff;
}

static void report_correlator(const char * name, const REAL * correlator,
                              double bar) {
  print0("\n   t           qc C[t]    reference C[t]  rel. diff\n");
  double max_rel = 0;
  int bad = 0;
  for (int tau = 0; tau < Lt_glob; tau ++) {
    double qc = correlator[tau];
    double ref = C_ref[tau];
    double rel = fabs(qc - ref) / fabs(ref);
    // Counted rather than maxed: a NaN loses every comparison, so folding it
    // into max_rel would leave the max at 0 and report a NaN solve as a
    // perfect one.  Testing rel rather than qc also covers a zero reference,
    // which would make the ratio non-finite from finite inputs.
    if (nonfinite(rel)) bad ++;
    else if (rel > max_rel) max_rel = rel;
    print0("  %2d   %15.6e   %15.6e   %.2e\n", tau, qc, ref, rel);
  }
  if (bad)
    print0("\n%s: %d of %d correlator entries are NaN or Inf => FAIL\n",
           name, bad, Lt_glob);
  else
    print0("\n%s: max relative diff vs reference = %.2e (bar %.1e = 1000x tol) "
           "=> %s\n", name, max_rel, bar, max_rel < bar ? "PASS" : "FAIL");
}

// Physics integration test: compute the zero-momentum pion correlator and
// compare it against C_ref.  The 12 spin-color point sources go into the
// first 12 rhs lanes and the whole batch is momentum-projected (the other
// lanes are blanked first).  The reference uses the convention
// D = (1/2kappa) D_kappa, so its solution is 2*kappa*D_kappa^{-1} b; we
// apply the same 2*kappa (chroma_norm) rescale, and the propagators -- hence
// the correlators -- match directly.  The MG hierarchy is the mixed
// (double-outer / single-V-cycle) one.

void test_pion_correlator() {
  REAL kappa = CORR_KAPPA, csw = CORR_CSW;
  int clover = 1;
  int x = 0, y = 0, z = 0, t = 0;

  // raw gauge configuration; drv_setup derives the field strength and the
  // prepared operator fields from it and owns them thereafter.
  REAL * U = calloc((long) Lx * Ly * Lz * Lt * 4 * Nc * Nc * 2, sizeof(REAL));
  read_gauge(U, GAUGE_PATH);

  // The tuned defaults, unmodified: this test is the regression bar for the
  // method the repo ships, so it must not name a knob explicitly -- a value
  // repeated here would go stale the next time one is retuned.  DDalphaAMG's
  // setup (inv_iter_inv_fcycle) refines each test vector with a K-cycle
  // (g.kcycle=1) at its setup-time K-cycle tol (set_kcycle_tol, pinned to
  // coarse_tol); the solve then uses the looser solve-time tol.  setup_k_tol /
  // solve_k_tol mirror that, both threaded through mg_cycle's k_tol argument
  // (bootstrap_sweep passes the first, mg_vcycle the second), so there is no
  // global to save/restore here and the setup K-cycle matches DDalphaAMG's.
  drv_setup_params sp;
  sp.kappa = kappa; sp.csw = csw;
  mg_params_default(&sp.mg);

  // Printed from sp.mg rather than from the macros, so this line reports the
  // method that actually ran instead of the one the defaults were meant to
  // give.
  const char * cyc = sp.mg.cycle == MG_CYCLE_W ? "W" :
                     sp.mg.cycle == MG_CYCLE_K ? "K" : "V";
  // bottom_solve reads none of the coarse_* controls once the dense inverse
  // is built.  MULTILEVEL is in the test because only mg_add_coarse_level
  // builds the inverse and a 2-level hierarchy never calls it; a failed
  // build aborts.
#if BOTTOM_DENSE_INV && MULTILEVEL
  const char * bottom = "bottom dense inverse";
#else
  char bottom_buf[80];
  snprintf(bottom_buf, sizeof bottom_buf,
           "coarse_max_iter=%d, coarse_tol=%.1e, coarse_restart=%d",
           sp.mg.coarse_max_iter, sp.mg.coarse_tol, sp.mg.coarse_restart);
  const char * bottom = bottom_buf;
#endif
  print0("test_pion_correlator: building %s MG hierarchy "
         "(multilevel=%d, cycle=%s, k_iters=%d, setup k_tol=%.2e, %s)\n",
         PREC_HIER, MULTILEVEL, cyc, sp.mg.k_iters, sp.mg.setup_k_tol, bottom);
  drv_handle * qh = drv_setup(U, &sp);
  // The raw gauge is no longer needed (drv_setup holds the prepared fields
  // and the hierarchy); free it now so it does not add to the peak when the
  // double outer FGMRES allocates its Krylov basis.
  free(U); U = NULL;

  print0("test_pion_correlator: cycle = %s (k_iters=%d, solve k_tol=%.2e)\n",
         cyc, sp.mg.k_iters, sp.mg.solve_k_tol);

  // Sources: every rhs lane carries a real fermion, like production (an
  // all-lanes-packed batch; exactly-zero lanes would only exercise the
  // solver's breakdown guards).  Lane index = 12*p + (d*Nc + c): the 12
  // spin-color unit point sources of source position p from src_pos[].
  // Position 0 is (x,y,z,t) -- the correlator reference source; the
  // correlator below is computed from lanes 0-11 only (the other lanes are
  // blanked before the projection).
  long n = (long) ferm_size * num_rhs;
  REAL * src  = malloc(sizeof(REAL) * n);
  REAL * prop = malloc(sizeof(REAL) * n);
  // nsrc (test-specific, kept here): how many lanes to fill.
  // num_rhs = production-like (all lanes packed); 12 = the
  // single-point-source form; 1 isolates the single-rhs outer-iteration
  // count (the correlator only matches C_ref when nsrc >= 12).
  int nsrc = num_rhs;
  // Global source coordinates, position 0 = the correlator source; table
  // covers num_rhs <= 48 lanes (lanes beyond it stay zero).
  const int src_pos[4][4] = {
    { x, y, z, t },
    { Lx / 2,       Ly / 2,       Lz_glob / 2,       Lt_glob / 2 },
    { Lx / 4,       (3 * Ly) / 4, Lz_glob / 4,       (3 * Lt_glob) / 4 },
    { (3 * Lx) / 4, Ly / 4,       (3 * Lz_glob) / 4, Lt_glob / 4 },
  };
  for (long i = 0; i < n; i ++) src[i] = 0;
  // src_pos holds global coordinates; under z/t decomposition only the owning
  // rank places each source, at local z,t (x,y never decomposed).
  const int oZ = qc_grid.origin[2], oT = qc_grid.origin[3];
  for (int index = 0; index < num_rhs && index < nsrc; index ++) {
    int p = index / (Ns * Nc), sc = index % (Ns * Nc);
    int d = sc / Nc, c = sc % Nc;
    if (p >= 4) break;             // past the position table: lanes stay zero
    int sx = src_pos[p][0], sy = src_pos[p][1],
        sz = src_pos[p][2], st = src_pos[p][3];
    int par = (sx ^ sy ^ sz ^ st) & 0x1;
    if (sz < oZ || sz >= oZ + Lz || st < oT || st >= oT + Lt) continue;
    int zl = sz - oZ, tl = st - oT;
    long loc = (long) par * Vh + site_half(sx / 2, sy, zl, tl);
    src[(((loc * Ns + d) * Nc + c) * 2) * num_rhs + index] = 1;
  }
  print0("test_pion_correlator: %d source lane(s) packed (12 spin-colors per "
         "position, %d position(s))\n", nsrc, (nsrc + Ns * Nc - 1) / (Ns * Nc));

  // outer FGMRES preconditioned by one MG V-cycle.  Under OUTER_EO drv_solve
  // owns the eo/Schur wrapping -- forming the even-site right-hand side and
  // back-substituting the odd sites around the solve.
  // chroma_norm off: the true-residual check below has to see the unscaled
  // solution.  The 2*kappa rescale happens after it, before the projection.
  drv_solve_params vp; drv_solve_params_default(&vp);
  REAL outer_tol = vp.tol;
  print0("\n--- test_pion_correlator: %s "
         "(nu=%d, restart=%d, tol=%.1e), %d packed point-source rhs ---\n",
         PREC_SOLVE, sp.mg.smoother_sweeps, vp.restart, (double) outer_tol,
         nsrc);
#if OUTER_EO
  print0("test_pion_correlator: OUTER_EO=1 (even-site Schur outer solve, "
         "Krylov vectors halved)\n");
#endif
  struct timeval s0, e0;
  gettimeofday(&s0, NULL);
  int it = drv_solve(qh, prop, src, &vp);
  gettimeofday(&e0, NULL);

  // true residual + iteration report
  REAL * Dx = malloc(sizeof(REAL) * n);
  drv_apply_D(qh, Dx, prop);
  // Per-rhs, not aggregate: fgmres gates every rhs against its own ||b_k||
  // now, so an aggregate ratio here would average away exactly the single
  // under-solved rhs the gate exists to catch.  rhs is the fastest index, so
  // i % num_rhs is the lane.  The best-vs-worst spread is the direct read on
  // what the shared Krylov polynomial costs the laggard lanes -- i.e. what a
  // per-rhs Krylov would recover.
  double rn[num_rhs], bn[num_rhs];
  for (int k = 0; k < num_rhs; k ++) { rn[k] = 0; bn[k] = 0; }
  for (long i = 0; i < n; i ++) {
    const int k = i % num_rhs;
    const double dd = (double) src[i] - Dx[i];
    rn[k] += dd * dd;
    bn[k] += (double) src[i] * src[i];
  }
  // the sums above are this rank's local slab; gather before forming any ratio
  {
    double rb[2 * num_rhs];
    for (int k = 0; k < num_rhs; k ++) {
      rb[k] = rn[k];
      rb[num_rhs + k] = bn[k];
    }
    allreduce_sum_d(rb, 2 * num_rhs);
    for (int k = 0; k < num_rhs; k ++) {
      rn[k] = rb[k];
      bn[k] = rb[num_rhs + k];
    }
  }
  // Aggregate ||b||^2 for the parity/timeslice breakdown below, which asks
  // where the residual sits rather than which rhs owns it, and so wants the
  // whole-block normalization.  Summed from the per-rhs partials -- already
  // global, no second pass and no second allreduce.
  double bnorm = 0;
  for (int k = 0; k < num_rhs; k ++) bnorm += bn[k];
  double rmax = 0, rmin = 0;
  int nact = 0;
  for (int k = 0; k < num_rhs; k ++) {
    if (bn[k] <= 0) continue;           // unused lane (nsrc may be < num_rhs)
    const double rk = sqrt(rn[k] / bn[k]);
    if (nact == 0 || rk > rmax) rmax = rk;
    if (nact == 0 || rk < rmin) rmin = rk;
    nact ++;
  }
  print0("test_pion_correlator: %d outer FGMRES iters (batched over %d rhs), "
         "worst-rhs ||b-Dx||/||b|| = %.3e "
         "(best %.3e over %d active rhs), %.2f s\n",
         it, nsrc, rmax, rmin, nact,
         (e0.tv_sec - s0.tv_sec) + 1e-6 * (e0.tv_usec - s0.tv_usec));

  // QC_RESID_BREAKDOWN=1: split the true residual by parity and by global
  // timeslice.  Multi-rank divergence localizer: the eo outer solve monitors
  // only the even-half (Schur) residual, so an odd-only excess indicts the
  // x_o reconstruct; excess confined to rank-boundary slices indicts a ghost
  // chain; a flat profile indicts a layout/scale-class bug.
  if (getenv("QC_RESID_BREAKDOWN")) {
    long sitesz = (long) Nc * Ns * num_rhs * 2;
    // Bind first: the Vh macro is an unparenthesized product
    // ((Lx/2) * Ly * Lz * Lt), so an inline `site / Vh` parses as
    // (site / (Lx/2)) * Ly * Lz * Lt.
    const long vh = Vh;
    double rp[2] = { 0, 0 };
    double * rt = calloc(Lt_glob, sizeof(double));
    for (long i = 0; i < n; i ++) {
      double dd = (double) src[i] - Dx[i];
      long site = i / sitesz;
      int  par  = (int) (site / vh);             // eo-packed: even sites lead
      long rem  = site - (long) par * vh;
      int  xdec, ydec, zdec, tloc;               // decode the storage index
      site_half_coords(rem, &xdec, &ydec, &zdec, &tloc);
      rp[par] += dd * dd;
      rt[qc_grid.origin[3] + tloc] += dd * dd;
    }
    allreduce_sum_d(rp, 2);
    allreduce_sum_d(rt, Lt_glob);
    print0("resid breakdown: ||r_even||/||b|| = %.3e   ||r_odd||/||b|| = %.3e\n",
           sqrt(rp[0] / bnorm), sqrt(rp[1] / bnorm));
    for (int t = 0; t < Lt_glob; t ++)
      print0("  ||r[t=%2d]||/||b|| = %.3e\n", t, sqrt(rt[t] / bnorm));
    free(rt);
    // Spin-structure dump (lane 0) at global sites (0,0,0,t) straddling a
    // rank face, plus the source contact site (0,0,0,0): prop, src, and
    // r = src - Dx.  Diffing these against a 1-rank run's identical print
    // names a corruption's structure (sign/chirality/swap) directly.  The
    // grid header pins down which compiled geometry the binary actually
    // carries (stale-binary check), and the src line separates a corrupted b
    // from a corrupted x.
    printf("[sitedump r%d] grid: P=(%d,%d) origin_t=%d Lt=%d DECOMP_Z=%d DECOMP_T=%d\n",
           comm_rank(), qc_grid.P[2], qc_grid.P[3], qc_grid.origin[3], Lt,
           (int) DECOMP_Z, (int) DECOMP_T);
    const int tgs[] = { 0, 22, 23, 24, 25 };
    for (int itg = 0; itg < 5; itg ++) {
      int tg = tgs[itg];
      int tl = tg - qc_grid.origin[3];
      if (tl < 0 || tl >= Lt) continue;            // not this rank's slice
      int parity = tg & 1;                         // site (0,0,0,tg), x=y=z=0
      long loc  = (long) parity * Vh + site_half(0, 0, 0, tl);
      long base = loc * sitesz;
      printf("[sitedump r%d] t=%2d prop:", comm_rank(), tg);
      for (int sc = 0; sc < 12; sc ++)
        printf(" %+.6e%+.6ei", prop[base + (long)(sc * 2) * num_rhs],
                               prop[base + (long)(sc * 2 + 1) * num_rhs]);
      printf("\n[sitedump r%d] t=%2d src:", comm_rank(), tg);
      for (int sc = 0; sc < 12; sc ++)
        printf(" %+.6e%+.6ei", src[base + (long)(sc * 2) * num_rhs],
                               src[base + (long)(sc * 2 + 1) * num_rhs]);
      printf("\n[sitedump r%d] t=%2d resid:", comm_rank(), tg);
      for (int sc = 0; sc < 12; sc ++)
        printf(" %+.6e%+.6ei", src[base + (long)(sc * 2) * num_rhs]
                               - Dx[base + (long)(sc * 2) * num_rhs],
                               src[base + (long)(sc * 2 + 1) * num_rhs]
                               - Dx[base + (long)(sc * 2 + 1) * num_rhs]);
      printf("\n");
    }
#if FERM_GHOSTS && DECOMP_T
    // Dump the t-hi ghost this rank actually receives for the even-input half
    // at face site (x=0,y=0,z=0) -- byte-for-byte this must equal the t-hi
    // neighbor's "t=<origin_t+Lt> prop" line above (its local-t=0 even face).
    // A mismatch convicts the host halo delivery; a match plus a large
    // boundary resid convicts the x values at unsampled sites.  Fresh
    // exchange into a private scratch, so no dependence on clover_ferm.c's
    // internal ghost state.
    {
      REAL * gsc = malloc(sizeof(REAL) * (long) GHOST_SITES * sitesz);
      halo_ferm_half(prop, gsc, sitesz * sizeof(REAL));  // even half leads prop
      long gbase = (long) G_OFF_THI * sitesz;
      int tghost = (qc_grid.origin[3] + Lt) % Lt_glob;
      printf("[sitedump r%d] thi-ghost(even, face site 0,0,0 = global t=%d):",
             comm_rank(), tghost);
      for (int sc = 0; sc < 12; sc ++)
        printf(" %+.6e%+.6ei", gsc[gbase + (long)(sc * 2) * num_rhs],
                               gsc[gbase + (long)(sc * 2 + 1) * num_rhs]);
      printf("\n");
      free(gsc);
    }
#endif
  }

  // performance summary: where the solve time went and how efficiently each
  // kernel ran
  fgmres_profile_t fgp;
  fgmres_get_profile(&fgp);
  mg_profile_report(&fgp, clover);

  // chroma_norm: match the reference convention,
  // solution = 2*kappa * D_kappa^{-1} b
  rescale_multi_ferm(prop, 2 * kappa, num_rhs);

  // C_ref assumes only the position-0 point source, but the projection below
  // sums |prop|^2 over all num_rhs lanes -- blank the lanes whose sources
  // sit elsewhere (lane = fastest index; zeroed lanes contribute nothing;
  // prop is dead after).
  #pragma omp parallel for
  for (long i = 0; i < n; i ++)
    if (i % num_rhs >= Ns * Nc) prop[i] = 0;

  // zero-momentum pion correlator: sum_{r,c,s} |prop|^2 per timeslice.
  // Indexed by global timeslice (length Lt_glob); the projection gathers
  // across ranks.
  REAL * correlator = malloc(sizeof(REAL) * Lt_glob);
  for (int tt = 0; tt < Lt_glob; tt ++) correlator[tt] = 0;
  project_pion_zero_momentum(prop, correlator);

  // Gate at 1000x the requested tolerance.  The correlator tracks the solve's
  // relative residual to within about a decade, so this is a real bound on
  // the propagator rather than a formality; at the tightest tolerances
  // C_ref's own error becomes the floor instead, and a smaller multiple
  // would leave no headroom there.
  report_correlator("test_pion_correlator", correlator, 1e3 * vp.tol);

  drv_teardown(qh);
  free(src); free(prop); free(Dx); free(correlator);
}

// The same correlator through the pre-multigrid solver stack: even-odd
// preconditioned Richardson with BiCGStab as its inner solver, which is what
// production used on these lattices before the hierarchy existed.  Checking
// it against the same C_ref is what keeps solve(), richardson()'s
// preconditioner path and bicgstab() exercised -- bicgstab has no other
// caller in the tree.
//
// Only lanes 0-11 carry sources (one point source, its twelve spin-colors);
// the rest stay zero and contribute nothing to the projection, so unlike the
// MG test above there is no blanking pass before it.
#ifdef MIXED
// solver_function-shaped wrapper around the single-precision inner solve,
// which takes void * for the preconditioner it ignores and so cannot be
// installed in a .solver slot itself.  REAL is double in this TU, matching
// the seam signature.
static int bicgstab_inner_single(REAL * out, REAL * in, REAL * U, REAL * F,
                                 int clover, int max_iter, REAL eps,
                                 int eo_precondition, solver_info * prec) {
  (void) prec;
  return bicgstab_mixed(out, in, U, F, clover, max_iter, eps,
                        eo_precondition, NULL);
}
#endif

void test_pion_correlator_bicgstab() {
  REAL kappa = CORR_KAPPA, csw = CORR_CSW;
  int x = 0, y = 0, z = 0, t = 0;
  int parity = (x ^ y ^ z ^ t) & 0x1;

  REAL * U = calloc((long) Lx * Ly * Lz * Lt * 4 * Nc * Nc * 2, sizeof(REAL));
  read_gauge(U, GAUGE_PATH);
  REAL * F = calloc((long) Lt * Lz * Ly * Lx * 2 * 6 * 6, sizeof(REAL));
  compute_Fmunu(U, F);

  long n = (long) ferm_size * num_rhs;
  REAL * src  = malloc(sizeof(REAL) * n);
  REAL * prop = malloc(sizeof(REAL) * n);
  for (long i = 0; i < n; i ++) src[i] = 0;
  // (x,y,z,t) are global; under z/t decomposition only the owning rank places
  // the source, at local z,t (x,y are never decomposed).
  const int oZ = qc_grid.origin[2], oT = qc_grid.origin[3];
  if (z >= oZ && z < oZ + Lz && t >= oT && t < oT + Lt) {
    int zl = z - oZ, tl = t - oT;
    for (int d = 0; d < Ns; d ++)
      for (int c = 0; c < Nc; c ++) {
        long loc = (long) parity * Vh + site_half(x / 2, y, zl, tl);
        src[(((loc * Ns + d) * Nc + c) * 2) * num_rhs + (d * Nc + c)] = 1;
      }
  }

  // Richardson outer, BiCGStab inner.  The inner tolerance is loose on
  // purpose: it is a preconditioner, so it only has to reduce the error
  // enough to be worth an outer step.  bicgstab ignores its own
  // preconditioner field, hence NULL.
  const double eps = OUTER_TOL;
  solver_info inner = { .solver = bicgstab, .preconditioner = NULL,
                        .max_iter = 100, .eps = 1e-2 };
  const char * inner_prec = "double";
#ifdef MIXED
  // Richardson is defect correction, so the inner solve only has to find a
  // correction, not carry the accuracy: the defect and the convergence test
  // stay double either way.
  inner.solver = bicgstab_inner_single;
  inner_prec = "single";
#endif
  solver_info outer = { .solver = richardson, .preconditioner = &inner,
                        .max_iter = OUTER_MAX, .eps = (REAL) eps };

  print0("\n--- test_pion_correlator_bicgstab: eo Richardson (max_iter=%d, tol=%.1e) "
         "preconditioned by %s-precision BiCGStab (max_iter=%d, tol=%.1e) ---\n",
         outer.max_iter, (double) outer.eps, inner_prec,
         inner.max_iter, (double) inner.eps);
  struct timeval s0, e0;
  gettimeofday(&s0, NULL);
  solve(prop, src, U, F, kappa, csw, /*chroma_norm*/ 1, /*eo*/ 1, outer);
  gettimeofday(&e0, NULL);
  print0("test_pion_correlator_bicgstab: %.2f s\n",
         (e0.tv_sec - s0.tv_sec) + 1e-6 * (e0.tv_usec - s0.tv_usec));

  REAL * correlator = malloc(sizeof(REAL) * Lt_glob);
  for (int tt = 0; tt < Lt_glob; tt ++) correlator[tt] = 0;
  project_pion_zero_momentum(prop, correlator);

  report_correlator("test_pion_correlator_bicgstab", correlator, 1e3 * eps);

  free(U); free(F); free(src); free(prop); free(correlator);
}

// ===========================================================================
// Coarse / multilevel selftests (moved out of coarse.c so coarse.c holds only
// production code).  These exercise coarse.c internals exposed in coarse.h:
// coarser_vol, local_orthonormalize_coarse, apply_P_coarse, apply_R_coarse,
// apply_Dc_lvl, apply_Dc_ref, build_coarse_op_lvl, coarse_solve_full_lvl,
// mg_cycle. Called from runner.c.  Each builds synthetic data, is
// deterministic (fixed srand seed), prints a PASS/FAIL line, and returns 0
// on pass / 1 on fail.
// ===========================================================================

// Self-test of the coarse-level transfers: with locally-orthonormalized test
// vectors, R_l P_l = I on the coarser space.  Builds a small synthetic level,
// random orthonormalized nvc, random coarser input c, and checks ||R P c - c||.
// Returns 0 on pass, 1 on fail.  Deterministic (fixed seed).
int mg_coarse_transfer_selftest(void) {
  level_t L;
  memset(&L, 0, sizeof L);
  // 8x8x4x8 level-l sites with an anisotropic block {2,2,2,4} -> 4x4x2x2 = 64
  // coarser sites: exercises per-dimension block factors and the block decode.
  L.nx = 8; L.ny = 8; L.nz = 4; L.nt = 8;
  L.blk[0] = 2; L.blk[1] = 2; L.blk[2] = 2; L.blk[3] = 4;
  L.vol = (long) L.nx * L.ny * L.nz * L.nt;
  // This level's width + the columns of the transition into its child.  Both
  // are read at runtime by the transfers; a zeroed nvar makes them write
  // nothing and the check degenerates to rel = 1.
  L.nvar = N_coarse; L.nnv = num_null_vecs;
  L.field_len = L.vol * L.nvar * 2;
  long cvol = coarser_vol(&L);

  long nv_len = L.vol * N_coarse * 2 * (long) num_null_vecs;
  long fL_len = L.vol * N_coarse * 2 * (long) num_rhs;   // level-l multi-rhs
  long cc_len = cvol  * N_coarse * 2 * (long) num_rhs;   // coarser multi-rhs

  REAL * nvc  = malloc(sizeof(REAL) * nv_len);
  REAL * cin  = malloc(sizeof(REAL) * cc_len);
  REAL * cout = malloc(sizeof(REAL) * cc_len);
  REAL * fL   = malloc(sizeof(REAL) * fL_len);
  if (!nvc || !cin || !cout || !fL) {
    fprintf(stderr, "transfer_selftest: alloc failed\n");
    free(nvc); free(cin); free(cout); free(fL); return 1;
  }

  srand(7);
  for (long i = 0; i < nv_len; i ++) nvc[i] = 2.0 * rand() / RAND_MAX - 1.0;
  for (long i = 0; i < cc_len; i ++) cin[i] = 2.0 * rand() / RAND_MAX - 1.0;

  local_orthonormalize_coarse(nvc, &L);
  apply_P_coarse(fL, cin, nvc, &L);     // fL   = P c
  apply_R_coarse(cout, fL, nvc, &L);    // cout = R P c  (should equal c)

  double maxabs = 0, maxerr = 0;
  for (long i = 0; i < cc_len; i ++) {
    double a = fabs(cin[i]);
    double e = fabs((double) cout[i] - cin[i]);
    if (a > maxabs) maxabs = a;
    if (e > maxerr) maxerr = e;
  }
  double rel = maxabs > 0 ? maxerr / maxabs : maxerr;
  int fail = !(rel < 1e-4);            // single precision over a 256-site agg
  printf("[coarse transfer selftest] R P = I : max|RPc - c| = %.3e "
         "(rel %.3e)  %s\n", maxerr, rel, fail ? "FAIL" : "PASS");

  free(nvc); free(cin); free(cout); free(fL);
  return fail;
}

// Self-test of apply_Dc_lvl: it routes through the same shared vectorized
// coarse_dslash_site kernel as apply_Dc, so on level-1 macro geometry it must
// reproduce apply_Dc bit-for-bit (same kernel, same pointer gather;
// coarse_neighbor_lvl == coarse_neighbor there).  Also reports the
// rounding-level agreement with the scalar oracle apply_Dc_ref.  Random
// coarse_op, deterministic.
int mg_coarse_dslash_selftest(void) {
  level_t L;
  memset(&L, 0, sizeof L);
  L.nx = Lxc; L.ny = Lyc; L.nz = Lzc; L.nt = Ltc; L.vol = Vc;
  // Runtime width apply_Dc_lvl reads; zeroed it writes nothing and o2 stays
  // at 0.
  L.nvar = N_coarse;
  // == vol*nvar*2, i.e. deliberately equal to the macro geometry apply_Dc
  // compiles in -- the bit-for-bit claim below is about that path (cf.
  // mg_coarse_bottom_selftest, which sets field_len != coarse_ferm_size on
  // purpose to take the generic one).
  L.field_len = coarse_ferm_size;
#if FERM_GHOSTS
  // This level is the real decomposed coarse geometry, so apply_Dc_lvl
  // exchanges z/t faces and needs the scratch a hierarchy level gets at
  // creation.
  alloc_level_dc_ghost(&L);
#endif

  long matsz = (long) N_coarse * N_coarse * 2;
  coarse_op Dc;
  Dc.self = malloc(sizeof(REAL) * (long) Vc * matsz);
  Dc.nbr  = malloc(sizeof(REAL) * (long) Vc * 8 * matsz);
  long flen = coarse_ferm_size * (long) num_rhs;
  REAL * in   = malloc(sizeof(REAL) * flen);
  // apply_Dc       (vectorized macro)
  REAL * o1   = malloc(sizeof(REAL) * flen);
  // apply_Dc_lvl   (vectorized runtime)
  REAL * o2   = malloc(sizeof(REAL) * flen);
   // apply_Dc_ref   (scalar oracle)
  REAL * oref = malloc(sizeof(REAL) * flen);
  if (!Dc.self || !Dc.nbr || !in || !o1 || !o2 || !oref) {
    fprintf(stderr, "dslash_selftest: alloc failed\n");
    free(Dc.self); free(Dc.nbr); free(in); free(o1); free(o2); free(oref);
    return 1;
  }

  srand(11);
  for (long i = 0; i < (long) Vc * matsz;     i ++)
    Dc.self[i] = 2.0*rand()/RAND_MAX-1.0;
  for (long i = 0; i < (long) Vc * 8 * matsz; i ++)
    Dc.nbr[i]  = 2.0*rand()/RAND_MAX-1.0;
  for (long i = 0; i < flen; i ++)
    in[i] = 2.0*rand()/RAND_MAX-1.0;

  // trusted vectorized macro path
  apply_Dc(&Dc, o1, in);
  // refactored runtime path (must match o1 exactly)
  apply_Dc_lvl(&Dc, o2, in, &L);
  // scalar oracle (rounding-level agreement)
  apply_Dc_ref(&Dc, oref, in);

  double maxerr = 0, refdiff = 0, refnrm = 0;
  for (long i = 0; i < flen; i ++) {
    double e = fabs((double) o1[i] - o2[i]);
    if (e > maxerr) maxerr = e;
    double dr = (double) o2[i] - oref[i];
    refdiff += dr * dr; refnrm += (double) oref[i] * oref[i];
  }
  // bit-for-bit vs apply_Dc is the regression gate
  int fail = !(maxerr == 0.0);
  // The apply_Dc_ref cross-check stays valid at a decomposed grid only
  // because the seed above is rank-independent: every rank holds the same
  // operator and field, so the exchanged ghost carries exactly what the
  // oracle's local wrap (coarse_neighbor, no ghost awareness) computes.
  // That still catches an exchange reading the wrong face, but keying the
  // data per rank would break the oracle, not find a bug.
  print0("[coarse dslash selftest] apply_Dc_lvl vs apply_Dc : max|diff| = %.3e  %s\n"
         "[coarse dslash selftest]   (vs scalar apply_Dc_ref: rel = %.3e)\n",
         maxerr, fail ? "FAIL" : "PASS", sqrt(refdiff / refnrm));

  free(Dc.self); free(Dc.nbr); free(in); free(o1); free(o2); free(oref);
  free(L.dc_ghost);
  return fail;
}

// Self-test of build_coarse_op_lvl: D_{l+1} built by probing must satisfy the
// Galerkin identity D_{l+1} x = R_l D_l P_l x for random x.  Uses a small fully
// degenerate coarser lattice (2x2x2x2, every dim extent 2) so the +-mu dedup is
// exercised in all four directions.  Random level-l operator + test vectors.
int mg_coarse_build_selftest(void) {
  level_t L;
  memset(&L, 0, sizeof L);
  // level-l 4x4x4x4, block {2,2,2,2} -> coarser 2x2x2x2 (all dims degenerate)
  L.nx = 4; L.ny = 4; L.nz = 4; L.nt = 4;
  L.blk[0] = L.blk[1] = L.blk[2] = L.blk[3] = 2;
  L.vol = (long) L.nx * L.ny * L.nz * L.nt;
  // width + columns (see width-mixing)
  L.nvar = N_coarse; L.nnv = num_null_vecs;

  level_t C;
  memset(&C, 0, sizeof C);
  C.nx = 2; C.ny = 2; C.nz = 2; C.nt = 2; C.vol = 16;
  // the child of a level coarsened by num_null_vecs columns is N_coarse wide;
  // apply_Dc_lvl reads the width and field length off the level it is given
  C.nvar = N_coarse; C.field_len = C.vol * N_coarse * 2;

  long matsz = (long) N_coarse * N_coarse * 2;
  coarse_op D1;
  D1.self = malloc(sizeof(REAL) * L.vol * matsz);
  D1.nbr  = malloc(sizeof(REAL) * L.vol * 8 * matsz);
  long nv_len = L.vol * N_coarse * 2 * (long) num_null_vecs;
  long fl_len = L.vol * N_coarse * 2 * (long) num_rhs;
  long cc_len = C.vol * N_coarse * 2 * (long) num_rhs;
  REAL * nvc = malloc(sizeof(REAL) * nv_len);
  REAL * x   = malloc(sizeof(REAL) * cc_len);
  REAL * lhs = malloc(sizeof(REAL) * cc_len);
  REAL * rhs = malloc(sizeof(REAL) * cc_len);
  REAL * Px  = malloc(sizeof(REAL) * fl_len);
  REAL * DPx = malloc(sizeof(REAL) * fl_len);
  if (!D1.self || !D1.nbr || !nvc || !x || !lhs || !rhs || !Px || !DPx) {
    fprintf(stderr, "build_selftest: alloc failed\n");
    free(D1.self); free(D1.nbr); free(nvc); free(x); free(lhs); free(rhs);
    free(Px); free(DPx); return 1;
  }
  L.Dc = &D1;

  srand(13);
  for (long i = 0; i < L.vol * matsz;     i ++)
    D1.self[i] = 2.0*rand()/RAND_MAX-1.0;
  for (long i = 0; i < L.vol * 8 * matsz; i ++)
    D1.nbr[i]  = 2.0*rand()/RAND_MAX-1.0;
  for (long i = 0; i < nv_len; i ++) nvc[i] = 2.0*rand()/RAND_MAX-1.0;
  for (long i = 0; i < cc_len; i ++) x[i]   = 2.0*rand()/RAND_MAX-1.0;

  local_orthonormalize_coarse(nvc, &L);

  coarse_op D2;
  build_coarse_op_lvl(&D2, &L, nvc);

  apply_Dc_lvl(&D2, lhs, x, &C);        // lhs = D_2 x
  apply_P_coarse(Px, x, nvc, &L);       // Px  = P_1 x
  apply_Dc_lvl(&D1, DPx, Px, &L);       // DPx = D_1 P_1 x
  apply_R_coarse(rhs, DPx, nvc, &L);    // rhs = R_1 D_1 P_1 x

  double maxabs = 0, maxerr = 0;
  for (long i = 0; i < cc_len; i ++) {
    double a = fabs(rhs[i]);
    double e = fabs((double) lhs[i] - rhs[i]);
    if (a > maxabs) maxabs = a;
    if (e > maxerr) maxerr = e;
  }
  double rel = maxabs > 0 ? maxerr / maxabs : maxerr;
  int fail = !(rel < 1e-4);
  printf("[coarse build selftest] D2 x = R D1 P x (degenerate 2^4) : "
         "max|diff| = %.3e (rel %.3e)  %s\n", maxerr, rel,
         fail ? "FAIL" : "PASS");

  free_coarse_op(&D2);
  free(D1.self); free(D1.nbr); free(nvc); free(x); free(lhs); free(rhs);
  free(Px); free(DPx);
  return fail;
}

// One geometry of the windowed-path self-test below: level-l lattice
// ldim/blk -> coarser cdim (every extent >= 3, so build_coarse_op_lvl takes
// the windowed fast path).  Reference = whole-field Galerkin probe per
// column batch (apply_P_coarse / apply_Dc_lvl / apply_R_coarse), extracted
// like the build's degenerate fallback; all 8 neighbors are distinct at
// extent >= 3 so no dedup.  The windowed build must reproduce it
// bit-identically: every coupling it skips reads exact zeros (the probe is
// supported on one block), and zero accumulations are exact.
static int windowed_build_check(const int ldim[4], const int blk[4]) {
  level_t L;
  memset(&L, 0, sizeof L);
  L.nx = ldim[0]; L.ny = ldim[1]; L.nz = ldim[2]; L.nt = ldim[3];
  L.blk[0] = blk[0]; L.blk[1] = blk[1]; L.blk[2] = blk[2]; L.blk[3] = blk[3];
  L.vol = (long) L.nx * L.ny * L.nz * L.nt;
  // build_coarse_op_lvl reads both (width-mixing)
  L.nvar = N_coarse; L.nnv = num_null_vecs;

  level_t C;
  memset(&C, 0, sizeof C);
  C.nx = L.nx / blk[0]; C.ny = L.ny / blk[1];
  C.nz = L.nz / blk[2]; C.nt = L.nt / blk[3];
  C.vol = (long) C.nx * C.ny * C.nz * C.nt;

  long matsz = (long) N_coarse * N_coarse * 2;
  coarse_op D1;
  D1.self = malloc(sizeof(REAL) * L.vol * matsz);
  D1.nbr  = malloc(sizeof(REAL) * L.vol * 8 * matsz);
  long nv_len = L.vol * N_coarse * 2 * (long) num_null_vecs;
  long fl_len = L.vol * N_coarse * 2 * (long) num_rhs;
  long cc_len = C.vol * N_coarse * 2 * (long) num_rhs;
  REAL * nvc   = malloc(sizeof(REAL) * nv_len);
  REAL * probe = malloc(sizeof(REAL) * cc_len);
  REAL * fine  = malloc(sizeof(REAL) * fl_len);
  REAL * Dfine = malloc(sizeof(REAL) * fl_len);
  REAL * res   = malloc(sizeof(REAL) * cc_len);
  coarse_op R2;
  R2.self = calloc(C.vol * matsz, sizeof(REAL));
  R2.nbr  = calloc(C.vol * 8 * matsz, sizeof(REAL));
  if (!D1.self || !D1.nbr || !nvc || !probe || !fine || !Dfine || !res
      || !R2.self || !R2.nbr) {
    fprintf(stderr, "windowed_build_check: alloc failed\n");
    free(D1.self); free(D1.nbr); free(nvc); free(probe); free(fine);
    free(Dfine); free(res); free(R2.self); free(R2.nbr);
    return 1;
  }
  L.Dc = &D1;

  srand(29);
  for (long i = 0; i < L.vol * matsz;     i ++)
    D1.self[i] = 2.0*rand()/RAND_MAX-1.0;
  for (long i = 0; i < L.vol * 8 * matsz; i ++)
    D1.nbr[i]  = 2.0*rand()/RAND_MAX-1.0;
  for (long i = 0; i < nv_len; i ++) nvc[i] = 2.0*rand()/RAND_MAX-1.0;

  local_orthonormalize_coarse(nvc, &L);

  coarse_op D2;
  build_coarse_op_lvl(&D2, &L, nvc);  // windowed path (coarser dims all >= 3)

  // Reference: whole-field probe of every coarser site, columns num_rhs at a
  // time.  Coarse-field layout:
  // idx(cl,v,rhs) = ((cl*N_coarse+v)*2)*num_rhs+rhs, imag part num_rhs higher.
  for (long q = 0; q < C.vol; q ++)
    for (int b0 = 0; b0 < N_coarse; b0 += num_rhs) {
      int nb = (N_coarse - b0 < num_rhs) ? (N_coarse - b0) : num_rhs;
      memset(probe, 0, sizeof(REAL) * cc_len);
      for (int s = 0; s < nb; s ++)
        probe[((q * N_coarse + (b0 + s)) * 2) * (long) num_rhs + s] = 1.0;

      apply_P_coarse(fine, probe, nvc, &L);        // P probe
      apply_Dc_lvl(&D1, Dfine, fine, &L);          // D_l P probe
      apply_R_coarse(res, Dfine, nvc, &L);         // R D_l P probe

      REAL * selfM = R2.self + q * matsz;
      for (int s = 0; s < nb; s ++) {
        int b = b0 + s;
        for (int a = 0; a < N_coarse; a ++) {
          long ri = ((q * N_coarse + a) * 2) * (long) num_rhs + s;
          long m = (long) (a * N_coarse + b) * 2;
          selfM[m]     = res[ri];
          selfM[m + 1] = res[ri + num_rhs];
        }
      }
      // Fold coincident directions: on a coarser extent of 2 the +mu and -mu
      // hops share a neighbor and R D P has already summed them into a single
      // matrix element, so file it in the first direction's back-slot only
      // and leave the twin zero -- apply_Dc_lvl sums all 8 slots, so
      // double-filing would double the coupling.  A no-op at extents >= 3,
      // where every neighbor is distinct.
      for (int dirp = 0; dirp < 8; dirp ++) {
        long p = coarse_neighbor_lvl(&C, q, dirp);
        int dup = 0;
        for (int e = 0; e < dirp; e ++)
          if (coarse_neighbor_lvl(&C, q, e) == p) { dup = 1; break; }
        if (dup) continue;
        REAL * Cc = R2.nbr + ((long) p * 8 + (dirp ^ 4)) * matsz;
        for (int s = 0; s < nb; s ++) {
          int b = b0 + s;
          for (int a = 0; a < N_coarse; a ++) {
            long ri = ((p * N_coarse + a) * 2) * (long) num_rhs + s;
            long m = (long) (a * N_coarse + b) * 2;
            Cc[m]     = res[ri];
            Cc[m + 1] = res[ri + num_rhs];
          }
        }
      }
    }

  // bitwise comparison (memcmp per element catches even +-0 sign flips)
  long bits_self = 0, bits_nbr = 0;
  double maxdiff = 0, refmax = 0;
  for (long i = 0; i < C.vol * matsz; i ++) {
    if (memcmp(&D2.self[i], &R2.self[i], sizeof(REAL)) != 0) bits_self ++;
    double d = fabs((double) D2.self[i] - R2.self[i]);
    if (d > maxdiff) maxdiff = d;
    if (fabs((double) R2.self[i]) > refmax)
      refmax = fabs((double) R2.self[i]);
  }
  for (long i = 0; i < C.vol * 8 * matsz; i ++) {
    if (memcmp(&D2.nbr[i], &R2.nbr[i], sizeof(REAL)) != 0) bits_nbr ++;
    double d = fabs((double) D2.nbr[i] - R2.nbr[i]);
    if (d > maxdiff) maxdiff = d;
  }
  int fail = (bits_self != 0 || bits_nbr != 0);
  printf("[windowed build selftest] %dx%dx%dx%d / blk %d%d%d%d -> %dx%dx%dx%d : "
         "self %ld/%ld nbr %ld/%ld bitdiffs, max|diff| %.3e (ref_max %.3e)  %s\n",
         L.nx, L.ny, L.nz, L.nt, blk[0], blk[1], blk[2], blk[3],
         C.nx, C.ny, C.nz, C.nt,
         bits_self, C.vol * matsz, bits_nbr, C.vol * 8 * matsz,
         maxdiff, refmax, fail ? "FAIL" : "PASS");

  free_coarse_op(&D2);
  free(D1.self); free(D1.nbr); free(nvc); free(probe); free(fine);
  free(Dfine); free(res); free(R2.self); free(R2.nbr);
  return fail;
}

// Bitwise self-test of the windowed build_coarse_op_lvl path against a
// whole-field probe of the same operator -- the path production exercises
// but the dev hierarchy, whose level-2 is 2^3x4, never does.  Four geometries:
//   6^4  / blk 2^4     -> 3^4  every coarser dim >= 3, the plain windowed case
//   6^3x3/ blk 2,2,2,1 -> 3^4  a dimension not blocked at all (in-block mask)
//   4^4  / blk 2^4     -> 2^4  every dim degenerate: all 8 directions fold onto
//                              4 neighbors, the shape of a production coarsest
//   4,4,6,6/blk 2^4    -> 2,2,3,3  degenerate and non-degenerate dims together
int mg_coarse_build_windowed_selftest(void) {
  const int l1[4] = {6, 6, 6, 6}, b1[4] = {2, 2, 2, 2};
  const int l2[4] = {6, 6, 6, 3}, b2[4] = {2, 2, 2, 1};
  const int l3[4] = {4, 4, 4, 4}, b3[4] = {2, 2, 2, 2};
  const int l4[4] = {4, 4, 6, 6}, b4[4] = {2, 2, 2, 2};
  int fail = windowed_build_check(l1, b1);
  fail |= windowed_build_check(l2, b2);
  fail |= windowed_build_check(l3, b3);
  fail |= windowed_build_check(l4, b4);
  // Slot-layout-independent cross-check of the same degenerate shape: the fold
  // above decides where a coupling is filed, and only the Galerkin identity
  // D2 x = R D1 P x can catch filing it twice or not at all.
  fail |= mg_coarse_build_selftest();
  return fail;
}

// Self-test of the geometry-agnostic bottom solve
// (coarse_solve_full_lvl + apply_Dc_lvl).  Builds a diagonally-dominant
// coarse_op on a small runtime-geometry level (so GMRES converges cleanly),
// forms b = D x_true, solves, and checks the relative residual and solution
// error.  Deterministic.  Returns 0 on pass, 1 on fail.
int mg_coarse_bottom_selftest(void) {
  level_t L;
  memset(&L, 0, sizeof L);
  L.nx = 4; L.ny = 4; L.nz = 4; L.nt = 4;
  L.vol = (long) L.nx * L.ny * L.nz * L.nt;
  // Coarse-site width apply_Dc_lvl reads.  Required: a zeroed nvar makes the 
  // apply a no-op, leaving b = 0 and the whole self-test a vacuous D x = 0
  // solve.
  L.nvar = N_coarse;
  // != coarse_ferm_size -> generic path
  L.field_len = L.vol * L.nvar * 2;

  long matsz = (long) N_coarse * N_coarse * 2;
  coarse_op D;
  D.self = calloc(L.vol * matsz, sizeof(REAL));
  D.nbr  = calloc(L.vol * 8 * matsz, sizeof(REAL));
  long flen = L.field_len * (long) num_rhs;
  REAL * xt = malloc(sizeof(REAL) * flen);
  REAL * b  = malloc(sizeof(REAL) * flen);
  REAL * xs = malloc(sizeof(REAL) * flen);
  REAL * r  = malloc(sizeof(REAL) * flen);
  if (!D.self || !D.nbr || !xt || !b || !xs || !r) {
    fprintf(stderr, "bottom_selftest: alloc failed\n");
    free(D.self); free(D.nbr); free(xt); free(b); free(xs); free(r); return 1;
  }
  L.Dc = &D;

  srand(17);
  for (long p = 0; p < L.vol; p ++) {
    REAL * s = D.self + p * matsz;
    for (int a = 0; a < N_coarse; a ++)
      for (int bb = 0; bb < N_coarse; bb ++) {
        long m = (long) (a * N_coarse + bb) * 2;
        s[m]     = 0.02 * (2.0*rand()/RAND_MAX - 1.0);
        s[m + 1] = 0.02 * (2.0*rand()/RAND_MAX - 1.0);
        if (a == bb) s[m] += (REAL) N_coarse;   // strong diagonal dominance
      }
    REAL * c = D.nbr + (long) p * 8 * matsz;
    for (long i = 0; i < 8 * matsz; i ++)
      c[i] = 0.01 * (2.0*rand()/RAND_MAX - 1.0);
  }
  for (long i = 0; i < flen; i ++)
    xt[i] = 2.0*rand()/RAND_MAX - 1.0;

  apply_Dc_lvl(&D, b, xt, &L);                 // b = D x_true
  double bn2 = 0; for (long i = 0; i < flen; i ++) bn2 += (double) b[i]*b[i];
  REAL tol = (REAL) (1e-8 * sqrt(bn2));
  int iters = coarse_solve_full_lvl(&L, xs, b, 200, tol, 30);

  apply_Dc_lvl(&D, r, xs, &L);                 // residual + solution error
  double rn = 0, xe = 0, xn = 0;
  for (long i = 0; i < flen; i ++) {
    double e = (double) r[i] - b[i]; rn += e*e;
    double d = (double) xs[i] - xt[i]; xe += d*d;
    xn += (double) xt[i]*xt[i];
  }
  double relres = bn2 > 0 ? sqrt(rn / bn2) : sqrt(rn);
  double relx   = xn  > 0 ? sqrt(xe / xn)  : sqrt(xe);
  int fail = !(relres < 1e-5);
  printf("[coarse bottom selftest] coarse_solve_full_lvl: %d iters, "
         "rel res %.3e, rel x-err %.3e  %s\n", iters, relres, relx,
         fail ? "FAIL" : "PASS");

  free(D.self); free(D.nbr); free(xt); free(b); free(xs); free(r);
  return fail;
}

// End-to-end plumbing test of the 3-level recursion, fully synthetic (no data
// files, no fine Wilson level).  Builds a two-deep coarse hierarchy rooted at
// a runtime-geometry "level 1" (4^4 sites, diagonally dominant so the cycle
// is well-conditioned), the Galerkin child D2 = R D1 P on a degenerate 2^4
// coarsest, then uses mg_cycle as the preconditioner in Richardson iteration
// to solve D1 x = b and checks convergence.  Exercises:
// restrict_lvl/prolong_lvl (coarse transfers), apply_D_lvl/smooth_lvl
// runtime-geometry dispatch, the recursive mg_cycle, and the
// coarse_solve_full_lvl bottom solve -- i.e. everything the level-1-and-below
// part of a real 3-level cycle runs.  Returns 0 on pass, 1 on fail.
int mg_3level_selftest(void) {
  long matsz = (long) N_coarse * N_coarse * 2;

  // level 1: runtime geometry (not the macro coarse), diagonally dominant op
  level_t l1; memset(&l1, 0, sizeof l1);
  l1.depth = 1; l1.nx = 4; l1.ny = 4; l1.nz = 4; l1.nt = 4;
  l1.vol = (long) l1.nx * l1.ny * l1.nz * l1.nt;
  l1.nvar = N_coarse; l1.field_len = l1.vol * N_coarse * 2;
  l1.nnv = num_null_vecs;   // hand-built level: the l1->l2 transition width
                            // (build_coarse_op_lvl sizes the child from it)
  l1.blk[0] = l1.blk[1] = l1.blk[2] = l1.blk[3] = 2;
  l1.nu = 6;
  // Hand-built levels bypass mg_setup, so the cycle shape and the smoother's
  // per-block iteration count have to be filled in by hand -- the memset above
  // would otherwise leave a V-cycle with zero block iterations.
  mg_params tp; mg_params_default(&tp);
  l1.blk_iter = tp.coarse_sap_block_iter;
  l1.cycle = tp.cycle; l1.w_mu = tp.w_mu; l1.k_iters = tp.k_iters;
  l1.solve_k_tol = tp.solve_k_tol;
  coarse_op D1;
  D1.self = calloc(l1.vol * matsz, sizeof(REAL));
  D1.nbr  = calloc(l1.vol * 8 * matsz, sizeof(REAL));
  l1.Dc = &D1;

  long nv_len = l1.vol * N_coarse * 2 * (long) num_null_vecs;
  REAL * nvc = malloc(sizeof(REAL) * nv_len);
  long flen = l1.field_len * (long) num_rhs;
  REAL * b   = malloc(sizeof(REAL) * flen);
  // Allocated like every other field here rather than with calloc, and zeroed
  // below: x is an input to apply_Dc_lvl, whose single-precision asm kernel
  // loads fields with aligned moves.  The malloc here is the 64-byte aligned
  // allocator (global_params.h); the C library's calloc promises only 16.
  REAL * x   = malloc(sizeof(REAL) * flen);
  REAL * r   = malloc(sizeof(REAL) * flen);
  REAL * dx  = malloc(sizeof(REAL) * flen);
  REAL * Dx  = malloc(sizeof(REAL) * flen);
  if (!D1.self || !D1.nbr || !nvc || !b || !x || !r || !dx || !Dx) {
    fprintf(stderr, "3level_selftest: alloc failed\n");
    free(D1.self); free(D1.nbr); free(nvc); free(b); free(x); free(r);
    free(dx); free(Dx); return 1;
  }
  memset(x, 0, sizeof(REAL) * flen);    // the Richardson loop starts at x = 0

  srand(23);
  for (long p = 0; p < l1.vol; p ++) {
    REAL * s = D1.self + p * matsz;
    for (int a = 0; a < N_coarse; a ++)
      for (int bb = 0; bb < N_coarse; bb ++) {
        long m = (long) (a * N_coarse + bb) * 2;
        s[m]     = 0.02 * (2.0*rand()/RAND_MAX - 1.0);
        s[m + 1] = 0.02 * (2.0*rand()/RAND_MAX - 1.0);
        if (a == bb) s[m] += (REAL) N_coarse;
      }
    REAL * c = D1.nbr + (long) p * 8 * matsz;
    for (long i = 0; i < 8 * matsz; i ++)
      c[i] = 0.01 * (2.0*rand()/RAND_MAX - 1.0);
  }
  for (long i = 0; i < nv_len; i ++) nvc[i] = 2.0*rand()/RAND_MAX - 1.0;
  for (long i = 0; i < flen;   i ++) b[i]   = 2.0*rand()/RAND_MAX - 1.0;
  l1.nv = nvc;
  local_orthonormalize_coarse(nvc, &l1);

  // level 2: Galerkin child on the degenerate 2^4 coarsest
  coarse_op D2;
  build_coarse_op_lvl(&D2, &l1, nvc);
  level_t l2; memset(&l2, 0, sizeof l2);
  l2.depth = 2; l2.nx = 2; l2.ny = 2; l2.nz = 2; l2.nt = 2;
  l2.vol = (long) l2.nx * l2.ny * l2.nz * l2.nt;
  l2.nvar = N_coarse; l2.field_len = l2.vol * N_coarse * 2;
  l2.Dc = &D2; l2.cg_max_iter = 100; l2.cg_tol = 1e-2; l2.cg_restart = 30;
  l2.coarser = NULL;
  l2.cycle = tp.cycle; l2.w_mu = tp.w_mu; l2.k_iters = tp.k_iters;
  l2.solve_k_tol = tp.solve_k_tol;
  l1.coarser = &l2;

  // MG-preconditioned Richardson: x_{k+1} = x_k + mg_cycle(b - D1 x_k)
  double bn2 = 0; for (long i = 0; i < flen; i ++) bn2 += (double) b[i]*b[i];
  double bn = sqrt(bn2);
  memcpy(r, b, sizeof(REAL) * flen);             // x = 0 -> r = b
  double relres = 1.0;
  int it = 0;
  for (it = 0; it < 30; it ++) {
    mg_cycle(&l1, dx, r, (REAL) tp.solve_k_tol);  // dx ~= D1^{-1} r
    for (long i = 0; i < flen; i ++) x[i] += dx[i];
    apply_Dc_lvl(&D1, Dx, x, &l1);
    double rn = 0;
    for (long i = 0; i < flen; i ++) {
      r[i] = b[i] - Dx[i];
      rn += (double) r[i]*r[i];
    }
    relres = sqrt(rn) / bn;
    if (relres < 1e-8) { it ++; break; }
  }
  int fail = !(relres < 1e-6);
  printf("[3-level selftest] MG-Richardson (synthetic 3-level): %d iters, "
         "rel res %.3e  %s\n", it, relres, fail ? "FAIL" : "PASS");

  free_coarse_op(&D2);
  free(D1.self); free(D1.nbr); free(nvc); free(b); free(x); free(r);
  free(dx); free(Dx);
  return fail;
}

// ===========================================================================
// Halo validation: multi-rank stencil bit-identity.
// Fields are synthesized deterministically from global coordinates, so every
// build assigns the same value to the same physical site no matter how the
// lattice is decomposed.  Each operator is applied and a 64-bit FNV-1a hash of
// every output site is dumped to /tmp/qc-halo/; the 1x1-grid run is the
// full-lattice reference, and compare-halo-dumps.py asserts that every rank's
// local sites match the reference slab byte-for-byte.
// ===========================================================================
#include "comm.h"

// deterministic value in [-0.5, 0.5) from (tag, index) -- splitmix64 finalizer
static double halo_hval(unsigned long tag, unsigned long idx) {
  unsigned long z = ((tag << 56) ^ idx) + 0x9E3779B97F4A7C15UL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
  z = z ^ (z >> 31);
  return (double) (z >> 11) / 9007199254740992.0 - 0.5;
}

#define HALO_GSS (4 * Nc * Nc * 2)        // gauge site REALs
#define HALO_FSS (Nc * Ns * 2 * num_rhs)  // fermion site REALs

static void halo_fill_gauge_d(REAL * U) {
  for (int t = 0; t < Lt; t ++)
    for (int z = 0; z < Lz; z ++)
      for (int y = 0; y < Ly; y ++)
        for (int x = 0; x < Lx; x ++) {
          long gs = ((((long) t + qc_grid.origin[3]) * Lz_glob
                     + (z + qc_grid.origin[2])) * Ly + y) * Lx + x;
          long loc = site_full(x, y, z, t);
          for (int i = 0; i < HALO_GSS; i ++)
            U[loc * HALO_GSS + i] = halo_hval(1, gs * HALO_GSS + i);
        }
}
// The float fillers feed the single-precision twins, which only exist alongside
// the double ones in a mixed build.
#ifdef MIXED
static void halo_fill_gauge_f(float * U) {
  for (int t = 0; t < Lt; t ++)
    for (int z = 0; z < Lz; z ++)
      for (int y = 0; y < Ly; y ++)
        for (int x = 0; x < Lx; x ++) {
          long gs = ((((long) t + qc_grid.origin[3]) * Lz_glob
                     + (z + qc_grid.origin[2])) * Ly + y) * Lx + x;
          long loc = site_full(x, y, z, t);
          for (int i = 0; i < HALO_GSS; i ++)
            U[loc * HALO_GSS + i] = (float) halo_hval(1, gs * HALO_GSS + i);
        }
}
#endif
// eo-layout clover field; values keyed on the global half-site
static void halo_fill_F_d(REAL * F) {
  for (int p = 0; p < 2; p ++)
    for (int t = 0; t < Lt; t ++)
      for (int z = 0; z < Lz; z ++)
        for (int y = 0; y < Ly; y ++)
          for (int x = 0; x < Lxh; x ++) {
            long loc = (long) p * Vh + site_half(x, y, z, t);
            long gs = (((((long) t + qc_grid.origin[3]) * Lz_glob
                       + (z + qc_grid.origin[2])) * Ly + y) * Lxh + x) * 2 + p;
            for (int i = 0; i < 72; i ++)
              F[loc * 72 + i] = halo_hval(2, gs * 72 + i);
          }
}
#ifdef MIXED
static void halo_fill_F_f(float * F) {
  for (int p = 0; p < 2; p ++)
    for (int t = 0; t < Lt; t ++)
      for (int z = 0; z < Lz; z ++)
        for (int y = 0; y < Ly; y ++)
          for (int x = 0; x < Lxh; x ++) {
            long loc = (long) p * Vh + site_half(x, y, z, t);
            long gs = (((((long) t + qc_grid.origin[3]) * Lz_glob
                       + (z + qc_grid.origin[2])) * Ly + y) * Lxh + x) * 2 + p;
            for (int i = 0; i < 72; i ++)
              F[loc * 72 + i] = (float) halo_hval(2, gs * 72 + i);
          }
}
#endif
static void halo_fill_ferm_d(REAL * f, unsigned long tag) {
  for (int p = 0; p < 2; p ++)
    for (int t = 0; t < Lt; t ++)
      for (int z = 0; z < Lz; z ++)
        for (int y = 0; y < Ly; y ++)
          for (int x = 0; x < Lxh; x ++) {
            long loc = (long) p * Vh + site_half(x, y, z, t);
            long gs = (((((long) t + qc_grid.origin[3]) * Lz_glob
                       + (z + qc_grid.origin[2])) * Ly + y) * Lxh + x) * 2 + p;
            for (int i = 0; i < HALO_FSS; i ++)
              f[loc * HALO_FSS + i] = halo_hval(tag, gs * HALO_FSS + i);
          }
}
#ifdef MIXED
static void halo_fill_ferm_f(float * f, unsigned long tag) {
  for (int p = 0; p < 2; p ++)
    for (int t = 0; t < Lt; t ++)
      for (int z = 0; z < Lz; z ++)
        for (int y = 0; y < Ly; y ++)
          for (int x = 0; x < Lxh; x ++) {
            long loc = (long) p * Vh + site_half(x, y, z, t);
            long gs = (((((long) t + qc_grid.origin[3]) * Lz_glob
                       + (z + qc_grid.origin[2])) * Ly + y) * Lxh + x) * 2 + p;
            for (int i = 0; i < HALO_FSS; i ++)
              f[loc * HALO_FSS + i] = (float) halo_hval(tag, gs * HALO_FSS + i);
          }
}
#endif

// Null-vector field (the P columns), keyed on the global fine half-site so a
// rank's local nv is exactly the 1x1 ref's slab.  Per-site layout matches
// coarse.h: nv[(((loc*Ns+d)*Nc+c)*2+reim)*num_null_vecs+vec], loc parity-major.
#define HALO_NVSS (Ns * Nc * 2 * num_null_vecs)
static void halo_fill_nv_d(REAL * nv, unsigned long tag) {
  for (int p = 0; p < 2; p ++)
    for (int t = 0; t < Lt; t ++)
      for (int z = 0; z < Lz; z ++)
        for (int y = 0; y < Ly; y ++)
          for (int x = 0; x < Lxh; x ++) {
            long loc = (long) p * Vh + site_half(x, y, z, t);
            long gs = (((((long) t + qc_grid.origin[3]) * Lz_glob
                       + (z + qc_grid.origin[2])) * Ly + y) * Lxh + x) * 2 + p;
            for (long i = 0; i < HALO_NVSS; i ++)
              nv[loc * HALO_NVSS + i] = halo_hval(tag, gs * HALO_NVSS + i);
          }
}

// Fill a multi-rhs coarse field keyed on the global coarse site, so the
// same physical coarse site gets the same value under any decomposition
// (coarse lattice = fine/BLOCK_AGG; only z,t are decomposed, origins are
// block-aligned).
#define HALO_CBLK ((long) N_coarse * 2 * num_rhs)    // reals / coarse site
#define HALO_CMAT ((long) N_coarse * N_coarse * 2)   // reals / coupling block
#define HALO_ZGC  (Lz_glob / BLOCK_AGG)              // global coarse z extent
static long halo_cglob(int x, int y, int z, int t) {  // global coarse lex key
  int coz = qc_grid.origin[2] / BLOCK_AGG, cot = qc_grid.origin[3] / BLOCK_AGG;
  return (((long) (t + cot) * HALO_ZGC + (z + coz)) * Lyc + y) * Lxc + x;
}
static void halo_fill_coarse(REAL * c, unsigned long tag) {
  for (int t = 0; t < Ltc; t ++)
    for (int z = 0; z < Lzc; z ++)
      for (int y = 0; y < Lyc; y ++)
        for (int x = 0; x < Lxc; x ++) {
          long loc = (((long) t * Lzc + z) * Lyc + y) * Lxc + x;
          long g   = halo_cglob(x, y, z, t);
          for (long i = 0; i < HALO_CBLK; i ++)
            c[loc * HALO_CBLK + i] = halo_hval(tag, g * HALO_CBLK + i);
        }
}
// Depth-1 null-vector field (the P_l columns), keyed on the global depth-1
// coarse site.  Per-site layout matches nvc_off in coarse.c:
// nvc[(((s*N_coarse+v)*2)+reim)*num_null_vecs+vec], contiguous per site.
#define HALO_NVCSS (N_coarse * 2 * num_null_vecs)
static void halo_fill_nvc(REAL * nvc, unsigned long tag) {
  for (int t = 0; t < Ltc; t ++)
    for (int z = 0; z < Lzc; z ++)
      for (int y = 0; y < Lyc; y ++)
        for (int x = 0; x < Lxc; x ++) {
          long loc = (((long) t * Lzc + z) * Lyc + y) * Lxc + x;
          long g   = halo_cglob(x, y, z, t);
          for (long i = 0; i < HALO_NVCSS; i ++)
            nvc[loc * HALO_NVCSS + i] = halo_hval(tag, g * HALO_NVCSS + i);
        }
}
// Synthesize a coarse_op (self + 8 nbr coupling blocks) keyed on the global
// site and direction -- a deterministic stand-in for a real Galerkin
// operator, enough to exercise apply_Dc's neighbor gather + halo redirect.
static void halo_fill_coarse_op(coarse_op * Dc, unsigned long tag) {
  Dc->self = malloc(sizeof(REAL) * (long) Vc * HALO_CMAT);
  Dc->nbr  = malloc(sizeof(REAL) * (long) Vc * 8 * HALO_CMAT);
  for (int t = 0; t < Ltc; t ++)
    for (int z = 0; z < Lzc; z ++)
      for (int y = 0; y < Lyc; y ++)
        for (int x = 0; x < Lxc; x ++) {
          long loc = (((long) t * Lzc + z) * Lyc + y) * Lxc + x;
          long g   = halo_cglob(x, y, z, t);
          for (long i = 0; i < HALO_CMAT; i ++)
            Dc->self[loc * HALO_CMAT + i] = halo_hval(tag, g * HALO_CMAT + i);
          for (int dir = 0; dir < 8; dir ++)
            for (long i = 0; i < HALO_CMAT; i ++)
              Dc->nbr[(loc * 8 + dir) * HALO_CMAT + i] =
                  halo_hval(tag + 1 + dir, g * HALO_CMAT + i);
        }
}

// Depth-2 (coarsest) geometry = depth-1 coarse / COARSE_BLK1.
// x,y are undecomposed (global = local); z,t carry the rank origin (in
// coarsest sites = fine origin / (BLOCK_AGG * 2)).  Same global-keying idea
// as depth 1, so a physical coarsest site gets the same value under any
// decomposition.
#define HALO_C2X  (Lxc / 2)
#define HALO_C2Y  (Lyc / 2)
#define HALO_C2Z  (Lzc / 2)
#define HALO_C2T  (Ltc / 2)
#define HALO_C2ZG (Lz_glob / BLOCK_AGG / 2)        // global coarsest z extent
static long halo_c2glob(int x, int y, int z, int t) {
  int coz = qc_grid.origin[2] / (BLOCK_AGG * 2),
      cot = qc_grid.origin[3] / (BLOCK_AGG * 2);
  return (((long) (t + cot) * HALO_C2ZG + (z + coz)) * HALO_C2Y + y)
          * HALO_C2X + x;
}
static void halo_fill_coarse_op2(coarse_op * Dc, unsigned long tag) {
  long vol2 = (long) HALO_C2X * HALO_C2Y * HALO_C2Z * HALO_C2T;
  Dc->self = malloc(sizeof(REAL) * vol2 * HALO_CMAT);
  Dc->nbr  = malloc(sizeof(REAL) * vol2 * 8 * HALO_CMAT);
  for (int t = 0; t < HALO_C2T; t ++)
    for (int z = 0; z < HALO_C2Z; z ++)
      for (int y = 0; y < HALO_C2Y; y ++)
        for (int x = 0; x < HALO_C2X; x ++) {
          long loc = (((long) t * HALO_C2Z + z) * HALO_C2Y + y) * HALO_C2X + x;
          long g   = halo_c2glob(x, y, z, t);
          for (long i = 0; i < HALO_CMAT; i ++)
            Dc->self[loc * HALO_CMAT + i] = halo_hval(tag, g * HALO_CMAT + i);
          for (int dir = 0; dir < 8; dir ++)
            for (long i = 0; i < HALO_CMAT; i ++)
              Dc->nbr[(loc * 8 + dir) * HALO_CMAT + i] =
                  halo_hval(tag + 1 + dir, g * HALO_CMAT + i);
        }
}

// --- dump layout descriptors -----------------------------------------------
// Every dumped entry is written as (global site key, hash), so comparing a
// decomposed run against the 1x1 reference is a dictionary merge: no
// knowledge of storage order, rank ordering, or which dimension was cut is
// needed on the reading side.  That matters because a rank's sites are a
// contiguous slab of the reference only when t alone is decomposed -- under
// a z cut they are strided, so any concatenate-the-ranks comparison silently
// mis-reports.
//   kind 0: fine, parity-major [even half | odd half],
//           block-major within a half
//   kind 1: fine, one parity half
//   kind 2: lexicographic [t][z][y][x] (every coarse level)
// Extents/origins are in the field's own site units (fine sites, coarse sites,
// coarsest sites), with x counted in half-lattice units for the fine kinds.
typedef struct {
  int kind;
  int nx, ny, nz, nt;   // local extents
  int nzg, ntg;         // global z/t extents, same units
  int oz, ot;           // this rank's origin, same units
} halo_layout;

static halo_layout hl_fine(int kind) {
  halo_layout l = { kind, Lxh, Ly, Lz, Lt, Lz_glob, Lt_glob,
                    qc_grid.origin[2], qc_grid.origin[3] };
  return l;
}
static halo_layout hl_coarse(void) {   // depth-1: fine / BLOCK_AGG
  halo_layout l = { 2, Lxc, Lyc, Lzc, Ltc,
                    Lz_glob / BLOCK_AGG, Lt_glob / BLOCK_AGG,
                    qc_grid.origin[2] / BLOCK_AGG,
                    qc_grid.origin[3] / BLOCK_AGG };
  return l;
}
static halo_layout hl_c2(void) {       // coarsest as built by the halo
                                       // harness (blk 2)
  halo_layout l = { 2, HALO_C2X, HALO_C2Y, HALO_C2Z, HALO_C2T,
                    Lz_glob / BLOCK_AGG / 2, Lt_glob / BLOCK_AGG / 2,
                    qc_grid.origin[2] / (BLOCK_AGG * 2),
                    qc_grid.origin[3] / (BLOCK_AGG * 2) };
  return l;
}
static halo_layout hl_lvl2(const int cblk[4]) {   // depth-2 with an explicit
                                                  // blocking
  halo_layout l = { 2, Lxc / cblk[0], Lyc / cblk[1],
                    Lzc / cblk[2], Ltc / cblk[3],
                    (Lz_glob / BLOCK_AGG) / cblk[2],
                    (Lt_glob / BLOCK_AGG) / cblk[3],
                    (qc_grid.origin[2] / BLOCK_AGG) / cblk[2],
                    (qc_grid.origin[3] / BLOCK_AGG) / cblk[3] };
  return l;
}

// local site index within the dumped buffer -> key identifying the physical
// site globally (identical under every decomposition)
static long halo_site_key(halo_layout l, long s) {
  long half = (long) l.nx * l.ny * l.nz * l.nt;
  if (l.kind != 2) {
    long parity = (l.kind == 0) ? s / half : 0;
    int xh, y, z, t;
    host_loc_inv_of_dims(l.nx, l.ny, l.nz, l.nt, s - parity * half,
                         &xh, &y, &z, &t);
    return ((((parity * l.ntg + (t + l.ot)) * l.nzg + (z + l.oz)) * l.ny + y)
            * l.nx) + xh;
  }
  int x = (int)(s % l.nx);  long r = s / l.nx;
  int y = (int)(r % l.ny);  r /= l.ny;
  int z = (int)(r % l.nz);  r /= l.nz;
  int t = (int) r;
  return (((long)(t + l.ot) * l.nzg + (z + l.oz)) * l.ny + y) * l.nx + x;
}

// Dump directory: $QC_HALO_DIR or /tmp/qc-halo, created on demand.  A write
// failure is fatal: the harness proves nothing except by the dumps it leaves,
// so a run that silently wrote none would still report success.
static const char * halo_dir(void) {
  static char dir[200];
  if (!dir[0]) {
    const char * e = getenv("QC_HALO_DIR");
    snprintf(dir, sizeof dir, "%s", e && *e ? e : "/tmp/qc-halo");
    if (mkdir(dir, 0755) != 0 && errno != EEXIST) {
      fprintf(stderr, "halo: cannot create dump directory %s: %s\n",
              dir, strerror(errno));
      exit(1);
    }
  }
  return dir;
}

static void halo_dump_hash(const char * op, const void * buf, long nsites,
                           long site_bytes, halo_layout lay) {
  char fn[256];
  snprintf(fn, sizeof fn, "%s/%s.P%dx%d.r%d.bin", halo_dir(), op,
           qc_grid.P[2], qc_grid.P[3], qc_grid.rank);
  FILE * fp = fopen(fn, "wb");
  if (!fp) {
    fprintf(stderr, "halo: cannot write %s: %s\n", fn, strerror(errno));
    exit(1);
  }
  unsigned long hdr[2] = { 0x314F4C41484351UL,       // "QCHALO1"
                          (unsigned long) nsites };
  fwrite(hdr, 8, 2, fp);
  const unsigned char * b = buf;
  for (long s = 0; s < nsites; s ++) {
    unsigned long h = 0xCBF29CE484222325UL;   // FNV-1a 64 over the site bytes
    for (long i = 0; i < site_bytes; i ++) {
      h ^= b[s * site_bytes + i];
      h *= 0x100000001B3UL;
    }
    unsigned long rec[2] = { (unsigned long) halo_site_key(lay, s), h };
    fwrite(rec, 8, 2, fp);
  }
  if (fclose(fp) != 0) {
    fprintf(stderr, "halo: error closing %s: %s\n", fn, strerror(errno));
    exit(1);
  }
}

#ifdef MIXED
// single-precision twin of D (the ASM dispatcher path in the _f TU)
void D_f(float * output, float * ferm, float * U, float * F, int clover);
// single-precision twin of the SAP smoother (last arg is solver_info* in the
// _f TU; only ever passed NULL here)
int red_black_schwarz_f(float * output, float * ferm, float * U_kappa,
                        float * F_csw, int clover, int max_iter,
                        int sap_block_iter, float eps, int eo_precondition,
                        void * preconditioner);
#endif

void test_halo_stencil() {
  long mfs = (long) ferm_size * num_rhs;
  long hfs = mfs / 2;
  REAL * U_k = malloc(sizeof(REAL) * gauge_alloc_size);
  halo_fill_gauge_d(U_k);
  halo_gauge(U_k, sizeof(REAL) * HALO_GSS);
  REAL * F   = malloc(sizeof(REAL) * (long) V * 72);
  halo_fill_F_d(F);
  REAL * fin = malloc(sizeof(REAL) * mfs);  halo_fill_ferm_d(fin, 3);
  REAL * eta = malloc(sizeof(REAL) * mfs);  halo_fill_ferm_d(eta, 4);
  REAL * out = malloc(sizeof(REAL) * mfs);
  REAL * odd = malloc(sizeof(REAL) * hfs);
  long fsb = HALO_FSS * sizeof(REAL);

  D(out, fin, U_k, F, 1);
  halo_dump_hash("D",      out, V,  fsb, hl_fine(0));
  D_residual(out, eta, fin, U_k, F, 1);
  halo_dump_hash("Dres",   out, V,  fsb, hl_fine(0));
  D_residual_eo(out, eta, fin, U_k, F, 1);

  halo_dump_hash("Dreseo", out, V,  fsb, hl_fine(0));
  H(out, fin, U_k, F, 1);
  halo_dump_hash("H",      out, V,  fsb, hl_fine(0));
  memset(out, 0, sizeof(REAL) * hfs);    // H_oe/H_eo accumulate
  H_oe(out, fin, U_k);
  halo_dump_hash("Hoe",    out, Vh, fsb, hl_fine(1));
  memset(out, 0, sizeof(REAL) * hfs);
  H_eo(out, fin + hfs, U_k);
  halo_dump_hash("Heo",    out, Vh, fsb, hl_fine(1));
  D_hat(out, fin, U_k, F, odd, 1);
  halo_dump_hash("Dhat",   out, Vh, fsb, hl_fine(1));
                                         
  halo_dump_hash("DhatOdd", odd, Vh, fsb, hl_fine(1));
  H_hat(out, fin, U_k, F, odd, 1);
  halo_dump_hash("Hhat",   out, Vh, fsb, hl_fine(1));

  // the fine SAP smoother (this double twin runs the C kernels --
  // wilson_kernel_block_c boundary hops + wilson_kernel_c Schur-RHS pass).
  // 2 sweeps so the cross-block/cross-rank boundary reads fire in 3 of the 4
  // half-sweeps; per-block arithmetic is block-local, so multi-rank output is
  // bit-identical, both parities.
  mg_params hp; mg_params_default(&hp);
  red_black_schwarz(out, fin, U_k, F, 1, 2, hp.sap_block_iter, 0, 0, NULL);
  halo_dump_hash("RBS",    out, V,  fsb, hl_fine(0));

  // exercise prepare_gauge itself (copy + decomposition-aware APBC fold at
  // the global t boundary + kappa + gauge-ghost exchange) through a D apply
  REAL * U_raw = malloc(sizeof(REAL) * gauge_alloc_size);
  halo_fill_gauge_d(U_raw);          // raw links; prepare_gauge makes U_k2
  REAL * U_k2 = malloc(sizeof(REAL) * gauge_alloc_size);
  prepare_gauge(U_k2, U_raw, 0.125);
  D(out, fin, U_k2, F, 1);
  halo_dump_hash("Dprep",  out, V,  fsb, hl_fine(0));
  free(U_raw); free(U_k2);

#ifdef MIXED
  // single-input D (float input, double output, C body in the double TU)
  float * zin = malloc(sizeof(float) * mfs);  halo_fill_ferm_f(zin, 3);
  D_si(out, zin, U_k, F, 1);
  halo_dump_hash("Dsi",    out, V,  fsb, hl_fine(0));
  // full single-precision twin: exercises the ASM dispatcher + float halos
  float * U_f   = malloc(sizeof(float) * gauge_alloc_size);
  halo_fill_gauge_f(U_f);
  halo_gauge(U_f, sizeof(float) * HALO_GSS);
  float * F_f   = malloc(sizeof(float) * (long) V * 72);
  halo_fill_F_f(F_f);
  float * out_f = malloc(sizeof(float) * mfs);
  D_f(out_f, zin, U_f, F_f, 1);
  halo_dump_hash("Df", out_f, V, HALO_FSS * sizeof(float), hl_fine(0));
  // single-precision SAP twin -- the production path: ASM masked
  // kernels reading ghost pointers (f_pz/f_mz/f_pt/f_mt in wilson_kernel_block)
  // plus the full-dispatch Schur-RHS pass, float halos throughout.
  red_black_schwarz_f(out_f, zin, U_f, F_f, 1, 2, hp.sap_block_iter, 0.0f, 0,
                      NULL);
  halo_dump_hash("RBSf", out_f, V, HALO_FSS * sizeof(float), hl_fine(0));
  free(zin); free(U_f); free(F_f); free(out_f);
#endif

  // coarse Dslash halo -- apply_Dc on the macro depth-1 coarse lattice
  // reads neighbor coarse sites; z/t face neighbors are redirected to the
  // coarse ghost (halo_coarse).  Same arithmetic/couplings/inputs as the 1x1
  // ref, so per-coarse-site output is bit-identical across decompositions.
  coarse_op cDc; halo_fill_coarse_op(&cDc, 50);
  long ccs = coarse_ferm_size * num_rhs;
  REAL * cin  = malloc(sizeof(REAL) * ccs); halo_fill_coarse(cin, 60);
  REAL * cout = malloc(sizeof(REAL) * ccs);
  apply_Dc(&cDc, cout, cin);
  halo_dump_hash("Dc", cout, Vc, HALO_CBLK * sizeof(REAL), hl_coarse());

  // coarse SAP smoother on the macro depth-1 level.  Cross-rank face
  // neighbor dpsi is exchanged per color phase (halo_coarse on dpsi_field ->
  // dpsi_ghost) and read through the ghost; coloring is global.  Per-block
  // arithmetic is block-local + the same neighbor dpsi as the 1x1 ref, so phi
  // (and the exported exit residual) are bit-identical across decompositions.
  level_t CL; memset(&CL, 0, sizeof CL);
  CL.depth = 1;
  CL.nx = Lxc; CL.ny = Lyc; CL.nz = Lzc; CL.nt = Ltc;
  CL.vol = Vc; CL.volh = Vhc; CL.nvar = N_coarse;
  CL.field_len = (long) Vc * N_coarse * 2;
  int cblk[4] = COARSE_BLK1;
  for (int d = 0; d < 4; d ++) {
    CL.blk[d] = cblk[d];
    CL.sap_blk[d] = cblk[d];
  }
  // Hand-built level: mg_setup would normally supply nu and blk_iter, and a
  // zero blk_iter would silently make every block solve a no-op.
  mg_params cp; mg_params_default(&cp);
  CL.nu = cp.coarse_smoother_sweeps;
  CL.blk_iter = cp.coarse_sap_block_iter;
  CL.Dc = &cDc;
  // Rank origin in depth-1 coarse sites.  Not optional: coarse_sap derives its
  // rank_blk_color from corigin, i.e. how many SAP blocks precede this rank
  // globally, so that the red-black coloring is the global checkerboard.
  // Leaving it zero (as the memset does) makes every rank color as if it were
  // rank 0, and any rank preceded by an odd number of blocks then solves the
  // two colors in the wrong order -- silently, and only for that decomposition.
  CL.corigin[0] = CL.corigin[1] = 0;
  CL.corigin[2] = qc_grid.origin[2] / BLOCK_AGG;
  CL.corigin[3] = qc_grid.origin[3] / BLOCK_AGG;
  REAL * ceta = cin;                    // reuse the global-keyed coarse field
  REAL * cphi = cout, * crex = malloc(sizeof(REAL) * ccs);
  // SAP blocking needs divisibility
  if (Lzc % cblk[2] == 0 && Ltc % cblk[3] == 0) {
    coarse_sap(&CL, cphi, ceta, NULL);
    halo_dump_hash("Csap", cphi, Vc, HALO_CBLK * sizeof(REAL), hl_coarse());
    // again, now exporting the exit residual
    coarse_sap(&CL, cphi, ceta, crex);
    halo_dump_hash("Crex", crex, Vc, HALO_CBLK * sizeof(REAL), hl_coarse());
  }
  free(cDc.self); free(cDc.nbr); free(cin); free(cout); free(crex);

  // coarsest eo-Schur matvec halo.  apply_coarse_schur reads in_odd at
  // even sites' odd neighbors (ghost #1) and ue at odd sites' even neighbors
  // (ghost #2); both cross-rank faces come from halo_coarse via the eo's
  // gfull/ghost scratch.  Parity is labeled globally (corigin offset), so the
  // even/odd sets and the cross-rank reads agree across a face.  It is a pure
  // stencil (no reductions), so per-site output is bit-identical across
  // grids.  We scatter the odd output back to a full-lex coarsest field
  // (even sites 0) and dump it in lex order, comparable against the 1x1 ref
  // via the 'c2' layout.
  coarse_op c2; halo_fill_coarse_op2(&c2, 80);
  long vol2 = (long) HALO_C2X * HALO_C2Y * HALO_C2Z * HALO_C2T,
       volh2 = vol2 / 2;
  level_t C2; memset(&C2, 0, sizeof C2);
  C2.depth = 2;
  C2.nx = HALO_C2X; C2.ny = HALO_C2Y; C2.nz = HALO_C2Z; C2.nt = HALO_C2T;
  C2.vol = vol2; C2.volh = volh2; C2.nvar = N_coarse;
  C2.field_len = vol2 * N_coarse * 2; C2.Dc = &c2;
  C2.corigin[0] = C2.corigin[1] = 0;
  C2.corigin[2] = qc_grid.origin[2] / (BLOCK_AGG * 2);
  C2.corigin[3] = qc_grid.origin[3] / (BLOCK_AGG * 2);
  coarse_eo_op eo2;
  if (build_coarse_eo_op(&eo2, &c2, &C2) != 0) {
    printf("[halo rank %d] build_coarse_eo_op FAILED (odd global extent?)\n",
           qc_grid.rank);
  } else {
    REAL * in_odd  = malloc(sizeof(REAL) * volh2 * HALO_CBLK);
    REAL * out_odd = malloc(sizeof(REAL) * volh2 * HALO_CBLK);
    // fill in_odd keyed by each odd site's global coarsest lex
    // (so ref/decomp agree)
    for (long o = 0; o < volh2; o ++) {
      int cl = eo2.odd_lex[o];
      int x = cl % HALO_C2X, r = cl / HALO_C2X, y = r % HALO_C2Y; r /= HALO_C2Y;
      int z = r % HALO_C2Z; r /= HALO_C2Z; int t = r;
      long g = halo_c2glob(x, y, z, t);
      for (long i = 0; i < HALO_CBLK; i ++)
        in_odd[o * HALO_CBLK + i] = halo_hval(90, g * HALO_CBLK + i);
    }
    apply_coarse_schur(&eo2, out_odd, in_odd);
    REAL * sfull = calloc(vol2 * HALO_CBLK, sizeof(REAL));  // even sites stay 0
    for (long o = 0; o < volh2; o ++)
      memcpy(sfull + (long) eo2.odd_lex[o] * HALO_CBLK,
             out_odd + o * HALO_CBLK, HALO_CBLK * sizeof(REAL));
    halo_dump_hash("Schur", sfull, vol2, HALO_CBLK * sizeof(REAL), hl_c2());
    free(in_odd); free(out_odd); free(sfull);
    free_coarse_eo_op(&eo2);
  }
  free(c2.self); free(c2.nbr);

  // build_coarse_op (depth 0->1).  D_c = R D P from a global-keyed gauge
  // + null-vector field.  The windowed local pass fills self + on-rank
  // neighbor couplings; the seam pass fills the cross-rank neighbors by
  // pulling the off-rank nv boundary (one halo) and replaying the shell hop
  // through the ghost-aware kernel.  No reductions -> every coupling block
  // is bit-identical to the 1x1 reference per coarse site.  Cself = self,
  // Cnbr = all 8 nbr blocks.
  REAL * Uk_b   = malloc(sizeof(REAL) * gauge_alloc_size);
  REAL * Uraw_b = malloc(sizeof(REAL) * gauge_alloc_size);
  halo_fill_gauge_d(Uraw_b);
  prepare_gauge(Uk_b, Uraw_b, 0.125);     // copy + APBC + kappa + tail ghosts
  REAL * nv_b = malloc(sizeof(REAL) * (long) V * HALO_NVSS);
  halo_fill_nv_d(nv_b, 70);
  coarse_op Db; Db.self = NULL; Db.nbr = NULL;
  build_coarse_op(&Db, nv_b, Uk_b, F, 1);
  halo_dump_hash("Cself", Db.self, Vc,     HALO_CMAT * sizeof(REAL),
                 hl_coarse());
  halo_dump_hash("Cnbr",  Db.nbr,  Vc, 8 * HALO_CMAT * sizeof(REAL),
                 hl_coarse());
  free(Db.self); free(Db.nbr); free(Uk_b); free(Uraw_b); free(nv_b);

  printf("[halo rank %d] P=%dx%d origin=(z%d,t%d) dumps written\n",
         qc_grid.rank, qc_grid.P[2], qc_grid.P[3],
         qc_grid.origin[2], qc_grid.origin[3]);
  free(U_k); free(F); free(fin); free(eta); free(out); free(odd);
}

// build_coarse_op_lvl (depth 1->2) cross-rank, coarse-only (no fine fields,
// so it stays small at the 24^2x48^2 lattice the depth-2 windowed build needs
// -- every depth-2 dim >= 3 requires fine >= 24/dim, decomposed >= 48).
// A synthetic depth-1 operator + null vectors, both keyed on the global depth-1
// site (so a rank's slab matches the 1x1 ref), drive build_coarse_op_lvl; the
// windowed local pass fills self + on-rank neighbors, the seam pass pulls the
// off-rank nvc boundary and replays the single depth-1 hop.  No reductions ->
// the depth-2 couplings are bit-identical to the 1x1 ref per coarsest site.
void test_build_lvl_halo() {
  coarse_op Dc1; halo_fill_coarse_op(&Dc1, 50);        // synthetic depth-1 op
  REAL * nvc = malloc(sizeof(REAL) * (long) Vc * HALO_NVCSS);
  halo_fill_nvc(nvc, 71);

  level_t L1; memset(&L1, 0, sizeof L1);
  L1.depth = 1;
  L1.nx = Lxc; L1.ny = Lyc; L1.nz = Lzc; L1.nt = Ltc;
  L1.vol = Vc; L1.volh = Vhc; L1.nvar = N_coarse;
  // depth-1 -> 2 column count; nvc below is sized on it
  L1.nnv = num_null_vecs;
  L1.field_len = (long) Vc * N_coarse * 2;
  int cblk[4] = COARSE_BLK1;
  for (int d = 0; d < 4; d ++) {
    L1.blk[d] = cblk[d];
    L1.sap_blk[d] = cblk[d];
  }
  L1.Dc = &Dc1;
  L1.corigin[0] = L1.corigin[1] = 0;
  L1.corigin[2] = qc_grid.origin[2] / BLOCK_AGG;
  L1.corigin[3] = qc_grid.origin[3] / BLOCK_AGG;

  coarse_op D2; D2.self = NULL; D2.nbr = NULL;
  build_coarse_op_lvl(&D2, &L1, nvc);

  long cvol2 = (long) (Lxc / cblk[0]) * (Lyc / cblk[1])
             * (Lzc / cblk[2]) * (Ltc / cblk[3]);
  halo_dump_hash("C2self", D2.self, cvol2,     HALO_CMAT * sizeof(REAL),
                 hl_lvl2(cblk));
  halo_dump_hash("C2nbr",  D2.nbr,  cvol2, 8 * HALO_CMAT * sizeof(REAL),
                 hl_lvl2(cblk));
  printf("[lvlbuild rank %d] P=%dx%d cvol2=%ld dumps written\n",
         qc_grid.rank, qc_grid.P[2], qc_grid.P[3], cvol2);

  free(Dc1.self); free(Dc1.nbr); free(nvc); free(D2.self); free(D2.nbr);
}

// ---------------------------------------------------------------------------
// Public-interface layout gate (./qc layout)
// ---------------------------------------------------------------------------
// qc.h publishes a field order so callers can fill a source in place.  This
// checks that what it publishes is what the build actually uses.
//
// The last check is the one that earns its keep.  The first three compare the
// accessors against an expression written the same way they are, so a
// convention that was wrong consistently -- spin and color exchanged, say --
// would pass all of them.  Only applying D can tell whether the index qc.h
// calls a spin is the one the gamma matrices act on.  The pion correlator
// cannot: it sums |prop|^2 over all twelve spin-colors at source and sink,
// which is invariant under any permutation of them.
void test_qc_layout() {
  int fails = 0;
  const long vh = Vh;              // bind: Vh expands to a bare product
  int og[4];
  qc_local_origin(og);

  // -- the build qc.h reports is the build we are in ------------------------
  {
    int gd[4], ld[4];
    qc_global_dims(gd);
    qc_local_dims(ld);
    int ok = gd[0] == Lx && gd[1] == Ly && gd[2] == Lz_glob && gd[3] == Lt_glob
          && ld[0] == Lx && ld[1] == Ly && ld[2] == Lz      && ld[3] == Lt
          && og[2] == qc_grid.origin[2] && og[3] == qc_grid.origin[3]
          && qc_num_rhs() == num_rhs && qc_num_colors() == Nc
          && qc_num_spins() == Ns
          && qc_field_elements() == (long) ferm_size * num_rhs
          && qc_lane_elements()  == (long) V * Ns * Nc * 2
          && qc_gauge_elements() == (long) gauge_size
          && (size_t) qc_native_precision() == sizeof(REAL);
    print0("[qc layout] geometry and precision queries => %s\n",
           ok ? "PASS" : "FAIL");
    if (!ok) fails ++;
  }

  // -- site map and its inverse, over the whole local volume ----------------
  {
    long bad_map = 0, bad_inv = 0;
    for (int t = 0; t < Lt; t ++)
      for (int z = 0; z < Lz; z ++)
        for (int y = 0; y < Ly; y ++)
          for (int x = 0; x < Lx; x ++) {
            // parity is a property of the global site; see qc_site_index
            const int par = (x ^ y ^ (z + og[2]) ^ (t + og[3])) & 1;
            const long ref = (long) par * vh + site_half(x / 2, y, z, t);
            const long got = qc_site_index(x, y, z, t);
            if (got != ref) bad_map ++;
            int xi, yi, zi, ti;
            qc_site_coords(got, &xi, &yi, &zi, &ti);
            if (xi != x || yi != y || zi != z || ti != t) bad_inv ++;
          }
    const long nsite = (long) V;
    print0("[qc layout] site map vs site_half: %ld/%ld differ; round trip through "
           "qc_site_coords: %ld/%ld differ => %s\n",
           bad_map, nsite, bad_inv, nsite,
           (bad_map == 0 && bad_inv == 0) ? "PASS" : "FAIL");
    if (bad_map || bad_inv) fails ++;
  }

  // -- import/export invert each other, in both precisions ------------------
  // Values are 24-bit integers so a float and a double hold them identically
  // and the comparison can be exact.  Lane 5 is imported after lane 3 and lane
  // 3 read back afterwards, so a stride that bleeds across lanes shows up here.
  {
    const long ln = qc_lane_elements(), fn = qc_field_elements();
    double * in_d  = malloc(sizeof(double) * ln);
    double * out_d = malloc(sizeof(double) * ln);
    float  * in_f  = malloc(sizeof(float)  * ln);
    float  * out_f = malloc(sizeof(float)  * ln);
    REAL   * f     = qc_alloc(fn);
    qc_field_zero(f);
    for (long i = 0; i < ln; i ++) {
      in_d[i] = (double) (i % 16777216);
      in_f[i] = (float) in_d[i];
    }
    qc_import(f, in_d, QC_DOUBLE, 3);
    qc_import(f, in_f, QC_SINGLE, 5);
    qc_export(f, out_d, QC_DOUBLE, 3);
    qc_export(f, out_f, QC_SINGLE, 5);
    long bad_d = 0, bad_f = 0;
    for (long i = 0; i < ln; i ++) {
      if (out_d[i] != in_d[i]) bad_d ++;
      if (out_f[i] != in_f[i]) bad_f ++;
    }
    // every other lane must still be untouched
    long spill = 0;
    for (long i = 0; i < fn; i ++) {
      const int lane = (int) (i % num_rhs);
      if (lane != 3 && lane != 5 && f[i] != 0) spill ++;
    }
    print0("[qc layout] round trip: %ld/%ld differ (double), %ld/%ld (single); "
           "%ld writes outside the imported lanes => %s\n",
           bad_d, ln, bad_f, ln, spill,
           (bad_d == 0 && bad_f == 0 && spill == 0) ? "PASS" : "FAIL");
    if (bad_d || bad_f || spill) fails ++;
    free(in_d); free(out_d); free(in_f); free(out_f); qc_free(f);
  }

  // -- one component imports to exactly one place ---------------------------
  // A round trip cannot see this: it uses the same map both ways, so any
  // consistent relabelling of (spin, color) survives it.  Here the destination
  // is named independently, the way every other consumer in the build names it.
  {
    const long ln = qc_lane_elements(), fn = qc_field_elements();
    // An odd-parity site on purpose: at an even one the parity term is zero, so
    // a map that forgot it entirely would still land in the right place here.
    const int px = 3, py = 1, pz = 2, pt = 1, ps = 1, pc = 2, lane = 2;
    double * u = calloc(ln, sizeof(double));
    REAL * f = qc_alloc(fn);
    qc_field_zero(f);
    const long uoff = ((((long) pt * Lz + pz) * Ly + py) * Lx + px) * Ns * Nc
                      + (long) ps * Nc + pc;
    u[2 * uoff] = 1;
    qc_import(f, u, QC_DOUBLE, lane);

    const int par = (px ^ py ^ (pz + og[2]) ^ (pt + og[3])) & 1;
    const long loc = (long) par * vh + site_half(px / 2, py, pz, pt);
    const long want = (((loc * Ns + ps) * Nc + pc) * 2) * num_rhs + lane;
    long nz = 0, at = -1;
    for (long i = 0; i < fn; i ++) if (f[i] != 0) { nz ++; at = i; }
    const int ok = (nz == 1 && at == want && f[want] == 1);
    print0("[qc layout] delta at (x%d y%d z%d t%d, s%d c%d, lane %d): %ld nonzero "
           "at %ld, expected 1 at %ld => %s\n",
           px, py, pz, pt, ps, pc, lane, nz, at, want, ok ? "PASS" : "FAIL");
    if (!ok) fails ++;
    free(u); qc_free(f);
  }

  // -- the operator agrees about which index is spin ------------------------
  // Applies D to a single-component source through the public accessors alone.
  // Two structural facts pin the reading:
  //   on site, D is the clover diagonal (the neighbors are zero), and the
  //     clover is block diagonal in chirality -- clover_kernel_c splits the
  //     twelve components as 6*block + a with the component index s*Nc + c, so
  //     block 0 is spins {0,1} and block 1 is spins {2,3}.  A source in spin 1
  //     therefore cannot reach spins 2 or 3.
  //   at the neighbor, the link is a full 3x3 matrix, so all three colors
  //     appear, while the Wilson spin projector couples the source spin to one
  //     other, so exactly two spins do.
  // Exchange the two indices and both counts move.
  //
  // The neighbor is taken in y, and the source sits on an odd site, for reasons
  // the mutation runs found: x and x+1 share a half-lattice index under every
  // layout (they differ only in parity), so an x neighbor cannot tell the
  // block-major order from the lexicographic one; and at an even site the
  // parity term vanishes, so a map that dropped it would still look right.
  {
    const REAL kappa = 0.1, csw = 1.0;
    const long csz = clover_size_full();
    REAL * U = malloc(sizeof(REAL) * gauge_size);
    // Deterministic pseudorandom links.  They are not SU(3), which D does not
    // require: this checks index structure, not gauge dynamics.  Each rank
    // builds the same local slab, so the field is not continuous across ranks
    // either -- also irrelevant, since the sites read below are interior.
    unsigned st = 12345u;
    for (long i = 0; i < (long) gauge_size; i ++) {
      st = st * 1664525u + 1013904223u;
      U[i] = (REAL) ((double) (st >> 8) / 16777216.0 - 0.5);
    }
    REAL * F  = calloc(csz, sizeof(REAL));
    compute_Fmunu(U, F);
    REAL * Uk = malloc(sizeof(REAL) * gauge_alloc_size);
    REAL * Fc = malloc(sizeof(REAL) * csz);
    prepare_gauge(Uk, U, kappa);
    prepare_F(Fc, F, kappa, csw, /*invert_odd*/ 0);

    const long fn = qc_field_elements();
    REAL * in  = qc_alloc(fn);
    REAL * out = qc_alloc(fn);
    qc_field_zero(in);
    const int px = 5, py = 2, pz = 2, pt = 2, ps = 1, pc = 2, lane = 0;
    const long ssite = qc_site_index(px, py, pz, pt);
    in[qc_component_index(ssite, ps, pc, 0, lane)] = 1;
    D(out, in, Uk, Fc, 1);

    double same_block = 0, other_block = 0;
    for (int s = 0; s < Ns; s ++)
      for (int c = 0; c < Nc; c ++)
        for (int ri = 0; ri < 2; ri ++) {
          const double v
            = fabs((double) out[qc_component_index(ssite, s, c, ri, lane)]);
          if (s < 2) same_block += v; else other_block += v;
        }
    const double leak = (same_block > 0) ? other_block / same_block : 1;

    const long nsite = qc_site_index(px, py + 1, pz, pt);
    int nspin = 0, ncolor = 0;
    for (int s = 0; s < Ns; s ++) {
      double m = 0;
      for (int c = 0; c < Nc; c ++)
        for (int ri = 0; ri < 2; ri ++)
          m += fabs((double) out[qc_component_index(nsite, s, c, ri, lane)]);
      if (m > 0) nspin ++;
    }
    for (int c = 0; c < Nc; c ++) {
      double m = 0;
      for (int s = 0; s < Ns; s ++)
        for (int ri = 0; ri < 2; ri ++)
          m += fabs((double) out[qc_component_index(nsite, s, c, ri, lane)]);
      if (m > 0) ncolor ++;
    }
    const int ok = (leak < 1e-12 && nspin == 2 && ncolor == Nc);
    print0("[qc layout] D on a spin-%d color-%d source: chirality leak %.3e "
           "(want 0), +y neighbor spans %d/%d spins (want 2) and %d/%d colors "
           "(want %d) => %s\n",
           ps, pc, leak, nspin, Ns, ncolor, Nc, Nc, ok ? "PASS" : "FAIL");
    if (!ok) fails ++;

    free(U); free(F); free(Uk); free(Fc); qc_free(in); qc_free(out);
    sap_free_doo_inv_cache();
  }

  // Every check above is rank-local, and the per-check lines print on rank 0
  // only, so a failure confined to another rank would otherwise never be seen.
  double all_fails = fails;
  allreduce_sum_d(&all_fails, 1);
  print0("test_qc_layout: %d check(s) failed across %d rank(s) => %s\n",
         (int) all_fails, qc_num_ranks(), all_fails == 0 ? "PASS" : "FAIL");
}

// --- ./qc params: the public parameter structs -----------------------------
// This arm answers two questions about the structs qc.h publishes.
//
// (1) Do the defaults carry this build's tuned values?  A caller who fills a
//     struct with qc_default_setup_params and changes only kappa must get the
//     method the repo is regression-tested at, so each field is compared
//     against the macro it comes from.
//
// (2) Does the validator actually reject what it claims to?  Checking that a
//     good struct passes proves nothing -- `return NULL;` would pass it.  So
//     every field is broken in turn and the check must fire, and must fire
//     naming that field.  This is what dictates the validator's shape:
//     qc_setup_params_problem returns a message rather than aborting where it
//     finds the fault, because a guard that calls MPI_Abort can only be driven
//     once per process, not once per field.
const char * qc_setup_params_problem(const qc_setup_params * p);
const char * qc_solve_params_problem(const qc_solve_params * p);
void qc_setup_params_to_mg(const qc_setup_params * p, mg_params * m);

void test_qc_params() {
  int fails = 0;

  // -- the public defaults are the internal ones ----------------------------
  // Compared field by field against mg_params_default rather than against the
  // tuned numbers themselves.  The question this can answer is whether qc.c
  // copies each mg_params field to the public field of the same name -- a
  // transposed or omitted pair is the failure it catches, and repeating the
  // values here would only add a second copy to go stale when one is retuned.
  // It cannot check that a default is the right value; the correlator's own
  // regression bar is what covers that.
  qc_setup_params sp;
  qc_default_setup_params(&sp);
  {
    mg_params m; mg_params_default(&m);
    int ok = sp.kappa == 0.0 && sp.csw == 0.0
          && sp.smoother_sweeps        == m.smoother_sweeps
          && sp.sap_block_iter         == m.sap_block_iter
          && sp.coarse_smoother_sweeps == m.coarse_smoother_sweeps
          && sp.coarse_sap_block_iter  == m.coarse_sap_block_iter
          && (int) sp.cycle            == (int) m.cycle
          && sp.w_mu                   == m.w_mu
          && sp.k_iters                == m.k_iters
          && sp.setup_k_tol            == m.setup_k_tol
          && sp.solve_k_tol            == m.solve_k_tol
          && sp.coarse_max_iter        == m.coarse_max_iter
          && sp.coarse_tol             == m.coarse_tol
          && sp.coarse_restart         == m.coarse_restart
          && sp.setup_init_sweeps      == m.setup_init_sweeps
          && sp.setup_sweeps           == m.setup_sweeps
          && sp.coarse_setup_sweeps    == m.coarse_setup_sweeps;
    print0("[qc params] setup defaults match mg_params_default => %s\n",
           ok ? "PASS" : "FAIL");
    if (!ok) fails ++;
  }

  // -- and every field reaches the internal struct --------------------------
  // The check above covers the outward copy, internal defaults to public
  // struct.  This one covers the inward copy that qc_setup makes, which
  // nothing else reaches: no test calls qc_setup, so an omitted line there
  // would surface only as a solve behaving oddly.  Each field is set to a
  // value distinct from its default and from every other field, so an omitted
  // line leaves the default in place and a transposed pair swaps two values
  // -- both fail here.
  {
    qc_setup_params t; qc_default_setup_params(&t);
    t.smoother_sweeps = 11; t.sap_block_iter = 12;
    t.coarse_smoother_sweeps = 13; t.coarse_sap_block_iter = 14;
    t.cycle = QC_CYCLE_W;                  // the default is QC_CYCLE_K
    t.w_mu = 15; t.k_iters = 16;
    t.setup_k_tol = 0.11; t.solve_k_tol = 0.12;
    t.coarse_max_iter = 17; t.coarse_tol = 0.13; t.coarse_restart = 18;
    t.setup_init_sweeps = 19; t.setup_sweeps = 20; t.coarse_setup_sweeps = 21;

    mg_params m;
    qc_setup_params_to_mg(&t, &m);
    int ok = m.smoother_sweeps == 11 && m.sap_block_iter == 12
          && m.coarse_smoother_sweeps == 13 && m.coarse_sap_block_iter == 14
          && m.cycle == MG_CYCLE_W
          && m.w_mu == 15 && m.k_iters == 16
          && m.setup_k_tol == 0.11 && m.solve_k_tol == 0.12
          && m.coarse_max_iter == 17 && m.coarse_tol == 0.13
          && m.coarse_restart == 18
          && m.setup_init_sweeps == 19 && m.setup_sweeps == 20
          && m.coarse_setup_sweeps == 21;
    print0("[qc params] setup fields reach mg_params => %s\n",
           ok ? "PASS" : "FAIL");
    if (!ok) fails ++;
  }

  qc_solve_params vp;
  qc_default_solve_params(&vp);
  {
    int ok = vp.tol == (double) OUTER_TOL && vp.max_iter == OUTER_MAX
          && vp.restart == FGMRES_RESTART && vp.chroma_norm == 0;
    print0("[qc params] solve defaults match global_params.h => %s\n",
           ok ? "PASS" : "FAIL");
    if (!ok) fails ++;
  }

  // -- the defaults are themselves acceptable, once kappa is set ------------
  // Ordered before the rejection sweep on purpose: if the defaults did not pass
  // their own validator, every "rejected" result below would be meaningless.
  sp.kappa = 0.134;
  {
    const char * s = qc_setup_params_problem(&sp);
    const char * v = qc_solve_params_problem(&vp);
    print0("[qc params] defaults + kappa accepted => %s%s%s\n",
           (!s && !v) ? "PASS" : "FAIL",
           s ? " setup: " : "", s ? s : "");
    if (s || v) fails ++;
  }

  // -- every field, broken in turn ------------------------------------------
  // Each entry restores the struct, breaks one field, and requires a
  // complaint mentioning that field's name.  A guard deleted from the
  // validator turns its row here red rather than passing quietly.
  {
    int caught = 0, tried = 0;
    #define BREAKS(field, bad_value, needle)                                 \
      do {                                                                   \
        qc_setup_params t; qc_default_setup_params(&t); t.kappa = 0.134;     \
        t.field = bad_value;                                                 \
        const char * msg = qc_setup_params_problem(&t);                      \
        int ok = msg && strstr(msg, needle) != NULL;                         \
        tried ++; if (ok) caught ++;                                         \
        if (!ok)                                                             \
          print0("[qc params]   %-24s not rejected (got: %s)\n",             \
                 #field, msg ? msg : "no complaint");                        \
      } while (0)

    BREAKS(kappa,                  0.0, "kappa");
    BREAKS(smoother_sweeps,         -1, "smoother_sweeps");
    BREAKS(sap_block_iter,           0, "sap_block_iter");
    BREAKS(coarse_smoother_sweeps,  -1, "coarse_smoother_sweeps");
    BREAKS(coarse_sap_block_iter,    0, "coarse_sap_block_iter");
    BREAKS(cycle,      (qc_cycle) 7,    "cycle");
    BREAKS(setup_k_tol,            0.0, "setup_k_tol");
    BREAKS(solve_k_tol,           -1.0, "solve_k_tol");
    BREAKS(coarse_max_iter,          0, "coarse_max_iter");
    BREAKS(coarse_tol,             0.0, "coarse_tol");
    BREAKS(coarse_restart,           0, "coarse_restart");
    BREAKS(setup_init_sweeps,       -1, "setup_init_sweeps");
    BREAKS(setup_sweeps,            -1, "setup_sweeps");
    BREAKS(coarse_setup_sweeps,      0, "coarse_setup_sweeps");
    #undef BREAKS

    // w_mu and k_iters are only consulted for the cycle that uses them, so
    // each has to be broken with that cycle selected -- the default cycle
    // would let a zero through legitimately.
    {
      qc_setup_params t; qc_default_setup_params(&t); t.kappa = 0.134;
      t.cycle = QC_CYCLE_W; t.w_mu = 0;
      const char * msg = qc_setup_params_problem(&t);
      tried ++; if (msg && strstr(msg, "w_mu")) caught ++;
      else print0("[qc params]   w_mu (W-cycle) not rejected\n");
    }
    {
      qc_setup_params t; qc_default_setup_params(&t); t.kappa = 0.134;
      t.cycle = QC_CYCLE_K; t.k_iters = 0;
      const char * msg = qc_setup_params_problem(&t);
      tried ++; if (msg && strstr(msg, "k_iters")) caught ++;
      else print0("[qc params]   k_iters (K-cycle) not rejected\n");
    }

    print0("[qc params] %d/%d bad setup fields rejected by name => %s\n",
           caught, tried, caught == tried ? "PASS" : "FAIL");
    if (caught != tried) fails ++;
  }

  {
    int caught = 0, tried = 0;
    #define BREAKV(field, bad_value, needle)                                 \
      do {                                                                   \
        qc_solve_params t; qc_default_solve_params(&t);                      \
        t.field = bad_value;                                                 \
        const char * msg = qc_solve_params_problem(&t);                      \
        int ok = msg && strstr(msg, needle) != NULL;                         \
        tried ++; if (ok) caught ++;                                         \
        if (!ok)                                                             \
          print0("[qc params]   %-24s not rejected (got: %s)\n",             \
                 #field, msg ? msg : "no complaint");                        \
      } while (0)

    BREAKV(tol,       0.0, "tol");
    BREAKV(max_iter,    0, "max_iter");
    BREAKV(restart,     0, "restart");
    #undef BREAKV

    print0("[qc params] %d/%d bad solve fields rejected by name => %s\n",
           caught, tried, caught == tried ? "PASS" : "FAIL");
    if (caught != tried) fails ++;
  }

  // -- a zeroed struct is caught --------------------------------------------
  // The failure this whole design exists to prevent: the
  // designated-initializer habit, which leaves every unnamed field at zero.
  // It must not reach a solve.
  {
    qc_setup_params z;
    memset(&z, 0, sizeof z);
    z.kappa = 0.134;
    z.csw = 1.87567;
    qc_solve_params w;
    memset(&w, 0, sizeof w);
    w.tol = 1e-10;
    w.max_iter = 100;
    int ok = qc_setup_params_problem(&z) != NULL
          && qc_solve_params_problem(&w) != NULL;
    print0("[qc params] zero-filled structs rejected => %s\n",
           ok ? "PASS" : "FAIL");
    if (!ok) fails ++;
  }

  double all_fails = fails;
  allreduce_sum_d(&all_fails, 1);
  print0("test_qc_params: %d check(s) failed across %d rank(s) => %s\n",
         (int) all_fails, qc_num_ranks(), all_fails == 0 ? "PASS" : "FAIL");
}
