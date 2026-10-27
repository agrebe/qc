// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

// Symbol-mangling layer for the mixed-precision build.
//
// Included by global_params.h (right after precision.h), so every translation
// unit gets it automatically and uniformly.  It rewrites every external symbol
// of the operator / MG / inner-solver / utils layer to a precision-suffixed
// name (foo -> foo_f or foo_d via PFX), keyed on the per-TU MG_SINGLE.  Thus:
//   * the single instantiations of the templated bodies define _f;
//   * the double instantiations and the entire double-only driver layer
//     (test.c, runner.c, solve(), mixed_bridge's calls) define and call _d --
//     so plain-name references in the drivers resolve correctly.
// The bodies and drivers keep using the plain names; the preprocessor renames.
// New external function -> add one line here.
//
// Deliberately not listed:
//   * static helpers and static file-scope state  -> internal linkage, no clash
//   * the hand-written asm entry points (wilson_kernel_avx2/512[_masked],
//     clover_kernel_avx2, coarse_dslash_kernel_avx2/512) -> single-precision
//     symbols defined in .s with fixed `float` ABI; they must keep their real
//     names, and are only referenced under #if ASM && MG_SINGLE.
//   * solve() and the mixed bridges -> they are the double-only orchestrator
//     layer, not templated (see mixed_bridge.c).
//   * bicgstab_mixed (solver.c) -> a seam entry going the other way: the
//     single twin defines it with literal double arguments so the double
//     driver can call into a single-precision solve.  One name on both sides,
//     so unmangled.
//   * the coarse-solve config toggles (MG_COARSE_EO / COARSE_USE_ASM) -> plain
//     compile-time #defines in global_params.h, not symbols, so nothing to mangle.

#ifndef MANGLE_H
#define MANGLE_H

// --- clover_ferm.c ----------------------------------------------------------
#define wilson_kernel_c        PFX(wilson_kernel_c)
#define wilson_kernel_dir_c    PFX(wilson_kernel_dir_c)
#define wilson_kernel          PFX(wilson_kernel)
#define wilson_kernel_block    PFX(wilson_kernel_block)
#define wilson_kernel_block_c  PFX(wilson_kernel_block_c)
#define wilson_block_itab_init PFX(wilson_block_itab_init)
#define wilson_kernel_block_itab PFX(wilson_kernel_block_itab)
#define wilson_halo_half       PFX(wilson_halo_half)
#define wilson_halo_sap        PFX(wilson_halo_sap)
#define wilson_halo_sap_stats_get   PFX(wilson_halo_sap_stats_get)
#define wilson_halo_sap_stats_reset PFX(wilson_halo_sap_stats_reset)
#define wilson_set_fghost      PFX(wilson_set_fghost)
#define wilson_zero_fghost     PFX(wilson_zero_fghost)
#define clover_kernel          PFX(clover_kernel)
#define clover_kernel_c        PFX(clover_kernel_c)
#define clover_kernel_ow       PFX(clover_kernel_ow)
#define H                      PFX(H)
#define H_half                 PFX(H_half)
#define H_oe                   PFX(H_oe)
#define H_eo                   PFX(H_eo)
#define D_oo_inv               PFX(D_oo_inv)
#define H_hat                  PFX(H_hat)
#define D                      PFX(D)
#define D_hat                  PFX(D_hat)
#define D_hat_residual         PFX(D_hat_residual)
#define D_residual             PFX(D_residual)
#define D_residual_eo          PFX(D_residual_eo)
#define D_si                   PFX(D_si)

// --- coarse.c ---------------------------------------------------------------
#define local_orthonormalize        PFX(local_orthonormalize)
#define transfer_stats_reset        PFX(transfer_stats_reset)
#define vcycle_stats_reset          PFX(vcycle_stats_reset)
#define other_stats_reset           PFX(other_stats_reset)
#define apply_P                     PFX(apply_P)
#define apply_P_block_cached        PFX(apply_P_block_cached)
#define apply_R                     PFX(apply_R)
#define apply_R_even                PFX(apply_R_even)
#define mg_restrict_fuse_normalize  PFX(mg_restrict_fuse_normalize)
#define coarse_neighbor             PFX(coarse_neighbor)
#define build_coarse_op             PFX(build_coarse_op)
#define build_coarse_op_direct      PFX(build_coarse_op_direct)
#define free_coarse_op              PFX(free_coarse_op)
#define coarser_vol                 PFX(coarser_vol)
#define local_orthonormalize_coarse PFX(local_orthonormalize_coarse)
#define apply_P_coarse              PFX(apply_P_coarse)
#define apply_R_coarse              PFX(apply_R_coarse)
#define coarse_neighbor_lvl         PFX(coarse_neighbor_lvl)
#define apply_Dc_lvl                PFX(apply_Dc_lvl)
#define build_coarse_op_lvl         PFX(build_coarse_op_lvl)
#define alloc_level_dc_ghost        PFX(alloc_level_dc_ghost)
#define coarse_solve_stats_reset    PFX(coarse_solve_stats_reset)
#define coarse_solve_full           PFX(coarse_solve_full)
#define coarse_solve_full_lvl       PFX(coarse_solve_full_lvl)
#define mg_cycle                    PFX(mg_cycle)
#define coarse_sap                  PFX(coarse_sap)
#define coarse_sap_cache_free       PFX(coarse_sap_cache_free)
#define coarse_sap_flops            PFX(coarse_sap_flops)
#define coarse_sap_flops_reset      PFX(coarse_sap_flops_reset)
#define csap_profile_reset          PFX(csap_profile_reset)
#define coarse_csap_bench           PFX(coarse_csap_bench)
#define coarse_csap_check           PFX(coarse_csap_check)
#define mg_setup                    PFX(mg_setup)
#define mg_add_coarse_level         PFX(mg_add_coarse_level)
#define mg_get_level                PFX(mg_get_level)
#define mg_trim_to                  PFX(mg_trim_to)
#define mg_vcycle                   PFX(mg_vcycle)
#define apply_dc_stats_reset        PFX(apply_dc_stats_reset)
#define apply_dc_stats_report       PFX(apply_dc_stats_report)
#define apply_schur_stats_reset     PFX(apply_schur_stats_reset)
#define mg_get_vcycle_profile       PFX(mg_get_vcycle_profile)
#define bottom_solve_diag_reset     PFX(bottom_solve_diag_reset)
#define bottom_solve_diag_report    PFX(bottom_solve_diag_report)
#define apply_Dc_ref                PFX(apply_Dc_ref)
#define apply_Dc                    PFX(apply_Dc)
#define build_coarse_eo_op          PFX(build_coarse_eo_op)
#define free_coarse_eo_op           PFX(free_coarse_eo_op)
#define apply_coarse_schur          PFX(apply_coarse_schur)
#define coarse_solve_eo             PFX(coarse_solve_eo)
#define nullvec_orthonormality_error PFX(nullvec_orthonormality_error)

// --- solver.c (inner solvers; solve() is intentionally excluded) ------------
#define time_used            PFX(time_used)
#define prepare_gauge        PFX(prepare_gauge)
#define prepare_F            PFX(prepare_F)
#define richardson           PFX(richardson)
#define bicgstab             PFX(bicgstab)
#define fgmres               PFX(fgmres)
#define fgmres_get_profile   PFX(fgmres_get_profile)
#define red_black_schwarz    PFX(red_black_schwarz)
#define red_black_schwarz_resfold PFX(red_black_schwarz_resfold)
#define sap_discard_odd_output_next PFX(sap_discard_odd_output_next)
#define sap_fuse_prolong_next PFX(sap_fuse_prolong_next)
#define sap_bulk_blk_parity  PFX(sap_bulk_blk_parity)
#define sap_emit_arnoldi_w_next PFX(sap_emit_arnoldi_w_next)
#define sap_arnoldi_w_filled PFX(sap_arnoldi_w_filled)
#define sap_emit_arnoldi_dots_next PFX(sap_emit_arnoldi_dots_next)
#define sap_arnoldi_dots_filled PFX(sap_arnoldi_dots_filled)
#define sap_free_doo_inv_cache  PFX(sap_free_doo_inv_cache)
#define sap_stats_reset      PFX(sap_stats_reset)
#define sap_stats_get        PFX(sap_stats_get)
#define sap_halo_stats_get   PFX(sap_halo_stats_get)
#define sap_fusion_stats_get PFX(sap_fusion_stats_get)

// --- test/kernel_tests.c (twinned: both precisions live in one mixed build) -
// Without these the two instantiations would define the same symbol and clash
// at link; with them the driver reaches a chosen twin by its suffixed name.
#define test_clover_kernel   PFX(test_clover_kernel)
#define test_wilson_kernel   PFX(test_wilson_kernel)
#define test_clover_asm      PFX(test_clover_asm)
#define test_wilson_asm      PFX(test_wilson_asm)
#define test_wilson_block_asm PFX(test_wilson_block_asm)

// --- utils.c (REAL-typed field/linear-algebra helpers) ----------------------
// Twinned because the templated MG bodies (coarse.c / solver.c) call these
// expecting their own REAL, while the double driver layer calls the same names
// expecting double; one shared utils.o cannot serve both in the mixed binary.
// (DOTC_KERNEL / CAXPY_KERNEL in utils.h are macros, not symbols, so they are
//  precision-agnostic and need no mangling.)
#define read_gauge               PFX(read_gauge)
#define compute_Fmunu            PFX(compute_Fmunu)
#define matmul                   PFX(matmul)
#define invert_dense_block       PFX(invert_dense_block)
#define copy_gauge               PFX(copy_gauge)
#define rescale_multi_ferm       PFX(rescale_multi_ferm)
#define copy_multi_ferm          PFX(copy_multi_ferm)
#define multi_half_ferm_norm     PFX(multi_half_ferm_norm)
#define multi_half_ferm_norm_rhs PFX(multi_half_ferm_norm_rhs)
#define zero_multi_half_ferm     PFX(zero_multi_half_ferm)
#define copy_multi_half_ferm     PFX(copy_multi_half_ferm)
#define cdot_ferm                PFX(cdot_ferm)
#define caxpy_ferm               PFX(caxpy_ferm)

#endif // MANGLE_H
