// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

// Pulled in for its own sake, not as a convenience: the kernel-benchmark names
// below are precision-mangled, so this header only declares what its callers
// call if mangle.h (via global_params.h) is already in scope.  runner.c
// includes test.h first, so relying on the caller to have included it would
// leave the declarations unmangled while the call sites were mangled -- a
// link-time mismatch that reads as a missing function.
#include "global_params.h"

// --- gauge configuration for the arms that need one -------------------------
// Test fixture data, so it lives here rather than in global_params.h: nothing
// in the library names a configuration file.  read_gauge and
// qc_read_gauge_file both take a path, so a caller building on qc never sees
// this.  Either a LIME container or a payload extracted from one works;
// read_gauge detects which, and checks the file's recorded geometry against
// this build, so a path that does not match aborts with a message rather than
// solving on whatever bytes it found.
#define GAUGE_PATH "test/sample_config.lime"

// --- halo-harness coarse geometry -------------------------------------------
// Level 1 -> 2 aggregation block of the hand-built levels in the halo
// harness (test_halo_*).  Production takes this block from BOOTSTRAP_BLK_LVL
// in global_params.h; the two need not agree.  The harness's coarsest-level
// extents (HALO_C2*) assume an edge of 2 in every direction.
#define COARSE_BLK1       {2, 2, 2, 2}

// Kernel benchmarks (test/kernel_tests.c).  That file is twinned, so these two
// names are precision-mangled and a mixed build contains both instantiations:
// test_clover_kernel_f times the asm kernels at REAL=float,
// test_clover_kernel_d the C kernels at REAL=double.  A plain call from the
// (double) driver layer resolves to the _d twin via mangle.h; to reach the
// single twin, declare it explicitly --
// `extern void test_clover_kernel_f(void);` -- as runner.c does.
// PRECISION=double builds no _f twin and PRECISION=single no _d twin, so any
// explicit reference needs the same #if MIXED || MG_SINGLE gate used elsewhere.
void test_clover_kernel();
void test_wilson_kernel();
// Correctness gates for the asm kernels.  Each only covers assembly in a twin
// where its dispatcher routes to it (CLOVER_ASM_LIVE / WILSON_ASM_LIVE in
// clover_ferm.h) -- so as with the benches, reaching the asm means calling the
// _f twin by name.
void test_clover_asm();
void test_wilson_asm();
void test_wilson_block_asm();

void test_dense();     // dense.c's GEMM / LU / Bunch-Kaufman (fp64, untwinned)
void test_pion_correlator();
void test_pion_correlator_bicgstab();
void test_qc_layout();        // qc.h's published field order vs this build
void test_qc_params();        // qc.h's parameter defaults and their validator
void test_eo_clover();
void test_schwarz();
void test_coarse();
void test_coarse_direct();
void test_local_ortho();
void test_apply_dc_bench();
void test_mixed();

// Coarse / multilevel selftests (moved out of coarse.c; defined in test.c).
int mg_coarse_transfer_selftest(void);
int mg_coarse_dslash_selftest(void);
int mg_coarse_build_selftest(void);
int mg_coarse_build_windowed_selftest(void);
int mg_coarse_bottom_selftest(void);
int mg_3level_selftest(void);

void test_halo_stencil();
void test_build_lvl_halo();   // build_coarse_op_lvl cross-rank (coarse-only)
