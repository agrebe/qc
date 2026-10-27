// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

#include "test.h"
// Declared here rather than by including bootstrap.h: that file's REAL is the
// single-precision one, and this entry point takes none, so a local declaration
// keeps its REAL-typed prototypes out of this translation unit.
int bootstrap_selfcheck_batching(void);
#include "coarse.h"
#include "comm.h"
#include "mixed.h"
#include <stdio.h>
#include <string.h>

// Run both twins of a twinned entry point (the test/kernel_tests.c family).
// Mixed is the only build where the twins have distinct names: there the plain
// name is the _d twin and reaching the asm one takes an explicit _f.  At
// PRECISION=single or =double there is exactly one twin and mangle.h already
// points the plain name at it (_f and _d respectively), so the plain call alone
// covers those.  Keeping the _f reference inside #if MIXED also keeps it out of
// a double build, where it would be an undefined symbol at link.
//
// The ## is load-bearing: a macro argument is not macro-expanded when it is an
// operand of ##, so name##_f pastes the raw name, while the bare name() below
// is expanded and goes through mangle.h to name_d.
#if MIXED
  #define RUN_TWINNED(name) \
    do { extern void name##_f(void); name##_f(); name(); } while (0)
#else
  #define RUN_TWINNED(name) name()
#endif

#define DEFAULT_ARM "correlator"     // what a bare ./qc runs

// Every command, in the order `all` sweeps them.  This table drives both the
// usage text and the sweep, so the arm list has one home; the names are
// matched by run_one() below, and a name here that run_one() does not know
// falls into its unknown-command branch the first time `all` is run.
//
// A_ALL   swept by `all`.  Cleared for the throughput benches (they report
//         GF/s, not PASS/FAIL, and numbers taken after a dozen other tests
//         have churned the caches and DRAM state are not worth reading); for
//         `coarse`, which holds several fine multi-rhs double fields at once,
//         so its footprint scales with the lattice and dwarfs every other
//         gate; and for `halo` and `cbuild2`, which have no self-contained
//         verdict -- they write per-rank hash dumps that you diff between a
//         1-rank and an N-rank run with test/compare-halo-dumps.py, so one
//         run alone proves nothing.
// A_GAUGE reads the ensemble config named in test.h.  read_gauge asserts on a
//         missing file, so these ABORT rather than fail -- hence they sort
//         last, and an `all` on a machine without the ensemble still gets
//         through every synthetic gate before it dies.
// A_1RANK valid only on an undecomposed lattice, so `all` skips it and run_one
//         refuses it above one rank.  Two reasons, both structural rather than
//         fixable: the kernel gates allocate fermion fields at bare ferm_size
//         and call the stencil kernels directly, so a face site's neighbor
//         read runs off the end; and the synthetic coarse selftests invent a
//         fixed lattice (4^4, 6^4, ...) that no rank grid divides, built
//         identically on every rank, so a decomposed run just re-solves the
//         same replicated problem.  Both compare local arithmetic that
//         decomposition does not change, and both are fully exercised at one
//         rank.  The tests that do validate the decomposition are cdirect /
//         cbuild2 / halo.
//         Tested against comm_size() rather than the grid macros only for
//         uniformity with the flags above; comm_init dies unless nranks equals
//         MPI_GRID_PZ * MPI_GRID_PT, so the two are the same predicate.
//
// Otherwise the order is cheapest-and-most-fundamental first: kernel gates,
// then the synthetic coarse/MG selftests, then the halo pair, then the
// gauge-dependent integration tests.  Nothing here needs teardown between
// arms -- the only two that register the global hierarchy (coarse.c's
// g_levels) are `mixed` and `correlator`, and both pair mg_hierarchy_build
// with mg_hierarchy_free.
#define A_ALL   1
#define A_GAUGE 2
#define A_1RANK 4
static const struct {
  const char * name;
  const char * help;
  int          flags;
} ARMS[] = {
  { "cloverasm",   "clover asm-vs-C correctness gate",                 A_ALL },
  { "wilsonasm",   "wilson asm-vs-C correctness gate",                 A_ALL | A_1RANK },
  { "wblockasm",   "Schwarz-block kernel family gate",                 A_ALL | A_1RANK },
  { "batching",    "null-vector batching gather/scatter gate",         A_ALL },
  { "densecheck",  "dense.c GEMM / LU / Bunch-Kaufman gate",           A_ALL },
  { "csapcheck",   "See_inv tile-kernel gate (SDE-friendly)",          A_ALL },
  { "ctransfer",   "coarse transfer R P = I (level >= 1)",             A_ALL },
  { "cdslash",     "apply_Dc_lvl vs apply_Dc / apply_Dc_ref",          A_ALL },
  { "cbuild",      "build_coarse_op_lvl Galerkin identity",            A_ALL | A_1RANK },
  { "wbuild",      "windowed build_coarse_op_lvl vs full-field probe", A_ALL | A_1RANK },
  { "cbottom",     "coarse_solve_full_lvl geometry-agnostic solve",    A_ALL | A_1RANK },
  { "mg3",         "synthetic 3-level cycle (exercises coarse_sap)",   A_ALL | A_1RANK },
  { "localortho",  "local orthonormalization of the test vectors",     A_ALL },
  { "layout",      "qc.h published field layout vs the build",         A_ALL },
  { "params",      "qc.h parameter defaults and their validator",      A_ALL },
  { "cdirect",     "depth-0 direct-build halo validation",             A_ALL },
  { "cbuild2",     "depth-1->2 build hash dumps (diff across ranks)",  0 },
  { "halo",        "per-rank stencil hash dumps (diff across ranks)",  0 },
  { "eoclover",    "even-odd clover operator",                         A_ALL | A_GAUGE },
  { "schwarz",     "red-black Schwarz smoother",                       A_ALL | A_GAUGE },
  { "mixed",       "double outer FGMRES + single MG V-cycle",          A_ALL | A_GAUGE },
  { "correlator",  "pion correlator integration test",                 A_ALL | A_GAUGE },
  { "bicgstab",    "pion correlator via eo Richardson + BiCGStab",     A_GAUGE },
  { "coarse",      "full coarse-op test (double; large temps)",        0 },
  { "dcbench",     "apply_Dc throughput",                              0 },
  { "csapbench",   "See_inv-apply kernel bench (single twin)",         0 },
  { "cloverbench", "clover kernel throughput",                         0 },
  { "wilsonbench", "wilson kernel throughput",                         0 },
};
#define N_ARMS ((int) (sizeof(ARMS) / sizeof(ARMS[0])))

static void usage(void) {
  print0("usage: qc [command]\n\n");
  print0("With no command, runs '%s'.\n\ncommands:\n", DEFAULT_ARM);
  for (int i = 0; i < N_ARMS; i++) {
    char ann[80];
    ann[0] = '\0';
    if (!(ARMS[i].flags & A_ALL))  strcat(ann, " [not in 'all']");
    if (ARMS[i].flags & A_1RANK)   strcat(ann, " [1x1 grid only]");
    if (ARMS[i].flags & A_GAUGE)   strcat(ann, " [needs gauge file]");
    if (ann[0]) print0("  %-12s %-51s%s\n", ARMS[i].name, ARMS[i].help, ann);
    else        print0("  %-12s %s\n",      ARMS[i].name, ARMS[i].help);
  }
  print0("  %-12s %s\n", "all", "every gate above except those marked [not in 'all']");
}

static int arm_flags(const char * name) {
  for (int i = 0; i < N_ARMS; i++)
    if (!strcmp(ARMS[i].name, name)) return ARMS[i].flags;
  return -1;                                  // not a table entry
}

// Dispatch one command.  Returns the arm's exit status where it has one; see
// the note in main() on what that does and does not cover.
static int run_one(const char * cmd) {
  int rc = 0;
  // Refuse rather than crash: these read past their own fields once the
  // lattice is decomposed (see A_1RANK above).  A skip is not a failure, so
  // rc stays 0.  Test fl >= 0 first: the unknown-name sentinel is -1, whose
  // bits would match every flag and divert unknown commands away from the
  // usage message below.
  const int fl = arm_flags(cmd);
  if (fl >= 0 && (fl & A_1RANK) && comm_size() > 1) {
    print0("%s is valid only on an undecomposed lattice -- rebuild with "
           "MPI_GRID_PZ = MPI_GRID_PT = 1 to run it.\n", cmd);
    return 0;
  }
  if      (!strcmp(cmd, "cloverasm"))   RUN_TWINNED(test_clover_asm);
  else if (!strcmp(cmd, "wilsonasm"))   RUN_TWINNED(test_wilson_asm);
  else if (!strcmp(cmd, "wblockasm"))   RUN_TWINNED(test_wilson_block_asm);
  else if (!strcmp(cmd, "ctransfer"))   rc = mg_coarse_transfer_selftest();
  else if (!strcmp(cmd, "cdslash"))     rc = mg_coarse_dslash_selftest();
  else if (!strcmp(cmd, "cbuild"))      rc = mg_coarse_build_selftest();
  else if (!strcmp(cmd, "wbuild"))      rc = mg_coarse_build_windowed_selftest();
  else if (!strcmp(cmd, "cbottom"))     rc = mg_coarse_bottom_selftest();
  else if (!strcmp(cmd, "mg3"))         rc = mg_3level_selftest();
  else if (!strcmp(cmd, "batching"))    rc = bootstrap_selfcheck_batching();
  else if (!strcmp(cmd, "densecheck"))  test_dense();
  else if (!strcmp(cmd, "localortho"))  test_local_ortho();
  else if (!strcmp(cmd, "layout"))      test_qc_layout();
  else if (!strcmp(cmd, "params"))      test_qc_params();
  else if (!strcmp(cmd, "cdirect"))     test_coarse_direct();
  else if (!strcmp(cmd, "cbuild2"))     test_build_lvl_halo();
  else if (!strcmp(cmd, "halo"))        test_halo_stencil();
  else if (!strcmp(cmd, "eoclover"))    test_eo_clover();
  else if (!strcmp(cmd, "schwarz"))     test_schwarz();
  else if (!strcmp(cmd, "mixed"))       test_mixed();
  else if (!strcmp(cmd, "correlator"))  test_pion_correlator();
  else if (!strcmp(cmd, "bicgstab"))    test_pion_correlator_bicgstab();
  else if (!strcmp(cmd, "coarse"))      test_coarse();
  else if (!strcmp(cmd, "dcbench"))     test_apply_dc_bench();
  else if (!strcmp(cmd, "csapbench")) {
    // The _f twin only exists in a build that instantiates the single
    // precision, i.e. PRECISION=mixed or =single; PRECISION=double has _d
    // twins only.
#if COARSE_SAP_DENSE_INV && (MIXED || MG_SINGLE)
    extern void coarse_csap_bench_f(void);  // the _f twin: what prod's MG runs
    coarse_csap_bench_f();
#else
    print0("csapbench needs a COARSE_SAP_DENSE_INV=1 build at PRECISION=mixed or single\n");
#endif
  }
  else if (!strcmp(cmd, "csapcheck")) {
#if COARSE_SAP_DENSE_INV && (MIXED || MG_SINGLE)
    extern void coarse_csap_check_f(void);
    coarse_csap_check_f();
#else
    print0("csapcheck needs a COARSE_SAP_DENSE_INV=1 build at PRECISION=mixed or single\n");
#endif
  }
  else if (!strcmp(cmd, "cloverbench")) RUN_TWINNED(test_clover_kernel);
  else if (!strcmp(cmd, "wilsonbench")) RUN_TWINNED(test_wilson_kernel);
  else {
    print0("qc: unknown command '%s'\n\n", cmd);
    usage();
    rc = 1;
  }
  return rc;
}

int main(int argc, char ** argv) {
  // no-op at MPI=0 / single rank
  comm_init(&argc, &argv);
  // line-buffer: live progress under SDE/redirect
  setvbuf(stdout, NULL, _IOLBF, 0);

  const char * cmd = argc > 1 ? argv[1] : DEFAULT_ARM;
  int rc = 0;

  if (!strcmp(cmd, "all")) {
    int n = 0, skipped = 0, warned = 0;
    for (int i = 0; i < N_ARMS; i++) {
      if (!(ARMS[i].flags & A_ALL)) continue;
      if ((ARMS[i].flags & A_1RANK) && comm_size() > 1) {
        print0("\n======== all: %s skipped (1x1 grid only) ========\n",
               ARMS[i].name);
        skipped ++;
        continue;
      }
      if ((ARMS[i].flags & A_GAUGE) && !warned) {
        warned = 1;
        print0("\n-------- the remaining arms read the ensemble gauge config; without\n"
               "-------- it read_gauge asserts and this sweep aborts here.  Every\n"
               "-------- gate that does not need it has already run above.\n");
      }
      print0("\n======== all [%d]: %s -- %s ========\n", ++n, ARMS[i].name,
             ARMS[i].help);
      rc |= run_one(ARMS[i].name);
    }
    // Only seven selftests (ctransfer, cdslash, cbuild, wbuild, cbottom, mg3,
    // batching) return a status; the rest are void and report by printing.
    // Say so rather than letting a 0 exit code imply a clean sweep.
    print0("\n======== all: %d arms run, %d skipped ========\n", n, skipped);
    print0("Some arms report through the exit code (now %d); the rest print their\n"
           "own PASS/FAIL above -- scan the output, do not trust $? alone.\n", rc);
  } else {
    rc = run_one(cmd);
  }

  comm_finalize();
  return rc;
}
