// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

// Symbol-hygiene gate for libqc.a.
//
// The library is one merged object (see the Makefile's library section), and
// the great majority of the names in it are internal.  This program is what
// proves the step that hides them worked, by being hostile in both directions
// at once:
//
//   * It defines its own solve(), print0() and cos(), and fails to link on a
//     duplicate definition if the archive still defines any of them.  solve()
//     and print0() are qc's own internals, hidden when the library is built.
//     cos() guards something different and is the more interesting of the two
//     cases: the library must not carry a copy of libm at all.  Going through
//     the compiler driver, the library link absorbs Intel's libm whole -- cos()
//     and sin() and sqrt() included -- so linking libqc would silently replace
//     every transcendental in the host application.  -nostdlib on that link is
//     what stops it, and this is the regression gate: drop the flag and cos()
//     reappears, and this stops linking.
//
//   * It takes the address of every entry point qc.h declares.  If the
//     hiding step swept up something public, this fails to link on an
//     undefined reference.  That is the check that keeps qc.syms honest as the
//     API grows: adding a function to qc.h and forgetting qc.syms breaks the
//     build here rather than shipping an unreachable symbol.
//
// So the link succeeding is most of the test.  Running it then confirms the
// stronger property for cos(): not merely that the names did not clash, but
// that calls compiled here reach this file's definition.
//
// It has a third job, unrelated to symbols.  qc_config.h is generated, records
// the geometry an installed archive was built for, and is the only thing that
// makes a libqc.a in /usr/local/lib self-describing -- so it had better be
// true.  Every macro in it is checked here against the corresponding runtime
// query, which is the one place the generator and the library are compared.
//
// Build and run: make check-lib   (or make test/collide && ./test/collide)
//
// Compiled with none of qc's build flags, like example/pion -- only -I. to find
// qc.h -- plus -fno-builtin so the compiler does not fold cos() into an
// instruction and skip the call being tested.  qc_set_comm is deliberately not
// referenced: it lives in qc_mpi.h, which needs <mpi.h>, and this program is
// meant to compile in the environment a serial consumer has.

#include <stdio.h>
#include "qc.h"
#include "qc_config.h"

static int fails = 0;

static void check(int ok, const char * what) {
  printf("collide: %s %s\n", ok ? "ok  " : "FAIL", what);
  if (! ok) fails ++;
}

// Sentinel returned by this file's cos().  Chosen to be outside the range of
// the real cosine so a wrong answer cannot look plausible.
#define COS_SENTINEL 42.0

static int cos_called = 0;

// The three colliding definitions.  Signatures are only documentation: a
// duplicate-definition failure is decided on the name alone, and these
// deliberately do not match qc's (solve there is
// void solve(REAL *, REAL *, REAL *, REAL *, ...), print0 is variadic).
double cos(double x) {
  (void) x;
  cos_called = 1;
  return COS_SENTINEL;
}

int solve(void) {
  return 0;
}

int print0(void) {
  return 0;
}

// Address-taken so the linker must resolve every one of them; volatile so
// nothing here is optimized away.  Function pointers go through void (*)(void)
// rather than void * -- converting a function pointer to an object pointer is
// not something standard C guarantees, and this file is built with -Wextra.
typedef void (* fnptr)(void);

static fnptr volatile api[] = {
  (fnptr) qc_native_precision,
  (fnptr) qc_global_dims,
  (fnptr) qc_local_dims,
  (fnptr) qc_local_origin,
  (fnptr) qc_num_rhs,
  (fnptr) qc_num_colors,
  (fnptr) qc_num_spins,
  (fnptr) qc_field_elements,
  (fnptr) qc_lane_elements,
  (fnptr) qc_gauge_elements,
  (fnptr) qc_alloc,
  (fnptr) qc_free,
  (fnptr) qc_field_zero,
  (fnptr) qc_site_index,
  (fnptr) qc_site_coords,
  (fnptr) qc_component_index,
  (fnptr) qc_import,
  (fnptr) qc_export,
  (fnptr) qc_init,
  (fnptr) qc_finalize,
  (fnptr) qc_rank,
  (fnptr) qc_num_ranks,
  (fnptr) qc_read_gauge_file,
  (fnptr) qc_default_setup_params,
  (fnptr) qc_default_solve_params,
  (fnptr) qc_setup,
  (fnptr) qc_solve,
  (fnptr) qc_apply_D,
  (fnptr) qc_report_profile,
  (fnptr) qc_teardown,
};

#define API_COUNT ((int) (sizeof api / sizeof api[0]))

int main(void) {
  printf("collide: linked against libqc.a with colliding solve/print0/cos\n");

  int resolved = 0;
  for (int i = 0; i < API_COUNT; i ++)
    if (api[i] != 0) resolved ++;
  check(resolved == API_COUNT, "every qc.h entry point resolved");
  printf("collide:      %d entry points\n", API_COUNT);

  // volatile input so the call cannot be constant-folded even if -fno-builtin
  // were dropped from the build line.
  volatile double x = 0.0;
  const double got = cos(x);
  check(cos_called, "cos() call reached this file, not qc's copy");
  check(got == COS_SENTINEL, "cos() returned this file's sentinel");

  // qc_config.h against the library it claims to describe.  All of these
  // queries read compile-time constants, so none of them needs qc_init --
  // which is what lets this program run at any rank count, including one that
  // would not match a multi-rank build's process grid.
  int gd[4], ld[4];
  qc_global_dims(gd);
  qc_local_dims(ld);

  check(gd[0] == QC_CONFIG_LX && gd[1] == QC_CONFIG_LY
     && gd[2] == QC_CONFIG_LZ && gd[3] == QC_CONFIG_LT,
        "qc_config.h global lattice matches qc_global_dims()");
  printf("collide:      %dx%dx%dx%d global\n", gd[0], gd[1], gd[2], gd[3]);

  // The rank grid is not directly queryable, but the local extents are exactly
  // the global ones divided by it, so this is what validates the grid macros.
  check(ld[2] == QC_CONFIG_LZ / QC_CONFIG_MPI_GRID_PZ
     && ld[3] == QC_CONFIG_LT / QC_CONFIG_MPI_GRID_PT,
        "qc_config.h rank grid matches qc_local_dims()");
  printf("collide:      %dx%dx%dx%d per rank, grid %dx%d\n", ld[0], ld[1],
         ld[2], ld[3], QC_CONFIG_MPI_GRID_PZ, QC_CONFIG_MPI_GRID_PT);

  check(qc_num_rhs()    == QC_CONFIG_NUM_RHS,
        "qc_config.h num_rhs matches qc_num_rhs()");
  check(qc_num_colors() == QC_CONFIG_NC,
        "qc_config.h Nc matches qc_num_colors()");
  check(qc_num_spins()  == QC_CONFIG_NS,
        "qc_config.h Ns matches qc_num_spins()");
  check((int) qc_native_precision() == QC_CONFIG_PRECISION,
        "qc_config.h precision matches qc_native_precision()");
  printf("collide:      num_rhs %d, Nc %d, Ns %d, %d-byte reals\n",
         qc_num_rhs(), qc_num_colors(), qc_num_spins(),
         (int) qc_native_precision());

  printf("collide: %s\n", fails ? "FAILED" : "PASS");
  return fails ? 1 : 0;
}
