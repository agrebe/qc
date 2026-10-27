// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

// Gate for qc's MPI ownership contract: qc adopts MPI, it does not own it.
//
// This cannot be an arm of ./qc.  runner.c calls comm_init before it
// dispatches, so by the time any arm runs, MPI is already up and the
// interesting question -- what happens when the caller initialized it first
// -- has been answered the wrong way.  Hence a separate driver with its own
// main.
//
//   build: make test/adopt_mpi MPI=1
//   run:   mpirun -n <N> ./test/adopt_mpi <mode> [gauge-file]
//
// It links libqc.a through the public interface only -- qc.h, qc_config.h and
// qc_mpi.h, with none of qc's build flags -- so it doubles as a check that the
// MPI half of the API is reachable from outside the library.  Its own
// reductions go through MPI directly rather than through qc's internals,
// which are local symbols in the archive and unavailable to a consumer by
// design.
//
// Modes, each a different division of responsibility:
//   qcinit  qc initializes and finalizes MPI (nobody else touches it).
//   world   the caller initializes MPI and finalizes it; qc adopts, on the
//           default communicator.
//   rev     as world, but qc is handed a communicator whose rank order is
//           reversed against MPI_COMM_WORLD.
//
// `rev` is the one that discriminates.  Under it, qc rank r is world rank
// N-1-r, so any MPI call left on MPI_COMM_WORLD talks to a different process
// than the one holding the neighboring slab: the halo topology stops agreeing
// with where the data actually is.  Meanwhile the physics does not care which
// process holds which slab, so a correct implementation has to reproduce the
// same correlator as `world` and as a single rank.  Comparing the three tables
// therefore tests the communicator plumbing against the operator rather than
// against a second copy of its own assumptions -- the plain correlator alone is
// blind to this, exactly as it is blind to halo bugs.
//
// The ownership half is checked directly rather than inferred: after
// qc_finalize the test asserts MPI_Finalized() is false in the adopting modes
// (qc must not have finalized something it did not start) and true in qcinit.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "qc.h"
#include "qc_config.h"

// QC_CONFIG_MPI, not the build's private QC_MPI: this program links libqc.a as
// a consumer would, with none of qc's -D flags, so the generated header is the
// only thing that tells it whether the library it is linking has MPI in it.
#if QC_CONFIG_MPI

#include <mpi.h>
#include "qc_mpi.h"

static int fails = 0;

// Deliberately MPI_COMM_WORLD rather than qc's communicator.  A consumer has no
// access to that one, and using it here would let the test inherit whatever
// mistake it is supposed to be detecting; a plain sum over all ranks is
// order-independent, so the reversed mode reduces to the same value.
static void sum_over_ranks(double * v, int n) {
  MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

static void check(int ok, const char * what) {
  if (!ok) { fails ++; printf("adopt_mpi: FAIL %s\n", what); }
}

static void set_one(void * field, long i) {
  if (qc_native_precision() == QC_DOUBLE) ((double *) field)[i] = 1.0;
  else                                    ((float  *) field)[i] = 1.0f;
}

static double get_elem(const void * field, long i) {
  return qc_native_precision() == QC_DOUBLE ? ((const double *) field)[i]
                                            : ((const float  *) field)[i];
}

int main(int argc, char ** argv) {
  const char * mode = (argc > 1) ? argv[1] : "qcinit";
  const char * conf = (argc > 2) ? argv[2] : "test/sample_config.lime";

  const int caller_owns = strcmp(mode, "qcinit") != 0;
  const int reversed    = strcmp(mode, "rev")    == 0;
  if (strcmp(mode, "qcinit") && strcmp(mode, "world") && strcmp(mode, "rev")) {
    fprintf(stderr, "usage: adopt_mpi <qcinit|world|rev> [gauge-file]\n");
    return 2;
  }

  int world_rank = 0, world_size = 1;
  MPI_Comm mine = MPI_COMM_NULL;

  if (caller_owns) {
    // The caller gets there first, which is the whole point of the test.
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (reversed) {
      // Same group, opposite order: key = N-1-rank.
      MPI_Comm_split(MPI_COMM_WORLD, 0, world_size - 1 - world_rank, &mine);
      qc_set_comm(mine);
    }
  }

  qc_init(&argc, &argv);

  if (caller_owns) {
    check(qc_num_ranks() == world_size, "qc rank count matches the communicator");
    const int want = reversed ? world_size - 1 - world_rank : world_rank;
    if (qc_rank() != want) {
      fails ++;
      printf("adopt_mpi: FAIL rank map (world %d -> qc %d, expected %d)\n",
             world_rank, qc_rank(), want);
    }
  }

  int gd[4], ld[4], og[4];
  qc_global_dims(gd);
  qc_local_dims(ld);
  qc_local_origin(og);
  const int Nc = qc_num_colors(), Ns = qc_num_spins(), nrhs = qc_num_rhs();
  const int nsc = Ns * Nc;

  if (qc_rank() == 0)
    printf("adopt_mpi: mode=%s, %d rank(s), %dx%dx%dx%d\n",
           mode, qc_num_ranks(), gd[0], gd[1], gd[2], gd[3]);

  void * U = qc_alloc(qc_gauge_elements());
  qc_read_gauge_file(U, conf);

  qc_setup_params sp;
  qc_default_setup_params(&sp);
  sp.kappa = 0.134;
  sp.csw   = 1.87567;
  qc_handle * h = qc_setup(U, qc_native_precision(), &sp);
  qc_free(U);

  // One point source at the origin, its twelve spin-colors on the first twelve
  // lanes.  Only the rank owning the origin writes anything.
  void * b = qc_alloc(qc_field_elements());
  void * x = qc_alloc(qc_field_elements());
  qc_field_zero(b);
  for (int lane = 0; lane < nsc && lane < nrhs; lane ++) {
    if (og[2] != 0 || og[3] != 0) break;      // origin lives on one rank
    const long site = qc_site_index(0, 0, 0, 0);
    set_one(b, qc_component_index(site, lane / Nc, lane % Nc, 0, lane));
  }

  // chroma_norm off so the residual check below sees the unscaled solution;
  // the correlator table is then unnormalized, which is fine because it is
  // only ever compared against other runs of this same program.
  qc_solve_params vp;
  qc_default_solve_params(&vp);
  vp.tol = 1e-10;
  vp.max_iter = 100;
  vp.chroma_norm = 0;
  const int iters = qc_solve(h, x, b, &vp);

  // Did the solve actually work?  Without this the test is far weaker than it
  // looks: a broken communicator corrupts the multigrid preconditioner, and the
  // outer Krylov then papers over it -- the run limps to max_iter and returns a
  // wrong answer while every structural check above still passes.  Assert on
  // convergence and on the true residual, which needs no reference table.
  check(iters < vp.max_iter, "solve converged rather than hitting max_iter");

  void * Dx = qc_alloc(qc_field_elements());
  qc_apply_D(h, Dx, x);
  double rn = 0, bn = 0;
  const long nelem = qc_field_elements();
  for (long i = 0; i < nelem; i ++) {
    const double bi = get_elem(b, i), d = bi - get_elem(Dx, i);
    rn += d * d;  bn += bi * bi;
  }
  double red[2] = { rn, bn };
  sum_over_ranks(red, 2);
  const double relres = red[1] > 0 ? sqrt(red[0] / red[1]) : 1.0;
  if (!(relres < 1e-8)) {
    fails ++;
    printf("adopt_mpi: FAIL ||b-Dx||/||b|| = %.3e, expected below 1e-8\n",
           relres);
  }
  qc_free(Dx);

  // C(t) over global t, summed across ranks -- the observable that does not
  // care which process holds which slab, and so is comparable across the three
  // modes.
  double * corr = calloc((size_t) gd[3], sizeof(double));
  double * lane_field = malloc(sizeof(double) * (size_t) qc_lane_elements());
  for (int lane = 0; lane < nsc && lane < nrhs; lane ++) {
    qc_export(x, lane_field, QC_DOUBLE, lane);
    for (int t = 0; t < ld[3]; t ++)
      for (int z = 0; z < ld[2]; z ++)
        for (int y = 0; y < ld[1]; y ++)
          for (int xx = 0; xx < ld[0]; xx ++) {
            const long base = ((((long) t * ld[2] + z) * ld[1] + y) * ld[0]
                               + xx) * (long) nsc;
            for (int i = 0; i < nsc; i ++) {
              const double re = lane_field[2 * (base + i)];
              const double im = lane_field[2 * (base + i) + 1];
              corr[t + og[3]] += re * re + im * im;
            }
          }
  }
  sum_over_ranks(corr, gd[3]);

  if (qc_rank() == 0) {
    printf("adopt_mpi: %d outer iterations, ||b-Dx||/||b|| = %.3e\n", iters,
           relres);
    printf("\n   t      adopt C[t]\n");
    for (int t = 0; t < gd[3]; t ++) printf("  %2d   %15.6e\n", t, corr[t]);
  }

  free(corr);
  free(lane_field);
  qc_free(b);
  qc_free(x);
  qc_teardown(h);

  qc_finalize();

  // The ownership assertion.  qc_finalize must finalize MPI exactly when
  // qc_init was what initialized it -- this is what fails if comm_finalize
  // goes back to calling MPI_Finalize unconditionally.
  int done = 0;
  MPI_Finalized(&done);
  check(done == !caller_owns, caller_owns
          ? "qc left MPI running for its caller to finalize"
          : "qc finalized the MPI it started");

  if (caller_owns) {
    if (mine != MPI_COMM_NULL) MPI_Comm_free(&mine);
    MPI_Finalize();
  }

  if (qc_rank() == 0)
    printf("adopt_mpi: %d check(s) failed => %s\n", fails,
           fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}

#else   // !QC_MPI

int main(void) {
  printf("adopt_mpi: serial build (MPI=0) -- nothing to adopt, skipping\n");
  return 0;
}

#endif
