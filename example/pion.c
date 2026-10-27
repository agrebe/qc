// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

// Zero-momentum pion correlator from point sources -- a worked example of
// linking against qc.
//
// This file includes qc.h and nothing else from qc, and is compiled with none
// of qc's build flags.  That is deliberate: it is the check that the public
// header stands on its own.
//
// It uses both ways of getting data across the boundary, because a real
// calculation wants both.  The sources are written straight into the solver
// field through qc_site_index -- for point sources that is a handful of stores,
// where going through a full-volume array would be wasteful.  The solution
// comes back out through qc_export, which hands it over in the conventional
// lexicographic order the correlator loop wants.
//
//   build: make example/pion
//   run:   ./example/pion [gauge-file]

#include <stdio.h>
#include <stdlib.h>
#include "qc.h"

// Store 1 into a solver field, whichever precision this build of qc uses.  A
// caller that fixes its own precision can drop the branch -- but must then
// check qc_native_precision() once at startup, because the field is void * and
// nothing will catch a mismatch.
static void set_one(void * field, long i) {
  if (qc_native_precision() == QC_DOUBLE) ((double *) field)[i] = 1.0;
  else                                    ((float  *) field)[i] = 1.0f;
}

int main(int argc, char ** argv) {
  qc_init(&argc, &argv);

  const double kappa = 0.134, csw = 1.87567;
  const char * conf = (argc > 1) ? argv[1] : "test/sample_config.lime";

  int gd[4], ld[4], og[4];
  qc_global_dims(gd);
  qc_local_dims(ld);
  qc_local_origin(og);
  // Named to collide with the macros the internal headers define.  qc.h must
  // not define Nc, Ns, V, D or replace malloc; if it ever starts to, this file
  // stops compiling, which is the point of it being an example and a test.
  const int Nc = qc_num_colors(), Ns = qc_num_spins(), nrhs = qc_num_rhs();
  const int nsc = Ns * Nc;                  // 12 spin-colors per point source

  if (qc_rank() == 0) {
    printf("qc pion example: %dx%dx%dx%d lattice, %d rhs per solve, %s fields\n",
           gd[0], gd[1], gd[2], gd[3], nrhs,
           qc_native_precision() == QC_DOUBLE ? "double" : "single");
    if (qc_num_ranks() > 1)
      printf("qc pion example: %d ranks -- the timeslice sums below are this "
             "rank's slab only; combining them is the caller's job\n",
             qc_num_ranks());
  }

  // --- gauge configuration --------------------------------------------------
  void * U = qc_alloc(qc_gauge_elements());
  qc_read_gauge_file(U, conf);

  // Defaults, then the physics.  Never a designated initializer here: the
  // fields it left out would be zero rather than tuned, and qc cannot tell the
  // difference.  Everything except kappa and csw is a solver knob you can
  // leave alone until you have a reason not to.
  qc_setup_params sp;
  qc_default_setup_params(&sp);
  sp.kappa = kappa;
  sp.csw   = csw;
  qc_handle * h = qc_setup(U, qc_native_precision(), &sp);
  qc_free(U);                               // qc keeps its own prepared copy

  // --- sources, written directly in qc's own order --------------------------
  // Every lane carries a real source, which is what a production batch looks
  // like: the twelve spin-colors of a point source at the origin, then as many
  // more positions as the remaining lanes hold.  Only the correlator's own
  // source (position 0) is projected below.
  const int src_pos[4][4] = {
    { 0,           0,           0,           0           },
    { gd[0]/2,     gd[1]/2,     gd[2]/2,     gd[3]/2     },
    { gd[0]/4,     3*gd[1]/4,   gd[2]/4,     3*gd[3]/4   },
    { 3*gd[0]/4,   gd[1]/4,     3*gd[2]/4,   gd[3]/4     },
  };

  void * b = qc_alloc(qc_field_elements());
  void * x = qc_alloc(qc_field_elements());
  qc_field_zero(b);

  for (int lane = 0; lane < nrhs; lane ++) {
    const int p = lane / nsc, sc = lane % nsc;
    if (p >= 4) break;                      // past the table: lane stays zero
    const int s = sc / Nc, c = sc % Nc;
    const int gx = src_pos[p][0], gy = src_pos[p][1];
    const int gz = src_pos[p][2], gt = src_pos[p][3];
    // Coordinates in the table are global; a point source lives on one rank.
    // Placing it takes local coordinates, so subtract this rank's origin.
    if (gz < og[2] || gz >= og[2] + ld[2]) continue;
    if (gt < og[3] || gt >= og[3] + ld[3]) continue;
    const long site = qc_site_index(gx, gy, gz - og[2], gt - og[3]);
    set_one(b, qc_component_index(site, s, c, /*real part*/ 0, lane));
  }

  // --- solve ----------------------------------------------------------------
  qc_solve_params vp;
  qc_default_solve_params(&vp);
  vp.tol = 1e-10;
  vp.max_iter = 100;
  vp.chroma_norm = 1;
  const int iters = qc_solve(h, x, b, &vp);
  if (qc_rank() == 0)
    printf("qc pion example: %d outer iterations\n", iters);
  // Where that solve's time went, per kernel.  Comment out for quieter output;
  // it is here because it is the thing to read when tuning mg parameters.
  qc_report_profile(h);

  // --- correlator, from the solution in lexicographic order -----------------
  // C(t) = sum over this timeslice, over the twelve lanes of the position-0
  // source, of |prop|^2.  The other lanes belong to other source positions and
  // are simply not read.
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
            double sum = 0;
            for (int i = 0; i < nsc; i ++) {
              const double re = lane_field[2 * (base + i)];
              const double im = lane_field[2 * (base + i) + 1];
              sum += re * re + im * im;
            }
            corr[t + og[3]] += sum;
          }
  }

  if (qc_rank() == 0) {
    printf("\n   t         qc C[t]\n");
    for (int t = 0; t < gd[3]; t ++)
      printf("  %2d   %15.6e\n", t, corr[t]);
  }

  free(corr);
  free(lane_field);
  qc_free(b);
  qc_free(x);
  qc_teardown(h);
  qc_finalize();
  return 0;
}
