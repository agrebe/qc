// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

// Validate qc's Wilson-clover operator against chroma's, and confirm a qc solve
// using chroma's own operator.
//
// This is the check phase 1 could not do.  QDP++ alone has no clover term, so
// the bridge there was validated only against qc's own correlator -- self-
// consistent, but not independent.  Chroma supplies a second Wilson-clover
// implementation, which turns the convention mapping into a measurement:
//
//   kappa  <-> Mass = 1/(2 kappa) - Nd     (chroma's A carries Nd + Mass)
//   csw    <-> clovCoeffR = clovCoeffT, u0 = 1
//   qc applies anti-periodic time boundaries to its own prepared copy of the
//   links, so chroma is asked for them explicitly via a (1,1,1,-1) fermbc
//
// The overall normalization between the two operators is deliberately not
// assumed.  Both comparisons fit the scale from the data and report it, then
// report the residual after removing it.  An assumed factor that happened to be
// wrong would otherwise show up as a large residual and be indistinguishable
// from a real disagreement.
//
// The links come from qc, not from a second read of the file: handing chroma
// the array qc_read_gauge_file produced removes the two I/O paths as a
// variable, so any disagreement here is the operator.

#include "chroma.h"
// Named explicitly rather than relying on chromainc.h's aggregation.
#include "actions/ferm/linop/unprec_clover_linop_w.h"
#include "actions/ferm/fermstates/simple_fermstate.h"
#include "meas/glue/mesplq.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

#include "qc.h"
#include "qc_config.h"
#include "bridge.h"

using namespace Chroma;

typedef LatticeFermion               T;
typedef multi1d<LatticeColorMatrix>  P;
typedef multi1d<LatticeColorMatrix>  Q;

static int failures = 0;

static void report(const char * name, bool ok, const char * detail) {
  QDPIO::cout << (ok ? "  [pass] " : "  [FAIL] ") << name;
  if (detail && *detail) QDPIO::cout << "  " << detail;
  QDPIO::cout << "\n";
  if (!ok) failures ++;
}

// Least-squares scale between two fields, then the relative residual once it is
// removed.  Returns the scale; *rel gets ||a - s b|| / ||a||.
static double fit_scale(const LatticeFermion & a, const LatticeFermion & b,
                        double * rel) {
  const double den = toDouble(norm2(b));
  if (den == 0.0) { *rel = 1.0; return 0.0; }
  const double s = toDouble(real(innerProduct(b, a))) / den;
  LatticeFermion d = a - Real(s) * b;
  *rel = std::sqrt(toDouble(norm2(d)) / toDouble(norm2(a)));
  return s;
}

int main(int argc, char ** argv) {
  Chroma::initialize(&argc, &argv);

  qc_init(&argc, &argv);

  int gd[4];
  qc_global_dims(gd);
  multi1d<int> nrow(Nd);
  for (int i = 0; i < Nd; i ++) nrow[i] = gd[i];
  Layout::setLattSize(nrow);
  Layout::create();

  // --mutate=N breaks one convention deliberately, so the checks below can be
  // shown to discriminate rather than merely to pass:
  //   1  periodic time boundary instead of anti-periodic
  //   2  Mass off by 1%, i.e. the kappa <-> Mass mapping wrong
  //   3  clovCoeff 0, i.e. chroma given plain Wilson against qc's clover
  const char * conf = "../../test/sample_config.lime";
  int mutate = 0;
  for (int i = 1; i < argc; i ++) {
    if (!strncmp(argv[i], "--mutate=", 9)) mutate = atoi(argv[i] + 9);
    else if (argv[i][0] != '-')            conf = argv[i];
  }

  const double kappa = 0.134, csw = 1.87567;

  QDPIO::cout << "\nqc vs chroma: " << gd[0] << "x" << gd[1] << "x" << gd[2]
              << "x" << gd[3] << ", kappa " << kappa << ", csw " << csw << "\n\n";

  const std::vector<long> map = build_site_map();

  // --- gauge ----------------------------------------------------------------
  void * U = qc_alloc(qc_gauge_elements());
  qc_read_gauge_file(U, conf);

  multi1d<LatticeColorMatrix> u;
  gauge_to_qdp(U, u);

  // The plaquette is the cheap independent witness that the links crossed
  // intact -- a site-map error in gauge_to_qdp scrambles it.  It is blind to a
  // color transpose (that conjugates the plaquette, leaving the real part), so
  // it is a smoke test, not a proof; the operator comparison below is what
  // actually pins the gauge convention.
  Double w_plaq, s_plaq, t_plaq, link;
  MesPlq(u, w_plaq, s_plaq, t_plaq, link);
  QDPIO::cout << "plaquette from qc's links: " << toDouble(w_plaq) << "\n";
  {
    char msg[128];
    const double wp = toDouble(w_plaq);
    snprintf(msg, sizeof msg, "w_plaq = %.6f", wp);
    report("plaquette is physical (0 < w_plaq < 1)", wp > 0.0 && wp < 1.0, msg);
  }

  // --- chroma's operator ----------------------------------------------------
  multi1d<int> boundary(Nd);
  boundary[0] = boundary[1] = boundary[2] = 1;
  boundary[3] = (mutate == 1) ? 1 : -1;   // qc hard-codes APBC in time
  Handle< FermState<T, P, Q> > fs(new SimpleFermState<T, P, Q>(boundary, u));

  CloverFermActParams cp;                 // defaults, then the physics -- the
  cp.Mass       = Real(1.0 / (2.0 * kappa) - Nd);   // same discipline qc's own
  cp.clovCoeffR = Real(csw);                        // params structs require
  cp.clovCoeffT = Real(csw);
  cp.u0         = Real(1.0);
  if (mutate == 2) cp.Mass = Real(toDouble(cp.Mass) * 1.01);
  if (mutate == 3) { cp.clovCoeffR = Real(0.0); cp.clovCoeffT = Real(0.0); }

  UnprecCloverLinOp M;
  M.create(fs, cp);

  QDPIO::cout << "chroma UnprecCloverLinOp: Mass " << toDouble(cp.Mass)
              << ", clovCoeff " << toDouble(cp.clovCoeffR) << "\n\n";

  // --- qc's operator --------------------------------------------------------
  qc_setup_params sp;
  qc_default_setup_params(&sp);
  sp.kappa = kappa;
  sp.csw   = csw;
  qc_handle * h = qc_setup(U, qc_native_precision(), &sp);
  qc_free(U);

  void * in  = qc_alloc(qc_field_elements());
  void * out = qc_alloc(qc_field_elements());

  LatticeFermion psi, chi_ch, chi_qc;
  gaussian(psi);

  M(chi_ch, psi, PLUS);

  qc_field_zero(in);
  import_lane(in, psi, map, 0);
  qc_apply_D(h, out, in);
  chi_qc = zero;
  export_lane(out, chi_qc, map, 0);

  double rel_op = 0;
  const double s_op = fit_scale(chi_qc, chi_ch, &rel_op);

  {
    // Testing the residual alone would only say the two are proportional; the
    // constant is the thing the convention mapping predicts, so it has to be in
    // the pass condition or a pure normalization error would slip through.
    // Single-precision QDP++ fields on both sides, so the bar is float
    // roundoff, not bitwise.
    const double want = 2.0 * kappa;
    const bool ok = rel_op < 1e-5 && std::fabs(s_op - want) / want < 1e-5;
    char msg[256];
    snprintf(msg, sizeof msg,
             "qc_D = %.8f * chroma_M (2*kappa = %.8f), residual %.3e",
             s_op, want, rel_op);
    report("operators agree, with the scale 2*kappa predicts", ok, msg);
  }

  // --- solve, checked with chroma's operator --------------------------------
  // The strongest statement available: qc produces the solution, and chroma's
  // independent operator confirms it.  Scale is fitted again rather than
  // assumed, because chroma_norm = 1 folds a 2*kappa into the answer.
  void * b = qc_alloc(qc_field_elements());
  void * x = qc_alloc(qc_field_elements());
  qc_field_zero(b);
  import_lane(b, psi, map, 0);

  qc_solve_params vp;
  qc_default_solve_params(&vp);
  vp.tol         = 1e-10;
  vp.max_iter    = 100;
  vp.chroma_norm = 1;
  const int iters = qc_solve(h, x, b, &vp);
  QDPIO::cout << "\nqc solved 1 of " << qc_num_rhs() << " lanes in " << iters
              << " outer iterations\n";

  LatticeFermion sol, Msol;
  sol = zero;
  export_lane(x, sol, map, 0);
  M(Msol, sol, PLUS);

  double rel_solve = 0;
  const double s_solve = fit_scale(Msol, psi, &rel_solve);

  {
    // With chroma_norm = 1 the scale should be exactly 1: qc is then returning
    // the solution in chroma's own normalization, needing no further rescaling.
    // Same reasoning as above -- the constant is part of the claim.
    const bool ok = rel_solve < 1e-5 && std::fabs(s_solve - 1.0) < 1e-5;
    char msg[256];
    snprintf(msg, sizeof msg,
             "chroma_M * (qc solution) = %.8f * source, residual %.3e",
             s_solve, rel_solve);
    report("chroma's operator confirms qc's solution", ok, msg);
  }

  QDPIO::cout << "\n" << (failures == 0 ? "qc vs chroma: all passed"
                                        : "qc vs chroma: FAILURES present")
              << "\n";

  qc_free(in); qc_free(out); qc_free(b); qc_free(x);
  qc_teardown(h);
  qc_finalize();
  Chroma::finalize();
  return failures == 0 ? 0 : 1;
}
