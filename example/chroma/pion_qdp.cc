// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

// Drive qc from QDP++ field types -- the layout bridge a chroma integration needs.
//
// Chroma's fermion fields are QDP++ LatticeFermions, laid out red/black with a
// lexicographic order inside each parity.  qc's fields are parity-major with a
// 4^4 block order inside each parity.  Neither can read the other's memory, so
// something has to walk sites.  That walk lives in bridge.h; this file is the
// demonstration and the test of it.
//
// It builds the twelve spin-color point sources of a pion correlator as
// LatticeFermions, hands them to qc, solves, reads the propagator back into
// LatticeFermions, and computes C(t) with QDP++ primitives.  The table must
// match `./qc correlator`.
//
// Only qc.h and qc_config.h are included from qc -- no internal header, none of
// qc's build flags.  That is the same standing constraint example/pion.c is
// under, and it is what makes this a test of the public interface.

#include <qdp.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

#include "qc.h"
#include "qc_config.h"

#define MUTATION_HOOKS 1        // this program offers --mutate
#include "bridge.h"

using namespace QDP;

// The archive is built for one lattice, one num_rhs and one precision, and
// passing it a field sized for anything else is not diagnosed -- solver fields
// are void *.  Checking here is what stands between a mismatch and silent
// corruption.
static_assert(QC_CONFIG_NC == Nc, "libqc and QDP++ disagree on Nc");
static_assert(QC_CONFIG_NS == Ns, "libqc and QDP++ disagree on Ns");
// This program pairs with a scalar QDP++ and has only ever been run at one rank.
// The chroma measurement is the one to use under MPI: it gates the two
// decompositions against each other at runtime instead of assuming they agree.
static_assert(QC_CONFIG_MPI == 0, "pion_qdp assumes a serial build of libqc");

static const int NSC = QC_CONFIG_NS * QC_CONFIG_NC;   // 12 spin-colors

// The one definition of the selector bridge.h declares.  Off unless --mutate
// says otherwise.
int g_mutate = 0;

// ---------------------------------------------------------------------------
// Timeslice set for the correlator
// ---------------------------------------------------------------------------
class TimeSliceFunc : public SetFunc {
public:
  explicit TimeSliceFunc(int dir) : dir_(dir) {}
  int operator()(const multi1d<int> & coord) const { return coord[dir_]; }
  int numSubsets() const { return Layout::lattSize()[dir_]; }
private:
  int dir_;
};

// ---------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------

static int failures = 0;

static void report(const char * name, bool ok, const char * detail) {
  QDPIO::cout << (ok ? "  [pass] " : "  [FAIL] ") << name;
  if (detail && *detail) QDPIO::cout << "  " << detail;
  QDPIO::cout << "\n";
  if (!ok) failures ++;
}

// Every qc site index must be hit exactly once.  Catches a site map that is
// merely self-consistent -- a permutation error that a round trip cannot see,
// because a round trip applies the same map forwards and backwards.
static void check_bijection(const std::vector<long> & map) {
  const long nsites = (long) Layout::vol();
  std::vector<int> hits((size_t) nsites, 0);
  long out_of_range = 0;
  for (size_t i = 0; i < map.size(); i ++) {
    if (map[i] < 0 || map[i] >= nsites) { out_of_range ++; continue; }
    hits[(size_t) map[i]] ++;
  }
  long missed = 0, doubled = 0;
  for (long s = 0; s < nsites; s ++) {
    if (hits[(size_t) s] == 0) missed ++;
    else if (hits[(size_t) s] > 1) doubled ++;
  }
  char msg[256];
  snprintf(msg, sizeof msg, "%ld sites, %ld out of range, %ld missed, %ld doubled",
           (long) map.size(), out_of_range, missed, doubled);
  report("site map is a bijection", out_of_range == 0 && missed == 0 && doubled == 0, msg);
}

// qc's parity must agree with QDP++'s.  qc splits the field even-sites-first, so
// the low half of the index range has to be exactly QDP++'s even checkerboard.
static void check_parity(const std::vector<long> & map) {
  const long half = (long) Layout::vol() / 2;
  long disagree = 0;
  for (size_t i = 0; i < map.size(); i ++) {
    const multi1d<int> c = Layout::siteCoords(Layout::nodeNumber(), (int) i);
    const int par_qdp = (c[0] + c[1] + c[2] + c[3]) & 1;
    const int par_qc  = (map[i] >= half) ? 1 : 0;
    if (par_qdp != par_qc) disagree ++;
  }
  char msg[128];
  snprintf(msg, sizeof msg, "%ld sites disagree", disagree);
  report("parity convention matches QDP++", disagree == 0, msg);
}

// A field through import then export must come back unchanged.  QDP++ is single
// precision here and qc's fields are double, so the round trip is exact and the
// bar is bitwise equality -- any tolerance would hide a real defect.
static void check_round_trip(const std::vector<long> & map, void * scratch) {
  LatticeFermion a, b;
  gaussian(a);
  b = zero;

  const int lane = 3;                 // not 0: a lane-stride bug that happens to
  import_lane(scratch, a, map, lane); // work at lane 0 still has to fail here
  export_lane(scratch, b, map, lane);

  double worst = 0;
  for (int i = 0; i < Layout::sitesOnNode(); i ++)
    for (int s = 0; s < Ns; s ++)
      for (int c = 0; c < Nc; c ++) {
        worst = std::max(worst, std::fabs((double) (a.elem(i).elem(s).elem(c).real()
                                                  - b.elem(i).elem(s).elem(c).real())));
        worst = std::max(worst, std::fabs((double) (a.elem(i).elem(s).elem(c).imag()
                                                  - b.elem(i).elem(s).elem(c).imag())));
      }
  char msg[128];
  snprintf(msg, sizeof msg, "worst abs diff %.3e (bitwise equality required)", worst);
  report("QDP++ -> qc -> QDP++ round trip", worst == 0.0, msg);
}

// Writing a lane must leave the others alone.  The round trip above reads back
// the lane it wrote, so it is blind to a stride that spills into a neighbor.
static void check_lane_isolation(const std::vector<long> & map, void * scratch) {
  LatticeFermion a, b, out;
  gaussian(a);
  gaussian(b);

  qc_field_zero(scratch);
  import_lane(scratch, a, map, 5);
  import_lane(scratch, b, map, 6);     // must not disturb lane 5

  out = zero;
  export_lane(scratch, out, map, 5);

  double worst = 0;
  for (int i = 0; i < Layout::sitesOnNode(); i ++)
    for (int s = 0; s < Ns; s ++)
      for (int c = 0; c < Nc; c ++)
        worst = std::max(worst, std::fabs((double) (a.elem(i).elem(s).elem(c).real()
                                                  - out.elem(i).elem(s).elem(c).real())));
  char msg[128];
  snprintf(msg, sizeof msg, "worst abs diff %.3e", worst);
  report("writing one lane leaves its neighbor intact", worst == 0.0, msg);
}

// The two public paths must agree: the accessor route this file uses for the
// solve, and the qc_import route a caller with lexicographic fields would use.
// Neither is checked against anything else, so checking them against each other
// is what makes a systematic error in one of them visible.
static void check_paths_agree(const std::vector<long> & map,
                              void * viaAccessor, void * viaImport) {
  LatticeFermion a;
  gaussian(a);

  qc_field_zero(viaAccessor);
  qc_field_zero(viaImport);

  const int lane = 2;
  import_lane(viaAccessor, a, map, lane);

  std::vector<double> lex;
  to_lex(a, lex);
  qc_import(viaImport, lex.data(), QC_DOUBLE, lane);

  double worst = 0;
  const long n = qc_field_elements();
  for (long i = 0; i < n; i ++)
    worst = std::max(worst, std::fabs(fld_get(viaAccessor, i) - fld_get(viaImport, i)));

  char msg[160];
  snprintf(msg, sizeof msg,
           "worst abs diff %.3e over %ld elements (bitwise equality required)", worst, n);
  report("accessor path and qc_import agree", worst == 0.0, msg);
}

// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
  QDP_initialize(&argc, &argv);

  qc_init(&argc, &argv);

  int gd[4];
  qc_global_dims(gd);

  multi1d<int> nrow(Nd);
  for (int i = 0; i < Nd; i ++) nrow[i] = gd[i];
  Layout::setLattSize(nrow);
  Layout::create();

  const char * conf = "../../test/sample_config.lime";
  int checks_only = 0, gauge_float = 0;
  for (int i = 1; i < argc; i ++) {
    if (!strncmp(argv[i], "--mutate=", 9))   g_mutate = atoi(argv[i] + 9);
    else if (!strcmp(argv[i], "--checks-only")) checks_only = 1;
    else if (!strcmp(argv[i], "--gauge-float")) gauge_float = 1;
    else if (argv[i][0] != '-')              conf = argv[i];
  }
  const double kappa = 0.134, csw = 1.87567;

  QDPIO::cout << "\nqc <-> QDP++ bridge: " << gd[0] << "x" << gd[1] << "x"
              << gd[2] << "x" << gd[3] << " lattice, " << qc_num_rhs()
              << " rhs per solve, "
              << (qc_native_precision() == QC_DOUBLE ? "double" : "single")
              << " solver fields\n\n";

  if (Layout::vol() != (int) (gd[0] * gd[1] * gd[2] * gd[3])) {
    QDPIO::cout << "geometry mismatch between QDP++ and libqc\n";
    QDP_finalize();
    return 1;
  }

  const std::vector<long> map = build_site_map();

  // --- layout checks --------------------------------------------------------
  QDPIO::cout << "layout checks\n";
  void * scratch  = qc_alloc(qc_field_elements());
  void * scratch2 = qc_alloc(qc_field_elements());
  qc_field_zero(scratch);

  check_bijection(map);
  check_parity(map);
  check_round_trip(map, scratch);
  check_lane_isolation(map, scratch);
  check_paths_agree(map, scratch, scratch2);

  qc_free(scratch2);

  // The checks above need neither a gauge configuration nor a solve, so they can
  // stand alone as a fast layout regression -- which is also what makes the
  // mutation runs cheap enough to be worth having.
  if (checks_only) {
    qc_free(scratch);
    QDPIO::cout << "\n" << (failures == 0 ? "layout checks: all passed"
                                          : "layout checks: FAILURES present")
                << "\n";
    qc_finalize();
    QDP_finalize();
    return failures == 0 ? 0 : 1;
  }

  // --- gauge and setup ------------------------------------------------------
  QDPIO::cout << "\nreading gauge and building the hierarchy\n";
  void * U = qc_alloc(qc_gauge_elements());
  qc_read_gauge_file(U, conf);

  // --gauge-float rounds every link through float before qc sees it, which is
  // exactly what a chroma driver does to them: QDP++ here is BASE_PRECISION 32,
  // so links arriving from chroma's named-object map have already been through
  // a float.  This exists to attribute the small difference between the
  // chroma-driven correlator and this one, rather than leave it unexplained.
  if (gauge_float) {
    const long n = qc_gauge_elements();
    for (long i = 0; i < n; i ++)
      fld_set(U, i, (double) (float) fld_get(U, i));
    QDPIO::cout << "gauge rounded through single precision\n";
  }

  qc_setup_params sp;
  qc_default_setup_params(&sp);       // defaults first, then the physics: a
  sp.kappa = kappa;                   // designated initializer here would zero
  sp.csw   = csw;                     // every knob it failed to name
  qc_handle * h = qc_setup(U, qc_native_precision(), &sp);
  qc_free(U);

  // --- sources, as QDP++ fields ---------------------------------------------
  // The twelve spin-colors of a point source at the origin, each built as a
  // LatticeFermion the way chroma would, then marshalled into lanes 0..11.
  // Lanes 12..15 stay zero: qc skips a lane whose rhs norm is zero, so the
  // padding costs lattice bandwidth but does not perturb the other twelve.
  multi1d<LatticeFermion> src(NSC);
  multi1d<int> origin(Nd);
  for (int i = 0; i < Nd; i ++) origin[i] = 0;

  for (int sc = 0; sc < NSC; sc ++) {
    src[sc] = zero;
    const int s = sc / Nc, c = sc % Nc;
    Fermion f;
    f = zero;
    f.elem().elem(s).elem(c).real() = 1.0;
    pokeSite(src[sc], f, origin);
  }

  void * b = qc_alloc(qc_field_elements());
  void * x = qc_alloc(qc_field_elements());
  qc_field_zero(b);
  for (int sc = 0; sc < NSC; sc ++) import_lane(b, src[sc], map, sc);

  // --- solve ----------------------------------------------------------------
  QDPIO::cout << "\nsolving " << NSC << " right-hand sides ("
              << qc_num_rhs() - NSC << " lanes zero-padded)\n";
  qc_solve_params vp;
  qc_default_solve_params(&vp);
  vp.tol         = 1e-10;
  vp.max_iter    = 100;
  vp.chroma_norm = 1;                 // 2*kappa, the convention chroma reports
  const int iters = qc_solve(h, x, b, &vp);
  QDPIO::cout << "qc returned after " << iters << " outer iterations\n";

  // --- propagator back into QDP++, correlator with QDP++ primitives ---------
  multi1d<LatticeFermion> prop(NSC);
  for (int sc = 0; sc < NSC; sc ++) {
    prop[sc] = zero;
    export_lane(x, prop[sc], map, sc);
  }

  LatticeReal density = zero;
  for (int sc = 0; sc < NSC; sc ++) density += localNorm2(prop[sc]);

  Set timeslice;
  timeslice.make(TimeSliceFunc(Nd - 1));
  multi1d<Double> corr = sumMulti(density, timeslice);

  QDPIO::cout << "\n   t         C[t]  (compare ./qc correlator)\n";
  for (int t = 0; t < gd[3]; t ++)
    printf("  %2d   %15.6e\n", t, toDouble(corr[t]));

  QDPIO::cout << "\n" << (failures == 0 ? "layout checks: all passed"
                                        : "layout checks: FAILURES present")
              << "\n";

  qc_free(b);
  qc_free(x);
  qc_teardown(h);
  qc_finalize();
  QDP_finalize();
  return failures == 0 ? 0 : 1;
}
