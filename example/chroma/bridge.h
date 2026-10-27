// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

// Shared QDP++ <-> qc marshalling, used by pion_qdp.cc and qc_vs_chroma.cc.
//
// The layout knowledge lives here and nowhere else.  Everything goes through
// qc.h's published accessors -- qc_site_index / qc_component_index -- rather
// than reimplementing qc's block arithmetic, so a future retune of qc's storage
// order does not break either program.
//
// Include <qdp.h> and "qc.h" before this header.

#ifndef QC_QDP_BRIDGE_H
#define QC_QDP_BRIDGE_H

#include <vector>

// ---------------------------------------------------------------------------
// Mutation hooks
// ---------------------------------------------------------------------------
// A passing check proves nothing until it has been shown to fail on the defect
// it exists to catch, so pion_qdp's --mutate=N introduces one deliberately.
//
//   1  export reads lane+1        -- a lane stride that a single-lane test misses
//   2  spin/color transposed in the lexicographic buffer only
//   3  site map ignores parity, using a plain lexicographic order
//
// 3 is the interesting one: it is still a bijection and still self-consistent,
// so the bijection check and the round trip both pass.  Only the parity check
// and the accessor-vs-qc_import cross-check can see it.
//
// Define MUTATION_HOOKS before this header to enable them, and g_mutate in one
// translation unit.  Otherwise every branch below folds away at compile time.
enum { MUT_NONE = 0, MUT_LANE_OFFSET = 1, MUT_SC_TRANSPOSE = 2, MUT_NO_PARITY = 3 };
#ifdef MUTATION_HOOKS
extern int g_mutate;
#else
constexpr int g_mutate = MUT_NONE;
#endif

// ---------------------------------------------------------------------------
// Precision
// ---------------------------------------------------------------------------
// A solver field is void * in whatever precision this build of qc uses, so
// every access goes through these.  A caller that pins its own build can drop
// the branch, but must then check qc_native_precision() once at startup.

static inline void fld_set(void * f, long i, double v) {
  if (qc_native_precision() == QC_DOUBLE) ((double *) f)[i] = v;
  else                                    ((float  *) f)[i] = (float) v;
}

static inline double fld_get(const void * f, long i) {
  return (qc_native_precision() == QC_DOUBLE)
           ? ((const double *) f)[i]
           : (double) ((const float *) f)[i];
}

// ---------------------------------------------------------------------------
// The site map
// ---------------------------------------------------------------------------
// One entry per QDP++ linear site, holding the qc site index for the same
// physical point.  Built once and reused: keeping the whole of the layout
// knowledge in one array is what makes the bijection check possible.
//
// Coordinates from QDP++ are global.  qc_site_index takes local ones, so the
// rank origin comes off first -- a no-op at one rank, but writing it the other
// way would be a latent bug the moment the grid is decomposed.

inline std::vector<long> build_site_map() {
  const int n = QDP::Layout::sitesOnNode();
  int org[4], ld[4];
  qc_local_origin(org);
  qc_local_dims(ld);

  std::vector<long> map(n);
  for (int i = 0; i < n; i ++) {
    const QDP::multi1d<int> c = QDP::Layout::siteCoords(QDP::Layout::nodeNumber(), i);
    const int x = c[0] - org[0], y = c[1] - org[1];
    const int z = c[2] - org[2], t = c[3] - org[3];
    if (g_mutate == MUT_NO_PARITY)
      map[i] = (((long) t * ld[2] + z) * ld[1] + y) * ld[0] + x;
    else
      map[i] = qc_site_index(x, y, z, t);
  }
  return map;
}

// ---------------------------------------------------------------------------
// Fermion transfer
// ---------------------------------------------------------------------------
// QDP++ stores a site as spin-outer, color-inner, complex innermost, which is
// already qc's per-site component order -- so these loops permute sites and
// scatter over the rhs lane, and never touch the component order itself.

inline void import_lane(void * field, const QDP::LatticeFermion & src,
                        const std::vector<long> & map, int lane) {
  const int n = (int) map.size();
  for (int i = 0; i < n; i ++) {
    const long site = map[i];
    for (int s = 0; s < QDP::Ns; s ++)
      for (int c = 0; c < QDP::Nc; c ++) {
        fld_set(field, qc_component_index(site, s, c, 0, lane),
                (double) src.elem(i).elem(s).elem(c).real());
        fld_set(field, qc_component_index(site, s, c, 1, lane),
                (double) src.elem(i).elem(s).elem(c).imag());
      }
  }
}

inline void export_lane(const void * field, QDP::LatticeFermion & dst,
                        const std::vector<long> & map, int lane) {
  const int n = (int) map.size();
  const int rl = (g_mutate == MUT_LANE_OFFSET) ? (lane + 1) % qc_num_rhs() : lane;
  for (int i = 0; i < n; i ++) {
    const long site = map[i];
    for (int s = 0; s < QDP::Ns; s ++)
      for (int c = 0; c < QDP::Nc; c ++) {
        dst.elem(i).elem(s).elem(c).real() =
          (REAL) fld_get(field, qc_component_index(site, s, c, 0, rl));
        dst.elem(i).elem(s).elem(c).imag() =
          (REAL) fld_get(field, qc_component_index(site, s, c, 1, rl));
      }
  }
}

// Fill a caller-side array in the lexicographic order qc_import/qc_export use,
// from a QDP++ field.  Only pion_qdp's cross-check needs this; the solve path
// goes straight through the accessors above.
inline void to_lex(const QDP::LatticeFermion & src, std::vector<double> & out) {
  int ld[4];
  qc_local_dims(ld);
  out.assign((size_t) qc_lane_elements(), 0.0);

  QDP::multi1d<int> co(QDP::Nd);
  for (int t = 0; t < ld[3]; t ++)
    for (int z = 0; z < ld[2]; z ++)
      for (int y = 0; y < ld[1]; y ++)
        for (int x = 0; x < ld[0]; x ++) {
          co[0] = x; co[1] = y; co[2] = z; co[3] = t;
          const int i = QDP::Layout::linearSiteIndex(co);
          const long base = (((long) t * ld[2] + z) * ld[1] + y) * ld[0] + x;
          for (int s = 0; s < QDP::Ns; s ++)
            for (int c = 0; c < QDP::Nc; c ++) {
              const long off = (g_mutate == MUT_SC_TRANSPOSE)
                                 ? (base * QDP::Nc + c) * QDP::Ns + s
                                 : (base * QDP::Ns + s) * QDP::Nc + c;
              out[2 * off]     = (double) src.elem(i).elem(s).elem(c).real();
              out[2 * off + 1] = (double) src.elem(i).elem(s).elem(c).imag();
            }
        }
}

// ---------------------------------------------------------------------------
// Gauge transfer
// ---------------------------------------------------------------------------
// qc's gauge array is lexicographic, [t][z][y][x][mu][row][col][re,im], stated
// as fact in qc.h because read_gauge maps the ILDG record straight onto it.
//
// Handing chroma the links qc already read, rather than letting chroma read the
// file itself, removes the two readers as a variable: any disagreement in an
// operator comparison is then the operator, not the I/O.
//
// The configuration is untwisted.  qc applies anti-periodic time boundaries
// internally to its own prepared copy, so a chroma consumer has to ask for
// them itself -- a SimpleFermBC with (1,1,1,-1).

// QDP++ -> qc.  The direction a chroma plugin needs: there the gauge field
// arrives from chroma's named-object map, already read by chroma's own reader,
// and qc has to be handed a copy.
inline void gauge_from_qdp(const QDP::multi1d<QDP::LatticeColorMatrix> & u, void * U) {
  int ld[4];
  qc_local_dims(ld);

  QDP::multi1d<int> co(QDP::Nd);
  for (int t = 0; t < ld[3]; t ++)
    for (int z = 0; z < ld[2]; z ++)
      for (int y = 0; y < ld[1]; y ++)
        for (int x = 0; x < ld[0]; x ++) {
          co[0] = x; co[1] = y; co[2] = z; co[3] = t;
          const int i = QDP::Layout::linearSiteIndex(co);
          const long site = (((long) t * ld[2] + z) * ld[1] + y) * ld[0] + x;
          for (int mu = 0; mu < QDP::Nd; mu ++)
            for (int a = 0; a < QDP::Nc; a ++)
              for (int b = 0; b < QDP::Nc; b ++) {
                const long off = (((site * QDP::Nd + mu) * QDP::Nc + a) * QDP::Nc + b) * 2;
                fld_set(U, off,     (double) u[mu].elem(i).elem().elem(a, b).real());
                fld_set(U, off + 1, (double) u[mu].elem(i).elem().elem(a, b).imag());
              }
        }
}

inline void gauge_to_qdp(const void * U, QDP::multi1d<QDP::LatticeColorMatrix> & u) {
  int ld[4];
  qc_local_dims(ld);
  u.resize(QDP::Nd);

  QDP::multi1d<int> co(QDP::Nd);
  for (int t = 0; t < ld[3]; t ++)
    for (int z = 0; z < ld[2]; z ++)
      for (int y = 0; y < ld[1]; y ++)
        for (int x = 0; x < ld[0]; x ++) {
          co[0] = x; co[1] = y; co[2] = z; co[3] = t;
          const int i = QDP::Layout::linearSiteIndex(co);
          const long site = (((long) t * ld[2] + z) * ld[1] + y) * ld[0] + x;
          for (int mu = 0; mu < QDP::Nd; mu ++)
            for (int a = 0; a < QDP::Nc; a ++)
              for (int b = 0; b < QDP::Nc; b ++) {
                const long off = (((site * QDP::Nd + mu) * QDP::Nc + a) * QDP::Nc + b) * 2;
                u[mu].elem(i).elem().elem(a, b).real() = (REAL) fld_get(U, off);
                u[mu].elem(i).elem().elem(a, b).imag() = (REAL) fld_get(U, off + 1);
              }
        }
}

#endif // QC_QDP_BRIDGE_H
