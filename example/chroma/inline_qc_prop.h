// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

// -*- C++ -*-
// Inline measurement: a forward propagator solved by qc.
//
// A worked example of driving qc from chroma. It is a normal chroma inline
// measurement -- registered in TheInlineMeasurementFactory, configured from
// XML, taking its gauge field from the named-object map and leaving a
// LatticePropagator there -- so it drops into a measurement chain like any
// other.
//
// The one thing it does differently from chroma's own INLINE_PROPAGATOR is
// where the batching lives. Chroma's SystemSolver interface takes one
// right-hand side per call, which would leave 15 of qc's 16 lanes idle and make
// any timing look about 16x worse than the solver really is. So this is an
// inline measurement rather than a SystemSolver: it gathers all twelve
// spin-colors of the point source and hands them to qc in a single solve.

#ifndef __inline_qc_prop_h__
#define __inline_qc_prop_h__

#include "chromabase.h"
#include "meas/inline/abs_inline_measurement.h"

namespace Chroma
{
  namespace InlineQcPropEnv
  {
    extern const std::string name;
    bool registerAll();

    struct Params
    {
      Params();
      Params(XMLReader& xml_in, const std::string& path);

      unsigned long frequency;

      // double, not Real.  Real is float when QDP++ is built at
      // BASE_PRECISION 32, and reading kappa into a float silently shifts it:
      // float(0.134) differs from 0.134 by ~4e-8 relative, which near critical
      // mass is enough to move a propagator by ~1e-5.  These are thresholds and
      // couplings, not field data, so they cost nothing to keep in double.
      struct Param_t
      {
        double        kappa;      // hopping parameter
        double        csw;        // clover coefficient
        double        tol;        // relative residual target
        int           maxIter;    // cap on outer iterations
        multi1d<int>  srcPos;     // point source location, 4 coords
      } param;

      struct NamedObject_t
      {
        std::string   gauge_id;   // input gauge field
        std::string   prop_id;    // output propagator
      } named_obj;

      std::string xml_file;
    };

    class InlineMeas : public AbsInlineMeasurement
    {
    public:
      ~InlineMeas() {}
      InlineMeas(const Params& p) : params(p) {}
      InlineMeas(const InlineMeas& p) : params(p.params) {}

      unsigned long getFrequency(void) const { return params.frequency; }

      void operator()(unsigned long update_no, XMLWriter& xml_out);

    protected:
      void func(const unsigned long update_no, XMLWriter& xml_out);

    private:
      Params params;
    };
  }
}

#endif
