// Copyright 2026, University of Maryland, All Rights Reserved
// Copyright (c) 2018 Jefferson Science Associates LLC All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later AND BSD-3-Clause

/*! \file
 *  \brief Inline measurement: a forward propagator solved by qc.
 */

#include "inline_qc_prop.h"

#include "meas/inline/abs_inline_measurement_factory.h"
#include "meas/inline/make_xml_file.h"
#include "meas/inline/io/named_objmap.h"
#include "meas/inline/io/default_gauge_field.h"
#include "meas/glue/mesplq.h"
#include "util/ferm/transf.h"
#include "util/info/proginfo.h"
#include "util/info/unique_id.h"
#include "io/qprop_io.h"

#include <vector>

#include "qc.h"
#include "qc_config.h"
#include "bridge.h"

namespace Chroma
{
  //! Params input
  void read(XMLReader& xml, const std::string& path,
            InlineQcPropEnv::Params::Param_t& param)
  {
    XMLReader paramtop(xml, path);

    int version;
    read(paramtop, "version", version);
    if (version != 1)
    {
      QDPIO::cerr << "InlineQcPropEnv::Params::Param_t: version " << version
                  << " unsupported." << std::endl;
      QDP_abort(1);
    }

    read(paramtop, "kappa",   param.kappa);
    read(paramtop, "csw",     param.csw);
    read(paramtop, "tol",     param.tol);
    read(paramtop, "maxIter", param.maxIter);
    read(paramtop, "srcPos",  param.srcPos);

    if (param.srcPos.size() != Nd)
    {
      QDPIO::cerr << "InlineQcPropEnv: srcPos needs " << Nd << " entries"
                  << std::endl;
      QDP_abort(1);
    }
  }

  //! Params output
  void write(XMLWriter& xml, const std::string& path,
             const InlineQcPropEnv::Params::Param_t& param)
  {
    push(xml, path);
    write(xml, "version", 1);
    write(xml, "kappa",   param.kappa);
    write(xml, "csw",     param.csw);
    write(xml, "tol",     param.tol);
    write(xml, "maxIter", param.maxIter);
    write(xml, "srcPos",  param.srcPos);
    pop(xml);
  }

  //! Named object input
  void read(XMLReader& xml, const std::string& path,
            InlineQcPropEnv::Params::NamedObject_t& input)
  {
    XMLReader inputtop(xml, path);
    read(inputtop, "gauge_id", input.gauge_id);
    read(inputtop, "prop_id",  input.prop_id);
  }

  //! Named object output
  void write(XMLWriter& xml, const std::string& path,
             const InlineQcPropEnv::Params::NamedObject_t& input)
  {
    push(xml, path);
    write(xml, "gauge_id", input.gauge_id);
    write(xml, "prop_id",  input.prop_id);
    pop(xml);
  }


  namespace InlineQcPropEnv
  {
    namespace
    {
      AbsInlineMeasurement* createMeasurement(XMLReader& xml_in,
                                              const std::string& path)
      {
        return new InlineMeas(Params(xml_in, path));
      }

      bool registered = false;

      //! Timeslice set for the correlator.  Named to avoid colliding with
      //! anything chroma defines in this namespace.
      class QcTimeSliceFunc : public SetFunc
      {
      public:
        explicit QcTimeSliceFunc(int dir) : dir_(dir) {}
        int operator()(const multi1d<int>& coord) const { return coord[dir_]; }
        int numSubsets() const { return Layout::lattSize()[dir_]; }
      private:
        int dir_;
      };
    }

    const std::string name = "QC_PROPAGATOR";

    bool registerAll()
    {
      bool success = true;
      if (! registered)
      {
        success &= TheInlineMeasurementFactory::Instance().registerObject(name, createMeasurement);
        registered = true;
      }
      return success;
    }


    Params::Params()
    {
      frequency = 0;
      param.kappa   = 0.0;
      param.csw     = 0.0;
      param.tol     = 1.0e-10;
      param.maxIter = 100;
      param.srcPos.resize(Nd);
      for (int i = 0; i < Nd; i ++) param.srcPos[i] = 0;
      named_obj.gauge_id = InlineDefaultGaugeField::getId();
      named_obj.prop_id  = "";
      xml_file = "";
    }

    Params::Params(XMLReader& xml_in, const std::string& path)
    {
      try
      {
        XMLReader paramtop(xml_in, path);

        if (paramtop.count("Frequency") == 1)
          read(paramtop, "Frequency", frequency);
        else
          frequency = 1;

        read(paramtop, "Param", param);
        read(paramtop, "NamedObject", named_obj);

        if (paramtop.count("xml_file") != 0)
          read(paramtop, "xml_file", xml_file);
      }
      catch(const std::string& e)
      {
        QDPIO::cerr << "InlineQcPropEnv: caught exception reading XML: " << e << std::endl;
        QDP_abort(1);
      }
    }


    void InlineMeas::operator()(unsigned long update_no, XMLWriter& xml_out)
    {
      if (params.xml_file != "")
      {
        std::string xml_file = makeXMLFileName(params.xml_file, update_no);

        push(xml_out, "qc_propagator");
        write(xml_out, "update_no", update_no);
        write(xml_out, "xml_file", xml_file);
        pop(xml_out);

        XMLFileWriter xml(xml_file);
        func(update_no, xml);
      }
      else
      {
        func(update_no, xml_out);
      }
    }


    void InlineMeas::func(unsigned long update_no, XMLWriter& xml_out)
    {
      START_CODE();

      StopWatch snoop;
      snoop.reset();
      snoop.start();

      // --- gauge field, from the named-object map --------------------------
      multi1d<LatticeColorMatrix> u;
      XMLBufferWriter gauge_xml;
      try
      {
        u = TheNamedObjMap::Instance().getData< multi1d<LatticeColorMatrix> >(params.named_obj.gauge_id);
        TheNamedObjMap::Instance().get(params.named_obj.gauge_id).getRecordXML(gauge_xml);
      }
      catch (std::bad_cast)
      {
        QDPIO::cerr << name << ": caught dynamic cast error" << std::endl;
        QDP_abort(1);
      }
      catch (const std::string& e)
      {
        QDPIO::cerr << name << ": map call failed: " << e << std::endl;
        QDP_abort(1);
      }

      push(xml_out, "qc_propagator");
      write(xml_out, "update_no", update_no);
      proginfo(xml_out);
      write(xml_out, "Input", params.param);
      MesPlq(xml_out, "Observables", u);

      // --- the compile-time geometry gate ----------------------------------
      // qc's lattice extents, num_rhs and precision are baked into the archive.
      // Handing it a field sized for anything else is not diagnosed by the
      // library -- solver fields are void * -- so this check is the only thing
      // between a mismatch and silent corruption.
      {
        int gd[4];
        qc_global_dims(gd);
        bool ok = true;
        for (int i = 0; i < Nd; i ++)
          if (Layout::lattSize()[i] != gd[i]) ok = false;
        if (!ok)
        {
          QDPIO::cerr << name << ": lattice mismatch -- chroma has "
                      << Layout::lattSize()[0] << "x" << Layout::lattSize()[1] << "x"
                      << Layout::lattSize()[2] << "x" << Layout::lattSize()[3]
                      << ", this libqc was built for "
                      << gd[0] << "x" << gd[1] << "x" << gd[2] << "x" << gd[3]
                      << ".  qc's geometry is compile-time; rebuild libqc."
                      << std::endl;
          QDP_abort(1);
        }
      }

      // --- the decomposition gate ------------------------------------------
      // qc decomposes z and t only (MPI_GRID_PZ/PT; there is no PX or PY),
      // while QDP++ parscalar splits all Nd dimensions.  So the two agree at
      // one rank and need not agree at more, and where they disagree the site
      // map built below is silently wrong rather than merely slow.  Compare the
      // slabs directly: this is cheap, and it is the difference between a clear
      // abort and a corrupt propagator on somebody's cluster.
      {
        int ld[4], og[4];
        qc_local_dims(ld);
        qc_local_origin(og);
        bool ok = true;
        for (int i = 0; i < Nd; i ++)
          if (Layout::subgridLattSize()[i] != ld[i]) ok = false;
        if (!ok)
        {
          QDPIO::cerr << name << ": decomposition mismatch -- QDP++ gives this"
                      << " rank a " << Layout::subgridLattSize()[0] << "x"
                      << Layout::subgridLattSize()[1] << "x"
                      << Layout::subgridLattSize()[2] << "x"
                      << Layout::subgridLattSize()[3] << " slab, qc expects "
                      << ld[0] << "x" << ld[1] << "x" << ld[2] << "x" << ld[3]
                      << ".  qc decomposes z and t only, so set MPI_GRID_PZ/PT"
                      << " to match QDP++'s layout (or run one rank)."
                      << std::endl;
          QDP_abort(1);
        }
        (void) og;
      }

      const int nsc = Ns * Nc;                 // 12 spin-colors
      if (qc_num_rhs() < nsc)
      {
        QDPIO::cerr << name << ": this libqc has num_rhs = " << qc_num_rhs()
                    << ", below the " << nsc << " a propagator needs" << std::endl;
        QDP_abort(1);
      }

      // --- hand the links to qc and build the hierarchy --------------------
      StopWatch swatch;
      swatch.reset();
      swatch.start();

      void* U = qc_alloc(qc_gauge_elements());
      gauge_from_qdp(u, U);

      qc_setup_params sp;
      qc_default_setup_params(&sp);            // defaults first, then the physics:
      sp.kappa = params.param.kappa;           // a designated initializer here would
      sp.csw   = params.param.csw;             // zero every knob it failed to name
      qc_handle* h = qc_setup(U, qc_native_precision(), &sp);
      qc_free(U);

      swatch.stop();
      QDPIO::cout << name << ": hierarchy built in " << swatch.getTimeInSeconds()
                  << " sec" << std::endl;

      // --- the twelve sources, solved in one batch -------------------------
      // This is why the example is an inline measurement and not a
      // SystemSolver: chroma's solver interface takes one right-hand side per
      // call, which would idle 15 of qc's 16 lanes.  Here all twelve
      // spin-colors go in together.  Lanes past twelve stay zero, which qc
      // handles -- a lane whose source norm is zero is skipped in the
      // convergence test and frozen.
      const std::vector<long> map = build_site_map();

      void* b = qc_alloc(qc_field_elements());
      void* x = qc_alloc(qc_field_elements());
      qc_field_zero(b);

      multi1d<int> srce = params.param.srcPos;
      for (int sc = 0; sc < nsc; sc ++)
      {
        const int s = sc / Nc, c = sc % Nc;
        LatticeFermion src = zero;
        Fermion f;
        f = zero;
        f.elem().elem(s).elem(c).real() = 1.0;
        pokeSite(src, f, srce);
        import_lane(b, src, map, sc);
      }

      swatch.reset();
      swatch.start();

      qc_solve_params vp;
      qc_default_solve_params(&vp);
      vp.tol         = params.param.tol;
      vp.max_iter    = params.param.maxIter;
      vp.chroma_norm = 1;                      // return the solution in chroma's
                                               // normalization (2*kappa)
      const int iters = qc_solve(h, x, b, &vp);

      swatch.stop();
      QDPIO::cout << name << ": " << nsc << " right-hand sides in one solve, "
                  << iters << " outer iterations, " << swatch.getTimeInSeconds()
                  << " sec" << std::endl;

      write(xml_out, "outer_iterations", iters);
      write(xml_out, "num_rhs_solved", nsc);
      write(xml_out, "num_rhs_available", qc_num_rhs());
      write(xml_out, "solve_seconds", swatch.getTimeInSeconds());

      // --- assemble a LatticePropagator ------------------------------------
      LatticePropagator quark_prop = zero;
      for (int sc = 0; sc < nsc; sc ++)
      {
        const int s = sc / Nc, c = sc % Nc;
        LatticeFermion psi = zero;
        export_lane(x, psi, map, sc);
        FermToProp(psi, quark_prop, c, s);
      }

      // --- the pion, so the measurement is self-checking --------------------
      // localNorm2 of a propagator sums |.|^2 over all 144 spin-color
      // components, which at zero momentum is exactly the pion correlator.
      //
      // It is computed twice on purpose.  The QDP++ one is what a chroma
      // measurement naturally produces, and it is limited by LatticePropagator
      // being single precision in this build (QDP++ is BASE_PRECISION 32) --
      // each timeslice sums 4096 sites of float.  The second reads the same
      // solution straight out of qc in double.  Reporting the gap between them
      // is what keeps the single-precision loss a measured number instead of an
      // unexplained discrepancy against a double-precision reference.
      const int Lt = Layout::lattSize()[Nd - 1];
      multi1d<Double> corr_f, corr_d(Lt);
      {
        Set timeslice;
        timeslice.make(QcTimeSliceFunc(Nd - 1));
        corr_f = sumMulti(localNorm2(quark_prop), timeslice);
      }
      {
        for (int t = 0; t < Lt; t ++) corr_d[t] = Double(0);
        int ld[4], og[4];
        qc_local_dims(ld);
        qc_local_origin(og);
        std::vector<double> lane((size_t) qc_lane_elements());
        for (int sc = 0; sc < nsc; sc ++)
        {
          qc_export(x, lane.data(), QC_DOUBLE, sc);
          for (int t = 0; t < ld[3]; t ++)
          {
            double s = 0;
            for (int z = 0; z < ld[2]; z ++)
              for (int y = 0; y < ld[1]; y ++)
                for (int xx = 0; xx < ld[0]; xx ++)
                {
                  const long base = ((((long) t * ld[2] + z) * ld[1] + y) * ld[0] + xx) * nsc;
                  for (int i = 0; i < nsc; i ++)
                    s += lane[2 * (base + i)]     * lane[2 * (base + i)]
                       + lane[2 * (base + i) + 1] * lane[2 * (base + i) + 1];
                }
            corr_d[t + og[3]] += Double(s);
          }
        }
      }

      qc_free(b);
      qc_free(x);
      qc_teardown(h);

      {
        double worst = 0;
        for (int t = 0; t < Lt; t ++)
        {
          const double a = toDouble(corr_f[t]), c = toDouble(corr_d[t]);
          if (c != 0) worst = std::max(worst, std::fabs(a - c) / std::fabs(c));
        }

        push(xml_out, "Pion");
        write(xml_out, "correlator", corr_f);
        write(xml_out, "correlator_double", corr_d);
        write(xml_out, "single_precision_loss", worst);
        pop(xml_out);

        QDPIO::cout << name << ": pion correlator" << std::endl;
        for (int t = 0; t < Lt; t ++)
          QDPIO::cout << "  " << t << "  " << toDouble(corr_d[t]) << std::endl;
        QDPIO::cout << name << ": QDP++ single-precision correlator differs from"
                    << " the double one by " << worst << " relative" << std::endl;
      }

      // --- leave it in the named-object map --------------------------------
      // Written with a real Propagator_t record so the result is an ordinary
      // chroma forward propagator, not a special case: a following measurement
      // reads it the same way it reads INLINE_PROPAGATOR's output.
      try
      {
        PropSourceConst_t source_header;
        source_header.j_decay  = Nd - 1;
        source_header.t_source = params.param.srcPos[Nd - 1];
        {
          XMLBufferWriter xml;
          push(xml, "Source");
          write(xml, "version", 1);
          write(xml, "SourceType", "POINT_SOURCE");
          write(xml, "j_decay", Nd - 1);
          write(xml, "t_srce", params.param.srcPos);
          pop(xml);
          source_header.source.xml  = xml.printCurrentContext();
          source_header.source.id   = "POINT_SOURCE";
          source_header.source.path = "/Source";
        }

        ChromaProp_t prop_header;
        prop_header.quarkSpinType = QUARK_SPIN_TYPE_FULL;
        prop_header.obsvP         = false;
        {
          XMLBufferWriter xml;
          push(xml, "FermionAction");
          write(xml, "FermAct", "CLOVER");
          write(xml, "Kappa", params.param.kappa);
          write(xml, "clovCoeff", params.param.csw);
          multi1d<int> bnd(Nd);
          bnd[0] = bnd[1] = bnd[2] = 1;
          bnd[3] = -1;              // qc applies anti-periodic time boundaries
          push(xml, "FermionBC");
          write(xml, "FermBC", "SIMPLE_FERMBC");
          write(xml, "boundary", bnd);
          pop(xml);
          pop(xml);
          prop_header.fermact.xml  = xml.printCurrentContext();
          prop_header.fermact.id   = "CLOVER";
          prop_header.fermact.path = "/FermionAction";
        }
        {
          XMLBufferWriter xml;
          push(xml, "InvertParam");
          write(xml, "invType", "QC_MG");
          write(xml, "RsdTarget", params.param.tol);
          write(xml, "MaxIter", params.param.maxIter);
          pop(xml);
          prop_header.invParam.xml  = xml.printCurrentContext();
          prop_header.invParam.id   = "QC_MG";
          prop_header.invParam.path = "/InvertParam";
        }

        Propagator_t rec;
        rec.prop_header   = prop_header;
        rec.source_header = source_header;
        rec.gauge_header  = gauge_xml.printCurrentContext();

        XMLBufferWriter file_xml;
        push(file_xml, "propagator");
        write(file_xml, "id", uniqueId());
        pop(file_xml);

        XMLBufferWriter record_xml;
        write(record_xml, "Propagator", rec);

        TheNamedObjMap::Instance().create<LatticePropagator>(params.named_obj.prop_id);
        TheNamedObjMap::Instance().getData<LatticePropagator>(params.named_obj.prop_id) = quark_prop;
        TheNamedObjMap::Instance().get(params.named_obj.prop_id).setFileXML(file_xml);
        TheNamedObjMap::Instance().get(params.named_obj.prop_id).setRecordXML(record_xml);
      }
      catch (std::bad_cast)
      {
        QDPIO::cerr << name << ": caught dynamic cast error storing the propagator" << std::endl;
        QDP_abort(1);
      }
      catch (const std::string& e)
      {
        QDPIO::cerr << name << ": storing the propagator failed: " << e << std::endl;
        QDP_abort(1);
      }

      pop(xml_out);   // qc_propagator

      snoop.stop();
      QDPIO::cout << name << ": total time = " << snoop.getTimeInSeconds()
                  << " sec" << std::endl;

      END_CODE();
    }
  }
}
