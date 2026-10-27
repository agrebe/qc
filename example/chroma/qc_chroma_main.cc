// Copyright 2026, University of Maryland, All Rights Reserved
// Copyright (c) 2018 Jefferson Science Associates LLC All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later AND BSD-3-Clause

// A chroma driver that knows about qc.
//
// This is chroma's own mainprogs/main/chroma.cc, trimmed to what an example
// needs and with one line added: InlineQcPropEnv::registerAll(), which puts
// QC_PROPAGATOR into the same measurement factory as everything chroma ships.
// From that point on qc is not a special case -- it is a measurement name in an
// XML file, and its output is an ordinary named LatticePropagator that a
// following measurement can consume.
//
// Existing chroma measurements are registered too (InlineAggregateEnv), so a
// chain can mix them freely with QC_PROPAGATOR.
//
//   ./qc_chroma -i qc_prop.xml -o qc_prop.out.xml
//
// qc_init/qc_finalize are the driver's job, once per process, which is why the
// measurement itself does not call them.

#include "chroma.h"

#include "inline_qc_prop.h"
#include "qc.h"

using namespace Chroma;

namespace
{
  struct Cfg_input_t
  {
    multi1d<int>  nrow;
    std::string   inline_measurement_xml;
    GroupXML_t    cfg;
    QDP::Seed     rng_seed;
  };

  void readInput(XMLReader& xml, Cfg_input_t& p)
  {
    XMLReader inputtop(xml, "/chroma");

    // <Cfg> is a factory group, keyed on cfg_type, not a plain struct.
    p.cfg = readXMLGroup(inputtop, "Cfg", "cfg_type");

    XMLReader paramtop(inputtop, "Param");
    read(paramtop, "nrow", p.nrow);

    {
      XMLReader measurements_xml(paramtop, "InlineMeasurements");
      std::ostringstream inline_os;
      measurements_xml.print(inline_os);
      p.inline_measurement_xml = inline_os.str();
    }

    if (inputtop.count("RNG") == 1)
      read(inputtop, "RNG", p.rng_seed);
    else
      p.rng_seed = 11;
  }

  // chroma's own main only prints its linkage result and carries on, because
  // the aggregate registration reports false for benign reasons (components
  // built out, names already claimed).  So the aggregates are reported and not
  // enforced; only qc's own registration is fatal, since that is the one this
  // driver exists to add and the one whose failure would be silent otherwise.
  bool registerEverything()
  {
    const bool inl = InlineAggregateEnv::registerAll();  // everything chroma ships
    const bool gau = GaugeInitEnv::registerAll();        // the <Cfg> readers
    const bool qcp = InlineQcPropEnv::registerAll();     // and qc

    QDPIO::cout << "qc_chroma: registration -- inline aggregate " << inl
                << ", gauge init " << gau << ", " << InlineQcPropEnv::name
                << " " << qcp << std::endl;
    return qcp;
  }
}


int main(int argc, char* argv[])
{
  Chroma::initialize(&argc, &argv);

  START_CODE();

  if (! registerEverything())
  {
    QDPIO::cerr << "qc_chroma: measurement registration failed" << std::endl;
    QDP_abort(1);
  }

  // qc adopts whatever MPI state it finds, so chroma having already started MPI
  // is the expected case and not a conflict.  Once per process, before any
  // measurement runs; the measurement itself never calls these.
  qc_init(&argc, &argv);

  XMLReader xml_in;
  Cfg_input_t input;
  try
  {
    xml_in.open(Chroma::getXMLInputFileName());
    readInput(xml_in, input);
  }
  catch (const std::string& e)
  {
    QDPIO::cerr << "qc_chroma: caught exception reading XML: " << e << std::endl;
    QDP_abort(1);
  }

  XMLFileWriter& xml_out = Chroma::getXMLOutputInstance();
  push(xml_out, "chroma");
  write(xml_out, "Input", xml_in);

  Layout::setLattSize(input.nrow);
  Layout::create();

  proginfo(xml_out);

  QDP::RNG::setrn(input.rng_seed);
  write(xml_out, "RNG", input.rng_seed);

  // --- gauge field ---------------------------------------------------------
  multi1d<LatticeColorMatrix> u(Nd);
  XMLReader gauge_file_xml, gauge_xml;
  try
  {
    std::istringstream xml_c(input.cfg.xml);
    XMLReader cfgtop(xml_c);
    QDPIO::cout << "qc_chroma: gauge initialization, cfg_type = " << input.cfg.id << std::endl;

    Handle< GaugeInit > gaugeInit(
      TheGaugeInitFactory::Instance().createObject(input.cfg.id, cfgtop, input.cfg.path));
    (*gaugeInit)(gauge_file_xml, gauge_xml, u);
  }
  catch (const std::string& e)
  {
    QDPIO::cerr << "qc_chroma: caught exception during gaugeInit: " << e << std::endl;
    QDP_abort(1);
  }

  XMLBufferWriter config_xml;
  config_xml << gauge_xml;
  write(xml_out, "Config_info", gauge_xml);

  MesPlq(xml_out, "Observables", u);

  // --- run the measurement chain -------------------------------------------
  try
  {
    std::istringstream measurements_is(input.inline_measurement_xml);
    XMLReader meas_xml(measurements_is);
    multi1d< Handle< AbsInlineMeasurement > > the_measurements;
    read(meas_xml, "/InlineMeasurements", the_measurements);

    QDPIO::cout << "qc_chroma: " << the_measurements.size() << " measurement(s)" << std::endl;

    InlineDefaultGaugeField::reset();
    InlineDefaultGaugeField::set(u, config_xml);

    push(xml_out, "InlineObservables");
    xml_out.flush();

    const unsigned long cur_update = 0;
    for (int m = 0; m < the_measurements.size(); m ++)
    {
      AbsInlineMeasurement& the_meas = *(the_measurements[m]);
      if (cur_update % the_meas.getFrequency() == 0)
      {
        push(xml_out, "elem");            // caller writes the elem rule
        the_meas(cur_update, xml_out);
        pop(xml_out);
        xml_out.flush();
      }
    }
    pop(xml_out);   // InlineObservables

    InlineDefaultGaugeField::reset();
  }
  catch (const std::string& e)
  {
    QDPIO::cerr << "qc_chroma: caught exception running measurements: " << e << std::endl;
    QDP_abort(1);
  }

  pop(xml_out);   // chroma

  qc_finalize();

  END_CODE();

  Chroma::finalize();
  return 0;
}
