# Copyright 2026, University of Maryland, All Rights Reserved
# SPDX-License-Identifier: GPL-3.0-or-later

# Build the qc <-> QDP++/chroma examples.
#
# Nothing in the main qc build refers to this directory.  QDP++ and chroma are
# not qc dependencies, so these are built by hand, here, when you have them.
#
#   make                 pion_qdp     -- needs QDP++ only
#   make chroma          qc_vs_chroma and qc_chroma -- also need chroma
#   make check-openmp    confirm the link will not load two OpenMP runtimes
#
# Point the four paths below at your own installs, on the command line or in the
# environment:
#
#   make QDPXX=$HOME/install/qdpxx CHROMA_BUILD=$HOME/build/chroma chroma

QDPXX        ?= $(HOME)/install/qdpxx-scalar
CHROMA_SRC   ?= $(HOME)/src/chroma
CHROMA_BUILD ?= $(HOME)/build/chroma
QCDIR        ?= ../..

QDPCONF := $(QDPXX)/bin/qdp++-config

# Two compilers on purpose, and this is the part that is easy to get wrong.
#
# QDP++ is normally built with gcc and libqc with icx, so a program using both
# risks loading two OpenMP runtimes into one process -- which oversubscribes the
# machine and can deadlock.  It is avoidable here because QDP++ needs only the
# omp_* query functions, which Intel's runtime also provides.  The conditions:
#
#   - compile the QDP++ side without -fopenmp, so no GOMP_* call is emitted
#     (`make check-openmp` checks this; QDP++'s own parallel regions then run
#     serially in these files, which costs nothing -- they only marshal data)
#   - link with icpx, which supplies the Intel OpenMP runtime libqc was
#     built against
#
# Before trusting the recipe against a different QDP++, run
# `nm --undefined-only $(QDPXX)/lib/libqdp.a | grep -c GOMP_`.  A nonzero count
# means that QDP++ has gcc-built parallel regions of its own, and libqc should
# be built with gcc rather than icx.
CXX := g++-9
LD  := icpx

# No -Wextra: QDP++'s expression-template headers emit thousands of
# unused-parameter warnings that bury real diagnostics.
CXXFLAGS = -O2 -Wall -std=c++14 $(shell $(QDPCONF) --cxxflags) -I$(QCDIR)
LDFLAGS  = -fopenmp $(shell $(QDPCONF) --ldflags)
QDPLIBS  = $(shell $(QDPCONF) --libs) -lm

# chroma is linked from its build tree rather than an install prefix, because
# chroma-config points at --prefix and `make install` would duplicate a few
# hundred MB of archive and headers for no benefit.
CHROMA_INC  = -I$(CHROMA_SRC)/lib -I$(CHROMA_BUILD)/lib \
              -I$(CHROMA_SRC)/other_libs/qdp-lapack/include \
              -I$(CHROMA_BUILD)/other_libs/qdp-lapack/include
CHROMA_LIBS = $(CHROMA_BUILD)/lib/libchroma.a \
              $(CHROMA_BUILD)/other_libs/qdp-lapack/lib/libqdp-lapack.a

all: pion_qdp

chroma: qc_vs_chroma qc_chroma

QCDEP = $(QCDIR)/qc.h $(QCDIR)/qc_config.h $(QCDIR)/qc.pc $(QCDIR)/libqc.a

# What libqc needs on the link line (OpenMP, libm, and a BLAS if it was built
# with one), as recorded in qc.pc when libqc was built.
QCLIBS = $(shell sed -n 's/^Libs.private: //p' $(QCDIR)/qc.pc)

# Run before anything compiles, so a missing dependency names the variable to
# set rather than failing somewhere inside a header.
check-qdp:
	@test -x $(QDPCONF) || { \
	  echo "no qdp++-config at $(QDPCONF)"; \
	  echo "set QDPXX to your QDP++ install prefix"; exit 1; }

check-chroma: check-qdp
	@test -f $(CHROMA_BUILD)/lib/libchroma.a || { \
	  echo "no libchroma.a under $(CHROMA_BUILD)"; \
	  echo "set CHROMA_SRC and CHROMA_BUILD to your chroma source and build trees"; \
	  exit 1; }

# These are real prerequisites, so rebuilding libqc relinks the examples.  The
# recipe runs only when one is absent, and exists to say so in one line.
$(QCDEP):
	@echo "$@ is missing -- run 'make lib' at the top of the qc tree"; exit 1

pion_qdp: pion_qdp.o $(QCDIR)/libqc.a
	$(LD) $< -o $@ $(LDFLAGS) $(QCDIR)/libqc.a $(QCLIBS) $(QDPLIBS)

pion_qdp.o: pion_qdp.cc bridge.h $(QCDEP) | check-qdp
	$(CXX) $(CXXFLAGS) -c $< -o $@

qc_vs_chroma: qc_vs_chroma.o $(QCDIR)/libqc.a
	$(LD) $< -o $@ $(LDFLAGS) $(QCDIR)/libqc.a $(QCLIBS) $(CHROMA_LIBS) $(QDPLIBS)

qc_vs_chroma.o: qc_vs_chroma.cc bridge.h $(QCDEP) | check-chroma
	$(CXX) $(CXXFLAGS) $(CHROMA_INC) -c $< -o $@

qc_chroma: qc_chroma_main.o inline_qc_prop.o $(QCDIR)/libqc.a
	$(LD) qc_chroma_main.o inline_qc_prop.o -o $@ $(LDFLAGS) \
	  $(QCDIR)/libqc.a $(QCLIBS) $(CHROMA_LIBS) $(QDPLIBS)

qc_chroma_main.o: qc_chroma_main.cc inline_qc_prop.h $(QCDEP) | check-chroma
	$(CXX) $(CXXFLAGS) $(CHROMA_INC) -c $< -o $@

inline_qc_prop.o: inline_qc_prop.cc inline_qc_prop.h bridge.h $(QCDEP) | check-chroma
	$(CXX) $(CXXFLAGS) $(CHROMA_INC) -c $< -o $@

# The link above stays safe only while libgomp stays out of these objects.
check-openmp: pion_qdp.o
	@if nm -u $^ | grep -q GOMP_; then \
	  echo "FAIL: a GOMP_ reference would pull libgomp in alongside libiomp5"; \
	  exit 1; \
	else echo "ok: no GOMP_ references, one OpenMP runtime serves both libraries"; fi

clean:
	rm -f pion_qdp qc_vs_chroma qc_chroma *.o *.out.xml

.PHONY: all chroma clean check-qdp check-chroma check-openmp
