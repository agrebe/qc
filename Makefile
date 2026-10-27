# Copyright 2026, University of Maryland, All Rights Reserved
# SPDX-License-Identifier: GPL-3.0-or-later

CC = icx
# -fPIC is unconditional so that one set of object files serves both libqc.a
# and libqc.so (see the library section below).  It costs nothing here:
# comparing the compiled output with and without it, no instruction changed
# anywhere on the solve path.  The slowdown people associate with -fPIC comes
# from calls going through an extra jump, and the library build removes that
# anyway when it hides qc's internal names.
CFLAGS = -Wall -g -O3 -march=native -ffast-math -fPIC
LDFLAGS = -g -lm
# Extra flags appended last (command-line hook:
# make XCFLAGS="-fsanitize=address ...")
CFLAGS += $(XCFLAGS)

# Test-only sources (test.c + its header) live in test/, so a production link
# is "everything but test/ and runner.o".  They still build to top-level
# objects (test.o, ...) via vpath, so OBJS and the rules below name them
# unqualified.  -I. lets a test/ source find the library headers up here (a
# quoted include searches the including file's directory, not the working
# directory); -Itest lets runner.c find test.h.
vpath %.c test
CFLAGS += -I. -Itest

# Pin the default goal explicitly rather than leaning on "first rule in the
# file wins": every target below is deliberately ordered for readability, and
# a new rule landing above qc: would otherwise silently steal the default goal.
.DEFAULT_GOAL := qc

# MPI build gate (rank-per-NUMA-domain decomposition; comm.{c,h}).  MPI=0
# compiles the comm layer to single-rank no-ops and links no MPI dependency;
# MPI=1 swaps in the Intel MPI icx wrapper and defines QC_MPI for comm.c.  The
# process grid itself is configured in global_params.h (MPI_GRID_PZ/PT); all
# MPI lives in comm.c, linked in through $(CC)=mpiicx.
# Flipping this flag needs a `make clean` -- make cannot see a -D change.
MPI = 0
ifeq ($(MPI), 1)
  CC = mpiicx
  CFLAGS += -DQC_MPI=1
endif

ifeq ($(CC), gcc)
  # The kernels carry `#pragma unroll` hints in clang/icx spelling, which gcc
  # does not recognize (it wants `#pragma GCC unroll <n>`) and warns about at
  # every site.  It compiles the same code either way, just without the
  # hints.  Those warnings outnumber everything else in the build output by
  # enough that nothing else gets read, so the category is silenced rather
  # than left to hide a real one.
  CFLAGS += -Wno-unknown-pragmas
endif

# Held in a variable rather than appended straight to CFLAGS because anyone
# linking libqc.a needs the same flag on their own link line: it is what
# QC_RUNTIME_LIBS passes to the example and what qc.pc advertises to consumers.
# One spelling, three users, so they cannot drift.
#
# -fopenmp for every compiler, icx included: icx links the same Intel runtime
# under either spelling, but -qopenmp also enables Intel's own OpenMP passes,
# which have crashed some icx releases compiling solver.c.  icx warns that it
# recommends -qopenmp; the warning is harmless.
SERIAL = 0
ifeq ($(SERIAL), 0)
  OPENMP_FLAG = -fopenmp
  CFLAGS += $(OPENMP_FLAG)
endif

# Build precision mode.  `make PRECISION=<mode>` selects one of three:
#
#   mixed  (default) -- double outer solver + single-precision MG
#          preconditioner in one binary.  This is the production configuration.
#          The templated bodies (clover_ferm/coarse/solver/utils) are compiled
#          twice, once per precision, by the _f/_d rules, which set MG_SINGLE
#          per translation unit; no global -DMG_SINGLE is set (precision.h
#          then defaults the driver layer to double and the _f objects select
#          single).
#   double -- the whole binary in fp64, MG included.  The fallback for a solve
#          fp32 cannot carry (very light quark masses, where fp32 underflows)
#          and the numerical oracle for the mixed path.  Note the asm
#          Wilson/clover kernels are single-precision only
#          (#if ASM && MG_SINGLE), so this falls back to the C kernels and
#          halves the SIMD lanes: correctness build, not a performance
#          comparison.  Also ~2x the memory -- at large volumes drop
#          FGMRES_RESTART or num_rhs (global_params.h) to fit.
#   single -- the whole binary in fp32.  Used by the asm-kernel tests, which
#          need the single instantiation to be the only one in the link.
#
# PRECISION is the front door; MIXED / MG_SINGLE below are the internal names
# the preprocessor sees, and PRECISION is what makes the three legal
# combinations the only spellable ones.  Overriding MIXED / MG_SINGLE on the
# command line still works and still wins (make gives command-line variables
# priority), so existing scripts are unaffected -- which is why the mixed
# branch keeps MG_SINGLE at its historical 1 even though nothing reads it
# there: a bare `make MIXED=0` must go on meaning what it always meant.
PRECISION ?= mixed
ifeq ($(PRECISION), mixed)
  MIXED = 1
  # unread under MIXED=1; held at the historical value -- see above
  MG_SINGLE = 1
else ifeq ($(PRECISION), double)
  MIXED = 0
  MG_SINGLE = 0
else ifeq ($(PRECISION), single)
  MIXED = 0
  MG_SINGLE = 1
else
  $(error PRECISION must be one of: mixed, double, single (got '$(PRECISION)'))
endif

ifeq ($(MIXED), 0)
  ifeq ($(MG_SINGLE), 1)
    CFLAGS += -DMG_SINGLE=1
  endif
endif

# vector instructions
AVX512 = 0
AVX2 = 1
ifeq ($(AVX512), 1)
  CFLAGS += -DAVX512=1 -march=skylake-avx512
  # Without this, dense.c's 64-byte vector_size tile is split into ymm pairs.
  # Only that file needs it; the rest of the AVX-512 path is asm and __m512
  # intrinsics.
  dense.o: CFLAGS += -mprefer-vector-width=512
else ifeq ($(AVX2), 1)
  CFLAGS += -DAVX2=1
endif

# Which implementation backs dense.c's GEMM, LU and Bunch-Kaufman entry points.
# This picks an implementation, not an algorithm: the global_params.h switches
# COARSE_SAP_DENSE_INV and BOTTOM_DENSE_INV choose whether the dense block and
# bottom solves are used at all, and every backend here computes the same
# thing.
#   none            qc's own kernels -- no external library, any compiler, any
#                   architecture, and nothing added to qc.pc for consumers
#   mkl             Intel MKL, via the icx -qmkl spelling
#   cblas           any CBLAS + LAPACKE (OpenBLAS, ARMPL, netlib); set BLAS_LIBS
#
# The default follows the compiler, because -qmkl is an icx spelling that no
# other compiler accepts.  An unconditional `mkl` would make `make CC=gcc`
# fail at the link for everyone without oneAPI; an unconditional `none` would
# leave the icx builds -- which have MKL sitting right there -- on kernels
# that are slower in setup.  Naming DENSE_BLAS on the command line overrides
# this in either direction, and `none` is always available as the
# zero-dependency build.  Switching this needs a `make clean` -- make cannot
# see a -D change.
ifeq ($(filter icx mpiicx,$(notdir $(CC))),)
  DENSE_BLAS = none
else
  DENSE_BLAS = mkl
endif
# Two libraries, not one: OpenBLAS ships CBLAS, but on Debian/Ubuntu the
# LAPACKE wrappers (LAPACKE_zgetrf/zgetrs/zhetrf/zhetri, which dense.c calls)
# are not in libopenblas at all -- they live in liblapacke.  Distributions
# that fold LAPACKE into OpenBLAS need only -lopenblas; override BLAS_LIBS
# there.
BLAS_LIBS  = -lopenblas -llapacke
ifeq ($(DENSE_BLAS), mkl)
  CFLAGS += -DDENSE_BLAS_MKL=1 -qmkl=sequential
else ifeq ($(DENSE_BLAS), cblas)
  CFLAGS += -DDENSE_BLAS_CBLAS=1
  LDFLAGS += $(BLAS_LIBS)
else ifneq ($(DENSE_BLAS), none)
  $(error DENSE_BLAS must be none, mkl or cblas -- got '$(DENSE_BLAS)')
endif

# Software prefetching in the Wilson kernel: controls both the C path
# (-DPREFETCH) and the asm kernels, whose prefetch instructions are gated
# behind the assembler symbol WK_PREFETCH.  Editing this line rebuilds the asm
# objects (they list $(MAKEFILE) as a prerequisite).
PREFETCH = 1
ASFLAGS = -g
ifeq ($(PREFETCH), 1)
  CFLAGS += -DPREFETCH=1
  ASFLAGS += --defsym WK_PREFETCH=1
endif

ifeq ($(MIXED), 1)
  # Expose the mixed build to the C preprocessor so the double outer solver can
  # gate the down-convert fusion
  # (solver.c: #if defined(MIXED) && !defined(MG_SINGLE)).
  # The single twin still gets MG_SINGLE via its compile rule, so the fusion
  # fires only in solver_d.o, the one TU that straddles the precision seam.
  CFLAGS += -DMIXED=1
  # Twinned templated bodies (_f single + _d double), built straight from the
  # one body source by the %_f.o / %_d.o rules below, plus the mixed bridge
  # and the double-only driver layer.
  OBJS = clover_ferm_f.o clover_ferm_d.o coarse_f.o coarse_d.o \
         coarse_sap_f.o coarse_sap_d.o \
         solver_f.o solver_d.o utils_f.o utils_d.o \
         kernel_tests_f.o kernel_tests_d.o \
         mixed_bridge.o bootstrap.o driver.o qc.o runner.o test.o
else
  # One precision throughout: no twins, so the templated bodies build once
  # under the generic %.o rule.  mixed_bridge.o / bootstrap.o come along --
  # despite the names they are the MG hierarchy driver and its adaptive setup,
  # not precision-bridging code, and they compile at whatever REAL this build
  # uses.
  OBJS = clover_ferm.o coarse.o coarse_sap.o runner.o solver.o \
         test.o kernel_tests.o utils.o mixed_bridge.o bootstrap.o driver.o qc.o
endif
# Precision-neutral, compiled once (never twinned).  dense.c is fp64 throughout
# whatever REAL the hierarchy uses -- the dense inverses have always been
# computed in double and stored down-converted -- so twinning it would emit the
# same code twice.
OBJS += comm.o dense.o dense_tests.o

HEADERS = site_layout.h global_params.h precision.h mangle.h coarse.h \
          coarse_kernel.h solver.h clover_ferm.h utils.h mixed.h bootstrap.h \
          comm.h mg_params.h driver.h dense.h qc.h qc_mpi.h
MAKEFILE = Makefile

# use assembly
ASM = 1
ifeq ($(ASM), 1)
  CFLAGS += -DASM=1
  OBJS += clover_kernel_avx2.o clover_kernel_avx2_ow.o
  OBJS += clover_kernel_avx512.o clover_kernel_avx512_ow.o
  OBJS += wilson_kernel_avx2.o wilson_kernel_avx512.o
  OBJS += coarse_dslash_kernel_avx2.o coarse_dslash_kernel_avx512.o
endif

# create main executable
qc: $(OBJS) $(HEADERS) $(MAKEFILE)
	$(CC) $(CFLAGS) $(OBJS) $(LDFLAGS) -o $@

# Everything a library consumer links against: the objects minus the test
# harness and its entry point (runner.o carries main, which would collide with
# the example's).  Also the input to libqc.a / libqc.so below.
LIBOBJS = $(filter-out runner.o test.o kernel_tests.o kernel_tests_f.o \
                       kernel_tests_d.o dense_tests.o, $(OBJS))

# ---------------------------------------------------------------------------
# The installable library: libqc.a (static) and libqc.so (shared).
#
# What we want is a library an analysis code can link without side effects: it
# offers the functions in qc.h and is otherwise invisible.  What stands in the
# way is name collisions: qc's internal routines have everyday names -- solve,
# print0, and the _f/_d twin of nearly every function, by the hundred.  If the
# program linking qc also defines solve(), the link either fails or quietly
# picks one of the two.  So every name outside qc.syms has to be made private
# to the library.
#
# That cannot be done object by object, since a name one object hides is a name
# the others can no longer call.  Hence three steps: step 1 links every object
# into one plain object file, step 2 marks everything outside qc.syms private
# in that single file, and step 3 packages the result.
#
# None of it is part of a plain `make` -- only `make lib` and `make install`
# build the library.

# The step-1 object is its own target because it is what to look at when asking
# what the library really compiled to; step 2 cannot be undone.
#
# Three adjustments to that link, each for a concrete reason:
#
#   -nostdlib  Do not copy the system libraries into qc.  Left alone the
#              compiler folds its own libm in, and then a program linking qc
#              gets qc's cos() and sin() rather than its own -- which would
#              quietly change results anywhere in that program, not just in the
#              solver.  With this flag qc merely refers to libm and the
#              application provides it, which is what QC_RUNTIME_LIBS and qc.pc
#              tell it to do.  It also more than halves the archive.
#
#   filter-out The same point for MKL, which only a DENSE_BLAS=mkl build asks
#     -qmkl    for.  -nostdlib only suppresses the libraries the compiler adds
#              on its own; -qmkl is an explicit request, so it survives and
#              copies roughly 1100 MKL routines in.  Removing it by hand leaves
#              MKL a reference like libm.  Harmless when the flag is absent.
#
#   -Wl,-d     Give every variable a definite location at this step.  Two
#              OpenMP bookkeeping variables are otherwise left for a later link
#              to place, and step 2 cannot mark those private -- they would be
#              the only names to escape it.  Safe here because this is the
#              entire library in one file, so no later link needs to combine
#              them with anything.
QC_LINK_CFLAGS = $(filter-out -qmkl=sequential $(OPENMP_FLAG),$(CFLAGS))

# For the same reason, step 1 must not go through the MPI compiler wrapper.
# mpiicx adds the MPI library to every link it performs, and -nostdlib does not
# suppress that any more than it suppresses -qmkl -- so the whole of Intel MPI,
# its internals included, gets copied inside libqc.  The result is two MPI
# libraries in one program: qc talking to its private copy and the application
# to the real one, each with its own record of whether MPI has been started.
# qc_init appears to do nothing from the application's side.  The plain compiler
# underneath produces the same code and adds no MPI, so the merge uses it and
# leaves MPI a reference like everything else.
ifeq ($(MPI), 1)
  QC_LINK_CC = icx
else
  QC_LINK_CC = $(CC)
endif

qc-all.o: $(LIBOBJS) $(MAKEFILE)
	$(QC_LINK_CC) $(QC_LINK_CFLAGS) -nostdlib -r -Wl,-d $(LIBOBJS) -o $@

qc-local.o: qc-all.o qc.syms $(MAKEFILE)
	objcopy --keep-global-symbols=qc.syms $< $@

libqc.a: qc-local.o $(MAKEFILE)
	rm -f $@
	ar rcs $@ qc-local.o

libqc.so: qc-local.o $(MAKEFILE)
	$(CC) $(CFLAGS) -shared qc-local.o $(LDFLAGS) -o $@

# `make lib` builds the static library only.  Both are made from the same
# object, so this is not about cost -- it is that they are not interchangeable
# here and the static one is the safe default.
#
# qc's geometry is fixed when it is compiled: lattice size, num_rhs, the rank
# grid and the precision are built into loop bounds and into generated
# assembly.  Link the archive and a mismatch cannot happen -- the wrong
# library will not produce a working program.  Install a shared library and it
# can, because libqc.so from a 16x16x16x32 build looks exactly like one from a
# 24x24x24x48 build, solver fields are passed as void *, and nothing would
# catch it before the results came out wrong.  Installing both also makes -lqc
# ambiguous, and the linker prefers the shared one, so a C program would pick
# it up without asking.
#
# Build the shared library when you actually need it -- driving qc from Python,
# which cannot load an archive:
#
#     make libqc.so
#
# `make install` then ships whatever has been built, and says which.
#
# qc_config.h and qc.pc come with the archive so that what describes the
# library is generated with the same flags as the library itself; the chroma
# example reads its link line from qc.pc.
lib: libqc.a qc_config.h qc.pc

# The libraries qc needs but does not contain, which whoever links it has to
# add alongside -lqc: OpenMP because the solver is threaded throughout, libm
# always, and a BLAS only if DENSE_BLAS asked for one.  With DENSE_BLAS=none
# (the default for non-icx compilers) this list is just OpenMP and libm, so the
# dense inverses put no vendor library on a downstream consumer's link line;
# the icx default (mkl) adds -qmkl=sequential.  qc.pc
# publishes this same list, and the example and adopt_mpi below link with it
# rather than with $(CFLAGS) -- so what we advertise to users is what we
# actually test.
QC_RUNTIME_LIBS = $(OPENMP_FLAG) -lm
ifeq ($(DENSE_BLAS), mkl)
  QC_RUNTIME_LIBS += -qmkl=sequential
else ifeq ($(DENSE_BLAS), cblas)
  QC_RUNTIME_LIBS += $(BLAS_LIBS)
endif

# ---------------------------------------------------------------------------
# Install: headers, both artifacts, and a pkg-config file.
#
# PREFIX is where the files end up -- `make install PREFIX=/opt/qc`.  DESTDIR is
# a temporary directory to install into instead, which package builders use to
# collect the files before putting them in place; it is deliberately left out of
# qc.pc, since that file has to record where the library will finally live.
# A prefix you own is on neither the pkg-config nor the ld.so search path, so
# `make install` prints the exports that need setting.
PREFIX ?= $(HOME)/.local
DESTDIR ?=
QC_VERSION = 0.1.0

INSTALL_HEADERS = qc.h qc_config.h
ifeq ($(MPI), 1)
  INSTALL_HEADERS += qc_mpi.h
endif

# Bytes per real in a solver field.  The mixed build's public fields are fp64 --
# the single precision lives entirely inside the preconditioner -- so only the
# uniform single build is 4.  cpp cannot answer this one (it needs sizeof),
# which is why the Makefile hands it to the generator rather than the header
# doing it.
ifeq ($(PRECISION), single)
  QC_PRECISION_BYTES = 4
else
  QC_PRECISION_BYTES = 8
endif

# Deliberately not in $(HEADERS): that variable is a prerequisite of every
# object, and this is generated from those headers.  Listing it there would be
# circular.
qc_config.h: gen_qc_config.py avx/hdr_params.py $(HEADERS) $(MAKEFILE)
	python3 gen_qc_config.py $(QC_PRECISION_BYTES) $(MPI) $@

# The geometry goes in Description so that `pkg-config --list-all` says what an
# installed archive is for -- otherwise a libqc.a in /usr/local/lib is anonymous
# and one built for the wrong lattice looks identical to the right one.
qc.pc: qc_config.h $(MAKEFILE)
	@set -e; \
	 eval "$$(sed -n 's/^#define QC_CONFIG_\([A-Z_]*\)  *\([0-9]*\)$$/\1=\2/p' qc_config.h)"; \
	 { \
	   echo "prefix=$(PREFIX)"; \
	   echo 'exec_prefix=$${prefix}'; \
	   echo 'libdir=$${exec_prefix}/lib'; \
	   echo 'includedir=$${prefix}/include'; \
	   echo ""; \
	   echo "Name: qc"; \
	   echo "Description: Wilson-clover multigrid solver, built for $$LX""x$$LY""x$$LZ""x$$LT, num_rhs=$$NUM_RHS, $$PRECISION-byte reals, rank grid $$MPI_GRID_PZ""x$$MPI_GRID_PT"; \
	   echo "Version: $(QC_VERSION)"; \
	   echo 'Cflags: -I$${includedir}'; \
	   echo 'Libs: -L$${libdir} -lqc'; \
	   echo "Libs.private: $(QC_RUNTIME_LIBS)"; \
	 } > $@
	@echo "wrote $@ (Libs.private: $(QC_RUNTIME_LIBS))"

# The shared library is installed only if it has been built, since `make lib`
# does not build it.  Anyone who wants it asked for it.
install: lib qc.pc $(INSTALL_HEADERS)
	install -d $(DESTDIR)$(PREFIX)/include $(DESTDIR)$(PREFIX)/lib/pkgconfig
	install -m 644 $(INSTALL_HEADERS) $(DESTDIR)$(PREFIX)/include
	install -m 644 libqc.a  $(DESTDIR)$(PREFIX)/lib
	install -m 644 qc.pc    $(DESTDIR)$(PREFIX)/lib/pkgconfig
	@if [ -f libqc.so ]; then \
	   install -m 755 libqc.so $(DESTDIR)$(PREFIX)/lib; \
	   echo "installed qc $(QC_VERSION) into $(DESTDIR)$(PREFIX): libqc.a and libqc.so"; \
	   echo "  note: with both present, -lqc selects libqc.so, and a program linked"; \
	   echo "  against it needs -Wl,-rpath,$(PREFIX)/lib or LD_LIBRARY_PATH to run"; \
	 else \
	   echo "installed qc $(QC_VERSION) into $(DESTDIR)$(PREFIX): libqc.a"; \
	 fi
	@echo "  pkg-config will not find qc until you add:"
	@echo "    export PKG_CONFIG_PATH=$(PREFIX)/lib/pkgconfig:\$$PKG_CONFIG_PATH"

# The Python wrapper, installed only when asked for.  Not part of `install`
# because qc is a C library and this is a demonstration of driving it: it
# brings potential failure modes that belong to the user's Python rather than
# to qc -- a second OpenMP runtime, an mpi4py built against a different MPI --
# and nobody should acquire those by installing a C library.  check_omp.py goes
# along with it since diagnosing that is the first thing a new machine needs.
install-python: libqc.so
	install -d $(DESTDIR)$(PREFIX)/share/qc/python
	install -m 644 example/python/qc.py example/python/pion.py \
	               example/python/check_omp.py \
	               $(DESTDIR)$(PREFIX)/share/qc/python
	@echo "installed the Python wrapper into $(DESTDIR)$(PREFIX)/share/qc/python"
	@echo "  export PYTHONPATH=$(PREFIX)/share/qc/python"
	@echo "  export QC_LIBRARY=$(PREFIX)/lib/libqc.so   # or put that dir on LD_LIBRARY_PATH"
	@echo "  then run check_omp.py there before relying on it -- see README.md"

uninstall:
	rm -f $(addprefix $(DESTDIR)$(PREFIX)/include/,$(notdir $(INSTALL_HEADERS)))
	rm -f $(DESTDIR)$(PREFIX)/lib/libqc.a $(DESTDIR)$(PREFIX)/lib/libqc.so
	rm -f $(DESTDIR)$(PREFIX)/lib/pkgconfig/qc.pc

# Checks the library offers the names in qc.syms and no others.
#
# It reads qc-local.o -- the object both libqc.a and libqc.so are made from --
# rather than a finished library.  That asks the same question, since ar and
# -shared neither add nor remove names, and it does not require the shared
# library to have been built.  It also sidesteps a distraction: a shared library
# carries a few names the linker itself appends (__bss_start, _edata, _end),
# which never went through step 2 and which some linkers emit and others do not.
#
# What it expects is qc.syms narrowed to the names this build actually has,
# which is how one list covers every configuration: qc_set_comm is simply
# absent when MPI=0.  The opposite mistake -- a function in qc.h that failed
# to be published -- is caught by test/collide, which refers to all of them
# and would not link.
check-exports: qc-local.o qc-all.o qc.syms
	@nm -g --defined-only qc-all.o | awk '{print $$NF}' | sort -u > .exp-have
	@grep -vE '^[[:space:]]*(#|$$)' qc.syms | tr -d ' \t' | sort -u > .exp-want
	@comm -12 .exp-have .exp-want > .exp-expect
	@nm -g --defined-only qc-local.o | awk '{print $$NF}' | sort -u > .exp-actual
	@if diff -u .exp-expect .exp-actual > .exp-diff; then \
	   echo "check-exports: OK -- the library offers exactly the $$(wc -l < .exp-expect) public symbols"; \
	   rm -f .exp-have .exp-want .exp-expect .exp-actual .exp-diff; \
	 else \
	   echo "check-exports: FAIL -- the library's published names do not match qc.syms"; \
	   echo "  (-) expected from qc.syms   (+) actually present"; \
	   cat .exp-diff; \
	   rm -f .exp-have .exp-want .exp-expect .exp-actual .exp-diff; \
	   exit 1; \
	 fi

# Links a program that deliberately defines its own solve(), print0() and cos()
# against libqc.a, and takes the address of every qc.h entry point.  Both halves
# are load-bearing: a leaked internal symbol makes the link fail on a duplicate
# definition, and a public function that step 2 hid by mistake makes it fail on
# an undefined one.  Built with none of qc's flags, like the example.  It also
# includes qc_config.h and checks every macro in it against the corresponding
# runtime query, which is what keeps the generated header from drifting away
# from the library it is installed beside.
test/collide: test/collide.c libqc.a qc.h qc_config.h $(MAKEFILE)
	$(CC) $(EXAMPLE_CFLAGS) -fno-builtin -c test/collide.c -o test/collide.o
	$(CC) test/collide.o libqc.a $(QC_RUNTIME_LIBS) -o $@

# Checks that qc refers to the libraries it uses instead of containing copies of
# them.  This has gone wrong three times -- libm, then MKL, then MPI -- always
# the same way: -nostdlib only suppresses the libraries the compiler adds on its
# own, so anything asked for explicitly (-qmkl, or the -lmpi that the MPI
# compiler wrapper adds to every link) is copied in regardless.
#
# Swallowing a library is worse than wasteful when the library holds state.  A
# private copy of MPI gave the program two of them, and qc_init started one
# while the application asked the other whether MPI was running -- which
# truthfully answered no.  The families listed here are the libraries qc
# links, which is the whole list of what could be absorbed.  The BLAS families
# stay on the list even though DENSE_BLAS defaults to none: they are exactly
# what a build that does ask for one would swallow, and a pattern that matches
# nothing costs nothing.
check-selfcontained: qc-all.o
	@bad=0; \
	 for pat in '^P?MPI_' '^I_MPI_' '^mkl_' '^MKL_' '^cblas_' '^LAPACKE_' '^GOMP_' '^__kmpc_' '^(cos|sin|sqrt|exp|log|pow)$$'; do \
	   n=$$(nm --defined-only qc-all.o | awk '{print $$NF}' | grep -cE "$$pat"); \
	   if [ "$$n" -gt 0 ]; then \
	     echo "check-selfcontained: FAIL -- library contains $$n symbol(s) matching $$pat"; \
	     echo "                     it should reference these, not carry them; see the library section"; \
	     bad=1; \
	   fi; \
	 done; \
	 if [ $$bad -eq 0 ]; then \
	   echo "check-selfcontained: OK -- $$(nm --defined-only qc-all.o | wc -l) symbols, none from a linked library"; \
	 else exit 1; fi

# Everything the library promises about its names and its generated header.
# libqc.so is listed so the shared path keeps being built and linked even
# though `make lib` no longer produces it -- an artifact nobody builds is one
# that quietly stops working.  Note this leaves a libqc.so in the tree, which
# a later `make install` will then ship; it says so when it does.
check-lib: check-selfcontained check-exports libqc.so test/collide
	@./test/collide

# The Python example, gated the way the C one is: run it and require its
# correlator to match the C path timeslice for timeslice.  Deliberately not
# part of check-lib or any default target -- see install-python below.
#
# It refuses to run rather than report a failure when check_omp.py finds two
# OpenMP runtimes, because in that state the numbers are not the thing under
# test.
check-python: qc libqc.so
	@python3 example/python/check_omp.py > .py-omp 2>&1 || { \
	   echo "check-python: not run -- $$(grep -E '^(PROBLEM|INCONCLUSIVE)' .py-omp | head -1)"; \
	   echo "  this is a property of your Python installation, not of qc;"; \
	   echo "  see 'Driving qc from Python' in README.md"; \
	   rm -f .py-omp; exit 1; }
	@python3 example/python/pion.py > .py-out 2>&1 || { \
	   echo "check-python: FAIL -- the example did not complete"; \
	   tail -5 .py-out; rm -f .py-omp .py-out; exit 1; }
	@./qc correlator > .py-ref 2>&1
	@grep -E '^ +[0-9]+ +[-0-9.e+]+$$'            .py-out | awk '{print $$1, $$2}' > .py-tbl
	@grep -E '^ +[0-9]+ +[-0-9.e+]+ +[-0-9.e+]+'  .py-ref | awk '{print $$1, $$2}' > .py-reftbl
	@# Both must be non-empty: two empty tables compare equal, so without this a
	@# broken run or a changed print format would pass as a match.
	@if [ ! -s .py-tbl ] || [ ! -s .py-reftbl ]; then \
	   echo "check-python: FAIL -- no correlator table parsed (python: $$(wc -l < .py-tbl), C: $$(wc -l < .py-reftbl) rows)"; \
	   rm -f .py-omp .py-out .py-ref .py-tbl .py-reftbl; exit 1; \
	 fi
	@if diff -u .py-reftbl .py-tbl > .py-diff; then \
	   echo "check-python: OK -- $$(wc -l < .py-tbl) timeslices match ./qc correlator"; \
	   rm -f .py-omp .py-out .py-ref .py-tbl .py-reftbl .py-diff; \
	 else \
	   echo "check-python: FAIL -- the Python and C correlators differ"; \
	   echo "  (-) ./qc correlator   (+) example/python/pion.py"; \
	   cat .py-diff; \
	   rm -f .py-omp .py-out .py-ref .py-tbl .py-reftbl .py-diff; exit 1; \
	 fi

# The worked example (example/pion.c).  Compiled with none of qc's build flags:
# no -DMIXED, no -DAVX2, no -DASM, no -Itest, no -march.  Only -I. to find
# qc.h.  That is the point of the target -- if qc.h ever starts depending on a
# build-configuration macro, or on an internal header, this stops compiling.
#
# The link is now equally spare: libqc.a plus $(QC_RUNTIME_LIBS), rather than
# all the objects under the full $(CFLAGS).  It is what a user would actually
# type, so the flags qc.pc recommends get tested instead of merely asserted,
# and it can only reach the exported functions, so anything missing from
# qc.syms shows up here too.
#
# One deliberate consequence: this link has no -ffast-math, so the example runs
# with denormal numbers handled the standard way, while ./qc flushes them to
# zero.  That is what a real application will see -- the floating-point note in
# qc.h explains it.
EXAMPLE_CFLAGS = -O2 -Wall -Wextra -std=c99 -I.

example/pion: example/pion.c libqc.a qc.h $(MAKEFILE)
	$(CC) $(EXAMPLE_CFLAGS) -c example/pion.c -o example/pion.o
	$(CC) example/pion.o libqc.a $(QC_RUNTIME_LIBS) -o $@

# MPI ownership gate (test/adopt_mpi.c): a driver with its own main, because the
# question it asks -- what happens when the caller initializes MPI first -- is
# already settled by the time any ./qc arm runs (runner.c calls comm_init up
# front).  Links libqc.a like the example, and builds at MPI=0 too, where it
# compiles to a skip message.  Run: mpirun -n 2 ./test/adopt_mpi rev
test/adopt_mpi: test/adopt_mpi.c libqc.a qc.h qc_mpi.h qc_config.h $(MAKEFILE)
	$(CC) $(EXAMPLE_CFLAGS) -c test/adopt_mpi.c -o test/adopt_mpi.o
	$(CC) test/adopt_mpi.o libqc.a $(QC_RUNTIME_LIBS) -o $@

# The generated kernels bake in num_rhs (and num_null_vecs), which the
# generators read through hdr_params.py -- i.e. from the C preprocessor, so
# they do not care which header holds a given #define.  Make cannot follow
# #includes, though, so the .s targets below depend on the whole of
# $(HEADERS): that way a parameter moving to another header still triggers
# regeneration, instead of leaving stale assembly compiled against the old
# value.  Regeneration takes well under a second and its output is
# deterministic.
GEN_DEPS = $(HEADERS) avx/hdr_params.py

# the fully generated (gitignored) asm kernels; clean removes them so a stale
# kernel compiled against an old header value cannot survive a clean rebuild
# (the .S templates in avx/ are tracked source and are never removed).
GEN_ASM = avx/wilson_kernel_avx2.s avx/coarse_dslash_kernel_avx2.s \
          avx/coarse_dslash_kernel_avx512.s

WILSON_GEN = avx/gen_wilson_avx2.py
avx/wilson_kernel_avx2.s: $(WILSON_GEN) $(GEN_DEPS)
	python3 $(WILSON_GEN)

# clover_kernel_avx2/avx512 and wilson_kernel_avx512 are cpp/GAS templates (.S)
# rather than Python-generated .s: each #includes global_params.h for num_rhs
# (the header's C-only tail is __ASSEMBLER__-guarded), the S(D) macro rescales
# the hand-tuned num_rhs=16 schedule to the actual num_rhs, and the same
# assembler (as) used for the other kernels assembles the cpp output -- so at
# num_rhs=16 each AVX2 object is byte-identical to the former hand-written .s.
# All require num_rhs % 16 == 0.  (wilson_kernel_avx512 additionally relies on
# the C caller tiling its 16-column kernel num_rhs/16 times;
# clover_kernel_avx512 tiles internally, like the AVX2 clover kernel.)
# -I. resolves their #include "global_params.h" from avx/, where a quoted
# include would otherwise search only the template's own directory.
clover_kernel_avx2.o: avx/clover_kernel_avx2.S $(HEADERS) $(MAKEFILE)
	$(CC) -E -P -I. $< | as $(ASFLAGS) -o $@
# Overwrite variant (output = F*ferm, no accumulate): same template assembled
# with CLOVER_OW=1 -> symbol clover_kernel_avx2_ow (first output loads zeroed).
clover_kernel_avx2_ow.o: avx/clover_kernel_avx2.S $(HEADERS) $(MAKEFILE)
	$(CC) -E -P -I. -DCLOVER_OW=1 $< | as $(ASFLAGS) -o $@
clover_kernel_avx512.o: avx/clover_kernel_avx512.S $(HEADERS) $(MAKEFILE)
	$(CC) -E -P -I. $< | as $(ASFLAGS) -o $@
clover_kernel_avx512_ow.o: avx/clover_kernel_avx512.S $(HEADERS) $(MAKEFILE)
	$(CC) -E -P -I. -DCLOVER_OW=1 $< | as $(ASFLAGS) -o $@
wilson_kernel_avx512.o: avx/wilson_kernel_avx512.S $(HEADERS) $(MAKEFILE)
	$(CC) -E -P -I. $< | as $(ASFLAGS) -o $@

# coarse Dslash microkernel: regenerate when the generator or the parameters it
# bakes in (num_rhs, num_null_vecs -> N_coarse) change.
COARSE_GEN = avx/gen_coarse_dslash_avx2.py
avx/coarse_dslash_kernel_avx2.s: $(COARSE_GEN) $(GEN_DEPS)
	python3 $(COARSE_GEN)

COARSE_GEN512 = avx/gen_coarse_dslash_avx512.py
avx/coarse_dslash_kernel_avx512.s: $(COARSE_GEN512) $(GEN_DEPS)
	python3 $(COARSE_GEN512)

# The generated assembly lives in avx/ beside its generator; the objects stay here, so
# the link line and OBJS name them unqualified.
%_kernel_avx2.o: avx/%_kernel_avx2.s $(MAKEFILE)
	as $(ASFLAGS) -c $< -o $@

%_kernel_avx512.o: avx/%_kernel_avx512.s $(MAKEFILE)
	as $(ASFLAGS) -c $< -o $@

%.o : %.c $(HEADERS) $(MAKEFILE)
	$(CC) -c $(CFLAGS) $< -o $@

# $(HEADERS) is the library's own headers and no library translation unit
# includes test/test.h, so it is deliberately not on that list.  The three test
# bodies that do include it need it named here: without this, editing test.h --
# GAUGE_PATH, the kernel-bench sizes -- recompiles nothing, and make reports the
# stale binary as up to date rather than failing.
test.o dense_tests.o kernel_tests_f.o kernel_tests_d.o: test/test.h

# Mixed-precision twins: compile a templated body foo.c into both a single
# (foo_f.o, MG_SINGLE set) and a double (foo_d.o) object directly from the one
# source -- no per-precision wrapper .c files needed.  These rules are more
# specific than the generic %.o rule above (their stem omits the _f/_d suffix
# and maps to the bare body, e.g. coarse_f.o <- coarse.c), and the generic
# rule is rejected for these targets because no coarse_f.c exists.  Editing
# the body rebuilds both twins automatically.
%_f.o : %.c $(HEADERS) $(MAKEFILE)
	$(CC) -c $(CFLAGS) -DMG_SINGLE=1 $< -o $@
%_d.o : %.c $(HEADERS) $(MAKEFILE)
	$(CC) -c $(CFLAGS) $< -o $@

# solver.c is special: its single twin must also drop the un-mangled solve()
# (emitted exactly once, by solver_d.o) via SOLVE_OMIT.  An explicit rule beats
# the %_f.o pattern.  (mixed_bridge.c is a real source file, not a twin; it sets
# MG_SINGLE internally and builds via the generic %.o rule.)
solver_f.o : solver.c $(HEADERS) $(MAKEFILE)
	$(CC) -c $(CFLAGS) -DMG_SINGLE=1 -DSOLVE_OMIT=1 $< -o $@

# *.o covers qc-all.o and qc-local.o; the objects under example/ and test/ are
# outside that glob and have to be named.
clean:
	rm -f *.o qc example/pion example/pion.o test/adopt_mpi test/adopt_mpi.o $(GEN_ASM)
	rm -f libqc.a libqc.so test/collide test/collide.o .exp-*
	rm -f qc_config.h qc.pc

.PHONY: clean lib check-exports check-selfcontained check-lib check-python \
        install install-python uninstall
