# Driving qc from QDP++ and chroma

Three worked examples of calling qc from a C++ lattice code. They use `qc.h`
only — no internal qc header and none of qc's build flags — so they are also a
test that the public interface is usable from outside.

| program | needs | what it does |
|---|---|---|
| `pion_qdp` | QDP++ | Builds the twelve spin-color point sources as `LatticeFermion`s, solves them in one qc call, computes C(t) with QDP++ primitives. Five layout checks run first. |
| `qc_vs_chroma` | QDP++, chroma | Compares qc's Wilson-clover operator against chroma's `UnprecCloverLinOp`, and confirms a qc solve using chroma's operator. |
| `qc_chroma` | QDP++, chroma | A real chroma `InlineMeasurement`, `QC_PROPAGATOR`, that writes an ordinary forward propagator into the named-object map. |

Nothing in the main qc build refers to this directory. QDP++ and chroma are not
qc dependencies; build these by hand when you have them.

## Build

Build `libqc.a` at the top of the qc tree first (`make lib`), then:

    make                                    # pion_qdp
    make chroma                             # qc_vs_chroma and qc_chroma
    make check-openmp                       # see below -- run this once

Point the paths at your own installs on the command line or in the environment:

    make QDPXX=$HOME/install/qdpxx CHROMA_SRC=$HOME/src/chroma \
         CHROMA_BUILD=$HOME/build/chroma chroma

Then:

    ./pion_qdp [gauge-file]                 # --checks-only skips the solve
    ./qc_vs_chroma [gauge-file]
    ./qc_chroma -i qc_prop.xml -o qc_prop.out.xml

The gauge file defaults to `../../test/sample_config.lime`; for
`qc_chroma` it is the `<cfg_file>` entry in `qc_prop.xml`.

## Two compilers, one OpenMP runtime

The one part of the build that is easy to get wrong. QDP++ is normally built
with gcc and libqc with icx, and a program using both risks loading two OpenMP
runtimes into one process — which oversubscribes the machine and can deadlock.

It is avoidable because QDP++ needs only the `omp_*` query functions from
OpenMP, which Intel's runtime also provides. So the QDP++ side is compiled
**without** `-fopenmp` (with it, the inlined pragmas in `qdp_dispatch.h` emit
`GOMP_*` calls and drag in libgomp), and the link is done by `icpx`. QDP++'s own
parallel regions then run serially in these files, which costs nothing — they
only marshal data. `make check-openmp` greps the object and fails on any
`GOMP_` reference.

Before trusting this against a different QDP++, check whether it has parallel
regions of its own:

    nm --undefined-only $QDPXX/lib/libqdp.a | grep -c GOMP_

A nonzero count means the recipe does not apply, and libqc should be built with
gcc rather than icx.

## The layout bridge

Chroma's fermion fields are QDP++ `LatticeFermion`s, laid out red/black with a
lexicographic order inside each parity. qc's fields are parity-major with a 4⁴
block order inside each parity. Neither can read the other's memory, so
something has to walk sites; that walk is `bridge.h`, and it is the only file
here that knows anything about layout.

It goes through `qc.h`'s published accessors — `qc_site_index` and
`qc_component_index` — rather than reimplementing qc's block arithmetic, so a
future change to qc's storage order does not break it. Per-site component order
needs no conversion at all: QDP++ is spin-outer, color-inner, complex innermost,
which is already qc's order.

`pion_qdp` runs five checks on this before doing anything else, and each is
mutation-tested (`--mutate=1,2,3` injects a specific defect). One result is
worth knowing if you write your own bridge: **a site map that ignores parity is
still a bijection and still self-consistent**, so both the bijection check and
the round trip pass it. A round trip applies the same map forwards and
backwards and cannot see a consistent permutation. Only the parity check and the
accessor-vs-`qc_import` cross-check catch it. The injection sites compile out
unless `MUTATION_HOOKS` is defined, so a copy of `bridge.h` carries none of them.

## Right-hand sides: why a measurement and not a SystemSolver

qc solves `num_rhs` right-hand sides per call — 16 in a typical build. Chroma's
`SystemSolver` interface takes one per call, so wrapping qc that way would idle
15 of 16 lanes and make any timing look about 16× worse than the solver is.

`QC_PROPAGATOR` is therefore an `InlineMeasurement`, not a `SystemSolver`: it
gathers all twelve spin-colors of the point source and hands them to qc in a
single `qc_solve`. Lanes past twelve stay zero, which qc handles — a lane whose
source norm is zero is skipped in the convergence test and frozen — so a build
with `num_rhs = 16` wastes lattice bandwidth on four idle lanes but returns the
same twelve propagator columns.

If your workflow needs qc behind chroma's solver interface instead, that is a
different piece of work, and it gives up most of the reason to use qc.

## Convention mapping

Measured against chroma's own operator by `qc_vs_chroma`, not assumed:

| qc | chroma |
|---|---|
| `kappa` | `Mass = 1/(2*kappa) - Nd` (chroma's `A` carries `Nd + Mass`) |
| `csw` | `clovCoeffR = clovCoeffT`, with `u0 = 1` |
| anti-periodic time boundaries, applied internally | ask for them: `SimpleFermBC` with `(1,1,1,-1)` |
| DeGrand-Rossi gamma basis | the same |

With those, `qc_D = +2*kappa * chroma_M`, and with `chroma_norm = 1` qc returns
a solution already in chroma's normalization needing no further rescale.
`qc_vs_chroma` fits both constants from the data rather than assuming them, and
reports the residual after removing them — an assumed factor that happened to be
wrong would otherwise look like a real disagreement.

Both halves of that pass condition are load-bearing. Mutation runs
(`--mutate=1,2,3`: periodic time boundary, `Mass` off 1%, `clovCoeff = 0`) move
the residual by three to six orders of magnitude but leave the fitted scale
within 0.06% of `2*kappa`. The residual catches a wrong operator *shape*; the
scale catches a wrong *normalization*; neither subsumes the other.

## Using the measurement in your own chroma

Chroma needs no source edits. `TheInlineMeasurementFactory` is a runtime
singleton, so a measurement registers itself from a translation unit chroma
knows nothing about, and `libchroma.a` is consumed as a black box. The price is
running your own driver — `qc_chroma_main.cc` is chroma's `chroma.cc` trimmed,
with one added line, `InlineQcPropEnv::registerAll()`. Putting `QC_PROPAGATOR`
into the stock `chroma` binary instead would mean editing
`lib/meas/inline/inline_aggregate.cc` and `lib/Makefile.am`, and re-merging on
every chroma upgrade.

Because the output is a real `Propagator_t` record, stock measurements consume
it unchanged: `qc_prop.xml` chains `SINK_SMEAR` and `HADRON_SPECTRUM` onto qc's
propagator, and neither knows qc exists.

Three things to get right when porting:

**Read solver parameters into `double`, not `Real`.** `Real` is float when
QDP++ is built at `BASE_PRECISION 32`, so `read(paramtop, "kappa", ...)` into a
`Real` stores `float(0.134)` — about 4e-8 off, which near critical mass moved
the propagator by 8.7e-06 and looked exactly like a bridge bug. Chroma's own
echo shows it: `clovCoeff 1.87566995620728` for an input of `1.87567`.
Thresholds and couplings are not field data.

**The lattice is compile-time in qc.** A libqc archive is valid for one
geometry, one `num_rhs` and one precision. Solver fields are `void *`, so
handing it a field sized for anything else is not diagnosed by the library. The
measurement compares `Layout::lattSize()` against `qc_global_dims()` and aborts
with both printed.

**qc and QDP++ decompose differently under MPI.** qc splits z and t only
(`MPI_GRID_PZ`/`PT`; there is no `PX` or `PY`), while QDP++ `parscalar` splits
all four dimensions. They agree trivially at one rank and need not agree at
more, and where they disagree the site map is silently wrong rather than merely
slow. The measurement gates on `Layout::subgridLattSize()` against
`qc_local_dims()` and aborts with both slabs printed. **Only the one-rank case
has been run**; see below.

## If you need to build chroma

Two things make it much smaller and faster than its reputation suggests — the
whole tree here is 256 MB including a 91 MB `libchroma.a`:

- **Override autoconf's default `CXXFLAGS`.** It is `-g -O2`, and the debug info
  on chroma's template-heavy C++ is the entire reason chroma is reputed to need
  many GB. Configure with `CXXFLAGS=-O1 CFLAGS=-O1` and no `-g`.
- **`other_libs/qdp-lapack` is a required submodule** that `--depth 1` does not
  fetch. `configure.ac` always builds it and `inv_eigcg2.cc` includes its header
  regardless of the eigcg flag, so the build dies on a missing `qdp-lapack.h`.
  Run `git submodule update --init --depth 1 other_libs/qdp-lapack`, then
  `autoreconf -f -i` inside it, then re-run chroma's configure.

Configure with `--with-qdp=<your QDP++ prefix>`, and build `other_libs/qdp-lapack`
before `lib`.

## What has been checked, and against what

Validated 2026-08-12 against **chroma 91c587d** (2023-07-24), **QDP++ 1.46.0**
built `ARCH_SCALAR` / `BASE_PRECISION 32` with **g++-9**, and libqc built with
icx at `16x16x16x32`, `num_rhs = 16`, mixed precision.

- `pion_qdp`'s C(t) table is identical to `example/pion`'s on all 32 timeslices.
- `qc_vs_chroma`: operators agree at residual 8.8e-08 with the fitted scale
  matching `2*kappa` to eight digits; chroma's operator applied to qc's solution
  returns the source with scale 1.00000000 at residual 1.4e-07. The floor is
  ~1e-7 because QDP++ here is single precision.
- `qc_chroma`'s correlator agrees with `example/pion` to 3.4e-07 worst-case,
  which is qc's own solver-tolerance floor rather than anything about the
  bridge. Chroma's `HADRON_SPECTRUM`, run on qc's propagator, reproduces the
  same pion in the gamma=15 channel to 3.8e-08, with no other gamma channel
  within a factor of two.

These are external projects on their own release schedules, and nothing in qc's
build compiles this directory — so if you are reading this well after that date,
expect the chroma-side interfaces to have moved.

## Not covered

One rank only, so the rank-origin subtraction in `bridge.h` is exercised but
always by zero, and the decomposition gate above has never fired in anger.
`qc_apply_D` is checked against chroma's clover operator but at a single point
in parameter space. The round trip is bitwise only because QDP++ is single
precision here and qc's fields are double; at a QDP++ built with
`BASE_PRECISION 64` it would need a tolerance.

## Attribution

Chroma and QDP++ are separate projects under their own terms: Copyright (c)
Jefferson Science Associates LLC, distributed under a three-clause BSD license.
Chroma's notice, conditions and disclaimer are reproduced in
`THIRD-PARTY-NOTICES` alongside this file, as that license requires of anything
redistributing its source.

Two files here derive from chroma source under those terms:

- `qc_chroma_main.cc` — chroma's `mainprogs/main/chroma.cc`, trimmed, with
  `InlineQcPropEnv::registerAll()` added.
- `inline_qc_prop.cc` — follows chroma's `AbsInlineMeasurement` plugin
  interface, so its registration and parameter-I/O scaffolding matches that of
  every other chroma inline measurement.

Those terms reach these examples only. The qc library neither links nor includes
chroma or QDP++.
