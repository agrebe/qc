<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo-dark.png">
    <img src="logo.png" width="400" alt="qc logo">
  </picture>
</p>

# qc

A multigrid solver for the Wilson-clover Dirac operator in lattice QCD. It inverts the operator
against many right-hand sides at once (`num_rhs`, typically 16 or 32), on one node or many.
The algorithms and their performance are described in the associated proceedings (see
[Citing](#citing)).

## What it does

The production solve is a flexible GMRES outer iteration preconditioned by a multigrid
K-cycle of configurable depth (`NUM_LEVELS` in `global_params.h`; 3 by default, typically
higher in production): Schwarz alternating-procedure (SAP) smoothers on the fine and
intermediate grids, and a direct solve on the coarsest grid through a precomputed dense
inverse (GMRES is the alternative, and the fallback). The outer iteration is
even-odd (Schur) preconditioned. Coarse grids come from adaptive aggregation -- null vectors
from a bootstrap setup phase, then a Galerkin operator assembled directly on each level.

Precision is mixed by design: the outer iteration and its restart residual are fp64, which is what
anchors the final accuracy, while the preconditioner runs in fp32. The two precisions coexist in
one binary -- the templated translation units are compiled twice, once per precision. Uniform
fp64 and fp32 builds are also available (`PRECISION=double|single`), the first as the numerical
oracle for the mixed path and the fallback for masses fp32 cannot carry.

Also here: generated AVX2 / AVX-512 assembly kernels for the fine and coarse Dslash (`avx/`),
a pion-correlator integration test checked against DDalphaAMG, and the simpler solvers kept as
references for the multigrid path (Richardson, BiCGStab), along with the red-black Schwarz smoother.

## Build

Needs a C compiler (icx or gcc), GNU make, and Python 3.6 or newer -- the AVX kernels in `avx/`
are generated at build time by the scripts there, which read `num_rhs` and friends from the
headers. `MPI=1` additionally needs an MPI implementation. The build uses Intel MPI's
`mpiicx` wrapper by default; for another MPI, pass its wrapper (`make qc MPI=1 CC=mpicc`).
Building the library with `MPI=1` also calls the plain compiler underneath the wrapper,
which the Makefile sets as `QC_LINK_CC = icx`. If your MPI wrapper drives gcc, clang or
any compiler other than icx, change that line to name it.

`initialize.sh` sets up the environment qc is built and run in: it loads Intel oneAPI, pins
OpenMP threads to cores, and keeps glibc's `malloc` from returning memory to the operating
system between allocations, which otherwise costs measurable time in the solve. Source it in
the shell you build in and in every job script that runs qc. Its line
`source /opt/intel/oneapi/setvars.sh intel64` is what puts icx on the path on the development
machine; replace it with whatever does that on yours (a different path or a `module load`), or
delete it if icx is already in your environment or you are not building with icx.

`initialize.sh` deliberately does not set a thread count, because the right one depends on
the machine. Your scheduler or site environment may already set it; if not, you may need to
set `OMP_NUM_THREADS` to the number of cores each rank owns (the cores per socket, when you
run one rank per socket) for best performance.

The default build compiles with `-march=native`, so build on the same kind of node you will
run on, not on a login node with a different CPU. An `AVX512=1` build targets a fixed AVX-512
baseline instead, so it can be built anywhere.

Everything that sizes the build -- lattice size, `num_rhs`, block sizes, number of levels and
null vectors, process grid -- is a `#define` in `global_params.h`. There are no environment
overrides and no runtime config file, so changing one of these means a recompile. qc is designed
to compile in a couple of minutes, so these recompiles are cheap for the end user. The solver's
tuning (smoother sweeps, cycle type, coarse-grid budgets) and the outer solve's tolerance,
iteration cap and restart length have their defaults in `mg_params.h` and `global_params.h`.
The `qc` binary runs at those defaults; a library caller can override them per setup or solve
(see [Using qc from your own code](#using-qc-from-your-own-code)).

```
make qc                     # mixed precision (fp64 outer / fp32 preconditioner)
make qc AVX512=1            # on any CPU with AVX-512 -- see below
make qc MPI=1               # multi-rank; set the process grid in global_params.h
make qc PRECISION=double    # uniform fp64, the numerical oracle for the mixed path
make qc CC=gcc              # without oneAPI: no MKL, qc's own dense kernels
```

`make clean` removes the objects, the binary, and the generated assembly. Flipping a `make`
flag needs one, since make cannot see a `-D` change.

The default build targets x86 with icx: it enables the AVX2 intrinsics (`AVX2=1`) and
the hand-written assembly kernels (`ASM=1`), and links Intel MKL (sequential) for the dense
factorizations and GEMMs behind the block and bottom inverses. With any other compiler, or
with `DENSE_BLAS=none`, the build links no external linear-algebra library and uses qc's own
kernels (`dense.c`). On another architecture, turn off only what is actually x86:

```
make qc AVX2=0 ASM=0 CC=gcc
```

That build is portable C -- slower, but it produces the same answers.

**On a CPU with AVX-512, build with `AVX512=1`.** The default is the AVX2 path, which runs on
any recent x86 CPU but can be 2x slower on an AVX-512 machine.

The dense coarse-SAP block inverse and the dense coarsest-level inverse
(`COARSE_SAP_DENSE_INV`, `BOTTOM_DENSE_INV` in `global_params.h`, both on by default)
need no external library, so a build on another architecture should leave them on.
Turning either off selects a genuinely different algorithm -- an iterative block
smoother or an iterative coarsest solve -- which converges differently.

`DENSE_BLAS` selects the dense backend: `mkl` is Intel MKL (the default with icx),
`cblas BLAS_LIBS=-lopenblas` uses any CBLAS + LAPACKE, and `none` uses qc's own kernels and
needs no external library (the default with any other compiler). All three compute the
same thing.

## Run

`./qc` (equivalently `./qc correlator`) runs the pion-correlator integration test.
`runner.c` is the entry point, and it carries the full list of gates: run `./qc` with any
unrecognized command to print them, with annotations for the ones that need a gauge file
or a 1x1 grid.

Run from the top of the repository: the gates that need a gauge configuration read the
bundled `test/sample_config.lime` by that relative path. It is a 16x16x16x32 lattice, the
size `global_params.h` ships with, so those gates only run at the default lattice size.
Thread count comes from `OMP_NUM_THREADS` as usual; for a multi-rank build, set the process
grid in `global_params.h` and launch one rank per NUMA domain with your MPI launcher
(`mpirun -n 2 ./qc`, for example).

```
./qc              # pion correlator against a DDalphaAMG reference
./qc mg3          # synthetic 3-level cycle (exercises the coarse SAP)
./qc layout       # qc.h's published field layout against this build
./qc params       # qc.h's parameter defaults, and that the validator rejects bad ones
```

## Performance

Solve times for qc 1.0.0 on two near-physical-mass ensembles (m_pi ~ 170 MeV), 16
right-hand sides, solved to a relative residual of 1e-10. Times exclude setup and the
even-odd reconstruction. Efficiency is relative to the node's fp32 peak (cores x all-core FMA
clock x flops/cycle). Details are in the associated proceedings (see [Citing](#citing)).

| Machine | Node | fp32 peak (TF/s) | STREAM Triad (GB/s) |
|---|---|---|---|
| LQ1 (FNAL) | 2x Cascade Lake, 40 cores, AVX-512 | 6.4 | 200 |
| lqcd (BNL) | 2x Zen 4, 64 cores, AVX-512 | 7.7 | 761 |
| Perlmutter (NERSC) | 2x Zen 3, 128 cores, AVX2 | 10.9 | 332 |

| Ensemble | Machine | Nodes | Node-seconds | TF/s per node | % of peak |
|---|---|---|---|---|---|
| A: 32^3 x 48, 4 levels | LQ1 | 1/2 | 7.81 | 1.54 | 24 |
| | lqcd | 1/8 | 2.95 | 4.09 | 53 |
| | Perlmutter | 1/8 | 3.85 | 3.13 | 29 |
| B: 64^3 x 128, 5 levels | LQ1 | 8 | 280 | 1.18 | 18 |
| | lqcd | 1 | 92.8 | 3.58 | 47 |
| | Perlmutter | 4 | 166 | 2.00 | 18 |

A is a MILC clover-on-HISQ configuration (blocking 4^4 / 2^4 / 2^3x3, null vectors
48 / 96 / 256). B is an NME isoclover configuration (4^4 / 2^4 / 2^4 / 2^4,
48 / 80 / 128 / 256). Fractional node counts are one rank per NUMA domain.
Parameters were tuned on lqcd. LQ1 uses a larger bottom grid on B to reach its rank count,
and Perlmutter runs one rank per socket on B.

These numbers are a reference point, not a ceiling. On comparable hardware, a solve that
runs well below these rates almost always has a fixable cause. The usual ones are:

- Threads and pinning: an `OMP_NUM_THREADS` inherited from an environment script, a
  launcher option such as Slurm's `-c` that counts hardware threads rather than cores,
  or ranks not bound to their NUMA domain or socket.
- Memory placement: a rank whose pages do not live on the cores running it. On Perlmutter,
  one rank per socket without `numactl --interleave` across that socket's domains was 36%
  slower.
- Multigrid parameters (blocking, null-vector counts, smoother iterations) not tuned for the
  ensemble and RHS count.

## Testing

A full sweep is the commands below plus the cross-decomposition halo comparison
that follows them. Nothing here needs anything beyond the build itself and the bundled
gauge configuration.

```
./qc all               # every self-contained gate in one command
make example/pion      # the public header stands alone, end to end
make test/adopt_mpi    # MPI ownership (MPI builds; a no-op skip at MPI=0)
make check-lib         # libqc.a / libqc.so export exactly the public API
make check-python      # optional: the Python example matches the C one
```

`./qc all` runs each gate and prints its own verdict; scan the output rather than trusting
the exit status alone, and note that it deliberately excludes the two dump-writing arms
below, which have no verdict of their own until their output is compared across grids.

`make example/pion` is a gate as well as a demonstration: it compiles against `qc.h` with
none of the build's flags, so it stops compiling if the public header ever starts
depending on one. Run it (`./example/pion`) and its correlator table must match `./qc`'s.

`make check-lib` builds both libraries (`make lib` alone gives you only `libqc.a`) and
checks its symbol surface two ways. `check-exports` asserts the shared
library exports exactly the list in `qc.syms` and nothing else; `test/collide` links a
program that deliberately defines its own `solve`, `print0` and `cos` against the static
archive while referencing every `qc.h` entry point, so a leaked internal symbol fails the
link on a duplicate definition and a missing public one fails it on an undefined
reference. The `cos` case is the one to understand: `icx` materializes its own libm under
`-ffast-math`, so an unlocalized archive would silently replace libm's transcendentals
throughout whatever program linked it.

`mpirun -n 2 ./test/adopt_mpi <qcinit|world|rev>` covers the three divisions of MPI
responsibility described under "Using qc from your own code" below: qc owning MPI, the
caller owning it, and the caller owning it on its own communicator. The `rev` mode is the
one with teeth: it hands qc a communicator whose rank order is reversed against
`MPI_COMM_WORLD`, so any call left on the world talks to the wrong process, while the
physics -- which does not care which rank holds which slab -- must come out unchanged.

### Multi-rank halo validation

Every operator that reads a neighbor rank's data is checked bitwise against an
undecomposed reference. Fields are synthesized from global coordinates and each
output site is hashed, so a physical site must carry the same hash under any
decomposition. This catches halo bugs the correlator test cannot: a broken ghost
degrades only the preconditioner, so the solve still converges to the right answer
and the only symptom is an iteration-count regression.

The process grid is a compile-time constant, so this takes two builds:

```
rm -rf /tmp/qc-halo         # start from an empty dump directory -- see below

# 1. reference: MPI_GRID_PZ = MPI_GRID_PT = 1 in global_params.h
make clean && make qc && ./qc halo && ./qc cbuild2

# 2. decomposed: set the grid, rebuild, run on that many ranks
make clean && make qc MPI=1 && mpirun -n 2 ./qc halo && mpirun -n 2 ./qc cbuild2

# 3. compare (exit status is nonzero if any operator differs)
python3 test/compare-halo-dumps.py
```

That currently checks 22 operators -- the fine and even-odd Dslash and its residual
forms, the Schwarz smoother, the coarse operators and transfers, and the coarse Schur
complement.

**Start from an empty dump directory.** Dumps go to `$QC_HALO_DIR` (default
`/tmp/qc-halo`), and the script merges every file it finds there, pairing them by the
grid in the filename. Dumps left over from an earlier campaign are therefore compared
too -- against a reference that may since have been rebuilt -- and report differences
that look exactly like halo bugs.

**The two builds must agree on everything except the grid,** because the comparison is
bitwise: a different `-O` level, `-march`, or a sanitizer changes the hashes and reports
every site as differing. Switching `MPI=0` to `MPI=1` between the two runs, as above, is
fine and is the intended usage -- that flag only reaches `comm.c`, which does nothing on
a 1x1 grid.

## Using qc from your own code

Unlike most LQCD libraries, qc does not take the lattice geometry or the multigrid hierarchy
at run time. The lattice size, `num_rhs`, rank grid, number of multigrid levels, block sizes
and null-vector counts are fixed in `global_params.h`, and the precision on the `make` line,
when the library is built (see [Build](#build)). Set them for your ensemble before running
`make lib`, and rebuild when you move to a different one. A library built for one geometry
cannot solve on another.

**`num_rhs` must be a multiple of the SIMD width:** 16 for AVX-512 and 8 for AVX2. This
may come as a surprise, since most multi-right-hand-side solvers treat the count as a free
parameter, but `qc` uses right-hand sides as the vectorization axis to maximize performance.
This means that the natural count of 12 (one source in every spin-color component) is not
allowed. If you solve several sources per configuration, the best approach is to buffer on
your side: gather the right-hand sides from successive sources into batches of `num_rhs`,
and unpack the solutions the same way. If you need only a single 12-component source
(or a total number of 12-component sources not divisible by 4), the currently supported
workaround is to pad the batch to 16 with zero right-hand sides and discard their solutions,
which costs 25% of the throughput for a single source. qc recognizes a zero right-hand side
and leaves it out of the convergence test, so the padding does not affect the other solutions.

```
make lib                          # libqc.a
make install                      # + qc.h, qc_config.h, qc.pc  (DESTDIR honored)
export PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig:$PKG_CONFIG_PATH
```

`PREFIX` defaults to `$HOME/.local` and needs no special privilege; set it to install
elsewhere. That directory is on neither the `pkg-config` nor the `ld.so` search path,
which is what the export above is for -- `make install` prints it, and prints the
`LD_LIBRARY_PATH` or `-Wl,-rpath` note too if you have built the shared library.

Then build against it. The archive carries none of its own dependencies, so they go on
your link line; `pkg-config` will produce them for you:

```
icx myprog.c $(pkg-config --cflags qc) $(pkg-config --static --libs qc)
```

which for the default icx build expands to `-I$HOME/.local/include -L$HOME/.local/lib -lqc
-fopenmp -lm -qmkl=sequential`. Built with another compiler, the list drops MKL (or carries
`BLAS_LIBS` instead, in a `DENSE_BLAS=cblas` build).

**Static by default, and deliberately.** `make lib` builds only `libqc.a`. qc's geometry
is fixed when it is compiled -- lattice size, `num_rhs`, the rank grid and precision are
built into loop bounds and into generated assembly -- so linking the archive makes a
mismatch impossible: the wrong library simply will not produce a working program. A
shared library reopens that door, because `libqc.so` built for 16x16x16x32 is
indistinguishable from one built for 24x24x24x48, solver fields are passed as `void *`,
and nothing would catch it before the numbers came out wrong. Build it only when you
need it:

```
make libqc.so     # for driving qc from Python, which cannot load an archive
```

`make install` ships whichever you have built and tells you which. With both installed
`-lqc` selects the shared one, and a program linked against it needs
`-Wl,-rpath,$HOME/.local/lib` or `LD_LIBRARY_PATH` to run.

Either way, `pkg-config --list-all` prints the geometry the library was built for, which
is otherwise invisible. `qc_config.h` records the same facts as macros, so you can check
them with `#if` at compile time rather than discovering a mismatch as corruption at run
time.

The library exports the entry points in `qc.h` and nothing else, and it carries no copy of
the system libraries: it references libm and OpenMP -- and a BLAS, in a `DENSE_BLAS=mkl`
(the icx default) or `=cblas` build -- rather than absorbing them, which is why you supply
those flags yourself. Both properties are gated by `make check-lib`, and both are easy to
lose: qc's internals include names as plain as `solve` and `print0`,
and an ordinary library link absorbs Intel's libm whole, which would replace `cos`, `sin`
and `sqrt` for the entire host program.

One thing to know about the link line: qc inherits your process's floating-point
environment and never changes it, while the standalone `qc` binary links `-ffast-math` and
so runs with denormals flushed to zero. If your own build links `-ffast-math` you get the
same environment; otherwise you get IEEE gradual underflow, which is more accurate and
only costs speed if a solve produces values below about 1e-38 in fp32. `qc.h` has the
details.

`qc.h` is the interface. It includes nothing else from this tree, so it will not
rewrite `malloc` or claim the name `D` the way the internal headers do, and it names
no type that changes with the build.

```c
#include "qc.h"

qc_init(&argc, &argv);
void * U = qc_alloc(qc_gauge_elements());
qc_read_gauge_file(U, "conf.lime");

qc_setup_params sp;
qc_default_setup_params(&sp);       /* the tuned method */
sp.kappa = 0.134;                   /* then your physics */
sp.csw   = 1.87567;
qc_handle * h = qc_setup(U, qc_native_precision(), &sp);
qc_free(U);

void * b = qc_alloc(qc_field_elements());
void * x = qc_alloc(qc_field_elements());
qc_field_zero(b);
/* fill b -- see below -- then: */
qc_solve_params vp;
qc_default_solve_params(&vp);
vp.tol = 1e-10;
vp.chroma_norm = 1;
int iters = qc_solve(h, x, b, &vp);
/* read x, solve more right-hand sides on the same configuration, then: */
qc_teardown(h);
qc_free(x);
qc_free(b);
qc_finalize();
```

**Operator normalization (`chroma_norm`).** Codes differ on whether a factor of 2*kappa is
folded into the Dirac operator. qc's operator is the kappa-normalized D = 1 - kappa*H (plus
the clover term), and `chroma_norm = 0` returns D^-1 b. `chroma_norm = 1` returns the
solution for the mass-normalized operator D / (2*kappa) that chroma and DDalphaAMG use,
which is 2*kappa D^-1 b. In QUDA's terms, 0 is `QUDA_KAPPA_NORMALIZATION` and 1 is
`QUDA_MASS_NORMALIZATION` (and `QUDA_ASYMMETRIC_MASS_NORMALIZATION`, which is the same
for a full-system solve). qc applies the factor by rescaling the solution after the
solve, which is an extra pass over every solution field. It is slightly faster to fold
2*kappa into the source when you build it and solve with `chroma_norm = 0` instead: the
solution and the meaning of the relative tolerance are unchanged, and for a point source
this replaces the full-field sweep with a rescaling of a single point.

**Always start from the defaults.** The two structs carry the solver's tuning --
smoother sweeps, cycle type, coarse-grid budgets, restart length -- and every field
you do not touch keeps the value this build is tuned and tested at. Do not build one
with a designated initializer (`qc_setup_params sp = { .kappa = ... };`): C sets every
field you leave out to zero, and zero is a legal-looking wrong answer for most of them
-- zero smoothing sweeps, a zero iteration cap. `qc_setup` and `qc_solve` check what
they are given and stop with the offending field named rather than solving something
you did not ask for, but starting from the defaults is what keeps you out of that
conversation. `./qc params` is the test that holds them to it.

The knobs worth reaching for first, when a new ensemble does not converge the way the
old one did, are `smoother_sweeps` and `k_iters` (more preconditioning per outer
iteration), `coarse_max_iter` and `coarse_tol` (a harder coarsest solve), and
`restart` (a longer outer Krylov space, at proportionally more memory). The lattice
size, process grid, number of right-hand sides and precision are not among them --
those are compile-time, as described above.

`example/pion.c` is a complete worked program: it reads a configuration, solves the
twelve spin-color point sources at the origin, and forms the zero-momentum pion
correlator. `make example/pion` builds it, deliberately with none of qc's build flags,
so it doubles as the check that the header stands alone.

**Getting fields in and out.** There are two ways, and a real calculation wants both.
Write directly into a solver field with `qc_site_index` and `qc_component_index` when
the source is sparse -- a point source is a handful of stores, and copying a
full-volume array to place twelve of them is waste. Use `qc_import` / `qc_export` when
your fields already exist in the conventional lexicographic order (t slowest, x
fastest, spin outside color, complex interleaved); those convert precision as well, so
a double-precision analysis program can drive a single-precision build.

**Check the build-time geometry from your program.** The lattice extents, process grid,
`num_rhs` and precision fixed at the top of this section can be queried (`qc_global_dims`,
`qc_num_rhs`, `qc_native_precision`); check them against what your program assumes --
solver fields are passed as `void *`, so nothing will catch a mismatch for you.

Two limits of this release: one handle may be live at a time, and calls must come from
serial code, since qc parallelizes internally with OpenMP and does not nest.

**MPI is adopted, not owned.** `qc_init` runs `MPI_Init_thread` only if nothing else
has, and `qc_finalize` runs `MPI_Finalize` only if `qc_init` was what started MPI -- so
a program (or an mpi4py session) that initializes MPI itself keeps that responsibility,
and simply calls `qc_init` afterwards. Whoever initializes must ask for at least
`MPI_THREAD_FUNNELED`; qc queries the level and aborts below it, because its exchanges
run on the master thread outside OpenMP parallel regions. Call `qc_finalize` before
your own `MPI_Finalize` -- if MPI is already finalized it does nothing rather than
crash.

To run on a communicator other than `MPI_COMM_WORLD`, call `qc_set_comm` before
`qc_init`. It lives in `qc_mpi.h` rather than `qc.h` so that the main header stays free
of `<mpi.h>` and a serial consumer needs no MPI at all; a serial (`MPI=0`) build
contains no MPI, so the symbol does not exist there. qc communicates on a private
duplicate of whatever it is given, so its fixed message tags cannot collide with yours
and freeing your communicator after `qc_init` does not disturb it.

One rough edge: a fatal qc configuration error calls `MPI_Abort` on `MPI_COMM_WORLD`,
not on qc's communicator, so it takes the whole job down. Aborting only qc's ranks
would leave the rest of a host application deadlocked in a collective with dead peers.

`make test/adopt_mpi` builds the gate for all of this -- see Testing above.

## Driving qc from Python

An example rather than a supported interface. `example/python/qc.py` wraps `libqc.so`
with `ctypes` and hands back numpy views onto qc's own solver fields, so a source is
written and a solution read without copying either. `example/python/pion.py` is the
worked example, and its correlator table matches `./qc correlator` timeslice for
timeslice, serial and multi-rank.

```
make libqc.so
source initialize.sh                        # or the module loads you build with
python3 example/python/pion.py
mpirun -n 2 python3 example/python/pion.py  # in an MPI build

make check-python                           # run it as a gate: must match ./qc correlator
make install-python                         # install the wrapper beside the library
```

Both targets are opt-in and reached from nothing else. `install` does not ship the
wrapper and `check-lib` does not run it, because installing a C library should not
hand you Python's potential failure modes, and a gate that depends on how the user
installed numpy is not a gate on qc.

Call `qc.require(dims=..., num_rhs=..., precision=...)` before allocating anything. It
does for Python what `qc_config.h` does for C: solver fields cross as untyped pointers,
so a library built for another lattice is read as though it were the right one.

This path is the more fragile of the two, for reasons that have little to do with qc:

- **Loading needs the compiler runtime.** `libqc.so` refers to Intel's libm (and to a
  BLAS, in a `DENSE_BLAS=mkl` (the icx default) or `=cblas` build). Python will not find
  those unless you source the same environment you built with, so a batch script needs
  that before it runs python.
- **One OpenMP runtime per process.** qc built with icx brings Intel's; whether numpy
  brings a second depends on how numpy was installed, which varies by machine. Two
  runtimes each size a thread pool from the core count, so the process runs N times
  more threads than intended and every parallel region thrashes -- and setting
  `OMP_NUM_THREADS` does not help, because both runtimes read it and each spawns that
  many. `example/python/check_omp.py` answers this in a couple of seconds and exits
  nonzero on a conflict. Run it inside the batch job rather than on the login node:
  which numpy is on the path can differ between them.
- **One MPI per process**, in an MPI build. mpi4py has to have been built against the
  same MPI qc links; `python3 -c "from mpi4py import MPI; print(MPI.Get_library_version())"`
  reports which one.

With mpi4py present, MPI belongs to Python -- it starts MPI on import and qc adopts it,
so summing a correlator over ranks is yours to do, as the `Allreduce` in `pion.py`
shows. Without mpi4py the example still runs and qc starts MPI itself, but Python then
has no communicator, so each rank can only report its own timeslices.

## Driving qc from QDP++ and chroma

Also an example rather than a supported interface, and built by hand: QDP++ and chroma
are not qc dependencies, so nothing in this Makefile refers to `example/chroma/`. It
has its own.

`pion_qdp` marshals QDP++ `LatticeFermion`s into qc's fields and back, reproducing
`example/pion`'s correlator exactly. `qc_vs_chroma` checks qc's Wilson-clover operator
against chroma's own and reports the convention mapping it fits rather than assumes.
`qc_chroma` is a real chroma `InlineMeasurement` -- `QC_PROPAGATOR` -- that writes an
ordinary forward propagator into the named-object map, so stock `SINK_SMEAR` and
`HADRON_SPECTRUM` consume it without knowing qc exists.

```
make lib
cd example/chroma && make QDPXX=... CHROMA_SRC=... CHROMA_BUILD=... chroma
```

Chroma itself needs no source edits: measurements register through a runtime factory,
so the price is running your own driver rather than the stock `chroma` binary.
`example/chroma/README.md` has the convention table, the two-compilers/one-OpenMP-runtime
link recipe, and what has and has not been checked. Two things there are worth knowing
before writing any qc-to-chroma glue: qc solves `num_rhs` right-hand sides per call while
chroma's `SystemSolver` interface takes one, which is why `QC_PROPAGATOR` is a
measurement; and qc decomposes z and t only while QDP++ `parscalar` splits all four, so
the two agree at one rank and must be checked past it.

## Layout

| file | what |
|---|---|
| `qc.h` / `qc.c` | the public interface: field layout, precision conversion, the handle |
| `qc_mpi.h` | the one public declaration that needs `<mpi.h>`: `qc_set_comm` |
| `qc.syms` | the export list; everything else is localized out of `libqc.{a,so}` |
| `gen_qc_config.py` | emits `qc_config.h`, recording the geometry a build is valid for |
| `driver.c` | the solve itself -- gauge and clover preparation, hierarchy, outer FGMRES |
| `example/` | a worked program that links against `qc.h` and nothing else, plus `python/` (`ctypes` bindings and the same example in Python) and `chroma/` (QDP++ and chroma bridges, built by hand) |
| `clover_ferm.c` | the Wilson-clover operator, even-odd split, clover inverse |
| `solver.c` | Krylov solvers, SAP, the outer FGMRES |
| `coarse.c` | coarse-grid construction, coarse Dslash, transfers, the cycle driver |
| `coarse_sap.c` | the coarse-level SAP smoother and its dense block-inverse cache |
| `bootstrap.c` | adaptive setup: null vectors, block orthogonalization |
| `comm.c` | MPI process grid and halo exchange (the only MPI translation unit) |
| `mixed_bridge.c` | the fp64/fp32 seam and the multigrid hierarchy driver |
| `avx/` | AVX2 / AVX-512 kernels: cpp/GAS templates (`.S`) and the generators that emit the rest |
| `test/` | validation harnesses and the correlator test, plus `compare-halo-dumps.py` and the `adopt_mpi` MPI-ownership gate |

Translation units that exist in both precisions are compiled twice from one source, with symbol
names mangled by `mangle.h` into `_d` / `_f` twins; `MG_SINGLE` selects which.

## Citing

If you use qc in your work, please cite

A. V. Grebe, "Toward Compute-Bound Lattice-QCD Software", PoS(LATTICE2026)112,
arXiv:2610.35526.

`CITATION.cff` carries the same reference for GitHub's "Cite this repository" button.

## License

Copyright 2026, University of Maryland, All Rights Reserved

qc, a multigrid solver for the Wilson-clover Dirac operator, is free software: you may
redistribute it and/or modify it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License, reproduced in `LICENSE`, for more details.  Parties interested in this
software under alternative terms may contact the University of Maryland Tech Transfer Office,
UM Ventures, at umdtechtransfer@umd.edu.

qc has built-in dense linear algebra, but the default icx build links Intel MKL instead; an
external CBLAS such as OpenBLAS is the other option. MKL is non-free software; if that is a
concern for redistributing a binary, build with `DENSE_BLAS=none` (the built-in routines) or
`DENSE_BLAS=cblas` (OpenBLAS), both of which avoid it.

### Relation to DDalphaAMG

qc is a reimplementation of the adaptive aggregation-based domain-decomposition multigrid method
of A. Frommer, K. Kahl, S. Krieg, B. Leder and M. Rottmann, SIAM J. Sci. Comput. **36** (2014)
A1581, and of M. Rottmann's doctoral thesis. DDalphaAMG, the authors' own implementation of that
method, was used as a reference throughout qc's development, and is cited in the source wherever
a design here corresponds to one there.

DDalphaAMG is Copyright (C) 2016, Matthias Rottmann, Artur Strebel, Simon Heybrock, Simone
Bacchio, Bjoern Leder, Issaku Kanamori, and is distributed under the GNU General Public License,
version 3 or later. qc is released under those same terms.

## Acknowledgments

qc's multi-right-hand-side structure was informed by QUDA -- in particular its handling of
per-right-hand-side scalars in Krylov solvers and its stopping criteria for multi-RHS solves --
and several conventions here were chosen to match it. The source notes these where they occur.
QUDA is Copyright (c) 2009-2022, QUDA Developers, and is distributed under the MIT license.

The examples in `example/chroma/` build against chroma and QDP++, Copyright (c) Jefferson
Science Associates, LLC, under a three-clause BSD license. Chroma's notice and conditions are
reproduced in `example/chroma/THIRD-PARTY-NOTICES`.

This work was supported by the National Science Foundation under award PHY-2402275.

## AI Disclosure

Writing a more performant lattice-QCD software package has long been a dream of mine,
and some of the original versions of the fine Dslash kernels were handwritten by me,
but progress was slow prior to the recent advances in large language models in early 2026.
As a result, the great majority of this codebase was written by Anthropic's Claude Code.
I have retained the overall strategic vision of the project and most of the high-level
design decisions, and both Claude and I have contributed to algorithmic design.
I have manually reviewed the codebase prior to each commit and the final release,
and Claude and I have both validated the results against three independent codebases:
Chroma, QUDA, and DDalphaAMG.
