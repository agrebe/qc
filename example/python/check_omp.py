#!/usr/bin/env python3
# Copyright 2026, University of Maryland, All Rights Reserved
# SPDX-License-Identifier: GPL-3.0-or-later

"""Report whether qc and numpy would share one OpenMP runtime in one process.

Two OpenMP runtimes in a single process is a real failure, not a tidiness
concern: each one sizes its own thread pool from the core count, so the process
runs N*N threads and every parallel region thrashes.  Intel's runtime sometimes
notices and aborts with "OMP: Error #15" instead, which is the friendlier
outcome -- silent oversubscription just makes the solve slow for no visible
reason.

The pairing that hurts is libgomp (GCC's) alongside libiomp5 (Intel's).  qc
built with icx brings libiomp5.  What numpy brings depends on how it was
installed: pip's scipy-openblas is built for pthreads and brings neither, while
a conda MKL-backed numpy brings libiomp5 -- the same runtime qc uses, so that
combination is fine too.  A numpy whose OpenBLAS was built with GCC and OpenMP
threading is the one to watch for.

Run this on any machine before trusting a qc Python job there:

    python3 example/python/check_omp.py [path/to/libqc.so]

Loading order matters to the diagnosis, so this mirrors what the qc module
does: import numpy, load qc, then do arithmetic.  Reading /proc/self/maps is
Linux-only; on anything else the runtime list comes out empty and only the
numpy backend and the environment are reported.
"""
import ctypes
import os
import re
import sys


def mapped_libraries():
    """Shared objects currently mapped into this process, by basename."""
    try:
        with open("/proc/self/maps") as f:
            maps = f.read()
    except OSError:
        return None                       # not Linux; caller degrades the report
    return sorted({os.path.basename(p)
                   for p in re.findall(r"/\S+\.so[.\w-]*", maps)})


def main():
    # Relative to this file, not to the working directory: the point of the
    # script is to be runnable anywhere, including from a job's scratch dir.
    # Note dlopen searches the library path rather than the current directory,
    # so a bare "libqc.so" would fail even sitting next to one.
    here = os.path.dirname(os.path.abspath(__file__))
    so = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "..",
                                                            "libqc.so")

    print("python  ", sys.version.split()[0])

    try:
        import numpy
    except ImportError:
        print("numpy    NOT INSTALLED -- nothing to conflict with, but the qc "
              "module needs it")
        return 2
    print("numpy   ", numpy.__version__)

    # Where numpy's BLAS came from is the single best predictor of which OpenMP
    # runtime it will pull in.
    try:
        blas = numpy.__config__.show(mode="dicts")["Build Dependencies"]["blas"]
        print("numpy BLAS", blas.get("name", "?"), blas.get("version", ""))
    except Exception:
        print("numpy BLAS unknown (older numpy: try `python3 -m threadpoolctl -i numpy`)")

    # Load qc the way the Python module will.  A failure here is the Intel
    # runtime not being on the library path, which is a different problem --
    # source the compiler environment first.
    loaded = False
    try:
        ctypes.CDLL(so)
        loaded = True
        print("qc      ", so, "loaded")
    except OSError as e:
        print("qc      ", so, "NOT loaded:", e)
        print("         (if this names libimf/libsvml/libmkl, source the oneAPI "
              "environment first -- that is a library-path problem, not an "
              "OpenMP one)")

    # numpy's BLAS is loaded lazily, so it will not appear in the map list until
    # something actually calls it.  Importing numpy is not enough.
    a = numpy.random.rand(200, 200)
    a @ a

    libs = mapped_libraries()
    if libs is None:
        print("\n/proc/self/maps unavailable (not Linux) -- cannot list runtimes")
        return 0

    omp = [l for l in libs if re.search(r"(lib(i|g)omp|libomp)", l)]
    print("\nOpenMP runtimes mapped:", omp if omp else "none")
    print("BLAS mapped:            ",
          [l for l in libs if "blas" in l.lower() or "mkl" in l.lower()] or "none")

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "KMP_AFFINITY", "OMP_PROC_BIND"):
        if os.environ.get(var):
            print(f"  {var}={os.environ[var]}")

    # The verdict.  Distinct runtimes is the thing to act on; one or none is fine
    # regardless of which one it is.
    families = {re.sub(r"[-.]\d.*", "", l) for l in omp}
    print()
    if not loaded:
        print("INCONCLUSIVE -- qc did not load, so only numpy's side was sampled.")
        return 2
    if len(families) > 1:
        print("PROBLEM:", len(families), "distinct OpenMP runtimes in one process:",
              sorted(families))
        print("Fix by making numpy and qc agree: use a pip scipy-openblas numpy "
              "(pthreads, brings none), or build qc with the same compiler family "
              "as numpy's BLAS.  Setting OMP_NUM_THREADS does not fix it -- both "
              "runtimes read it and each spawns that many threads.")
        return 1
    print("OK -- at most one OpenMP runtime", f"({sorted(families)[0]})" if families else "",
          "\nqc and numpy can share this process.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
