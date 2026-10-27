#!/usr/bin/env python3
# Copyright 2026, University of Maryland, All Rights Reserved
# SPDX-License-Identifier: GPL-3.0-or-later

"""Zero-momentum pion correlator from point sources -- the Python analog of
example/pion.c, and the gate that says the bindings work.

Its table must match `./qc correlator` on every timeslice.  Run it as

    source initialize.sh                  # qc's compiler runtime must be loadable
    python3 example/python/pion.py [gauge-file]

Like the C example it deliberately uses both ways across the boundary, because a
real calculation wants both: the twelve point sources are written straight into
the solver field through site_index, which for a point source is a handful of
stores rather than a full-volume array, and the solution comes back through
export_lane in the conventional lexicographic order the correlator wants.
"""

import os
import sys

import numpy as np

# Imported before qc so that MPI is started by whoever owns it.  mpi4py calls
# MPI_Init on import; qc.init() then finds MPI already running and adopts it
# rather than starting a second one, and qc.finalize() leaves it alone.  That is
# the arrangement an HPC application is in, so it is the one worth exercising.
# Without mpi4py the example still runs -- qc starts MPI itself -- but Python has
# no communicator to reduce the per-rank correlators with, so each rank can only
# report its own timeslices.
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import qc


def main():
    conf = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..",
        "test/sample_config.lime")

    # Before anything is allocated.  A solver field crosses into qc as an untyped
    # pointer, so a lattice or precision this program did not expect would be
    # read as though it were right.
    qc.require(dims=(16, 16, 16, 32), num_rhs=16)
    qc.init()

    kappa, csw = 0.134, 1.87567
    gd, ld, og = qc.global_dims(), qc.local_dims(), qc.local_origin()
    Nc, Ns, nrhs = qc.num_colors(), qc.num_spins(), qc.num_rhs()
    nsc = Ns * Nc                       # 12 spin-colors per point source

    if qc.rank() == 0:
        print(f"qc pion example: {gd[0]}x{gd[1]}x{gd[2]}x{gd[3]} lattice, "
              f"{nrhs} rhs per solve, "
              f"{'double' if qc.precision() == qc.DOUBLE else 'single'} fields")
        if qc.num_ranks() > 1:
            print(f"qc pion example: {qc.num_ranks()} ranks -- the timeslice sums "
                  f"below are this rank's slab only; combining them is the "
                  f"caller's job")

    # --- gauge configuration -------------------------------------------------
    U = qc.gauge_field()
    qc.read_gauge_file(U, conf)

    # Defaults, then the physics.  Everything except kappa and csw is a solver
    # knob you can leave alone until you have a reason not to.
    h = qc.setup(U, kappa=kappa, csw=csw)

    # --- sources, written directly in qc's own order -------------------------
    # Every lane carries a real source, which is what a production batch looks
    # like: the twelve spin-colors of a point source at the origin, then as many
    # more positions as the remaining lanes hold.  Only position 0 is projected
    # below.
    src_pos = [
        (0,            0,            0,            0),
        (gd[0] // 2,   gd[1] // 2,   gd[2] // 2,   gd[3] // 2),
        (gd[0] // 4,   3 * gd[1] // 4, gd[2] // 4, 3 * gd[3] // 4),
        (3 * gd[0] // 4, gd[1] // 4, 3 * gd[2] // 4, gd[3] // 4),
    ]

    b, x = qc.field(), qc.field()
    for lane in range(nrhs):
        p, sc = divmod(lane, nsc)
        if p >= len(src_pos):
            break                       # past the table: lane stays zero
        s, c = divmod(sc, Nc)
        gx, gy, gz, gt = src_pos[p]
        # Coordinates in the table are global; a point source lives on one rank.
        # Placing it takes local coordinates, so subtract this rank's origin.
        if not (og[2] <= gz < og[2] + ld[2]):
            continue
        if not (og[3] <= gt < og[3] + ld[3]):
            continue
        b[qc.site_index(gx, gy, gz - og[2], gt - og[3]), s, c, 0, lane] = 1.0

    # --- solve ---------------------------------------------------------------
    iters = qc.solve(h, x, b, tol=1e-10, max_iter=100, chroma_norm=True)
    if qc.rank() == 0:
        print(f"qc pion example: {iters} outer iterations")
    # Where that solve's time went, per kernel -- the same report example/pion.c
    # prints.  Drop it for quieter output; it is here because it is what to read
    # when tuning the multigrid parameters.
    qc.report_profile(h)

    # --- correlator, from the solution in lexicographic order ----------------
    # C(t) = sum over this timeslice, over the twelve lanes of the position-0
    # source, of |prop|^2.  The other lanes belong to other source positions and
    # are simply not read.
    corr = np.zeros(gd[3])
    for lane in range(min(nsc, nrhs)):
        v = qc.export_lane(x, lane)              # (Lt, Lz, Ly, Lx, Ns, Nc)
        # re*re + im*im, not abs(v)**2: the latter takes a square root and
        # squares it back, which is a rounding step the C version does not have.
        mag2 = v.real ** 2 + v.imag ** 2
        corr[og[3]:og[3] + ld[3]] += mag2.sum(axis=(1, 2, 3, 4, 5))

    # Each rank summed only its own timeslices, so the global correlator is a sum
    # over ranks.  qc publishes no reduction on purpose -- the caller owns MPI,
    # and qc solving on a communicator does not entitle it to collectives on the
    # caller's behalf.
    whole = True
    if qc.num_ranks() > 1:
        if MPI is not None:
            MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, corr, op=MPI.SUM)
        else:
            whole = False

    if not whole:
        # Nothing to reduce with.  Report this rank's own timeslices, labeled,
        # rather than printing a partial sum that looks like the whole thing.
        for t in range(og[3], og[3] + ld[3]):
            print(f"rank {qc.rank()}  {t:2d}   {corr[t]:15.6e}")
    elif qc.rank() == 0:
        print("\n   t         qc C[t]")
        for t in range(gd[3]):
            print(f"  {t:2d}   {corr[t]:15.6e}")

    qc.teardown(h)
    qc.finalize()


if __name__ == "__main__":
    main()
