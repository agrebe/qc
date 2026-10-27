#!/usr/bin/env python3
# Copyright 2026, University of Maryland, All Rights Reserved
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare `qc halo` / `qc cbuild2` dumps across process grids.

The harnesses apply each operator to fields synthesized from GLOBAL coordinates
and hash every output site, so a physical site must carry the same hash no
matter how the lattice was decomposed.  This script asserts exactly that: the
1x1 run is the full-lattice reference, and every decomposed run must reproduce
it site for site.

Each dump entry is (global site key, hash), so a rank's contribution is located
by key rather than by position.  That is what makes this work for a z cut, where
a rank's sites are strided through the reference rather than a contiguous slab.

The hashes are bitwise, so anything that perturbs floating-point codegen
(-fsanitize=..., a different -O level or -march) changes them and reports every
site as differing, which looks exactly like a halo bug.  MPI=0 vs MPI=1 is not
such a difference and is the intended usage below: that flag only reaches
comm.c, which does nothing on a 1x1 grid.

Start from an empty dump directory.  Every file found there is merged, so dumps
left from an earlier campaign are compared too -- against a reference that may
since have been rebuilt.

Usage:
    rm -rf /tmp/qc-halo
    # 1. build with MPI_GRID_PZ = MPI_GRID_PT = 1, then
    ./qc halo && ./qc cbuild2
    # 2. edit the grid in global_params.h, rebuild, then
    mpirun -n <N> ./qc halo && mpirun -n <N> ./qc cbuild2
    # 3.
    python3 test/compare-halo-dumps.py [dump-dir]

Exit status is 0 only if every operator matches on every decomposed grid.
"""
import os
import re
import struct
import sys
from collections import defaultdict

MAGIC = 0x314F4C41484351  # "QCHALO1"
NAME = re.compile(r"^(?P<op>\w+)\.P(?P<pz>\d+)x(?P<pt>\d+)\.r(?P<rank>\d+)\.bin$")


def read_dump(path):
    """-> {global key: hash} for one rank's dump of one operator."""
    with open(path, "rb") as fh:
        blob = fh.read()
    if len(blob) < 16:
        raise ValueError(f"{path}: truncated (no header)")
    magic, nsites = struct.unpack_from("<QQ", blob, 0)
    if magic != MAGIC:
        raise ValueError(f"{path}: bad magic {magic:#x} -- stale dump? "
                         "delete the dump dir and re-run both grids")
    if len(blob) != 16 + 16 * nsites:
        raise ValueError(f"{path}: size {len(blob)} != header's {nsites} sites")
    out = {}
    for key, val in struct.iter_unpack("<QQ", blob[16:]):
        if key in out:
            raise ValueError(f"{path}: duplicate site key {key}")
        out[key] = val
    return out


def collect(dump_dir):
    """-> {(op, grid): {key: hash}}, merging all ranks of a grid."""
    merged = defaultdict(dict)
    for fn in sorted(os.listdir(dump_dir)):
        m = NAME.match(fn)
        if not m:
            continue
        op = m.group("op")
        grid = f"{m.group('pz')}x{m.group('pt')}"
        part = read_dump(os.path.join(dump_dir, fn))
        target = merged[(op, grid)]
        for key, val in part.items():
            if key in target:
                raise ValueError(f"{op} P{grid}: site {key} dumped by two ranks")
            target[key] = val
    return merged


def main():
    dump_dir = sys.argv[1] if len(sys.argv) > 1 else \
        os.environ.get("QC_HALO_DIR", "/tmp/qc-halo")
    if not os.path.isdir(dump_dir):
        sys.exit(f"no dump directory {dump_dir} -- run ./qc halo first")

    merged = collect(dump_dir)
    if not merged:
        sys.exit(f"{dump_dir} holds no dumps")

    ops = sorted({op for op, _ in merged})
    grids = sorted({g for _, g in merged if g != "1x1"})
    if not grids:
        sys.exit("only a 1x1 reference is present -- rebuild with a decomposed "
                 "grid and re-run to have something to compare against")

    failures = skipped = checked = 0
    for grid in grids:
        print(f"=== P={grid} vs the 1x1 reference ===")
        for op in ops:
            ref = merged.get((op, "1x1"))
            dec = merged.get((op, grid))
            if dec is None:
                continue
            if ref is None:
                print(f"  {op:8s} SKIP  (no 1x1 reference dump)")
                skipped += 1
                continue
            checked += 1
            if ref.keys() != dec.keys():
                only_ref, only_dec = ref.keys() - dec.keys(), dec.keys() - ref.keys()
                print(f"  {op:8s} FAIL  site coverage differs "
                      f"({len(only_ref)} only in reference, "
                      f"{len(only_dec)} only in decomposed)")
                failures += 1
                continue
            bad = [k for k in ref if ref[k] != dec[k]]
            if bad:
                print(f"  {op:8s} FAIL  {len(bad)} / {len(ref)} sites differ "
                      f"(first keys: {sorted(bad)[:5]})")
                failures += 1
            else:
                print(f"  {op:8s} ok    {len(ref)} sites bit-identical")

    print(f"\n{checked} operator/grid pairs checked, {failures} failed"
          + (f", {skipped} skipped" if skipped else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
