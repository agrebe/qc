// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

// qc_mpi.h -- the MPI-aware corner of qc's public interface.
//
// Kept separate from qc.h on purpose.  qc.h includes only <stddef.h> so that a
// serial consumer needs no MPI headers and no MPI build flags to use qc;
// pulling <mpi.h> in there would impose the dependency on everyone.  Include
// this header only if you need to hand qc a communicator.
//
// Available in MPI builds only.  A serial build (MPI=0) contains no MPI at
// all, so qc_set_comm does not exist in it and a program that calls it fails
// to link.  That is the honest outcome rather than a stub: there is no
// communicator to set.
//
// On ownership -- qc adopts MPI, it does not own it:
//   * qc_init initializes MPI only if MPI_Initialized says nobody has.  If you
//     called MPI_Init yourself (mpi4py does so on import), qc leaves it alone.
//   * qc_finalize calls MPI_Finalize only if qc_init was what initialized MPI.
//     Your own MPI_Init stays paired with your own MPI_Finalize.
//   * Whoever initializes must ask for at least MPI_THREAD_FUNNELED.  qc's
//     exchanges run on the master thread outside OpenMP parallel regions, and
//     qc aborts at startup rather than run below that level.
//   * Call qc_finalize before your MPI_Finalize.  If MPI is already finalized,
//     qc_finalize does nothing rather than crash.
//
// One limitation to know about: a fatal qc configuration error calls MPI_Abort
// on MPI_COMM_WORLD, not on qc's communicator, so it takes the whole job down.
// Aborting the sub-communicator alone would leave the rest of a host
// application's ranks in a collective with dead peers.

#ifndef QC_MPI_H
#define QC_MPI_H

#include <mpi.h>
#include "qc.h"

#ifdef __cplusplus
extern "C" {
#endif

// Hand qc the communicator to run on.  Must be called before qc_init -- the
// rank grid, the neighbor topology and the node-footprint census are all built
// from it at init, so setting it afterwards is a fatal error rather than a
// no-op.  Without this call qc runs on MPI_COMM_WORLD.
//
// qc duplicates what it is given (MPI_Comm_dup) and communicates on the copy,
// so qc's fixed message tags cannot collide with yours on the same
// communicator, and freeing your communicator after qc_init does not disturb
// qc.  The duplicate is released by qc_finalize.
//
// The communicator's size must equal the process grid the library was built for
// (MPI_GRID_PZ * MPI_GRID_PT); qc_init aborts with a message if it does not.
void qc_set_comm(MPI_Comm c);

#ifdef __cplusplus
}
#endif

#endif // QC_MPI_H
