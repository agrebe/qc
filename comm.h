// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

// MPI process-grid / communication layer.
//
// comm.c is compiled exactly once (it is not a precision twin), and every name
// here is precision-neutral (int/long/double/float only, none mangled by
// mangle.h), so this header is safe to include from any TU, twinned or not.
//
// MPI=0 (QC_MPI unset) is a single-rank identity: init/finalize are no-ops,
// reductions leave their buffers untouched, the grid is 1x1x1x1 with
// self-neighbors, and results are bit-identical to a comm-free build.

#ifndef COMM_H
#define COMM_H

#include <stdint.h>   // uint32_t, for the checksum reduction below

typedef struct {
  int nranks, rank;
  int P[4];        // ranks per dimension {x,y,z,t}.  x and y are always 1:
                   // decomposing only the two slowest indices keeps every
                   // face a regular strided slab and keeps whole x-lines
                   // (and their Lxh even/odd packing) on one rank.
  int coords[4];   // this rank's grid coordinates ({0,0,0,0} at 1 rank)
  int nbr_lo[4];   // rank owning the -dim face neighbor (periodic; self when
  int nbr_hi[4];   //   P[dim] == 1) and likewise for the +dim face
  int loc[4];      // local fine-lattice extents {Lx, Ly, Lz/Pz, Lt/Pt}
  int origin[4];   // global fine coordinate of the local origin
} qc_grid_t;

// The one grid instance, filled by comm_init (and valid -- as the identity
// grid -- even before init in the MPI=0 build, so early driver code may read it).
extern qc_grid_t qc_grid;

// Call first thing in main / last thing before return.  Both are idempotent
// and harmless no-ops at MPI=0, so drivers can call them unconditionally.
// init verifies nranks == product(P), every decomposed extent's divisibility,
// and that rank faces land on SAP-block boundaries; it aborts with a message
// on any violation.
//
// MPI ownership is adoptive, not exclusive: comm_init runs MPI_Init_thread only
// if MPI_Initialized reports nobody has, and comm_finalize runs MPI_Finalize only
// if comm_init was what initialized it.  A caller that starts MPI itself keeps
// the responsibility for stopping it, and must ask for at least
// MPI_THREAD_FUNNELED -- comm_init queries the level and dies below it, because
// every exchange here is master-thread-only by construction.
//
// The communicator is whatever qc_set_comm (qc_mpi.h) was given before init,
// defaulting to MPI_COMM_WORLD, and comm_init communicates on a private
// duplicate of it.  Nothing outside comm.c names an MPI type, which is why this
// header needs no <mpi.h> and stays includable from every TU.
void comm_init(int * argc, char *** argv);
void comm_finalize(void);

int  comm_rank(void);
int  comm_size(void);

// Report a fatal error and take the job down.  The single fatal path: every
// abort in the library routes here, so the message reaches stderr (unbuffered,
// unlike a redirected stdout) and carries the rank that raised it.  A uniform
// error -- a geometry or parameter mismatch -- is hit by every rank and so
// prints once per rank; a rank-local one, such as a read error on the slab a
// rank owns, prints only from the rank that saw it, which print0 could not do.
// Appends the newline; call sites pass a message without one.  Under MPI this
// aborts MPI_COMM_WORLD rather than qc's own communicator -- see the comment at
// the definition, and the note in qc.h about what that costs a host program.
_Noreturn void comm_die(const char * fmt, ...);

// Note on placement: comm_init takes a census of the ranks sharing each
// shared-memory node, verifies each node holds one contiguous world-rank block
// (uniform size), and lays those blocks out as gz x gt rectangles of the
// (Pz, Pt) rank grid chosen to minimize inter-node face bytes -- a dimension
// whose ring closes on the node (gz == Pz) crosses no node boundary at all,
// wrap links included.  All internal; see the footprint map in comm.c.

// In-place sum-allreduce over all ranks; identity at 1 rank / MPI=0.
// n is in elements (chunked internally below MPI's int count limit).
void allreduce_sum_d(double * buf, long n);
void allreduce_sum_f(float * buf, long n);
// In-place min-allreduce (double); identity at 1 rank / MPI=0.  Used for the
// global setup diagnostics (per-aggregate CholQR min diag ratio).
void allreduce_min_d(double * buf, long n);
// In-place bitwise-xor allreduce; identity at 1 rank / MPI=0.  Used by
// read_gauge, where the ranks hold a disjoint cover of the lattice and the
// SciDAC checksum folds sites together with xor, so per-rank partial sums
// combine into the whole-file value regardless of who read what.
void allreduce_xor_u32(uint32_t * buf, long n);

// Per-TU alias for REAL-typed reduction buffers in the twinned TUs, keyed on
// MG_SINGLE exactly like precision.h (an MG_SINGLE TU's REAL is float).  This
// is a header-side convenience only -- comm.c itself stays precision-neutral
// and is still compiled once.
#if MG_SINGLE
  #define allreduce_sum_real allreduce_sum_f
#else
  #define allreduce_sum_real allreduce_sum_d
#endif

// --- fine-lattice halo exchange ---------------------------------------------
// Byte-level (precision-neutral: pass site_bytes = sites' REAL count * esz), so
// the twinned TUs share one implementation.  Both are complete no-ops when no
// dimension is decomposed (FERM_GHOSTS == 0) or at MPI=0.  Blocking
// exchanges, called on the master thread outside parallel regions (FUNNELED).
//
// halo_ferm_half: exchange the z/t faces of one parity-half fermion block
// (layout [t][z][y][xh], site_bytes per site) into its ghost block `ghost`
// (layout [z-lo | z-hi | t-lo | t-hi], faces of undecomposed dims absent --
// the G_OFF_* / GHOST_SITES geometry in global_params.h).
void halo_ferm_half(const void * half, void * ghost, long site_bytes);
//
// halo_gauge: fill the gauge array's tail ghosts (lo faces only -- the
// backward links), appended at full-lattice site index V (GG_OFF_* layout).
// site_bytes = one full gauge site (4 links), i.e. 4*Nc*Nc*2 * esz.  Call once
// per prepared gauge array, after the APBC/kappa folding, so ghosts carry the
// boundary phases.
void halo_gauge(void * U, long site_bytes);

// --- coarse-lattice halo exchange -------------------------------------------
// Exchange the z/t faces of one full (lexicographic, non-eo) coarse field into
// its ghost block `ghost`.  Geometry is passed explicitly (local extents
// lx,ly,lz,lt; site_bytes per coarse site) so the same routine serves every
// coarse depth -- the decomposition is read at runtime from qc_grid.P.  Field
// layout is lex [t][z][y][x]; ghost layout [z-lo | z-hi | t-lo | t-hi] with an
// undecomposed dim's faces absent (caller's CG_OFF_* must match: z faces
// present iff P[2]>1, then t faces follow).  No-op when neither z nor t is
// decomposed or at MPI=0.  Blocking, master-thread/FUNNELED, like the fine
// halo.
void halo_coarse(const void * field, void * ghost, long site_bytes,
                    int lx, int ly, int lz, int lt);

// --- nonblocking split for comm/compute overlap -----------------------------
// `_start` packs the local faces and posts the Irecv/Isend pairs (filling
// `ghost` asynchronously), then returns; the caller computes rank-interior
// sites while the exchange is in flight.  `_wait` blocks until the ghosts
// have landed.  Same data, same ghost layout, same tags as halo_coarse --
// only the blocking is split.  Only one coarse exchange may be in flight at a
// time (file-static request state), which is all apply_coarse_schur needs
// (its two exchanges are strictly sequential).  Both are no-ops at MPI=0 / no
// decomposed dim, so the caller's overlap structure stays byte-identical
// there.  `field`/`ghost`/packbuf must not be mutated between _start and _wait.
void halo_coarse_start(const void * field, void * ghost, long site_bytes,
                          int lx, int ly, int lz, int lt);
void halo_coarse_wait(void);

// --- SAP compact fermion halo -----------------------------------------------
// Exchange of pre-packed per-face byte blobs for the fine-SAP compact halo
// (color-halved + spin-projected payloads built by wilson_halo_sap).  Buffers
// are indexed [z-lo | z-hi | t-lo | t-hi]; an undecomposed dim's entries are
// ignored.  Unlike every other exchange here, the send and recv byte counts of
// a direction differ (the live-row sets on the two sides of a cut are the two
// colors' complements), so both are passed per face; a face's send count on
// this rank always equals its recv count on the matching neighbor (the live
// rows are the same physical block rows, enumerated in the same face order).
// Direction semantics match halo_ferm_half: my z-hi face lands as the +z
// neighbor's z-lo ghost.  Blocking, master-thread / FUNNELED, tag block
// 140-143.  No-op at MPI=0 / no decomposed dim.
void halo_ferm_sap(const void * send[4], void * recv[4],
                   const long sbytes[4], const long rbytes[4]);

// Extended-gauge halo for compute_Fmunu (clover field strength): fills the
// one-deep z/t face ghost layers and the (z,t) corner of a gauge array padded
// by one site in each decomposed dim.  Layout: Uext[te][ze][y][x][4 links]
// with te in [0,Lt+2*DECOMP_T), ze in [0,Lz+2*DECOMP_Z); interior local (t,z)
// at (t+DECOMP_T, z+DECOMP_Z).  site_bytes = one gauge site (4*Nc*Nc*2 * esz).
// Corners come for free from exchanging z first (interior t range) then the
// whole z-extended t-slabs.  No-op when no dim is decomposed or at MPI=0.
void halo_gauge_ext(void * Uext, long site_bytes);

// printf that emits on rank 0 only (every rank is rank 0 at MPI=0).
void print0(const char * fmt, ...);

#endif // COMM_H
