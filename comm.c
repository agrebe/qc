// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

// MPI process-grid / communication layer -- see comm.h for the contract.
// Single TU for both precisions: nothing here touches REAL.

#include "comm.h"

#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Grid shape (MPI_GRID_PZ/PT) and lattice extents.  This TU sets no
// MG_SINGLE, so the precision/mangle tail of the header resolves to the
// (unused) double defaults.
#include "global_params.h"

#if QC_MPI
#include <mpi.h>
#include "qc_mpi.h"  // declares qc_set_comm, defined at the bottom of this file
#endif

// Identity grid before init so early driver reads are safe at MPI=0.
qc_grid_t qc_grid = {
  .nranks = 1, .rank = 0,
  .P      = {1, 1, 1, 1},
  .coords = {0, 0, 0, 0},
  .nbr_lo = {0, 0, 0, 0},
  .nbr_hi = {0, 0, 0, 0},
  .loc    = {Lx, Ly, Lz, Lt},
  .origin = {0, 0, 0, 0},
};

static int g_comm_initialized = 0;

#if QC_MPI
// --- MPI ownership ----------------------------------------------------------
// qc adopts MPI rather than owning it.  comm_init initializes MPI only if
// nobody else has, and comm_finalize finalizes only if comm_init was the one
// that did: whoever starts MPI is responsible for stopping it.  A caller that
// runs its own MPI_Init (mpi4py does this on import) therefore needs no
// cooperation from qc beyond calling comm_init after it.
static int g_we_init_mpi = 0;

// The communicator qc was handed (qc_set_comm), and qc's private duplicate of
// it.  Everything below sends on the duplicate: qc uses fixed tags (102-105
// fine halo, 112-115 coarse, 140-143 SAP), so sharing a communicator with a
// caller that happens to use one of those tags would let its messages satisfy
// qc's receives.  MPI_Comm_dup buys a private tag space for the cost of one
// collective at init.  It also makes qc's handle independent of the caller's,
// so a caller freeing its own communicator after qc_init cannot invalidate
// ours.
static MPI_Comm g_user_comm = MPI_COMM_WORLD;
static MPI_Comm g_comm      = MPI_COMM_NULL;
#endif

// Abort with a message from every rank that hits the error; see comm.h for
// which errors that is one rank and which it is all of them.
_Noreturn void comm_die(const char * fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  // Tagged "qc", not "comm": every fatal path in the library ends here, not
  // just the communicator's own.  Flush stdout first so the progress printed
  // before the failure is not reordered after this message.
  fflush(stdout);
  fprintf(stderr, "[qc rank %d] ", qc_grid.rank);
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
#if QC_MPI
  // Deliberately the world, not qc's communicator.  comm_die fires on
  // unrecoverable configuration errors, and can fire before g_comm exists;
  // aborting only qc's sub-communicator would leave the rest of a host
  // application's ranks hanging in a collective with dead peers.  The cost is
  // that a fatal qc error takes down the whole job -- documented in qc.h.
  MPI_Abort(MPI_COMM_WORLD, 1);
#endif
  exit(1);
}

#if QC_MPI
// --- node census (processor names) ------------------------------------------
// Computed once, collectively.  Not via MPI_COMM_TYPE_SHARED: some stacks hand
// back singleton shared comms for co-located ranks, which would report one
// rank per node and silently defeat the footprint map below.  Processor names
// are ground truth each process reports about itself.
// The footprint map requires every node's ranks to be one contiguous
// world-rank block of uniform size, so that is verified here: a cyclic
// launcher distribution must die loudly rather than silently scatter the node
// footprint (and with it multiply the inter-node face traffic).
static int g_node_size = -1;  // also the "census done" sentinel (< 0 = not yet)

static void node_info(void) {
  if (g_node_size >= 0) return;
  char name[MPI_MAX_PROCESSOR_NAME] = {0};
  int len = 0;
  MPI_Get_processor_name(name, &len);
  char * all = malloc((size_t) qc_grid.nranks * MPI_MAX_PROCESSOR_NAME);
  MPI_Allgather(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                all, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, g_comm);
  // runs of consecutive equal names = the nodes' rank blocks
  int nrun = 0;
  int * start = malloc(sizeof(int) * (size_t) qc_grid.nranks);
  for (int r = 0; r < qc_grid.nranks; r ++)
    if (!r || strcmp(all + (size_t) r       * MPI_MAX_PROCESSOR_NAME,
                     all + (size_t) (r - 1) * MPI_MAX_PROCESSOR_NAME))
      start[nrun ++] = r;
  g_node_size = (nrun > 1 ? start[1] : qc_grid.nranks);
  for (int i = 0; i < nrun; i ++) {
    const int leni = (i + 1 < nrun ? start[i + 1] : qc_grid.nranks) - start[i];
    if (leni != g_node_size)
      comm_die("non-uniform ranks per node (%d vs %d): the node-footprint map "
               "needs the same rank count on every node", leni, g_node_size);
    for (int j = 0; j < i; j ++)
      if (!strcmp(all + (size_t) start[i] * MPI_MAX_PROCESSOR_NAME,
                  all + (size_t) start[j] * MPI_MAX_PROCESSOR_NAME))
        comm_die("node '%s' holds a non-contiguous world-rank set: launch with "
                 "a block distribution (consecutive ranks per node)",
                 all + (size_t) start[i] * MPI_MAX_PROCESSOR_NAME);
  }
  free(start);
  free(all);
}
#endif

// --- checked allocation -----------------------------------------------------
// The bodies behind the malloc / calloc macros in global_params.h, which is
// where the contract is written down.  They live here because a failure ends
// in comm_die, and because the message wants the node census above.

#if defined(__linux__)
#include <sys/resource.h>   // getrlimit, in the failure message below

// One field of a /proc table ("VmRSS:", "MemAvailable:") in kB, or -1 if the
// file or the key is missing.  Only ever read on the failure path.
static long proc_field_kb(const char * path, const char * key) {
  FILE * f = fopen(path, "r");
  if (!f) return -1;
  const size_t klen = strlen(key);
  char line[256];
  long kb = -1;
  while (fgets(line, sizeof line, f))
    if (!strncmp(line, key, klen)) {
      kb = strtol(line + klen, NULL, 10);
      break;
    }
  fclose(f);
  return kb;
}
#endif

// Report the request that failed, and what the machine had at the time: the two
// together say whether the run needs a bigger node or more ranks, which a bare
// "out of memory" does not.  `bytes` is what the caller asked for, before the
// round-up to the alignment.
static void alloc_fail(size_t bytes, const char * file, int line) {
  // No machine has this, so the request is not a real one: a size computed in
  // an int (or as a difference that went negative) arrives here as a huge
  // size_t.  Say that instead of reporting the machine's memory, which is not
  // the problem.
  if (bytes > (size_t) 1 << 48)
    comm_die("%s:%d asked for %zu bytes -- far past any real memory, so the size "
             "computation at that line overflowed", file, line, bytes);

  char note[448];
  note[0] = '\0';
#if defined(__linux__)
  const long rss   = proc_field_kb("/proc/self/status", "VmRSS:");
  const long avail = proc_field_kb("/proc/meminfo", "MemAvailable:");
  int per_node = 1;
#if QC_MPI
  if (g_node_size > 0) per_node = g_node_size;   // -1 before the census runs
#endif
  if (rss >= 0 && avail >= 0) {
    int n = snprintf(note, sizeof note,
                     "\n  this rank holds %.2f GiB; the node has %.2f GiB available "
                     "for %d rank%s", rss / 1048576.0, avail / 1048576.0,
                     per_node, per_node == 1 ? "" : "s");
    // The node's free memory is what /proc reports, but it is not necessarily
    // what this process may have: a batch system's per-job cgroup
    // (SLURM --mem), a container, or ulimit -v all cap the process while the
    // node still looks empty.  When the request fits in this rank's share of
    // what is free, that gap is the whole story, so name it -- otherwise the
    // message sends the user to buy a bigger node when the fix is a launcher
    // flag.
    if (n > 0 && (size_t) n < sizeof note
        && bytes < (size_t) avail * 1024 / per_node) {
      struct rlimit rl;
      if (!getrlimit(RLIMIT_AS, &rl) && rl.rlim_cur != RLIM_INFINITY)
        snprintf(note + n, sizeof note - n,
                 "\n  the node has room, so the cap is this process's address-space "
                 "rlimit of %.2f GiB", rl.rlim_cur / 1073741824.0);
      else
        snprintf(note + n, sizeof note - n,
                 "\n  the node has room, so the cap is on this process: a cgroup "
                 "(a batch system's --mem), a container, or an rlimit");
    }
  }
#endif
  comm_die("out of memory: %s:%d asked for %.2f GiB (%zu bytes)%s",
           file, line, bytes / 1073741824.0, bytes, note);
}

// Round the request up to the alignment: C11 requires the size be a multiple
// of it (glibc tolerates violations, but ASan aborts on them and other
// allocators may return NULL).  A zero-byte request becomes one block, so
// every caller gets a distinct freeable pointer rather than an
// implementation-defined NULL.
void * alloc_try(size_t bytes) {
  if (bytes > SIZE_MAX - 63) return NULL;     // the round-up would itself wrap
  size_t sz = (bytes + 63) & ~(size_t) 63;
  if (sz == 0) sz = 64;
  return aligned_alloc(64, sz);
}

void * alloc_checked(size_t bytes, const char * file, int line) {
  void * p = alloc_try(bytes);
  if (p == NULL) alloc_fail(bytes, file, line);
  return p;
}

void * calloc_checked(size_t nmemb, size_t size, const char * file, int line) {
  if (size != 0 && nmemb > SIZE_MAX / size)
    comm_die("%s:%d asked for %zu x %zu bytes, which overflows a size_t: the "
             "size computation at that line is wrong", file, line, nmemb, size);
  const size_t bytes = nmemb * size;
  void * p = alloc_checked(bytes, file, line);   // never returns NULL
  memset(p, 0, bytes);
  return p;
}

// --- blocked rank <-> coords map --------------------------------------------
// The (Pz, Pt) rank grid is tiled by gz x gt node blocks; a node's G
// consecutive world ranks form one block (z fastest within a block, blocks
// z-fastest).  comm_init picks (gz, gt) among gz | Pz, gt | Pt, gz*gt = G to
// minimize inter-node bytes per node per exchange (one shared factor dropped):
//     cost = gt * loc_t * (gz < Pz)  +  gz * loc_z * (gt < Pt)
// -- a z-face's bytes scale with the local t extent and vice versa, and a
// dimension whose ring closes on the node (gz == Pz) moves nothing between
// nodes, wrap links included.  The loc_t : loc_z ratio is the same on every MG
// level (each coarsening divides z and t alike, and the bottom's eo face-slot
// counts inherit it), so one footprint is optimal for all levels at once.
// gz = gt = 1 (1 rank total, or 1 rank/node where every placement is
// equivalent) degenerates to a plain z-fastest map.
static int g_gz = 1, g_gt = 1;

static int coords_to_rank(const int c[4]) {
  const int * P = qc_grid.P;
  const int blk   = (c[3] / g_gt) * (P[2] / g_gz) + c[2] / g_gz;
  const int intra = (c[3] % g_gt) * g_gz          + c[2] % g_gz;
  return (c[0] * P[1] + c[1]) * P[2] * P[3] + blk * g_gz * g_gt + intra;
}

static void rank_to_coords(int r, int c[4]) {
  const int * P = qc_grid.P;
  const int zt = r % (P[2] * P[3]);
  r /= P[2] * P[3];
  c[1] = r % P[1];
  c[0] = r / P[1];
  const int blk = zt / (g_gz * g_gt), intra = zt % (g_gz * g_gt);
  c[2] = (blk % (P[2] / g_gz)) * g_gz + intra % g_gz;
  c[3] = (blk / (P[2] / g_gz)) * g_gt + intra / g_gz;
}

void comm_init(int * argc, char *** argv) {
  if (g_comm_initialized) return;
  g_comm_initialized = 1;

#if QC_MPI
  // FUNNELED: OpenMP threads compute, but only the master thread (outside
  // parallel regions) makes MPI calls -- the design point of this layer.
  //
  // Initialize only if nobody else has.  When the caller got there first we
  // cannot choose the thread level, so query what it asked for: every exchange
  // below assumes FUNNELED, and a caller that initialized at MPI_THREAD_SINGLE
  // would otherwise corrupt halos silently rather than fail.  That check is the
  // whole reason this is not just an "if (!already) init" -- the level is the
  // part qc gives up control of by adopting.
  int already = 0;
  MPI_Initialized(&already);
  int provided = MPI_THREAD_SINGLE;
  if (!already) {
    MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
    g_we_init_mpi = 1;
  } else {
    (void) argc; (void) argv;      // the caller consumed them
    MPI_Query_thread(&provided);
  }
  if (provided < MPI_THREAD_FUNNELED)
    comm_die("MPI thread level %d is below MPI_THREAD_FUNNELED (%d)%s",
             provided, MPI_THREAD_FUNNELED,
             already
               ? ": qc adopted an MPI initialized by the caller, so pass at "
                 "least MPI_THREAD_FUNNELED to your MPI_Init_thread"
               : "");
  // Private tag space; see g_comm above.  Collective over g_user_comm, which is
  // fine here -- comm_init is collective already.
  MPI_Comm_dup(g_user_comm, &g_comm);
  MPI_Comm_size(g_comm, &qc_grid.nranks);
  MPI_Comm_rank(g_comm, &qc_grid.rank);
#else
  (void) argc; (void) argv;
#endif

  const int P[4]    = {1, 1, MPI_GRID_PZ, MPI_GRID_PT};
  // global extents (the plain Lz/Lt macros are the per-rank local ones)
  const int gdim[4] = {Lx, Ly, Lz_glob, Lt_glob};

  if (qc_grid.nranks != P[0] * P[1] * P[2] * P[3])
    comm_die("launched with %d ranks but MPI_GRID is 1 x 1 x %d x %d = %d "
             "(edit global_params.h or the mpirun -np)", qc_grid.nranks,
             MPI_GRID_PZ, MPI_GRID_PT, P[0] * P[1] * P[2] * P[3]);

  // The grid divides the global extents, and both block edges divide the
  // resulting local ones: every term is a compile-time constant, so those
  // invariants are _Static_asserted together in coarse.h rather than
  // re-checked here.
  for (int d = 0; d < 4; d ++) {
    qc_grid.P[d] = P[d];
    qc_grid.loc[d] = gdim[d] / P[d];
  }

  // node footprint selection (see the map comment above).  A pure function of
  // the name census + grid + local extents, so every rank picks the same pair.
#if QC_MPI
  if (qc_grid.nranks > 1) {
    node_info();
    const int G = g_node_size;
    long best = -1;
    for (int gz = 1; gz <= G; gz ++) {
      if (G % gz || P[2] % gz || P[3] % (G / gz)) continue;
      const int  gt   = G / gz;
      const long cost = (gz < P[2] ? (long) gt * qc_grid.loc[3] : 0)
                      + (gt < P[3] ? (long) gz * qc_grid.loc[2] : 0);
      if (best < 0 || cost < best) { best = cost; g_gz = gz; g_gt = gt; }
    }
    if (best < 0)
      comm_die("no gz x gt node footprint tiles the %d x %d rank grid at %d "
               "ranks/node (needs gz | Pz, gt | Pt, gz*gt = ranks/node)",
               P[2], P[3], G);
    print0("[comm] node footprint %d x %d (z x t) on the %d x %d rank grid "
           "(%d ranks/node)\n", g_gz, g_gt, P[2], P[3], G);
  }
#endif

  rank_to_coords(qc_grid.rank, qc_grid.coords);

  for (int d = 0; d < 4; d ++) {
    qc_grid.origin[d] = qc_grid.coords[d] * qc_grid.loc[d];
    // neighbor ranks (periodic in the grid): re-encode with one coord shifted
    for (int s = 0; s < 2; s ++) {
      int c[4] = {qc_grid.coords[0], qc_grid.coords[1],
                  qc_grid.coords[2], qc_grid.coords[3]};
      c[d] = (c[d] + (s ? 1 : P[d] - 1)) % P[d];
      int nbr = coords_to_rank(c);
      if (s) qc_grid.nbr_hi[d] = nbr;
      else   qc_grid.nbr_lo[d] = nbr;
    }
  }
}

void comm_finalize(void) {
  if (!g_comm_initialized) return;
  g_comm_initialized = 0;
#if QC_MPI
  // A caller whose teardown order is wrong -- MPI already finalized, qc's
  // teardown still to run -- gets a no-op instead of a crash.  This is not
  // hypothetical for Python: mpi4py registers MPI_Finalize through atexit, and
  // which atexit handler runs first is not something the user controls
  // directly.
  int done = 0;
  MPI_Finalized(&done);
  if (done) return;
  if (g_comm != MPI_COMM_NULL) MPI_Comm_free(&g_comm);   // sets it to NULL
  if (g_we_init_mpi) { MPI_Finalize(); g_we_init_mpi = 0; }
  // Re-init after finalize is a real sequence for a library (a Python session
  // that tears qc down and builds it again), so drop the cached census and
  // footprint: both are properties of the communicator that just went away.
  g_node_size = -1;
  g_gz = g_gt = 1;
#endif
}

#if QC_MPI
// Defined here rather than in qc.c because comm.c is the only TU that includes
// <mpi.h>, so it is the only one that can name MPI_Comm.  qc_grid sets the
// precedent for a published name living in this file.
void qc_set_comm(MPI_Comm c) {
  if (g_comm_initialized)
    comm_die("qc_set_comm called after qc_init -- the communicator has to be set "
             "before qc initializes, since the rank grid is built from it");
  g_user_comm = c;
}
#endif

int comm_rank(void) { return qc_grid.rank; }
int comm_size(void) { return qc_grid.nranks; }

#if QC_MPI
// --- debug: collective/halo divergence tracer (env QC_TRACE=1) --------------
// Every rank prints a per-rank running sequence number + the routine + the
// caller's return address on each collective/halo entry.  Lock-step ranks emit
// identical sequences; a multi-rank deadlock shows up as the two ranks' traces
// parting ways -- the last line of each names the call site they are stuck at
// (map ra= with: addr2line -e qc <addr>).  Zero cost unless QC_TRACE is set.
static int trace_on(void) {
  static int v = -1;
  if (v < 0) { const char * e = getenv("QC_TRACE"); v = (e && *e) ? 1 : 0; }
  return v;
}
static long trace_seq = 0;   // master-thread only (collectives are FUNNELED)
static void trace(const char * what, long n, void * ra) {
  if (!trace_on()) return;
  fprintf(stderr, "QCTRACE r=%d seq=%ld %-14s n=%ld ra=%p\n",
          qc_grid.rank, trace_seq++, what, n, ra);
  fflush(stderr);
}
// MPI counts are int; chunk long element counts well below INT_MAX
#define COMM_CHUNK (1L << 30)
static void allreduce_sum(void * buf, long n, MPI_Datatype t, size_t esz) {
  if (qc_grid.nranks == 1) return;
  for (long off = 0; off < n; off += COMM_CHUNK) {
    long c = n - off < COMM_CHUNK ? n - off : COMM_CHUNK;
    MPI_Allreduce(MPI_IN_PLACE, (char *) buf + off * esz, (int) c, t,
                  MPI_SUM, g_comm);
  }
}
void allreduce_sum_d(double * buf, long n) {
  trace("ar_sum_d", n, __builtin_return_address(0));
  allreduce_sum(buf, n, MPI_DOUBLE, sizeof(double));
}
void allreduce_sum_f(float * buf, long n) {
  trace("ar_sum_f", n, __builtin_return_address(0));
  allreduce_sum(buf, n, MPI_FLOAT, sizeof(float));
}
void allreduce_min_d(double * buf, long n) {
  trace("ar_min_d", n, __builtin_return_address(0));
  if (qc_grid.nranks == 1) return;
  for (long off = 0; off < n; off += COMM_CHUNK) {
    long c = n - off < COMM_CHUNK ? n - off : COMM_CHUNK;
    MPI_Allreduce(MPI_IN_PLACE, buf + off, (int) c, MPI_DOUBLE,
                  MPI_MIN, g_comm);
  }
}
void allreduce_xor_u32(uint32_t * buf, long n) {
  trace("ar_xor_u32", n, __builtin_return_address(0));
  if (qc_grid.nranks == 1) return;
  for (long off = 0; off < n; off += COMM_CHUNK) {
    long c = n - off < COMM_CHUNK ? n - off : COMM_CHUNK;
    MPI_Allreduce(MPI_IN_PLACE, buf + off, (int) c, MPI_UINT32_T,
                  MPI_BXOR, g_comm);
  }
}
#else
void allreduce_sum_d(double * buf, long n) { (void) buf; (void) n; }
void allreduce_sum_f(float * buf, long n)  { (void) buf; (void) n; }
void allreduce_min_d(double * buf, long n) { (void) buf; (void) n; }
void allreduce_xor_u32(uint32_t * buf, long n) { (void) buf; (void) n; }
#endif

// --- Fine-lattice halo exchange (see comm.h for the contract) ---------------
// Compiled to no-ops unless both MPI and a decomposed dimension are configured.
#if QC_MPI && FERM_GHOSTS

// Growable pack buffers for the strided z faces (t faces are contiguous slabs
// and need no packing).  Reused across calls; master-thread only.
static char * g_packbuf[4] = {NULL, NULL, NULL, NULL};
static long   g_packbuf_sz = 0;
static void * packbuf(int which, long bytes) {
  if (bytes > g_packbuf_sz) {
    for (int i = 0; i < 4; i ++) {
      free(g_packbuf[i]);
      g_packbuf[i] = malloc(bytes);
    }
    g_packbuf_sz = bytes;
  }
  return g_packbuf[which];
}

// MPI counts are int.  Every halo byte count goes through here, so a face
// past INT_MAX (large local volume x per-site payload, e.g. many null vectors
// in fp64) dies instead of wrapping into a bad count.
static int face_count(long bytes) {
  if (bytes > INT_MAX)
    comm_die("halo face of %ld bytes exceeds MPI's int count limit (%d)",
             bytes, INT_MAX);
  return (int) bytes;
}

// one blocking paired exchange: my `face` travels toward `dst`; the matching
// face from `src` lands in `ghost`.
static void face_sendrecv(const void * face, int dst, void * ghost, int src,
                          long bytes, int tag) {
  const int n = face_count(bytes);
  MPI_Sendrecv((void *) face, n, MPI_BYTE, dst, tag,
               ghost, n, MPI_BYTE, src, tag,
               g_comm, MPI_STATUS_IGNORE);
}

void halo_ferm_half(const void * half, void * ghost, long site_bytes) {
  trace("halo_ferm", site_bytes, __builtin_return_address(0));
  const char * f = half;
  char * g = ghost;
  // Post every face non-blocking so the two directions within a dim -- and both
  // dims when both are decomposed -- fly at once instead of as serial paired
  // Sendrecvs.  Recvs first (so the matching sends always have a posted recv to
  // land in), then pack, then sends, then one Waitall: the call still returns
  // only after the ghost is complete, so the synchronous contract is unchanged.
  // Ghost offsets and tags match the blocking layout (z 102/103, t 104/105).
  MPI_Request reqs[8];
  int nreq = 0;
  // Faces are gathered site by site through the funnel (site_half) into the
  // pack buffers, in the receiver's lex face-row order.  Size every pack slot
  // up front: a later, larger packbuf request reallocates all slots, which
  // must never happen while an earlier slot's Isend is in flight.
  long pb_bytes = 0;
#if DECOMP_Z
  const long zface = (long) GHOST_FACE_Z * site_bytes;
  if (zface > pb_bytes) pb_bytes = zface;
#endif
#if DECOMP_T
  const long tface = (long) GHOST_FACE_T * site_bytes;
  if (tface > pb_bytes) pb_bytes = tface;
#endif
#if DECOMP_Z
  char * zlo = packbuf(0, pb_bytes), * zhi = packbuf(1, pb_bytes);
  MPI_Irecv(g + (long) G_OFF_ZLO * site_bytes, face_count(zface), MPI_BYTE,
            qc_grid.nbr_lo[2], 102, g_comm, &reqs[nreq ++]);
  MPI_Irecv(g + (long) G_OFF_ZHI * site_bytes, face_count(zface), MPI_BYTE,
            qc_grid.nbr_hi[2], 103, g_comm, &reqs[nreq ++]);
  for (int t = 0; t < Lt; t ++)
    for (int y = 0; y < Ly; y ++)
      for (int x = 0; x < Lxh; x ++) {
        const long row = (((long) t * Ly + y) * Lxh + x) * site_bytes;
        memcpy(zlo + row, f + site_half(x, y, 0, t) * site_bytes, site_bytes);
        memcpy(zhi + row, f + site_half(x, y, Lz - 1, t) * site_bytes,
               site_bytes);
      }
  // hi face travels +z (lands as the neighbor's z-lo ghost) and vice versa
  MPI_Isend(zhi, face_count(zface), MPI_BYTE, qc_grid.nbr_hi[2],
            102, g_comm, &reqs[nreq ++]);
  MPI_Isend(zlo, face_count(zface), MPI_BYTE, qc_grid.nbr_lo[2],
            103, g_comm, &reqs[nreq ++]);
#endif
#if DECOMP_T
  char * tlo = packbuf(2, pb_bytes), * thi = packbuf(3, pb_bytes);
  MPI_Irecv(g + (long) G_OFF_TLO * site_bytes, face_count(tface), MPI_BYTE,
            qc_grid.nbr_lo[3], 104, g_comm, &reqs[nreq ++]);
  MPI_Irecv(g + (long) G_OFF_THI * site_bytes, face_count(tface), MPI_BYTE,
            qc_grid.nbr_hi[3], 105, g_comm, &reqs[nreq ++]);
  for (int z = 0; z < Lz; z ++)
    for (int y = 0; y < Ly; y ++)
      for (int x = 0; x < Lxh; x ++) {
        const long row = (((long) z * Ly + y) * Lxh + x) * site_bytes;
        memcpy(tlo + row, f + site_half(x, y, z, 0) * site_bytes, site_bytes);
        memcpy(thi + row, f + site_half(x, y, z, Lt - 1) * site_bytes,
               site_bytes);
      }
  MPI_Isend(thi, face_count(tface), MPI_BYTE, qc_grid.nbr_hi[3],
            104, g_comm, &reqs[nreq ++]);
  MPI_Isend(tlo, face_count(tface), MPI_BYTE, qc_grid.nbr_lo[3],
            105, g_comm, &reqs[nreq ++]);
#endif
  if (nreq > 0)
    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
}

// SAP compact halo (see comm.h): per-face pre-packed blobs, per-direction
// asymmetric sizes, same post-all-Irecv / post-all-Isend / one-Waitall shape
// as halo_ferm_half.  Tags 140 (z up), 141 (z down), 142 (t up), 143 (t down)
// -- distinct per direction so the 2-rank self-neighbor case cannot
// cross-match, same scheme as 102-105.
void halo_ferm_sap(const void * send[4], void * recv[4],
                   const long sbytes[4], const long rbytes[4]) {
  trace("halo_ferm_sap", sbytes[0] + sbytes[1] + sbytes[2] + sbytes[3],
        __builtin_return_address(0));
  MPI_Request reqs[8];
  int nreq = 0;
#if DECOMP_Z
  MPI_Irecv(recv[0], face_count(rbytes[0]), MPI_BYTE, qc_grid.nbr_lo[2],
            140, g_comm, &reqs[nreq ++]);
  MPI_Irecv(recv[1], face_count(rbytes[1]), MPI_BYTE, qc_grid.nbr_hi[2],
            141, g_comm, &reqs[nreq ++]);
  MPI_Isend((void *) send[1], face_count(sbytes[1]), MPI_BYTE,
            qc_grid.nbr_hi[2], 140, g_comm, &reqs[nreq ++]);
  MPI_Isend((void *) send[0], face_count(sbytes[0]), MPI_BYTE,
            qc_grid.nbr_lo[2], 141, g_comm, &reqs[nreq ++]);
#endif
#if DECOMP_T
  MPI_Irecv(recv[2], face_count(rbytes[2]), MPI_BYTE, qc_grid.nbr_lo[3],
            142, g_comm, &reqs[nreq ++]);
  MPI_Irecv(recv[3], face_count(rbytes[3]), MPI_BYTE, qc_grid.nbr_hi[3],
            143, g_comm, &reqs[nreq ++]);
  MPI_Isend((void *) send[3], face_count(sbytes[3]), MPI_BYTE,
            qc_grid.nbr_hi[3], 142, g_comm, &reqs[nreq ++]);
  MPI_Isend((void *) send[2], face_count(sbytes[2]), MPI_BYTE,
            qc_grid.nbr_lo[3], 143, g_comm, &reqs[nreq ++]);
#endif
  if (nreq > 0)
    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
}

void halo_gauge(void * U, long site_bytes) {
  trace("halo_gauge", site_bytes, __builtin_return_address(0));
  char * u = U;
  char * g = u + (long) (V) * site_bytes;   // ghosts appended at site V
#if DECOMP_Z
  {
    // send my z-hi slice toward +z; receive my z-lo ghost from -z
    const long chunk = (long) Ly * Lx * site_bytes;
    const long face  = chunk * Lt;
    char * hi = packbuf(0, face);
    for (int t = 0; t < Lt; t ++)
      memcpy(hi + t * chunk, u + ((long) t * Lz + (Lz - 1)) * chunk, chunk);
    face_sendrecv(hi, qc_grid.nbr_hi[2], g + (long) GG_OFF_ZLO * site_bytes,
                  qc_grid.nbr_lo[2], face, 106);
  }
#endif
#if DECOMP_T
  {
    const long face = (long) Lx * Ly * Lz * site_bytes;
    face_sendrecv(u + (long) (Lt - 1) * face, qc_grid.nbr_hi[3],
                  g + (long) GG_OFF_TLO * site_bytes, qc_grid.nbr_lo[3],
                  face, 107);
  }
#endif
}

// Coarse field halo: same z-strided / t-contiguous slab structure as the fine
// half-field exchange, but on a full lexicographic coarse field with runtime
// extents (so it serves every coarse level).  z faces are lt chunks of (ly*lx)
// contiguous sites; t faces are whole contiguous slabs.  The ghost layout is
// computed here from the same rule the CG_OFF_* macros use, so caller and
// exchange agree.  Distinct tag block (112-115) from the fine halo (102-107).
void halo_coarse(const void * field, void * ghost, long site_bytes,
                    int lx, int ly, int lz, int lt) {
  trace("halo_coarse", site_bytes, __builtin_return_address(0));
  const char * f = field;
  char * g = ghost;
  const int  decz = qc_grid.P[2] > 1, dect = qc_grid.P[3] > 1;
  const long zface_sites = (long) lx * ly * lt;
  const long tface_sites = (long) lx * ly * lz;
  const long off_zhi = zface_sites;
  const long off_tlo = 2 * zface_sites * decz;
  const long off_thi = off_tlo + tface_sites;
  if (decz) {
    const long chunk = (long) ly * lx * site_bytes;   // one z-layer at fixed t
    const long face  = zface_sites * site_bytes;
    char * lo = packbuf(0, chunk * lt), * hi = packbuf(1, chunk * lt);
    for (int t = 0; t < lt; t ++) {
      memcpy(lo + t * chunk, f + ((long) t * lz            ) * chunk, chunk);
      memcpy(hi + t * chunk, f + ((long) t * lz + (lz - 1)) * chunk, chunk);
    }
    // hi face travels +z (lands as the +z neighbor's z-lo ghost) and vice versa
    face_sendrecv(hi, qc_grid.nbr_hi[2], g + 0        * site_bytes,
                  qc_grid.nbr_lo[2], face, 112);
    face_sendrecv(lo, qc_grid.nbr_lo[2], g + off_zhi  * site_bytes,
                  qc_grid.nbr_hi[2], face, 113);
  }
  if (dect) {
    const long face = tface_sites * site_bytes;     // whole t slab, contiguous
    face_sendrecv(f + (long) (lt - 1) * face, qc_grid.nbr_hi[3],
                  g + off_tlo * site_bytes, qc_grid.nbr_lo[3], face, 114);
    face_sendrecv(f, qc_grid.nbr_lo[3],
                  g + off_thi * site_bytes, qc_grid.nbr_hi[3], face, 115);
  }
}

// Non-blocking split of halo_coarse (see comm.h).  Same packing, ghost layout
// and tags (112-115) as the blocking version, posted as Irecv/Isend so the
// caller can compute the rank interior while the faces are in flight.  Single
// in-flight exchange: requests + count kept file-static, drained by
// halo_coarse_wait.
static MPI_Request g_coarse_reqs[8];
static int         g_coarse_nreq = 0;

void halo_coarse_start(const void * field, void * ghost, long site_bytes,
                          int lx, int ly, int lz, int lt) {
  trace("halo_coarse_start", site_bytes, __builtin_return_address(0));
  const char * f = field;
  char * g = ghost;
  const int  decz = qc_grid.P[2] > 1, dect = qc_grid.P[3] > 1;
  const long zface_sites = (long) lx * ly * lt;
  const long tface_sites = (long) lx * ly * lz;
  const long off_zhi = zface_sites;
  const long off_tlo = 2 * zface_sites * decz;
  const long off_thi = off_tlo + tface_sites;
  g_coarse_nreq = 0;
  if (decz) {
    const long chunk = (long) ly * lx * site_bytes;   // one z-layer at fixed t
    const long face  = zface_sites * site_bytes;
    char * lo = packbuf(0, chunk * lt), * hi = packbuf(1, chunk * lt);
    // post receives first so the matching sends always have a posted recv to
    // land
    MPI_Irecv(g + 0       * site_bytes, face_count(face), MPI_BYTE,
              qc_grid.nbr_lo[2], 112, g_comm, &g_coarse_reqs[g_coarse_nreq ++]);
    MPI_Irecv(g + off_zhi * site_bytes, face_count(face), MPI_BYTE,
              qc_grid.nbr_hi[2], 113, g_comm, &g_coarse_reqs[g_coarse_nreq ++]);
    for (int t = 0; t < lt; t ++) {
      memcpy(lo + t * chunk, f + ((long) t * lz            ) * chunk, chunk);
      memcpy(hi + t * chunk, f + ((long) t * lz + (lz - 1)) * chunk, chunk);
    }
    // hi face travels +z (lands as the +z neighbor's z-lo ghost) and vice versa
    MPI_Isend(hi, face_count(face), MPI_BYTE, qc_grid.nbr_hi[2],
              112, g_comm, &g_coarse_reqs[g_coarse_nreq ++]);
    MPI_Isend(lo, face_count(face), MPI_BYTE, qc_grid.nbr_lo[2],
              113, g_comm, &g_coarse_reqs[g_coarse_nreq ++]);
  }
  if (dect) {
    const long face = tface_sites * site_bytes;     // whole t slab, contiguous
    MPI_Irecv(g + off_tlo * site_bytes, face_count(face), MPI_BYTE,
              qc_grid.nbr_lo[3], 114, g_comm,
              &g_coarse_reqs[g_coarse_nreq ++]);
    MPI_Irecv(g + off_thi * site_bytes, face_count(face), MPI_BYTE,
              qc_grid.nbr_hi[3], 115, g_comm,
              &g_coarse_reqs[g_coarse_nreq ++]);
    MPI_Isend(f + (long) (lt - 1) * face, face_count(face), MPI_BYTE,
              qc_grid.nbr_hi[3], 114, g_comm,
              &g_coarse_reqs[g_coarse_nreq ++]);
    MPI_Isend(f,                          face_count(face), MPI_BYTE,
              qc_grid.nbr_lo[3], 115, g_comm,
              &g_coarse_reqs[g_coarse_nreq ++]);
  }
}

void halo_coarse_wait(void) {
  if (g_coarse_nreq > 0)
    MPI_Waitall(g_coarse_nreq, g_coarse_reqs, MPI_STATUSES_IGNORE);
  g_coarse_nreq = 0;
}

// Extended-gauge halo for compute_Fmunu: fills the one-deep z/t face ghost
// layers and the (z,t) corner of a gauge array padded by one site in each
// decomposed dim.  Layout: Uext[te][ze][y][x][4 links], te in [0,Lt+2*dt),
// ze in [0,Lz+2*dz) (dz=DECOMP_Z, dt=DECOMP_T); interior local (t,z) lives at
// (t+dt, z+dz), ghost planes at index 0 and L+ {1}.  The clover leaf steps
// one site in both mu and nu, so a (z,t)-plane plaquette needs the diagonal
// corner -- filled here for free by exchanging z first (over the interior t
// range), then exchanging the whole z-extended t-slabs (which now carry the
// z-ghosts) in t.  site_bytes = one gauge site (4 links).  No-op unless MPI
// and a decomposed dim are configured.
void halo_gauge_ext(void * Uext, long site_bytes) {
  trace("halo_gauge_ext", site_bytes, __builtin_return_address(0));
  char * u = Uext;
  const long Lzx = (long) Lz + 2 * DECOMP_Z;
  const long plane   = (long) Ly * Lx * site_bytes;  // one ze-plane (all y,x)
  const long tstride = Lzx * plane;                  // one te-slab (all ze,y,x)
#if DECOMP_Z
  {
    // z faces, packed across the interior t range [dt, Lt+dt):
    // one ze-plane per t.
    const long nt = Lt, bytes = plane * nt;
    char * slo = packbuf(0, bytes);     // my interior z=0     plane (ze = dz)
    char * shi = packbuf(1, bytes);     // my interior z=Lz-1  plane
                                        // (ze = Lz-1+dz)
    char * rlo = malloc(bytes);         // -> my -z ghost (ze = 0)
    char * rhi = malloc(bytes);         // -> my +z ghost (ze = Lzx-1)
    for (int t = 0; t < Lt; t ++) {
      char * slab = u + (long) (t + DECOMP_T) * tstride;
      memcpy(slo + (long) t * plane, slab + (long) DECOMP_Z * plane, plane);
      memcpy(shi + (long) t * plane, slab + (long) (Lz - 1 + DECOMP_Z) * plane,
             plane);
    }
    // z=Lz-1 travels +z -> the +z neighbor's -z ghost; z=0 travels -z likewise
    face_sendrecv(shi, qc_grid.nbr_hi[2], rlo, qc_grid.nbr_lo[2], bytes, 116);
    face_sendrecv(slo, qc_grid.nbr_lo[2], rhi, qc_grid.nbr_hi[2], bytes, 117);
    for (int t = 0; t < Lt; t ++) {
      char * slab = u + (long) (t + DECOMP_T) * tstride;
      memcpy(slab + 0,                 rlo + (long) t * plane, plane);  // ze=0
      memcpy(slab + (Lzx - 1) * plane, rhi + (long) t * plane, plane);  // ze=Lzx-1
    }
    free(rlo); free(rhi);
  }
#endif
#if DECOMP_T
  {
    // t faces: whole z-extended slabs (contiguous); they carry the z-ghosts
    // filled above, so the four (z,t) corners arrive correctly.
    // -t ghost = te 0, +t = Ltx-1.  Declared here rather than beside Lzx
    // above: it is the one quantity used only on this path, so at DECOMP_T 0
    // an outer declaration is an unused variable.
    const long Ltx = (long) Lt + 2 * DECOMP_T;
    face_sendrecv(u + (long) (Lt - 1 + DECOMP_T) * tstride, qc_grid.nbr_hi[3],
                  u + 0,                   qc_grid.nbr_lo[3], tstride, 118);
    face_sendrecv(u + (long) DECOMP_T * tstride,            qc_grid.nbr_lo[3],
                  u + (Ltx - 1) * tstride, qc_grid.nbr_hi[3], tstride, 119);
  }
#endif
}

#else  // !QC_MPI || !FERM_GHOSTS
void halo_ferm_half(const void * half, void * ghost, long site_bytes) {
  (void) half; (void) ghost; (void) site_bytes;
}
void halo_ferm_sap(const void * send[4], void * recv[4],
                   const long sbytes[4], const long rbytes[4]) {
  (void) send; (void) recv; (void) sbytes; (void) rbytes;
}
void halo_gauge(void * U, long site_bytes) { (void) U; (void) site_bytes; }
void halo_coarse(const void * field, void * ghost, long site_bytes,
                    int lx, int ly, int lz, int lt) {
  (void) field; (void) ghost; (void) site_bytes;
  (void) lx; (void) ly; (void) lz; (void) lt;
}
void halo_coarse_start(const void * field, void * ghost, long site_bytes,
                          int lx, int ly, int lz, int lt) {
  (void) field; (void) ghost; (void) site_bytes;
  (void) lx; (void) ly; (void) lz; (void) lt;
}
void halo_coarse_wait(void) {}
void halo_gauge_ext(void * Uext, long site_bytes) {
  (void) Uext; (void) site_bytes;
}
#endif

// Flushed per call so the message is printed even when the job aborts: abort()
// does not flush stdio, and a redirected stdout would lose the buffer.
void print0(const char * fmt, ...) {
  if (qc_grid.rank) return;
  va_list ap;
  va_start(ap, fmt);
  vprintf(fmt, ap);
  va_end(ap);
  fflush(stdout);
}
