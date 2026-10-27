// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

/* Fine-lattice site-storage order -- the single source of truth.
 *
 * Deliberately self-contained: no includes, dimensions passed as arguments,
 * pure integer inline functions.  global_params.h includes it and routes every
 * fine-level site access through it (site_half), so this file alone defines
 * where a site lives in memory.
 *
 * Block order: loc = blk*128 + nl, where blk rasterizes the 4^4 blocks
 * x-fastest (fine SAP bs == BLOCK_AGG == 4 is what lets one ordering serve
 * the smoother, the transfers and the stencils) and nl is the parity-local
 * in-block lexicographic index (t,z,y slowest->fastest, then the in-block x
 * pair).  A block's 128 sites * site_size floats are then contiguous per
 * (parity, block): a SAP block sweep's working set is one dense span (which
 * restores its L2 residency -- in the lex order the block's z/t site columns
 * collide in the 64 KB-period L2 sets), and z/t stencil neighbors stop
 * stacking 8-deep on the same 4 KB-period L1 sets.  Lex order:
 * loc = ((t*nz+z)*ny+y)*nxh + x/2.  The gauge field is not covered by these
 * maps -- it stays lexicographic (site_full in global_params.h).
 */
#ifndef SITE_LAYOUT_H
#define SITE_LAYOUT_H

// LOC_HOST_BLOCK: the order of the fine-lattice arrays.  The funnel (site_half
// in global_params.h) routes every site access through host_loc_of_dims, so
// this one macro is the storage order.  1 = block-major (see header), 0 = the
// plain lexicographic order block-major replaced, kept switchable as the
// reference implementation.  Changing it relayouts every fine field at once,
// which is the point: no code inside the build may assume an order.
//
// Outside the build it is no longer free.  qc.h publishes this order through
// qc_site_index so that a caller can fill a source in place, so a change here
// is a change to what a linked program sees.  Callers are told to go through
// qc_site_index rather than reimplement the arithmetic, which is what keeps a
// retune possible -- but it does mean a relayout is a release-note item, not an
// internal detail.
#define LOC_HOST_BLOCK 1

// The two raw maps: full coords -> within-half site index (the caller owns the
// parity*Vh offset when a parity-major full index is wanted).  Note both maps
// are x-parity-independent: the two same-parity sites of an x-row land on
// (x&3)>>1 (block) / x/2 (lex) for either parity, so one half-map serves both
// halves (and single-parity arrays).  Dimension parameters are named n*
// (never Lxh/Ly/...): the extent macros of any including header would
// otherwise expand inside the parameter lists.
static inline long loc_block_of_dims(int nxh, int ny, int nz, int nt,
                                     int x, int y, int z, int t) {
  (void) nt;
  long blk = (((long)(t >> 2) * (nz >> 2) + (z >> 2)) * (ny >> 2) + (y >> 2))
           * (nxh >> 1) + (x >> 2);
  int  nl  = ((((t & 3) * 4 + (z & 3)) * 4 + (y & 3)) << 1) + ((x & 3) >> 1);
  // 128 = 4^4/2 sites per parity per block; fixed together with the >>2 / &3
  // tiling above (not a tunable -- the layout block is 4^4 by construction,
  // independent of the runtime SAP/BLOCK_AGG sizes)
  return blk * 128 + nl;
}
static inline long loc_lex_of_dims(int nxh, int ny, int nz, int nt, int x,
                                   int y, int z, int t) {
  (void) nt;
  return (((long) t * nz + z) * ny + y) * nxh + x / 2;
}

// the order of the host arrays (see LOC_HOST_BLOCK above)
static inline long host_loc_of_dims(int nxh, int ny, int nz, int nt,
                                    int x, int y, int z, int t) {
#if LOC_HOST_BLOCK
  return loc_block_of_dims(nxh, ny, nz, nt, x, y, z, t);
#else
  return loc_lex_of_dims(nxh, ny, nz, nt, x, y, z, t);
#endif
}

// Inverse of host_loc_of_dims: within-half site index -> (xh, y, z, t).  For
// the rare consumers that must decode a stored host index (per-timeslice
// residual breakdowns and the like).
static inline void host_loc_inv_of_dims(int nxh, int ny, int nz, int nt, long s,
                                        int * xh, int * y, int * z, int * t) {
  (void) nt;
#if LOC_HOST_BLOCK
  int  nl  = (int)(s & 127);
  long blk = s >> 7;
  int  bx  = (int)(blk % (nxh >> 1));  long q = blk / (nxh >> 1);
  int  by  = (int)(q % (ny >> 2));     q /= (ny >> 2);
  int  bz  = (int)(q % (nz >> 2));     q /= (nz >> 2);
  *xh = bx * 2 + (nl & 1);  nl >>= 1;
  *y  = by * 4 + (nl & 3);  nl >>= 2;
  *z  = bz * 4 + (nl & 3);  nl >>= 2;
  *t  = (int) q * 4 + nl;
#else
  *xh = (int)(s % nxh);  long q = s / nxh;
  *y  = (int)(q % ny);   q /= ny;
  *z  = (int)(q % nz);   q /= nz;
  *t  = (int) q;
#endif
}

#endif /* SITE_LAYOUT_H */
