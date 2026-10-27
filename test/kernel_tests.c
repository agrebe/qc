// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

// Kernel-level benchmarks, split out of test.c so that they can be compiled
// twice.
//
// test.o is untwinned (Makefile OBJS lists it with no _f/_d suffix), so in a
// mixed build it takes the driver layer's double precision.
// `#if ASM && MG_SINGLE` is therefore false inside it, and a bench living
// there can only ever time the double C fallbacks -- timing the asm kernels
// the MG actually runs needed a separate PRECISION=single build, and the two
// sets of numbers are easy to confuse.
//
// Building this file as kernel_tests_f.o / kernel_tests_d.o, through the same
// %_f.o / %_d.o pattern rules the operator and MG bodies use, puts both
// instantiations in one binary: the _f twin dispatches to the asm kernels at
// REAL=float, the _d twin to the portable C kernels at REAL=double (which is
// what a PRECISION=double build runs, and which also halves the SIMD lanes).
// The driver picks a twin by name -- see test.h.
//
// No bridge (cf. mixed_bridge.c) is needed here: nothing crosses the precision
// seam.  Each bench allocates its own fields, times a kernel, prints, and
// frees, so the only requirement on the entry points is that their signatures
// contain no REAL -- a double-typed driver can then call the float twin safely.

#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>           // sqrt, in the SU(3) projection
#include "global_params.h"
#include "utils.h"          // compute_Fmunu
#include "clover_ferm.h"    // clover_kernel, wilson_kernel
#include "solver.h"         // time_used
#include "test.h"
#include "omp.h"            // omp_get_thread_num

// Deterministic pseudo-random values in [-1, 1) from Knuth's MMIX 64-bit
// linear congruential generator, or LCG (multiplier 6364136223846793005,
// increment 1442695040888963407, the digits of log2 e).  A multiplier of 1
// mod 4 with an odd increment gives the state its full 2^64 period; the low
// bits of a power-of-two LCG cycle quickly, so only high bits are kept.
// Matching the generator the asm correctness harnesses use lets a run
// reproduce exactly with no data file.
static unsigned long bench_seed;
static REAL bench_rand(void) {
  bench_seed = bench_seed * 6364136223846793005UL + 1442695040888963407UL;
  return (REAL) (((double) ((bench_seed >> 33) & 0x7fffffff) / 0x40000000)
                 - 1.0);
}

// Project one gauge link onto SU(3): Gram-Schmidt rows 0 and 1, then
// row2 = conj(row0 x row1).  Link layout is 18 REAL, row-major e = c1*3 + c2,
// [2e] = re and [2e+1] = im; the orthonormalization runs in double whatever
// REAL is, so the result is unitary to REAL's precision rather than to sqrt
// of it.
static void su3_project(REAL * U) {
  double na = 0; for (int k = 0; k < 6; k ++) na += (double) U[k] * U[k];
  na = 1.0 / sqrt(na);
  for (int k = 0; k < 6; k ++) U[k] = (REAL) (U[k] * na);    // normalize row0
  double dr = 0, di = 0;                                     // <row0,row1>
  for (int k = 0; k < 3; k ++) {
    double ar = U[2 * k], ai = U[2 * k + 1],
           br = U[6 + 2 * k], bi = U[6 + 2 * k + 1];
    dr += ar * br + ai * bi; di += ar * bi - ai * br;
  }
  for (int k = 0; k < 3; k ++) {                           // row1 -= <..> row0
    double ar = U[2 * k], ai = U[2 * k + 1];
    U[6 + 2 * k]     = (REAL) (U[6 + 2 * k]     - (dr * ar - di * ai));
    U[6 + 2 * k + 1] = (REAL) (U[6 + 2 * k + 1] - (dr * ai + di * ar));
  }
  double nb = 0; for (int k = 6; k < 12; k ++) nb += (double) U[k] * U[k];
  nb = 1.0 / sqrt(nb);
  for (int k = 6; k < 12; k ++) U[k] = (REAL) (U[k] * nb);    // normalize row1
  double a0r = U[0], a0i = U[1], a1r = U[2], a1i = U[3], a2r = U[4], a2i = U[5];
  double b0r = U[6], b0i = U[7], b1r = U[8], b1i = U[9], b2r = U[10], b2i = U[11];
  // row2 = conj(row0 x row1)
  U[12] = (REAL) (a1r * b2r - a1i * b2i - a2r * b1r + a2i * b1i);
  U[13] = (REAL) (a2r * b1i + a2i * b1r - a1r * b2i - a1i * b2r);
  U[14] = (REAL) (a2r * b0r - a2i * b0i - a0r * b2r + a0i * b2i);
  U[15] = (REAL) (a0r * b2i + a0i * b2r - a2r * b0i - a2i * b0r);
  U[16] = (REAL) (a0r * b1r - a0i * b1i - a1r * b0r + a1i * b0i);
  U[17] = (REAL) (a1r * b0i + a1i * b0r - a0r * b1i - a0i * b1r);
}

// Synthetic gauge field, in place of reading an ILDG configuration.  These are
// throughput benchmarks and neither kernel branches on a value, so the data
// only has to be of the right class: finite, normal, and special-unitary per
// link.  SU(3) rather than generic random matrices because that is what a
// link-reconstruction scheme is entitled to assume -- store 12 floats per link
// and rebuild the third row as above, and a non-unitary link silently yields a
// different operator.  Not reading a file also lets the benches run anywhere.
static void fill_random_gauge(REAL * U) {
  bench_seed = 0x9e3779b97f4a7c15UL;   // 2^64/phi, an arbitrary fixed seed
  for (long i = 0; i < (long) gauge_size; i ++) U[i] = bench_rand();
  for (long L = 0; L < (long) gauge_size / (Nc * Nc * 2); L ++)
    su3_project(U + L * (Nc * Nc * 2));
}

// Random fermion.  A point source would leave the field all but entirely zero,
// which is the least representative input a throughput measurement can have.
static void fill_random_ferm(REAL * f, long n) {
  for (long i = 0; i < n; i ++) f[i] = bench_rand();
}

// Core-clock speedometer: a dependent scalar-add chain retires ~1 iter/cycle
// on any x86 core, so wall time over the chain gives the core's current
// frequency without perf counters or cpufreq sysfs.  ~1 ms per reading.
// The loop is 32-byte-aligned and shorter than 32 bytes so its branch can
// never end on a 32-byte boundary: on Skylake/Cascade Lake the JCC-erratum
// microcode otherwise drops the line from the uop cache and the chain slows
// to 2 cycles/iter -- the reading comes out at exactly half the true clock,
// and since this function inlines per call site, whether that happens is an
// accident of surrounding code size.
// Gated on the architecture, not on AVX2/AVX512: the chain is plain integer asm
// and is just as valid in a scalar x86-64 build.  Elsewhere it reports 0.
static double bench_core_ghz() {
#if defined(__x86_64__)
  long n = 2000000, x = 0;
  struct timeval a, b;
  gettimeofday(&a, NULL);
  __asm__ volatile (".p2align 5\n1:\n\tadd $1, %0\n\tcmp %1, %0\n\tjne 1b"
                    : "+r"(x) : "r"(n));
  gettimeofday(&b, NULL);
  return n / time_used(a, b) / 1e9;
#else
  return 0.0;   // no portable cycle-accurate chain; banners print 0.00 GHz
#endif
}

// One banner line per bench.  Both twins of this file are linked into a mixed
// build and print consecutively, so the implementation and REAL width are what
// tell the two report blocks apart.  The core clock is reported alongside them
// because a GF/s number cannot be interpreted without it -- AVX-512 license
// downclock alone moves it enough to swamp a kernel change.
static void bench_banner(const char * what) {
#if ASM && MG_SINGLE
  const char * impl = "asm dispatch";
#else
  const char * impl = "C kernels";
#endif
  printf("%s [%s, REAL=%d B, num_rhs=%d, core %.2f GHz]\n",
         what, impl, (int) sizeof(REAL), num_rhs, bench_core_ghz());
}

// Number of timed calls, rounded down to the unrolled inner loop's stride.
static long bench_n(const char * env) {
  long n = 10000000L;
  { const char * s = getenv(env); if (s) n = atol(s); }
  n = n / 100 * 100;
  return n < 100 ? 100 : n;
}

// Report a completed timed loop.  fpc is the modeled flops per call.
static void bench_report(long n, REAL secs, double fpc) {
  printf("Time elapsed for n = %ld calls: %f sec\n", n, (double) secs);
  printf("Time per call: %f nsec\n", (double) secs / n * 1e9);
  printf("Speed: %f GFlops\n", fpc * n * num_rhs / (double) secs / 1e9);
}

// Correctness gate for the hand-written clover asm kernels -- the clover
// counterpart of test_wilson_asm.  Three properties, none of which a throughput
// measurement can see:
//
//   1. accumulate  clover_kernel        vs clover_kernel_c       (fp rounding)
//   2. overwrite   clover_kernel_ow     vs zero-then-accumulate  (bitwise)
//   3. in place    clover_kernel(f,f,.) vs out of place          (bitwise)
//
// (1) is approximate because the hand-scheduled FMA order differs from what the
// compiler emits, exactly as in test_wilson_asm.  (2) and (3) are exact by
// construction, so a tolerance there would hide a real violation: the _ow
// kernel is the same schedule with each accumulator's first load replaced by a
// register zero, and the in-place branch runs that same kernel into a zeroed
// stack temp.  Both are compared with memcmp rather than !=, which also catches
// a sign-of-zero divergence that float equality calls equal.
//
// The accumulators are seeded with random nonzero values rather than zeros, so
// a kernel that overwrote where it should accumulate fails (1) instead of
// passing it by luck.
//
// CLOVER_ASM_LIVE (clover_ferm.h) is the dispatcher's own condition, so the
// banner cannot claim asm coverage the build does not have.  In the _d twin
// clover_kernel *is* clover_kernel_c and all three comparisons are self-tests;
// they still run -- they gate the C path's own overwrite and in-place branches,
// which have no other test -- but the banner says which case it is.  Silently
// printing PASS for a C-against-C comparison is precisely what test_wilson_asm
// does today, compiled from the untwinned test.c.
void test_clover_asm() {
  long fsz = (long) ferm_size * num_rhs;
  long csz = (long) Lt * Lz * Ly * Lx * 2 * 6 * 6;
  int  site_size = Nc * Ns * num_rhs * 2;
  // malloc is 64-byte aligned here (global_params.h), which the asm kernels'
  // alignment-required vmovaps need.  A site's offset is loc * site_size
  // REALs, and site_size is a multiple of 16 wherever the asm path is live,
  // so aligning the base aligns every site.  Two output buffers suffice for
  // all three checks -- each is rewritten between them -- which matters
  // because a full field here is ferm_size * num_rhs REALs.
  REAL * out_a = malloc(sizeof(REAL) * fsz);
  REAL * out_b = malloc(sizeof(REAL) * fsz);
  REAL * src   = malloc(sizeof(REAL) * fsz);
  REAL * F     = malloc(sizeof(REAL) * csz);

  bench_seed = 0x9e3779b97f4a7c15UL;   // same fixed seed as the gauge fill
  fill_random_ferm(src, fsz);
  // Random reals give a random *Hermitian* clover block, not a malformed one:
  // the packed layout holds each 6x6 block as 36 independent REALs -- entry
  // (a,b) with a <= b takes its real part from F[a][b] and its imaginary part
  // from F[b][a], and (b,a) reads those same two slots with the imaginary part
  // negated.  So this is as admissible as a compute_Fmunu field while carrying
  // none of its structure or near-zero entries, which is what a differential
  // test wants.  The clover kernel never reads the gauge field, so none exists
  // here.
  fill_random_ferm(F, csz);

  // Sites at the low edge, low+1, and high edge of each dimension.  The kernel
  // is site-local -- no neighbors, no wrap -- so this samples site_half's
  // indexing rather than any stencil geometry, and one parity half covers the
  // code: parity reaches the kernel only as which half of the field and of F
  // the caller passes, which is the caller's arithmetic, not the kernel's.
  int xs[3] = {0, 1, Lxh - 1};
  int ys[3] = {0, 1, Ly - 1};
  int zs[3] = {0, 1, Lz - 1};
  int ts[3] = {0, 1, Lt - 1};
  double max_abs = 0, max_mag = 0, ow_max = 0, ip_max = 0;
  long n_ow_diff = 0, n_ip_diff = 0;
  int  n_sites = 0;
  for (int it = 0; it < 3; it ++)
    for (int iz = 0; iz < 3; iz ++)
      for (int iy = 0; iy < 3; iy ++)
        for (int ix = 0; ix < 3; ix ++) {
          int x = xs[ix], y = ys[iy], z = zs[iz], t = ts[it];
          long base = site_half(x, y, z, t) * site_size;
 
          // 1. accumulate, from an identical nonzero output in both buffers
          for (int k = 0; k < site_size; k ++)
            out_a[base+k] = out_b[base+k] = bench_rand();
          clover_kernel_c(out_a, src, F, x, y, z, t);
          clover_kernel  (out_b, src, F, x, y, z, t);
          for (int k = 0; k < site_size; k ++) {
            double d = fabs((double) out_a[base+k] - (double) out_b[base+k]);
            double m = fabs((double) out_a[base+k]);
            if (d > max_abs) max_abs = d;
            if (m > max_mag) max_mag = m;
          }
 
          // 2. overwrite vs zero-then-accumulate, both in the dispatched path
          for (int k = 0; k < site_size; k ++) out_a[base+k] = 0;
          clover_kernel   (out_a, src, F, x, y, z, t);
          clover_kernel_ow(out_b, src, F, x, y, z, t);
          if (memcmp(out_a + base, out_b + base, sizeof(REAL) * site_size))
            n_ow_diff ++;
          for (int k = 0; k < site_size; k ++) {
            double d = fabs((double) out_a[base+k] - (double) out_b[base+k]);
            if (d > ow_max) ow_max = d;
          }
 
          // 3. in place (ferm == output) against the out-of-place result now
          // in out_a.  Seeding out_b's site with src makes the in-place call
          // compute F * src, which is exactly what out_a holds -- by the same
          // kernel on the same inputs, so they must agree bitwise and not
          // merely closely.
          memcpy(out_b + base, src + base, sizeof(REAL) * site_size);
          clover_kernel(out_b, out_b, F, x, y, z, t);
          if (memcmp(out_a + base, out_b + base, sizeof(REAL) * site_size))
            n_ip_diff ++;
          for (int k = 0; k < site_size; k ++) {
            double d = fabs((double) out_a[base+k] - (double) out_b[base+k]);
            if (d > ip_max) ip_max = d;
          }

          n_sites ++;
        }

  // Global relative error, as in test_wilson_asm: a per-element relative error
  // is meaningless near the zeros that cancellation produces.  A logic bug in
  // any of the 72 output components is O(1); FMA-order differences are ~1e-7.
  double rel = (max_mag > 0) ? max_abs / max_mag : 0;
  int pass_acc = (rel < 1e-5), pass_ow = (n_ow_diff == 0),
      pass_ip = (n_ip_diff == 0);
  printf("test_clover_asm [%s, REAL=%d B, num_rhs=%d, %d sites]\n",
         CLOVER_ASM_LIVE ? "asm dispatch"
                         : "C kernels -- no asm path in this twin, so the "
                           "checks below are self-tests, not asm coverage",
         (int) sizeof(REAL), num_rhs, n_sites);
  printf("  accumulate vs clover_kernel_c:    global rel err = %.3e => %s\n",
         rel, pass_acc ? "PASS" : "FAIL");
  printf("  _ow vs zero-then-accumulate:      %ld/%d sites differ, "
         "max |diff| = %.3e => %s\n",
         n_ow_diff, n_sites, ow_max, pass_ow ? "PASS (bitwise)" : "FAIL");
  printf("  in place vs out of place:         %ld/%d sites differ, "
         "max |diff| = %.3e => %s\n",
         n_ip_diff, n_sites, ip_max, pass_ip ? "PASS (bitwise)" : "FAIL");
  printf("  => %s\n", (pass_acc && pass_ow && pass_ip) ? "PASS" : "FAIL");

  free(out_a); free(out_b); free(src); free(F);
}

// Correctness gate for the hand-written Wilson hopping asm kernels, the direct
// counterpart of test_clover_asm above.  Only one comparison rather than three,
// because the Wilson kernel has no overwrite twin and no in-place mode: its
// output site and its input neighbors are opposite parities, so the aliasing
// the clover checks exist for cannot arise.  What it has instead is stencil
// geometry, and that is where the coverage goes -- sites at the low edge,
// low+1, and high edge of each dimension, both parities, so every one of the 8
// directions is taken across a wrap in both senses.
//
// wilson_kernel accumulates into output (see wilson_kernel_dir_c's comment),
// so as in test_clover_asm the accumulators are seeded with random nonzero
// values rather than zeros: against a zeroed output, a kernel that overwrote
// where it should accumulate would pass.
//
// Unlike the clover test the gauge field is projected to SU(3) rather than left
// as raw random reals.  The asymmetry is real, not an oversight: 36 random
// reals *are* a valid packed Hermitian clover block, whereas 18 random reals
// are not a valid link, and a link-reconstruction scheme that rebuilds the
// third row from the first two would be entitled to assume unitarity.  Testing
// on data the operator can never see would buy nothing and could cost a
// spurious failure later.
//
// WILSON_ASM_LIVE (clover_ferm.h) is the dispatcher's own condition, so the
// banner cannot claim asm coverage the build does not have -- in the _d twin
// wilson_kernel *is* wilson_kernel_c and this compares a function with itself.
// Living in this twinned file is what makes that distinction available at all:
// compiled from the untwinned test.c, this test only ever saw the double C
// path and printed PASS for a tautology.
void test_wilson_asm() {
  long fsz = (long) ferm_size * num_rhs;
  int  site_size = Nc * Ns * num_rhs * 2;
  // malloc is 64-byte aligned: the asm kernels use alignment-required vmovaps
  REAL * out_ref = malloc(sizeof(REAL) * fsz);
  REAL * out_asm = malloc(sizeof(REAL) * fsz);
  REAL * src     = malloc(sizeof(REAL) * fsz);
  REAL * U       = malloc(sizeof(REAL) * gauge_size);

  fill_random_gauge(U);         // resets the LCG seed, so this must come first
  fill_random_ferm(src, fsz);

  int xs[3] = {0, 1, Lxh - 1};
  int ys[3] = {0, 1, Ly - 1};
  int zs[3] = {0, 1, Lz - 1};
  int ts[3] = {0, 1, Lt - 1};
  double max_abs = 0, max_mag = 0;
  long max_idx = -1;
  int  n_sites = 0;
  for (int it = 0; it < 3; it ++)
    for (int iz = 0; iz < 3; iz ++)
      for (int iy = 0; iy < 3; iy ++)
        for (int ix = 0; ix < 3; ix ++)
          for (int parity = 0; parity < 2; parity ++) {
            int x = xs[ix], y = ys[iy], z = zs[iz], t = ts[it];
            // The kernel reads the opposite parity half and writes its own, so
            // each (site, parity) case owns a disjoint output slice and the two
            // buffers need seeding only for the slice about to be written.
            long base = (long) parity * fsz / 2
                       + site_half(x, y, z, t) * site_size;
            for (int k = 0; k < site_size; k ++)
              out_ref[base+k] = out_asm[base+k] = bench_rand();
            REAL * in = src + (1 - parity) * fsz / 2;
            wilson_kernel_c(out_ref + (long) parity * fsz / 2, in, U, x, y, z,
                            t, parity);
            wilson_kernel  (out_asm + (long) parity * fsz / 2, in, U, x, y, z,
                            t, parity);
            for (int k = 0; k < site_size; k ++) {
              double d = fabs((double) out_ref[base+k]
                         - (double) out_asm[base+k]);
              double m = fabs((double) out_ref[base+k]);
              if (d > max_abs) { max_abs = d; max_idx = base + k; }
              if (m > max_mag) max_mag = m;
            }
            n_sites ++;
         }

  // Global relative error (max abs diff / max output magnitude): a per-element
  // relative error is meaningless near the zeros catastrophic cancellation
  // produces, where the denominator goes tiny.  A logic bug in any direction
  // gives an O(1) difference; the hand-scheduled FMA order differs from the
  // compiler's, which shows up at ~1e-7 relative in single precision.
  double rel = (max_mag > 0) ? max_abs / max_mag : 0;
  printf("test_wilson_asm [%s, REAL=%d B, num_rhs=%d, %d sites]\n",
         WILSON_ASM_LIVE ? "asm dispatch"
                         : "C kernels -- no asm path in this twin, so the "
                           "check below is a self-test, not asm coverage",
         (int) sizeof(REAL), num_rhs, n_sites);
  printf("  vs wilson_kernel_c: max |abs diff| = %.3e (at idx %ld, ref=%.6e "
         "test=%.6e), max|out| = %.3e\n", max_abs, max_idx,
         max_idx >= 0 ? (double) out_ref[max_idx] : 0.0,
         max_idx >= 0 ? (double) out_asm[max_idx] : 0.0, max_mag);
  printf("  global rel err = %.3e => %s\n", rel, rel < 1e-5 ? "PASS" : "FAIL");

  free(out_ref); free(out_asm); free(src); free(U);
}

// Correctness gate for the Schwarz-block Wilson kernels: the masked variant
// wilson_kernel_block and its offset-table twin wilson_kernel_block_itab.
//
//   1. masked asm    wilson_kernel_block vs wilson_kernel_block_c  (fp rounding)
//   2. complements   block(ub=0) then block(ub=1) vs wilson_kernel (fp rounding)
//   3. offset table  block_itab vs block(ub=0)                     (bitwise)
//
// (1) cannot gate the mask on its own: the asm dispatcher and the C reference
// build the same mask from the same expression, so an error shared by both
// still compares equal.
//
// (2) brings in an independent witness -- wilson_kernel has no mask and derives
// its own neighbor and gauge pointers -- and gates a live optimization:
// red_black_schwarz replaces a masked pair with one unmasked call on exactly
// this identity (solver.c, "exterior and interior direction masks are
// complements").  It establishes that the two modes partition the stencil, not
// where the split falls; any predicate XORed with update_boundary partitions,
// so a mask condition wrong on both sides survives this as well.  The two
// summation orders differ, hence rounding rather than bitwise.
//
// (3) is exact because the table holds the 14 pointer offsets and the mask
// wilson_kernel_block would have computed, so both calls reach the same kernel
// with identical arguments.  Checked away from block 0: the fast path adds the
// origin's offset to a table row, and site_half is the block-major
// host_loc_of_dims rather than a linear index, so that composition holds only
// for a block-aligned origin -- at block 0 the offset is zero and the check is
// vacuous.
void test_wilson_block_asm() {
  long fsz = (long) ferm_size * num_rhs;
  int  site_size = Nc * Ns * num_rhs * 2;
  int  bs = SCHWARZ_BLOCK_SIZE;
  REAL * out_a = malloc(sizeof(REAL) * fsz);
  REAL * out_b = malloc(sizeof(REAL) * fsz);
  REAL * src   = malloc(sizeof(REAL) * fsz);
  REAL * U     = malloc(sizeof(REAL) * gauge_size);
  fill_random_gauge(U);         // resets the LCG seed, so this must come first
  fill_random_ferm(src, fsz);

  // 1. Every mask pattern: all block-relative positions (full_x and result_*
  // mod bs sweep 0..bs-1, so each skip bit is exercised set and clear), both
  // update_boundary values, both parities.
  double mx_abs = 0, mx_mag = 0;
  int n_mask = 0;
  for (int ub = 0; ub < 2; ub ++)
    for (int parity = 0; parity < 2; parity ++)
      for (int t = 0; t < bs; t ++)
        for (int z = 0; z < bs; z ++)
          for (int y = 0; y < bs; y ++)
            for (int rx = 0; rx < bs; rx ++) {
              long base = (long) parity * fsz / 2
                         + site_half(rx, y, z, t) * site_size;
              for (int k = 0; k < site_size; k ++)
                out_a[base+k] = out_b[base+k] = bench_rand();
              REAL * in = src + (1 - parity) * fsz / 2;
              wilson_kernel_block_c(out_a + (long) parity * fsz / 2, in, U,
                                    rx, y, z, t, parity, bs, ub);
              wilson_kernel_block  (out_b + (long) parity * fsz / 2, in, U,
                                    rx, y, z, t, parity, bs, ub);
              for (int k = 0; k < site_size; k ++) {
                double d = fabs((double) out_a[base+k]
                          - (double) out_b[base+k]);
                double m = fabs((double) out_a[base+k]);
                if (d > mx_abs) mx_abs = d;
                if (m > mx_mag) mx_mag = m;
              }
              n_mask ++;
           }

  // 2. Interior and boundary masks together must reproduce the full stencil.
  double cx_abs = 0, cx_mag = 0;
  int n_comp = 0;
  for (int parity = 0; parity < 2; parity ++)
    for (int t = 0; t < bs; t ++)
      for (int z = 0; z < bs; z ++)
        for (int y = 0; y < bs; y ++)
          for (int rx = 0; rx < bs; rx ++) {
            long base = (long) parity * fsz / 2
                       + site_half(rx, y, z, t) * site_size;
            for (int k = 0; k < site_size; k ++)
              out_a[base+k] = out_b[base+k] = bench_rand();
            REAL * in = src + (1 - parity) * fsz / 2;
            REAL * oa = out_a + (long) parity * fsz / 2;
            REAL * ob = out_b + (long) parity * fsz / 2;
            wilson_kernel      (oa, in, U, rx, y, z, t, parity);
            wilson_kernel_block(ob, in, U, rx, y, z, t, parity, bs, 0);
            wilson_kernel_block(ob, in, U, rx, y, z, t, parity, bs, 1);
            for (int k = 0; k < site_size; k ++) {
              double d = fabs((double) out_a[base+k] - (double) out_b[base+k]);
              double m = fabs((double) out_a[base+k]);
              if (d > cx_abs) cx_abs = d;
              if (m > cx_mag) cx_mag = m;
            }
            n_comp ++;
          }

  // 3. Offset table vs the interior masked call it stands in for.
  wilson_block_itab_init(bs);
  int ox = bs, oy = bs, oz = bs, ot = bs;   // block-aligned, and not block 0
  long org_half = site_half(ox / 2, oy, oz, ot);
  long org_full = site_full(ox, oy, oz, ot);
  double ix_abs = 0;
  long n_itab_diff = 0;
  int n_itab = 0;
  for (int lt = 0; lt < bs; lt ++)
    for (int lz = 0; lz < bs; lz ++)
      for (int ly = 0; ly < bs; ly ++)
        for (int lx = 0; lx < bs; lx ++) {
          // local_idx must match the table's row order, which is the
          // SCHWARZ_BLOCK_SITES visit order (x fastest, full-x steps of 1).
          int local_idx = ((lt * bs + lz) * bs + ly) * bs + lx;
          // Block origins are even in every coordinate, so the site's parity is
          // just the parity of its local offset.
          int parity = (lx ^ ly ^ lz ^ lt) & 1;
          int rx = (ox + lx) / 2, ry = oy + ly, rz = oz + lz, rt = ot + lt;
          long base = (long) parity * fsz / 2
                     + site_half(rx, ry, rz, rt) * site_size;
          for (int k = 0; k < site_size; k ++)
            out_a[base+k] = out_b[base+k] = bench_rand();
          REAL * in = src + (1 - parity) * fsz / 2;
          wilson_kernel_block     (out_a + (long) parity * fsz / 2, in, U,
                                   rx, ry, rz, rt, parity, bs, 0);
          wilson_kernel_block_itab(out_b + (long) parity * fsz / 2, in, U,
                                   rx, ry, rz, rt, parity, bs,
                                   org_half, org_full, local_idx);
          if (memcmp(out_a + base, out_b + base, sizeof(REAL) * site_size))
            n_itab_diff ++;
          for (int k = 0; k < site_size; k ++) {
            double d = fabs((double) out_a[base+k] - (double) out_b[base+k]);
            if (d > ix_abs) ix_abs = d;
          }
          n_itab ++;
        }

  double rel_mask = (mx_mag > 0) ? mx_abs / mx_mag : 0;
  double rel_comp = (cx_mag > 0) ? cx_abs / cx_mag : 0;
  int p1 = rel_mask < 1e-5, p2 = rel_comp < 1e-5, p3 = (n_itab_diff == 0);
  printf("test_wilson_block_asm [%s, REAL=%d B, num_rhs=%d, bs=%d]\n",
         WILSON_ASM_LIVE ? "asm dispatch"
                         : "C kernels -- no asm path in this twin, so the "
                           "checks below are self-tests, not asm coverage",
         (int) sizeof(REAL), num_rhs, bs);
  printf("  masked vs wilson_kernel_block_c:   %d cases, rel err = %.3e => %s\n",
         n_mask, rel_mask, p1 ? "PASS" : "FAIL");
  printf("  interior+boundary vs wilson_kernel: %d cases, rel err = %.3e => %s\n",
         n_comp, rel_comp, p2 ? "PASS" : "FAIL");
  printf("  offset table vs masked call:       %ld/%d sites differ, "
         "max |diff| = %.3e => %s\n",
         n_itab_diff, n_itab, ix_abs, p3 ? "PASS (bitwise)" : "FAIL");
  printf("  => %s\n", (p1 && p2 && p3) ? "PASS" : "FAIL");

  free(out_a); free(out_b); free(src); free(U);
}

// Throughput of the dispatched clover kernel, one site per call.
void test_clover_kernel() {
  REAL * output = malloc(sizeof(REAL) * ferm_size * num_rhs);
  REAL * src = malloc(sizeof(REAL) * ferm_size * num_rhs);
  REAL * F = calloc(Lt * Lz * Ly * Lx * 2 * 6 * 6, sizeof(REAL));
  REAL * U = calloc(Lx * Ly * Lz * Lt * 4 * Nc * Nc * 2, sizeof(REAL));
  fill_random_gauge(U);
  fill_random_ferm(src, (long) ferm_size * num_rhs);
  // F is derived from the gauge rather than filled directly, so it keeps the
  // Hermitian block structure the packed a<=b / a>b indexing assumes.
  compute_Fmunu(U, F);
  // One untimed call first: it faults in the pages the timed loop then reuses,
  // so the first timed call does not pay for them.
  clover_kernel(output, src, F, 0, 0, 0, 0);

  long n_test = bench_n("CLOVER_BENCH_N");
  bench_banner("clover kernel");
  struct timeval start, end;
  gettimeofday(&start, NULL);
  #pragma omp parallel
  {
    #pragma omp for schedule(static)
    for (long i = 0; i < n_test; i += 100) {
      int tid = omp_get_thread_num();
      for (int j = 0; j < 100; j ++)
        clover_kernel(output, src, F, 0, 0, 0, tid);
    }
  }
  gettimeofday(&end, NULL);
  bench_report(n_test, time_used(start, end), CLOVER_FLOPS);

  free(U);
  free(F);
  free(output);
  free(src);
}

// Timing counterpart to test_wilson_asm (which only checks correctness):
// reports a GFlop/s number for the dispatched wilson_kernel (asm when ASM=1),
// in the same format as test_clover_kernel, for an apples-to-apples roofline
// comparison against the clover kernel and against fine-SAP's measured
// inner-iteration rate.  Calls with parity=0, so `ferm` must be the
// opposite-parity (odd) half of a full ferm_size*num_rhs buffer -- same
// convention test_wilson_asm uses.
void test_wilson_kernel() {
  long fsz = (long) ferm_size * num_rhs;
  long gsz = (long) Lx * Ly * Lz * Lt * 4 * Nc * Nc * 2;
  REAL * output = malloc(sizeof(REAL) * fsz);
  REAL * src    = malloc(sizeof(REAL) * fsz);
  memset(output, 0, sizeof(REAL) * fsz);
  fill_random_ferm(src, fsz);
  REAL * U = calloc(gsz, sizeof(REAL));
  fill_random_gauge(U);

  long n_test = bench_n("WILSON_BENCH_N");
  bench_banner("wilson kernel");
  struct timeval start, end;
  gettimeofday(&start, NULL);
  #pragma omp parallel
  {
    #pragma omp for schedule(static)
    for (long i = 0; i < n_test; i += 100) {
      int tid = omp_get_thread_num();
      for (int j = 0; j < 100; j ++)
        wilson_kernel(output, src + fsz / 2, U, 0, 0, 0, tid, 0);
    }
  }
  gettimeofday(&end, NULL);
  bench_report(n_test, time_used(start, end), WILSON_FLOPS);

  free(U);
  free(output);
  free(src);
}
