// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef PRECISION_H
#define PRECISION_H
// type-generic sqrt/cabs/creal/cimag/conj -> picks the float or double variant
// from the REAL argument type, so the templated body keeps writing plain
// sqrt()/cabs()
#include <tgmath.h>

// x86 SIMD intrinsics for the vector types and NT fence below.  Gated, and the
// only place they enter the build, so a scalar build compiles on non-x86
// targets.
#if AVX2 || AVX512
#include <immintrin.h>
#endif

// ---------------------------------------------------------------------------
// Per-translation-unit precision selector.
//
// A templated body (e.g. coarse.c) is compiled twice, once with -DMG_SINGLE=1
// and once without, by the Makefile's %_f.o / %_d.o rules.  The body pulls
// REAL, the SIMD vector type, and the PFX() symbol-mangling macro from here;
// when MG_SINGLE is undefined the translation unit is built in double.
//
// Precision is chosen per TU rather than once for the whole binary, which is
// what lets the double outer solver and the single preconditioner coexist in
// one executable.  global_params.h #includes this at its end, so any TU that
// does not set MG_SINGLE (the double-only driver layer) gets a double REAL by
// default.
// ---------------------------------------------------------------------------

#if MG_SINGLE
  #define REAL float
  #define PFX(name) name##_f
#else
  #define REAL double
  #define PFX(name) name##_d
#endif

#if AVX512
  #if MG_SINGLE
    #define REAL_VEC __m512
    #define veclen 16
    #define ZERO_VEC _mm512_setzero_ps()
    #define SET1(x) _mm512_set1_ps(x)
  #else
    #define REAL_VEC __m512d
    #define veclen 8
    #define ZERO_VEC _mm512_setzero_pd()
    #define SET1(x) _mm512_set1_pd(x)
  #endif
#elif AVX2
  #if MG_SINGLE
    #define REAL_VEC __m256
    #define veclen 8
    #define ZERO_VEC _mm256_setzero_ps()
    #define SET1(x) _mm256_set1_ps(x)
  #else
    #define REAL_VEC __m256d
    #define veclen 4
    #define ZERO_VEC _mm256_setzero_pd()
    #define SET1(x) _mm256_set1_pd(x)
  #endif
#else
  #define REAL_VEC REAL
  #define veclen 1
  #define ZERO_VEC 0
  #define SET1(x) x
#endif

// Drain this thread's non-temporal stores before another thread reads them.
// The NT stores exist only in the SIMD paths, so a scalar build needs no fence
// at all.
#if AVX2 || AVX512
  #define NT_FENCE() _mm_sfence()
#else
  #define NT_FENCE() ((void) 0)
#endif

// Store one REAL_VEC bypassing the caches.  Compilers without the builtin (gcc
// accepts the call and leaves it undefined at link) drop the hint and store
// plainly.
#if (AVX2 || AVX512) && defined(__has_builtin)
  #if __has_builtin(__builtin_nontemporal_store)
    #define NT_STORE(v, p) __builtin_nontemporal_store((v), (p))
  #endif
#endif
#ifndef NT_STORE
  #define NT_STORE(v, p) (*(p) = (v))
#endif

// num_rhs_vec depends on veclen, hence on precision -> it lives here, not in
// global_params.h.  num_rhs itself (a plain count) stays in global_params.h.
#define num_rhs_vec (num_rhs / veclen)

// veclen must divide num_rhs so num_rhs_vec is a whole number of vectors
_Static_assert(num_rhs % veclen == 0, "veclen must divide num_rhs");

#endif // PRECISION_H
