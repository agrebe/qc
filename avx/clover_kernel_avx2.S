// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

// ===========================================================================
// AVX2 single-precision clover-term kernel (one site, all num_rhs columns).
//
// Authored as a hand-scheduled num_rhs=16 kernel, here turned into a cpp/GAS
// template (not a Python generator -- cpp + as do the arithmetic):
//   * num_rhs arrives as -Dnum_rhs=<N> (the Makefile extracts it from
//     global_params.h and pipes this file through cpp into as).
//   * S(D) rescales a num_rhs=16 byte displacement to the actual num_rhs.
//     S(D)==D at num_rhs=16, so the machine code is byte-identical to the
//     original hand-written kernel; for larger num_rhs the component/imag
//     strides scale while the 32-byte 8-column-chunk stride stays fixed.
//   * num_rhs>16 runs the body once per 16-column tile (num_rhs/16 tiles),
//     reusing the exact 2-chunk schedule with the bases stepped by 64 B and F
//     re-read each tile.  Requires num_rhs % 16 == 0 (caller falls back to C).
// ===========================================================================
#include "global_params.h"   // num_rhs/Nc/Ns;
                             // C-only tail is __ASSEMBLER__-guarded
#if (num_rhs % 16) != 0
#error "clover_kernel_avx2.S needs num_rhs % 16 == 0 (2-chunk tile granularity)"
#endif
// component stride 8*num_rhs, imag stride 4*num_rhs, chunk stride 32 (fixed).
#define S(D)  D/128*8*num_rhs + D%128/64*4*num_rhs + D%64

// CLOVER_OW=1 assembles the same schedule as an overwrite kernel
// (output = F*ferm instead of output += F*ferm), symbol clover_kernel_avx2_ow
// (a second .o built from this file -- see the Makefile rule).  Each output
// accumulator's first load (the only one that would read the caller's data;
// every later load of the same slot rereads a partial sum this kernel stored)
// becomes a register zero, so the caller's "zero the site, then accumulate"
// pattern collapses into the kernel.  FMA into a zeroed register is the same
// arithmetic as FMA into a loaded zero, so this is bit-identical to that
// pattern.  Output must not alias ferm.
#if CLOVER_OW
#define KERNEL_NAME clover_kernel_avx2_ow
#define OUT_FIRST(MEM, REG) vxorps REG, REG, REG
#else
#define KERNEL_NAME clover_kernel_avx2
#define OUT_FIRST(MEM, REG) vmovaps MEM, REG
#endif

    .globl KERNEL_NAME
    .type KERNEL_NAME, @function
    # arguments:
    # float * output:  %rdi
    # float * ferm:    %rsi
    # float * F:       %rdx
    # all of output, ferm, and F have been advanced to right position
    # single precision; num_rhs is a -D define, any multiple of 16
KERNEL_NAME:
    .cfi_startproc

#if num_rhs > 16
    # num_rhs/16 tiles of 16 columns each; r10/r11 = output/ferm tile base,
    # r8 = F base, r9 = running byte offset along the column dimension.
    mov %rdi, %r10
    mov %rsi, %r11
    mov %rdx, %r8
    xor %r9, %r9
tile_loop:
    lea (%r10,%r9,1), %rdi
    lea (%r11,%r9,1), %rsi
    mov %r8, %rdx
#endif

    # zero out %rax
    xor %rax, %rax
multiply_block:
    # extract first 2x2 block from F and broadcast to register
    vbroadcastss (%rdx), %ymm0
    vbroadcastss 4(%rdx), %ymm1
    vbroadcastss 24(%rdx), %ymm2
    vbroadcastss 28(%rdx), %ymm3
    # load first two spin-color components of output vector
    # we have 2 RHS vecs (= 16 RHS) * 2 complex components per spin-color
    # so this is 8 vectors total
    # output loads (first touch of rows 0-1 -> OUT_FIRST: zeroed when CLOVER_OW)
    OUT_FIRST((%rdi),    %ymm4)  # re(output[0][0])
    OUT_FIRST(S(32)(%rdi),  %ymm5)  # re(output[0][1])
    OUT_FIRST(S(64)(%rdi),  %ymm6)  # im(output[0][0])
    OUT_FIRST(S(96)(%rdi),  %ymm7)  # im(output[0][1])
    OUT_FIRST(S(128)(%rdi), %ymm8)  # re(output[1][0])
    OUT_FIRST(S(160)(%rdi), %ymm9)  # re(output[1][1])
    OUT_FIRST(S(192)(%rdi), %ymm10) # im(output[1][0])
    OUT_FIRST(S(224)(%rdi), %ymm11) # im(output[1][1])
    # load the first spin-color component of the input
    # ferm loads
    vmovaps (%rsi),    %ymm12 # re(ferm[0][0])
    vmovaps S(32)(%rsi),  %ymm13 # re(ferm[0][1])
    vmovaps S(64)(%rsi),  %ymm14 # im(ferm[0][0])
    vmovaps S(96)(%rsi),  %ymm15 # im(ferm[0][1])

    # output[1] += ferm[0] * F[0][1]
    vfmadd231ps %ymm12, %ymm1, %ymm8  # re(output[1][0])
    vfmadd231ps %ymm13, %ymm1, %ymm9  # re(output[1][1])
    vfmadd231ps %ymm14, %ymm1, %ymm10 # im(output[1][0])
    vfmadd231ps %ymm15, %ymm1, %ymm11 # im(output[1][1])

    # output[0] += ferm[0] * F[0][0]
    vfmadd231ps %ymm12, %ymm0, %ymm4  # re(output[0][0])
    vfmadd231ps %ymm13, %ymm0, %ymm5  # re(output[0][1])
    vfmadd231ps %ymm14, %ymm0, %ymm6  # im(output[0][0])
    vfmadd231ps %ymm15, %ymm0, %ymm7  # im(output[0][1])

    # output[1] -= ferm[0] * F[1][0] (= -Im(F_10))
    # after this, we are finished with ferm[0]
    # replace this with ferm[1] (interleaved here)
    vfmadd231ps  %ymm14, %ymm2, %ymm8  # re(output[1][0])
    vmovaps S(192)(%rsi), %ymm14 # im(ferm[1][0])
    vfmadd231ps  %ymm15, %ymm2, %ymm9  # re(output[1][1])
    vmovaps S(224)(%rsi), %ymm15 # im(ferm[1][1])
    vfnmadd231ps %ymm12, %ymm2, %ymm10 # im(output[1][0])
    vmovaps S(128)(%rsi), %ymm12 # re(ferm[1][0])
    vfnmadd231ps %ymm13, %ymm2, %ymm11 # im(output[1][1])
    vmovaps S(160)(%rsi), %ymm13 # re(ferm[1][1])

    # output[0] += ferm[1] * F[0][1]
    vfmadd231ps %ymm14, %ymm1, %ymm6  # im(output[0][0])
    vfmadd231ps %ymm15, %ymm1, %ymm7  # im(output[0][1])
    vfmadd231ps %ymm12, %ymm1, %ymm4  # re(output[0][0])
    vfmadd231ps %ymm13, %ymm1, %ymm5  # re(output[0][1])

    # replace F[0][0] with F[1][2]
    vbroadcastss 32(%rdx), %ymm0
    # replace F[1][0] with F[2][1]
    vbroadcastss 52(%rdx), %ymm1

    # output[1] += ferm[1] * F[1][1]
    vfmadd231ps %ymm12, %ymm3, %ymm8  # re(output[1][0])
    vfmadd231ps %ymm13, %ymm3, %ymm9  # re(output[1][1])
    vfmadd231ps %ymm14, %ymm3, %ymm10 # im(output[1][0])
    vfmadd231ps %ymm15, %ymm3, %ymm11 # im(output[1][1])

    # replace F[1][1] with ferm[2][1]
    vmovaps S(288)(%rsi), %ymm3 # re(ferm[2][1])

    # output[0] += ferm[1] * F[1][0] (= +Im(F_01))
    # store output[0] back to memory
    # replace with output[2] (first touch -> OUT_FIRST)
    vfnmadd231ps %ymm14, %ymm2, %ymm4 # re(output[0][0])
    vmovaps %ymm4,  (%rdi)
    OUT_FIRST(S(256)(%rdi), %ymm4)
    vfnmadd231ps %ymm15, %ymm2, %ymm5 # re(output[0][1])
    vmovaps %ymm5,  S(32)(%rdi)
    OUT_FIRST(S(288)(%rdi), %ymm5)
    vfmadd231ps  %ymm12, %ymm2, %ymm6 # im(output[0][0])
    vmovaps %ymm6,  S(64)(%rdi)
    OUT_FIRST(S(320)(%rdi), %ymm6)
    vfmadd231ps  %ymm13, %ymm2, %ymm7 # im(output[0][1])
    vmovaps %ymm7,  S(96)(%rdi)
    OUT_FIRST(S(352)(%rdi), %ymm7)

    # replace F[1][0] with ferm[2][0]
    vmovaps S(256)(%rsi), %ymm2 # re(ferm[2][0])

    # output[1] += re(ferm[2]) * F[1][2]
    # output[1] += re(ferm[2]) * F[2][1] (= +Im(F_12))
    # load imaginary parts of ferm[2] when done with this
    vfmadd231ps %ymm3, %ymm0, %ymm9  # re(output[1][1])
    vfmadd231ps %ymm3, %ymm1, %ymm11 # im(output[1][1])
    vmovaps S(352)(%rsi), %ymm3 # im(ferm[2][1])
    vfmadd231ps %ymm2, %ymm0, %ymm8  # re(output[1][0])
    vfmadd231ps %ymm2, %ymm1, %ymm10 # im(output[1][0])
    vmovaps S(320)(%rsi), %ymm2 # im(ferm[2][0])

    # output[2] += re(ferm[1]) * F[1][2]
    # output[2] -= re(ferm[1]) * F[2][1] (= -Im(F_12))
    vfmadd231ps  %ymm13, %ymm0, %ymm5 # re(output[2][1])
    vfnmadd231ps %ymm13, %ymm1, %ymm7 # im(output[2][1])
    vfmadd231ps  %ymm12, %ymm0, %ymm4 # re(output[2][0])
    vfnmadd231ps %ymm12, %ymm1, %ymm6 # im(output[2][0])
    
    # now repeat with imaginary part of ferm[2] and ferm[1]
    # replace ferm[2] with F[1][3]
    vfnmadd231ps %ymm3, %ymm1, %ymm9  # re(output[1][1])
    vfmadd231ps  %ymm3, %ymm0, %ymm11 # im(output[1][1])
    vbroadcastss 76(%rdx), %ymm3
    vfnmadd231ps %ymm2, %ymm1, %ymm8  # re(output[1][0])
    vfmadd231ps  %ymm2, %ymm0, %ymm10 # im(output[1][0])
    vbroadcastss 36(%rdx), %ymm2

    # store output[2] back to memory when finished
    # replace with output[3] (first touch -> OUT_FIRST)
    # replace F[1][2] with re(ferm[3])
    vfmadd231ps %ymm15, %ymm1, %ymm5 # re(output[2][1])
    vmovaps %ymm5, S(288)(%rdi)
    OUT_FIRST(S(416)(%rdi), %ymm5)
    vfmadd231ps %ymm15, %ymm0, %ymm7 # im(output[2][1])
    vmovaps %ymm7, S(352)(%rdi)
    OUT_FIRST(S(480)(%rdi), %ymm7)
    vfmadd231ps %ymm14, %ymm1, %ymm4 # re(output[2][0])
    vmovaps S(416)(%rsi), %ymm1 # re(ferm[3][1])
    vmovaps %ymm4, S(256)(%rdi)
    OUT_FIRST(S(384)(%rdi), %ymm4)
    vfmadd231ps %ymm14, %ymm0, %ymm6 # im(output[2][0])
    vmovaps S(384)(%rsi), %ymm0 # re(ferm[3][0])
    vmovaps %ymm6, S(320)(%rdi)
    OUT_FIRST(S(448)(%rdi), %ymm6)

    # output[1] += re(ferm[3]) * F[1][3]
    # output[1] += re(ferm[3]) * F[3][1] (= +Im(F_13))
    # load imaginary parts of ferm[3] when done with this
    vfmadd231ps %ymm1, %ymm2, %ymm9  # re(output[1][1])
    vfmadd231ps %ymm1, %ymm3, %ymm11 # im(output[1][1])
    vmovaps S(480)(%rsi), %ymm1 # im(ferm[3][1])
    vfmadd231ps %ymm0, %ymm2, %ymm8  # re(output[1][0])
    vfmadd231ps %ymm0, %ymm3, %ymm10 # im(output[1][0])
    vmovaps S(448)(%rsi), %ymm0 # im(ferm[3][0])

    # output[3] += re(ferm[1]) * F[1][3]
    # output[3] -= re(ferm[1]) * F[3][1] (= -Im(F_13))
    vfmadd231ps  %ymm13, %ymm2, %ymm5 # re(output[3][1])
    vfnmadd231ps %ymm13, %ymm3, %ymm7 # im(output[3][1])
    vfmadd231ps  %ymm12, %ymm2, %ymm4 # re(output[3][0])
    vfnmadd231ps %ymm12, %ymm3, %ymm6 # im(output[3][0])

    # now repeat with imaginary part of ferm[3] and ferm[1]
    # replace ferm[3] with F[1][4]
    vfnmadd231ps %ymm1, %ymm3, %ymm9  # re(output[1][1])
    vfmadd231ps  %ymm1, %ymm2, %ymm11 # im(output[1][1])
    vbroadcastss 100(%rdx), %ymm1
    vfnmadd231ps %ymm0, %ymm3, %ymm8  # re(output[1][0])
    vfmadd231ps  %ymm0, %ymm2, %ymm10 # im(output[1][0])
    vbroadcastss 40(%rdx), %ymm0

    # store output[3] back to memory when finished
    # replace with output[4] (first touch -> OUT_FIRST)
    # replace F[1][3] with re(ferm[4])
    vfmadd231ps %ymm15, %ymm3, %ymm5 # re(output[3][1])
    vmovaps %ymm5, S(416)(%rdi)
    OUT_FIRST(S(544)(%rdi), %ymm5)
    vfmadd231ps %ymm15, %ymm2, %ymm7 # im(output[3][1])
    vmovaps %ymm7, S(480)(%rdi)
    OUT_FIRST(S(608)(%rdi), %ymm7)
    vfmadd231ps %ymm14, %ymm3, %ymm4 # re(output[3][0])
    vmovaps S(544)(%rsi), %ymm3 # re(ferm[4][1])
    vmovaps %ymm4, S(384)(%rdi)
    OUT_FIRST(S(512)(%rdi), %ymm4)
    vfmadd231ps %ymm14, %ymm2, %ymm6 # im(output[3][0])
    vmovaps S(512)(%rsi), %ymm2 # re(ferm[4][0])
    vmovaps %ymm6, S(448)(%rdi)
    OUT_FIRST(S(576)(%rdi), %ymm6)

    # output[1] += re(ferm[4]) * F[1][4]
    # output[1] += re(ferm[4]) * F[4][1] (= +Im(F_14))
    # load imaginary parts of ferm[4] when done with this
    vfmadd231ps %ymm3, %ymm0, %ymm9  # re(output[1][1])
    vfmadd231ps %ymm3, %ymm1, %ymm11 # im(output[1][1])
    vmovaps S(608)(%rsi), %ymm3 # im(ferm[4][1])
    vfmadd231ps %ymm2, %ymm0, %ymm8  # re(output[1][0])
    vfmadd231ps %ymm2, %ymm1, %ymm10 # im(output[1][0])
    vmovaps S(576)(%rsi), %ymm2 # im(ferm[4][0])

    # output[4] += re(ferm[1]) * F[1][4]
    # output[4] -= re(ferm[1]) * F[4][1] (= -Im(F_14))
    vfmadd231ps  %ymm13, %ymm0, %ymm5 # re(output[4][1])
    vfnmadd231ps %ymm13, %ymm1, %ymm7 # im(output[4][1])
    vfmadd231ps  %ymm12, %ymm0, %ymm4 # re(output[4][0])
    vfnmadd231ps %ymm12, %ymm1, %ymm6 # im(output[4][0])

    # now repeat with imaginary part of ferm[4] and ferm[1]
    # replace ferm[4] with F[1][5]
    vfnmadd231ps %ymm3, %ymm1, %ymm9  # re(output[1][1])
    vfmadd231ps  %ymm3, %ymm0, %ymm11 # im(output[1][1])
    vbroadcastss 124(%rdx), %ymm3
    vfnmadd231ps %ymm2, %ymm1, %ymm8  # re(output[1][0])
    vfmadd231ps  %ymm2, %ymm0, %ymm10 # im(output[1][0])
    vbroadcastss 44(%rdx), %ymm2

    # store output[4] back to memory when finished
    # replace with output[5] (first touch -> OUT_FIRST)
    # replace F[1][4] with re(ferm[5])
    vfmadd231ps %ymm15, %ymm1, %ymm5 # re(output[4][1])
    vmovaps %ymm5, S(544)(%rdi)
    OUT_FIRST(S(672)(%rdi), %ymm5)
    vfmadd231ps %ymm15, %ymm0, %ymm7 # im(output[4][1])
    vmovaps %ymm7, S(608)(%rdi)
    OUT_FIRST(S(736)(%rdi), %ymm7)
    vfmadd231ps %ymm14, %ymm1, %ymm4 # re(output[4][0])
    vmovaps S(672)(%rsi), %ymm1 # re(ferm[5][1])
    vmovaps %ymm4, S(512)(%rdi)
    OUT_FIRST(S(640)(%rdi), %ymm4)
    vfmadd231ps %ymm14, %ymm0, %ymm6 # im(output[4][0])
    vmovaps S(640)(%rsi), %ymm0 # re(ferm[5][0])
    vmovaps %ymm6, S(576)(%rdi)
    OUT_FIRST(S(704)(%rdi), %ymm6)

    # output[1] += re(ferm[5]) * F[1][5]
    # output[1] += re(ferm[5]) * F[5][1] (= +Im(F_15))
    # load imaginary parts of ferm[4] when done with this
    vfmadd231ps %ymm1, %ymm2, %ymm9  # re(output[1][1])
    vfmadd231ps %ymm1, %ymm3, %ymm11 # im(output[1][1])
    vmovaps S(736)(%rsi), %ymm1 # im(ferm[5][1])
    vfmadd231ps %ymm0, %ymm2, %ymm8  # re(output[1][0])
    vfmadd231ps %ymm0, %ymm3, %ymm10 # im(output[1][0])
    vmovaps S(704)(%rsi), %ymm0 # im(ferm[5][0])

    # output[5] += re(ferm[1]) * F[1][5]
    # output[5] -= re(ferm[1]) * F[5][1] (= -Im(F_15))
    # replace re(ferm[1]) with F[0][5]
    vfmadd231ps  %ymm13, %ymm2, %ymm5 # re(output[5][1])
    vfnmadd231ps %ymm13, %ymm3, %ymm7 # im(output[5][1])
    vbroadcastss 120(%rdx), %ymm13
    vfmadd231ps  %ymm12, %ymm2, %ymm4 # re(output[5][0])
    vfnmadd231ps %ymm12, %ymm3, %ymm6 # im(output[5][0])
    vbroadcastss 20(%rdx), %ymm12

    # now repeat with imaginary part of ferm[5] and ferm[1]
    # store output[1] back to memory
    # read output[0] from memory
    vfnmadd231ps %ymm1, %ymm3, %ymm9  # re(output[1][1])
    vmovaps %ymm9,  S(160)(%rdi)
    vmovaps S(32)(%rdi), %ymm9
    vfmadd231ps  %ymm1, %ymm2, %ymm11 # im(output[1][1])
    vmovaps %ymm11, S(224)(%rdi)
    vmovaps S(96)(%rdi), %ymm11
    vfnmadd231ps %ymm0, %ymm3, %ymm8  # re(output[1][0])
    vmovaps %ymm8,  S(128)(%rdi)
    vmovaps (%rdi), %ymm8
    vfmadd231ps  %ymm0, %ymm2, %ymm10 # im(output[1][0])
    vmovaps %ymm10, S(192)(%rdi)
    vmovaps S(64)(%rdi), %ymm10

    # replace imaginary part of ferm[1] with ferm[0]
    # also replace F[1][5] with re(ferm[0])
    vfmadd231ps %ymm15, %ymm3, %ymm5 # re(output[5][1])
    vfmadd231ps %ymm15, %ymm2, %ymm7 # im(output[5][1])
    vmovaps S(96)(%rsi), %ymm15 # im(ferm[0][1])
    vfmadd231ps %ymm14, %ymm3, %ymm4 # re(output[5][0])
    vmovaps S(32)(%rsi), %ymm3 # re(ferm[0][1])
    vfmadd231ps %ymm14, %ymm2, %ymm6 # im(output[5][0])
    vmovaps S(64)(%rsi), %ymm14 # im(ferm[0][0])
    vmovaps (%rsi), %ymm2 # re(ferm[0][0])
    # note that ferm[0] is now stored discontiguously
    # real part is in registers 2, 3
    # imag part is in registers 14, 15
    # F[0][5] is stored in registers 12, 13
    # imag part of ferm[5] is in registers 0, 1
    # output[0] is in registers 8-11
    # output[5] is in registers 4-7

    # output[0] += im(ferm[5]) * F[0][5]
    # replace im(ferm[5]) with re(ferm[5])
    vfnmadd231ps %ymm1, %ymm13, %ymm9  # re(output[0][1])
    vfmadd231ps  %ymm1, %ymm12, %ymm11 # im(output[0][1])
    vmovaps S(672)(%rsi), %ymm1 # re(ferm[5][1])
    vfnmadd231ps %ymm0, %ymm13, %ymm8  # re(output[0][0])
    vfmadd231ps  %ymm0, %ymm12, %ymm10 # im(output[0][0])
    vmovaps S(640)(%rsi), %ymm0 # re(ferm[5][0])

    # output[5] += im(ferm[0]) * F[0][5]
    vfmadd231ps %ymm15, %ymm13, %ymm5 # re(output[5][1])
    vfmadd231ps %ymm15, %ymm12, %ymm7 # im(output[5][1])
    vfmadd231ps %ymm14, %ymm13, %ymm4 # re(output[5][0])
    vfmadd231ps %ymm14, %ymm12, %ymm6 # im(output[5][0])

    # output[0] += re(ferm[5]) * F[0][5]
    # replace re(ferm[5]) with F[0][4]
    vfmadd231ps %ymm1, %ymm12, %ymm9  # re(output[0][1])
    vfmadd231ps %ymm1, %ymm13, %ymm11 # im(output[0][1])
    vbroadcastss 96(%rdx), %ymm1
    vfmadd231ps %ymm0, %ymm12, %ymm8  # re(output[0][0])
    vfmadd231ps %ymm0, %ymm13, %ymm10 # im(output[0][0])
    vbroadcastss 16(%rdx), %ymm0

    # output[5] += re(ferm[0]) * F[0][5]
    # store output[5] to memory
    # read output[4] from memory
    # replace F[0][5] with re(ferm[4])
    vfmadd231ps  %ymm3, %ymm12, %ymm5 # re(output[5][1])
    vmovaps %ymm5, S(672)(%rdi)
    vmovaps S(544)(%rdi), %ymm5
    vfnmadd231ps %ymm3, %ymm13, %ymm7 # im(output[5][1])
    vmovaps %ymm7, S(736)(%rdi)
    vmovaps S(608)(%rdi), %ymm7
    vfmadd231ps  %ymm2, %ymm12, %ymm4 # re(output[5][0])
    vmovaps S(512)(%rsi), %ymm12 # re(ferm[4][0])
    vmovaps %ymm4, S(640)(%rdi)
    vmovaps S(512)(%rdi), %ymm4
    vfnmadd231ps %ymm2, %ymm13, %ymm6 # im(output[5][0])
    vmovaps S(544)(%rsi), %ymm13 # re(ferm[4][1])
    vmovaps %ymm6, S(704)(%rdi)
    vmovaps S(576)(%rdi), %ymm6

    # output[0] += re(ferm[4]) * F[0][4]
    # output[0] += re(ferm[4]) * F[4][0] (= +Im(F_04))
    # load imaginary parts of ferm[4] when done with this
    vfmadd231ps %ymm13, %ymm0, %ymm9  # re(output[0][1])
    vfmadd231ps %ymm13, %ymm1, %ymm11 # im(output[0][1])
    vmovaps S(608)(%rsi), %ymm13 # im(ferm[4][1])
    vfmadd231ps %ymm12, %ymm0, %ymm8  # re(output[0][0])
    vfmadd231ps %ymm12, %ymm1, %ymm10 # im(output[0][0])
    vmovaps S(576)(%rsi), %ymm12 # im(ferm[4][0])

    # output[4] += re(ferm[0]) * F[0][4]
    # output[4] -= re(ferm[0]) * F[4][0] (= -Im(F_04))
    vfmadd231ps  %ymm3, %ymm0, %ymm5 # re(output[4][1])
    vfnmadd231ps %ymm3, %ymm1, %ymm7 # im(output[4][1])
    vfmadd231ps  %ymm2, %ymm0, %ymm4 # re(output[4][0])
    vfnmadd231ps %ymm2, %ymm1, %ymm6 # im(output[4][0])

    # now repeat with imaginary part of ferm[4] and ferm[0]
    # replace ferm[4] with F[0][3]
    vfnmadd231ps %ymm13, %ymm1, %ymm9  # re(output[0][1])
    vfmadd231ps  %ymm13, %ymm0, %ymm11 # im(output[0][1])
    vbroadcastss 72(%rdx), %ymm13
    vfnmadd231ps %ymm12, %ymm1, %ymm8  # re(output[0][0])
    vfmadd231ps  %ymm12, %ymm0, %ymm10 # im(output[0][0])
    vbroadcastss 12(%rdx), %ymm12

    # store output[4] back to memory when finished
    # replace with output[3]
    # replace F[0][4] with re(ferm[3])
    vfmadd231ps %ymm15, %ymm1, %ymm5 # re(output[4][1])
    vmovaps %ymm5, S(544)(%rdi)
    vmovaps S(416)(%rdi), %ymm5
    vfmadd231ps %ymm15, %ymm0, %ymm7 # im(output[4][1])
    vmovaps %ymm7, S(608)(%rdi)
    vmovaps S(480)(%rdi), %ymm7
    vfmadd231ps %ymm14, %ymm1, %ymm4 # re(output[4][0])
    vmovaps S(416)(%rsi), %ymm1 # re(ferm[3][1])
    vmovaps %ymm4, S(512)(%rdi)
    vmovaps S(384)(%rdi), %ymm4
    vfmadd231ps %ymm14, %ymm0, %ymm6 # im(output[4][0])
    vmovaps S(384)(%rsi), %ymm0 # re(ferm[3][0])
    vmovaps %ymm6, S(576)(%rdi)
    vmovaps S(448)(%rdi), %ymm6

    # output[0] += re(ferm[3]) * F[0][3]
    # output[0] += re(ferm[3]) * F[3][0] (= +Im(F_03))
    # load imaginary parts of ferm[3] when done with this
    vfmadd231ps %ymm1, %ymm12, %ymm9  # re(output[0][1])
    vfmadd231ps %ymm1, %ymm13, %ymm11 # im(output[0][1])
    vmovaps S(480)(%rsi), %ymm1 # im(ferm[3][1])
    vfmadd231ps %ymm0, %ymm12, %ymm8  # re(output[0][0])
    vfmadd231ps %ymm0, %ymm13, %ymm10 # im(output[0][0])
    vmovaps S(448)(%rsi), %ymm0 # im(ferm[3][0])

    # output[3] += re(ferm[0]) * F[0][3]
    # output[3] -= re(ferm[0]) * F[3][0] (= -Im(F_03))
    vfmadd231ps  %ymm3, %ymm12, %ymm5 # re(output[3][1])
    vfnmadd231ps %ymm3, %ymm13, %ymm7 # im(output[3][1])
    vfmadd231ps  %ymm2, %ymm12, %ymm4 # re(output[3][0])
    vfnmadd231ps %ymm2, %ymm13, %ymm6 # im(output[3][0])

    # now repeat with imaginary part of ferm[3] and ferm[0]
    # replace ferm[3] with F[0][2]
    vfnmadd231ps %ymm1, %ymm13, %ymm9  # re(output[0][1])
    vfmadd231ps  %ymm1, %ymm12, %ymm11 # im(output[0][1])
    vbroadcastss 48(%rdx), %ymm1
    vfnmadd231ps %ymm0, %ymm13, %ymm8  # re(output[0][0])
    vfmadd231ps  %ymm0, %ymm12, %ymm10 # im(output[0][0])
    vbroadcastss 8(%rdx), %ymm0

    # store output[3] back to memory when finished
    # replace with output[2]
    # replace F[0][3] with re(ferm[2])
    vfmadd231ps %ymm15, %ymm13, %ymm5 # re(output[3][1])
    vmovaps %ymm5, S(416)(%rdi)
    vmovaps S(288)(%rdi), %ymm5
    vfmadd231ps %ymm15, %ymm12, %ymm7 # im(output[3][1])
    vmovaps %ymm7, S(480)(%rdi)
    vmovaps S(352)(%rdi), %ymm7
    vfmadd231ps %ymm14, %ymm13, %ymm4 # re(output[3][0])
    vmovaps S(288)(%rsi), %ymm13 # re(ferm[2][1])
    vmovaps %ymm4, S(384)(%rdi)
    vmovaps S(256)(%rdi), %ymm4
    vfmadd231ps %ymm14, %ymm12, %ymm6 # im(output[3][0])
    vmovaps S(256)(%rsi), %ymm12 # re(ferm[2][0])
    vmovaps %ymm6, S(448)(%rdi)
    vmovaps S(320)(%rdi), %ymm6

    # output[0] += re(ferm[2]) * F[0][2]
    # output[0] += re(ferm[2]) * F[2][0] (= +Im(F_02))
    # load imaginary parts of ferm[2] when done with this
    vfmadd231ps %ymm13, %ymm0, %ymm9  # re(output[0][1])
    vfmadd231ps %ymm13, %ymm1, %ymm11 # im(output[0][1])
    vmovaps S(352)(%rsi), %ymm13 # im(ferm[2][1])
    vfmadd231ps %ymm12, %ymm0, %ymm8  # re(output[0][0])
    vfmadd231ps %ymm12, %ymm1, %ymm10 # im(output[0][0])
    vmovaps S(320)(%rsi), %ymm12 # im(ferm[2][0])

    # output[2] += re(ferm[0]) * F[0][2]
    # output[2] -= re(ferm[0]) * F[2][0] (= -Im(F_02))
    # replace re(ferm[0]) with F[2][2] and F[2][3]
    vfmadd231ps  %ymm3, %ymm0, %ymm5 # re(output[2][1])
    vfnmadd231ps %ymm3, %ymm1, %ymm7 # im(output[2][1])
    vbroadcastss 60(%rdx), %ymm3  # F[2][3]
    vfmadd231ps  %ymm2, %ymm0, %ymm4 # re(output[2][0])
    vfnmadd231ps %ymm2, %ymm1, %ymm6 # im(output[2][0])
    vbroadcastss 56(%rdx), %ymm2  # F[2][2]

    # now repeat with imaginary part of ferm[2] and ferm[0]
    # write output[0] to memory
    # replace with output[3]
    vfnmadd231ps %ymm13, %ymm1, %ymm9  # re(output[0][1])
    vmovaps %ymm9,  S(32)(%rdi)
    vmovaps S(416)(%rdi), %ymm9
    vfmadd231ps  %ymm13, %ymm0, %ymm11 # im(output[0][1])
    vmovaps %ymm11, S(96)(%rdi)
    vmovaps S(480)(%rdi), %ymm11
    vfnmadd231ps %ymm12, %ymm1, %ymm8  # re(output[0][0])
    vmovaps %ymm8,  (%rdi)
    vmovaps S(384)(%rdi), %ymm8
    vfmadd231ps  %ymm12, %ymm0, %ymm10 # im(output[0][0])
    vmovaps %ymm10, S(64)(%rdi)
    vmovaps S(448)(%rdi), %ymm10

    # replace im(ferm[0]) with re(ferm[2])
    # note that this means im(ferm[2]) is after re(ferm[2])
    # replace F[2][0] and F[0][2] with F[3][2] and F[3][3]
    vfmadd231ps %ymm15, %ymm1, %ymm5 # re(output[2][1])
    vfmadd231ps %ymm15, %ymm0, %ymm7 # im(output[2][1])
    vmovaps S(288)(%rsi), %ymm15 # re(ferm[2][1])
    vfmadd231ps %ymm14, %ymm1, %ymm4 # re(output[2][0])
    vbroadcastss 84(%rdx), %ymm1  # F[3][3]
    vfmadd231ps %ymm14, %ymm0, %ymm6 # im(output[2][0])
    vmovaps S(256)(%rsi), %ymm14 # re(ferm[2][0])
    vbroadcastss 80(%rdx), %ymm0  # F[3][2]

    # now we have
    # output[3]: 8-11
    # output[2]: 4-7
    # ferm[2]: 14-15, 12-13 (imag part first)
    # F[2][2-3]: 2-3
    # F[3][2-3]: 0-1

    # output[3] += ferm[2] * F[2][3]
    vfmadd231ps %ymm15, %ymm3, %ymm9  # re(output[3][1])
    vfmadd231ps %ymm13, %ymm3, %ymm11 # im(output[3][1])
    vfmadd231ps %ymm14, %ymm3, %ymm8  # re(output[3][0])
    vfmadd231ps %ymm12, %ymm3, %ymm10 # im(output[3][0])

    # output[2] += ferm[2] * F[2][2]
    vfmadd231ps %ymm15, %ymm2, %ymm5 # re(output[2][1])
    vfmadd231ps %ymm13, %ymm2, %ymm7 # im(output[2][1])
    vfmadd231ps %ymm14, %ymm2, %ymm4 # re(output[2][0])
    vfmadd231ps %ymm12, %ymm2, %ymm6 # im(output[2][0])

    # output[3] -= ferm[2] * F[3][2] (= -Im(F_32))
    # after this, we are finished with ferm[2]
    # replace this with ferm[3] (interleaved here)
    # we will take this opportunity to reorder ferm[3] (real part first)
    vfmadd231ps  %ymm13, %ymm0, %ymm9  # re(output[3][1])
    vmovaps S(416)(%rsi), %ymm13 # re(ferm[3][1])
    vfnmadd231ps %ymm15, %ymm0, %ymm11 # im(output[3][1])
    vmovaps S(480)(%rsi), %ymm15 # im(ferm[3][1])
    vfmadd231ps  %ymm12, %ymm0, %ymm8  # re(output[3][0])
    vmovaps S(384)(%rsi), %ymm12 # re(ferm[3][0])
    vfnmadd231ps %ymm14, %ymm0, %ymm10 # im(output[3][0])
    vmovaps S(448)(%rsi), %ymm14 # im(ferm[3][0])

    # output[2] += ferm[3] * F[2][3]
    vfmadd231ps %ymm13, %ymm3, %ymm5  # re(output[2][1])
    vfmadd231ps %ymm15, %ymm3, %ymm7  # im(output[2][1])
    vfmadd231ps %ymm12, %ymm3, %ymm4  # re(output[2][0])
    vfmadd231ps %ymm14, %ymm3, %ymm6  # im(output[2][0])

    # replace F[2][2] and F[2][3] with F[3][4] and F[4][3]
    vbroadcastss 88(%rdx), %ymm2
    vbroadcastss 108(%rdx), %ymm3

    # output[3] += ferm[3] * F[3][3]
    vfmadd231ps %ymm13, %ymm1, %ymm9  # re(output[3][1])
    vfmadd231ps %ymm15, %ymm1, %ymm11 # im(output[3][1])
    vfmadd231ps %ymm12, %ymm1, %ymm8  # re(output[3][0])
    vfmadd231ps %ymm14, %ymm1, %ymm10 # im(output[3][0])

    # replace F[3][3] with ferm[4][1]
    vmovaps S(544)(%rsi), %ymm1 # re(ferm[4][1])

    # output[2] += ferm[3] * F[3][2] (= +Im(F_23))
    # store output[2] back to memory
    # replace with output[4]
    vfnmadd231ps %ymm15, %ymm0, %ymm5  # re(output[2][1])
    vmovaps %ymm5, S(288)(%rdi)
    vmovaps S(544)(%rdi), %ymm5
    vfmadd231ps  %ymm13, %ymm0, %ymm7  # im(output[2][1])
    vmovaps %ymm7, S(352)(%rdi)
    vmovaps S(608)(%rdi), %ymm7
    vfnmadd231ps %ymm14, %ymm0, %ymm4  # re(output[2][0])
    vmovaps %ymm4, S(256)(%rdi)
    vmovaps S(512)(%rdi), %ymm4
    vfmadd231ps  %ymm12, %ymm0, %ymm6  # im(output[2][0])
    vmovaps %ymm6, S(320)(%rdi)
    vmovaps S(576)(%rdi), %ymm6

    # replace F[3][2] with ferm[4][0]
    vmovaps S(512)(%rsi), %ymm0 # re(ferm[4][0])

    # output[3] += re(ferm[4]) * F[3][4]
    # output[3] += re(ferm[4]) * F[4][3] (= +Im(F_34))
    # load imaginary parts of ferm[2] when done with this
    vfmadd231ps %ymm1, %ymm2, %ymm9  # re(output[3][1])
    vfmadd231ps %ymm1, %ymm3, %ymm11 # im(output[3][1])
    vmovaps S(608)(%rsi), %ymm1 # im(ferm[4][1])
    vfmadd231ps %ymm0, %ymm2, %ymm8  # re(output[3][0])
    vfmadd231ps %ymm0, %ymm3, %ymm10 # im(output[3][0])
    vmovaps S(576)(%rsi), %ymm0 # im(ferm[4][0])

    # output[4] += re(ferm[3]) * F[3][4]
    # output[4] -= re(ferm[3]) * F[4][3] (= -Im(F_34))
    vfmadd231ps  %ymm13, %ymm2, %ymm5 # re(output[4][1])
    vfnmadd231ps %ymm13, %ymm3, %ymm7 # im(output[4][1])
    vfmadd231ps  %ymm12, %ymm2, %ymm4 # re(output[4][0])
    vfnmadd231ps %ymm12, %ymm3, %ymm6 # im(output[4][0])
    
    # now repeat with imaginary part of ferm[4] and ferm[3]
    # replace ferm[4] with F[3][5]
    vfnmadd231ps %ymm1, %ymm3, %ymm9  # re(output[3][1])
    vfmadd231ps  %ymm1, %ymm2, %ymm11 # im(output[3][1])
    vbroadcastss 132(%rdx), %ymm1
    vfnmadd231ps %ymm0, %ymm3, %ymm8  # re(output[3][0])
    vfmadd231ps  %ymm0, %ymm2, %ymm10 # im(output[3][0])
    vbroadcastss 92(%rdx), %ymm0

    # store output[4] back to memory when finished
    # replace with output[5]
    # replace F[3][4] with re(ferm[5])
    vfmadd231ps %ymm15, %ymm3, %ymm5 # re(output[4][1])
    vmovaps %ymm5, S(544)(%rdi)
    vmovaps S(672)(%rdi), %ymm5
    vfmadd231ps %ymm15, %ymm2, %ymm7 # im(output[4][1])
    vmovaps %ymm7, S(608)(%rdi)
    vmovaps S(736)(%rdi), %ymm7
    vfmadd231ps %ymm14, %ymm3, %ymm4 # re(output[4][0])
    vmovaps S(672)(%rsi), %ymm3 # re(ferm[5][1])
    vmovaps %ymm4, S(512)(%rdi)
    vmovaps S(640)(%rdi), %ymm4
    vfmadd231ps %ymm14, %ymm2, %ymm6 # im(output[4][0])
    vmovaps S(640)(%rsi), %ymm2 # re(ferm[5][0])
    vmovaps %ymm6, S(576)(%rdi)
    vmovaps S(704)(%rdi), %ymm6

    # output[3] += re(ferm[5]) * F[3][5]
    # output[3] += re(ferm[5]) * F[5][3] (= +Im(F_35))
    # load imaginary parts of ferm[4] when done with this
    vfmadd231ps %ymm3, %ymm0, %ymm9  # re(output[3][1])
    vfmadd231ps %ymm3, %ymm1, %ymm11 # im(output[3][1])
    vmovaps S(736)(%rsi), %ymm3 # im(ferm[5][1])
    vfmadd231ps %ymm2, %ymm0, %ymm8  # re(output[3][0])
    vfmadd231ps %ymm2, %ymm1, %ymm10 # im(output[3][0])
    vmovaps S(704)(%rsi), %ymm2 # im(ferm[5][0])

    # output[5] += re(ferm[3]) * F[3][5]
    # output[5] -= re(ferm[3]) * F[5][3] (= -Im(F_35))
    # replace re(ferm[3]) with F[2][5]
    vfmadd231ps  %ymm13, %ymm0, %ymm5 # re(output[5][1])
    vfnmadd231ps %ymm13, %ymm1, %ymm7 # im(output[5][1])
    vbroadcastss 128(%rdx), %ymm13
    vfmadd231ps  %ymm12, %ymm0, %ymm4 # re(output[5][0])
    vfnmadd231ps %ymm12, %ymm1, %ymm6 # im(output[5][0])
    vbroadcastss 68(%rdx), %ymm12

    # now repeat with imaginary part of ferm[5] and ferm[3]
    # store output[3] back to memory
    # read output[2] from memory
    vfnmadd231ps %ymm3, %ymm1, %ymm9  # re(output[3][1])
    vmovaps %ymm9,  S(416)(%rdi)
    vmovaps S(288)(%rdi), %ymm9
    vfmadd231ps  %ymm3, %ymm0, %ymm11 # im(output[3][1])
    vmovaps %ymm11, S(480)(%rdi)
    vmovaps S(352)(%rdi), %ymm11
    vfnmadd231ps %ymm2, %ymm1, %ymm8  # re(output[3][0])
    vmovaps %ymm8,  S(384)(%rdi)
    vmovaps S(256)(%rdi), %ymm8
    vfmadd231ps  %ymm2, %ymm0, %ymm10 # im(output[3][0])
    vmovaps %ymm10, S(448)(%rdi)
    vmovaps S(320)(%rdi), %ymm10

    # replace imaginary part of ferm[3] with ferm[2]
    # also replace F[3][5] with re(ferm[2])
    vfmadd231ps %ymm15, %ymm1, %ymm5 # re(output[5][1])
    vfmadd231ps %ymm15, %ymm0, %ymm7 # im(output[5][1])
    vmovaps S(352)(%rsi), %ymm15 # im(ferm[2][1])
    vfmadd231ps %ymm14, %ymm1, %ymm4 # re(output[5][0])
    vmovaps S(288)(%rsi), %ymm1 # re(ferm[2][1])
    vfmadd231ps %ymm14, %ymm0, %ymm6 # im(output[5][0])
    vmovaps S(320)(%rsi), %ymm14 # im(ferm[2][0])
    vmovaps S(256)(%rsi), %ymm0 # re(ferm[2][0])
    # note that ferm[2] is now stored discontiguously
    # real part is in registers 0, 1
    # imag part is in registers 14, 15
    # F[2][5] is stored in registers 12, 13
    # imag part of ferm[5] is in registers 2, 3
    # output[2] is in registers 8-11
    # output[5] is in registers 4-7

    # output[2] += im(ferm[5]) * F[2][5]
    # replace im(ferm[5]) with re(ferm[5])
    vfnmadd231ps %ymm3, %ymm13, %ymm9  # re(output[2][1])
    vfmadd231ps  %ymm3, %ymm12, %ymm11 # im(output[2][1])
    vmovaps S(672)(%rsi), %ymm3 # re(ferm[5][1])
    vfnmadd231ps %ymm2, %ymm13, %ymm8  # re(output[2][0])
    vfmadd231ps  %ymm2, %ymm12, %ymm10 # im(output[2][0])
    vmovaps S(640)(%rsi), %ymm2 # re(ferm[5][0])

    # output[5] += im(ferm[2]) * F[2][5]
    vfmadd231ps %ymm15, %ymm13, %ymm5 # re(output[5][1])
    vfmadd231ps %ymm15, %ymm12, %ymm7 # im(output[5][1])
    vfmadd231ps %ymm14, %ymm13, %ymm4 # re(output[5][0])
    vfmadd231ps %ymm14, %ymm12, %ymm6 # im(output[5][0])

    # output[2] += re(ferm[5]) * F[2][5]
    # replace re(ferm[5]) with F[2][4]
    vfmadd231ps %ymm3, %ymm12, %ymm9  # re(output[2][1])
    vfmadd231ps %ymm3, %ymm13, %ymm11 # im(output[2][1])
    vbroadcastss 104(%rdx), %ymm3
    vfmadd231ps %ymm2, %ymm12, %ymm8  # re(output[2][0])
    vfmadd231ps %ymm2, %ymm13, %ymm10 # im(output[2][0])
    vbroadcastss 64(%rdx), %ymm2

    # output[5] += re(ferm[2]) * F[2][5]
    # store output[5] to memory
    # read output[4] from memory
    # replace F[2][5] with re(ferm[4])
    vfmadd231ps  %ymm1, %ymm12, %ymm5 # re(output[5][1])
    vmovaps %ymm5, S(672)(%rdi)
    vmovaps S(544)(%rdi), %ymm5
    vfnmadd231ps %ymm1, %ymm13, %ymm7 # im(output[5][1])
    vmovaps %ymm7, S(736)(%rdi)
    vmovaps S(608)(%rdi), %ymm7
    vfmadd231ps  %ymm0, %ymm12, %ymm4 # re(output[5][0])
    vmovaps S(512)(%rsi), %ymm12 # re(ferm[4][0])
    vmovaps %ymm4, S(640)(%rdi)
    vmovaps S(512)(%rdi), %ymm4
    vfnmadd231ps %ymm0, %ymm13, %ymm6 # im(output[5][0])
    vmovaps S(544)(%rsi), %ymm13 # re(ferm[4][1])
    vmovaps %ymm6, S(704)(%rdi)
    vmovaps S(576)(%rdi), %ymm6

    # output[2] += re(ferm[4]) * F[2][4]
    # output[2] += re(ferm[4]) * F[4][2] (= +Im(F_24))
    # load imaginary parts of ferm[4] when done with this
    vfmadd231ps %ymm13, %ymm2, %ymm9  # re(output[2][1])
    vfmadd231ps %ymm13, %ymm3, %ymm11 # im(output[2][1])
    vmovaps S(608)(%rsi), %ymm13 # im(ferm[4][1])
    vfmadd231ps %ymm12, %ymm2, %ymm8  # re(output[2][0])
    vfmadd231ps %ymm12, %ymm3, %ymm10 # im(output[2][0])
    vmovaps S(576)(%rsi), %ymm12 # im(ferm[4][0])

    # output[4] += re(ferm[2]) * F[2][4]
    # output[4] -= re(ferm[2]) * F[4][2] (= -Im(F_24))
    # replace re(ferm[2]) with F[4][4] and F[4][5]
    vfmadd231ps  %ymm1, %ymm2, %ymm5 # re(output[4][1])
    vfnmadd231ps %ymm1, %ymm3, %ymm7 # im(output[4][1])
    vbroadcastss 116(%rdx), %ymm1  # F[4][5]
    vfmadd231ps  %ymm0, %ymm2, %ymm4 # re(output[4][0])
    vfnmadd231ps %ymm0, %ymm3, %ymm6 # im(output[4][0])
    vbroadcastss 112(%rdx), %ymm0  # F[4][4]

    # now repeat with imaginary part of ferm[4] and ferm[2]
    # write output[2] to memory
    # replace with output[5]
    vfnmadd231ps %ymm13, %ymm3, %ymm9  # re(output[2][1])
    vmovaps %ymm9,  S(288)(%rdi)
    vmovaps S(672)(%rdi), %ymm9
    vfmadd231ps  %ymm13, %ymm2, %ymm11 # im(output[2][1])
    vmovaps %ymm11, S(352)(%rdi)
    vmovaps S(736)(%rdi), %ymm11
    vfnmadd231ps %ymm12, %ymm3, %ymm8  # re(output[2][0])
    vmovaps %ymm8,  S(256)(%rdi)
    vmovaps S(640)(%rdi), %ymm8
    vfmadd231ps  %ymm12, %ymm2, %ymm10 # im(output[2][0])
    vmovaps %ymm10, S(320)(%rdi)
    vmovaps S(704)(%rdi), %ymm10

    # replace im(ferm[2]) with re(ferm[4])
    # replace F[4][2] and F[2][4] with F[5][4] and F[5][5]
    vfmadd231ps %ymm15, %ymm3, %ymm5 # re(output[4][1])
    vfmadd231ps %ymm15, %ymm2, %ymm7 # im(output[4][1])
    vmovaps S(544)(%rsi), %ymm15 # re(ferm[4][1])
    vfmadd231ps %ymm14, %ymm3, %ymm4 # re(output[4][0])
    vbroadcastss 140(%rdx), %ymm3  # F[5][5]
    vfmadd231ps %ymm14, %ymm2, %ymm6 # im(output[4][0])
    vmovaps S(512)(%rsi), %ymm14 # re(ferm[4][0])
    vbroadcastss 136(%rdx), %ymm2  # F[5][4]

    # now we have
    # output[5]: 8-11
    # output[4]: 4-7
    # ferm[4]: 14-15, 12-13 (imag part first)
    # F[4][4-5]: 0-1
    # F[5][4-5]: 2-3

    # output[5] += ferm[4] * F[4][5]
    vfmadd231ps %ymm15, %ymm1, %ymm9  # re(output[5][1])
    vfmadd231ps %ymm13, %ymm1, %ymm11 # im(output[5][1])
    vfmadd231ps %ymm14, %ymm1, %ymm8  # re(output[5][0])
    vfmadd231ps %ymm12, %ymm1, %ymm10 # im(output[5][0])

    # output[4] += ferm[4] * F[4][4]
    vfmadd231ps %ymm15, %ymm0, %ymm5 # re(output[4][1])
    vfmadd231ps %ymm13, %ymm0, %ymm7 # im(output[4][1])
    vfmadd231ps %ymm14, %ymm0, %ymm4 # re(output[4][0])
    vfmadd231ps %ymm12, %ymm0, %ymm6 # im(output[4][0])

    # output[5] -= ferm[4] * F[5][4] (= -Im(F_54))
    # after this, we are finished with ferm[4]
    # replace this with ferm[5] (interleaved here)
    # we will take this opportunity to reorder ferm[5] (real part first)
    vfmadd231ps  %ymm13, %ymm2, %ymm9  # re(output[5][1])
    vmovaps S(672)(%rsi), %ymm13 # re(ferm[5][1])
    vfnmadd231ps %ymm15, %ymm2, %ymm11 # im(output[5][1])
    vmovaps S(736)(%rsi), %ymm15 # im(ferm[5][1])
    vfmadd231ps  %ymm12, %ymm2, %ymm8  # re(output[5][0])
    vmovaps S(640)(%rsi), %ymm12 # re(ferm[5][0])
    vfnmadd231ps %ymm14, %ymm2, %ymm10 # im(output[5][0])
    vmovaps S(704)(%rsi), %ymm14 # im(ferm[5][0])

    # output[4] += ferm[5] * F[4][5]
    vfmadd231ps %ymm13, %ymm1, %ymm5  # re(output[4][1])
    vfmadd231ps %ymm15, %ymm1, %ymm7  # im(output[4][1])
    vfmadd231ps %ymm12, %ymm1, %ymm4  # re(output[4][0])
    vfmadd231ps %ymm14, %ymm1, %ymm6  # im(output[4][0])

    # output[5] += ferm[5] * F[5][5]
    # write output[5] to memory
    vfmadd231ps %ymm13, %ymm3, %ymm9  # re(output[5][1])
    vmovaps %ymm9,  S(672)(%rdi)
    vfmadd231ps %ymm15, %ymm3, %ymm11 # im(output[5][1])
    vmovaps %ymm11, S(736)(%rdi)
    vfmadd231ps %ymm12, %ymm3, %ymm8  # re(output[5][0])
    vmovaps %ymm8,  S(640)(%rdi)
    vfmadd231ps %ymm14, %ymm3, %ymm10 # im(output[5][0])
    vmovaps %ymm10, S(704)(%rdi)

    # output[4] += ferm[5] * F[5][4] (= +Im(F_45))
    # store output[4] back to memory
    vfnmadd231ps %ymm15, %ymm2, %ymm5  # re(output[4][1])
    vmovaps %ymm5, S(544)(%rdi)
    vfmadd231ps  %ymm13, %ymm2, %ymm7  # im(output[4][1])
    vmovaps %ymm7, S(608)(%rdi)
    vfnmadd231ps %ymm14, %ymm2, %ymm4  # re(output[4][0])
    vmovaps %ymm4, S(512)(%rdi)
    vfmadd231ps  %ymm12, %ymm2, %ymm6  # im(output[4][0])
    vmovaps %ymm6, S(576)(%rdi)

    # second pass over this site's two chiral blocks -> block loop is done
    test %rax, %rax
#if num_rhs > 16
    jnz tile_next
#else
    jnz finish
#endif
    mov $1, %rax
    add $S(768), %rdi
    add $S(768), %rsi
    add $144, %rdx
    jmp multiply_block

#if num_rhs > 16
tile_next:
    add $64, %r9               # advance to the next 16-column tile
    cmp $num_rhs/16*64, %r9
    jl tile_loop
#endif
finish:
    endbr64
    ret
    .cfi_endproc
    .size KERNEL_NAME, .-KERNEL_NAME
    .section .note.GNU-stack,"",@progbits
