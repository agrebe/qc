// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

// ===========================================================================
// AVX-512 single-precision Wilson hopping kernel (full + masked); one site, one
// 16-column "zmm tile" per call.
//
// Schedule: each direction projects the input spinor into zmm0-11, then
// mult_U3 / mult_U3_dag form the products against all three output colors of U
// in one pass -- twelve independent accumulator chains (zmm18-29).  A dual-pipe
// 512-bit FMA core at 4-cycle latency needs >= 8 chains in flight to reach
// throughput; a one-output-color pass has only 4 and caps at half rate.
// Computing the full color row also makes every U displacement a compile-time
// constant (mult_U3 takes the direction's link byte offset as a macro
// argument), so the kernel never steps %r11 or the fermion/output pointers:
// every direction block is straight-line and self-contained, which is also
// what lets gen_wilson_avx512_masked.py gate each block with a bare
// test-and-branch.
//
// cpp/GAS template mechanics:
//   * num_rhs comes from global_params.h; S(D) rescales a
//     num_rhs=16 byte displacement to the actual num_rhs.  It is applied to
//     every fermion/output operand and not to gauge (r11) or stack (rbp)
//     operands, which are num_rhs-independent.  S(D)==D at num_rhs=16.
//   * The kernel always processes one 16-column zmm tile; for num_rhs>16 the C
//     caller (wilson_kernel / wilson_kernel_block) invokes it once per tile
//     with the fermion/output pointers stepped by 16 columns (gauge
//     unchanged), so no in-asm tile loop is needed.  Requires num_rhs % 16 ==
//     0 (the caller falls back to the generated AVX2 kernel otherwise).
// ===========================================================================
#include "global_params.h"   // num_rhs/Nc/Ns;
                             // C-only tail is __ASSEMBLER__-guarded
#if (num_rhs % 16) != 0
#error "wilson_kernel_avx512.S needs num_rhs % 16 == 0 (one-zmm-tile granularity)"
#endif
// component stride 8*num_rhs, imag stride 4*num_rhs (fermion ops are
// 64-aligned).  D is fully parenthesized so macro-argument sums like
// S(\cb+64) rescale correctly; the leading 0+ keeps the expansion from
// starting with '(', which a GAS memory operand could otherwise mis-parse as
// a register spec.
#define S(D)  0+(D)/128*8*num_rhs + (D)%128/64*4*num_rhs + (D)%64

# Register map:
#   zmm0-2   temp_re[spin 0], colors 0-2      zmm3-5   temp_re[spin 1]
#   zmm6-8   temp_im[spin 0]                  zmm9-11  temp_im[spin 1]
#   zmm12-17 U broadcasts (re/im x 3 output colors), reloaded per input color;
#            dead after mult_U3, reused as reconstruction scratch
#   zmm18-29 accumulators: output color j in 18+4j..21+4j as
#            (re spin0, im spin0, re spin1, im spin1)
#   zmm30-31 reconstruction scratch

# Multiply the projected half-spinor (zmm0-11) by U for all three output
# colors: prod[j] = sum_k U[j][k] * temp[k].  Rounds run over input color k;
# each round broadcasts column k of U (6 values), then issues the 12 re(U)
# products followed by the 12 im(U) products.  Each accumulator chain sees its
# six contributions in the same order as a one-output-color-at-a-time schedule,
# so results are bit-identical to one.
.macro mult_U3 uoff
    # input color 0: U column 0 (elements 0, 3, 6)
    vbroadcastss \uoff+0(%r11),  %zmm12 # re(U[0][0])
    vbroadcastss \uoff+4(%r11),  %zmm13 # im(U[0][0])
    vbroadcastss \uoff+24(%r11), %zmm14 # re(U[1][0])
    vbroadcastss \uoff+28(%r11), %zmm15 # im(U[1][0])
    vbroadcastss \uoff+48(%r11), %zmm16 # re(U[2][0])
    vbroadcastss \uoff+52(%r11), %zmm17 # im(U[2][0])
    vmulps       %zmm12, %zmm0,  %zmm18 # re[prod]  = re[U] * re[temp]
    vmulps       %zmm12, %zmm6,  %zmm19 # im[prod]  = re[U] * im[temp]
    vmulps       %zmm12, %zmm3,  %zmm20
    vmulps       %zmm12, %zmm9,  %zmm21
    vmulps       %zmm14, %zmm0,  %zmm22
    vmulps       %zmm14, %zmm6,  %zmm23
    vmulps       %zmm14, %zmm3,  %zmm24
    vmulps       %zmm14, %zmm9,  %zmm25
    vmulps       %zmm16, %zmm0,  %zmm26
    vmulps       %zmm16, %zmm6,  %zmm27
    vmulps       %zmm16, %zmm3,  %zmm28
    vmulps       %zmm16, %zmm9,  %zmm29
    vfnmadd231ps %zmm13, %zmm6,  %zmm18 # re[prod] -= im[U] * im[temp]
    vfmadd231ps  %zmm13, %zmm0,  %zmm19 # im[prod] += im[U] * re[temp]
    vfnmadd231ps %zmm13, %zmm9,  %zmm20
    vfmadd231ps  %zmm13, %zmm3,  %zmm21
    vfnmadd231ps %zmm15, %zmm6,  %zmm22
    vfmadd231ps  %zmm15, %zmm0,  %zmm23
    vfnmadd231ps %zmm15, %zmm9,  %zmm24
    vfmadd231ps  %zmm15, %zmm3,  %zmm25
    vfnmadd231ps %zmm17, %zmm6,  %zmm26
    vfmadd231ps  %zmm17, %zmm0,  %zmm27
    vfnmadd231ps %zmm17, %zmm9,  %zmm28
    vfmadd231ps  %zmm17, %zmm3,  %zmm29

    # input color 1: U column 1 (elements 1, 4, 7)
    vbroadcastss \uoff+8(%r11),  %zmm12 # re(U[0][1])
    vbroadcastss \uoff+12(%r11), %zmm13 # im(U[0][1])
    vbroadcastss \uoff+32(%r11), %zmm14 # re(U[1][1])
    vbroadcastss \uoff+36(%r11), %zmm15 # im(U[1][1])
    vbroadcastss \uoff+56(%r11), %zmm16 # re(U[2][1])
    vbroadcastss \uoff+60(%r11), %zmm17 # im(U[2][1])
    vfmadd231ps  %zmm12, %zmm1,  %zmm18 # re[prod] += re[U] * re[temp]
    vfmadd231ps  %zmm12, %zmm7,  %zmm19 # im[prod] += re[U] * im[temp]
    vfmadd231ps  %zmm12, %zmm4,  %zmm20
    vfmadd231ps  %zmm12, %zmm10, %zmm21
    vfmadd231ps  %zmm14, %zmm1,  %zmm22
    vfmadd231ps  %zmm14, %zmm7,  %zmm23
    vfmadd231ps  %zmm14, %zmm4,  %zmm24
    vfmadd231ps  %zmm14, %zmm10, %zmm25
    vfmadd231ps  %zmm16, %zmm1,  %zmm26
    vfmadd231ps  %zmm16, %zmm7,  %zmm27
    vfmadd231ps  %zmm16, %zmm4,  %zmm28
    vfmadd231ps  %zmm16, %zmm10, %zmm29
    vfnmadd231ps %zmm13, %zmm7,  %zmm18
    vfmadd231ps  %zmm13, %zmm1,  %zmm19
    vfnmadd231ps %zmm13, %zmm10, %zmm20
    vfmadd231ps  %zmm13, %zmm4,  %zmm21
    vfnmadd231ps %zmm15, %zmm7,  %zmm22
    vfmadd231ps  %zmm15, %zmm1,  %zmm23
    vfnmadd231ps %zmm15, %zmm10, %zmm24
    vfmadd231ps  %zmm15, %zmm4,  %zmm25
    vfnmadd231ps %zmm17, %zmm7,  %zmm26
    vfmadd231ps  %zmm17, %zmm1,  %zmm27
    vfnmadd231ps %zmm17, %zmm10, %zmm28
    vfmadd231ps  %zmm17, %zmm4,  %zmm29

    # input color 2: U column 2 (elements 2, 5, 8)
    vbroadcastss \uoff+16(%r11), %zmm12 # re(U[0][2])
    vbroadcastss \uoff+20(%r11), %zmm13 # im(U[0][2])
    vbroadcastss \uoff+40(%r11), %zmm14 # re(U[1][2])
    vbroadcastss \uoff+44(%r11), %zmm15 # im(U[1][2])
    vbroadcastss \uoff+64(%r11), %zmm16 # re(U[2][2])
    vbroadcastss \uoff+68(%r11), %zmm17 # im(U[2][2])
    vfmadd231ps  %zmm12, %zmm2,  %zmm18
    vfmadd231ps  %zmm12, %zmm8,  %zmm19
    vfmadd231ps  %zmm12, %zmm5,  %zmm20
    vfmadd231ps  %zmm12, %zmm11, %zmm21
    vfmadd231ps  %zmm14, %zmm2,  %zmm22
    vfmadd231ps  %zmm14, %zmm8,  %zmm23
    vfmadd231ps  %zmm14, %zmm5,  %zmm24
    vfmadd231ps  %zmm14, %zmm11, %zmm25
    vfmadd231ps  %zmm16, %zmm2,  %zmm26
    vfmadd231ps  %zmm16, %zmm8,  %zmm27
    vfmadd231ps  %zmm16, %zmm5,  %zmm28
    vfmadd231ps  %zmm16, %zmm11, %zmm29
    vfnmadd231ps %zmm13, %zmm8,  %zmm18
    vfmadd231ps  %zmm13, %zmm2,  %zmm19
    vfnmadd231ps %zmm13, %zmm11, %zmm20
    vfmadd231ps  %zmm13, %zmm5,  %zmm21
    vfnmadd231ps %zmm15, %zmm8,  %zmm22
    vfmadd231ps  %zmm15, %zmm2,  %zmm23
    vfnmadd231ps %zmm15, %zmm11, %zmm24
    vfmadd231ps  %zmm15, %zmm5,  %zmm25
    vfnmadd231ps %zmm17, %zmm8,  %zmm26
    vfmadd231ps  %zmm17, %zmm2,  %zmm27
    vfnmadd231ps %zmm17, %zmm11, %zmm28
    vfmadd231ps  %zmm17, %zmm5,  %zmm29
.endm

# Dagger twin: prod[j] = sum_k conj(U[k][j]) * temp[k].  Round k broadcasts
# row k of U (a contiguous 24-byte run); conjugation flips the sign of the
# im(U) products relative to mult_U3.
.macro mult_U3_dag uoff
    # input color 0: conj of U row 0 (elements 0, 1, 2)
    vbroadcastss \uoff+0(%r11),  %zmm12 # re(U[0][0])
    vbroadcastss \uoff+4(%r11),  %zmm13 # im(U[0][0])
    vbroadcastss \uoff+8(%r11),  %zmm14 # re(U[0][1])
    vbroadcastss \uoff+12(%r11), %zmm15 # im(U[0][1])
    vbroadcastss \uoff+16(%r11), %zmm16 # re(U[0][2])
    vbroadcastss \uoff+20(%r11), %zmm17 # im(U[0][2])
    vmulps       %zmm12, %zmm0,  %zmm18 # re[prod]  = re[U] * re[temp]
    vmulps       %zmm12, %zmm6,  %zmm19 # im[prod]  = re[U] * im[temp]
    vmulps       %zmm12, %zmm3,  %zmm20
    vmulps       %zmm12, %zmm9,  %zmm21
    vmulps       %zmm14, %zmm0,  %zmm22
    vmulps       %zmm14, %zmm6,  %zmm23
    vmulps       %zmm14, %zmm3,  %zmm24
    vmulps       %zmm14, %zmm9,  %zmm25
    vmulps       %zmm16, %zmm0,  %zmm26
    vmulps       %zmm16, %zmm6,  %zmm27
    vmulps       %zmm16, %zmm3,  %zmm28
    vmulps       %zmm16, %zmm9,  %zmm29
    vfmadd231ps  %zmm13, %zmm6,  %zmm18 # re[prod] += im[U*] * im[temp]
    vfnmadd231ps %zmm13, %zmm0,  %zmm19 # im[prod] -= im[U*] * re[temp]
    vfmadd231ps  %zmm13, %zmm9,  %zmm20
    vfnmadd231ps %zmm13, %zmm3,  %zmm21
    vfmadd231ps  %zmm15, %zmm6,  %zmm22
    vfnmadd231ps %zmm15, %zmm0,  %zmm23
    vfmadd231ps  %zmm15, %zmm9,  %zmm24
    vfnmadd231ps %zmm15, %zmm3,  %zmm25
    vfmadd231ps  %zmm17, %zmm6,  %zmm26
    vfnmadd231ps %zmm17, %zmm0,  %zmm27
    vfmadd231ps  %zmm17, %zmm9,  %zmm28
    vfnmadd231ps %zmm17, %zmm3,  %zmm29

    # input color 1: conj of U row 1 (elements 3, 4, 5)
    vbroadcastss \uoff+24(%r11), %zmm12 # re(U[1][0])
    vbroadcastss \uoff+28(%r11), %zmm13 # im(U[1][0])
    vbroadcastss \uoff+32(%r11), %zmm14 # re(U[1][1])
    vbroadcastss \uoff+36(%r11), %zmm15 # im(U[1][1])
    vbroadcastss \uoff+40(%r11), %zmm16 # re(U[1][2])
    vbroadcastss \uoff+44(%r11), %zmm17 # im(U[1][2])
    vfmadd231ps  %zmm12, %zmm1,  %zmm18
    vfmadd231ps  %zmm12, %zmm7,  %zmm19
    vfmadd231ps  %zmm12, %zmm4,  %zmm20
    vfmadd231ps  %zmm12, %zmm10, %zmm21
    vfmadd231ps  %zmm14, %zmm1,  %zmm22
    vfmadd231ps  %zmm14, %zmm7,  %zmm23
    vfmadd231ps  %zmm14, %zmm4,  %zmm24
    vfmadd231ps  %zmm14, %zmm10, %zmm25
    vfmadd231ps  %zmm16, %zmm1,  %zmm26
    vfmadd231ps  %zmm16, %zmm7,  %zmm27
    vfmadd231ps  %zmm16, %zmm4,  %zmm28
    vfmadd231ps  %zmm16, %zmm10, %zmm29
    vfmadd231ps  %zmm13, %zmm7,  %zmm18
    vfnmadd231ps %zmm13, %zmm1,  %zmm19
    vfmadd231ps  %zmm13, %zmm10, %zmm20
    vfnmadd231ps %zmm13, %zmm4,  %zmm21
    vfmadd231ps  %zmm15, %zmm7,  %zmm22
    vfnmadd231ps %zmm15, %zmm1,  %zmm23
    vfmadd231ps  %zmm15, %zmm10, %zmm24
    vfnmadd231ps %zmm15, %zmm4,  %zmm25
    vfmadd231ps  %zmm17, %zmm7,  %zmm26
    vfnmadd231ps %zmm17, %zmm1,  %zmm27
    vfmadd231ps  %zmm17, %zmm10, %zmm28
    vfnmadd231ps %zmm17, %zmm4,  %zmm29

    # input color 2: conj of U row 2 (elements 6, 7, 8)
    vbroadcastss \uoff+48(%r11), %zmm12 # re(U[2][0])
    vbroadcastss \uoff+52(%r11), %zmm13 # im(U[2][0])
    vbroadcastss \uoff+56(%r11), %zmm14 # re(U[2][1])
    vbroadcastss \uoff+60(%r11), %zmm15 # im(U[2][1])
    vbroadcastss \uoff+64(%r11), %zmm16 # re(U[2][2])
    vbroadcastss \uoff+68(%r11), %zmm17 # im(U[2][2])
    vfmadd231ps  %zmm12, %zmm2,  %zmm18
    vfmadd231ps  %zmm12, %zmm8,  %zmm19
    vfmadd231ps  %zmm12, %zmm5,  %zmm20
    vfmadd231ps  %zmm12, %zmm11, %zmm21
    vfmadd231ps  %zmm14, %zmm2,  %zmm22
    vfmadd231ps  %zmm14, %zmm8,  %zmm23
    vfmadd231ps  %zmm14, %zmm5,  %zmm24
    vfmadd231ps  %zmm14, %zmm11, %zmm25
    vfmadd231ps  %zmm16, %zmm2,  %zmm26
    vfmadd231ps  %zmm16, %zmm8,  %zmm27
    vfmadd231ps  %zmm16, %zmm5,  %zmm28
    vfmadd231ps  %zmm16, %zmm11, %zmm29
    vfmadd231ps  %zmm13, %zmm8,  %zmm18
    vfnmadd231ps %zmm13, %zmm2,  %zmm19
    vfmadd231ps  %zmm13, %zmm11, %zmm20
    vfnmadd231ps %zmm13, %zmm5,  %zmm21
    vfmadd231ps  %zmm15, %zmm8,  %zmm22
    vfnmadd231ps %zmm15, %zmm2,  %zmm23
    vfmadd231ps  %zmm15, %zmm11, %zmm24
    vfnmadd231ps %zmm15, %zmm5,  %zmm25
    vfmadd231ps  %zmm17, %zmm8,  %zmm26
    vfnmadd231ps %zmm17, %zmm2,  %zmm27
    vfmadd231ps  %zmm17, %zmm11, %zmm28
    vfnmadd231ps %zmm17, %zmm5,  %zmm29
.endm

# Reconstruction: fold output color \cb/128's accumulator quartet
# (\pr0 \pi0 \pr1 \pi1) = prod[spin 0/1] re/im back into the output spinor.
# The spin-0/1 rows accumulate the product directly in every direction; the
# spin-2/3 rows carry the direction's gamma-projector signs, one rmw_* macro
# per direction.  Scratch: zmm30/31 + the dead U broadcasts zmm12-17.
.macro rmw_spin01 cb, pr0, pi0, pr1, pi1
    # re/im(output[0]) += re/im(prod[0]); re/im(output[1]) += re/im(prod[1])
    vaddps S(\cb+0)(%rdi),   %zmm\pr0, %zmm30
    vaddps S(\cb+64)(%rdi),  %zmm\pi0, %zmm31
    vaddps S(\cb+384)(%rdi), %zmm\pr1, %zmm12
    vaddps S(\cb+448)(%rdi), %zmm\pi1, %zmm13
    vmovaps %zmm30, S(\cb+0)(%rdi)
    vmovaps %zmm31, S(\cb+64)(%rdi)
    vmovaps %zmm12, S(\cb+384)(%rdi)
    vmovaps %zmm13, S(\cb+448)(%rdi)
.endm

.macro rmw_plus_x cb, pr0, pi0, pr1, pi1
    rmw_spin01 \cb, \pr0, \pi0, \pr1, \pi1
    # re(output[2]) -= im(prod[1])
    vmovaps S(\cb+768)(%rdi), %zmm14
    vsubps %zmm\pi1, %zmm14, %zmm14
    # im(output[2]) += re(prod[1])
    vaddps S(\cb+832)(%rdi), %zmm\pr1, %zmm15
    # re(output[3]) -= im(prod[0])
    vmovaps S(\cb+1152)(%rdi), %zmm16
    vsubps %zmm\pi0, %zmm16, %zmm16
    # im(output[3]) += re(prod[0])
    vaddps S(\cb+1216)(%rdi), %zmm\pr0, %zmm17
    vmovaps %zmm14, S(\cb+768)(%rdi)
    vmovaps %zmm15, S(\cb+832)(%rdi)
    vmovaps %zmm16, S(\cb+1152)(%rdi)
    vmovaps %zmm17, S(\cb+1216)(%rdi)
.endm

.macro rmw_plus_y cb, pr0, pi0, pr1, pi1
    rmw_spin01 \cb, \pr0, \pi0, \pr1, \pi1
    # re(output[2]) -= re(prod[1])
    vmovaps S(\cb+768)(%rdi), %zmm14
    vsubps %zmm\pr1, %zmm14, %zmm14
    # im(output[2]) -= im(prod[1])
    vmovaps S(\cb+832)(%rdi), %zmm15
    vsubps %zmm\pi1, %zmm15, %zmm15
    # re(output[3]) += re(prod[0])
    vaddps S(\cb+1152)(%rdi), %zmm\pr0, %zmm16
    # im(output[3]) += im(prod[0])
    vaddps S(\cb+1216)(%rdi), %zmm\pi0, %zmm17
    vmovaps %zmm14, S(\cb+768)(%rdi)
    vmovaps %zmm15, S(\cb+832)(%rdi)
    vmovaps %zmm16, S(\cb+1152)(%rdi)
    vmovaps %zmm17, S(\cb+1216)(%rdi)
.endm

.macro rmw_plus_z cb, pr0, pi0, pr1, pi1
    rmw_spin01 \cb, \pr0, \pi0, \pr1, \pi1
    # re(output[2]) -= im(prod[0])
    vmovaps S(\cb+768)(%rdi), %zmm14
    vsubps %zmm\pi0, %zmm14, %zmm14
    # im(output[2]) += re(prod[0])
    vaddps S(\cb+832)(%rdi), %zmm\pr0, %zmm15
    # re(output[3]) += im(prod[1])
    vaddps S(\cb+1152)(%rdi), %zmm\pi1, %zmm16
    # im(output[3]) -= re(prod[1])
    vmovaps S(\cb+1216)(%rdi), %zmm17
    vsubps %zmm\pr1, %zmm17, %zmm17
    vmovaps %zmm14, S(\cb+768)(%rdi)
    vmovaps %zmm15, S(\cb+832)(%rdi)
    vmovaps %zmm16, S(\cb+1152)(%rdi)
    vmovaps %zmm17, S(\cb+1216)(%rdi)
.endm

.macro rmw_plus_t cb, pr0, pi0, pr1, pi1
    rmw_spin01 \cb, \pr0, \pi0, \pr1, \pi1
    # re(output[2]) -= re(prod[0])
    vmovaps S(\cb+768)(%rdi), %zmm14
    vsubps %zmm\pr0, %zmm14, %zmm14
    # im(output[2]) -= im(prod[0])
    vmovaps S(\cb+832)(%rdi), %zmm15
    vsubps %zmm\pi0, %zmm15, %zmm15
    # re(output[3]) -= re(prod[1])
    vmovaps S(\cb+1152)(%rdi), %zmm16
    vsubps %zmm\pr1, %zmm16, %zmm16
    # im(output[3]) -= im(prod[1])
    vmovaps S(\cb+1216)(%rdi), %zmm17
    vsubps %zmm\pi1, %zmm17, %zmm17
    vmovaps %zmm14, S(\cb+768)(%rdi)
    vmovaps %zmm15, S(\cb+832)(%rdi)
    vmovaps %zmm16, S(\cb+1152)(%rdi)
    vmovaps %zmm17, S(\cb+1216)(%rdi)
.endm

.macro rmw_minus_x cb, pr0, pi0, pr1, pi1
    rmw_spin01 \cb, \pr0, \pi0, \pr1, \pi1
    # re(output[2]) += im(prod[1])
    vaddps S(\cb+768)(%rdi), %zmm\pi1, %zmm14
    # im(output[2]) -= re(prod[1])
    vmovaps S(\cb+832)(%rdi), %zmm15
    vsubps %zmm\pr1, %zmm15, %zmm15
    # re(output[3]) += im(prod[0])
    vaddps S(\cb+1152)(%rdi), %zmm\pi0, %zmm16
    # im(output[3]) -= re(prod[0])
    vmovaps S(\cb+1216)(%rdi), %zmm17
    vsubps %zmm\pr0, %zmm17, %zmm17
    vmovaps %zmm14, S(\cb+768)(%rdi)
    vmovaps %zmm15, S(\cb+832)(%rdi)
    vmovaps %zmm16, S(\cb+1152)(%rdi)
    vmovaps %zmm17, S(\cb+1216)(%rdi)
.endm

.macro rmw_minus_y cb, pr0, pi0, pr1, pi1
    rmw_spin01 \cb, \pr0, \pi0, \pr1, \pi1
    # re(output[2]) += re(prod[1])
    vaddps S(\cb+768)(%rdi), %zmm\pr1, %zmm14
    # im(output[2]) += im(prod[1])
    vaddps S(\cb+832)(%rdi), %zmm\pi1, %zmm15
    # re(output[3]) -= re(prod[0])
    vmovaps S(\cb+1152)(%rdi), %zmm16
    vsubps %zmm\pr0, %zmm16, %zmm16
    # im(output[3]) -= im(prod[0])
    vmovaps S(\cb+1216)(%rdi), %zmm17
    vsubps %zmm\pi0, %zmm17, %zmm17
    vmovaps %zmm14, S(\cb+768)(%rdi)
    vmovaps %zmm15, S(\cb+832)(%rdi)
    vmovaps %zmm16, S(\cb+1152)(%rdi)
    vmovaps %zmm17, S(\cb+1216)(%rdi)
.endm

.macro rmw_minus_z cb, pr0, pi0, pr1, pi1
    rmw_spin01 \cb, \pr0, \pi0, \pr1, \pi1
    # re(output[2]) += im(prod[0])
    vaddps S(\cb+768)(%rdi), %zmm\pi0, %zmm14
    # im(output[2]) -= re(prod[0])
    vmovaps S(\cb+832)(%rdi), %zmm15
    vsubps %zmm\pr0, %zmm15, %zmm15
    # re(output[3]) -= im(prod[1])
    vmovaps S(\cb+1152)(%rdi), %zmm16
    vsubps %zmm\pi1, %zmm16, %zmm16
    # im(output[3]) += re(prod[1])
    vaddps S(\cb+1216)(%rdi), %zmm\pr1, %zmm17
    vmovaps %zmm14, S(\cb+768)(%rdi)
    vmovaps %zmm15, S(\cb+832)(%rdi)
    vmovaps %zmm16, S(\cb+1152)(%rdi)
    vmovaps %zmm17, S(\cb+1216)(%rdi)
.endm

.macro rmw_minus_t cb, pr0, pi0, pr1, pi1
    rmw_spin01 \cb, \pr0, \pi0, \pr1, \pi1
    # re(output[2]) += re(prod[0])
    vaddps S(\cb+768)(%rdi), %zmm\pr0, %zmm14
    # im(output[2]) += im(prod[0])
    vaddps S(\cb+832)(%rdi), %zmm\pi0, %zmm15
    # re(output[3]) += re(prod[1])
    vaddps S(\cb+1152)(%rdi), %zmm\pr1, %zmm16
    # im(output[3]) += im(prod[1])
    vaddps S(\cb+1216)(%rdi), %zmm\pi1, %zmm17
    vmovaps %zmm14, S(\cb+768)(%rdi)
    vmovaps %zmm15, S(\cb+832)(%rdi)
    vmovaps %zmm16, S(\cb+1152)(%rdi)
    vmovaps %zmm17, S(\cb+1216)(%rdi)
.endm

# Software-prefetch helpers (only invoked under .ifdef WK_PREFETCH).
# pf_ferm pulls this tile's slice of a neighbor fermion site (Ns*Nc*2 = 24
# components, one 64-byte line = 16 columns each) into L1; pf_gauge pulls one
# 72-byte link (<=2 lines).  These mirror the prefetch_ferm / prefetch_gauge
# macros in clover_ferm.c so the asm hopping kernel hides the same memory
# latency the C path does.
# The 24 component lines are S()-strided (8*num_rhs apart), not a flat 64*line:
# at num_rhs=16 that is the original contiguous 0..1472, but at larger num_rhs
# it follows the tile's components instead of fetching one num_rhs=16-sized
# slab and missing most of them.
.macro pf_ferm base
    .set _pf_line, 0
    .rept 24
        prefetcht0 S(_pf_line*64)(\base)
        .set _pf_line, _pf_line + 1
    .endr
.endm
.macro pf_gauge base, doff
    prefetcht0 \doff(\base)
    prefetcht0 \doff+64(\base)
.endm

    .globl wilson_kernel_avx512
    .type wilson_kernel_avx512, @function
    # arguments:
    # float * output:       %rdi
    # float * ferm_plus_x:  %rsi
    # float * ferm_plus_y:  %rdx
    # float * ferm_plus_z:  %rcx
    # float * ferm_plus_t:  %r8
    # float * ferm_minus_x: %r9
    # float * ferm_minus_y: 16(%rbp)
    # float * ferm_minus_z: 24(%rbp)
    # float * ferm_minus_t: 32(%rbp)
    # float * U:            40(%rbp)
    # float * U_minus_x:    48(%rbp)
    # float * U_minus_y:    56(%rbp)
    # float * U_minus_z:    64(%rbp)
    # float * U_minus_t:    72(%rbp)
    # all of output, ferm, and U have been advanced to the right position
    # single precision; processes one 16-column tile
    # 64 B entry alignment: the ~10 KB straight-line body is MITE-fed, so its
    # fetch-window alignment follows the entry; without this the link layout
    # (any object-size change) moves it, worth a measured ~14% per call
    .p2align 6
wilson_kernel_avx512:
    .cfi_startproc
    pushq %rbp
    movq %rsp, %rbp

plus_x:
    # multiplication in +x direction
    # load (pointer to) U into register r11
    movq 40(%rbp), %r11
.ifdef WK_PREFETCH
    # pull the next directions' inputs into L1 (mirrors wilson_kernel_c)
    pf_ferm %rdx                 # plus_y fermion
    pf_ferm %rcx                 # plus_z fermion
    movq 40(%rbp), %r10
    pf_gauge %r10, 72            # +y link
    pf_gauge %r10, 144           # +z link
.endif
    # ferm is in %rsi
    # imaginary part is in (real part) + 64
    # ferm[s=1] is in S(384)(%rsi), etc.
    # projection of input fermion
    # temp_re[0] = re(ferm[0]) + im(ferm[3])
    vmovaps    (%rsi), %zmm0 # c = 0
    vmovaps S(128)(%rsi), %zmm1 # c = 1
    vmovaps S(256)(%rsi), %zmm2 # c = 2
    vaddps S(1216)(%rsi), %zmm0, %zmm0
    vaddps S(1344)(%rsi), %zmm1, %zmm1
    vaddps S(1472)(%rsi), %zmm2, %zmm2
    # temp_re[1] = re(ferm[1]) + im(ferm[2])
    vmovaps S(384)(%rsi), %zmm3
    vmovaps S(512)(%rsi), %zmm4
    vmovaps S(640)(%rsi), %zmm5
    vaddps S(832)(%rsi), %zmm3, %zmm3
    vaddps S(960)(%rsi), %zmm4, %zmm4
    vaddps S(1088)(%rsi), %zmm5, %zmm5
    # temp_im[0] = im(ferm[0]) - re(ferm[3])
    vmovaps S(64)(%rsi), %zmm6
    vmovaps S(192)(%rsi), %zmm7
    vmovaps S(320)(%rsi), %zmm8
    vsubps S(1152)(%rsi), %zmm6, %zmm6
    vsubps S(1280)(%rsi), %zmm7, %zmm7
    vsubps S(1408)(%rsi), %zmm8, %zmm8
    # temp_im[1] = im(ferm[1]) - re(ferm[2])
    vmovaps S(448)(%rsi), %zmm9
    vmovaps S(576)(%rsi), %zmm10
    vmovaps S(704)(%rsi), %zmm11
    vsubps S(768)(%rsi), %zmm9, %zmm9
    vsubps S(896)(%rsi), %zmm10, %zmm10
    vsubps S(1024)(%rsi), %zmm11, %zmm11

    mult_U3 0
    rmw_plus_x 0,   18, 19, 20, 21
    rmw_plus_x 128, 22, 23, 24, 25
    rmw_plus_x 256, 26, 27, 28, 29

plus_y:
    # multiplication in +y direction; the +y link is 72 bytes into the record
    movq 40(%rbp), %r11
.ifdef WK_PREFETCH
    pf_ferm %r8                  # plus_t fermion
    movq 40(%rbp), %r10
    pf_gauge %r10, 216           # +t link
.endif
    # ferm is in %rdx
    # imaginary part is in (real part) + 64
    # ferm[s=1] is in S(384)(%rdx), etc.
    # projection of input fermion
    # temp_re[0] = re(ferm[0]) + re(ferm[3])
    vmovaps    (%rdx), %zmm0 # c = 0
    vmovaps S(128)(%rdx), %zmm1 # c = 1
    vmovaps S(256)(%rdx), %zmm2 # c = 2
    vaddps S(1152)(%rdx), %zmm0, %zmm0
    vaddps S(1280)(%rdx), %zmm1, %zmm1
    vaddps S(1408)(%rdx), %zmm2, %zmm2
    # temp_re[1] = re(ferm[1]) - re(ferm[2])
    vmovaps S(384)(%rdx), %zmm3
    vmovaps S(512)(%rdx), %zmm4
    vmovaps S(640)(%rdx), %zmm5
    vsubps S(768)(%rdx), %zmm3, %zmm3
    vsubps S(896)(%rdx), %zmm4, %zmm4
    vsubps S(1024)(%rdx), %zmm5, %zmm5
    # temp_im[0] = im(ferm[0]) + im(ferm[3])
    vmovaps S(64)(%rdx), %zmm6
    vmovaps S(192)(%rdx), %zmm7
    vmovaps S(320)(%rdx), %zmm8
    vaddps S(1216)(%rdx), %zmm6, %zmm6
    vaddps S(1344)(%rdx), %zmm7, %zmm7
    vaddps S(1472)(%rdx), %zmm8, %zmm8
    # temp_im[1] = im(ferm[1]) - im(ferm[2])
    vmovaps S(448)(%rdx), %zmm9
    vmovaps S(576)(%rdx), %zmm10
    vmovaps S(704)(%rdx), %zmm11
    vsubps S(832)(%rdx), %zmm9, %zmm9
    vsubps S(960)(%rdx), %zmm10, %zmm10
    vsubps S(1088)(%rdx), %zmm11, %zmm11

    mult_U3 72
    rmw_plus_y 0,   18, 19, 20, 21
    rmw_plus_y 128, 22, 23, 24, 25
    rmw_plus_y 256, 26, 27, 28, 29

plus_z:
    # multiplication in +z direction; the +z link is 144 bytes into the record
    movq 40(%rbp), %r11
.ifdef WK_PREFETCH
    pf_ferm %r9                  # minus_x fermion
    movq 48(%rbp), %r10
    pf_gauge %r10, 0             # -x link (U_minus_x, dir 0)
.endif
    # ferm is in %rcx
    # imaginary part is in (real part) + 64
    # ferm[s=1] is in S(384)(%rcx), etc.
    # projection of input fermion
    # temp_re[0] = re(ferm[0]) + im(ferm[2])
    vmovaps    (%rcx), %zmm0 # c = 0
    vmovaps S(128)(%rcx), %zmm1 # c = 1
    vmovaps S(256)(%rcx), %zmm2 # c = 2
    vaddps S(832)(%rcx), %zmm0, %zmm0
    vaddps S(960)(%rcx), %zmm1, %zmm1
    vaddps S(1088)(%rcx), %zmm2, %zmm2
    # temp_re[1] = re(ferm[1]) - im(ferm[3])
    vmovaps S(384)(%rcx), %zmm3
    vmovaps S(512)(%rcx), %zmm4
    vmovaps S(640)(%rcx), %zmm5
    vsubps S(1216)(%rcx), %zmm3, %zmm3
    vsubps S(1344)(%rcx), %zmm4, %zmm4
    vsubps S(1472)(%rcx), %zmm5, %zmm5
    # temp_im[0] = im(ferm[0]) - re(ferm[2])
    vmovaps S(64)(%rcx), %zmm6
    vmovaps S(192)(%rcx), %zmm7
    vmovaps S(320)(%rcx), %zmm8
    vsubps S(768)(%rcx), %zmm6, %zmm6
    vsubps S(896)(%rcx), %zmm7, %zmm7
    vsubps S(1024)(%rcx), %zmm8, %zmm8
    # temp_im[1] = im(ferm[1]) + re(ferm[3])
    vmovaps S(448)(%rcx), %zmm9
    vmovaps S(576)(%rcx), %zmm10
    vmovaps S(704)(%rcx), %zmm11
    vaddps S(1152)(%rcx), %zmm9, %zmm9
    vaddps S(1280)(%rcx), %zmm10, %zmm10
    vaddps S(1408)(%rcx), %zmm11, %zmm11

    mult_U3 144
    rmw_plus_z 0,   18, 19, 20, 21
    rmw_plus_z 128, 22, 23, 24, 25
    rmw_plus_z 256, 26, 27, 28, 29

plus_t:
    # multiplication in +t direction; the +t link is 216 bytes into the record
    movq 40(%rbp), %r11
.ifdef WK_PREFETCH
    movq 16(%rbp), %r10
    pf_ferm %r10                 # minus_y fermion
    movq 56(%rbp), %r10
    pf_gauge %r10, 72            # -y link (U_minus_y, dir 1)
.endif
    # ferm is in %r8
    # imaginary part is in (real part) + 64
    # ferm[s=1] is in S(384)(%r8), etc.
    # projection of input fermion
    # temp_re[0] = re(ferm[0]) - re(ferm[2])
    vmovaps    (%r8), %zmm0 # c = 0
    vmovaps S(128)(%r8), %zmm1 # c = 1
    vmovaps S(256)(%r8), %zmm2 # c = 2
    vsubps  S(768)(%r8), %zmm0, %zmm0
    vsubps  S(896)(%r8), %zmm1, %zmm1
    vsubps S(1024)(%r8), %zmm2, %zmm2
    # temp_re[1] = re(ferm[1]) - re(ferm[3])
    vmovaps S(384)(%r8), %zmm3
    vmovaps S(512)(%r8), %zmm4
    vmovaps S(640)(%r8), %zmm5
    vsubps S(1152)(%r8), %zmm3, %zmm3
    vsubps S(1280)(%r8), %zmm4, %zmm4
    vsubps S(1408)(%r8), %zmm5, %zmm5
    # temp_im[0] = im(ferm[0]) - im(ferm[2])
    vmovaps S(64)(%r8), %zmm6
    vmovaps S(192)(%r8), %zmm7
    vmovaps S(320)(%r8), %zmm8
    vsubps  S(832)(%r8), %zmm6, %zmm6
    vsubps  S(960)(%r8), %zmm7, %zmm7
    vsubps S(1088)(%r8), %zmm8, %zmm8
    # temp_im[1] = im(ferm[1]) - im(ferm[3])
    vmovaps S(448)(%r8), %zmm9
    vmovaps S(576)(%r8), %zmm10
    vmovaps S(704)(%r8), %zmm11
    vsubps S(1216)(%r8), %zmm9, %zmm9
    vsubps S(1344)(%r8), %zmm10, %zmm10
    vsubps S(1472)(%r8), %zmm11, %zmm11

    mult_U3 216
    rmw_plus_t 0,   18, 19, 20, 21
    rmw_plus_t 128, 22, 23, 24, 25
    rmw_plus_t 256, 26, 27, 28, 29

minus_x:
    # multiplication in -x direction: U^dag from the -x neighbor's x link
    movq 48(%rbp), %r11
.ifdef WK_PREFETCH
    movq 24(%rbp), %r10
    pf_ferm %r10                 # minus_z fermion
    movq 64(%rbp), %r10
    pf_gauge %r10, 144           # -z link (U_minus_z, dir 2)
.endif
    # temp_re[0] = re(ferm[0]) - im(ferm[3])
    vmovaps    (%r9), %zmm0 # c = 0
    vmovaps S(128)(%r9), %zmm1 # c = 1
    vmovaps S(256)(%r9), %zmm2 # c = 2
    vsubps S(1216)(%r9), %zmm0, %zmm0
    vsubps S(1344)(%r9), %zmm1, %zmm1
    vsubps S(1472)(%r9), %zmm2, %zmm2
    # temp_re[1] = re(ferm[1]) - im(ferm[2])
    vmovaps S(384)(%r9), %zmm3
    vmovaps S(512)(%r9), %zmm4
    vmovaps S(640)(%r9), %zmm5
    vsubps S(832)(%r9), %zmm3, %zmm3
    vsubps S(960)(%r9), %zmm4, %zmm4
    vsubps S(1088)(%r9), %zmm5, %zmm5
    # temp_im[0] = im(ferm[0]) + re(ferm[3])
    vmovaps S(64)(%r9), %zmm6
    vmovaps S(192)(%r9), %zmm7
    vmovaps S(320)(%r9), %zmm8
    vaddps S(1152)(%r9), %zmm6, %zmm6
    vaddps S(1280)(%r9), %zmm7, %zmm7
    vaddps S(1408)(%r9), %zmm8, %zmm8
    # temp_im[1] = im(ferm[1]) + re(ferm[2])
    vmovaps S(448)(%r9), %zmm9
    vmovaps S(576)(%r9), %zmm10
    vmovaps S(704)(%r9), %zmm11
    vaddps S(768)(%r9), %zmm9, %zmm9
    vaddps S(896)(%r9), %zmm10, %zmm10
    vaddps S(1024)(%r9), %zmm11, %zmm11

    mult_U3_dag 0
    rmw_minus_x 0,   18, 19, 20, 21
    rmw_minus_x 128, 22, 23, 24, 25
    rmw_minus_x 256, 26, 27, 28, 29

minus_y:
    # multiplication in -y direction: U^dag from the -y neighbor's y link
    movq 56(%rbp), %r11
    # load ferm into %rdx
    movq 16(%rbp), %rdx
.ifdef WK_PREFETCH
    movq 32(%rbp), %r10
    pf_ferm %r10                 # minus_t fermion
    movq 72(%rbp), %r10
    pf_gauge %r10, 216           # -t link (U_minus_t, dir 3)
.endif
    # imaginary part is in (real part) + 64
    # ferm[s=1] is in S(384)(%rdx), etc.
    # projection of input fermion
    # temp_re[0] = re(ferm[0]) - re(ferm[3])
    vmovaps    (%rdx), %zmm0 # c = 0
    vmovaps S(128)(%rdx), %zmm1 # c = 1
    vmovaps S(256)(%rdx), %zmm2 # c = 2
    vsubps S(1152)(%rdx), %zmm0, %zmm0
    vsubps S(1280)(%rdx), %zmm1, %zmm1
    vsubps S(1408)(%rdx), %zmm2, %zmm2
    # temp_re[1] = re(ferm[1]) + re(ferm[2])
    vmovaps S(384)(%rdx), %zmm3
    vmovaps S(512)(%rdx), %zmm4
    vmovaps S(640)(%rdx), %zmm5
    vaddps S(768)(%rdx), %zmm3, %zmm3
    vaddps S(896)(%rdx), %zmm4, %zmm4
    vaddps S(1024)(%rdx), %zmm5, %zmm5
    # temp_im[0] = im(ferm[0]) - im(ferm[3])
    vmovaps S(64)(%rdx), %zmm6
    vmovaps S(192)(%rdx), %zmm7
    vmovaps S(320)(%rdx), %zmm8
    vsubps S(1216)(%rdx), %zmm6, %zmm6
    vsubps S(1344)(%rdx), %zmm7, %zmm7
    vsubps S(1472)(%rdx), %zmm8, %zmm8
    # temp_im[1] = im(ferm[1]) + im(ferm[2])
    vmovaps S(448)(%rdx), %zmm9
    vmovaps S(576)(%rdx), %zmm10
    vmovaps S(704)(%rdx), %zmm11
    vaddps S(832)(%rdx), %zmm9, %zmm9
    vaddps S(960)(%rdx), %zmm10, %zmm10
    vaddps S(1088)(%rdx), %zmm11, %zmm11

    mult_U3_dag 72
    rmw_minus_y 0,   18, 19, 20, 21
    rmw_minus_y 128, 22, 23, 24, 25
    rmw_minus_y 256, 26, 27, 28, 29

minus_z:
    # multiplication in -z direction: U^dag from the -z neighbor's z link
    movq 64(%rbp), %r11
    # load ferm into %rcx
    movq 24(%rbp), %rcx
    # imaginary part is in (real part) + 64
    # ferm[s=1] is in S(384)(%rcx), etc.
    # projection of input fermion
    # temp_re[0] = re(ferm[0]) - im(ferm[2])
    vmovaps    (%rcx), %zmm0 # c = 0
    vmovaps S(128)(%rcx), %zmm1 # c = 1
    vmovaps S(256)(%rcx), %zmm2 # c = 2
    vsubps S(832)(%rcx), %zmm0, %zmm0
    vsubps S(960)(%rcx), %zmm1, %zmm1
    vsubps S(1088)(%rcx), %zmm2, %zmm2
    # temp_re[1] = re(ferm[1]) + im(ferm[3])
    vmovaps S(384)(%rcx), %zmm3
    vmovaps S(512)(%rcx), %zmm4
    vmovaps S(640)(%rcx), %zmm5
    vaddps S(1216)(%rcx), %zmm3, %zmm3
    vaddps S(1344)(%rcx), %zmm4, %zmm4
    vaddps S(1472)(%rcx), %zmm5, %zmm5
    # temp_im[0] = im(ferm[0]) + re(ferm[2])
    vmovaps S(64)(%rcx), %zmm6
    vmovaps S(192)(%rcx), %zmm7
    vmovaps S(320)(%rcx), %zmm8
    vaddps S(768)(%rcx), %zmm6, %zmm6
    vaddps S(896)(%rcx), %zmm7, %zmm7
    vaddps S(1024)(%rcx), %zmm8, %zmm8
    # temp_im[1] = im(ferm[1]) - re(ferm[3])
    vmovaps S(448)(%rcx), %zmm9
    vmovaps S(576)(%rcx), %zmm10
    vmovaps S(704)(%rcx), %zmm11
    vsubps S(1152)(%rcx), %zmm9, %zmm9
    vsubps S(1280)(%rcx), %zmm10, %zmm10
    vsubps S(1408)(%rcx), %zmm11, %zmm11

    mult_U3_dag 144
    rmw_minus_z 0,   18, 19, 20, 21
    rmw_minus_z 128, 22, 23, 24, 25
    rmw_minus_z 256, 26, 27, 28, 29

minus_t:
    # multiplication in -t direction: U^dag from the -t neighbor's t link
    movq 72(%rbp), %r11
    # load ferm into %r8
    movq 32(%rbp), %r8
    # imaginary part is in (real part) + 64
    # ferm[s=1] is in S(384)(%r8), etc.
    # projection of input fermion
    # temp_re[0] = re(ferm[0]) + re(ferm[2])
    vmovaps    (%r8), %zmm0 # c = 0
    vmovaps S(128)(%r8), %zmm1 # c = 1
    vmovaps S(256)(%r8), %zmm2 # c = 2
    vaddps  S(768)(%r8), %zmm0, %zmm0
    vaddps  S(896)(%r8), %zmm1, %zmm1
    vaddps S(1024)(%r8), %zmm2, %zmm2
    # temp_re[1] = re(ferm[1]) + re(ferm[3])
    vmovaps S(384)(%r8), %zmm3
    vmovaps S(512)(%r8), %zmm4
    vmovaps S(640)(%r8), %zmm5
    vaddps S(1152)(%r8), %zmm3, %zmm3
    vaddps S(1280)(%r8), %zmm4, %zmm4
    vaddps S(1408)(%r8), %zmm5, %zmm5
    # temp_im[0] = im(ferm[0]) + im(ferm[2])
    vmovaps S(64)(%r8), %zmm6
    vmovaps S(192)(%r8), %zmm7
    vmovaps S(320)(%r8), %zmm8
    vaddps  S(832)(%r8), %zmm6, %zmm6
    vaddps  S(960)(%r8), %zmm7, %zmm7
    vaddps S(1088)(%r8), %zmm8, %zmm8
    # temp_im[1] = im(ferm[1]) + im(ferm[3])
    vmovaps S(448)(%r8), %zmm9
    vmovaps S(576)(%r8), %zmm10
    vmovaps S(704)(%r8), %zmm11
    vaddps S(1216)(%r8), %zmm9, %zmm9
    vaddps S(1344)(%r8), %zmm10, %zmm10
    vaddps S(1472)(%r8), %zmm11, %zmm11

    mult_U3_dag 216
    rmw_minus_t 0,   18, 19, 20, 21
    rmw_minus_t 128, 22, 23, 24, 25
    rmw_minus_t 256, 26, 27, 28, 29

finish:
    movq %rbp, %rsp
    popq %rbp
    endbr64
    ret
    .cfi_endproc
    .size wilson_kernel_avx512, .-wilson_kernel_avx512
# === BEGIN generated wilson_kernel_avx512_masked (do not edit by hand; see gen_wilson_avx512_masked.py) ===
    .globl wilson_kernel_avx512_masked
    .type wilson_kernel_avx512_masked, @function
    # Schwarz-block variant of wilson_kernel_avx512: identical per-direction
    # arithmetic, but gated by an 8-bit direction mask (15th arg, 80(%rbp)).
    # bit 0:+x 1:+y 2:+z 3:+t 4:-x 5:-y 6:-z 7:-t -- excluded directions are
    # branched over (zero flops).  No software prefetch (block data is
    # L2-resident).
    .p2align 6                   # MITE-fed body: pin fetch alignment
wilson_kernel_avx512_masked:
    .cfi_startproc
    pushq %rbp
    movq %rsp, %rbp
    pushq %rbx                   # callee-saved; holds the direction mask
    movl 80(%rbp), %ebx          # 15th arg = 8-bit direction mask

plus_x_m:
    testb $1, %bl
    jz plus_y_m
    # multiplication in +x direction
    # load (pointer to) U into register r11
    movq 40(%rbp), %r11
    # ferm is in %rsi
    # imaginary part is in (real part) + 64
    # ferm[s=1] is in S(384)(%rsi), etc.
    # projection of input fermion
    # temp_re[0] = re(ferm[0]) + im(ferm[3])
    vmovaps    (%rsi), %zmm0 # c = 0
    vmovaps S(128)(%rsi), %zmm1 # c = 1
    vmovaps S(256)(%rsi), %zmm2 # c = 2
    vaddps S(1216)(%rsi), %zmm0, %zmm0
    vaddps S(1344)(%rsi), %zmm1, %zmm1
    vaddps S(1472)(%rsi), %zmm2, %zmm2
    # temp_re[1] = re(ferm[1]) + im(ferm[2])
    vmovaps S(384)(%rsi), %zmm3
    vmovaps S(512)(%rsi), %zmm4
    vmovaps S(640)(%rsi), %zmm5
    vaddps S(832)(%rsi), %zmm3, %zmm3
    vaddps S(960)(%rsi), %zmm4, %zmm4
    vaddps S(1088)(%rsi), %zmm5, %zmm5
    # temp_im[0] = im(ferm[0]) - re(ferm[3])
    vmovaps S(64)(%rsi), %zmm6
    vmovaps S(192)(%rsi), %zmm7
    vmovaps S(320)(%rsi), %zmm8
    vsubps S(1152)(%rsi), %zmm6, %zmm6
    vsubps S(1280)(%rsi), %zmm7, %zmm7
    vsubps S(1408)(%rsi), %zmm8, %zmm8
    # temp_im[1] = im(ferm[1]) - re(ferm[2])
    vmovaps S(448)(%rsi), %zmm9
    vmovaps S(576)(%rsi), %zmm10
    vmovaps S(704)(%rsi), %zmm11
    vsubps S(768)(%rsi), %zmm9, %zmm9
    vsubps S(896)(%rsi), %zmm10, %zmm10
    vsubps S(1024)(%rsi), %zmm11, %zmm11

    mult_U3 0
    rmw_plus_x 0,   18, 19, 20, 21
    rmw_plus_x 128, 22, 23, 24, 25
    rmw_plus_x 256, 26, 27, 28, 29

plus_y_m:
    testb $2, %bl
    jz plus_z_m
    # multiplication in +y direction; the +y link is 72 bytes into the record
    movq 40(%rbp), %r11
    # ferm is in %rdx
    # imaginary part is in (real part) + 64
    # ferm[s=1] is in S(384)(%rdx), etc.
    # projection of input fermion
    # temp_re[0] = re(ferm[0]) + re(ferm[3])
    vmovaps    (%rdx), %zmm0 # c = 0
    vmovaps S(128)(%rdx), %zmm1 # c = 1
    vmovaps S(256)(%rdx), %zmm2 # c = 2
    vaddps S(1152)(%rdx), %zmm0, %zmm0
    vaddps S(1280)(%rdx), %zmm1, %zmm1
    vaddps S(1408)(%rdx), %zmm2, %zmm2
    # temp_re[1] = re(ferm[1]) - re(ferm[2])
    vmovaps S(384)(%rdx), %zmm3
    vmovaps S(512)(%rdx), %zmm4
    vmovaps S(640)(%rdx), %zmm5
    vsubps S(768)(%rdx), %zmm3, %zmm3
    vsubps S(896)(%rdx), %zmm4, %zmm4
    vsubps S(1024)(%rdx), %zmm5, %zmm5
    # temp_im[0] = im(ferm[0]) + im(ferm[3])
    vmovaps S(64)(%rdx), %zmm6
    vmovaps S(192)(%rdx), %zmm7
    vmovaps S(320)(%rdx), %zmm8
    vaddps S(1216)(%rdx), %zmm6, %zmm6
    vaddps S(1344)(%rdx), %zmm7, %zmm7
    vaddps S(1472)(%rdx), %zmm8, %zmm8
    # temp_im[1] = im(ferm[1]) - im(ferm[2])
    vmovaps S(448)(%rdx), %zmm9
    vmovaps S(576)(%rdx), %zmm10
    vmovaps S(704)(%rdx), %zmm11
    vsubps S(832)(%rdx), %zmm9, %zmm9
    vsubps S(960)(%rdx), %zmm10, %zmm10
    vsubps S(1088)(%rdx), %zmm11, %zmm11

    mult_U3 72
    rmw_plus_y 0,   18, 19, 20, 21
    rmw_plus_y 128, 22, 23, 24, 25
    rmw_plus_y 256, 26, 27, 28, 29

plus_z_m:
    testb $4, %bl
    jz plus_t_m
    # multiplication in +z direction; the +z link is 144 bytes into the record
    movq 40(%rbp), %r11
    # ferm is in %rcx
    # imaginary part is in (real part) + 64
    # ferm[s=1] is in S(384)(%rcx), etc.
    # projection of input fermion
    # temp_re[0] = re(ferm[0]) + im(ferm[2])
    vmovaps    (%rcx), %zmm0 # c = 0
    vmovaps S(128)(%rcx), %zmm1 # c = 1
    vmovaps S(256)(%rcx), %zmm2 # c = 2
    vaddps S(832)(%rcx), %zmm0, %zmm0
    vaddps S(960)(%rcx), %zmm1, %zmm1
    vaddps S(1088)(%rcx), %zmm2, %zmm2
    # temp_re[1] = re(ferm[1]) - im(ferm[3])
    vmovaps S(384)(%rcx), %zmm3
    vmovaps S(512)(%rcx), %zmm4
    vmovaps S(640)(%rcx), %zmm5
    vsubps S(1216)(%rcx), %zmm3, %zmm3
    vsubps S(1344)(%rcx), %zmm4, %zmm4
    vsubps S(1472)(%rcx), %zmm5, %zmm5
    # temp_im[0] = im(ferm[0]) - re(ferm[2])
    vmovaps S(64)(%rcx), %zmm6
    vmovaps S(192)(%rcx), %zmm7
    vmovaps S(320)(%rcx), %zmm8
    vsubps S(768)(%rcx), %zmm6, %zmm6
    vsubps S(896)(%rcx), %zmm7, %zmm7
    vsubps S(1024)(%rcx), %zmm8, %zmm8
    # temp_im[1] = im(ferm[1]) + re(ferm[3])
    vmovaps S(448)(%rcx), %zmm9
    vmovaps S(576)(%rcx), %zmm10
    vmovaps S(704)(%rcx), %zmm11
    vaddps S(1152)(%rcx), %zmm9, %zmm9
    vaddps S(1280)(%rcx), %zmm10, %zmm10
    vaddps S(1408)(%rcx), %zmm11, %zmm11

    mult_U3 144
    rmw_plus_z 0,   18, 19, 20, 21
    rmw_plus_z 128, 22, 23, 24, 25
    rmw_plus_z 256, 26, 27, 28, 29

plus_t_m:
    testb $8, %bl
    jz minus_x_m
    # multiplication in +t direction; the +t link is 216 bytes into the record
    movq 40(%rbp), %r11
    # ferm is in %r8
    # imaginary part is in (real part) + 64
    # ferm[s=1] is in S(384)(%r8), etc.
    # projection of input fermion
    # temp_re[0] = re(ferm[0]) - re(ferm[2])
    vmovaps    (%r8), %zmm0 # c = 0
    vmovaps S(128)(%r8), %zmm1 # c = 1
    vmovaps S(256)(%r8), %zmm2 # c = 2
    vsubps  S(768)(%r8), %zmm0, %zmm0
    vsubps  S(896)(%r8), %zmm1, %zmm1
    vsubps S(1024)(%r8), %zmm2, %zmm2
    # temp_re[1] = re(ferm[1]) - re(ferm[3])
    vmovaps S(384)(%r8), %zmm3
    vmovaps S(512)(%r8), %zmm4
    vmovaps S(640)(%r8), %zmm5
    vsubps S(1152)(%r8), %zmm3, %zmm3
    vsubps S(1280)(%r8), %zmm4, %zmm4
    vsubps S(1408)(%r8), %zmm5, %zmm5
    # temp_im[0] = im(ferm[0]) - im(ferm[2])
    vmovaps S(64)(%r8), %zmm6
    vmovaps S(192)(%r8), %zmm7
    vmovaps S(320)(%r8), %zmm8
    vsubps  S(832)(%r8), %zmm6, %zmm6
    vsubps  S(960)(%r8), %zmm7, %zmm7
    vsubps S(1088)(%r8), %zmm8, %zmm8
    # temp_im[1] = im(ferm[1]) - im(ferm[3])
    vmovaps S(448)(%r8), %zmm9
    vmovaps S(576)(%r8), %zmm10
    vmovaps S(704)(%r8), %zmm11
    vsubps S(1216)(%r8), %zmm9, %zmm9
    vsubps S(1344)(%r8), %zmm10, %zmm10
    vsubps S(1472)(%r8), %zmm11, %zmm11

    mult_U3 216
    rmw_plus_t 0,   18, 19, 20, 21
    rmw_plus_t 128, 22, 23, 24, 25
    rmw_plus_t 256, 26, 27, 28, 29

minus_x_m:
    testb $16, %bl
    jz minus_y_m
    # multiplication in -x direction: U^dag from the -x neighbor's x link
    movq 48(%rbp), %r11
    # temp_re[0] = re(ferm[0]) - im(ferm[3])
    vmovaps    (%r9), %zmm0 # c = 0
    vmovaps S(128)(%r9), %zmm1 # c = 1
    vmovaps S(256)(%r9), %zmm2 # c = 2
    vsubps S(1216)(%r9), %zmm0, %zmm0
    vsubps S(1344)(%r9), %zmm1, %zmm1
    vsubps S(1472)(%r9), %zmm2, %zmm2
    # temp_re[1] = re(ferm[1]) - im(ferm[2])
    vmovaps S(384)(%r9), %zmm3
    vmovaps S(512)(%r9), %zmm4
    vmovaps S(640)(%r9), %zmm5
    vsubps S(832)(%r9), %zmm3, %zmm3
    vsubps S(960)(%r9), %zmm4, %zmm4
    vsubps S(1088)(%r9), %zmm5, %zmm5
    # temp_im[0] = im(ferm[0]) + re(ferm[3])
    vmovaps S(64)(%r9), %zmm6
    vmovaps S(192)(%r9), %zmm7
    vmovaps S(320)(%r9), %zmm8
    vaddps S(1152)(%r9), %zmm6, %zmm6
    vaddps S(1280)(%r9), %zmm7, %zmm7
    vaddps S(1408)(%r9), %zmm8, %zmm8
    # temp_im[1] = im(ferm[1]) + re(ferm[2])
    vmovaps S(448)(%r9), %zmm9
    vmovaps S(576)(%r9), %zmm10
    vmovaps S(704)(%r9), %zmm11
    vaddps S(768)(%r9), %zmm9, %zmm9
    vaddps S(896)(%r9), %zmm10, %zmm10
    vaddps S(1024)(%r9), %zmm11, %zmm11

    mult_U3_dag 0
    rmw_minus_x 0,   18, 19, 20, 21
    rmw_minus_x 128, 22, 23, 24, 25
    rmw_minus_x 256, 26, 27, 28, 29

minus_y_m:
    testb $32, %bl
    jz minus_z_m
    # multiplication in -y direction: U^dag from the -y neighbor's y link
    movq 56(%rbp), %r11
    # load ferm into %rdx
    movq 16(%rbp), %rdx
    # imaginary part is in (real part) + 64
    # ferm[s=1] is in S(384)(%rdx), etc.
    # projection of input fermion
    # temp_re[0] = re(ferm[0]) - re(ferm[3])
    vmovaps    (%rdx), %zmm0 # c = 0
    vmovaps S(128)(%rdx), %zmm1 # c = 1
    vmovaps S(256)(%rdx), %zmm2 # c = 2
    vsubps S(1152)(%rdx), %zmm0, %zmm0
    vsubps S(1280)(%rdx), %zmm1, %zmm1
    vsubps S(1408)(%rdx), %zmm2, %zmm2
    # temp_re[1] = re(ferm[1]) + re(ferm[2])
    vmovaps S(384)(%rdx), %zmm3
    vmovaps S(512)(%rdx), %zmm4
    vmovaps S(640)(%rdx), %zmm5
    vaddps S(768)(%rdx), %zmm3, %zmm3
    vaddps S(896)(%rdx), %zmm4, %zmm4
    vaddps S(1024)(%rdx), %zmm5, %zmm5
    # temp_im[0] = im(ferm[0]) - im(ferm[3])
    vmovaps S(64)(%rdx), %zmm6
    vmovaps S(192)(%rdx), %zmm7
    vmovaps S(320)(%rdx), %zmm8
    vsubps S(1216)(%rdx), %zmm6, %zmm6
    vsubps S(1344)(%rdx), %zmm7, %zmm7
    vsubps S(1472)(%rdx), %zmm8, %zmm8
    # temp_im[1] = im(ferm[1]) + im(ferm[2])
    vmovaps S(448)(%rdx), %zmm9
    vmovaps S(576)(%rdx), %zmm10
    vmovaps S(704)(%rdx), %zmm11
    vaddps S(832)(%rdx), %zmm9, %zmm9
    vaddps S(960)(%rdx), %zmm10, %zmm10
    vaddps S(1088)(%rdx), %zmm11, %zmm11

    mult_U3_dag 72
    rmw_minus_y 0,   18, 19, 20, 21
    rmw_minus_y 128, 22, 23, 24, 25
    rmw_minus_y 256, 26, 27, 28, 29

minus_z_m:
    testb $64, %bl
    jz minus_t_m
    # multiplication in -z direction: U^dag from the -z neighbor's z link
    movq 64(%rbp), %r11
    # load ferm into %rcx
    movq 24(%rbp), %rcx
    # imaginary part is in (real part) + 64
    # ferm[s=1] is in S(384)(%rcx), etc.
    # projection of input fermion
    # temp_re[0] = re(ferm[0]) - im(ferm[2])
    vmovaps    (%rcx), %zmm0 # c = 0
    vmovaps S(128)(%rcx), %zmm1 # c = 1
    vmovaps S(256)(%rcx), %zmm2 # c = 2
    vsubps S(832)(%rcx), %zmm0, %zmm0
    vsubps S(960)(%rcx), %zmm1, %zmm1
    vsubps S(1088)(%rcx), %zmm2, %zmm2
    # temp_re[1] = re(ferm[1]) + im(ferm[3])
    vmovaps S(384)(%rcx), %zmm3
    vmovaps S(512)(%rcx), %zmm4
    vmovaps S(640)(%rcx), %zmm5
    vaddps S(1216)(%rcx), %zmm3, %zmm3
    vaddps S(1344)(%rcx), %zmm4, %zmm4
    vaddps S(1472)(%rcx), %zmm5, %zmm5
    # temp_im[0] = im(ferm[0]) + re(ferm[2])
    vmovaps S(64)(%rcx), %zmm6
    vmovaps S(192)(%rcx), %zmm7
    vmovaps S(320)(%rcx), %zmm8
    vaddps S(768)(%rcx), %zmm6, %zmm6
    vaddps S(896)(%rcx), %zmm7, %zmm7
    vaddps S(1024)(%rcx), %zmm8, %zmm8
    # temp_im[1] = im(ferm[1]) - re(ferm[3])
    vmovaps S(448)(%rcx), %zmm9
    vmovaps S(576)(%rcx), %zmm10
    vmovaps S(704)(%rcx), %zmm11
    vsubps S(1152)(%rcx), %zmm9, %zmm9
    vsubps S(1280)(%rcx), %zmm10, %zmm10
    vsubps S(1408)(%rcx), %zmm11, %zmm11

    mult_U3_dag 144
    rmw_minus_z 0,   18, 19, 20, 21
    rmw_minus_z 128, 22, 23, 24, 25
    rmw_minus_z 256, 26, 27, 28, 29

minus_t_m:
    testb $128, %bl
    jz finish_m
    # multiplication in -t direction: U^dag from the -t neighbor's t link
    movq 72(%rbp), %r11
    # load ferm into %r8
    movq 32(%rbp), %r8
    # imaginary part is in (real part) + 64
    # ferm[s=1] is in S(384)(%r8), etc.
    # projection of input fermion
    # temp_re[0] = re(ferm[0]) + re(ferm[2])
    vmovaps    (%r8), %zmm0 # c = 0
    vmovaps S(128)(%r8), %zmm1 # c = 1
    vmovaps S(256)(%r8), %zmm2 # c = 2
    vaddps  S(768)(%r8), %zmm0, %zmm0
    vaddps  S(896)(%r8), %zmm1, %zmm1
    vaddps S(1024)(%r8), %zmm2, %zmm2
    # temp_re[1] = re(ferm[1]) + re(ferm[3])
    vmovaps S(384)(%r8), %zmm3
    vmovaps S(512)(%r8), %zmm4
    vmovaps S(640)(%r8), %zmm5
    vaddps S(1152)(%r8), %zmm3, %zmm3
    vaddps S(1280)(%r8), %zmm4, %zmm4
    vaddps S(1408)(%r8), %zmm5, %zmm5
    # temp_im[0] = im(ferm[0]) + im(ferm[2])
    vmovaps S(64)(%r8), %zmm6
    vmovaps S(192)(%r8), %zmm7
    vmovaps S(320)(%r8), %zmm8
    vaddps  S(832)(%r8), %zmm6, %zmm6
    vaddps  S(960)(%r8), %zmm7, %zmm7
    vaddps S(1088)(%r8), %zmm8, %zmm8
    # temp_im[1] = im(ferm[1]) + im(ferm[3])
    vmovaps S(448)(%r8), %zmm9
    vmovaps S(576)(%r8), %zmm10
    vmovaps S(704)(%r8), %zmm11
    vaddps S(1216)(%r8), %zmm9, %zmm9
    vaddps S(1344)(%r8), %zmm10, %zmm10
    vaddps S(1472)(%r8), %zmm11, %zmm11

    mult_U3_dag 216
    rmw_minus_t 0,   18, 19, 20, 21
    rmw_minus_t 128, 22, 23, 24, 25
    rmw_minus_t 256, 26, 27, 28, 29

finish_m:
    popq %rbx
    popq %rbp
    ret
    .cfi_endproc
    .size wilson_kernel_avx512_masked, .-wilson_kernel_avx512_masked
# === END generated wilson_kernel_avx512_masked ===
    .section .note.GNU-stack,"",@progbits
