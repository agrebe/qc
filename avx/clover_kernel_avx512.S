// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

// =========================================================================
// AVX-512 single-precision clover kernel, scheduled for accumulator ILP.
//
// The instruction order was emitted once by a scheduling script kept outside
// this tree; there is no Makefile rule that regenerates it, and nothing to
// run.  This .S is the maintained artifact -- edit it here, keeping the
// 1-factorization described below intact.
//
// Per chiral block: 132 FMAs, 36 broadcasts, 12 ferm loads, 12 stores, and
// 12 output loads in the accumulating variant only -- the algebraic minimum.
// The order is what this schedule buys: consecutive FMAs into the same
// accumulator are 12 FMAs (6 cycles) apart rather than a median of 4 FMAs
// (2 cycles), which is below the 4-cycle FMA latency and leaves the two FMA
// ports unsaturated.
//
// The 15 off-diagonal row pairs are grouped by a 1-factorization of K6 into 5
// rounds of 3 mutually disjoint pairs.  Each round touches all 12 accumulators
// (2 FMAs each) and needs only 6 distinct F values, so round-robin emission
// inside a round gives a uniform gap of 12 while each F value is still
// broadcast exactly once and consumed all 4 times within its own round.  The 6
// real diagonal entries form a 6th round.  Registers: zmm16-27 accumulators,
// zmm4-15 ferm, zmm0-3/zmm28-31 rotating F-broadcast pool -- all 32.
//
// The prologue is ordered for the same reason the body is.  An FMA cannot
// issue until its last operand arrives, so leading with all 24 loads would
// leave the F broadcast -- the operand emitted last -- gating the first FMA,
// and both FMA ports idle for the fifteen cycles the loads take to issue.
// Group 0 touches each accumulator exactly once, so its broadcasts come
// first and each accumulator's two loads and first FMA are emitted together:
// the first FMA then waits on two loads instead of thirty.
// =========================================================================
#include "global_params.h"   // num_rhs/Nc/Ns;
                             // C-only tail is __ASSEMBLER__-guarded
#if (num_rhs % 16) != 0
#error "clover_kernel_avx512.S needs num_rhs % 16 == 0 (one-zmm-tile granularity)"
#endif
// component stride 8*num_rhs, imag stride 4*num_rhs.
#define S(D)  D/128*8*num_rhs + D%128/64*4*num_rhs + D%64

// The two variants differ only in how each accumulator is first touched.
// Accumulating: load the caller's output, then multiply-add into it.
// Overwriting: skip the load and make that first multiply-add a plain
// multiply -- group 0 writes every accumulator exactly once, so the twelve
// zeroing instructions the caller would otherwise need never exist.  They
// would have cost twelve uops on ports 0 and 5, the same two the FMAs
// issue on, and a 512-bit vpxord is not rename-eliminated on Skylake-X /
// Cascade Lake, so each would also have inherited its register's previous
// chain.  A multiply writing a fresh destination has neither problem.
#if CLOVER_OW
#define KERNEL_NAME clover_kernel_avx512_ow
#define OUT_FIRST(MEM, REG)
#define ACC_FIRST(FERM, F, ACC) vmulps FERM, F, ACC
#else
#define KERNEL_NAME clover_kernel_avx512
#define OUT_FIRST(MEM, REG) vmovaps MEM, REG
#define ACC_FIRST(FERM, F, ACC) vfmadd231ps FERM, F, ACC
#endif

    .globl KERNEL_NAME
    .type KERNEL_NAME, @function
    # arguments:
    # float * output:  %rdi
    # float * ferm:    %rsi
    # float * F:       %rdx
    # all of output, ferm, and F have been advanced to right position
KERNEL_NAME:
    .cfi_startproc

#if num_rhs > 16
    mov %rdi, %r10
    mov %rsi, %r11
    mov %rdx, %r8
    xor %r9, %r9
tile_loop:
    lea (%r10,%r9,1), %rdi
    lea (%r11,%r9,1), %rsi
    mov %r8, %rdx
#endif

    xor %rax, %rax
multiply_block:
    # ---- group 0: 6 real diagonal entries -- 12 FMAs over all 12 accumulators, gap 12
    vbroadcastss (%rdx), %zmm0
    vbroadcastss 28(%rdx), %zmm1
    vbroadcastss 56(%rdx), %zmm2
    vbroadcastss 84(%rdx), %zmm3
    vbroadcastss 112(%rdx), %zmm28
    vbroadcastss 140(%rdx), %zmm29
    # ferm + output loads interleaved with the first FMA into
    # each accumulator (ferm zmm4-15, accumulators zmm16-27)
    vmovaps (%rsi), %zmm4  # re(ferm[0])
    OUT_FIRST((%rdi), %zmm16)  # re(output[0])
    ACC_FIRST(%zmm4, %zmm0, %zmm16)
    vmovaps S(64)(%rsi), %zmm5  # im(ferm[0])
    OUT_FIRST(S(64)(%rdi), %zmm17)  # im(output[0])
    ACC_FIRST(%zmm5, %zmm0, %zmm17)
    vmovaps S(128)(%rsi), %zmm6  # re(ferm[1])
    OUT_FIRST(S(128)(%rdi), %zmm18)  # re(output[1])
    ACC_FIRST(%zmm6, %zmm1, %zmm18)
    vmovaps S(192)(%rsi), %zmm7  # im(ferm[1])
    OUT_FIRST(S(192)(%rdi), %zmm19)  # im(output[1])
    ACC_FIRST(%zmm7, %zmm1, %zmm19)
    vmovaps S(256)(%rsi), %zmm8  # re(ferm[2])
    OUT_FIRST(S(256)(%rdi), %zmm20)  # re(output[2])
    ACC_FIRST(%zmm8, %zmm2, %zmm20)
    vmovaps S(320)(%rsi), %zmm9  # im(ferm[2])
    OUT_FIRST(S(320)(%rdi), %zmm21)  # im(output[2])
    ACC_FIRST(%zmm9, %zmm2, %zmm21)
    vmovaps S(384)(%rsi), %zmm10  # re(ferm[3])
    OUT_FIRST(S(384)(%rdi), %zmm22)  # re(output[3])
    ACC_FIRST(%zmm10, %zmm3, %zmm22)
    vmovaps S(448)(%rsi), %zmm11  # im(ferm[3])
    OUT_FIRST(S(448)(%rdi), %zmm23)  # im(output[3])
    ACC_FIRST(%zmm11, %zmm3, %zmm23)
    vmovaps S(512)(%rsi), %zmm12  # re(ferm[4])
    OUT_FIRST(S(512)(%rdi), %zmm24)  # re(output[4])
    ACC_FIRST(%zmm12, %zmm28, %zmm24)
    vmovaps S(576)(%rsi), %zmm13  # im(ferm[4])
    OUT_FIRST(S(576)(%rdi), %zmm25)  # im(output[4])
    ACC_FIRST(%zmm13, %zmm28, %zmm25)
    vmovaps S(640)(%rsi), %zmm14  # re(ferm[5])
    OUT_FIRST(S(640)(%rdi), %zmm26)  # re(output[5])
    ACC_FIRST(%zmm14, %zmm29, %zmm26)
    vmovaps S(704)(%rsi), %zmm15  # im(ferm[5])
    OUT_FIRST(S(704)(%rdi), %zmm27)  # im(output[5])
    ACC_FIRST(%zmm15, %zmm29, %zmm27)

    # ---- group 1: row pairs (0,5) (1,4) (2,3) -- 24 FMAs over all 12
    # accumulators, gap 12
    vbroadcastss 120(%rdx), %zmm30
    vbroadcastss 20(%rdx), %zmm31
    vbroadcastss 40(%rdx), %zmm0
    vbroadcastss 100(%rdx), %zmm1
    vbroadcastss 60(%rdx), %zmm2
    vbroadcastss 80(%rdx), %zmm3
    vfnmadd231ps %zmm15, %zmm30, %zmm16
    vfmadd231ps %zmm15, %zmm31, %zmm17
    vfmadd231ps %zmm12, %zmm0, %zmm18
    vfmadd231ps %zmm12, %zmm1, %zmm19
    vfmadd231ps %zmm10, %zmm2, %zmm20
    vfmadd231ps %zmm11, %zmm2, %zmm21
    vfmadd231ps %zmm8, %zmm2, %zmm22
    vfmadd231ps %zmm9, %zmm2, %zmm23
    vfmadd231ps %zmm6, %zmm0, %zmm24
    vfnmadd231ps %zmm6, %zmm1, %zmm25
    vfmadd231ps %zmm5, %zmm30, %zmm26
    vfmadd231ps %zmm5, %zmm31, %zmm27
    vfmadd231ps %zmm14, %zmm31, %zmm16
    vfmadd231ps %zmm14, %zmm30, %zmm17
    vfnmadd231ps %zmm13, %zmm1, %zmm18
    vfmadd231ps %zmm13, %zmm0, %zmm19
    vfnmadd231ps %zmm11, %zmm3, %zmm20
    vfmadd231ps %zmm10, %zmm3, %zmm21
    vfmadd231ps %zmm9, %zmm3, %zmm22
    vfnmadd231ps %zmm8, %zmm3, %zmm23
    vfmadd231ps %zmm7, %zmm1, %zmm24
    vfmadd231ps %zmm7, %zmm0, %zmm25
    vfmadd231ps %zmm4, %zmm31, %zmm26
    vfnmadd231ps %zmm4, %zmm30, %zmm27

    # ---- group 2: row pairs (1,5) (0,2) (3,4) -- 24 FMAs over all 12
    # accumulators, gap 12
    vbroadcastss 44(%rdx), %zmm28
    vbroadcastss 124(%rdx), %zmm29
    vbroadcastss 8(%rdx), %zmm30
    vbroadcastss 48(%rdx), %zmm31
    vbroadcastss 88(%rdx), %zmm0
    vbroadcastss 108(%rdx), %zmm1
    vfmadd231ps %zmm8, %zmm30, %zmm16
    vfmadd231ps %zmm8, %zmm31, %zmm17
    vfmadd231ps %zmm14, %zmm28, %zmm18
    vfmadd231ps %zmm14, %zmm29, %zmm19
    vfmadd231ps %zmm4, %zmm30, %zmm20
    vfnmadd231ps %zmm4, %zmm31, %zmm21
    vfmadd231ps %zmm12, %zmm0, %zmm22
    vfmadd231ps %zmm12, %zmm1, %zmm23
    vfmadd231ps %zmm10, %zmm0, %zmm24
    vfnmadd231ps %zmm10, %zmm1, %zmm25
    vfmadd231ps %zmm6, %zmm28, %zmm26
    vfnmadd231ps %zmm6, %zmm29, %zmm27
    vfnmadd231ps %zmm9, %zmm31, %zmm16
    vfmadd231ps %zmm9, %zmm30, %zmm17
    vfnmadd231ps %zmm15, %zmm29, %zmm18
    vfmadd231ps %zmm15, %zmm28, %zmm19
    vfmadd231ps %zmm5, %zmm31, %zmm20
    vfmadd231ps %zmm5, %zmm30, %zmm21
    vfnmadd231ps %zmm13, %zmm1, %zmm22
    vfmadd231ps %zmm13, %zmm0, %zmm23
    vfmadd231ps %zmm11, %zmm1, %zmm24
    vfmadd231ps %zmm11, %zmm0, %zmm25
    vfmadd231ps %zmm7, %zmm29, %zmm26
    vfmadd231ps %zmm7, %zmm28, %zmm27

    # ---- group 3: row pairs (2,5) (1,3) (0,4) -- 24 FMAs over all 12
    # accumulators, gap 12
    vbroadcastss 128(%rdx), %zmm2
    vbroadcastss 68(%rdx), %zmm3
    vbroadcastss 36(%rdx), %zmm28
    vbroadcastss 76(%rdx), %zmm29
    vbroadcastss 16(%rdx), %zmm30
    vbroadcastss 96(%rdx), %zmm31
    vfmadd231ps %zmm12, %zmm30, %zmm16
    vfmadd231ps %zmm12, %zmm31, %zmm17
    vfmadd231ps %zmm10, %zmm28, %zmm18
    vfmadd231ps %zmm10, %zmm29, %zmm19
    vfnmadd231ps %zmm15, %zmm2, %zmm20
    vfmadd231ps %zmm15, %zmm3, %zmm21
    vfmadd231ps %zmm6, %zmm28, %zmm22
    vfnmadd231ps %zmm6, %zmm29, %zmm23
    vfmadd231ps %zmm4, %zmm30, %zmm24
    vfnmadd231ps %zmm4, %zmm31, %zmm25
    vfmadd231ps %zmm9, %zmm2, %zmm26
    vfmadd231ps %zmm9, %zmm3, %zmm27
    vfnmadd231ps %zmm13, %zmm31, %zmm16
    vfmadd231ps %zmm13, %zmm30, %zmm17
    vfnmadd231ps %zmm11, %zmm29, %zmm18
    vfmadd231ps %zmm11, %zmm28, %zmm19
    vfmadd231ps %zmm14, %zmm3, %zmm20
    vfmadd231ps %zmm14, %zmm2, %zmm21
    vfmadd231ps %zmm7, %zmm29, %zmm22
    vfmadd231ps %zmm7, %zmm28, %zmm23
    vfmadd231ps %zmm5, %zmm31, %zmm24
    vfmadd231ps %zmm5, %zmm30, %zmm25
    vfmadd231ps %zmm8, %zmm3, %zmm26
    vfnmadd231ps %zmm8, %zmm2, %zmm27

    # ---- group 4: row pairs (3,5) (2,4) (0,1) -- 24 FMAs over all 12
    # accumulators, gap 12
    vbroadcastss 92(%rdx), %zmm0
    vbroadcastss 132(%rdx), %zmm1
    vbroadcastss 64(%rdx), %zmm2
    vbroadcastss 104(%rdx), %zmm3
    vbroadcastss 4(%rdx), %zmm28
    vbroadcastss 24(%rdx), %zmm29
    vfmadd231ps %zmm6, %zmm28, %zmm16
    vfmadd231ps %zmm7, %zmm28, %zmm17
    vfmadd231ps %zmm4, %zmm28, %zmm18
    vfmadd231ps %zmm5, %zmm28, %zmm19
    vfmadd231ps %zmm12, %zmm2, %zmm20
    vfmadd231ps %zmm12, %zmm3, %zmm21
    vfmadd231ps %zmm14, %zmm0, %zmm22
    vfmadd231ps %zmm14, %zmm1, %zmm23
    vfmadd231ps %zmm8, %zmm2, %zmm24
    vfnmadd231ps %zmm8, %zmm3, %zmm25
    vfmadd231ps %zmm10, %zmm0, %zmm26
    vfnmadd231ps %zmm10, %zmm1, %zmm27
    vfnmadd231ps %zmm7, %zmm29, %zmm16
    vfmadd231ps %zmm6, %zmm29, %zmm17
    vfmadd231ps %zmm5, %zmm29, %zmm18
    vfnmadd231ps %zmm4, %zmm29, %zmm19
    vfnmadd231ps %zmm13, %zmm3, %zmm20
    vfmadd231ps %zmm13, %zmm2, %zmm21
    vfnmadd231ps %zmm15, %zmm1, %zmm22
    vfmadd231ps %zmm15, %zmm0, %zmm23
    vfmadd231ps %zmm9, %zmm3, %zmm24
    vfmadd231ps %zmm9, %zmm2, %zmm25
    vfmadd231ps %zmm11, %zmm1, %zmm26
    vfmadd231ps %zmm11, %zmm0, %zmm27

    # ---- group 5: row pairs (4,5) (0,3) (1,2) -- 24 FMAs over all 12
    # accumulators, gap 12
    vbroadcastss 116(%rdx), %zmm30
    vbroadcastss 136(%rdx), %zmm31
    vbroadcastss 12(%rdx), %zmm0
    vbroadcastss 72(%rdx), %zmm1
    vbroadcastss 32(%rdx), %zmm2
    vbroadcastss 52(%rdx), %zmm3
    vfmadd231ps %zmm10, %zmm0, %zmm16
    vfmadd231ps %zmm10, %zmm1, %zmm17
    vfmadd231ps %zmm8, %zmm2, %zmm18
    vfmadd231ps %zmm8, %zmm3, %zmm19
    vfmadd231ps %zmm6, %zmm2, %zmm20
    vfnmadd231ps %zmm6, %zmm3, %zmm21
    vfmadd231ps %zmm4, %zmm0, %zmm22
    vfnmadd231ps %zmm4, %zmm1, %zmm23
    vfmadd231ps %zmm14, %zmm30, %zmm24
    vfmadd231ps %zmm15, %zmm30, %zmm25
    vfmadd231ps %zmm12, %zmm30, %zmm26
    vfmadd231ps %zmm13, %zmm30, %zmm27
    vfnmadd231ps %zmm11, %zmm1, %zmm16
    vfmadd231ps %zmm11, %zmm0, %zmm17
    vfnmadd231ps %zmm9, %zmm3, %zmm18
    vfmadd231ps %zmm9, %zmm2, %zmm19
    vfmadd231ps %zmm7, %zmm3, %zmm20
    vfmadd231ps %zmm7, %zmm2, %zmm21
    vfmadd231ps %zmm5, %zmm1, %zmm22
    vfmadd231ps %zmm5, %zmm0, %zmm23
    vfnmadd231ps %zmm15, %zmm31, %zmm24
    vfmadd231ps %zmm14, %zmm31, %zmm25
    vfmadd231ps %zmm13, %zmm31, %zmm26
    vfnmadd231ps %zmm12, %zmm31, %zmm27

    # single batch of stores: no accumulator ever left a register
    vmovaps %zmm16, (%rdi)
    vmovaps %zmm17, S(64)(%rdi)
    vmovaps %zmm18, S(128)(%rdi)
    vmovaps %zmm19, S(192)(%rdi)
    vmovaps %zmm20, S(256)(%rdi)
    vmovaps %zmm21, S(320)(%rdi)
    vmovaps %zmm22, S(384)(%rdi)
    vmovaps %zmm23, S(448)(%rdi)
    vmovaps %zmm24, S(512)(%rdi)
    vmovaps %zmm25, S(576)(%rdi)
    vmovaps %zmm26, S(640)(%rdi)
    vmovaps %zmm27, S(704)(%rdi)

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
    add $64, %r9
    cmp $num_rhs/16*64, %r9
    jl tile_loop
#endif
finish:
    endbr64
    ret
    .cfi_endproc
    .size KERNEL_NAME, .-KERNEL_NAME
    .section .note.GNU-stack,"",@progbits
