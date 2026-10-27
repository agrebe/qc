#!/usr/bin/env python3
# Copyright 2026, University of Maryland, All Rights Reserved
# SPDX-License-Identifier: GPL-3.0-or-later

# Generator for wilson_kernel_avx2.s
#
# Emits an AVX2 (single-precision) Wilson hopping kernel that processes all
# num_rhs columns of a site in one call.  num_rhs columns = NHALF 8-wide ymm
# blocks; each direction loops over those blocks internally (see NHALF below).
#
# Why a generator: AVX2 has only 16 ymm registers (vs 32 zmm), so unlike the
# AVX-512 kernel we cannot keep the whole 12-element half-spinor "temp" plus
# all accumulators live at once.  We instead process each direction one
# spin-component at a time -- the two projected components of a direction
# reconstruct into disjoint output spins, so they are independent.  That needs
# only 6 temps + 6 prod + 2 U-broadcast + scratch, which fits in 16 ymm.
#
# Arithmetic (projection signs, U vs U-dagger, reconstruction signs) is taken
# directly from the verified AVX-512 kernel / the C macros in clover_ferm.c.
# Correctness is checked by test_wilson_asm against wilson_kernel_c.
#
# All byte strides below are derived from num_rhs and Nc read from
# global_params.h, so changing num_rhs (for tuning) regenerates a correct
# kernel.  Fermion site layout (bytes):
#   re/im stride  = num_rhs*4
#   color stride  = 2*num_rhs*4
#   spin stride   = Nc*2*num_rhs*4
# One ymm holds 8 columns (32 bytes); the num_rhs columns of a (spin,color,
# re/im) are contiguous, and the internal half-loop walks them 32 bytes at a
# time.  vmovaps is safe: the site base is 64-byte aligned and we only ever
# shift by 32.
# Gauge: 4 link matrices per site, each Nc*Nc complex; dir stride Nc*Nc*2*4.
# Note: the clover kernels and wilson_kernel_avx512 are not generated here --
# they are cpp/GAS templates (.S) that read num_rhs from global_params.h
# themselves.

import os

# Prefetch tuning knobs (env-overridable for sweeps; defaults reproduce the
# original schedule exactly):
#   PF_AHEAD       - steady-state lookahead in *directions* (2 = current).
#   PF_LINE_STRIDE - prefetch every Nth 64-byte line of a neighbor fermion site
#                    (1 = every line = current; 2/4 lean on the HW streamer).
PF_AHEAD       = int(os.environ.get("PF_AHEAD", "2"))
PF_LINE_STRIDE = int(os.environ.get("PF_LINE_STRIDE", "1"))
assert PF_AHEAD >= 1 and PF_LINE_STRIDE >= 1
# Prefetch target level: t0 = L1 (+L2/L3), t1 = L2 (+L3, not L1), t2 = L3,
# nta = non-temporal.  For a streaming working set that can't fit L1, t1 often
# beats t0 (hides DRAM/L3 latency without evicting the hot output accumulator).
PF_HINT = os.environ.get("PF_HINT", "t0").lower()
PF_INSN = {"t0": "prefetcht0", "t1": "prefetcht1",
           "t2": "prefetcht2", "nta": "prefetchnta"}[PF_HINT]

# Read the tunable parameters so this kernel stays in sync when num_rhs is
# changed for tuning.  hdr_int asks the C preprocessor, so it does not matter
# which header holds a given #define; the Makefile lists the headers as
# prerequisites of wilson_kernel_avx2.s, so editing one triggers regen + a
# rebuild automatically.
from hdr_params import hdr_int
num_rhs = hdr_int("num_rhs")
Nc      = hdr_int("Nc")
Ns      = hdr_int("Ns")

# This generator emits AVX2 single-precision code: 8 floats (32 bytes) per ymm.
VECLEN = 8
FLOAT  = 4
assert Nc == 3, "register allocation assumes Nc == 3 (3 temps + 3 prods per re/im)"
assert Ns == 4, "Wilson kernel assumes Ns == 4 spins"
assert num_rhs % VECLEN == 0, \
    f"num_rhs ({num_rhs}) must be a multiple of {VECLEN} for AVX2 single precision"

# Fermion site layout (bytes):
# index = ((spin*Nc + color)*2 + reim)*num_rhs + col
IMAG  = num_rhs * FLOAT           # re -> im of one (spin,color)
COLOR = 2 * num_rhs * FLOAT       # successive colors
SPIN  = Nc * 2 * num_rhs * FLOAT  # successive spins
GAUGE_DIR = Nc * Nc * 2 * FLOAT   # successive link directions within a site

# A full fermion neighbor site is Ns*Nc complex * num_rhs floats; we prefetch
# it one cacheline (64 B) at a time.  Ns*Nc*2*num_rhs*4 is always a multiple of
# 64 for num_rhs % 8 == 0, so FERM_LINES is exact.
FERM_SITE_BYTES = Ns * Nc * 2 * num_rhs * FLOAT
assert FERM_SITE_BYTES % 64 == 0
FERM_LINES = FERM_SITE_BYTES // 64

# num_rhs columns = NHALF 8-wide ymm blocks ("halves"), HALF_STRIDE bytes
# apart.  Each direction loops over the halves internally rather than the C
# dispatcher calling us once per half: that removes the call boundary between
# halves, so the out-of-order engine can overlap them, and keeps each
# direction's body compact.
NHALF = num_rhs // VECLEN
HALF_STRIDE = VECLEN * FLOAT      # = 32


def part_off(spin, part, color):
    return spin * SPIN + color * COLOR + (IMAG if part == 'im' else 0)


# A spin-pair operation, decoded from the C macros:
#   temp_re[c] = f[sA,'re',c]  op1  f[sB, bRe, c]
#   temp_im[c] = f[sA,'im',c]  op2  f[sB, bIm, c]
# reconstruct:
#   out[sA,'re'] += pr ; out[sA,'im'] += pi
#   out[sB,bRe]  op1 pr ; out[sB,bIm] op2 pi
# A_OP_B(a,b,OP):   bRe='re', bIm='im', op1=op2=OP
# A_MINUS_BI(a,b):  bRe='im', bIm='re', op1='+', op2='-'
# A_PLUS_BI(a,b):   bRe='im', bIm='re', op1='-', op2='+'
def OPB(a, b, op):
  return dict(sA=a, sB=b, bRe='re', bIm='im', op1=op, op2=op)
def MBI(a, b):
  return dict(sA=a, sB=b, bRe='im', bIm='re', op1='+', op2='-')
def PBI(a, b):
  return dict(sA=a, sB=b, bRe='im', bIm='re', op1='-', op2='+')

# Direction table.  ferm = GPR or stack ref holding the input fermion pointer;
# gauge = stack ref of the link base; diroff = byte offset of this direction's
# link within the site; dag = use U-dagger.
DIRS = [
    dict(name='plus_x',  ferm='%rsi',      gauge='40(%rbp)',
         diroff=0*GAUGE_DIR, dag=False, pairs=[MBI(0,3), MBI(1,2)]),
    dict(name='plus_y',  ferm='%rdx',      gauge='40(%rbp)',
         diroff=1*GAUGE_DIR, dag=False, pairs=[OPB(0,3,'+'), OPB(1,2,'-')]),
    dict(name='plus_z',  ferm='%rcx',      gauge='40(%rbp)',
         diroff=2*GAUGE_DIR, dag=False, pairs=[MBI(0,2), PBI(1,3)]),
    dict(name='plus_t',  ferm='%r8',       gauge='40(%rbp)',
         diroff=3*GAUGE_DIR, dag=False, pairs=[OPB(0,2,'-'), OPB(1,3,'-')]),
    dict(name='minus_x', ferm='%r9',       gauge='48(%rbp)',
         diroff=0*GAUGE_DIR, dag=True,  pairs=[PBI(0,3), PBI(1,2)]),
    dict(name='minus_y', ferm='16(%rbp)',  gauge='56(%rbp)',
         diroff=1*GAUGE_DIR, dag=True,  pairs=[OPB(0,3,'-'), OPB(1,2,'+')]),
    dict(name='minus_z', ferm='24(%rbp)',  gauge='64(%rbp)',
         diroff=2*GAUGE_DIR, dag=True,  pairs=[PBI(0,2), MBI(1,3)]),
    dict(name='minus_t', ferm='32(%rbp)',  gauge='72(%rbp)',
         diroff=3*GAUGE_DIR, dag=True,  pairs=[OPB(0,2,'+'), OPB(1,3,'+')]),
]

# ymm assignment within one spin-component pass.
# We compute all three output colors' prods at once so the scheduler sees 6
# independent FMA accumulator chains (3 colors x re/im).
TRE = ['%ymm0', '%ymm1', '%ymm2']             # temp_re[input color]
TIM = ['%ymm3', '%ymm4', '%ymm5']             # temp_im[input color]
PRODRE = ['%ymm6', '%ymm7', '%ymm8']          # prod_re[output color c1]
PRODIM = ['%ymm9', '%ymm10', '%ymm11']        # prod_im[output color c1]
URE = '%ymm12'                                # broadcast Re(U)
UIM = '%ymm13'                                # broadcast Im(U)
SC  = '%ymm14'                                # scratch for read-modify-write

out = []
def emit(s=''):
    out.append(s)


def gen_projection(p):
    emit("    # projection: temp_re/temp_im for this spin-component")
    addsub1 = 'vaddps' if p['op1'] == '+' else 'vsubps'
    addsub2 = 'vaddps' if p['op2'] == '+' else 'vsubps'
    for c in range(Nc):
        aRe = part_off(p['sA'], 're', c)
        bRe = part_off(p['sB'], p['bRe'], c)
        emit(f"    vmovaps {aRe}(%r10), {TRE[c]}")
        emit(f"    {addsub1} {bRe}(%r10), {TRE[c]}, {TRE[c]}")
    for c in range(Nc):
        aIm = part_off(p['sA'], 'im', c)
        bIm = part_off(p['sB'], p['bIm'], c)
        emit(f"    vmovaps {aIm}(%r10), {TIM[c]}")
        emit(f"    {addsub2} {bIm}(%r10), {TIM[c]}, {TIM[c]}")


def uoff(c1, c2, diroff, dag):
    # U[c1][c2] for non-dag, U[c2][c1] for dag (column access)
    row, col = (c2, c1) if dag else (c1, c2)
    return diroff + (row * Nc + col) * 2 * FLOAT


def gen_multiply(p, diroff, dag):
    emit("    # prod[c1] = sum_c2 U[c1][c2] . temp[c2], all output colors at once")
    for c2 in range(Nc):
        for c1 in range(Nc):
            o = uoff(c1, c2, diroff, dag)
            emit(f"    vbroadcastss {o}(%r11), {URE}")
            emit(f"    vbroadcastss {o+FLOAT}(%r11), {UIM}")
            mul = 'vmulps' if c2 == 0 else 'vfmadd231ps'
            emit(f"    {mul} {URE}, {TRE[c2]}, {PRODRE[c1]}")
            emit(f"    {mul} {URE}, {TIM[c2]}, {PRODIM[c1]}")
            if not dag:
                emit(f"    vfnmadd231ps {UIM}, {TIM[c2]}, {PRODRE[c1]}")
                emit(f"    vfmadd231ps  {UIM}, {TRE[c2]}, {PRODIM[c1]}")
            else:
                emit(f"    vfmadd231ps  {UIM}, {TIM[c2]}, {PRODRE[c1]}")
                emit(f"    vfnmadd231ps {UIM}, {TRE[c2]}, {PRODIM[c1]}")


def _emit_accumulate(p, PR, PI, aRe, aIm, bRe, bIm):
    emit(f"    vaddps {aRe}(%rdi), {PR}, {SC}")
    emit(f"    vmovaps {SC}, {aRe}(%rdi)")
    emit(f"    vaddps {aIm}(%rdi), {PI}, {SC}")
    emit(f"    vmovaps {SC}, {aIm}(%rdi)")
    if p['op1'] == '+':
        emit(f"    vaddps {bRe}(%rdi), {PR}, {SC}")
    else:
        emit(f"    vmovaps {bRe}(%rdi), {SC}")
        emit(f"    vsubps {PR}, {SC}, {SC}")
    emit(f"    vmovaps {SC}, {bRe}(%rdi)")
    if p['op2'] == '+':
        emit(f"    vaddps {bIm}(%rdi), {PI}, {SC}")
    else:
        emit(f"    vmovaps {bIm}(%rdi), {SC}")
        emit(f"    vsubps {PI}, {SC}, {SC}")
    emit(f"    vmovaps {SC}, {bIm}(%rdi)")

def gen_reconstruct(p):
    emit(f"    # reconstruct into output spins {p['sA']} and {p['sB']}, all colors")
    for c1 in range(Nc):
        PR, PI = PRODRE[c1], PRODIM[c1]
        aRe = part_off(p['sA'], 're', c1); aIm = part_off(p['sA'], 'im', c1)
        bRe = part_off(p['sB'], p['bRe'], c1); bIm = part_off(p['sB'], p['bIm'], c1)
        _emit_accumulate(p, PR, PI, aRe, aIm, bRe, bIm)


# Software-prefetch schedule: while computing direction i, pull the fermion
# site(s) + gauge link(s) that a direction PF_AHEAD steps later will need into
# L1.  Direction 0 primes the pipeline by issuing the first PF_AHEAD targets at
# once; each later direction issues just the one newly-exposed target.  At
# PF_AHEAD=2 this is the schedule {0:[1,2],1:[3],...,5:[7]}, mirroring
# wilson_kernel_c.  It pays off on the memory-bound full-lattice sweep; on the
# cache-resident single-site test the data is already hot, so it only adds uops.
_ND = 8  # number of directions
PREFETCH_SCHEDULE = {}
for _i in range(_ND):
    if _i == 0:
        _t = [j for j in range(1, 1 + PF_AHEAD) if j < _ND]
    else:
        _t = [_i + PF_AHEAD] if _i + PF_AHEAD < _ND else []
    if _t:
        PREFETCH_SCHEDULE[_i] = _t
# r12 is callee-saved and otherwise unused here; we borrow it as the prefetch
# base pointer (only when WK_PREFETCH is defined, so the non-prefetch build is
# byte-for-byte unchanged).
PF = '%r12'


def gen_prefetch(i):
    targets = PREFETCH_SCHEDULE.get(i, [])
    if not targets:
        return
    emit(".ifdef WK_PREFETCH")
    for j in targets:
        dj = DIRS[j]
        # fermion site: FERM_LINES cachelines.  Register-held neighbor pointers
        # (rsi/rdx/rcx/r8/r9) are pristine here, so address them directly;
        # stack-held ones must be loaded into the scratch register first.
        emit(f"    # prefetch {dj['name']} fermion site (next-direction input)")
        if dj['ferm'].startswith('%'):
            base = dj['ferm']
        else:
            emit(f"    movq {dj['ferm']}, {PF}")
            base = PF
        for off in range(0, FERM_SITE_BYTES, 64 * PF_LINE_STRIDE):
            emit(f"    {PF_INSN} {off}({base})")
        # gauge link for this direction: 72 B spans <=2 lines.
        emit(f"    # prefetch {dj['name']} gauge link")
        emit(f"    movq {dj['gauge']}, {PF}")
        emit(f"    {PF_INSN} {dj['diroff']}({PF})")
        emit(f"    {PF_INSN} {dj['diroff'] + 64}({PF})")
    emit(".endif")


# Emit one kernel body.  masked=False -> the original full-lattice kernel
# (software prefetch, all 8 directions unconditionally).  masked=True -> a
# Schwarz variant taking an 8-bit direction mask (15th arg, 80(rbp)): each
# direction is preceded by a test of its mask bit and a jump over the block if
# the bit is clear, so disabled directions cost only a (well-predicted) branch
# and no flops.  No prefetch (Schwarz block data is L2-resident).  Labels carry
# a "_m" suffix so the two kernels can coexist in one object.
def emit_kernel(name, masked):
    sfx = "_m" if masked else ""
    emit(f"    .globl {name}")
    emit(f"    .type {name}, @function")
    emit("    # args: rdi=output, rsi=+x, rdx=+y, rcx=+z, r8=+t, r9=-x ferm;")
    emit("    # 16(rbp)=-y 24=-z 32=-t ferm; 40=U 48=U-x 56=U-y 64=U-z 72=U-t"
         + ("; 80=dir mask (bit i = perform DIRS[i])" if masked else ""))
    emit(f"{name}:")
    emit("    .cfi_startproc")
    emit("    endbr64")
    emit("    pushq %rbp")
    emit("    movq %rsp, %rbp")
    if masked:
        emit("    pushq %rbx")             # callee-saved; holds the dir mask
        emit("    movl 80(%rbp), %ebx")    # 15th arg = 8-bit direction mask
    else:
        emit(".ifdef WK_PREFETCH")
        emit(f"    pushq {PF}")            # scratch for prefetch base pointers
        emit(".endif")
    emit("")
    for i, d in enumerate(DIRS):
        emit(f"{d['name']}{sfx}:")
        if masked:
            nxt = (DIRS[i + 1]['name'] + sfx) if i + 1 < len(DIRS) \
                  else (name + "_done")
            emit(f"    testb ${1 << i}, %bl")  # skip this direction if bit clear
            emit(f"    jz {nxt}")
        emit(f"    movq {d['ferm']}, %r10")    # input fermion base
        emit(f"    movq {d['gauge']}, %r11")   # gauge link base (rhs-independent)
        if not masked:
            gen_prefetch(i)                    # pull next direction(s) into L1
        emit(f"    movl ${NHALF}, %eax")       # loop over the rhs half-blocks
        emit(f"{d['name']}{sfx}_half:")
        for p in d['pairs']:
            gen_projection(p)
            gen_multiply(p, d['diroff'], d['dag'])
            gen_reconstruct(p)
        emit(f"    addq ${HALF_STRIDE}, %r10") # advance input to next half
        emit(f"    addq ${HALF_STRIDE}, %rdi") # advance output to next half
        emit(f"    decl %eax")
        emit(f"    jnz {d['name']}{sfx}_half")
        emit(f"    subq ${HALF_STRIDE * NHALF}, %rdi")  # reset output to site base
        emit("")
    if masked:
        emit(f"{name}_done:")
        emit("    popq %rbx")
    else:
        emit(".ifdef WK_PREFETCH")
        emit(f"    popq {PF}")
        emit(".endif")
    emit("    popq %rbp")
    emit("    ret")
    emit("    .cfi_endproc")
    emit(f"    .size {name}, .-{name}")
    emit("")


# ------- header -------
emit("# AUTO-GENERATED by gen_wilson_avx2.py -- do not edit by hand.")
emit(f"# Generated for num_rhs={num_rhs}, Nc={Nc} (NHALF={NHALF} rhs blocks/site).")
emit(f"# Prefetch knobs: PF_AHEAD={PF_AHEAD}, PF_LINE_STRIDE={PF_LINE_STRIDE},"
     f" PF_HINT={PF_HINT} ({PF_INSN}) (schedule {PREFETCH_SCHEDULE}).")
emit("# AVX2 single-precision Wilson hopping kernel, one site (all num_rhs) per call.")
emit_kernel("wilson_kernel_avx2", masked=False)
emit_kernel("wilson_kernel_avx2_masked", masked=True)
emit('    .section .note.GNU-stack,"",@progbits')

dst = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "wilson_kernel_avx2.s")
with open(dst, "w") as f:
    f.write("\n".join(out) + "\n")
print(f"wrote {dst} ({len(out)} lines) for num_rhs={num_rhs}, Nc={Nc}")
