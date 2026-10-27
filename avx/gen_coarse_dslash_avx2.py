#!/usr/bin/env python3
# Copyright 2026, University of Maryland, All Rights Reserved
# SPDX-License-Identifier: GPL-3.0-or-later

# Generator for coarse_dslash_kernel_avx2.s
#
# Emits an AVX2 (single-precision) microkernel for the coarse Dslash apply_Dc:
# out(p) = sum_{k=0..8} C_k * in_k, where for one coarse site p the operator is
# a nearest-neighbor stencil of 9 dense complex coupling blocks (k=0 self,
# k=1..8 the eight face neighbors), each N_coarse x N_coarse, applied to the
# N_coarse x num_rhs input block at the corresponding site.
#
# Register-blocked complex GEMM microkernel.  It computes MR output rows
# (coarse vars a0..a0+MR-1) at once so each loaded input vector feeds all rows;
# the coupling element C[a,b] is a broadcast scalar, multiplied across an
# 8-lane ymm of num_rhs columns.  Only num_rhs is vectorized (a,b stay scalar
# loops), so any even num_null_vecs works as long as N_coarse = 2*num_null_vecs
# is a multiple of MR (checked below, and by test_coarse).
#
# Latency vs throughput.  A complex MAC is acc_re += cr*ir - ci*ii ; acc_im +=
# cr*ii + ci*ir.  Two ways to schedule it per output row:
#   split-2: re,im accumulators, 2 (dependent) FMAs each -> 2 acc/row, cheap on
#            registers but each acc is a 2-deep RAW chain.
#   split-4: re_pos,re_neg,im_a,im_b, 1 FMA each (all independent), combined
#            once at the end (re = re_pos-re_neg ; im = im_a+im_b) -> 4 acc/row,
#            every accumulator an independent chain so the latency floor halves.
# An FMA of 4-cycle latency at 2/cycle needs ~8 independent in-flight FMAs to be
# throughput- rather than latency-bound.  split-4 at MR=2 gives exactly 8 -- the
# no-slack edge.  More chains add slack, but 4*MR + 4 <= 16 ymm caps a uniform
# split-4 at MR=2.
#
# The generator therefore takes a per-row split spec (argv[1]): "44" = MR2, 8
# chains; "4422" = MR4 mixed, 8+4 = 12 independent chains in 16 registers;
# "2222" = MR4 uniform split-2 (8 chains, but input vectors loaded once and
# reused across 4 rows).  HARD CONSTRAINT: DC_MR in coarse_kernel.h must equal
# MR = len(spec) -- pass XCFLAGS=-DDC_MR=<MR>; a mismatch is caught by
# test_coarse.
#
# We process one 8-lane rhs chunk per pass and loop over the num_rhs/8 chunks
# (couplings re-read once per chunk -- 2x for num_rhs=16 -- cheap vs the inner
# FMA-bound loop).
#
# Split-complex layout (bytes), shared by the input/output coarse fields:
#   re/im of one (var, rhs-chunk): re at base, im at base + num_rhs*4
#   successive coarse vars (b or a): + 2*num_rhs*4
# Coupling C_k[(a*N_coarse + b)*2 + reim]: element (a,b) at a*(N_coarse*2*4) +
#   b*8 bytes; re +0, im +4 (read by vbroadcastss, no alignment requirement).
#   Input/output vmovaps are aligned: fields are aligned_alloc(64), offsets %32.

import sys, os
FLOAT  = 4
VECLEN = 8

# Software prefetch of the next coupling matrix (k+1) while computing coupling
# k.  On by default; env-overridable:
#   DC_PF unset / "1"          -> on, auto stride (the default)
#   DC_PF=0 / "off"            -> no prefetch
#   DC_PF=<N>                  -> on, explicit byte stride per b-iteration (tuning)
# Within a call the MR rows a0..a0+MR-1 of one coupling are contiguous (each
# row is CROW_B bytes and holds exactly that), so coupling k+1's footprint is
# one MR*CROW_B block at cbases[k+1]+coff.  A cursor (%r10) walks it during the
# b-loop (which is FMA/latency-bound, so it has slack to absorb the loads),
# hiding the cold matrix miss for 8 of the 9 per-call couplings.  The auto
# stride MR*CROW_B/N_coarse covers the whole block over the b-loop, touching
# each 64B line about twice.  The last k reads cbases[9] (one past the
# 9-element array) -> harmless pointer load + faultless prefetch.  prefetcht0
# and the %r10 cursor are non-semantic: the kernel's result is unchanged
# either way.
_pf = os.environ.get("DC_PF", "1").strip().lower()
PF_ON = _pf not in ("0", "", "off", "no", "false")

# Per-row accumulator split spec.  Default "4422" = MR4 mixed (two split-4 rows
# + two split-2 rows = 12 independent accumulator chains in 16 ymm), which puts
# slack past the 8-chain latency edge described above.  Must match DC_MR in
# coarse.c.
SPEC   = sys.argv[1] if len(sys.argv) > 1 else "4422"
SPLITS = [int(c) for c in SPEC]
MR     = len(SPLITS)
assert all(s in (2, 4) for s in SPLITS), f"split spec '{SPEC}' must be 2/4 digits"
NACC   = sum(SPLITS)
assert NACC + 4 <= 16, \
    f"register overflow: {NACC} accumulators + 4 temps > 16 ymm (spec '{SPEC}')"

# hdr_int asks the C preprocessor for num_rhs, so it does not matter which
# header holds the #define (the Makefile lists the headers as prerequisites of
# the generated .s).  N_coarse is not read here: it is a runtime kernel
# argument (%r9), so one kernel serves any coarse depth's null-vector count.
# It enters only as the b-loop trip count and the coupling-row byte stride
# CROW_B (= N_coarse*2*FLOAT), which is set up in %rbx / %rbp at entry -- see
# the row-addressing helper below.
from hdr_params import hdr_int
num_rhs        = hdr_int("num_rhs")

assert num_rhs % VECLEN == 0, \
    f"num_rhs ({num_rhs}) must be a multiple of {VECLEN} for AVX2 single"
assert MR <= 4, \
    f"MR ({MR}) > 4: runtime-N_coarse row addressing indexes tile rows via CROW_B " \
    f"(scale 1,2 in %rbx) and 3*CROW_B (%rbp); extend the index registers for MR > 4"
# The caller must pass N_coarse divisible by MR (a0 tiles by DC_MR).  The even
# num_null_vecs contract makes N_coarse a multiple of 4, so DC_MR=4 always
# divides it.

N_TILES = num_rhs // VECLEN                # 8-lane rhs chunks, looped

# byte strides (num_rhs-based -- compile-time; the N_coarse-based row stride
# is runtime)
CHUNK     = VECLEN * FLOAT                 # 32  one ymm of rhs lanes
IMAG_B    = num_rhs * FLOAT               # re -> im within a (var) block
VARSTR_B  = 2 * num_rhs * FLOAT           # successive coarse vars (in/out)
# CROW_B (a coupling row = N_coarse*2 reals) is runtime: %rbx = N_coarse*8
# bytes.

# Next-coupling prefetch cursor stride (bytes/b-iteration).  Auto covers the
# whole MR*CROW_B block over the N_coarse-iteration b-loop; the stride
# MR*CROW_B/N_coarse = MR*2*FLOAT is N_coarse-independent, so it stays a
# compile-time immediate.
PF_STRIDE = (int(_pf) if _pf.isdigit() and int(_pf) > 1
             else MR * 2 * FLOAT)

# ---- register assignment: accumulators first, then the 4 shared temps ----
def Y(i): return f"%ymm{i}"
rows = []                                  # per-row: split + accumulator regs
ai = 0
for s in SPLITS:
    regs = [Y(ai + i) for i in range(s)]
    ai += s
    if s == 4:
        rows.append({'split': 4, 're_pos': regs[0], 're_neg': regs[1],
                     'im_a': regs[2], 'im_b': regs[3]})
    else:
        rows.append({'split': 2, 're': regs[0], 'im': regs[1]})
IR = Y(ai); II = Y(ai + 1); CR = Y(ai + 2); CI = Y(ai + 3)
assert ai + 4 <= 16

def row_accs(r):
    return [r['re_pos'], r['re_neg'], r['im_a'], r['im_b']] if r['split'] == 4 \
        else [r['re'], r['im']]

# Coupling-element address for tile row `ri` (byte offset `extra` selects
# re=0/im=4), relative to the b-cursor %r11 (= &C_k[a0][b], advanced +8 each
# b-iter).  Row ri sits ri*CROW_B above row a0; with N_coarse runtime, CROW_B
# lives in %rbx (rows 1,2 via scale 1,2) and 3*CROW_B in %rbp (row 3).  The
# index registers are constant, so the effective address rides the existing
# %r11 advance -- no per-iteration pointer bump.  Covers MR<=4.
def crow(ri, extra=0):
    d = f"{extra}" if extra else ""
    return [f"{d}(%r11)", f"{d}(%r11,%rbx,1)",
            f"{d}(%r11,%rbx,2)", f"{d}(%r11,%rbp,1)"][ri]

out = []
def emit(s=''): out.append(s)

# ---- header ----
emit("# AUTO-GENERATED by gen_coarse_dslash_avx2.py -- do not edit by hand.")
emit("# Coarse Dslash apply_Dc microkernel (AVX2, single precision).")
emit(f"# num_rhs={num_rhs} (baked); N_coarse: runtime arg %r9 (b-count + CROW_B);")
emit(f"# MR={MR} rows/pass, split spec {SPLITS} ({NACC} accumulators), "
     f"{N_TILES} rhs-chunk(s).  REQUIRES coarse.c DC_MR={MR}.")
emit(f"# next-coupling prefetch: {'ON stride=%d B/iter' % PF_STRIDE if PF_ON else 'OFF'}")
emit("")
emit("    .text")
emit("    .globl coarse_dslash_kernel_avx2")
emit("    .type coarse_dslash_kernel_avx2, @function")
emit("    # C ABI: void coarse_dslash_kernel_avx2(")
emit("    #   float * out,        %rdi  output rows a0..a0+MR-1 (already offset)")
emit("    #   float ** cbases,    %rsi  ncpl coupling-matrix base pointers")
emit("    #   float ** ibases,    %rdx  ncpl input-block base pointers")
emit("    #   long coff,          %rcx  byte offset of row a0 within a coupling")
emit("    #   int ncpl,           %r8d  number of couplings to accumulate (1..)")
emit("    #   long n_coarse)      %r9   coarse vars/site: b-loop count + row stride")
emit("    # n_coarse (was baked N_coarse) is a runtime arg so one kernel serves any coarse")
emit("    # depth; it sets the b-loop trip count and CROW_B = n_coarse*2*4 (%rbx / %rbp).")
emit("    # ncpl is a runtime arg (was hardcoded 9): 9 = full coarse stencil")
emit("    # (self + 8 hops), 8 = pure-hop block (D_eo/D_oe), 1 = a dense self/inverse")
emit("    # block.  Only the outer k-loop trip count changes; the inner b-loop (and")
emit("    # thus throughput) is identical.  The prefetch reads cbases[ncpl] (one past),")
emit("    # so the caller's cbases array must hold ncpl+1 slots.")
emit("coarse_dslash_kernel_avx2:")
emit("    .cfi_startproc")
emit("    endbr64")
for reg in ("%r12", "%r13", "%r14", "%r15", "%rbx", "%rbp"):
    emit(f"    pushq {reg}")
    emit("    .cfi_adjust_cfa_offset 8")
emit("    movq %rsi, %r12            # r12 = cbases (preserved across rhs-chunks)")
emit("    movq %rdx, %r13            # r13 = ibases")
emit("    movl %r8d, %r14d           # r14d = ncpl (k-loop count, preserved)")
emit("    movq %r9, %r15            # r15 = n_coarse (runtime b-loop count)")
emit("    leaq 0(,%r15,8), %rbx     # rbx = CROW_B = n_coarse*2*4 bytes (coupling row stride)")
emit("    leaq (%rbx,%rbx,2), %rbp  # rbp = 3*CROW_B  (tile-row-3 index)")
emit("")

for t in range(N_TILES):
    rv_off = t * CHUNK
    emit(f"    # ===== rhs-chunk {t} (byte offset {rv_off}) =====")
    for r in rows:
        for acc in row_accs(r):
            emit(f"    vxorps {acc}, {acc}, {acc}")
    emit("    movq %r12, %r8             # cbases cursor")
    emit("    movq %r13, %r9             # ibases cursor")
    emit("    movl %r14d, %esi          # k = ncpl couplings")
    emit(f".Lk_{t}:")
    emit("    movq (%r8), %r11          # r11 = cbases[k]")
    emit("    addq %rcx, %r11           # cp = &C_k[a0][0]  (row a0)")
    emit("    movq (%r9), %rax          # rax = ibases[k]")
    if rv_off:
        emit(f"    addq ${rv_off}, %rax        # ip = input + rhs-chunk offset")
    emit("    addq $8, %r8")
    emit("    addq $8, %r9")
    if PF_ON:
        emit("    movq (%r8), %r10          # next coupling base cbases[k+1]")
        emit("    addq %rcx, %r10           # pf cursor = &C_{k+1}[a0][0] (MR-row block)")
    emit(f"    movl %r15d, %edx          # b loop (runtime n_coarse)")
    emit(f".Lb_{t}:")
    # one input vector pair (re/im) feeds all MR rows
    emit(f"    vmovaps (%rax), {IR}")
    emit(f"    vmovaps {IMAG_B}(%rax), {II}")
    for ri, r in enumerate(rows):
        emit(f"    vbroadcastss {crow(ri)}, {CR}")
        emit(f"    vbroadcastss {crow(ri, FLOAT)}, {CI}")
        if r['split'] == 4:
            emit(f"    vfmadd231ps {CR}, {IR}, {r['re_pos']}")   # += cr*ir
            emit(f"    vfmadd231ps {CI}, {II}, {r['re_neg']}")   # += ci*ii
            emit(f"    vfmadd231ps {CR}, {II}, {r['im_a']}")     # += cr*ii
            emit(f"    vfmadd231ps {CI}, {IR}, {r['im_b']}")     # += ci*ir
        else:
            emit(f"    vfmadd231ps {CR}, {IR}, {r['re']}")       # re += cr*ir
            emit(f"    vfnmadd231ps {CI}, {II}, {r['re']}")      # re -= ci*ii
            emit(f"    vfmadd231ps {CR}, {II}, {r['im']}")       # im += cr*ii
            emit(f"    vfmadd231ps {CI}, {IR}, {r['im']}")       # im += ci*ir
    if PF_ON:
        emit(f"    prefetcht0 (%r10)         # SW-prefetch next coupling's matrix block")
        emit(f"    addq ${PF_STRIDE}, %r10")
    emit(f"    addq $8, %r11        # advance one coupling element (b -> b+1)")
    emit(f"    addq ${VARSTR_B}, %rax     # advance one input var")
    emit("    decl %edx")
    emit(f"    jnz .Lb_{t}")
    emit("    decl %esi")
    emit(f"    jnz .Lk_{t}")
    # combine split terms and store
    for ri, r in enumerate(rows):
        rb = ri * VARSTR_B + rv_off
        if r['split'] == 4:
            emit(f"    vsubps {r['re_neg']}, {r['re_pos']}, {r['re_pos']}")
            emit(f"    vaddps {r['im_b']}, {r['im_a']}, {r['im_a']}")
            emit(f"    vmovaps {r['re_pos']}, {rb}(%rdi)")
            emit(f"    vmovaps {r['im_a']}, {rb + IMAG_B}(%rdi)")
        else:
            emit(f"    vmovaps {r['re']}, {rb}(%rdi)")
            emit(f"    vmovaps {r['im']}, {rb + IMAG_B}(%rdi)")
    emit("")

for reg in ("%rbp", "%rbx", "%r15", "%r14", "%r13", "%r12"):
    emit(f"    popq {reg}")
    emit("    .cfi_adjust_cfa_offset -8")
emit("    ret")
emit("    .cfi_endproc")
emit("    .size coarse_dslash_kernel_avx2, .-coarse_dslash_kernel_avx2")
emit('    .section .note.GNU-stack,"",@progbits')

dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coarse_dslash_kernel_avx2.s")
with open(dst, "w") as f:
    f.write("\n".join(out) + "\n")
print(f"wrote {dst} ({len(out)} lines) for "
      f"num_rhs={num_rhs}, N_coarse=runtime, MR={MR}, spec={SPLITS}, "
      f"{NACC} accumulators, {N_TILES} rhs-chunk(s)")
