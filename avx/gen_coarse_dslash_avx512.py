#!/usr/bin/env python3
# Copyright 2026, University of Maryland, All Rights Reserved
# SPDX-License-Identifier: GPL-3.0-or-later

# Generator for coarse_dslash_kernel_avx512.s
#
# AVX-512 (single-precision) sibling of gen_coarse_dslash_avx2.py: the coarse
# Dslash apply_Dc microkernel, out(p) = sum_{k=0..8} C_k * in_k for one coarse
# site (k=0 self, k=1..8 face neighbors; each C_k is N_coarse x N_coarse
# complex applied to the N_coarse x num_rhs input block).
#
# What the wider ISA buys over the AVX2 sibling, which is backend-bound about
# half on memory and half on core:
#   * 32 zmm registers -> MR=4 output rows per pass (vs 2 on AVX2), so each
#     input vector feeds 4 FMA chains and the input block is re-read N_coarse/4
#     times instead of N_coarse/2 (attacks the memory-bound stalls), with far
#     more independent ILP to hide latency (attacks the core-bound stalls).
#   * a 16-wide zmm holds 16 columns in one register, so at num_rhs=16 the
#     coupling matrices stream once (the AVX2 split re-reads them per 8-lane
#     chunk).
# Same split-complex accumulators as AVX2 (re_pos/re_neg/im_a/im_b per row,
# combined at the end) so every accumulator is an independent single-FMA chain.
# The coupling scalar is broadcast to a register and reused across a row's 2
# FMAs (cheaper on the load ports than an embedded-broadcast memory operand per
# FMA).  Checked under SDE (sde64 -skx) by test_coarse vs apply_Dc_ref.
#
# Three kernels are emitted (one loop body, parameterized):
#   coarse_dslash_kernel_avx512        out  = sum C_k in_k   (overwrite)
#   coarse_dslash_kernel_avx512_acc    out += sum C_k in_k   (accumulate: the
#     re_pos/im_a accumulators init from out instead of zero)
#   coarse_dslash_kernel_avx512_tconj  out += sum C_k^H in_k (transposed-conj
#     accumulate, the blocked-HEMM mirror pass): element (b, a0+ri) is read
#     instead of (a0+ri, b) -- the b-cursor starts at column offset coff and
#     advances a whole coupling row (CROW_B) per iteration while the MR row
#     offsets become immediates ri*8 -- and the conjugate flips the combines to
#     re = re_pos + re_neg, im = im_a - im_b.  No chirality signs: the caller
#     folds the Gamma5c signature into the input.  No prefetch: the caller
#     applies it to a tile the non-transposed pass just made cache-hot.
#
# AVX-512 needs num_rhs % 16 == 0 (one zmm per rhs chunk); production num_rhs
# is a multiple of 16.  N_coarse = 2*num_null_vecs is a multiple of 4
# (num_null_vecs even), so the MR=4 a-tiling has no remainder.  Aligned vmovaps
# are valid: the coarse fields are aligned_alloc(64) and every in/out offset
# here is a multiple of 64 (the tconj coupling reads are 8B-aligned, but those
# are broadcasts).

import os
FLOAT  = 4
VECLEN = 16
MR     = 4          # output rows (coarse vars) per microkernel pass

# Software prefetch of the next coupling matrix (k+1) while computing coupling
# k; see gen_coarse_dslash_avx2.py for the same scheme.  Within a call the MR
# rows a0..a0+MR-1 of one coupling are contiguous, so coupling k+1 is one
# MR*CROW_B block at cbases[k+1]+coff; a %r10 cursor walks it in the b-loop
# (prefetcht0 + addq $PF_STRIDE), hiding the cold matrix miss for 8 of the 9
# per-call couplings.  Non-semantic: the result is unchanged either way.
# On by default:
#   DC_PF unset / "1" -> on, auto stride (MR*CROW_B/N_coarse B/iter,
#                                         ~2x coverage)
#   DC_PF=0 / "off"   -> no prefetch
#   DC_PF=<N>         -> on, explicit byte stride per b-iteration (tuning)
_pf = os.environ.get("DC_PF", "1").strip().lower()
PF_ON = _pf not in ("0", "", "off", "no", "false")

# hdr_int asks the C preprocessor for num_rhs, so it does not matter which
# header holds the #define (the Makefile lists the headers as prerequisites of
# the generated .s).  N_coarse is not read here: it is a runtime kernel
# argument (%r9), so one kernel serves any coarse depth's null-vector count.
# It enters only as the b-loop trip count and the coupling-row byte stride
# CROW_B (= N_coarse*2*FLOAT), set up in %rbx / %rbp at entry.
from hdr_params import hdr_int
num_rhs        = hdr_int("num_rhs")

assert num_rhs % VECLEN == 0, \
    f"num_rhs ({num_rhs}) must be a multiple of {VECLEN} for AVX-512 single"
assert MR <= 4, \
    f"MR ({MR}) > 4: runtime-N_coarse row addressing indexes tile rows via CROW_B " \
    f"(scale 1,2 in %rbx) and 3*CROW_B (%rbp); extend the index registers for MR > 4"
assert 4*MR + 4 <= 32, "zmm register budget exceeded"
# The caller must pass N_coarse divisible by MR (a0 tiles by DC_MR).  The even
# num_null_vecs contract makes N_coarse a multiple of 4, so DC_MR=4 always
# divides it.

N_TILES = num_rhs // VECLEN          # 16-lane rhs chunks, looped (1 for nr=16)

# byte strides (num_rhs-based -- compile-time; the N_coarse-based row stride is
# runtime)
CHUNK     = VECLEN * FLOAT                # 64  one zmm of rhs lanes
IMAG_B    = num_rhs * FLOAT               # re -> im within a (var) block
VARSTR_B  = 2 * num_rhs * FLOAT           # successive coarse vars (in/out)
# CROW_B (a coupling row = N_coarse*2 reals) is runtime: %rbx = N_coarse*8
# bytes.

# Next-coupling prefetch cursor stride (bytes/b-iteration); auto covers the
# whole MR*CROW_B block over the N_coarse-iteration b-loop;
# MR*CROW_B/N_coarse = MR*2*FLOAT is N_coarse-independent, so it stays a
# compile-time immediate.
PF_STRIDE = (int(_pf) if _pf.isdigit() and int(_pf) > 1
             else MR * 2 * FLOAT)

def Z(i): return f"%zmm{i}"
RE_POS = [Z(4*row + 0) for row in range(MR)]   # += cr*ir
RE_NEG = [Z(4*row + 1) for row in range(MR)]   # += ci*ii
IM_A   = [Z(4*row + 2) for row in range(MR)]   # += cr*ii
IM_B   = [Z(4*row + 3) for row in range(MR)]   # += ci*ir
ACCS   = RE_POS + RE_NEG + IM_A + IM_B
IR = Z(4*MR)        # zmm16
II = Z(4*MR + 1)    # zmm17
CR = Z(4*MR + 2)    # zmm18
CI = Z(4*MR + 3)    # zmm19

# Coupling-element address for tile row `ri` (byte offset `extra` selects
# re=0/im=4), relative to the b-cursor %r11.
#   normal: %r11 = &C_k[a0][b], advanced +8 each b-iter; row ri sits ri*CROW_B
#     above (CROW_B in %rbx: scales 1,2; 3*CROW_B in %rbp).
#   tconj:  %r11 = &C_k[b][j0], advanced +CROW_B each b-iter; "row" ri is column
#     j0+ri, an immediate ri*8 -- the MR elements are 32 contiguous bytes.
def crow(ri, extra=0):
    d = f"{extra}" if extra else ""
    return [f"{d}(%r11)", f"{d}(%r11,%rbx,1)",
            f"{d}(%r11,%rbx,2)", f"{d}(%r11,%rbp,1)"][ri]
def crow_t(ri, extra=0):
    d = ri * 2 * FLOAT + extra
    return f"{d}(%r11)" if d else "(%r11)"

out = []
def emit(s=''): out.append(s)

emit("# AUTO-GENERATED by gen_coarse_dslash_avx512.py -- do not edit by hand.")
emit("# Coarse Dslash apply_Dc microkernel (AVX-512, single precision).")
emit(f"# num_rhs={num_rhs} (baked); N_coarse: runtime arg %r9 (b-count + CROW_B);")
emit(f"# MR={MR} rows/pass, split complex accumulators, {N_TILES} rhs-chunk(s).")
emit(f"# next-coupling prefetch: {'ON stride=%d B/iter' % PF_STRIDE if PF_ON else 'OFF'}")
emit(f"# variants: base (overwrite), _acc (accumulate), _tconj (transposed-conj acc).")
emit("")
emit("    .text")

def emit_kernel(name, acc=False, tconj=False):
    pf = PF_ON and not tconj
    emit(f"    .globl {name}")
    emit(f"    .type {name}, @function")
    emit(f"    # C ABI: void {name}(")
    emit("    #   float * out,        %rdi  output rows a0..a0+MR-1 (already offset)")
    emit("    #   float ** cbases,    %rsi  ncpl coupling-matrix base pointers")
    emit("    #   float ** ibases,    %rdx  ncpl input-block base pointers")
    if tconj:
        emit("    #   long coff,          %rcx  byte COLUMN offset j0*2*4 (transposed read)")
    else:
        emit("    #   long coff,          %rcx  byte offset of row a0 within a coupling")
    emit("    #   int ncpl,           %r8d  number of couplings to accumulate (1..)")
    emit("    #   long n_coarse)      %r9   coarse vars/site: b-loop count + row stride")
    if tconj:
        emit("    # out += sum_k C_k^H in_k: reads element (b, j0+ri) -- the b-cursor")
        emit("    # advances CROW_B per iteration, the MR outputs are columns j0..j0+3.")
        emit("    # Conjugate combine: re = re_pos + re_neg, im = im_a - im_b.")
    elif acc:
        emit("    # out += sum_k C_k in_k: re_pos/im_a accumulators init from out.")
    if pf:
        emit("    # The prefetch reads cbases[ncpl] (one past), so the caller's cbases")
        emit("    # array must hold ncpl+1 slots.")
    emit(f"{name}:")
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

    lbl = name.split("avx512")[-1].lstrip("_") or "k"   # unique label suffix
    for t in range(N_TILES):
        rv_off = t * CHUNK
        emit(f"    # ===== rhs-chunk {t} (byte offset {rv_off}) =====")
        if acc or tconj:
            # accumulate: seed the positive-combine accumulators from out
            for row in range(MR):
                rb = row * VARSTR_B + rv_off
                emit(f"    vmovaps {rb}(%rdi), {RE_POS[row]}")
                emit(f"    vmovaps {rb + IMAG_B}(%rdi), {IM_A[row]}")
            for a in RE_NEG + IM_B:
                emit(f"    vpxorq {a}, {a}, {a}")
        else:
            for a in ACCS:
                emit(f"    vpxorq {a}, {a}, {a}")
        emit("    movq %r12, %r8             # cbases cursor")
        emit("    movq %r13, %r9             # ibases cursor")
        emit("    movl %r14d, %esi          # k = ncpl couplings")
        emit(f".Lk_{lbl}_{t}:")
        emit("    movq (%r8), %r11          # r11 = cbases[k]")
        emit("    addq %rcx, %r11           # cp = coff into the coupling")
        emit("    movq (%r9), %rax          # rax = ibases[k]")
        if rv_off:
            emit(f"    addq ${rv_off}, %rax        # ip = input + rhs-chunk offset")
        emit("    addq $8, %r8")
        emit("    addq $8, %r9")
        if pf:
            emit("    movq (%r8), %r10          # next coupling base cbases[k+1]")
            emit("    addq %rcx, %r10           # pf cursor = &C_{k+1}[a0][0] (MR-row block)")
        emit(f"    movl %r15d, %edx            # b = n_coarse (runtime)")
        emit(f".Lb_{lbl}_{t}:")
        emit(f"    vmovaps 0(%rax), {IR}")
        emit(f"    vmovaps {IMAG_B}(%rax), {II}")
        row_addr = crow_t if tconj else crow
        for row in range(MR):
            emit(f"    vbroadcastss {row_addr(row)}, {CR}")
            emit(f"    vbroadcastss {row_addr(row, FLOAT)}, {CI}")
            emit(f"    vfmadd231ps {CR}, {IR}, {RE_POS[row]}")   # += cr*ir
            emit(f"    vfmadd231ps {CI}, {II}, {RE_NEG[row]}")   # += ci*ii
            emit(f"    vfmadd231ps {CR}, {II}, {IM_A[row]}")     # += cr*ii
            emit(f"    vfmadd231ps {CI}, {IR}, {IM_B[row]}")     # += ci*ir
        if pf:
            emit("    prefetcht0 (%r10)            # SW-prefetch next coupling's matrix block")
            emit(f"    addq ${PF_STRIDE}, %r10")
        if tconj:
            emit("    addq %rbx, %r11           # next coupling ROW (transposed walk)")
        else:
            emit("    addq $8, %r11             # next coupling element b")
        emit(f"    addq ${VARSTR_B}, %rax       # next input var b")
        emit("    decl %edx")
        emit(f"    jnz .Lb_{lbl}_{t}")
        emit("    decl %esi")
        emit(f"    jnz .Lk_{lbl}_{t}")
        # combine: normal/acc re = re_pos - re_neg, im = im_a + im_b;
        # tconj (conjugate)  re = re_pos + re_neg, im = im_a - im_b.
        for row in range(MR):
            rb = row * VARSTR_B + rv_off
            if tconj:
                emit(f"    vaddps {RE_NEG[row]}, {RE_POS[row]}, {RE_POS[row]}")
                emit(f"    vsubps {IM_B[row]}, {IM_A[row]}, {IM_A[row]}")
            else:
                emit(f"    vsubps {RE_NEG[row]}, {RE_POS[row]}, {RE_POS[row]}")
                emit(f"    vaddps {IM_B[row]}, {IM_A[row]}, {IM_A[row]}")
            emit(f"    vmovaps {RE_POS[row]}, {rb}(%rdi)")
            emit(f"    vmovaps {IM_A[row]}, {rb + IMAG_B}(%rdi)")
        emit("")

    for reg in ("%rbp", "%rbx", "%r15", "%r14", "%r13", "%r12"):
        emit(f"    popq {reg}")
        emit("    .cfi_adjust_cfa_offset -8")
    emit("    ret")
    emit("    .cfi_endproc")
    emit(f"    .size {name}, .-{name}")
    emit("")

emit_kernel("coarse_dslash_kernel_avx512")
emit_kernel("coarse_dslash_kernel_avx512_acc", acc=True)
emit_kernel("coarse_dslash_kernel_avx512_tconj", tconj=True)

emit('    .section .note.GNU-stack,"",@progbits')

dst = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "coarse_dslash_kernel_avx512.s")
with open(dst, "w") as f:
    f.write("\n".join(out) + "\n")
print(f"wrote {dst} ({len(out)} lines) for "
      f"num_rhs={num_rhs}, N_coarse=runtime, MR={MR}, {N_TILES} rhs-chunk(s), 3 kernels")
