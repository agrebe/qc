#!/usr/bin/env python3
# Copyright 2026, University of Maryland, All Rights Reserved
# SPDX-License-Identifier: GPL-3.0-or-later

# One-shot transformer that derives wilson_kernel_avx512_masked from the
# hand-written wilson_kernel_avx512 in wilson_kernel_avx512.S and writes it back
# into the same file (so the masked routine reuses the mult_U3 / mult_U3_dag /
# rmw_* macros defined at the top).  It is not a build step -- it is a tool to
# (re)emit the masked kernel from the unmasked one whenever the latter is
# retuned, then the result is committed as static asm.  Re-running is
# idempotent: any previously generated masked block (between the BEGIN/END
# markers) is replaced.
#
# What "masked" adds (mirrors wilson_kernel_avx2_masked):
#   * 15th arg (80(%rbp)) is an 8-bit direction mask; bit i selects DIRS[i]:
#       bit 0:+x 1:+y 2:+z 3:+t 4:-x 5:-y 6:-z 7:-t.
#     Each direction is preceded by `testb $(1<<i), %bl; jz <next>` so excluded
#     directions cost only a (well-predicted) branch and zero flops.
#   * No software prefetch (Schwarz block data is L2-resident): the .ifdef
#     WK_PREFETCH regions are stripped.
#   * %rbx is pushed/popped to hold the mask (callee-saved; the kernel body
#     never touches it otherwise).
#
# The unmasked kernel's direction blocks are straight-line and self-contained
# (each loads its own %r11 from the stack args; %rdi and the fermion pointers
# are never stepped), so skipping a block cannot disturb any other -- gating is
# a bare test-and-branch, with no fix-up code.

import os, re

# The template this splices into lives beside this script; it is a .S (cpp/GAS:
# it reads num_rhs from global_params.h).
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wilson_kernel_avx512.S")
BEGIN = "# === BEGIN generated wilson_kernel_avx512_masked"
END   = "# === END generated wilson_kernel_avx512_masked"

DIRS = ['plus_x', 'plus_y', 'plus_z', 'plus_t',
        'minus_x', 'minus_y', 'minus_z', 'minus_t']

text = open(SRC).read()

# Drop any previously generated masked block so re-running is idempotent.
# Consume from the BEGIN marker through to the end of the END marker line
# (the markers carry a trailing " ===", so match the rest of that line too).
text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"[^\n]*\n?",
              "", text, flags=re.S)

lines = text.split("\n")

# Locate the first occurrence of each direction label and of `finish:` -- these
# bound the unmasked function's per-direction blocks.
label_idx = {}
for i, l in enumerate(lines):
    s = l.strip()
    name = s[:-1] if s.endswith(':') else None
    if name in DIRS + ['finish'] and name not in label_idx:
        label_idx[name] = i
order = DIRS + ['finish']
bounds = {order[k]: (label_idx[order[k]],
                     label_idx[order[k + 1]] if k + 1 < len(order) else None)
          for k in range(len(order))}

# Renameable labels: the direction labels and finish.  The (?<!\w)...(?!\w)
# guards mean "plus_x" does not match inside "rmw_plus_x", so the macro
# invocations in the block bodies are left alone.
RENAME = DIRS + ['finish']

def rename(s):
    for lab in RENAME:
        s = re.sub(r'(?<!\w)' + re.escape(lab) + r'(?!\w)', lab + '_m', s)
    return s

def strip_prefetch(block):
    out, skip = [], False
    for l in block:
        st = l.strip()
        if st.startswith('.ifdef WK_PREFETCH'):
            skip = True
            continue
        if skip:
            if st == '.endif':
                skip = False
            continue
        out.append(l)
    return out

def block_lines(name):
    lo, hi = bounds[name]
    return lines[lo:hi]

emit = []
def E(s=''):
    emit.append(s)

E(BEGIN + " (do not edit by hand; see gen_wilson_avx512_masked.py) ===")
E("    .globl wilson_kernel_avx512_masked")
E("    .type wilson_kernel_avx512_masked, @function")
E("    # Schwarz-block variant of wilson_kernel_avx512: identical per-direction")
E("    # arithmetic, but gated by an 8-bit direction mask (15th arg, 80(%rbp)).")
E("    # bit 0:+x 1:+y 2:+z 3:+t 4:-x 5:-y 6:-z 7:-t -- excluded directions are")
E("    # branched over (zero flops).  No software prefetch (block data is")
E("    # L2-resident).")
E("    .p2align 6                   # MITE-fed body: pin fetch alignment")
E("wilson_kernel_avx512_masked:")
E("    .cfi_startproc")
E("    pushq %rbp")
E("    movq %rsp, %rbp")
E("    pushq %rbx                   # callee-saved; holds the direction mask")
E("    movl 80(%rbp), %ebx          # 15th arg = 8-bit direction mask")
E("")

for k, name in enumerate(DIRS):
    nxt = (DIRS[k + 1] if k + 1 < len(DIRS) else 'finish') + '_m'
    bit = 1 << k
    blk = strip_prefetch(block_lines(name))
    blk = [rename(l) for l in blk]
    label = name + '_m'
    # blk[0] is the (renamed) label line `name_m:`
    E(label + ":")
    E("    testb $%d, %%bl" % bit)
    E("    jz %s" % nxt)
    for l in blk[1:]:
        E(l)

# Epilogue.  rsp still points at the saved %rbx (the body never touches rsp), so
# pop rbx then rbp.  (The unmasked kernel's `movq %rbp,%rsp` reset would skip the
# rbx slot, leaving the callee-saved register clobbered -- so do not use it here.)
E("finish_m:")
E("    popq %rbx")
E("    popq %rbp")
E("    ret")
E("    .cfi_endproc")
E("    .size wilson_kernel_avx512_masked, .-wilson_kernel_avx512_masked")
E(END + " ===")

masked = "\n".join(emit) + "\n"

# Insert the masked routine just before the GNU-stack note (end of file).
marker = '    .section .note.GNU-stack,"",@progbits'
assert marker in text, "could not find GNU-stack note to anchor insertion"
text = text.replace(marker, masked + marker, 1)

open(SRC, "w").write(text)
print("inserted wilson_kernel_avx512_masked (%d lines) into %s"
      % (len(emit), SRC))
