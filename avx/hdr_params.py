# Copyright 2026, University of Maryland, All Rights Reserved
# SPDX-License-Identifier: GPL-3.0-or-later

"""Read an integer build parameter (num_rhs, Nc, ...) out of the C headers.

The asm generators bake these values into the code they emit, so they must read
exactly what the C side compiles with.  They are #defines, and the thing that
actually knows a #define's value is the C preprocessor -- it follows the
#include chain and honors the #if guards -- so ask it, rather than regexing
the header text.  A generator then never needs to know which header holds a
parameter: moving one between headers changes nothing here.  (The Makefile
must still list the headers as prerequisites of the generated .s, or
an edit would not trigger regeneration.)

Feed cpp the header followed by the bare macro name, and the expansion is the
last line it prints:

    #include "global_params.h"      -->     ...the header's own output...
    num_rhs                                 16

The headers are found relative to this file (they live one level up, in the
source root), not relative to the working directory, so a generator runs the
same from anywhere.
"""
import os, subprocess, sys

SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def hdr_int(name, header="global_params.h", incdir=SRC_ROOT):
    cc = os.environ.get("CC", "cc")
    cpp = subprocess.run([cc, "-E", "-P", "-I", incdir, "-x", "c", "-"],
                         input=f'#include "{header}"\n{name}\n',
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    if cpp.returncode:
        sys.exit(f"hdr_int: '{cc} -E' failed on {header}\n{cpp.stderr}")
    lines = [l.strip() for l in cpp.stdout.splitlines() if l.strip()]
    token = lines[-1] if lines else ""
    try:
        # An undefined macro passes through cpp as its own name, and an
        # expression-valued one (none today) as the expression -- int() rejects
        # both, loudly, rather than letting a wrong constant reach the kernel.
        return int(token)
    except ValueError:
        sys.exit(f"hdr_int: {name} does not expand to an integer constant via {header} "
                 f"(cpp gave '{token}')")
