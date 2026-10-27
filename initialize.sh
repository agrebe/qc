# Copyright 2026, University of Maryland, All Rights Reserved
# SPDX-License-Identifier: GPL-3.0-or-later

source /opt/intel/oneapi/setvars.sh intel64
export OMP_PLACES=cores OMP_PROC_BIND=close
export MALLOC_MMAP_MAX_=0         # never satisfy malloc via per-allocation mmap
export MALLOC_TRIM_THRESHOLD_=-1  # never trim the heap back to the OS
