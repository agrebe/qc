# Copyright 2026, University of Maryland, All Rights Reserved
# SPDX-License-Identifier: GPL-3.0-or-later

"""Python interface to the qc Wilson-clover multigrid solver.

ctypes over libqc.so, with numpy views onto qc's own solver fields.  Calls are
whole solves, so foreign-call overhead is irrelevant and there is nothing to
gain from a compiled extension; ctypes also releases the GIL for the duration
of a call, so a solve does not block other Python threads.

    import qc
    qc.require(dims=(16, 16, 16, 32), num_rhs=16, precision="double")
    qc.init()

    U = qc.gauge_field()
    qc.read_gauge_file(U, "test/sample_config.lime")

    h = qc.setup(U, qc.setup_params(kappa=0.134, csw=1.87567))
    b, x = qc.field(), qc.field()
    b[qc.site_index(0, 0, 0, 0), s, c, 0, lane] = 1.0
    qc.solve(h, x, b, tol=1e-10, chroma_norm=True)
    qc.report_profile(h)

The geometry, the number of right-hand sides and the arithmetic precision are
compile-time constants of the library, so a given libqc.so is valid for exactly
one configuration.  Nothing in the library can detect a caller that assumed a
different one -- solver fields cross the boundary as void * -- which is what
qc.require is for.  Call it before anything else and it fails loudly instead.

Loading libqc.so needs the compiler runtime on the library path (libimf, libsvml,
and a BLAS in a DENSE_BLAS=mkl or =cblas build).  Source the same environment you
build with,
which for a slurm job means the usual module loads or `source initialize.sh`.
"""

import ctypes
import os
import sys

import numpy as np

__all__ = [
    "require", "init", "finalize", "rank", "num_ranks",
    "global_dims", "local_dims", "local_origin",
    "num_rhs", "num_colors", "num_spins", "precision", "dtype",
    "field", "gauge_field", "field_zero", "free_all",
    "site_index", "site_coords", "component_index",
    "import_lane", "export_lane",
    "read_gauge_file", "setup_params", "solve_params",
    "setup", "solve", "apply_D", "report_profile", "teardown",
]

SINGLE = 4          # the qc_precision enumerators: bytes per real
DOUBLE = 8


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _find_library():
    """Where to look for libqc.so, most specific first."""
    if os.environ.get("QC_LIBRARY"):
        return [os.environ["QC_LIBRARY"]]
    here = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(here, "..", "..", "libqc.so"),  # built in the source tree
        os.path.join(here, "libqc.so"),         # installed beside this module
        "libqc.so",                             # or on the loader's own path
    ]


def _load():
    tried = []
    for cand in _find_library():
        try:
            return ctypes.CDLL(cand), cand
        except OSError as e:
            tried.append((cand, str(e)))

    # Separate the two failures that look alike from Python.  A missing libqc.so
    # is a build problem; a libqc.so that cannot resolve libimf or MKL is an
    # environment problem, and saying so saves the reader from debugging the
    # wrong one.
    runtime = [t for t in tried if "libimf" in t[1] or "libsvml" in t[1]
               or "libmkl" in t[1] or "libintlc" in t[1] or "libiomp" in t[1]]
    lines = ["could not load libqc.so"]
    if runtime:
        lines += [
            "",
            "  " + runtime[0][1],
            "",
            "  The library was found but its compiler runtime was not.  Source the",
            "  environment qc was built with before running python -- the same one",
            "  the build uses (`source initialize.sh`, or the equivalent module",
            "  loads on a cluster).",
        ]
    else:
        lines += ["", "  looked in:"] + [f"    {c}: {e}" for c, e in tried] + [
            "",
            "  Build it with `make libqc.so`, or point QC_LIBRARY at one.",
        ]
    raise OSError("\n".join(lines))


_lib, _lib_path = _load()


try:
    _libc = ctypes.CDLL(None)             # this process, which includes libc
    _libc.fflush.argtypes = [ctypes.c_void_p]
    _libc.fflush.restype = ctypes.c_int
except (OSError, AttributeError):         # not fatal; only output order suffers
    _libc = None


def _flush():
    """Drain both output buffers, Python's and C's.

    They are separate, so a solve's progress lines otherwise land in the middle
    of a Python line -- the output is not merely out of order, it is spliced
    mid-word.  Both directions need draining and both are called around any
    entry point that prints: Python's before, so its text is already out when
    the solver starts writing, and C's after, so nothing of the solver's is
    still held when Python prints next.  fflush(NULL) covers every open stream.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    if _libc is not None:
        _libc.fflush(None)


# ---------------------------------------------------------------------------
# Parameter structs
# ---------------------------------------------------------------------------
# Field order and types mirror qc.h exactly.  ctypes lays a Structure out with
# the platform ABI rules, the same ones the C compiler used, so the two agree.
# A field added to qc.h and not added here is read at the wrong offset, which is
# what setup_params()/solve_params() guard against by filling the struct from
# qc's own defaults before applying any keyword.

CYCLE_V, CYCLE_W, CYCLE_K = 0, 1, 2


class SetupParams(ctypes.Structure):
    _fields_ = [
        ("kappa",                  ctypes.c_double),
        ("csw",                    ctypes.c_double),
        ("smoother_sweeps",        ctypes.c_int),
        ("sap_block_iter",         ctypes.c_int),
        ("coarse_smoother_sweeps", ctypes.c_int),
        ("coarse_sap_block_iter",  ctypes.c_int),
        ("cycle",                  ctypes.c_int),
        ("w_mu",                   ctypes.c_int),
        ("k_iters",                ctypes.c_int),
        ("setup_k_tol",            ctypes.c_double),
        ("solve_k_tol",            ctypes.c_double),
        ("coarse_max_iter",        ctypes.c_int),
        ("coarse_tol",             ctypes.c_double),
        ("coarse_restart",         ctypes.c_int),
        ("setup_init_sweeps",      ctypes.c_int),
        ("setup_sweeps",           ctypes.c_int),
        ("coarse_setup_sweeps",    ctypes.c_int),
    ]


class SolveParams(ctypes.Structure):
    _fields_ = [
        ("tol",         ctypes.c_double),
        ("max_iter",    ctypes.c_int),
        ("restart",     ctypes.c_int),
        ("chroma_norm", ctypes.c_int),
    ]


_int4 = ctypes.c_int * 4
_p = ctypes.c_void_p


def _declare():
    """argtypes and restype for every entry point.

    Not optional bookkeeping: without an explicit restype ctypes assumes int,
    which truncates the 64-bit values qc_alloc and qc_field_elements return.  On
    the sample 16^3 x 32 configuration with 16 rhs, a fermion field is already
    50 million elements, so the truncation would not stay theoretical for long.
    """
    sig = {
        "qc_native_precision": ([], ctypes.c_int),
        "qc_global_dims":  ([_int4], None),
        "qc_local_dims":   ([_int4], None),
        "qc_local_origin": ([_int4], None),
        "qc_num_rhs":      ([], ctypes.c_int),
        "qc_num_colors":   ([], ctypes.c_int),
        "qc_num_spins":    ([], ctypes.c_int),
        "qc_field_elements": ([], ctypes.c_long),
        "qc_lane_elements":  ([], ctypes.c_long),
        "qc_gauge_elements": ([], ctypes.c_long),
        "qc_alloc": ([ctypes.c_long], _p),
        "qc_free":  ([_p], None),
        "qc_field_zero": ([_p], None),
        "qc_site_index": ([ctypes.c_int] * 4, ctypes.c_long),
        "qc_site_coords": ([ctypes.c_long] + [ctypes.POINTER(ctypes.c_int)] * 4,
                           None),
        "qc_component_index": ([ctypes.c_long] + [ctypes.c_int] * 4,
                               ctypes.c_long),
        "qc_import": ([_p, _p, ctypes.c_int, ctypes.c_int], None),
        "qc_export": ([_p, _p, ctypes.c_int, ctypes.c_int], None),
        "qc_init": ([_p, _p], None),
        "qc_finalize": ([], None),
        "qc_rank":      ([], ctypes.c_int),
        "qc_num_ranks": ([], ctypes.c_int),
        "qc_read_gauge_file": ([_p, ctypes.c_char_p], None),
        "qc_default_setup_params": ([ctypes.POINTER(SetupParams)], None),
        "qc_default_solve_params": ([ctypes.POINTER(SolveParams)], None),
        "qc_setup": ([_p, ctypes.c_int, ctypes.POINTER(SetupParams)], _p),
        "qc_solve": ([_p, _p, _p, ctypes.POINTER(SolveParams)], ctypes.c_int),
        "qc_apply_D": ([_p, _p, _p], None),
        "qc_report_profile": ([_p], None),
        "qc_teardown": ([_p], None),
    }
    for name, (args, res) in sig.items():
        fn = getattr(_lib, name)
        fn.argtypes = args
        fn.restype = res


_declare()


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
# All of these except local_origin are compile-time constants inside the
# library, so they are readable before qc_init -- which is what lets require()
# run at import time, before MPI has been touched.

def global_dims():
    d = _int4()
    _lib.qc_global_dims(d)
    return tuple(d)


def local_dims():
    d = _int4()
    _lib.qc_local_dims(d)
    return tuple(d)


def local_origin():
    """This rank's global origin.  Needs init(): it comes from the rank grid."""
    d = _int4()
    _lib.qc_local_origin(d)
    return tuple(d)


def num_rhs():     return _lib.qc_num_rhs()
def num_colors():  return _lib.qc_num_colors()
def num_spins():   return _lib.qc_num_spins()
def precision():   return _lib.qc_native_precision()
def rank():        return _lib.qc_rank()
def num_ranks():   return _lib.qc_num_ranks()


def dtype():
    """The numpy dtype of a solver field, from the library rather than assumed.

    Worth taking from here every time.  np.zeros defaults to float64, so a
    single-precision build handed a default-dtype array reads garbage and
    nothing in qc can tell -- the fields cross as void *.
    """
    return np.float64 if precision() == DOUBLE else np.float32


def local_volume():
    """Sites on this rank, derived from the library's own field sizes."""
    return _lib.qc_lane_elements() // (num_spins() * num_colors() * 2)


def field_shape():
    """Shape of a solver field: qc's own order, with the rhs lane fastest."""
    return (local_volume(), num_spins(), num_colors(), 2, num_rhs())


_PRECISION_NAMES = {DOUBLE: "double", SINGLE: "single"}


def config():
    """Everything a caller might want to check, as a dict."""
    return {
        "dims": global_dims(),
        "local_dims": local_dims(),
        "num_rhs": num_rhs(),
        "num_colors": num_colors(),
        "num_spins": num_spins(),
        "precision": _PRECISION_NAMES.get(precision(), precision()),
        "library": os.path.abspath(_lib_path),
    }


def require(**expected):
    """Check the library against what the caller was written for.

    Raises on the first disagreement, naming the field.  Accepts any key config()
    returns; precision takes "double"/"single" or the byte count.

    This is the only thing standing between a mismatch and silent corruption,
    because a solver field crosses the boundary as an untyped pointer: an array
    sized for a different lattice, or of the wrong dtype, is read as though it
    were right.  Call it before allocating anything.
    """
    have = config()
    for key, want in expected.items():
        if key not in have:
            raise TypeError(
                f"require: unknown key {key!r}; expected one of "
                f"{', '.join(sorted(k for k in have if k != 'library'))}")
        got = have[key]
        if key in ("dims", "local_dims"):
            want = tuple(want)
        if key == "precision" and want in (SINGLE, DOUBLE):
            want = _PRECISION_NAMES[want]
        if got != want:
            raise ValueError(
                f"qc was built for {key}={got}, but this program expects "
                f"{want}.  The geometry, num_rhs and precision are compile-time "
                f"constants of {os.path.abspath(_lib_path)} -- rebuild qc for "
                f"this configuration, or change the program to match.")

    # A backstop the per-field checks cannot give.  They only compare what
    # somebody thought to enumerate; this compares the library's own idea of how
    # big a field is against the shape this module would wrap around it, so a
    # layout or padding change that no named check covers still fails here.
    want_elems = 1
    for n in field_shape():
        want_elems *= n
    have_elems = _lib.qc_field_elements()
    if want_elems != have_elems:
        raise ValueError(
            f"field layout disagrees: this module would use shape "
            f"{field_shape()} ({want_elems} reals) but qc_field_elements() "
            f"reports {have_elems}.  The published field layout has changed and "
            f"this module has not caught up -- do not solve with it.")
    return have


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

_initialized = False


def init():
    """Start qc.  Safe to call once; later calls are ignored, as in C.

    In an MPI build this adopts an already-running MPI rather than starting its
    own, so importing mpi4py first is fine and is the expected arrangement.
    """
    global _initialized
    _flush()
    _lib.qc_init(None, None)          # MPI_Init_thread accepts NULL, NULL
    _flush()
    _initialized = True


def finalize():
    """Shut qc down, releasing any fields this module allocated.

    Fields are freed here rather than by the garbage collector.  Tying qc_free
    to collection would mean the release order depends on when Python happens to
    run, including after qc_finalize, and a numpy view outliving its buffer is a
    use-after-free that presents as wrong numbers rather than a crash.
    """
    free_all()
    _lib.qc_finalize()


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------
# qc_alloc rather than numpy: the vector kernels and the streaming stores assume
# 64-byte alignment, which numpy does not promise.

_allocations = []


def _alloc(n_elements):
    ptr = _lib.qc_alloc(n_elements)
    if not ptr:
        raise MemoryError(f"qc_alloc failed for {n_elements} elements")
    _allocations.append(ptr)
    ctype = ctypes.c_double if dtype() == np.float64 else ctypes.c_float
    buf = (ctype * n_elements).from_address(ptr)
    return np.ctypeslib.as_array(buf)


def field():
    """A zeroed solver field, as a numpy view in qc's own order.

    Shape is (sites, spin, color, reim, rhs) -- see qc.h on what the site index
    means.  Use site_index() to get one; the blocking behind it is a cache
    detail and may be retuned, so arithmetic on it does not survive.
    """
    a = _alloc(_lib.qc_field_elements())
    a[:] = 0
    return a.reshape(field_shape())


def gauge_field():
    """An uninitialized gauge array, lexicographic, ready for read_gauge_file."""
    return _alloc(_lib.qc_gauge_elements())


def field_zero(f):
    _lib.qc_field_zero(f.ctypes.data_as(_p))


def free_all():
    for ptr in _allocations:
        _lib.qc_free(ptr)
    _allocations.clear()


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def site_index(x, y, z, t):
    """Index of local site (x, y, z, t) within a field, parity included."""
    return _lib.qc_site_index(x, y, z, t)


def site_coords(site):
    c = [ctypes.c_int() for _ in range(4)]
    _lib.qc_site_coords(site, *[ctypes.byref(v) for v in c])
    return tuple(v.value for v in c)


def component_index(site, s, c, reim, lane):
    """Flat element offset, for callers that want the field one-dimensional."""
    return _lib.qc_component_index(site, s, c, reim, lane)


# ---------------------------------------------------------------------------
# Conversion to and from the conventional order
# ---------------------------------------------------------------------------
# qc_import and qc_export move one rhs lane at a time between a solver field and
# a lexicographic array: t slowest, x fastest, spin outside color, complex
# interleaved -- the ILDG and chroma convention.  These are C loops, which is why
# whole-lattice work goes through them rather than through component_index in a
# Python loop.

def export_lane(f, lane, out=None):
    """One lane of field f as a complex array, shape (Lt, Lz, Ly, Lx, Ns, Nc).

    Always exported at double precision regardless of the build, so the caller
    gets complex128 either way; qc converts if it must.
    """
    lx, ly, lz, lt = local_dims()
    n = _lib.qc_lane_elements()
    if out is None:
        out = np.empty(n, dtype=np.float64)
    _lib.qc_export(f.ctypes.data_as(_p), out.ctypes.data_as(_p), DOUBLE, lane)
    return (out.reshape(lt, lz, ly, lx, num_spins(), num_colors(), 2)
               .view(np.complex128)[..., 0])


def import_lane(f, arr, lane):
    """Write a complex (Lt, Lz, Ly, Lx, Ns, Nc) array into lane `lane` of f."""
    lx, ly, lz, lt = local_dims()
    want = (lt, lz, ly, lx, num_spins(), num_colors())
    if arr.shape != want:
        raise ValueError(f"import_lane expects shape {want}, got {arr.shape}")
    flat = np.ascontiguousarray(arr, dtype=np.complex128).view(np.float64)
    _lib.qc_import(f.ctypes.data_as(_p), flat.ctypes.data_as(_p), DOUBLE, lane)


# ---------------------------------------------------------------------------
# Gauge, setup and solve
# ---------------------------------------------------------------------------

def read_gauge_file(U, path):
    """Fill U from a single-file ILDG or SciDAC configuration.

    qc checks the file's own geometry and checksum and aborts on a mismatch, so
    a configuration for the wrong lattice fails rather than solving on whatever
    bytes it found.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    _flush()
    _lib.qc_read_gauge_file(U.ctypes.data_as(_p), path.encode())
    _flush()


def _params(struct, defaults_fn, kw):
    """Fill a parameter struct: library defaults first, then the caller's keywords.

    The order is the whole point.  Every field has a tuned value that this build
    is regression-tested at, and zero is a legal-looking wrong value for most of
    them -- zero smoothing sweeps, a zero tolerance -- which qc cannot tell from
    a deliberate request.  Building the struct from keywords alone would zero
    everything not named.
    """
    p = struct()
    defaults_fn(ctypes.byref(p))
    known = {name for name, _ in struct._fields_}
    for key, value in kw.items():
        if key not in known:
            raise TypeError(
                f"unknown parameter {key!r}; qc accepts "
                f"{', '.join(sorted(known))}")
        setattr(p, key, int(value) if isinstance(value, bool) else value)
    return p


def setup_params(**kw):
    """Setup parameters: defaults, then whatever you name.  kappa is required."""
    return _params(SetupParams, _lib.qc_default_setup_params, kw)


def solve_params(**kw):
    return _params(SolveParams, _lib.qc_default_solve_params, kw)


class Handle:
    """An MG hierarchy.  Build once per gauge configuration, solve many times."""

    def __init__(self, ptr):
        self._ptr = ptr

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.teardown()
        return False

    def teardown(self):
        if self._ptr:
            _lib.qc_teardown(self._ptr)
            self._ptr = None


def setup(U, params=None, **kw):
    """Build the hierarchy from configuration U.  This is the expensive step.

    U is only read and may be freed as soon as this returns.  Pass a struct from
    setup_params(), or the same keywords directly.
    """
    if params is None:
        params = setup_params(**kw)
    elif kw:
        raise TypeError("pass either a params struct or keywords, not both")
    _flush()
    ptr = _lib.qc_setup(U.ctypes.data_as(_p), precision(), ctypes.byref(params))
    _flush()
    if not ptr:
        raise RuntimeError("qc_setup returned NULL")
    return Handle(ptr)


def solve(h, x, b, params=None, **kw):
    """Solve D x = b for all num_rhs() lanes at once.  Returns the iteration count."""
    if params is None:
        params = solve_params(**kw)
    elif kw:
        raise TypeError("pass either a params struct or keywords, not both")
    _flush()
    iters = _lib.qc_solve(h._ptr, x.ctypes.data_as(_p), b.ctypes.data_as(_p),
                          ctypes.byref(params))
    _flush()
    return iters


def apply_D(h, out, inp):
    """out = D inp, so a caller can form its own residual b - D x."""
    _lib.qc_apply_D(h._ptr, out.ctypes.data_as(_p), inp.ctypes.data_as(_p))


def report_profile(h):
    """Print where the last solve's time went: one line per kernel, then totals.

    Describes that solve alone -- solve() clears the counters on entry -- and
    prints on rank 0 only.  This is what says whether a parameter change moved
    the smoother, the coarse grid or the transfers, so it is worth calling while
    tuning even though nothing else needs it.
    """
    _flush()
    _lib.qc_report_profile(h._ptr)
    _flush()


def teardown(h):
    h.teardown()
