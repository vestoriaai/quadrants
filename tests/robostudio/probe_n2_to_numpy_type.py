#!/usr/bin/env python3
"""N2 probe: to_numpy_type accepts compound types (element-dtype unwrap).

Before N2, qd.lang.util.to_numpy_type(dt) only accepted scalar qd DataTypes and
hit `assert False` on VectorType/MatrixType; every consumer that held a compound
dtype had to unwrap `.dtype` layer by layer first (the D19 workaround in
mujoco_qd/_src/io.py).  N2 teaches to_numpy_type to recursively unwrap compound
types to the element np dtype; shape/extent semantics stay with the caller.

Pinned behaviour:

  S1  scalar table          -- all 12 scalar branches, explicit expected np dtype
                               objects (`is` comparison, locks regressions)
  S2  compound direct call  -- VectorType (f32/i32/f64 x vec3/vec6) and MatrixType
                               (f32 mat2x3, f64 mat3x3) map to the element np dtype
  S3  nested compound       -- public construction is NOT possible (cook_dtype
                               rejects a compound element dtype) -> SKIP line;
                               recursion depth > 1 is exercised by a clearly
                               labelled synthetic check (attribute injection)
  S4  real consumption path -- qd.ndarray(dtype=<vector/matrix dtype>) to_numpy()
                               / from_numpy() round trip (the io/sensor shape;
                               impl.ndarray unwraps the compound dtype to the
                               element scalar dtype before the Ndarray classes,
                               so to_numpy internally feeds to_numpy_type a
                               scalar -- pinned here so both sides stay honest)
  S5  unknown dtype         -- must stay LOUD (raise), never silently succeed

Usage: probe_n2_to_numpy_type.py [arch]   (default cpu; API is arch-independent)
Verdict: RESULT: PASS / RESULT: FAIL (exit 1)
"""
import sys

import numpy as np

import quadrants as qd

qd.init(arch=getattr(qd, sys.argv[1] if len(sys.argv) > 1 else "cpu"))

from quadrants.lang.util import to_numpy_type  # noqa: E402  (after qd.init banner)

fails = []


def check(name, ok):
    print(f"  {name:34s} {'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(name)


# S1 -- scalar table: every branch of the pre-N2 function, exact object ---------
SCALAR_TABLE = [
    ("f16", qd.f16, np.half),
    ("f32", qd.f32, np.float32),
    ("f64", qd.f64, np.float64),
    ("i8", qd.i8, np.int8),
    ("i16", qd.i16, np.int16),
    ("i32", qd.i32, np.int32),
    ("i64", qd.i64, np.int64),
    ("u1", qd.u1, np.bool_),
    ("u8", qd.u8, np.uint8),
    ("u16", qd.u16, np.uint16),
    ("u32", qd.u32, np.uint32),
    ("u64", qd.u64, np.uint64),
]
for name, dt, want in SCALAR_TABLE:
    got = to_numpy_type(dt)
    check(f"S1 {name:3s} -> {np.__name__}.{want.__name__}", got is want)

# S2 -- compound direct calls: element dtype, no shape attached -----------------
V = qd.types.vector
M = qd.types.matrix
COMPOUND_TABLE = [
    ("vec3 f32", V(3, qd.f32), np.float32),
    ("vec6 f32", V(6, qd.f32), np.float32),
    ("vec3 i32", V(3, qd.i32), np.int32),
    ("vec6 i32", V(6, qd.i32), np.int32),
    ("vec3 f64", V(3, qd.f64), np.float64),
    ("vec6 f64", V(6, qd.f64), np.float64),
    ("mat2x3 f32", M(2, 3, qd.f32), np.float32),
    ("mat3x3 f64", M(3, 3, qd.f64), np.float64),
]
for name, dt, want in COMPOUND_TABLE:
    got = to_numpy_type(dt)
    check(f"S2 {name:12s} -> {want.__name__}", got is want)

# S3 -- nested compound: honesty first, recursion mechanics second --------------
try:
    nested = V(2, V(3, qd.i32))
    nested_ok = True
    nested_err = ""
except Exception as e:  # pylint: disable=W0703  (probe: classify and report)
    nested_ok = False
    nested_err = f"{type(e).__name__}: {e}"
if nested_ok:
    check("S3 nested vec-of-vec public construct", to_numpy_type(nested) is np.int32)
else:
    print(
        "  SKIP S3 nested vec-of-vec: not publicly constructible "
        f"({nested_err}); recursion depth>1 via labelled synthetic check below"
    )
    # Synthetic (clearly labelled): the design is "recursively unwrap", so pin
    # that the recursion itself is depth-independent via attribute injection.
    # This is a mechanics check of to_numpy_type, NOT a publicly constructible
    # type.
    two_level = V(2, qd.i32)
    two_level.dtype = V(3, qd.f32)
    check("S3 synthetic 2-level unwrap (labelled)", to_numpy_type(two_level) is np.float32)

# S4 -- real consumption path: compound-dtype ndarray round trip ----------------
v3 = V(3, qd.f32)
v6i = V(6, qd.i32)
m23 = M(2, 3, qd.f32)
x = qd.ndarray(v3, shape=6)
src = np.arange(18, dtype=np.float32).reshape(6, 3)
x.from_numpy(src)
out = x.to_numpy()
check("S4 vec3 ndarray round trip (vals)", out.shape == (6, 3) and bool((out == src).all()))
check("S4 vec3 ndarray round trip (dtype)", out.dtype == np.float32)

y = qd.ndarray(v6i, shape=3)
src_i = np.arange(18, dtype=np.int32).reshape(3, 6)
y.from_numpy(src_i)
out_i = y.to_numpy()
check("S4 vec6 i32 ndarray round trip (vals)", out_i.shape == (3, 6) and bool((out_i == src_i).all()))
check("S4 vec6 i32 ndarray round trip (dtype)", out_i.dtype == np.int32)

z = qd.ndarray(m23, shape=4)
src_m = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
z.from_numpy(src_m)
out_m = z.to_numpy()
check("S4 mat2x3 ndarray round trip (vals)", out_m.shape == (4, 2, 3) and bool((out_m == src_m).all()))
check("S4 mat2x3 ndarray round trip (dtype)", out_m.dtype == np.float32)

sc = qd.ndarray(qd.f32, shape=5)
src_s = np.arange(5, dtype=np.float32)
sc.from_numpy(src_s)
out_s = sc.to_numpy()
check("S4 scalar ndarray round trip (dtype)", out_s.dtype == np.float32 and bool((out_s == src_s).all()))

# impl.ndarray unwraps the compound dtype before the Ndarray classes, so the
# Ndarray-level .dtype that reaches to_numpy_type inside to_numpy stays scalar.
# Info line documenting the mechanism (not a gate): if this ever changes, the
# to_numpy_type compound branch becomes the live path for ndarrays too.
print(f"  info: VectorNdarray.dtype stays scalar ({x.dtype} == qd.f32: {x.dtype == qd.f32})")

# S5 -- unknown dtype must stay loud (raise), never silently succeed ------------
raised = False
try:
    to_numpy_type("not-a-dtype")
except Exception:  # pylint: disable=W0703  (probe: loudness gate)
    raised = True
check("S5 unknown dtype raises (loud)", raised)

print("RESULT:", "PASS" if not fails else f"FAIL {fails}")
sys.exit(0 if not fails else 1)
