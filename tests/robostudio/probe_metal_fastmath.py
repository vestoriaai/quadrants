"""Gate 22: Program-level `fast_math` reaches the Metal shader compiler (A69).

Defect line mujocoqd/a69-metal-fastmath (docs/a69_metal_fastmath_development_spec.md).

Pre-fix, MTLCompileOptions left Metal's default fast math ON regardless of
`qd.init(fast_math=...)`: FMA contraction / reassociation made bit-parity with
the LLVM backends unreachable (mujoqo util_misc wrap_circle tangent case,
collision_gjk box_box_max/max2). Post-fix, `fast_math=False` selects
MTLMathModeSafe (IEEE strict; `fastMathEnabled=NO` on pre-macOS-15).

Form: the mujoqo util_misc.wrap_circle tangent-point construction (two 2-D
endpoints, circle radius; the four candidate tangent points sol00/sol01/sol10/
sol11 via safe_div). For an exactly tangent segment (row 0: x = 0.1, radius
0.1) sol10 == sol11 == (0.1, 0) under step-by-step f32 rounding; Metal fast
math keeps `radius*end*sqrt` at extended precision and leaves sol11.y at
~4e-10 (observed 4.0572468e-10 on Apple M4), which flips the downstream
length_circle sign test. Strict mode must reproduce a numpy f32 twin bit-for-bit.

Usage: python probe_metal_fastmath.py <cpu|metal> <0|1>   (0 = fast_math off)
  hard : fast_math=0 -> all outputs bit-identical to the f32 twin (both arches)
  info : fast_math=1 -> drift count printed (metal: expected > 0; cpu: 0)
Prints "FASTMATH GATE [<arch> fast_math=<x>]: PASS|FAIL".
"""
import sys

import numpy as np

import quadrants as qd

arch_name = sys.argv[1] if len(sys.argv) > 1 else "cpu"
fast = (sys.argv[2] if len(sys.argv) > 2 else "1") == "1"
qd.init(arch=qd.metal if arch_name == "metal" else qd.cpu, fast_math=fast, offline_cache=False,
        cfg_optimization=False)

MJ_MINVAL = 1e-15


@qd.func
def safe_div(x: qd.template(), y: qd.template()):
  return x / qd.select(y != 0.0, y, MJ_MINVAL)


@qd.func
def tangent_points(end: qd.math.vec4, radius: float) -> tuple[qd.math.vec2, qd.math.vec2, qd.math.vec2, qd.math.vec2]:
  end0 = qd.math.vec2(end[0], end[1])
  end1 = qd.math.vec2(end[2], end[3])
  sqlen0 = qd.math.dot(end0, end0)
  sqlen1 = qd.math.dot(end1, end1)
  sqrad = radius * radius
  sqrt0 = qd.sqrt(sqlen0 - sqrad)
  sqrt1 = qd.sqrt(sqlen1 - sqrad)
  sol00 = qd.math.vec2(safe_div(end[0] * sqrad + radius * end[1] * sqrt0, sqlen0),
                       safe_div(end[1] * sqrad - radius * end[0] * sqrt0, sqlen0))
  sol01 = qd.math.vec2(safe_div(end[2] * sqrad - radius * end[3] * sqrt1, sqlen1),
                       safe_div(end[3] * sqrad + radius * end[2] * sqrt1, sqlen1))
  sol10 = qd.math.vec2(safe_div(end[0] * sqrad - radius * end[1] * sqrt0, sqlen0),
                       safe_div(end[1] * sqrad + radius * end[0] * sqrt0, sqlen0))
  sol11 = qd.math.vec2(safe_div(end[2] * sqrad + radius * end[3] * sqrt1, sqlen1),
                       safe_div(end[3] * sqrad - radius * end[2] * sqrt1, sqlen1))
  return sol00, sol01, sol10, sol11


@qd.kernel
def k(end_in: qd.types.ndarray(dtype=qd.math.vec4, ndim=1), r_in: qd.types.ndarray(dtype=float, ndim=1),
      out: qd.types.ndarray(dtype=float, ndim=2), n: int):
    for i in range(n):
        s00, s01, s10, s11 = tangent_points(end_in[i], r_in[i])
        out[i, 0] = s00[0]
        out[i, 1] = s00[1]
        out[i, 2] = s01[0]
        out[i, 3] = s01[1]
        out[i, 4] = s10[0]
        out[i, 5] = s10[1]
        out[i, 6] = s11[0]
        out[i, 7] = s11[1]


E = np.array([[0.1, -1.0, 0.1, 1.0], [0.3, -1.0, 0.3, 1.0], [0.7, -2.0, 0.7, 2.0], [0.05, -0.5, 0.05, 0.5],
              [0.2, -0.9, 0.4, 1.1]], np.float32)
R = np.array([0.1, 0.3, 0.7, 0.05, 0.15], np.float32)
n = len(R)
end = qd.ndarray(dtype=qd.math.vec4, shape=(n,))
end.from_numpy(E)
r = qd.ndarray(dtype=qd.f32, shape=(n,))
r.from_numpy(R)
out = qd.ndarray(dtype=qd.f32, shape=(n, 8))
k(end, r, out, n)
qd.sync()
o = out.to_numpy()

f = np.float32


def sdiv(x, y):
  return f(x / (y if y != 0.0 else f(MJ_MINVAL)))


want = np.zeros((n, 8), np.float32)
for i in range(n):
  e = E[i]
  rad = R[i]
  sqlen0 = f(f(e[0] * e[0]) + f(e[1] * e[1]))
  sqlen1 = f(f(e[2] * e[2]) + f(e[3] * e[3]))
  sqrad = f(rad * rad)
  sqrt0 = f(np.sqrt(f(sqlen0 - sqrad)))
  sqrt1 = f(np.sqrt(f(sqlen1 - sqrad)))
  want[i, 0] = sdiv(f(f(e[0] * sqrad) + f(f(rad * e[1]) * sqrt0)), sqlen0)
  want[i, 1] = sdiv(f(f(e[1] * sqrad) - f(f(rad * e[0]) * sqrt0)), sqlen0)
  want[i, 2] = sdiv(f(f(e[2] * sqrad) - f(f(rad * e[3]) * sqrt1)), sqlen1)
  want[i, 3] = sdiv(f(f(e[3] * sqrad) + f(f(rad * e[2]) * sqrt1)), sqlen1)
  want[i, 4] = sdiv(f(f(e[0] * sqrad) - f(f(rad * e[1]) * sqrt0)), sqlen0)
  want[i, 5] = sdiv(f(f(e[1] * sqrad) + f(f(rad * e[0]) * sqrt0)), sqlen0)
  want[i, 6] = sdiv(f(f(e[2] * sqrad) + f(f(rad * e[3]) * sqrt1)), sqlen1)
  want[i, 7] = sdiv(f(f(e[3] * sqrad) - f(f(rad * e[2]) * sqrt1)), sqlen1)

tag = f"[{arch_name} fast_math={int(fast)}]"
diff = int((o != want).sum())
for i, j in np.argwhere(o != want)[:8]:
  ulp = int(abs(int(o[i, j].view(np.int32)) - int(want[i, j].view(np.int32))))
  print(f"  diff row={i} col={j} got={o[i, j]!r} twin={want[i, j]!r} ulp={ulp}")
print(f"FASTMATH GATE {tag}: tangent row0 sol10.y={o[0, 5]!r} sol11.y={o[0, 7]!r} (twin {want[0, 7]!r}); "
      f"{diff}/{o.size} values differ from the f32 twin")
if not fast:
  ok = diff == 0
  print(f"FASTMATH GATE {tag}: {'PASS' if ok else 'FAIL'} (strict mode must be bit-identical to the f32 twin)")
  sys.exit(0 if ok else 1)
print(f"FASTMATH GATE {tag}: PASS (informational arm; drift {diff} == fast-math active)")
