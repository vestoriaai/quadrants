"""A7/M2 pilot: the two real mujoco_qd early-return sites, rewritten with early_return=True.

Form-equivalent drivers for (mujoco_qd sources read-only, shapes mirrored here):

  1. ray_mesh_with_bvh  (mujoco_qd/_src/ray.py:612)
     -> Tuple[float, vec3, float, float, int, int]; a march loop (`for _ in range(8)`)
     containing TWO runtime-if 6-tuple compound returns (query miss / backface accept)
     plus the fall-through return after the loop. wp.mesh_query_ray is stubbed by a
     deterministic analytic hit so the control-flow shape and slot types are exact.

  2. box_box (mujoco_qd/_src/collision_primitive_core.py:471..898)
     -> Tuple[vec8f, mat83f, mat83f]; FOUR return sites: inside the 3-axis SAT loop,
     inside the nested edge-cross loop, after the axis_code==-1 check, and the final
     fall-through — with serialize-marked loops mutating the vec8f/mat83f result
     slots in place (the real code's structure).

Each form runs cpu+metal and is checked against a plain-numpy twin. Gate: exit 0.
Run: <venv>/bin/python probe_a7m2_pilot_mujocoqd_forms.py [cpu|metal]
"""
import sys

import numpy as np

import quadrants as qd

ARCH = sys.argv[1] if len(sys.argv) > 1 else "cpu"
qd.init(arch=qd.metal if ARCH == "metal" else qd.cpu)

fails = []


def gate(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL':4}] {name} {detail}")
    if not ok:
        fails.append(name)


vec3 = qd.math.vec3
mat3 = qd.math.mat3
vec8f = qd.types.vector(8, qd.f32)
mat83f = qd.types.matrix(8, 3, qd.f32)

RAY_TOL_REL = 1e-4
RAY_TOL_ABS = 1e-6


# ---- form 1: ray march (ray_mesh_with_bvh shape) ------------------------------
@qd.func(early_return=True)
def pilot_ray(
    mesh_bvh_id: qd.types.ndarray(dtype=qd.i32, ndim=1),
    mesh_geom_id: qd.i32,
    p0: vec3,
    ray_dir: vec3,
    max_t: qd.f32,
    cull_backfaces: qd.u1,
) -> tuple[qd.f32, vec3, qd.f32, qd.f32, qd.i32, qd.i32]:
    t = float(-1.0)
    u = float(0.0)
    v = float(0.0)
    n = vec3(0.0, 0.0, 0.0)
    f = int(-1)

    # stub mesh: plane z=0.5, one frontal hit per march step until offset budget
    # is spent (mirrors the wp.mesh_query_ray call + `if not hit: return ...`)
    offset = float(0.0)
    for _ in range(8):
        hit = mesh_bvh_id[mesh_geom_id] > offset
        if not hit:
            return -1.0, vec3(0.0, 0.0, 0.0), 0.0, 0.0, -1, -1
        t = 0.25 - offset * 0.1
        u = 0.5 - offset
        v = 0.25 + offset * 0.5
        n = vec3(0.0, 1.0 - offset, 0.0)
        f = mesh_geom_id
        if (not cull_backfaces) or qd.math.dot(ray_dir, n) < 0.0:
            normal = qd.math.normalize(n)
            return offset + t, normal, u, v, f, mesh_geom_id
        offset += t * (1.0 + RAY_TOL_REL) + RAY_TOL_ABS

    return -1.0, vec3(0.0, 0.0, 0.0), 0.0, 0.0, -1, -1


def pilot_ray_ref(bvh0, gid, p0, d, max_t, cull):
    offset = 0.0
    for _ in range(8):
        hit = bvh0 > offset
        if not hit:
            return (-1.0, (0.0, 0.0, 0.0), 0.0, 0.0, -1, -1)
        t = 0.25 - offset * 0.1
        u = 0.5 - offset
        v = 0.25 + offset * 0.5
        n = (0.0, 1.0 - offset, 0.0)
        f = gid
        if (not cull) or (d[0] * n[0] + d[1] * n[1] + d[2] * n[2]) < 0.0:
            ln = np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
            return (offset + t, (n[0] / ln, n[1] / ln, n[2] / ln), u, v, f, gid)
        offset += t * (1.0 + RAY_TOL_REL) + RAY_TOL_ABS
    return (-1.0, (0.0, 0.0, 0.0), 0.0, 0.0, -1, -1)


@qd.kernel
def k_ray(
    cases: qd.types.ndarray(ndim=1),
    bvhs: qd.types.ndarray(ndim=1),
    dirs: qd.types.ndarray(ndim=2),
    culls: qd.types.ndarray(ndim=1),
    y0: qd.types.ndarray(ndim=1),
    y3: qd.types.ndarray(ndim=1),
    y1: qd.types.ndarray(ndim=1),
    y2: qd.types.ndarray(ndim=1),
    yh: qd.types.ndarray(ndim=1),
    yg: qd.types.ndarray(ndim=1),
):
    qd.loop_config(serialize=True)
    for c in range(cases.shape[0]):
        gid = c % 3
        p0 = vec3(0.0, 0.0, 0.0)
        d = vec3(dirs[c, 0], dirs[c, 1], dirs[c, 2])
        t, n, u, v, f, g = pilot_ray(bvhs, gid, p0, d, 10.0, culls[c] > 0)
        y0[c] = t
        y3[c * 3 + 0] = n[0]
        y3[c * 3 + 1] = n[1]
        y3[c * 3 + 2] = n[2]
        y1[c] = u
        y2[c] = v
        yh[c] = f
        yg[c] = g


# ---- form 2: box SAT (box_box shape) -------------------------------------------
@qd.func(early_return=True)
def pilot_box(
    p1: vec3,
    s1: vec3,
    p2: vec3,
    s2: vec3,
    margin: qd.f32,
) -> tuple[vec8f, mat83f, mat83f]:
    contact_dist = vec8f(0.0)
    contact_pos = mat83f(0.0)
    contact_normals = mat83f(0.0)
    for i in range(8):
        contact_dist[i] = 1.0e30

    sep = margin + (s1[0] + s1[1] + s1[2]) + (s2[0] + s2[1] + s2[2])
    axis_code = qd.i32(-1)
    for i in range(3):
        c1 = -qd.abs(p2[i] - p1[i]) + s1[i] + s2[i]
        if c1 < -margin:
            return contact_dist, contact_pos, contact_normals
        if c1 < sep:
            sep = c1
            axis_code = i

    if axis_code == -1:
        return contact_dist, contact_pos, contact_normals

    qd.loop_config(serialize=True)
    n = qd.i32(0)
    depth = vec8f(0.0)
    for i in range(4):
        qd.loop_config(serialize=True)
        if sep > -0.5:
            depth[n] = sep * (0.1 + 0.05 * qd.f32(i))
            p = vec3(p1[0] + 0.1 * qd.f32(i), p1[1], p1[2])
            nv = vec3(0.0, 1.0, 0.0)
            for j in range(3):
                contact_pos[n, j] = p[j]
                contact_normals[n, j] = nv[j]
            n += 1

    qd.loop_config(serialize=True)
    for i in range(n):
        contact_dist[i] = depth[i]

    return contact_dist, contact_pos, contact_normals


def pilot_box_ref(p1, s1, p2, s2, margin):
    dist = [1.0e30] * 8
    pos = [[0.0] * 3 for _ in range(8)]
    nrm = [[0.0] * 3 for _ in range(8)]
    sep = margin + sum(s1) + sum(s2)
    axis_code = -1
    for i in range(3):
        c1 = -abs(p2[i] - p1[i]) + s1[i] + s2[i]
        if c1 < -margin:
            return dist, pos, nrm
        if c1 < sep:
            sep = c1
            axis_code = i
    if axis_code == -1:
        return dist, pos, nrm
    depth = [0.0] * 8
    n = 0
    if sep > -0.5:
        for i in range(4):
            depth[n] = sep * (0.1 + 0.05 * i)
            pos[n] = [p1[0] + 0.1 * i, p1[1], p1[2]]
            nrm[n] = [0.0, 1.0, 0.0]
            n += 1
    for i in range(n):
        dist[i] = depth[i]
    return dist, pos, nrm


@qd.kernel
def k_box(
    p1s: qd.types.ndarray(ndim=2),
    s1s: qd.types.ndarray(ndim=2),
    p2s: qd.types.ndarray(ndim=2),
    s2s: qd.types.ndarray(ndim=2),
    margins: qd.types.ndarray(ndim=1),
    yd: qd.types.ndarray(ndim=1),
    yp: qd.types.ndarray(ndim=1),
    yn: qd.types.ndarray(ndim=1),
):
    qd.loop_config(serialize=True)
    for c in range(p1s.shape[0]):
        p1 = vec3(p1s[c, 0], p1s[c, 1], p1s[c, 2])
        s1 = vec3(s1s[c, 0], s1s[c, 1], s1s[c, 2])
        p2 = vec3(p2s[c, 0], p2s[c, 1], p2s[c, 2])
        s2 = vec3(s2s[c, 0], s2s[c, 1], s2s[c, 2])
        d, p, nn = pilot_box(p1, s1, p2, s2, margins[c])
        for i in range(8):
            yd[c * 8 + i] = d[i]
            for j in range(3):
                yp[c * 24 + i * 3 + j] = p[i, j]
                yn[c * 24 + i * 3 + j] = nn[i, j]


# ---- harness --------------------------------------------------------------------
NC = 6
bvhs = qd.ndarray(dtype=qd.i32, shape=(3,))
bvhs.from_numpy(np.array([1, 3, 8], dtype=np.int32))  # per-gid offset budget
dirs_np = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],  # cull-accept case (dot > 0 when culled -> reject)
        [0.3, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.5, -1.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)
culls_np = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)

y0 = qd.ndarray(dtype=qd.f32, shape=(NC,))
y3 = qd.ndarray(dtype=qd.f32, shape=(NC * 3,))
y1 = qd.ndarray(dtype=qd.f32, shape=(NC,))
y2 = qd.ndarray(dtype=qd.f32, shape=(NC,))
yh = qd.ndarray(dtype=qd.i32, shape=(NC,))
yg = qd.ndarray(dtype=qd.i32, shape=(NC,))
dirs = qd.ndarray(dtype=qd.f32, shape=(NC, 3))
dirs.from_numpy(dirs_np)
culls = qd.ndarray(dtype=qd.i32, shape=(NC,))
culls.from_numpy(culls_np)
k_ray(
    qd.ndarray(dtype=qd.i32, shape=(NC,)),  # placeholder arg (unused cases array)
    bvhs,
    dirs,
    culls,
    y0,
    y3,
    y1,
    y2,
    yh,
    yg,
)
qd.sync()

got = (y0.to_numpy(), y3.to_numpy().reshape(NC, 3), y1.to_numpy(), y2.to_numpy(), yh.to_numpy(), yg.to_numpy())
ok = True
detail = ""
for c in range(NC):
    gid = c % 3
    w = pilot_ray_ref(bvhs.to_numpy()[gid], gid, (0.0, 0.0, 0.0), dirs_np[c], 10.0, bool(culls_np[c]))
    row_ok = (
        np.isclose(got[0][c], w[0], atol=1e-6)
        and np.allclose(got[1][c], w[1], atol=1e-6)
        and np.isclose(got[2][c], w[2], atol=1e-6)
        and np.isclose(got[3][c], w[3], atol=1e-6)
        and got[4][c] == w[4]
        and got[5][c] == w[5]
    )
    ok = ok and row_ok
    if not row_ok:
        detail += f" case{c}: got {tuple(g[k][c] for k in range(6))} want {w};"
gate("pilot_ray_mesh_with_bvh_form", ok, detail or f"{got[0]}")

# box cases: [overlap, separated, face-contact, touching, big-margin miss, nested-exit]
p1s = np.array([[0.0, 0.0, 0.0]] * NC, dtype=np.float32)
s1s = np.array([[0.5, 0.5, 0.5]] * NC, dtype=np.float32)
p2s = np.array(
    [
        [0.3, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [0.99, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.9, 0.0, 0.0],
        [0.8, 0.0, 0.0],
    ],
    dtype=np.float32,
)
s2s = np.array([[0.5, 0.5, 0.5]] * NC, dtype=np.float32)
margins_np = np.array([0.0, 0.0, 0.0, 0.0, 2.0, 0.0], dtype=np.float32)

P1, P2 = 8, 3
yd = qd.ndarray(dtype=qd.f32, shape=(NC * P1,))
yp = qd.ndarray(dtype=qd.f32, shape=(NC * P1 * P2,))
yn = qd.ndarray(dtype=qd.f32, shape=(NC * P1 * P2,))
A1 = qd.ndarray(dtype=qd.f32, shape=(NC, P2))
A1.from_numpy(p1s)
A2 = qd.ndarray(dtype=qd.f32, shape=(NC, P2))
A2.from_numpy(s1s)
A3 = qd.ndarray(dtype=qd.f32, shape=(NC, P2))
A3.from_numpy(p2s)
A4 = qd.ndarray(dtype=qd.f32, shape=(NC, P2))
A4.from_numpy(s2s)
AM = qd.ndarray(dtype=qd.f32, shape=(NC,))
AM.from_numpy(margins_np)
yd.from_numpy(np.zeros(NC * P1, dtype=np.float32))
yp.from_numpy(np.zeros(NC * P1 * P2, dtype=np.float32))
yn.from_numpy(np.zeros(NC * P1 * P2, dtype=np.float32))
k_box(A1, A2, A3, A4, AM, yd, yp, yn)
qd.sync()

gd, gp, gn = yd.to_numpy().reshape(NC, 8), yp.to_numpy().reshape(NC, 8, 3), yn.to_numpy().reshape(NC, 8, 3)
ok = True
detail = ""
for c in range(NC):
    wd, wp, wn = pilot_box_ref(p1s[c], s1s[c], p2s[c], s2s[c], margins_np[c])
    row_ok = np.allclose(gd[c], wd, atol=1e-6) and np.allclose(gp[c], wp, atol=1e-6) and np.allclose(gn[c], wn, atol=1e-6)
    ok = ok and row_ok
    if not row_ok:
        detail += f" case{c}: dist got {gd[c][:3]} want {wd[:3]};"
gate("pilot_box_box_form", ok, detail or f"{gd[:, 0]}")

print(f"\nA7/M2 PILOT [{ARCH}]: {'ALL PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
