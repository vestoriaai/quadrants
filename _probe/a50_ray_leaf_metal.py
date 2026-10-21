"""Per-leaf metal bisection for the _ray Tensor(3)-into-scalar-slot SPIR-V leak.

Each ray.py leaf is compiled alone in its own tiny kernel on metal; the first
failing leaf localizes the leak. Usage:

  cd /Users/mini0/Code/mujoqo-rfw && \
    ../mujoqo/_port.venv/bin/python /Users/mini0/Code/quadrants/_probe/a50_ray_leaf_metal.py
"""
import numpy as np

import quadrants as qd
from mujoco_qd._src import ray as R

qd.init(arch=qd.metal, offline_cache=False)

vec6 = qd.types.vector(6, qd.f32)
out3 = qd.ndarray(dtype=qd.math.vec3, shape=[2])
out1 = qd.ndarray(dtype=qd.f32, shape=[2])
out6 = qd.ndarray(dtype=qd.f32, shape=[2])
outi = qd.ndarray(dtype=qd.i32, shape=[2])
fails = []


def run(name, fn):
    try:
        fn()
        qd.sync()
        print(f"  {name:44s} OK")
    except Exception as e:
        print(f"  {name:44s} FAIL {str(e).splitlines()[0][:100]}")
        fails.append(name)


def k_plane():
    @qd.kernel
    def k(o3: qd.types.ndarray(dtype=qd.math.vec3, ndim=1), o1: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        d, n = R.ray_plane(
            qd.math.vec3(0.0, 0.0, 0.0),
            qd.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            qd.math.vec3(0.5),
            qd.math.vec3(0.0, 0.0, 5.0),
            qd.math.vec3(0.0, 0.0, -1.0),
        )
        o1[0] = d
        o3[0] = n

    k(out3, out1)


def k_sphere():
    @qd.kernel
    def k(o3: qd.types.ndarray(dtype=qd.math.vec3, ndim=1), o1: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        d, n = R.ray_sphere(qd.math.vec3(0.0, 0.0, 0.0), 4.0, qd.math.vec3(0.0, 0.0, 5.0), qd.math.vec3(0.0, 0.0, -1.0))
        o1[0] = d
        o3[0] = n

    k(out3, out1)


def k_capsule():
    @qd.kernel
    def k(o3: qd.types.ndarray(dtype=qd.math.vec3, ndim=1), o1: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        d, n = R.ray_capsule(
            qd.math.vec3(0.0, 0.0, 0.0),
            qd.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            qd.math.vec3(0.3, 0.3, 1.0),
            qd.math.vec3(0.0, 0.0, 5.0),
            qd.math.vec3(0.0, 0.0, -1.0),
        )
        o1[0] = d
        o3[0] = n

    k(out3, out1)


def k_ellipsoid():
    @qd.kernel
    def k(o3: qd.types.ndarray(dtype=qd.math.vec3, ndim=1), o1: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        d, n = R.ray_ellipsoid(
            qd.math.vec3(0.0, 0.0, 0.0),
            qd.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            qd.math.vec3(0.5, 0.4, 0.3),
            qd.math.vec3(0.0, 0.0, 5.0),
            qd.math.vec3(0.0, 0.0, -1.0),
        )
        o1[0] = d
        o3[0] = n

    k(out3, out1)


def k_cylinder():
    @qd.kernel
    def k(o3: qd.types.ndarray(dtype=qd.math.vec3, ndim=1), o1: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        d, n = R.ray_cylinder(
            qd.math.vec3(0.0, 0.0, 0.0),
            qd.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            qd.math.vec3(0.3, 0.3, 1.0),
            qd.math.vec3(0.0, 0.0, 5.0),
            qd.math.vec3(0.0, 0.0, -1.0),
        )
        o1[0] = d
        o3[0] = n

    k(out3, out1)


def k_box():
    @qd.kernel
    def k(o3: qd.types.ndarray(dtype=qd.math.vec3, ndim=1), o1: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        d, all6, n = R.ray_box(
            qd.math.vec3(0.0, 0.0, 0.0),
            qd.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            qd.math.vec3(0.5),
            qd.math.vec3(2.0, 0.0, 0.0),
            qd.math.vec3(-1.0, 0.0, 0.0),
        )
        o1[0] = d
        o3[0] = n

    k(out3, out1)


def k_geom():
    @qd.kernel
    def k(o3: qd.types.ndarray(dtype=qd.math.vec3, ndim=1), o1: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        d, n = R.ray_geom(
            qd.math.vec3(0.0, 0.0, 0.0),
            qd.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            qd.math.vec3(0.5),
            qd.math.vec3(2.0, 0.0, 0.0),
            qd.math.vec3(-1.0, 0.0, 0.0),
            6,
        )
        o1[0] = d
        o3[0] = n

    k(out3, out1)


def k_quad():
    @qd.kernel
    def k(o3: qd.types.ndarray(dtype=qd.math.vec3, ndim=1), o1: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        d, uv = R._ray_quad(1.0, -1.5, 0.5)
        o1[0] = d

    k(out3, out1)


def k_tri():
    @qd.kernel
    def k(o3: qd.types.ndarray(dtype=qd.math.vec3, ndim=1), o1: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        d, n = R._ray_triangle(
            qd.math.vec3(-1.0, 0.0, 0.0),
            qd.math.vec3(1.0, 0.0, 0.0),
            qd.math.vec3(0.0, 1.0, 0.0),
            qd.math.vec3(0.0, 0.1, 5.0),
            qd.math.vec3(0.0, 0.0, -1.0),
            qd.math.vec3(1.0, 0.0, 0.0),
            qd.math.vec3(0.0, 1.0, 0.0),
        )
        o1[0] = d
        o3[0] = n

    k(out3, out1)


def k_tri_hit():
    @qd.kernel
    def k(o3: qd.types.ndarray(dtype=qd.math.vec3, ndim=1), o1: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        t, u, v, fid, n = R._ray_tri_hit(
            qd.math.vec3(-1.0, 0.0, 0.0),
            qd.math.vec3(1.0, 0.0, 0.0),
            qd.math.vec3(0.0, 1.0, 0.0),
            qd.math.vec3(0.0, 0.1, 5.0),
            qd.math.vec3(0.0, 0.0, -1.0),
            100.0,
            1e-6,
        )
        o1[0] = t
        o3[0] = n

    k(out3, out1)


def k_map():
    @qd.kernel
    def k(o3: qd.types.ndarray(dtype=qd.math.vec3, ndim=1), o1: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        p, v = R._ray_map(
            qd.math.vec3(0.0, 0.0, 0.0),
            qd.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            qd.math.vec3(0.0, 0.0, 5.0),
            qd.math.vec3(0.0, 0.0, -1.0),
        )
        o3[0] = p
        o3[1] = v

    k(out3, out1)


def k_basis():
    @qd.kernel
    def k(o3: qd.types.ndarray(dtype=qd.math.vec3, ndim=1), o1: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        b0, b1 = R._orthogonal_basis(qd.math.vec3(0.0, 0.0, -1.0))
        o3[0] = b0
        o3[1] = b1

    k(out3, out1)


if __name__ == "__main__":
    run("ray_plane", k_plane)
    run("ray_sphere", k_sphere)
    run("ray_capsule", k_capsule)
    run("ray_ellipsoid", k_ellipsoid)
    run("ray_cylinder", k_cylinder)
    run("ray_box (vec6 in 3-tuple)", k_box)
    run("ray_geom (dispatcher)", k_geom)
    run("_ray_quad (vec2 tuple)", k_quad)
    run("_ray_triangle", k_tri)
    run("_ray_tri_hit (5-tuple vec3)", k_tri_hit)
    run("_ray_map (2x vec3 tuple)", k_map)
    run("_orthogonal_basis", k_basis)
    print(f"A50 RAY LEAF METAL: {12 - len(fails)}/12 PASS, fails={fails}")
