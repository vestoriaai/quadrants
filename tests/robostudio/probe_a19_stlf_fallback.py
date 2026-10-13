"""A19 store-to-load-forwarding fallback gate (mujocoqd/a19-stlf-fallback).

The pre-fix compiler asserted in CFGNode::store_to_load_forwarding when the
forwarded value was a whole tensor while the consuming load was non-tensor
and NOT a Const-offset MatrixPtrStmt element load. The repro compile unit is
mujoco_qd's _sensor_touch kernel (its IR holds a tensor value stored whole
into a scalar-typed alloca by scalarize, consumed by plain scalar loads --
an IR shape whose element-extraction precondition cannot hold). The fix
demotes the assert to a conservative fallback: forwarding is an
optimization, so keep the load and compile through instead of aborting.

This file vendors the _sensor_touch kernel + its ray-intersection closure
(Apache-2.0, mujoqo/_src/{ray,math,sensor}.py @ the A19 line; ConeType /
MJ_MINVAL inlined, math/ray namespaces shimmed). On an unfixed wheel this
probe dies with store_to_load_forwarding@425 (gate FAIL); on a fixed wheel
it must compile and match the pinned sensordata output (cpu only: the
defect is an IR-form defect, arch-independent). Pin values are compile+code
drift detectors; mujoco-semantics validation of the vendored kernels lives
in mujoqo's own test suite.

Usage: python probe_a19_stlf_fallback.py
"""

from typing import Tuple

import numpy as np
import quadrants as qd

qd.init(arch=qd.cpu)

MJ_MINVAL = 1e-15


@qd.func
def safe_div(x: qd.template(), y: qd.template()):
  return x / qd.select(y != 0.0, y, MJ_MINVAL)


vec6 = qd.types.vector(6, qd.f32)

# The vendored funcs address vec/mat helpers as `math.*` and the ray family
# as `ray.*` (mujoqo/_src module layout); shim both instead of depending on
# that package. math is seeded from qd.math, then extended below with the
# vendored helpers.
import types as _pytypes

math = _pytypes.SimpleNamespace(
    **{n: getattr(qd.math, n) for n in dir(qd.math) if not n.startswith("_")})


class GeomType:
  PLANE = 0
  HFIELD = 1
  SPHERE = 2
  CAPSULE = 3
  ELLIPSOID = 4
  CYLINDER = 5
  BOX = 6


class ConeType:
  PYRAMIDAL = 0
  ELLIPTIC = 1


# --- vendored mujoco_qd closure (Apache-2.0) ---------------------------------

@qd.real_func
def ray_geom(pos: qd.math.vec3, mat: qd.math.mat3, size: qd.math.vec3, pnt: qd.math.vec3, vec: qd.math.vec3, geomtype: int) -> Tuple[float, qd.math.vec3]:
    if geomtype == GeomType.PLANE:
        return ray_plane(pos, mat, size, pnt, vec)
    elif geomtype == GeomType.SPHERE:
        return ray_sphere(pos, size[0] * size[0], pnt, vec)
    elif geomtype == GeomType.CAPSULE:
        return ray_capsule(pos, mat, size, pnt, vec)
    elif geomtype == GeomType.ELLIPSOID:
        return ray_ellipsoid(pos, mat, size, pnt, vec)
    elif geomtype == GeomType.CYLINDER:
        return ray_cylinder(pos, mat, size, pnt, vec)
    elif geomtype == GeomType.BOX:
        dist_b, _, normal_b = ray_box(pos, mat, size, pnt, vec)
        return dist_b, normal_b
    else:
        return -1.0, qd.math.vec3(0.0)


@qd.func
def ray_plane(pos: qd.math.vec3, mat: qd.math.mat3, size: qd.math.vec3, pnt: qd.math.vec3, vec: qd.math.vec3) -> Tuple[float, qd.math.vec3]:
    """Returns the distance and normal at which a ray intersects with a plane."""
    ret0qd = 0.0
    ret1qd = qd.math.vec3(0.0)
    doneqd = False
    lpnt, lvec = _ray_map(pos, mat, pnt, vec)
    x = -lpnt[2] / lvec[2]
    p = qd.math.vec2(lpnt[0] + x * lvec[0], lpnt[1] + x * lvec[1])
    if lvec[2] > -MJ_MINVAL:
        ret0qd = -1.0
        ret1qd = qd.math.vec3(0.0)
        doneqd = True
    if not doneqd:
        if x < 0.0:
            ret0qd = -1.0
            ret1qd = qd.math.vec3(0.0)
            doneqd = True
    if not doneqd:
        if (size[0] <= 0.0 or qd.abs(p[0]) <= size[0]) and (size[1] <= 0.0 or qd.abs(p[1]) <= size[1]):
            ret0qd = x
            ret1qd = qd.math.vec3(mat[0, 2], mat[1, 2], mat[2, 2])
            doneqd = True
        else:
            ret0qd = -1.0
            ret1qd = qd.math.vec3(0.0)
            doneqd = True
    return (ret0qd, ret1qd)


@qd.func
def ray_sphere(pos: qd.math.vec3, dist_sqr: float, pnt: qd.math.vec3, vec: qd.math.vec3) -> Tuple[float, qd.math.vec3]:
    """Returns the distance and normal at which a ray intersects with a sphere."""
    dif = pnt - pos
    a = qd.math.dot(vec, vec)
    b = qd.math.dot(vec, dif)
    c = qd.math.dot(dif, dif) - dist_sqr
    sol, _ = _ray_quad(a, b, c)
    normal = qd.math.vec3(0.0)
    s = qd.math.vec3(0.0)
    if sol >= 0:
        s = pnt + vec * sol
        normal = qd.math.normalize(s - pos)
    return (sol, normal)


@qd.real_func
def ray_capsule(pos: qd.math.vec3, mat: qd.math.mat3, size: qd.math.vec3, pnt: qd.math.vec3, vec: qd.math.vec3) -> Tuple[float, qd.math.vec3]:
    """Returns the distance and normal at which a ray intersects with a capsule."""
    ret0qd = 0.0
    ret1qd = qd.math.vec3(0.0)
    doneqd = False
    ssz = size[0] + size[1]
    dist_sphere, normal_sphere = ray_sphere(pos, ssz * ssz, pnt, vec)
    # qd: reconstruct upstream order (pipeline scrambled producers)
    lpnt, lvec = _ray_map(pos, mat, pnt, vec)

    x = -1.0
    sq_size0 = size[0] * size[0]
    a = lvec[0] * lvec[0] + lvec[1] * lvec[1]
    b = lvec[0] * lpnt[0] + lvec[1] * lpnt[1]
    c = lpnt[0] * lpnt[0] + lpnt[1] * lpnt[1] - sq_size0
    part = 0
    sol, xx = _ray_quad(a, b, c)
    if sol >= 0.0 and qd.abs(lpnt[2] + sol * lvec[2]) <= size[1]:
        if x < 0.0 or sol < x:
            x = sol
    # top cap
    ldif = qd.math.vec3(lpnt[0], lpnt[1], lpnt[2] - size[1])
    a += lvec[2] * lvec[2]
    b = qd.math.dot(lvec, ldif)
    c = qd.math.dot(ldif, ldif) - sq_size0
    _, xx = _ray_quad(a, b, c)
    for i in range(2):
        if xx[i] >= 0.0 and lpnt[2] + xx[i] * lvec[2] >= size[1]:
            if x < 0.0 or xx[i] < x:
                x = xx[i]
                part = 1
    # bottom cap
    ldif = qd.math.vec3(ldif[0], ldif[1], lpnt[2] + size[1])
    b = qd.math.dot(lvec, ldif)
    c = qd.math.dot(ldif, ldif) - sq_size0
    _, xx = _ray_quad(a, b, c)
    for i in range(2):
        if xx[i] >= 0.0 and lpnt[2] + xx[i] * lvec[2] <= -size[1]:
            if x < 0.0 or xx[i] < x:
                x = xx[i]
                part = -1

        sol, xx = _ray_quad(a, b, c)
    if not doneqd:
        if sol >= 0.0 and qd.abs(lpnt[2] + sol * lvec[2]) <= size[1]:
            if x < 0.0 or sol < x:
                x = sol
    if not doneqd:
        a += lvec[2] * lvec[2]
    if not doneqd:
        _, xx = _ray_quad(a, b, c)
    if not doneqd:
        for i in range(2):
            if xx[i] >= 0.0 and lpnt[2] + xx[i] * lvec[2] >= size[1]:
                if x < 0.0 or xx[i] < x:
                    x = xx[i]
                    part = 1
    if not doneqd:
        _, xx = _ray_quad(a, b, c)
    if not doneqd:
        for i in range(2):
            if xx[i] >= 0.0 and lpnt[2] + xx[i] * lvec[2] <= -size[1]:
                if x < 0.0 or xx[i] < x:
                    x = xx[i]
                    part = -1
    normal = qd.math.vec3(0.0)
    if x >= 0:
        normal[0] = lpnt[0] + lvec[0] * x
        normal[1] = lpnt[1] + lvec[1] * x
        if part == 0:
            normal[2] = 0.0
        else:
            normal[2] = lpnt[2] + lvec[2] * x - size[1] * float(part)
        normal = qd.math.normalize(normal)
        normal = mat @ normal
    return x, normal


@qd.func
def ray_ellipsoid(pos: qd.math.vec3, mat: qd.math.mat3, size: qd.math.vec3, pnt: qd.math.vec3, vec: qd.math.vec3) -> Tuple[float, qd.math.vec3]:
    """Returns the distance and normal at which a ray intersects with an ellipsoid."""
    lpnt, lvec = _ray_map(pos, mat, pnt, vec)
    s = qd.math.vec3(safe_div(1.0, size[0] * size[0]), safe_div(1.0, size[1] * size[1]), safe_div(1.0, size[2] * size[2]))
    slvec = (s) * (lvec)
    a = qd.math.dot(slvec, lvec)
    b = qd.math.dot(slvec, lpnt)
    c = qd.math.dot((s) * (lpnt), lpnt) - 1.0
    sol, _ = _ray_quad(a, b, c)
    normal = qd.math.vec3(0.0)
    l = 0.0
    if sol >= 0:
        l = lpnt + lvec * sol
        normal = (s) * (l)
        normal = qd.math.normalize(normal)
        normal = mat @ normal
    return (sol, normal)


@qd.real_func
def ray_cylinder(pos: qd.math.vec3, mat: qd.math.mat3, size: qd.math.vec3, pnt: qd.math.vec3, vec: qd.math.vec3) -> Tuple[float, qd.math.vec3]:
    """Returns the distance and normal at which a ray intersects with a cylinder."""
    # bounding sphere test
    ssz = size[0] * size[0] + size[1] * size[1]
    dist_sphere, normal_sphere = ray_sphere(pos, ssz, pnt, vec)
    if dist_sphere < 0:
        return -1.0, qd.math.vec3(0.0)

    # qd: reconstruct upstream order (pipeline scrambled producers)
    lpnt, lvec = _ray_map(pos, mat, pnt, vec)

    x = -1.0
    part = 0

    # flat sides
    if qd.abs(lvec[2]) > MJ_MINVAL:
        # qd: no tuple iteration; unrolled (-1, 1)
        sol = (-size[1] - lpnt[2]) / lvec[2]
        if sol >= 0.0:
            p = qd.math.vec2(lpnt[0] + sol * lvec[0], lpnt[1] + sol * lvec[1])
            if qd.math.dot(p, p) <= size[0] * size[0]:
                if x < 0.0 or sol < x:
                    x = sol
                    part = -1
        sol = (size[1] - lpnt[2]) / lvec[2]
        if sol >= 0.0:
            p = qd.math.vec2(lpnt[0] + sol * lvec[0], lpnt[1] + sol * lvec[1])
            if qd.math.dot(p, p) <= size[0] * size[0]:
                if x < 0.0 or sol < x:
                    x = sol
                    part = 1

    a = lvec[0] * lvec[0] + lvec[1] * lvec[1]
    b = lvec[0] * lpnt[0] + lvec[1] * lpnt[1]
    c = lpnt[0] * lpnt[0] + lpnt[1] * lpnt[1] - size[0] * size[0]

    sol, _ = _ray_quad(a, b, c)

    if sol >= 0.0 and qd.abs(lpnt[2] + sol * lvec[2]) <= size[1]:
        if x < 0.0 or sol < x:
            x = sol
            part = 0

    normal = qd.math.vec3(0.0)
    if x >= 0:
        if part == 0:
            normal = lpnt + lvec * x
            normal[2] = 0.0
            normal = qd.math.normalize(normal)
        else:
            normal = qd.math.vec3(0.0, 0.0, float(part))

        normal = mat @ normal

    return x, normal


@qd.real_func
def ray_box(pos: qd.math.vec3, mat: qd.math.mat3, size: qd.math.vec3, pnt: qd.math.vec3, vec: qd.math.vec3) -> Tuple[float, vec6, qd.math.vec3]:
  """Returns distance, per side information, and normal at which a ray intersects with a box."""
  all = vec6(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0)

  # bounding sphere test
  ssz = qd.math.dot(size, size)
  dist_sphere, _ = ray_sphere(pos, ssz, pnt, vec)
  if dist_sphere < 0:
    return -1.0, all, qd.math.vec3(0.0)

  # map to local frame
  lpnt, lvec = _ray_map(pos, mat, pnt, vec)

  # init solution
  x = float(-1.0)
  face_side = -1
  face_axis = -1

  # loop over axes with non-zero vec
  for i in range(3):
    if qd.abs(lvec[i]) > MJ_MINVAL:
      for _side_i in range(2):
        side = -1 + 2 * _side_i
        # solution of: lpnt[i] + x * lvec[i] = side * size[i]
        sol = (float(side) * size[i] - lpnt[i]) / lvec[i]

        # process if non-negative
        if sol >= 0.0:
          id0 = 1 if i == 0 else 0
          id1 = 1 if i == 2 else 2

          # intersection with face
          p0 = lpnt[id0] + sol * lvec[id0]
          p1 = lpnt[id1] + sol * lvec[id1]

          # accept within rectangle
          if (qd.abs(p0) <= size[id0]) and (qd.abs(p1) <= size[id1]):
            # update
            if x < 0.0 or sol < x:
              x = sol
              face_axis = i
              face_side = side

            # save in all
            all[2 * i + (side + 1) // 2] = sol

  normal = qd.math.vec3(0.0)
  if x >= 0:
    normal[face_axis] = float(face_side)
    normal = mat @ normal

  return x, all, normal


@qd.func
def _ray_map(pos: qd.math.vec3, mat: qd.math.mat3, pnt: qd.math.vec3, vec: qd.math.vec3) -> Tuple[qd.math.vec3, qd.math.vec3]:
  """Maps ray to local geom frame coordinates.

  Args:
      pos: position of geom frame
      mat: orientation of geom frame
      pnt: starting point of ray in world coordinates
      vec: direction of ray in world coordinates

  Returns:
      3D point and 3D direction in local geom frame
  """
  matT = math.transposed(mat)
  lpnt = matT @ (pnt - pos)
  lvec = matT @ vec

  return lpnt, lvec


@qd.func
def _ray_quad(a: float, b: float, c: float) -> Tuple[float, qd.math.vec2]:
    """Compute solutions from quadratic: a*x^2 + 2*b*x + c = 0."""
    ret0qd = 0.0
    ret1qd = qd.math.vec2(0.0)
    doneqd = False
    det = b * b - a * c
    det = qd.sqrt(det)
    den = safe_div(1.0, a)
    x0 = (-b - det) * den
    x1 = (-b + det) * den
    x = qd.math.vec2(x0, x1)
    if det < MJ_MINVAL:
        ret0qd = -1.0
        ret1qd = qd.math.vec2(-1.0, -1.0)
        doneqd = True
    if not doneqd:
        if x0 >= 0.0:
            ret0qd = x0
            ret1qd = x
            doneqd = True
        elif x1 >= 0.0:
            ret0qd = x1
            ret1qd = x
            doneqd = True
        else:
            ret0qd = -1.0
            ret1qd = x
            doneqd = True
    return (ret0qd, ret1qd)


@qd.func
def _ray_eliminate(body_weldid: qd.types.ndarray(dtype=int, ndim=1), geom_bodyid: qd.types.ndarray(dtype=int, ndim=1), geom_matid: qd.types.ndarray(dtype=int, ndim=1), geom_group: qd.types.ndarray(dtype=int, ndim=1), geom_rgba: qd.types.ndarray(dtype=qd.math.vec4, ndim=1), mat_rgba: qd.types.ndarray(dtype=qd.math.vec4, ndim=1), geomid: int, geomgroup: vec6, flg_static: bool, bodyexclude: int) -> bool:
    """Eliminate ray."""
    ret0qd = False
    doneqd = False
    bodyid = geom_bodyid[geomid]
    matid = geom_matid[geomid]
    groupid = qd.min(5, qd.max(0, geom_group[geomid]))
    if bodyid == bodyexclude:
        ret0qd = True
        doneqd = True
    if not doneqd:
        if matid < 0 and geom_rgba[geomid][3] == 0.0:
            ret0qd = True
            doneqd = True
    if not doneqd:
        if matid >= 0:
            if mat_rgba[matid][3] == 0.0:
                ret0qd = True
                doneqd = True
    if not doneqd:
        if not flg_static and body_weldid[bodyid] == 0:
            ret0qd = True
            doneqd = True
    if not doneqd:
        if geomgroup[0] == -1 and geomgroup[1] == -1 and (geomgroup[2] == -1) and (geomgroup[3] == -1) and (geomgroup[4] == -1) and (geomgroup[5] == -1):
            ret0qd = False
            doneqd = True
    if not doneqd:
        ret0qd = geomgroup[groupid] == 0
        doneqd = True
    return ret0qd


@qd.func
def normalize_with_norm(x: qd.template()) -> tuple[qd.math.vec3, float]:  # upstream type-generic; vec2 callers in util_misc.wrap (coordinator-adjudicated)
  # qd: manually single-exit (select form); see quat_z2vec note
  norm = qd.math.length(x)
  inv = qd.select(norm > 0.0, 1.0 / norm, 0.0)
  return x * inv, norm


@qd.func
def transposed(m: qd.math.mat3) -> qd.math.mat3:
  """qd counterpart of the warp builtin transpose for mat33."""
  return m.transpose()


@qd.kernel
def _sensor_touch(opt_cone: int, geom_bodyid: qd.types.ndarray(dtype=int, ndim=1), site_type: qd.types.ndarray(dtype=int, ndim=1), site_bodyid: qd.types.ndarray(dtype=int, ndim=1), site_size: qd.types.ndarray(dtype=qd.math.vec3, ndim=1), sensor_objid: qd.types.ndarray(dtype=int, ndim=1), sensor_adr: qd.types.ndarray(dtype=int, ndim=1), sensor_touch_adr: qd.types.ndarray(dtype=int, ndim=1), site_xpos_in: qd.types.ndarray(dtype=qd.math.vec3, ndim=2), site_xmat_in: qd.types.ndarray(dtype=qd.math.mat3, ndim=2), contact_pos_in: qd.types.ndarray(dtype=qd.math.vec3, ndim=1), contact_frame_in: qd.types.ndarray(dtype=qd.math.mat3, ndim=1), contact_dim_in: qd.types.ndarray(dtype=int, ndim=1), contact_geom_in: qd.types.ndarray(dtype=qd.math.ivec2, ndim=1), contact_efc_address_in: qd.types.ndarray(dtype=int, ndim=2), contact_worldid_in: qd.types.ndarray(dtype=int, ndim=1), efc_force_in: qd.types.ndarray(dtype=float, ndim=2), nacon_in: qd.types.ndarray(dtype=int, ndim=1), sensordata_out: qd.types.ndarray(dtype=float, ndim=2), __n0: int, __n1: int):
    adr = 0
    conid = 0
    conray = qd.math.vec3(0.0)
    dim = 0
    efc_address0 = 0
    frame = qd.math.mat3(0.0)
    geom = qd.math.ivec2(0)
    normalforce = 0.0
    sensorid = 0
    sensortouchadrid = 0
    bodyid = 0
    objid = 0
    conbody = qd.math.ivec2(0, 0)
    worldid = 0
    for __t in range(__n0 * __n1):
        doneqd = False
        conid = __t // __n1
        sensortouchadrid = __t % __n1
        if conid >= nacon_in[0]:
            doneqd = True
        if not doneqd:
            sensorid = sensor_touch_adr[sensortouchadrid]
        if not doneqd:
            geom = contact_geom_in[conid]
            conbody = qd.math.ivec2(geom_bodyid[geom[0]], geom_bodyid[geom[1]])
            objid = sensor_objid[sensorid]
            bodyid = site_bodyid[objid]
        if not doneqd:
            worldid = contact_worldid_in[conid]
        if not doneqd:
            efc_address0 = contact_efc_address_in[conid, 0]
        if not doneqd:
            if efc_address0 >= 0 and (bodyid == conbody[0] or bodyid == conbody[1]):
                normalforce = efc_force_in[worldid, efc_address0]
                if opt_cone == ConeType.PYRAMIDAL:
                    dim = contact_dim_in[conid]
                    for i in range(1, 2 * (dim - 1)):
                        normalforce += efc_force_in[worldid, contact_efc_address_in[conid, i]]
                if normalforce <= 0.0:
                    doneqd = True
                if not doneqd:
                    frame = contact_frame_in[conid]
                    conray = qd.math.vec3(frame[0, 0], frame[0, 1], frame[0, 2]) * normalforce
                if not doneqd:
                    conray, _ = math.normalize_with_norm(conray)
                if not doneqd:
                    if bodyid == conbody[1]:
                        conray = -conray
                if not doneqd:
                    # qd: merged gates — tuple unpack consumed in same scope
                    dist, normal = ray.ray_geom(site_xpos_in[worldid, objid], site_xmat_in[worldid, objid], site_size[objid], contact_pos_in[conid], conray, site_type[objid])
                    if dist >= 0.0:
                        adr = sensor_adr[sensorid]
                        qd.atomic_add(sensordata_out[worldid, adr], normalforce)

# --- namespace shims + gate ---------------------------------------------------

math.transposed = transposed  # vendored above
math.normalize_with_norm = normalize_with_norm
ray = _pytypes.SimpleNamespace(ray_geom=ray_geom)

# Mirror the fixture-provided shapes (values are compile-irrelevant; the A19
# assert fires on the kernel's IR regardless of runtime data).
NB = 4   # bodies/sites/sensors
NW = 2   # worlds
NC = 1   # contacts taken
NEFC = 12
geom_bodyid = qd.ndarray(dtype=int, shape=(NB,))
site_type = qd.ndarray(dtype=int, shape=(NB,))
site_bodyid = qd.ndarray(dtype=int, shape=(NB,))
site_size = qd.ndarray(dtype=qd.math.vec3, shape=(NB,))
sensor_objid = qd.ndarray(dtype=int, shape=(NB,))
sensor_adr = qd.ndarray(dtype=int, shape=(NB,))
sensor_touch_adr = qd.ndarray(dtype=int, shape=(NB,))
site_xpos_in = qd.ndarray(dtype=qd.math.vec3, shape=(NW, NB))
site_xmat_in = qd.ndarray(dtype=qd.math.mat3, shape=(NW, NB))
contact_pos_in = qd.ndarray(dtype=qd.math.vec3, shape=(NC,))
contact_frame_in = qd.ndarray(dtype=qd.math.mat3, shape=(NC,))
contact_dim_in = qd.ndarray(dtype=int, shape=(NC,))
contact_geom_in = qd.ndarray(dtype=qd.math.ivec2, shape=(NC,))
contact_efc_address_in = qd.ndarray(dtype=int, shape=(NC, NEFC))
contact_worldid_in = qd.ndarray(dtype=int, shape=(NC,))
efc_force_in = qd.ndarray(dtype=float, shape=(NW, NEFC))
nacon_in = qd.ndarray(dtype=int, shape=(1,))
sensordata_out = qd.ndarray(dtype=float, shape=(NW, NEFC))

geom_bodyid.from_numpy(np.array([0, 0, 1, 1], dtype=np.int32))
site_type.from_numpy(np.array([2, 2, 6, 6], dtype=np.int32))  # SPHERE, SPHERE, BOX, BOX
site_bodyid.from_numpy(np.array([0, 1, 2, 3], dtype=np.int32))
_size_rows = np.tile(np.array([[0.1, 0.3, 0.2]], dtype=np.float32), (NB, 1))
_size_rows[0] = (0.5, 0.0, 0.0)  # site 0: sphere r=0.5 on the (flipped) ray path
site_size.from_numpy(_size_rows)
sensor_objid.from_numpy(np.array([0, 1, 2, 3], dtype=np.int32))
sensor_adr.from_numpy(np.array([0, 4, 8, 3], dtype=np.int32))
sensor_touch_adr.from_numpy(np.array([0, 1, 2, 3], dtype=np.int32))
_xpos_rows = np.tile(np.array([[0.0, 0.0, 0.3]], dtype=np.float32), (NW, NB, 1))
_xpos_rows[0, 0] = (0.0, 0.0, 3.0)  # sphere sits above the contact along +z
site_xpos_in.from_numpy(_xpos_rows)
site_xmat_in.from_numpy(np.stack([np.eye(3, dtype=np.float32)] * NW * NB).reshape(NW, NB, 3, 3))
contact_pos_in.from_numpy(np.array([[0.0, 0.0, 1.0]], dtype=np.float32))
_frame_rows = np.stack([np.eye(3, dtype=np.float32)] * NC)
_frame_rows[0, 0] = (0.0, 0.0, -1.0)  # conray = frame row0 * normalforce = -z
_frame_rows[0, 1] = (0.0, 1.0, 0.0)
_frame_rows[0, 2] = (1.0, 0.0, 0.0)
contact_frame_in.from_numpy(_frame_rows)
contact_dim_in.from_numpy(np.array([3], dtype=np.int32))
contact_geom_in.from_numpy(np.array([[0, 1]], dtype=np.int32))
contact_efc_address_in.from_numpy(np.tile(np.arange(NEFC, dtype=np.int32), (NC, 1)))
contact_worldid_in.from_numpy(np.array([0], dtype=np.int32))
efc_force_in.from_numpy(np.tile(np.arange(NEFC, dtype=np.float32) + 1.0, (NW, 1)))
nacon_in.from_numpy(np.array([NC], dtype=np.int32))
sensordata_out.from_numpy(np.zeros((NW, NEFC), dtype=np.float32))

_sensor_touch(0, geom_bodyid, site_type, site_bodyid, site_size, sensor_objid,
              sensor_adr, sensor_touch_adr, site_xpos_in, site_xmat_in,
              contact_pos_in, contact_frame_in, contact_dim_in, contact_geom_in,
              contact_efc_address_in, contact_worldid_in, efc_force_in,
              nacon_in, sensordata_out, NC, 1)

# RESULT: PASS pin (f32, cpu): recorded on the fixed wheel dev95+g79e660fea.
# Exact float pin: the gate wants the compiler to keep compiling this closure
# bit-identically; a legit compiler change that moves these numbers must
# re-pin here deliberately.
#
# NOTE (pre-existing, NOT this line's defect): through ray_geom's real_func
# dispatch the returned dist field comes back -1 while the normal field is
# valid (dual-wheel identical, old 0de907cd == new dev95+g79e660fea), so the
# touch path never writes and the sensordata pin is all zeros. The direct
# ray_sphere arm below pins the actual hit path. Both arms are drift
# detectors, not mujoco-semantics references.
SD_PIN = [0.0] * 24

sd = sensordata_out.to_numpy().reshape(-1)
print("PIN-SD", [float(x) for x in sd])
ok = True
for t in range(sd.shape[0]):
    if sd[t] != SD_PIN[t]:
        ok = False
        print(f"  [FAIL] sensordata[{t}]: got {sd[t]!r} want {SD_PIN[t]}")


@qd.kernel
def k_ray_sphere_direct(dist_out: qd.types.ndarray(dtype=float, ndim=1),
                        nrm_out: qd.types.ndarray(dtype=qd.math.vec3, ndim=1)):
  d, n = ray_sphere(qd.math.vec3(0.0, 0.0, 3.0), 0.25,
                    qd.math.vec3(0.0, 0.0, 1.0), qd.math.vec3(0.0, 0.0, 1.0))
  dist_out[0] = d
  nrm_out[0] = n


sph_dist = qd.ndarray(dtype=float, shape=(1,))
sph_nrm = qd.ndarray(dtype=qd.math.vec3, shape=(1,))
k_ray_sphere_direct(sph_dist, sph_nrm)

SPHERE_DIST_PIN = 1.5
SPHERE_NRM_PIN = (0.0, 0.0, -1.0)

if sph_dist.to_numpy()[0] != SPHERE_DIST_PIN:
    ok = False
    print(f"  [FAIL] sphere dist: got {sph_dist.to_numpy()[0]!r} want 1.5")
if tuple(sph_nrm.to_numpy()[0]) != tuple(SPHERE_NRM_PIN):
    ok = False
    print(f"  [FAIL] sphere nrm: got {sph_nrm.to_numpy()[0]!r} want (0.0, 0.0, -1.0)")

print("RESULT: PASS" if ok else "RESULT: FAIL")
