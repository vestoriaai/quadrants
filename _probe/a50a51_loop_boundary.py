import quadrants as qd

qd.init(arch=qd.cpu)

out = qd.field(dtype=qd.i32, shape=())
cnt = qd.field(dtype=qd.i32, shape=())


@qd.kernel
def const_trip():
    n = 0
    for i in range(10):
        n += 1
    out[None] = n


@qd.kernel
def dyn_trip(n0: qd.i32):
    n = 0
    for i in range(n0):
        n += 1
    out[None] = n


@qd.kernel
def const_trip_use_i():
    n = 0
    for i in range(10):
        n += i
    out[None] = n


@qd.kernel
def const_trip_store_inside():
    n = 0
    for i in range(10):
        n += 1
        out[None] = n


@qd.kernel
def field_trip():
    n = 0
    for i in range(cnt[None]):
        n += 1
    out[None] = n


const_trip(); print(f"const range(10)         -> {out[None]}  (want 10)")
dyn_trip(10); print(f"dynamic range(n0=10)    -> {out[None]}  (want 10)")
const_trip_use_i(); print(f"const range(10) n+=i    -> {out[None]}  (want 45)")
const_trip_store_inside(); print(f"const, store inside     -> {out[None]}  (want 10)")
cnt[None] = 10; field_trip(); print(f"range(field=10)         -> {out[None]}  (want 10)")
