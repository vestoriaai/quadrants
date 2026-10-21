import quadrants as qd

qd.init(arch=qd.cpu)


@qd.kernel
def top_level(out: qd.types.ndarray(dtype=qd.i32, ndim=1)):
    n = 0
    for i in range(10):      # top-level range-for = its own offloaded task
        n += 1
    out[0] = n               # epilogue task reads n across the task boundary


@qd.kernel
def nested(out: qd.types.ndarray(dtype=qd.i32, ndim=1)):
    for _ in range(1):       # single outer task; everything below is inside it
        n = 0
        for i in range(10):
            n += 1
        out[0] = n


@qd.kernel
def serial_marked(out: qd.types.ndarray(dtype=qd.i32, ndim=1)):
    n = 0
    qd.loop_config(serialize=True)
    for i in range(10):
        n += 1
    out[0] = n


a = qd.ndarray(dtype=qd.i32, shape=[1])
top_level(a); print(f"top-level for, read after  -> {a.to_numpy()[0]} (want 10)")
nested(a);    print(f"nested in range(1) task    -> {a.to_numpy()[0]} (want 10)")
try:
    serial_marked(a); print(f"loop_config(serialize)     -> {a.to_numpy()[0]} (want 10)")
except Exception as e:
    print(f"loop_config(serialize)     EXC {str(e).splitlines()[0][:80]}")
