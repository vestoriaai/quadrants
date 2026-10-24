"""A62 diagnosis v2: y-sequence shape by loop form (static-unrolled vs runtime).

Both forms compile the same 8-iteration sequential atomic chain on a kernel-local
scalar. If the returned old values differ between forms, the bug is in the IR
transform of one specific lowering path.
"""

import pathlib
import tempfile

import quadrants as qd

qd.init(arch=qd.cpu, debug=True)

COMMON = (
    "import quadrants as qd\n"
    "x = qd.field(qd.i32)\n"
    "y = qd.field(qd.i32)\n"
    "c = qd.field(qd.i32)\n"
    "qd.root.dense(qd.i, 8).place(x, y)\n"
    "qd.root.place(c)\n"
)


def build(form):
    if form == "static":
        body = "\n".join(
            f"    x[{i}] = qd.atomic_add(c[None], 42)\n"
            f"    y[{i}] = qd.atomic_add(ck, 42)"
            for i in range(8)
        )
        src = COMMON + f"""

@qd.kernel
def k():
    ck = 0
{body}
"""
    else:
        src = COMMON + """

@qd.kernel
def k():
    ck = 0
    for i in range(8):
        x[i] = qd.atomic_add(c[None], 42)
        y[i] = qd.atomic_add(ck, 42)
"""
    d = tempfile.mkdtemp()
    p = pathlib.Path(d) / f"k_{form}.py"
    p.write_text(src)
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"m_{form}", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


for form in ("static", "runtime"):
    mod = build(form)
    mod.k()
    y = list(mod.y.to_numpy())
    x_ok = sorted(mod.x.to_numpy()) == [i * 42 for i in range(8)]
    print(f"form={form:8s}: y={y} x_ok={x_ok}", flush=True)
