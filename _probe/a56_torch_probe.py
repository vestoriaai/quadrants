"""A56 probe driver: run the instrumented child twice per torch condition and diff logs.

Usage:
    python a56_torch_probe.py <venv_python> <workdir>

Produces <workdir>/torch_p1.json, torch_p2.json, notorch_p1.json, notorch_p2.json
and prints the pairwise diffs of key-pipeline components (torch p1 vs p2 is the
A56 signature; notorch pair is the control).
"""

import json
import pathlib
import shutil
import subprocess
import sys

VENV_PY = sys.argv[1]
WORK = pathlib.Path(sys.argv[2])
CHILD = pathlib.Path(__file__).parent / "a56_torch_probe_child.py"


def run_child(tag, cache_dir, with_torch):
    log = WORK / f"{tag}.json"
    cmd = [VENV_PY, str(CHILD), str(cache_dir), str(log)] + (["torch"] if with_torch else [])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(f"--- {tag}: rc={proc.returncode}")
    print(proc.stdout.strip())
    if proc.returncode != 0:
        print(proc.stderr[-3000:])
    return log


def load_log(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def events(log, name):
    return [e for e in log if e["event"] == name]


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    cache_torch = WORK / "cache_torch"
    cache_notorch = WORK / "cache_notorch"
    for d in (cache_torch, cache_notorch):
        shutil.rmtree(d, ignore_errors=True)

    logs = {}
    for tag, cache, torch_flag in [
        ("torch_p1", cache_torch, True),
        ("torch_p2", cache_torch, True),
        ("notorch_p1", cache_notorch, False),
        ("notorch_p2", cache_notorch, False),
    ]:
        logs[tag] = load_log(run_child(tag, cache, torch_flag))

    def summarize(tag):
        out = {}
        for e in logs[tag]:
            name = e.pop("event")
            out.setdefault(name, []).append(e)
        return out

    def diff_events(tag_a, tag_b, name):
        ea, eb = events(logs[tag_a], name), events(logs[tag_b], name)
        la = [json.dumps(e, sort_keys=True) for e in ea]
        lb = [json.dumps(e, sort_keys=True) for e in eb]
        if la == lb:
            print(f"[same] {name}: {len(la)} events identical")
            return
        print(f"[DIFF] {name}:")
        for i in range(max(len(la), len(lb))):
            sa = la[i] if i < len(la) else "<missing>"
            sb = lb[i] if i < len(lb) else "<missing>"
            if sa != sb:
                print(f"  {tag_a}[{i}]: {sa}")
                print(f"  {tag_b}[{i}]: {sb}")

    for a, b in [("torch_p1", "torch_p2"), ("notorch_p1", "notorch_p2"), ("torch_p1", "notorch_p1")]:
        print(f"\n===== {a} vs {b} =====")
        for name in [
            "config_hash",
            "device_caps_hash",
            "kernel_hash",
            "l1_key",
            "l1_load",
            "l1_store",
            "args_hash",
            "narrow_hash",
            "l2_key",
            "l2_load",
            "l2_store",
            "observations",
        ]:
            diff_events(a, b, name)


if __name__ == "__main__":
    main()
