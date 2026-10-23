"""A56 probe child: run the fastcache checkpoint kernel once, dumping key composition.

Usage:
    python a56_torch_probe_child.py <cache_dir> <log_file> [torch]

Replicates tests/python/test_checkpoint.py::_fastcache_checkpoint_child with the
minimal kernel shape (graph=True, checkpoints=True, fastcache=True,
yield_on=self.flag) and monkeypatches the fastcache key pipeline to log every
component (config list, kernel hash, L1 key, pruning paths, narrow args hash,
L2 key, load/store) to <log_file> as JSON lines. Driver script diffs the logs.
"""

import json
import sys
import os

CACHE_DIR = sys.argv[1]
LOG_FILE = sys.argv[2]
IMPORT_TORCH = len(sys.argv) > 3 and sys.argv[3] == "torch"

if IMPORT_TORCH:
    import torch  # noqa: F401

import numpy as np  # noqa: E402
import quadrants as qd  # noqa: E402
from quadrants.lang._fast_caching import (  # noqa: E402
    args_hasher,
    config_hasher,
    function_hasher,
    src_hasher,
)

LOG = []


def _log(event, **data):
    LOG.append({"event": event, **data})


# ---- instrument the key pipeline (module-attr wrapping; src_hasher calls via
# module references so patching the modules is effective) ----

_orig_hash_compile_config = config_hasher.hash_compile_config
_orig_hash_device_caps = config_hasher.hash_device_caps
_orig_hash_kernel = function_hasher.hash_kernel
_orig_make_source_config_key = src_hasher.make_source_config_key
_orig_load_pruning_info = src_hasher.load_pruning_info
_orig_hash_args = args_hasher.hash_args
_orig_compute_narrow = src_hasher.compute_narrow_args_hash
_orig_make_full_key = src_hasher.make_full_cache_key
_orig_src_load = src_hasher.load
_orig_src_store = src_hasher.store
_orig_store_pruning = src_hasher.store_pruning_info


def _hash_compile_config():
    from quadrants.lang import impl

    config = impl.get_runtime().prog.config()
    exclude_prefixes = ["_", "offline_cache", "print_", "verbose_"]
    exclude_keys = {"debug_dump_path", "cuda_stack_limit", "external_metal_command_queue"}
    config_l = []
    for k in dir(config):
        if k in exclude_keys:
            continue
        skip = False
        for prefix in exclude_prefixes:
            if k.startswith(prefix) or k in [""]:
                skip = True
        if skip:
            continue
        config_l.append(f"{k}={getattr(config, k)}")
    h = _orig_hash_compile_config()
    _log("config_hash", items=config_l, hash=h)
    return h


def _hash_device_caps():
    h = _orig_hash_device_caps()
    _log("device_caps_hash", hash=h)
    return h


def _hash_kernel(source_info):
    h = _orig_hash_kernel(source_info)
    _log("kernel_hash", filepath=source_info.filepath, lineno=source_info.start_lineno, hash=h)
    return h


def _make_source_config_key(kernel_source_info):
    key = _orig_make_source_config_key(kernel_source_info)
    _log("l1_key", key=key)
    return key


def _load_pruning_info(source_config_key):
    paths, levels = _orig_load_pruning_info(source_config_key)
    _log(
        "l1_load",
        key=source_config_key,
        hit=paths is not None,
        paths=sorted(paths) if paths else None,
        levels=levels,
    )
    return paths, levels


def _hash_args(raise_on_templated_floats, args, arg_metas, pruning_paths=None):
    h = _orig_hash_args(raise_on_templated_floats, args, arg_metas, pruning_paths=pruning_paths)
    _log(
        "args_hash",
        raise_on_templated_floats=raise_on_templated_floats,
        pruning_paths=sorted(pruning_paths) if pruning_paths is not None else None,
        hash=str(h),
    )
    return h


def _compute_narrow(raise_on_templated_floats, kernel_source_info, args, arg_metas, pruning_paths):
    h = _orig_compute_narrow(raise_on_templated_floats, kernel_source_info, args, arg_metas, pruning_paths)
    _log("narrow_hash", hash=h)
    return h


def _make_full_key(source_config_key, narrow_args_hash):
    key = _orig_make_full_key(source_config_key, narrow_args_hash)
    _log("l2_key", key=key)
    return key


def _src_load(cache_key):
    raw = src_hasher._try_load(cache_key)
    val = _orig_src_load(cache_key)
    _log(
        "l2_load",
        key=cache_key,
        entry_on_disk=raw is not None,
        validated=val is not None,
        frontend_cache_key=raw.frontend_cache_key if raw is not None else None,
    )
    return val


def _src_store(frontend_cache_key, fast_cache_key, *a, **kw):
    _log("l2_store", fast_cache_key=fast_cache_key, frontend_cache_key=frontend_cache_key)
    return _orig_src_store(frontend_cache_key, fast_cache_key, *a, **kw)


def _store_pruning(source_config_key, function_source_infos, used, graph_do_while_levels=None):
    _log(
        "l1_store",
        key=source_config_key,
        paths=sorted(used),
        levels=graph_do_while_levels,
    )
    return _orig_store_pruning(source_config_key, function_source_infos, used, graph_do_while_levels)


config_hasher.hash_compile_config = _hash_compile_config
config_hasher.hash_device_caps = _hash_device_caps
function_hasher.hash_kernel = _hash_kernel
src_hasher.make_source_config_key = _make_source_config_key
src_hasher.load_pruning_info = _load_pruning_info
args_hasher.hash_args = _hash_args
src_hasher.compute_narrow_args_hash = _compute_narrow
src_hasher.make_full_cache_key = _make_full_key
src_hasher.load = _src_load
src_hasher.store = _src_store
src_hasher.store_pruning_info = _store_pruning


# ---- kernel under test (mirrors the test file) ----


@qd.data_oriented
class _FastcacheYieldOnSelfCheckpoint:
    def __init__(self, n):
        self.x = qd.ndarray(qd.i32, shape=(n,))
        self.flag = qd.ndarray(qd.i32, shape=())

    @qd.kernel(graph=True, checkpoints=True, fastcache=True)
    def step(self):
        with qd.checkpoint(0, yield_on=self.flag):
            for i in range(self.x.shape[0]):
                self.x[i] = self.x[i] + 1


def main():
    qd.init(
        arch=qd.arm64,
        offline_cache=True,
        offline_cache_file_path=CACHE_DIR,
        src_ll_cache=True,
    )
    n = 8
    sim = _FastcacheYieldOnSelfCheckpoint(n)
    sim.x.from_numpy(np.zeros(n, dtype=np.int32))
    sim.flag.from_numpy(np.array(0, dtype=np.int32))
    sim.step()
    x = sim.x.to_numpy()
    assert (x == 1).all(), f"kernel result wrong: {x}"

    primal = type(sim).step._primal
    obs = primal.src_ll_cache_observations
    _log(
        "observations",
        cache_key_generated=obs.cache_key_generated,
        cache_validated=obs.cache_validated,
        cache_loaded=obs.cache_loaded,
        cache_stored=obs.cache_stored,
    )
    with open(LOG_FILE, "w") as f:
        for entry in LOG:
            f.write(json.dumps(entry) + "\n")
    print(f"PROBE_OK loaded={obs.cache_loaded}")


if __name__ == "__main__":
    main()
