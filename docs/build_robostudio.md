# BUILD.md — quadrants wheel: build / install / gate, and venv roles

> Canonical location: `~/Code/quadrants @ robostudio-main`,
> `docs/build_robostudio.md` (moved 2026-10-11 out of the robostudio
> worktree's `scratch/quadrants_c0s/BUILD.md`, which is slated for
> deletion — see the two-line stub left there). The full C0-S context
> lives in the robostudio worktree at
> `$WT/scratch/quadrants_c0s/README_C0S_PATCH.md`; this file is the
> operational recipe.
>
> **Path convention**: `$WT` = the robostudio parallel-accel worktree
> root `/Users/mini0/Code/robostudio/.claude/worktrees/parallel-accel`
> (all robostudio-side paths below are `$WT`-prefixed); everything else
> is repo-side (this repository). See also
> `docs/consolidation_robostudio_main.md` (branch graph, family
> provenance, union notes) and `docs/robostudio_features.md` (features
> ledger).
>
> **2026-10-11 CONSOLIDATION**: the source of truth is this branch
> (`robostudio-main`) — the wheel carries every patch family in-tree
> (S2/S3/C0S/S-guard/u8/readback). The overlay tree
> `$WT/scratch/quadrants_src` is a frozen historical artifact
> (retirement recommended; kept as dev7 provenance). Its `dist/` remains
> the hand-off point the install flow reads.

## 1. Wheel rebuild (this repo @ robostudio-main)

```sh
cd ~/Code/quadrants                       # branch robostudio-main
LLVM_DIR=$HOME/.cache/qd-build-cache/llvm-22.1.0-202603120808 \
CMAKE_ARGS="-DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo -DQD_WITH_CUDA:BOOL=OFF" \
MAKEFLAGS=-j8 NO_PROXY="*" .venv-build/bin/python build.py wheel
# -> dist/quadrants-1.3.4.devN+g<sha>-cp312-cp312-macosx_13_0_arm64.whl
cp dist/quadrants-*.whl "$WT/scratch/quadrants_src/dist/"   # hand-off point
```

- **Build type honesty (2026-10-11 correction)**: the
  `-DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo` in CMAKE_ARGS does NOT
  stick — the pre-configured `build/cp312-*/CMakeCache.txt` holds
  `Release` and the override is ignored (sccache makes this invisible).
  Wheels from this repo are therefore Release (-O3, no debug info); the
  retired overlay's dev7 was RelWithDebInfo. The difference is debug
  info only — every gate (4-family, 6-gate union, readback perf pins)
  passes identically on both. For a true RelWithDebInfo build: delete
  `build/cp312-*` first, then rebuild.
- sccache makes rebuilds ~1 min (no full C++ recompile).
- The old overlay re-apply scripts (`apply_s2_patch.py` etc.) are
  RETIRED — the fixes are merged in-tree; do not re-apply them.
- First-time LLVM fetch is rate-limited: pull via
  `https://gh-proxy.com/https://github.com/...` style mirrors.

## 2. Install (zsh only)

```sh
zsh "$WT/scratch/quadrants_c0s/install_qds.sh"
```

- Installs the newest `$WT/scratch/quadrants_src/dist/quadrants-*.whl`
  into `$WT/scratch/poc/.venv-qds` (`--no-deps --reinstall`; never
  touches `.venv`), verifies the installed AST files carry the S2+S3
  markers (`register_loop_var` + `scope_guard` — a pre-consolidation
  wheel fails loudly here), and runs the S2/S3 post-install probes (they
  must print PASS or the script exits 1). The old post-install python
  overlay (copying overlay-tree AST files over the venv) is retired.
- zsh only (historical: the retired overlay loop used the zsh-only
  `${f:t}` modifier; the habit stuck).

## 3. Gate (after EVERY wheel rebuild)

Worktree form (four families, exit 0 required):

```sh
zsh "$WT/scratch/quadrants_c0s/post_install_gate.sh"
```

S2 loop-var guard, S3 static-if matrix, u8 DLPack type matrix (cpu ALL
PASS; metal fails EXACTLY the pinned f64 cell — qd SPIR-V has no f64,
T19), readback R2/R3 perf pins. Waits for GPU idle (up to 5 min) before
metal arms. `GATE_VERBOSE=1` cats the probe logs.

**In-repo form (preferred — no worktree dependency)**: the vendored
scripts + runner live in this repo, `tests/robostudio/`:

```sh
zsh tests/robostudio/run_robostudio_gates.sh [venv]   # 6/6 required
```

reproduces all four families PLUS the stale-field battery and the C0S
single-init battery (default venv `~/Code/quadrants/.venv-consolidated`).

## 4. Rollback

`.venv-qds`: reinstall the previous wheel from
`$WT/scratch/quadrants_src/dist/` (e.g. the dev7 artifact), or
`uv pip uninstall quadrants && uv pip install quadrants==1.3.3`, or
delete + recreate the venv. Production `.venv` is never touched by any
of the above.

## 5. venv roles (T12 residual)

| venv | quadrants | role |
|---|---|---|
| `$WT/scratch/poc/.venv` | stock 1.3.3 wheel | host stack donor: pinocchio/scipy borrows (cmeel.prefix) + the production对照 arm. Never install overlay wheels here. |
| `$WT/scratch/poc/.venv-qds` | consolidated wheel from this branch (S2/S3/C0S/S-guard/u8/readback in-tree) | metal/kernel work: probes, gates, batch_kin tests, deploy prewarm. Lacks pinocchio/scipy/pytest-timeout — borrow pinocchio via `PYTHONPATH=$WT/scratch/poc/.venv/lib/python3.12/site-packages/cmeel.prefix/lib/python3.12/site-packages` (same py 3.12). |
| `~/Code/quadrants/.venv-consolidated` | consolidated wheel + pytest/xdist (private) | union verification venv: `tests/robostudio/run_robostudio_gates.sh` runs here BEFORE any production install. |
| `$WT/scratch/poc/.venv-uvsync` | overlay dev7 + full host stack | robostudio root test suite + runtime contract + battery. Has pytest. Run the suite with `PYTHONPATH="$WT/packages/batch_kin:$WT/packages/runtime:$WT/packages/robot_kinematics"` (T14: the `planning` module is NOT importable on that branch — see §6). |
| `~/Code/quadrants/.venv-build` | none | build-only interpreter for `build.py wheel` (never system python3.14). |

Borrow pattern (documented in
`$WT/scratch/quadrants_c0s/probes/check_rnea4_s4land.py` header):
`PYTHONPATH="$WT/scratch/poc/.venv/lib/python3.12/site-packages/cmeel.prefix/lib/python3.12/site-packages"`.
`$WT/scripts/pytest_per_arch.sh` and `$WT/scratch/poc/bake_qd_cache.sh`
apply it automatically.

## 6. Test-suite conventions

- Robostudio root suite: `PYTHONPATH="$WT/packages/batch_kin:$WT/packages/runtime:$WT/packages/robot_kinematics" $WT/scratch/poc/.venv-uvsync/bin/pytest tests/ -q`
  (T14: files whose chains need the top-level `planning`/`robot_kinematics`
  modules carry `pytest.importorskip` guards and skip cleanly; the
  bare no-PYTHONPATH run still error-collects the 8 files that import
  `src.reachability` at module level — the PYTHONPATH above is the
  documented suite requirement, not optional).
- Per-arch kernel tests (T15): `test_fv_parity.py` and
  `test_selfcol_twolvl.py` parametrize the quadrants arch; ONE ARCH PER
  PROCESS (C0) — running the `batch_kin/tests/` directory bare collides
  the [metal]/[cpu] parametrizations into one process and the second
  arch raises `QuadrantsArchError` (a "1 failed" there is the trap, not
  a regression). Use:

  ```sh
  zsh "$WT/scripts/pytest_per_arch.sh" [files...]            # metal arm, then cpu
  zsh "$WT/scripts/pytest_per_arch.sh" --arch metal FILE     # one arm
  ```

  Default env `.venv-qds` + pinocchio borrow; exits non-zero if either
  arm fails; warns when the GPU is busy before the metal arm.

- Stale-field battery (S-guard): from this repo,
  `.venv-consolidated/bin/python -m pytest tests/python/test_stale_field.py
  -p xdist -n0` — the conftest's `pytest_handlecrashitem` impl requires
  xdist's hookspec, so bare pytest INTERNALERRORs (4/4 must pass).
- All six families in one command: `zsh tests/robostudio/run_robostudio_gates.sh`.

## 7. Deploy prewarm (T17)

`zsh "$WT/scratch/poc/bake_qd_cache.sh" [venv] [--arch metal,cpu] [--families all|reach,...]`
launches every production kernel family once so the disk cache
(`$HOME/.cache/quadrants`, wheel-version-keyed) holds all hot kernels;
`$WT/scratch/poc/qd_prewarm_probe.py` measures the R2 qd-metal
first-query wall cold-cache vs baked. The bake only ADDS cache entries,
never wipes.

Measured 2026-10-10 (R2 parity grid, `RobotRuntime.reachability`,
qd-metal, GPU idle; metal bake: reach 3.2s / ws_selfcol 0.8s / rneaf
186.8s / twolvl 1.8s / bspline 1254.2s / dynloss 230.7s — the last
three were genuinely absent from the cache, i.e. a fresh deploy would
have eaten ~27 min of JIT on first use):

| arm | first-query wall | ledger split |
|---|---|---|
| cold cache (fresh `$HOME`) | 45.107s | setup 17.9 (esdf JIT) + ik 27.0 (ik JIT) |
| warm cache, pre-bake | 2.838s | setup 1.76 + ik 0.92 |
| **post-bake (2 runs)** | **2.256 / 2.170s** | setup 1.46/1.35 + ik 0.66 |
| in-process steady state | 0.169s | ik 0.035 + scene 0.078 + pose_gen 0.052 |

The AST→IR JIT wall is gone post-bake (45.1s → 2.2s, 20×; kernels load
from disk). The residual ~2.0s over steady state is per-process ctor
(esdf layer loads + field upload + program init) plus first-launch
materialization — cache-independent, present in the warm-cache process
too (the T2 "runtime qd-metal 2.57s" per-process number is the same
phenomenon); deployment option (b), the resident warm process, is the
steady-state form that amortizes it. `reachable` = 2/8 identical in
every arm (kernel parity cold vs baked vs warm).

CPU-arm bake (same day): reach 47.5s / rneaf 272.5s / twolvl 2.3s /
bspline 1276.8s — also genuinely uncached before. CPU caveat: a
field-creating family (dynloss) constructed after other families'
user-kernel launches trips the C0 guard (`CpuFieldAfterLaunchError`;
metal immune) — bake such a family in a dedicated process:
`bake_qd_cache.py --arch cpu --families dynloss` (247.8s, OK).
