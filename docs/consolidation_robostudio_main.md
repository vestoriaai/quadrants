# robostudio-main — the consolidated quadrants line (2026-10-11)

This branch is the **single source of truth** for every quadrants wheel
robostudio deploys. It replaces the overlay tree
(`robostudio/.claude/worktrees/parallel-accel/scratch/quadrants_src`),
which is now a frozen historical artifact (see "Retirement" below).

Companions: `docs/build_robostudio.md` (the operational
build/install/gate recipe — moved out of the robostudio worktree's
`scratch/quadrants_c0s/BUILD.md`, now a stub) and
`docs/robostudio_features.md` (the features ledger — every feature
commit merged to this branch must update it in the same merge).

## What the wheel carries

`1.3.4.devN+g<sha>` wheels built from this branch contain ALL patch
families in-tree — no post-install python overlay:

| family | provenance | content |
|---|---|---|
| u8 DLPack | feat/u8-dlpack@4b2e3c77 | u8/i8/u16/i16/u32/u64 ndarray export |
| readback R2/R3 | feat/readback@4fe1bbe2+7aaaa7d7 | metal direct-map readback + persistent ext-arr mirror pool |
| S2 | overlay patches/s2_fix.patch | loop-var assignment guard (`register_loop_var`) |
| S3 | overlay patches/s3_fix.patch | static-if branch-name join (`scope_guard`) |
| C0S | overlay patches/c0s_fix.patch | cache-epoch: artifacts never re-served across SNodeTree lifecycles + dead-runtime capture walk |
| S-guard | overnight/inv-cache-keys@80445172 | stale-field generation stamp + `Field._check_not_stale` |

Fidelity was verified before building: every ported file is byte-identical
to the overlay working tree it came from (`impl.py` is a strict superset:
overlay content + S-guard additions).

## Branch shape

    main (upstream) ──┬── feat/u8-dlpack ──┬── feat/readback ──┬── overnight/inv-cache-keys (investigation, stays separate)
                      │                    │                   │
                      │            robostudio-main             │ (S-guard cherry-picked,
                      │            = feat/readback             │  NOT merged: keeps
                      │              + merge port/s2-loopvar-guard        investigation
                      │              + merge port/s3-static-if-join       probes/docs out)
                      │              + merge port/c0s-cache-epoch
                      │              + merge port/sguard-stale-field (cherry-pick of 80445172)
                      │              + merge port/c0s-cache-epoch (union integration: walk defers
                      │                to S-guard generation stamp)

Every port landed as its own feature branch merged `--no-ff` into
`robostudio-main`; the overlay's `patches/*.patch` files are the ports'
verbatim sources. This branch is local-only (no upstream PR — user
ruling, tech_debt_todo T20).

## Union integration notes

1. **C0S capture walk vs S-guard** (`60003393`): the C0S coarse walk at
   `Kernel` materialize fired before the S-guard's precise
   `impl.subscript` check and rejected kernels that merely carry a stale
   field without reading it (`scan_nd`). The walk now defers captures
   carrying the generation stamp to the choke point and keeps refusing
   captures that lack it. `tests/python/test_stale_field.py` 4/4.
2. **Multi-init battery semantics changed** (supersedes the C0S
   battery's old reading): `scratch/quadrants_src/probes/probe_multi_init.py`
   arms `a2b` / `ab_a` / `jit_heavy` RAISE on the union wheel — by
   design. They relaunch families whose fields predate the current
   `qd.init()`; overnight Inv2 established that surviving-field use
   after `Program::finalize` is use-after-free (the old PASS was
   memory-luck on the reused pool, not safety). The S-guard refuses
   loudly instead. The contract-lift form passes 4/4:
   `probe_multi_init.py --single-init`. New battery of record:
   `pytest tests/python/test_stale_field.py` (run under
   `-p xdist -n0`; the conftest's `pytest_handlecrashitem` impl needs
   xdist's hookspec).

## Build / install / gate

    cd ~/Code/quadrants
    LLVM_DIR=$HOME/.cache/qd-build-cache/llvm-22.1.0-202603120808 \
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo -DQD_WITH_CUDA:BOOL=OFF" \
    MAKEFLAGS=-j8 NO_PROXY="*" .venv-build/bin/python build.py wheel
    # -> dist/quadrants-1.3.4.devN+g<sha>-cp312-cp312-macosx_13_0_arm64.whl

RelWithDebInfo matches the deployed dev7 lineage (its `.so` carries
`.debug_info`); sccache makes rebuilds fast. Then copy the wheel into
`scratch/quadrants_src/dist/` (the robostudio install flow picks the
newest wheel from there) and:

    zsh scratch/quadrants_c0s/install_qds.sh        # install + marker verify + S2/S3 probes
    zsh scratch/quadrants_c0s/post_install_gate.sh  # 4/4 must PASS

## Union gate — one command

`zsh tests/robostudio/run_robostudio_gates.sh [venv]` runs every family
in-repo (default venv `.venv-consolidated`): S2 probe, S3 form matrix,
u8 DLPack cpu+metal (with the pinned f64 expected-failure), readback
R2/R3 perf pins, the stale-field pytest battery, and the C0S
single-init battery. The vendored scripts under `tests/robostudio/` are
verbatim copies of the probes/gates that validated the consolidation
(originals: robostudio worktree `scratch/quadrants_c0s/probes/`,
`scratch/quadrants_src/probes/probe_multi_init.py`, and simtest
qd-render-poc `results/u8_dlpack_gate.py`); the readback gate was
extracted verbatim from post_install_gate.sh's gate-4 heredoc.

## Retirement (documented recommendation; not executed)

`scratch/quadrants_src` is superseded: its unique content (S2/S3/C0S
patches, u8/readback ports, probes) is now in this branch. Recommendation:
keep it frozen as the dev7 provenance record, stop building from it, and
delete it in a later cleanup round once a few consolidated wheels have
shipped. Its `.venv-build` remains referenced by nothing after the move
(~/Code/quadrants has its own). Do not re-apply its overlay scripts —
`install_qds.sh` no longer overlays (retired 2026-10-11; it verifies the
installed files carry the S2/S3 markers instead, so a pre-consolidation
wheel fails loudly instead of being silently patched).
