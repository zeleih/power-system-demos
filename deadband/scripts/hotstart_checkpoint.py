#!/usr/bin/env python3
"""
Helpers for disk-based hot-start checkpoints.

The checkpoint format intentionally separates:

- parameter-independent dispatch targets (`dispatch JSON`)
- parameter-dependent dynamic checkpoints (`system.pkl`, AGC state, runtime context)

This makes it possible to regenerate a hot-start chain for one parameter set
without recomputing the OPF dispatch library.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from andes.utils.snapshot import load_ss, save_ss

import run_dispatch_tds as rdt


RUNTIME_CTX_FIELDS = (
    "pq_idx",
    "sap0",
    "saq0",
    "stg_w2t",
    "stg_pv",
    "p0_w2t",
    "p0_pv",
    "pvd1_w2t",
    "pvd1_pv",
)


def _canonical_path(path: Path) -> str:
    return str(path.resolve())


def _normalized_prefixes(values: Iterable[str]) -> list[str]:
    return [str(item) for item in values]


def build_param_signature(
    *,
    kp: float,
    ki: float,
    agc_interval: int,
    init_mode: str,
    dispatch_interval: int,
    curve_file: Path,
    dyn_case: Path,
    stable_dyn_case: Path,
    wind_prefixes: Iterable[str],
    solar_prefixes: Iterable[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the parameter signature for a checkpoint family.

    Any field that can change the dynamic end state belongs here.
    """

    signature: dict[str, Any] = {
        "format": "deadband_hotstart_v1",
        "kp": float(kp),
        "ki": float(ki),
        "agc_interval": int(agc_interval),
        "init_mode": str(init_mode),
        "dispatch_interval": int(dispatch_interval),
        "curve_file": _canonical_path(curve_file),
        "dyn_case": _canonical_path(dyn_case),
        "stable_dyn_case": _canonical_path(stable_dyn_case),
        "wind_prefixes": _normalized_prefixes(wind_prefixes),
        "solar_prefixes": _normalized_prefixes(solar_prefixes),
    }
    if extra:
        signature["extra"] = extra

    return signature


def param_hash(signature: dict[str, Any], length: int = 12) -> str:
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:length]


def checkpoint_family_dir(root: Path, signature: dict[str, Any]) -> Path:
    return root / param_hash(signature)


def checkpoint_dir(root: Path, signature: dict[str, Any], dispatch_label: str) -> Path:
    return checkpoint_family_dir(root, signature) / f"end_{dispatch_label}"


def ensure_family_manifest(root: Path, signature: dict[str, Any]) -> Path:
    family_dir = checkpoint_family_dir(root, signature)
    family_dir.mkdir(parents=True, exist_ok=True)
    path = family_dir / "param_signature.json"
    path.write_text(json.dumps(signature, indent=2))
    return path


def minimal_runtime_context(ctx: dict[str, object]) -> dict[str, np.ndarray]:
    """
    Extract the pieces of runtime context needed to continue updates after reload.

    The dynamic snapshot already preserves the live ANDES state. The arrays saved
    here are the immutable scaling references used by the external replay loop.
    """

    out: dict[str, np.ndarray] = {}
    for field in RUNTIME_CTX_FIELDS:
        if field not in ctx:
            raise KeyError(f"Runtime context missing required field '{field}'")
        out[field] = np.asarray(ctx[field])
    return out


def build_runtime_context(
    *,
    sa,
    curve,
    stored_ctx: dict[str, np.ndarray],
) -> dict[str, object]:
    """
    Rebuild the runtime context around a restored ANDES snapshot.
    """

    link = rdt.build_andes_link(sa)
    pext_max = 999.0 * np.ones(sa.DG.n)
    if hasattr(sa, "ESD1") and sa.ESD1.n:
        ess_uid = sa.DG.idx2uid(sa.ESD1.idx.v)
        pext_max[ess_uid] = 999.0

    ctx: dict[str, object] = {
        "curve": curve,
        "link": link,
        "pext_max": pext_max,
    }
    for field, value in stored_ctx.items():
        arr = np.asarray(value)
        if field in ("sap0", "saq0", "p0_w2t", "p0_pv"):
            ctx[field] = arr.astype(float, copy=False)
        else:
            # ANDES idx arrays are often string-like labels such as ``PQ_1`` or
            # ``GENROU_3`` rather than numeric IDs. Preserve their original
            # dtype so the restored snapshot can address devices correctly.
            ctx[field] = arr
    return ctx


def trim_snapshot_timeseries(sa) -> None:
    """
    Remove in-memory time-series history before serializing a checkpoint.

    The live DAE state is preserved, but prior sampled trajectories are not.
    """

    if hasattr(sa, "dae") and hasattr(sa.dae, "ts"):
        sa.dae.ts.reset()


def rehydrate_loaded_snapshot(sa) -> None:
    """
    Refresh model-side views and residual-dependent services after ``load_ss``.

    Some device-level cached values (for example ``PVD1.vp`` / ``ESD1.vp``)
    are derived from DAE arrays and need one explicit refresh after
    deserialization before resumed integration can continue reliably.
    """

    sa.vars_to_models()
    sa.TDS.fg_update(models=sa.exist.pflow_tds)

    # Serialized sparse-solver caches are not guaranteed to remain valid across
    # a disk round-trip. Force the resumed simulation to rebuild factorizations.
    if hasattr(sa.TDS, "solver") and hasattr(sa.TDS.solver, "clear"):
        sa.TDS.solver.clear()
        worker = getattr(sa.TDS.solver, "worker", None)
        if worker is not None and hasattr(worker, "factorize"):
            worker.factorize = True


def save_checkpoint(
    *,
    checkpoint_dir: Path,
    sa,
    ctx: dict[str, object],
    ace_integral: float,
    ace_raw: float,
    agc_aw_state: dict[str, int] | None = None,
    manifest: dict[str, Any],
) -> dict[str, Path]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    system_path = checkpoint_dir / "system.pkl"
    runtime_path = checkpoint_dir / "runtime_context.npz"
    agc_path = checkpoint_dir / "agc_state.json"
    manifest_path = checkpoint_dir / "manifest.json"

    trim_snapshot_timeseries(sa)
    save_ss(system_path, sa)
    np.savez(runtime_path, **minimal_runtime_context(ctx))
    payload = {
        "ace_integral": float(ace_integral),
        "ace_raw": float(ace_raw),
    }
    if agc_aw_state is not None:
        payload["freeze_active"] = int(agc_aw_state.get("freeze_active", 0))
        payload["freeze_on_streak"] = int(agc_aw_state.get("freeze_on_streak", 0))
        payload["freeze_off_streak"] = int(agc_aw_state.get("freeze_off_streak", 0))
        payload["freeze_dir"] = int(agc_aw_state.get("freeze_dir", 0))
    agc_path.write_text(json.dumps(payload, indent=2))
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return {
        "system_path": system_path,
        "runtime_context_path": runtime_path,
        "agc_state_path": agc_path,
        "manifest_path": manifest_path,
    }


def load_checkpoint(checkpoint_dir: Path) -> tuple[object, dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    system_path = checkpoint_dir / "system.pkl"
    runtime_path = checkpoint_dir / "runtime_context.npz"
    agc_path = checkpoint_dir / "agc_state.json"
    manifest_path = checkpoint_dir / "manifest.json"

    sa = load_ss(system_path)
    rehydrate_loaded_snapshot(sa)
    with np.load(runtime_path, allow_pickle=False) as data:
        runtime_ctx = {key: data[key] for key in data.files}
    agc_state = json.loads(agc_path.read_text())
    manifest = json.loads(manifest_path.read_text())

    return sa, runtime_ctx, agc_state, manifest


def validate_signature(expected: dict[str, Any], observed: dict[str, Any]) -> None:
    if expected != observed:
        raise RuntimeError(
            "Checkpoint parameter signature mismatch.\n"
            f"expected={json.dumps(expected, sort_keys=True)}\n"
            f"observed={json.dumps(observed, sort_keys=True)}"
        )
