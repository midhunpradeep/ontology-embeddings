"""Exact closed, positive-width axis-aligned BOX relation diagnostics (float64 arithmetic)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def validate_box(
    lower: Any, upper: Any, *, domain: str, epsilon: float
) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    if lo.ndim != 1 or lo.shape != hi.shape or lo.size < 1:
        raise ValueError("endpoints must be matching nonempty vectors")
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi <= lo):
        raise ValueError("box must have finite positive-width intervals")
    if np.any(hi - lo < epsilon):
        raise ValueError("box violates checkpoint width floor")
    if domain == "bounded" and (np.any(lo < -1) or np.any(hi > 1)):
        raise ValueError("bounded endpoints outside [-1,1]")
    if domain not in {"bounded", "unbounded"}:
        raise ValueError(f"unsupported domain {domain}")
    return lo, hi


def _log_volume(widths: np.ndarray) -> float:
    if np.any(widths <= 0):
        return -math.inf
    return float(np.log(widths).sum())


def _ratio(numer: np.ndarray, denom: np.ndarray) -> float | None:
    ln, ld = _log_volume(numer), _log_volume(denom)
    if ln == -math.inf:
        return 0.0
    delta = ln - ld
    if delta > math.log(np.finfo(float).max):
        return None
    return float(math.exp(delta))


def directed(
    a_lo: np.ndarray, a_hi: np.ndarray, b_lo: np.ndarray, b_hi: np.ndarray
) -> dict[str, Any]:
    violation = np.maximum(0.0, np.maximum(b_lo - a_lo, a_hi - b_hi))
    contained = bool(np.all(violation == 0))
    il, iu = np.maximum(a_lo, b_lo), np.minimum(a_hi, b_hi)
    nonempty = bool(np.all(il <= iu))
    widths = np.maximum(0.0, iu - il)
    fraction = _ratio(widths, a_hi - a_lo) if nonempty else 0.0
    witness = None
    if not contained:
        axis = int(np.flatnonzero(violation > 0)[0])
        point = a_lo.copy() if a_lo[axis] < b_lo[axis] else a_hi.copy()
        witness = {
            "type": "containment_counterexample",
            "coordinates": point.tolist(),
            "axis": axis,
        }
    return {
        "contained": contained,
        "violation_vector": violation.tolist(),
        "max_violation": float(violation.max()),
        "mean_violation": float(violation.mean()),
        "violating_dimensions": int(np.count_nonzero(violation)),
        "relative_containment_volume": fraction,
        "relative_uncovered_volume": None if fraction is None else 1 - fraction,
        "witness": witness,
    }


def unordered(
    a_lo: np.ndarray, a_hi: np.ndarray, b_lo: np.ndarray, b_hi: np.ndarray
) -> dict[str, Any]:
    gap = np.maximum(a_lo, b_lo) - np.minimum(a_hi, b_hi)
    il, iu = np.maximum(a_lo, b_lo), np.minimum(a_hi, b_hi)
    nonempty = bool(np.all(il <= iu))
    widths = np.maximum(0.0, iu - il)
    av, bv = a_hi - a_lo, b_hi - b_lo
    lv, la, lb = _log_volume(widths), _log_volume(av), _log_volume(bv)
    volume = (
        0.0
        if lv == -math.inf
        else (float(math.exp(lv)) if lv < math.log(np.finfo(float).max) else None)
    )
    scale = max(la, lb)
    union_scaled = (
        math.exp(la - scale)
        + math.exp(lb - scale)
        - (0.0 if lv == -math.inf else math.exp(lv - scale))
    )
    union_log = scale + math.log(union_scaled)
    iou = (
        0.0
        if lv == -math.inf
        else (float(math.exp(lv - union_log)) if lv - union_log > -746 else 0.0)
    )
    return {
        "signed_separation": gap.tolist(),
        "intersection_status": "empty"
        if not nonempty
        else "positive_volume"
        if np.all(widths > 0)
        else "lower_dimensional",
        "intersection_lower": il.tolist() if nonempty else None,
        "intersection_upper": iu.tolist() if nonempty else None,
        "intersection_dimensionality": int(np.count_nonzero(widths > 0)) if nonempty else None,
        "intersection_volume": volume,
        "fraction_of_a": _ratio(widths, av) if nonempty else 0.0,
        "fraction_of_b": _ratio(widths, bv) if nonempty else 0.0,
        "intersection_over_union": iou,
    }
