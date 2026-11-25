"""Trainable axis-aligned BOX geometry and class-agnostic mathematical kernels.

The supported numerical operating envelope is finite float32 inputs and parameters,
with decoded coordinates representable in float32. No clipping or repair is performed.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

Domain = Literal["bounded", "unbounded"]


def _positive_epsilon(epsilon: float, domain: Domain) -> None:
    if not math.isfinite(epsilon) or epsilon <= 0 or (domain == "bounded" and epsilon >= 2):
        raise ValueError("epsilon must be finite and positive (and less than 2 when bounded)")


def _check_raw(raw_a: torch.Tensor, raw_b: torch.Tensor, epsilon: float, domain: Domain) -> None:
    _positive_epsilon(epsilon, domain)
    if not isinstance(raw_a, torch.Tensor) or not isinstance(raw_b, torch.Tensor):
        raise TypeError("raw geometry must be tensors")
    if raw_a.ndim != 3 or raw_a.shape != raw_b.shape or any(n < 1 for n in raw_a.shape):
        raise ValueError("raw a and b must share nonempty [classes, boxes, dimensions] shape")
    if raw_a.dtype != torch.float32 or raw_b.dtype != torch.float32:
        raise TypeError("raw geometry tensors must be float32")
    if raw_a.device != raw_b.device:
        raise ValueError("raw geometry tensors must share a device")
    if not torch.isfinite(raw_a).all() or not torch.isfinite(raw_b).all():
        raise ValueError("raw geometry must be finite")


def decode_geometry(
    raw_a: torch.Tensor, raw_b: torch.Tensor, *, domain: Domain, epsilon: float = 1e-4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode width-first F12 raw parameters to float32 (lower, upper) endpoints."""
    _check_raw(raw_a, raw_b, epsilon, domain)
    if domain == "bounded":
        half = (epsilon + (2.0 - epsilon) * torch.sigmoid(raw_a)) / 2.0
        center = (1.0 - half) * torch.tanh(raw_b)
    elif domain == "unbounded":
        half = epsilon / 2.0 + F.softplus(raw_a)
        center = raw_b
    else:
        raise ValueError(f"unknown domain {domain!r}")
    lower, upper = center - half, center + half
    validate_endpoints(lower, upper, domain=domain, epsilon=epsilon)
    return lower, upper


def validate_endpoints(
    lower: torch.Tensor, upper: torch.Tensor, *, domain: Domain, epsilon: float = 1e-4
) -> None:
    """Reject invalid stored endpoints, checking floor comparisons in float64."""
    _positive_epsilon(epsilon, domain)
    if not isinstance(lower, torch.Tensor) or not isinstance(upper, torch.Tensor):
        raise TypeError("endpoints must be tensors")
    if lower.ndim != 3 or lower.shape != upper.shape or any(n < 1 for n in lower.shape):
        raise ValueError("endpoints must share nonempty [classes, boxes, dimensions] shape")
    if lower.dtype != torch.float32 or upper.dtype != torch.float32:
        raise TypeError("endpoints must be float32")
    if lower.device != upper.device:
        raise ValueError("endpoints must share a device")
    if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
        raise ValueError("endpoints must be finite")
    lo, hi = lower.to(torch.float64), upper.to(torch.float64)
    if torch.any(hi <= lo) or torch.any((hi - lo) < epsilon):
        raise ValueError("stored endpoints must be ordered and meet the full-width floor")
    if domain == "bounded" and (torch.any(lo < -1.0) or torch.any(hi > 1.0)):
        raise ValueError("bounded endpoints must lie in [-1, 1]")
    if domain not in ("bounded", "unbounded"):
        raise ValueError(f"unknown domain {domain!r}")


def inverse_geometry(
    lower: torch.Tensor, upper: torch.Tensor, *, domain: Domain, epsilon: float = 1e-4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert valid physical float32 endpoints into F12 raw width/center parameters."""
    validate_endpoints(lower, upper, domain=domain, epsilon=epsilon)
    center = (lower + upper) / 2.0
    half = (upper - lower) / 2.0
    if domain == "bounded":
        ratio = (2.0 * half - epsilon) / (2.0 - epsilon)
        center_ratio = center / (1.0 - half)
        if torch.any(ratio <= 0) or torch.any(ratio >= 1) or torch.any(center_ratio.abs() >= 1):
            raise ValueError(
                "bounded endpoints have no finite inverse in the float32 operating envelope"
            )
        raw_a = torch.logit(ratio)
        raw_b = torch.atanh(center_ratio)
    elif domain == "unbounded":
        gap = half - epsilon / 2.0
        if torch.any(gap <= 0):
            raise ValueError(
                "unbounded endpoint width has no finite inverse above the epsilon floor"
            )
        raw_a = gap + torch.log(-torch.expm1(-gap))
        raw_b = center
    else:
        raise ValueError(f"unknown domain {domain!r}")
    if not torch.isfinite(raw_a).all() or not torch.isfinite(raw_b).all():
        raise ValueError("inverse geometry is outside the finite float32 operating envelope")
    return raw_a, raw_b


def initialize_geometry(
    classes: int,
    boxes: int,
    dimensions: int,
    *,
    domain: Domain,
    seed: int,
    epsilon: float = 1e-4,
    volume_fraction: float | None = None,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Seeded dimension-aware feasible initialization, returned as raw (a,b)."""
    _positive_epsilon(epsilon, domain)
    if any(
        isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (classes, boxes, dimensions)
    ):
        raise ValueError("classes, boxes, and dimensions must be positive integers")
    if volume_fraction is None:
        g = torch.Generator(device="cpu").manual_seed(seed)
        fraction = float(torch.empty(()).uniform_(0.05, 0.25, generator=g))
    else:
        fraction = float(volume_fraction)
        if not math.isfinite(fraction) or not 0.05 <= fraction <= 0.25:
            raise ValueError("volume_fraction must lie in [0.05, 0.25]")
        g = torch.Generator(device="cpu").manual_seed(seed)
    half = fraction ** (1.0 / dimensions)
    if domain == "bounded" and not (epsilon / 2 < half < 1):
        raise ValueError("dimension/floor combination cannot represent initialization")
    shape = (classes, boxes, dimensions)
    halves = torch.full(shape, half, dtype=torch.float32)
    if domain == "bounded":
        centers = (torch.rand(shape, generator=g) * 2 - 1) * (1 - half)
    elif domain == "unbounded":
        centers = torch.rand(shape, generator=g) * 2 - 1
    else:
        raise ValueError(f"unknown domain {domain!r}")
    lower, upper = centers - halves, centers + halves
    raw_a, raw_b = inverse_geometry(lower, upper, domain=domain, epsilon=epsilon)
    raw_a, raw_b = raw_a.to(device), raw_b.to(device)
    decoded = decode_geometry(raw_a, raw_b, domain=domain, epsilon=epsilon)
    tolerance = max(2e-5, 5e-5 * max(1.0, abs(half)))
    if (
        max(
            float((decoded[0] - lower.to(device)).abs().max()),
            float((decoded[1] - upper.to(device)).abs().max()),
        )
        > tolerance
    ):
        raise ValueError("initialization inverse/forward reconstruction exceeded tolerance")
    return raw_a, raw_b


def box_membership(
    points: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    domain: Domain,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Exact closed-union membership [N,C], independent of differentiable score."""
    _validate_kernel_inputs(points, lower, upper, domain, epsilon)
    if domain == "bounded":
        in_domain = ((points >= -1) & (points <= 1)).all(dim=-1)
    else:
        in_domain = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
    within = (
        (points[:, None, None, :] >= lower[None]) & (points[:, None, None, :] <= upper[None])
    ).all(dim=-1)
    return within.any(dim=2) & in_domain[:, None]


def box_clearance(
    points: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    domain: Domain,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Raw signed coordinate clearance: amin across faces, then amax across boxes."""
    _validate_kernel_inputs(points, lower, upper, domain, epsilon)
    faces = torch.stack(
        (points[:, None, None, :] - lower[None], upper[None] - points[:, None, None, :]), dim=-1
    )
    per_box = torch.amin(faces, dim=(-1, -2))
    return torch.amax(per_box, dim=-1)


def _validate_kernel_inputs(
    points: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    domain: Domain,
    epsilon: float,
) -> None:
    if not all(isinstance(x, torch.Tensor) for x in (points, lower, upper)):
        raise TypeError("kernel inputs must be tensors")
    if points.ndim != 2 or points.shape[1] < 1:
        raise ValueError("points must have shape [N,d] with d >= 1")
    validate_endpoints(lower, upper, domain=domain, epsilon=epsilon)
    if points.dtype != torch.float32:
        raise TypeError("points and endpoints must be float32")
    if points.device != lower.device or upper.device != lower.device:
        raise ValueError("points and endpoints must share a device")
    if lower.shape[2] != points.shape[1]:
        raise ValueError("point and endpoint dimensions differ")
    if not torch.isfinite(points).all():
        raise ValueError("points must be finite")


def box_objective(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    kind: Literal["logistic", "hinge"],
    tau: float = 0.1,
    margin: float = 0.1,
) -> torch.Tensor:
    """Per-example/class signed-clearance logistic or hinge penalties."""
    if (
        scores.ndim != 2
        or scores.shape != targets.shape
        or scores.dtype != torch.float32
        or targets.dtype not in (torch.float32, torch.bool)
    ):
        raise ValueError("scores and targets must have matching shapes and supported dtypes")
    if scores.device != targets.device or not torch.isfinite(scores).all():
        raise ValueError("scores must be finite and inputs share a device")
    if targets.dtype == torch.float32 and (
        not torch.isfinite(targets).all() or torch.any((targets != 0) & (targets != 1))
    ):
        raise ValueError("floating targets must be finite binary values")
    if not math.isfinite(tau) or tau <= 0 or not math.isfinite(margin) or margin < 0:
        raise ValueError("tau must be positive and margin nonnegative, both finite")
    signed_target = 2 * targets.to(torch.float32) - 1
    if kind == "logistic":
        return F.softplus(-signed_target * scores / tau)
    if kind == "hinge":
        return F.relu(margin - signed_target * scores)
    raise ValueError(f"unknown objective {kind!r}")


class BoxGeometry(nn.Module):
    """Minimal trainable raw geometry, decoded through the versioned F12 transform."""

    parameterization_version = "f12-width-first-v1"

    def __init__(
        self, raw_a: torch.Tensor, raw_b: torch.Tensor, *, domain: Domain, epsilon: float = 1e-4
    ):
        super().__init__()
        _check_raw(raw_a, raw_b, epsilon, domain)
        self.raw_a = nn.Parameter(raw_a.detach().clone())
        self.raw_b = nn.Parameter(raw_b.detach().clone())
        self.domain, self.epsilon = domain, epsilon

    @classmethod
    def initialized(
        cls,
        classes: int,
        boxes: int,
        dimensions: int,
        *,
        domain: Domain,
        seed: int,
        epsilon: float = 1e-4,
        volume_fraction: float | None = None,
        device: torch.device | str = "cpu",
    ) -> "BoxGeometry":
        a, b = initialize_geometry(
            classes,
            boxes,
            dimensions,
            domain=domain,
            seed=seed,
            epsilon=epsilon,
            volume_fraction=volume_fraction,
            device=device,
        )
        return cls(a, b, domain=domain, epsilon=epsilon)

    def endpoints(self) -> tuple[torch.Tensor, torch.Tensor]:
        return decode_geometry(self.raw_a, self.raw_b, domain=self.domain, epsilon=self.epsilon)
