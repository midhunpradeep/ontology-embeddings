"""TOWN's point-level subtractive AABB geometry and physical parameterization.

The evaluator is deliberately independent of features, labels, objectives and optimizers.
Frames are closed; each window removes only the strict ambient interior of its own
hosting frame. All clearance reductions retain the native hierarchical amin/amax order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

Domain = Literal["bounded", "unbounded"]


def _validate_domain(domain: Domain) -> None:
    if domain not in ("bounded", "unbounded"):
        raise ValueError(f"unknown domain {domain!r}")


def _positive_floor(epsilon: float, reserve: float, domain: Domain) -> float:
    _validate_domain(domain)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not math.isfinite(reserve) or reserve < 0:
        raise ValueError("reserve must be finite and nonnegative")
    floor = epsilon + reserve
    if not math.isfinite(floor) or (domain == "bounded" and floor >= 2):
        raise ValueError("physical width floor must be representable in the selected domain")
    return floor


def validate_town_endpoints(
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    domain: Domain,
    epsilon: float = 1e-4,
    reserve: float = 0.0,
) -> None:
    """Validate physical float32 endpoints and a full-width floor.

    Frames hosting at least one window use ``reserve=2*delta``. Ordinary frames
    and all windows use ``reserve=0``. A zero-sized window axis is valid.
    """
    floor = _positive_floor(epsilon, reserve, domain)
    if not isinstance(lower, torch.Tensor) or not isinstance(upper, torch.Tensor):
        raise TypeError("endpoints must be tensors")
    if lower.ndim not in (3, 4) or lower.shape != upper.shape:
        raise ValueError("endpoints must share [C,H,d] or [C,H,W,d] shape")
    if lower.ndim == 3:
        valid_shape = all(size >= 1 for size in lower.shape)
    else:
        valid_shape = (
            lower.shape[0] >= 1
            and lower.shape[1] >= 1
            and lower.shape[2] >= 0
            and lower.shape[3] >= 1
        )
    if not valid_shape:
        raise ValueError("endpoint axes require positive C, H, d and nonnegative W")
    if lower.dtype != torch.float32 or upper.dtype != torch.float32:
        raise TypeError("endpoints must be float32")
    if lower.device != upper.device:
        raise ValueError("endpoints must share a device")
    if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
        raise ValueError("endpoints must be finite")
    lo, hi = lower.to(torch.float64), upper.to(torch.float64)
    widths = hi - lo
    if torch.any(hi <= lo) or torch.any(widths < floor):
        raise ValueError("endpoints must be ordered and meet the full-width floor")
    if domain == "bounded" and (torch.any(lo < -1.0) or torch.any(hi > 1.0)):
        raise ValueError("bounded endpoints must lie in [-1, 1]")


def _check_raw(
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    *,
    domain: Domain,
    epsilon: float,
    reserve: float,
    ndim: int | None = None,
) -> None:
    _positive_floor(epsilon, reserve, domain)
    if not isinstance(raw_a, torch.Tensor) or not isinstance(raw_b, torch.Tensor):
        raise TypeError("raw geometry must be tensors")
    if raw_a.shape != raw_b.shape or raw_a.ndim not in (3, 4):
        raise ValueError("raw geometry must share [C,H,d] or [C,H,W,d] shape")
    if ndim is not None and raw_a.ndim != ndim:
        raise ValueError(f"raw geometry must have rank {ndim}")
    if raw_a.ndim == 3:
        valid_shape = all(size >= 1 for size in raw_a.shape)
    else:
        valid_shape = (
            raw_a.shape[0] >= 1
            and raw_a.shape[1] >= 1
            and raw_a.shape[2] >= 1
            and raw_a.shape[3] >= 1
        )
    if not valid_shape:
        raise ValueError("raw geometry axes must be nonempty")
    if raw_a.dtype != torch.float32 or raw_b.dtype != torch.float32:
        raise TypeError("raw geometry must be float32")
    if raw_a.device != raw_b.device:
        raise ValueError("raw geometry must share a device")
    if not torch.isfinite(raw_a).all() or not torch.isfinite(raw_b).all():
        raise ValueError("raw geometry must be finite")


def decode_parameters(
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    *,
    domain: Domain,
    epsilon: float = 1e-4,
    reserve: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode width-first raw parameters into physical lower/upper endpoints."""
    floor = _positive_floor(epsilon, reserve, domain)
    _check_raw(
        raw_a, raw_b, domain=domain, epsilon=epsilon, reserve=reserve
    )
    if domain == "bounded":
        half = (floor + (2.0 - floor) * torch.sigmoid(raw_a)) / 2.0
        center = (1.0 - half) * torch.tanh(raw_b)
    else:
        half = floor / 2.0 + F.softplus(raw_a)
        center = raw_b
    lower, upper = center - half, center + half
    validate_town_endpoints(
        lower, upper, domain=domain, epsilon=epsilon, reserve=reserve
    )
    return lower, upper


def _encode_unchecked(
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    domain: Domain,
    floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    lo, hi = lower.to(torch.float64), upper.to(torch.float64)
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    if domain == "bounded":
        ratio = (2.0 * half - floor) / (2.0 - floor)
        center_ratio = center / (1.0 - half)
        raw_a = torch.logit(ratio).to(torch.float32)
        raw_b = torch.atanh(center_ratio).to(torch.float32)
    else:
        gap = half - floor / 2.0
        raw_a = (gap + torch.log(-torch.expm1(-gap))).to(torch.float32)
        raw_b = center.to(torch.float32)
    return raw_a, raw_b


def encode_endpoints(
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    domain: Domain,
    epsilon: float = 1e-4,
    reserve: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert the width-first transform without clipping or repairing endpoints."""
    floor = _positive_floor(epsilon, reserve, domain)
    validate_town_endpoints(
        lower, upper, domain=domain, epsilon=epsilon, reserve=reserve
    )
    lo, hi = lower.to(torch.float64), upper.to(torch.float64)
    half = (hi - lo) / 2.0
    center = (lo + hi) / 2.0
    if domain == "bounded":
        ratio = (2.0 * half - floor) / (2.0 - floor)
        center_ratio = center / (1.0 - half)
        if (
            torch.any(ratio <= 0)
            or torch.any(ratio >= 1)
            or torch.any(center_ratio.abs() >= 1)
        ):
            raise ValueError("bounded endpoints have no finite inverse")
    elif torch.any(half <= floor / 2.0):
        raise ValueError("unbounded endpoints have no finite inverse above the width floor")
    raw_a, raw_b = _encode_unchecked(lower, upper, domain=domain, floor=floor)
    if not torch.isfinite(raw_a).all() or not torch.isfinite(raw_b).all():
        raise ValueError("endpoint inverse is outside the finite float32 operating envelope")
    return raw_a, raw_b


def _validate_kernel_inputs(
    points: torch.Tensor,
    frame_lower: torch.Tensor,
    frame_upper: torch.Tensor,
    window_lower: torch.Tensor,
    window_upper: torch.Tensor,
    *,
    domain: Domain,
    epsilon: float,
    delta: float,
) -> None:
    _validate_domain(domain)
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    if not isinstance(points, torch.Tensor) or points.ndim != 2 or points.shape[1] < 1:
        raise ValueError("points must have shape [N,d] with d >= 1")
    if points.dtype != torch.float32:
        raise TypeError("points must be float32")
    if not torch.isfinite(points).all():
        raise ValueError("points must be finite")
    if not isinstance(frame_lower, torch.Tensor) or frame_lower.ndim != 3:
        raise ValueError("frame endpoints must have shape [C,H,d]")
    if not isinstance(window_lower, torch.Tensor) or window_lower.ndim != 4:
        raise ValueError("window endpoints must have shape [C,H,W,d]")
    if (
        frame_lower.shape[0] != window_lower.shape[0]
        or frame_lower.shape[1] != window_lower.shape[1]
        or frame_lower.shape[2] != points.shape[1]
        or window_lower.shape[3] != points.shape[1]
    ):
        raise ValueError("point, frame and window axes are incompatible")
    hosting_reserve = 2.0 * delta if window_lower.shape[2] > 0 else 0.0
    validate_town_endpoints(
        frame_lower,
        frame_upper,
        domain=domain,
        epsilon=epsilon,
        reserve=hosting_reserve,
    )
    validate_town_endpoints(
        window_lower, window_upper, domain=domain, epsilon=epsilon, reserve=0.0
    )
    if any(
        endpoint.device != points.device
        for endpoint in (frame_lower, frame_upper, window_lower, window_upper)
    ):
        raise ValueError("points and endpoints must share a device")


def town_clearance(
    points: torch.Tensor,
    frame_lower: torch.Tensor,
    frame_upper: torch.Tensor,
    window_lower: torch.Tensor,
    window_upper: torch.Tensor,
    *,
    domain: Domain,
    epsilon: float = 1e-4,
    delta: float = 1e-4,
) -> torch.Tensor:
    """Return raw-coordinate town clearance [N,C] with native hierarchical AD."""
    _validate_kernel_inputs(
        points,
        frame_lower,
        frame_upper,
        window_lower,
        window_upper,
        domain=domain,
        epsilon=epsilon,
        delta=delta,
    )
    frame_faces = torch.stack(
        (
            points[:, None, None, :] - frame_lower[None],
            frame_upper[None] - points[:, None, None, :],
        ),
        dim=-1,
    )
    frame_scores = torch.amin(frame_faces, dim=(-1, -2))
    window_faces = torch.stack(
        (
            points[:, None, None, None, :] - window_lower[None],
            window_upper[None] - points[:, None, None, None, :],
        ),
        dim=-1,
    )
    window_scores = torch.amin(window_faces, dim=(-1, -2))
    house_candidates = torch.cat((frame_scores.unsqueeze(-1), -window_scores), dim=-1)
    house_scores = torch.amin(house_candidates, dim=-1)
    return torch.amax(house_scores, dim=2)


def town_membership(
    points: torch.Tensor,
    frame_lower: torch.Tensor,
    frame_upper: torch.Tensor,
    window_lower: torch.Tensor,
    window_upper: torch.Tensor,
    *,
    domain: Domain,
    epsilon: float = 1e-4,
    delta: float = 1e-4,
) -> torch.Tensor:
    """Return exact closed-frame/strict-window-interior membership [N,C]."""
    _validate_kernel_inputs(
        points,
        frame_lower,
        frame_upper,
        window_lower,
        window_upper,
        domain=domain,
        epsilon=epsilon,
        delta=delta,
    )
    in_frame = (
        (points[:, None, None, :] >= frame_lower[None])
        & (points[:, None, None, :] <= frame_upper[None])
    ).all(dim=-1)
    in_window_interior = (
        (points[:, None, None, None, :] > window_lower[None])
        & (points[:, None, None, None, :] < window_upper[None])
    ).all(dim=-1)
    in_house = in_frame & ~in_window_interior.any(dim=-1)
    in_town = in_house.any(dim=2)
    if domain == "bounded":
        in_domain = ((points >= -1.0) & (points <= 1.0)).all(dim=-1)
        in_town = in_town & in_domain[:, None]
    return in_town


def reference_town_clearance(
    points: torch.Tensor,
    frame_lower: torch.Tensor,
    frame_upper: torch.Tensor,
    window_lower: torch.Tensor,
    window_upper: torch.Tensor,
    *,
    domain: Domain = "unbounded",
    epsilon: float = 1e-4,
    delta: float = 1e-4,
) -> torch.Tensor:
    """Slow independent hierarchical oracle, intended only for focused tests."""
    _validate_kernel_inputs(
        points,
        frame_lower,
        frame_upper,
        window_lower,
        window_upper,
        domain=domain,
        epsilon=epsilon,
        delta=delta,
    )
    if points.shape[0] == 0:
        return points.new_empty((0, frame_lower.shape[0]))
    rows = []
    for point in points:
        class_values = []
        for class_index in range(frame_lower.shape[0]):
            houses = []
            for house in range(frame_lower.shape[1]):
                frame_faces = torch.cat(
                    (
                        point - frame_lower[class_index, house],
                        frame_upper[class_index, house] - point,
                    )
                )
                frame_score = torch.amin(frame_faces, dim=0)
                candidates = [frame_score]
                for window in range(window_lower.shape[2]):
                    faces = torch.cat(
                        (
                            point - window_lower[class_index, house, window],
                            window_upper[class_index, house, window] - point,
                        )
                    )
                    candidates.append(-torch.amin(faces, dim=0))
                houses.append(torch.amin(torch.stack(candidates), dim=0))
            class_values.append(torch.amax(torch.stack(houses), dim=0))
        rows.append(torch.stack(class_values))
    return torch.stack(rows)


def reference_town_membership(
    points: torch.Tensor,
    frame_lower: torch.Tensor,
    frame_upper: torch.Tensor,
    window_lower: torch.Tensor,
    window_upper: torch.Tensor,
    *,
    domain: Domain = "unbounded",
    epsilon: float = 1e-4,
    delta: float = 1e-4,
) -> torch.Tensor:
    """Slow Python-loop Boolean oracle independent of the vectorized membership kernel."""
    _validate_kernel_inputs(
        points,
        frame_lower,
        frame_upper,
        window_lower,
        window_upper,
        domain=domain,
        epsilon=epsilon,
        delta=delta,
    )
    if points.shape[0] == 0:
        return torch.empty((0, frame_lower.shape[0]), dtype=torch.bool, device=points.device)
    result = []
    for point in points:
        class_membership = []
        for class_index in range(frame_lower.shape[0]):
            house_membership = []
            for house in range(frame_lower.shape[1]):
                in_frame = bool(
                    (
                        (point >= frame_lower[class_index, house])
                        & (point <= frame_upper[class_index, house])
                    ).all()
                )
                excluded = False
                for window in range(window_lower.shape[2]):
                    excluded = excluded or bool(
                        (
                            (point > window_lower[class_index, house, window])
                            & (point < window_upper[class_index, house, window])
                        ).all()
                    )
                house_membership.append(in_frame and not excluded)
            in_domain = domain != "bounded" or bool(((point >= -1) & (point <= 1)).all())
            class_membership.append(any(house_membership) and in_domain)
        result.append(class_membership)
    return torch.tensor(result, dtype=torch.bool, device=points.device)


def town_objective(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    kind: Literal["logistic", "hinge"],
    tau: float = 0.1,
    margin: float = 0.1,
) -> torch.Tensor:
    """Per-example signed-clearance logistic or hinge penalties."""
    if (
        not isinstance(scores, torch.Tensor)
        or not isinstance(targets, torch.Tensor)
        or scores.ndim != 2
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


class TownGeometry(nn.Module):
    """Trainable raw frame/window state with no parameters for a zero-window model."""

    parameterization_version = "town-width-first-v1"

    def __init__(
        self,
        raw_frame_a: torch.Tensor,
        raw_frame_b: torch.Tensor,
        raw_window_a: torch.Tensor | None = None,
        raw_window_b: torch.Tensor | None = None,
        *,
        domain: Domain,
        epsilon: float = 1e-4,
        delta: float = 1e-4,
    ) -> None:
        super().__init__()
        _check_raw(
            raw_frame_a,
            raw_frame_b,
            domain=domain,
            epsilon=epsilon,
            reserve=0.0,
            ndim=3,
        )
        if raw_frame_a.shape[1] < 1:
            raise ValueError("TOWN requires at least one frame per class")
        if not math.isfinite(delta) or delta <= 0:
            raise ValueError("delta must be finite and positive")
        if (raw_window_a is None) != (raw_window_b is None):
            raise ValueError("both window parameter tensors must be supplied together")
        self.raw_frame_a = nn.Parameter(raw_frame_a.detach().clone())
        self.raw_frame_b = nn.Parameter(raw_frame_b.detach().clone())
        if raw_window_a is None:
            self.register_parameter("raw_window_a", None)
            self.register_parameter("raw_window_b", None)
            self.window_count = 0
        else:
            _check_raw(
                raw_window_a,
                raw_window_b,
                domain=domain,
                epsilon=epsilon,
                reserve=0.0,
                ndim=4,
            )
            if (
                raw_window_a.shape[:2] != raw_frame_a.shape[:2]
                or raw_window_a.shape[2] < 1
                or raw_window_a.shape[3] != raw_frame_a.shape[2]
            ):
                raise ValueError("window axes must match their hosting frame axes")
            self.raw_window_a = nn.Parameter(raw_window_a.detach().clone())
            self.raw_window_b = nn.Parameter(raw_window_b.detach().clone())
            self.window_count = raw_window_a.shape[2]
        self.domain, self.epsilon, self.delta = domain, epsilon, delta

    def endpoints(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reserve = 2.0 * self.delta if self.window_count else 0.0
        frame_lower, frame_upper = decode_parameters(
            self.raw_frame_a,
            self.raw_frame_b,
            domain=self.domain,
            epsilon=self.epsilon,
            reserve=reserve,
        )
        if self.window_count:
            window_lower, window_upper = decode_parameters(
                self.raw_window_a,
                self.raw_window_b,
                domain=self.domain,
                epsilon=self.epsilon,
            )
        else:
            empty_shape = (*frame_lower.shape[:2], 0, frame_lower.shape[-1])
            window_lower = frame_lower.new_empty(empty_shape)
            window_upper = frame_upper.new_empty(empty_shape)
        return frame_lower, frame_upper, window_lower, window_upper

    def clearance(self, points: torch.Tensor) -> torch.Tensor:
        return town_clearance(
            points,
            *self.endpoints(),
            domain=self.domain,
            epsilon=self.epsilon,
            delta=self.delta,
        )

    def membership(self, points: torch.Tensor) -> torch.Tensor:
        return town_membership(
            points,
            *self.endpoints(),
            domain=self.domain,
            epsilon=self.epsilon,
            delta=self.delta,
        )


@dataclass(frozen=True)
class InitializedWindows:
    """Deterministic volume-first targets and the actual decoded float32 geometry."""

    raw_a: torch.Tensor
    raw_b: torch.Tensor
    target_lower: torch.Tensor
    target_upper: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    sampled_fractions: torch.Tensor
    actual_fractions: torch.Tensor


def _candidate_representable(
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    domain: Domain,
    floor: float,
) -> torch.Tensor:
    lo, hi = lower.to(torch.float64), upper.to(torch.float64)
    half = (hi - lo) / 2.0
    center = (hi + lo) / 2.0
    representable = (hi > lo).all(dim=-1) & ((hi - lo) >= floor).all(dim=-1)
    if domain == "bounded":
        ratio = (2 * half - floor) / (2 - floor)
        center_ratio = center / (1 - half)
        representable = representable & (ratio > 0).all(dim=-1) & (ratio < 1).all(dim=-1)
        representable = representable & (center_ratio.abs() < 1).all(dim=-1)
        representable = representable & (lo >= -1).all(dim=-1) & (hi <= 1).all(dim=-1)
    else:
        representable = representable & (half > floor / 2).all(dim=-1)
    return representable


def initialize_volume_first_windows(
    frame_lower: torch.Tensor,
    frame_upper: torch.Tensor,
    *,
    window_count: int,
    domain: Domain,
    seed: int,
    epsilon: float = 1e-4,
    delta: float = 1e-4,
    retry_limit: int = 64,
) -> InitializedWindows:
    """Seeded per-window volume-first initialization with decoded-fraction retries.

    Each attempted target draws q uniformly on [0.05, 0.10), sets each half-width
    to q**(1/d) times its hosting-frame half-width, and samples feasible centers.
    A result is published only when every actual decoded window is contained and
    excludes at least 5% of its actual hosting-frame volume.
    """
    if isinstance(window_count, bool) or not isinstance(window_count, int) or window_count < 0:
        raise ValueError("window_count must be a nonnegative integer")
    if isinstance(retry_limit, bool) or not isinstance(retry_limit, int) or retry_limit < 1:
        raise ValueError("retry_limit must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if frame_lower.ndim != 3 or frame_lower.shape[1] < 1:
        raise ValueError("frame endpoints must have shape [C,H,d]")
    reserve = 2.0 * delta if window_count else 0.0
    validate_town_endpoints(
        frame_lower,
        frame_upper,
        domain=domain,
        epsilon=epsilon,
        reserve=reserve,
    )
    classes, houses, dimensions = frame_lower.shape
    out_device = frame_lower.device
    frame_lo = frame_lower.detach().to(device="cpu")
    frame_hi = frame_upper.detach().to(device="cpu")
    window_shape = (classes, houses, window_count, dimensions)
    fraction_shape = (classes, houses, window_count)
    if window_count == 0:
        empty = torch.empty(window_shape, dtype=torch.float32, device=out_device)
        empty_fraction = torch.empty(fraction_shape, dtype=torch.float32, device=out_device)
        return InitializedWindows(
            empty,
            empty.clone(),
            empty.clone(),
            empty.clone(),
            empty.clone(),
            empty.clone(),
            empty_fraction,
            empty_fraction.clone(),
        )

    floor = _positive_floor(epsilon, 0.0, domain)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    accepted = torch.zeros(fraction_shape, dtype=torch.bool)
    raw_a_result = torch.zeros(window_shape, dtype=torch.float32)
    raw_b_result = torch.zeros_like(raw_a_result)
    target_lower_result = torch.zeros_like(raw_a_result)
    target_upper_result = torch.zeros_like(raw_a_result)
    lower_result = torch.zeros_like(raw_a_result)
    upper_result = torch.zeros_like(raw_a_result)
    sampled_result = torch.zeros(fraction_shape, dtype=torch.float32)
    actual_result = torch.zeros_like(sampled_result)
    frame_width = frame_hi - frame_lo
    frame_half = frame_width / 2.0
    for _ in range(retry_limit):
        pending = ~accepted
        if not bool(pending.any()):
            break
        sampled = 0.05 + 0.05 * torch.rand(fraction_shape, generator=generator)
        radius = sampled.pow(1.0 / dimensions).unsqueeze(-1)
        half = frame_half.unsqueeze(2) * radius
        center_min = frame_lo.unsqueeze(2) + half
        center_max = frame_hi.unsqueeze(2) - half
        random_center = torch.rand(window_shape, generator=generator)
        center = center_min + random_center * (center_max - center_min)
        target_lower, target_upper = center - half, center + half
        representable = _candidate_representable(
            target_lower, target_upper, domain=domain, floor=floor
        )
        safe_half = (floor + 2.0) / 4.0
        safe_lower = torch.full_like(target_lower, -safe_half)
        safe_upper = torch.full_like(target_upper, safe_half)
        use_candidate = representable.unsqueeze(-1)
        inverse_lower = torch.where(use_candidate, target_lower, safe_lower)
        inverse_upper = torch.where(use_candidate, target_upper, safe_upper)
        raw_a, raw_b = _encode_unchecked(
            inverse_lower, inverse_upper, domain=domain, floor=floor
        )
        finite_raw = torch.isfinite(raw_a).all(dim=-1) & torch.isfinite(raw_b).all(dim=-1)
        representable = representable & finite_raw
        safe_raw = representable.unsqueeze(-1)
        raw_a = torch.where(safe_raw, raw_a, torch.zeros_like(raw_a))
        raw_b = torch.where(safe_raw, raw_b, torch.zeros_like(raw_b))
        decoded_lower, decoded_upper = decode_parameters(
            raw_a, raw_b, domain=domain, epsilon=epsilon
        )
        intersection = (
            torch.minimum(decoded_upper, frame_hi.unsqueeze(2))
            - torch.maximum(decoded_lower, frame_lo.unsqueeze(2))
        ).clamp_min(0.0)
        actual_fraction = torch.prod(
            intersection.to(torch.float64) / frame_width.unsqueeze(2).to(torch.float64), dim=-1
        )
        contained = (
            (decoded_lower >= frame_lo.unsqueeze(2)).all(dim=-1)
            & (decoded_upper <= frame_hi.unsqueeze(2)).all(dim=-1)
        )
        good = representable & contained & (actual_fraction >= 0.05)
        take = pending & good
        take_coords = take.unsqueeze(-1)
        raw_a_result = torch.where(take_coords, raw_a, raw_a_result)
        raw_b_result = torch.where(take_coords, raw_b, raw_b_result)
        target_lower_result = torch.where(take_coords, target_lower, target_lower_result)
        target_upper_result = torch.where(take_coords, target_upper, target_upper_result)
        lower_result = torch.where(take_coords, decoded_lower, lower_result)
        upper_result = torch.where(take_coords, decoded_upper, upper_result)
        sampled_result = torch.where(take, sampled, sampled_result)
        actual_result = torch.where(take, actual_fraction.to(torch.float32), actual_result)
        accepted = accepted | take
    if not bool(accepted.all()):
        count = int((~accepted).sum())
        raise RuntimeError(
            f"volume-first window initialization exceeded retry limit {retry_limit}; "
            f"{count} window(s) had no valid decoded sample"
        )
    # Validate the published float32 values again; never repair or clip them.
    validate_town_endpoints(
        lower_result.to(out_device), upper_result.to(out_device), domain=domain, epsilon=epsilon
    )
    if torch.any(actual_result < 0.05) or not torch.isfinite(actual_result).all():
        raise RuntimeError("decoded volume-first initialization violated the actual 5% floor")
    return InitializedWindows(
        raw_a_result.to(out_device),
        raw_b_result.to(out_device),
        target_lower_result.to(out_device),
        target_upper_result.to(out_device),
        lower_result.to(out_device),
        upper_result.to(out_device),
        sampled_result.to(out_device),
        actual_result.to(out_device),
    )
