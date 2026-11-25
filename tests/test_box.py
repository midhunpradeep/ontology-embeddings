from __future__ import annotations

import pytest
import torch

from ontology_embeddings.box import (
    BoxGeometry,
    box_clearance,
    box_membership,
    box_objective,
    decode_geometry,
    initialize_geometry,
    inverse_geometry,
    validate_endpoints,
)


def endpoints(lower, upper):
    return (
        torch.tensor(lower, dtype=torch.float32).reshape(1, 1, -1),
        torch.tensor(upper, dtype=torch.float32).reshape(1, 1, -1),
    )


@pytest.mark.parametrize("domain", ["bounded", "unbounded"])
def test_f12_transform_and_seeded_initialization_inverse(domain):
    a, b = initialize_geometry(3, 2, 4, domain=domain, seed=19)
    again = initialize_geometry(3, 2, 4, domain=domain, seed=19)
    assert torch.equal(a, again[0]) and torch.equal(b, again[1])
    lower, upper = decode_geometry(a, b, domain=domain)
    recovered = inverse_geometry(lower, upper, domain=domain)
    roundtrip = decode_geometry(*recovered, domain=domain)
    assert torch.allclose(lower, roundtrip[0], atol=2e-5, rtol=0)
    assert torch.allclose(upper, roundtrip[1], atol=2e-5, rtol=0)
    assert torch.all((upper - lower) >= 1e-4)
    if domain == "bounded":
        assert torch.all(lower >= -1) and torch.all(upper <= 1)


def test_f12_transform_matches_approved_width_first_equations():
    raw_a = torch.zeros((1, 1, 1), dtype=torch.float32)
    raw_b = torch.zeros_like(raw_a)
    eps = 1e-4
    bounded = decode_geometry(raw_a, raw_b, domain="bounded", epsilon=eps)
    expected_half = (eps + (2 - eps) * 0.5) / 2
    assert bounded[0].item() == pytest.approx(-expected_half, abs=1e-7)
    assert bounded[1].item() == pytest.approx(expected_half, abs=1e-7)
    unbounded = decode_geometry(raw_a, raw_b, domain="unbounded", epsilon=eps)
    expected_half = eps / 2 + torch.nn.functional.softplus(torch.tensor(0.0)).item()
    assert unbounded[0].item() == pytest.approx(-expected_half, abs=1e-7)
    assert unbounded[1].item() == pytest.approx(expected_half, abs=1e-7)


def test_exact_closed_union_membership_disconnected_boxes_and_empty_points():
    lo = torch.tensor([[[-1.0, -0.5], [0.5, 0.5]]])
    hi = torch.tensor([[[-0.5, 0.5], [1.0, 1.0]]])
    pts = torch.tensor([[-1.0, 0.0], [-0.5, 0.5], [0.5, 0.5], [0.0, 0.0], [1.1, 0.8]])
    got = box_membership(pts, lo, hi, domain="bounded")
    assert got.tolist() == [[True], [True], [True], [False], [False]]
    assert box_membership(torch.empty((0, 2)), lo, hi, domain="bounded").shape == (0, 1)
    # Class axis is independent and is never broadcast into another class.
    class_lo = torch.cat((lo, torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])), dim=0)
    class_hi = torch.cat((hi, torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])), dim=0)
    assert box_membership(
        torch.tensor([[0.25, 0.25]]), class_lo, class_hi, domain="bounded"
    ).tolist() == [[False, True]]


def test_clearance_is_min_face_then_max_box_and_has_symmetric_tie_gradients():
    lo = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    hi = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    p = torch.tensor([[0.5, 0.5]], requires_grad=True)
    score = box_clearance(p, lo, hi, domain="unbounded")
    assert score.item() == pytest.approx(0.5)
    score.sum().backward()
    assert p.grad.tolist() == [[0.0, 0.0]]
    assert box_clearance(
        torch.tensor([[2.0, 0.5]]), lo, hi, domain="unbounded"
    ).item() == pytest.approx(-1.0)
    assert box_membership(torch.tensor([[2.0, 0.5]]), lo, hi, domain="unbounded").tolist() == [
        [False]
    ]


def test_box_tie_uses_symmetric_max_gradient():
    lower = torch.tensor([[[0.3, 0.0], [0.0, 0.0]]])
    upper = torch.tensor([[[0.8, 1.0], [0.7, 1.0]]])
    point = torch.tensor([[0.5, 0.5]], requires_grad=True)
    value = box_clearance(point, lower, upper, domain="unbounded")
    assert value.item() == pytest.approx(0.2)
    value.sum().backward()
    assert point.grad.tolist() == [[0.0, 0.0]]


def test_clearance_finite_difference_away_from_kinks():
    lo, hi = endpoints([-0.4, -0.3], [0.4, 0.3])
    p = torch.tensor([[0.1, 0.05]], requires_grad=True)
    score = box_clearance(p, lo, hi, domain="unbounded").sum()
    score.backward()
    delta = 1e-3
    plus, minus = p.detach().clone(), p.detach().clone()
    plus[0, 1] += delta
    minus[0, 1] -= delta
    numerical = (
        box_clearance(plus, lo, hi, domain="unbounded")
        - box_clearance(minus, lo, hi, domain="unbounded")
    ) / (2 * delta)
    assert p.grad[0, 1].item() == pytest.approx(numerical.item(), abs=2e-4)


def test_width_floor_validation_and_input_contract():
    lo, hi = endpoints([0.0], [1.0001e-4])
    validate_endpoints(lo, hi, domain="bounded")
    with pytest.raises(ValueError, match="floor"):
        validate_endpoints(
            torch.tensor([[[0.0]]]), torch.tensor([[[0.00009999]]]), domain="bounded"
        )
    with pytest.raises(ValueError, match="dimension"):
        box_membership(torch.zeros((1, 2)), lo, hi, domain="bounded")
    with pytest.raises(ValueError, match="finite"):
        box_clearance(torch.tensor([[float("nan")]]), lo, hi, domain="bounded")
    with pytest.raises(TypeError, match="float32"):
        box_membership(torch.zeros((1, 1), dtype=torch.float64), lo, hi, domain="bounded")


def test_kernels_validate_against_the_configured_width_floor():
    lower = torch.tensor([[[0.0]]], dtype=torch.float32)
    upper = torch.tensor([[[5e-5]]], dtype=torch.float32)
    point = torch.tensor([[2.5e-5]], dtype=torch.float32)
    with pytest.raises(ValueError, match="floor"):
        box_membership(point, lower, upper, domain="unbounded")
    assert box_membership(point, lower, upper, domain="unbounded", epsilon=1e-5).tolist() == [
        [True]
    ]
    assert box_clearance(
        point, lower, upper, domain="unbounded", epsilon=1e-5
    ).item() == pytest.approx(2.5e-5)


def test_objective_formulas_and_input_validation():
    score = torch.tensor([[0.2, -0.3]])
    targets = torch.tensor([[True, False]])
    expected = torch.nn.functional.softplus(torch.tensor([[-2.0, -3.0]]))
    assert torch.allclose(box_objective(score, targets, kind="logistic"), expected)
    assert box_objective(score, targets, kind="hinge", margin=0.1).tolist() == [[0.0, 0.0]]
    assert torch.allclose(
        box_objective(score, targets, kind="hinge", margin=0.4),
        torch.tensor([[0.2, 0.1]]),
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="binary"):
        box_objective(score, torch.tensor([[0.5, 0.0]]), kind="hinge")


@pytest.mark.parametrize(
    "domain,kind",
    [(domain, kind) for domain in ("bounded", "unbounded") for kind in ("logistic", "hinge")],
)
def test_fixed_coordinate_geometry_only_recovery_controls(domain, kind):
    points = torch.tensor(
        [[-0.15, -0.12], [0.12, 0.10], [0.55, 0.0], [0.0, 0.55], [-0.55, 0.0], [0.0, -0.55]]
    )
    targets = torch.tensor([[1.0], [1.0], [0.0], [0.0], [0.0], [0.0]])
    start_lo, start_hi = endpoints([-0.7, -0.7], [0.7, 0.7])
    a, b = inverse_geometry(start_lo, start_hi, domain=domain)
    geom = BoxGeometry(a, b, domain=domain)
    opt = torch.optim.Adam(geom.parameters(), lr=0.04)
    initial_scores = box_clearance(points, *geom.endpoints(), domain=domain)
    initial_loss = box_objective(initial_scores, targets, kind=kind).mean()
    initial_loss.backward()
    assert any(p.grad is not None and p.grad.norm().item() > 0 for p in geom.parameters())
    geom.zero_grad(set_to_none=True)
    for _ in range(350):
        opt.zero_grad(set_to_none=True)
        loss = box_objective(
            box_clearance(points, *geom.endpoints(), domain=domain),
            targets,
            kind=kind,
            margin=0.1,
        ).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(geom.parameters(), 1.0)
        opt.step()
    lower, upper = geom.endpoints()
    assert box_membership(points, lower, upper, domain=domain).squeeze(1).tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert torch.isfinite(loss)
    assert not torch.equal(lower, start_lo) or not torch.equal(upper, start_hi)


def test_overlapping_union_boxes_and_finite_transform_extremes():
    lower = torch.tensor([[[0.0, 0.0], [0.5, 0.5]]])
    upper = torch.tensor([[[1.0, 1.0], [1.5, 1.5]]])
    assert box_membership(
        torch.tensor([[0.75, 0.75]]), lower, upper, domain="unbounded"
    ).tolist() == [[True]]
    for domain in ("bounded", "unbounded"):
        raw_a = torch.tensor([[[-8.0, 8.0]]], dtype=torch.float32)
        raw_b = torch.tensor([[[-4.0, 4.0]]], dtype=torch.float32)
        physical = decode_geometry(raw_a, raw_b, domain=domain)
        assert all(torch.isfinite(value).all() for value in physical)
        validate_endpoints(*physical, domain=domain)
