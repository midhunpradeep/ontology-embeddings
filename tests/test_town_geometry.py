from __future__ import annotations

import pytest
import torch

from ontology_embeddings.town_geometry import (
    TownGeometry,
    decode_parameters,
    encode_endpoints,
    initialize_volume_first_windows,
    reference_town_clearance,
    reference_town_membership,
    town_clearance,
    town_membership,
    town_objective,
    validate_town_endpoints,
)


def _reference(points, frame_lower, frame_upper, window_lower, window_upper):
    """Deliberately scalar/list-based oracle, independent of vectorized kernels."""
    if points.shape[0] == 0:
        return torch.empty((0, frame_lower.shape[0])), torch.empty(
            (0, frame_lower.shape[0]), dtype=torch.bool
        )
    score_rows, membership_rows = [], []
    for point in points:
        class_scores, class_memberships = [], []
        for class_index in range(frame_lower.shape[0]):
            house_scores, house_memberships = [], []
            for house in range(frame_lower.shape[1]):
                face_scores = torch.cat(
                    (
                        point - frame_lower[class_index, house],
                        frame_upper[class_index, house] - point,
                    )
                )
                frame_score = torch.amin(face_scores, dim=0)
                candidates = [frame_score]
                in_frame = bool(
                    (
                        (point >= frame_lower[class_index, house])
                        & (point <= frame_upper[class_index, house])
                    ).all()
                )
                excluded = False
                for window in range(window_lower.shape[2]):
                    window_faces = torch.cat(
                        (
                            point - window_lower[class_index, house, window],
                            window_upper[class_index, house, window] - point,
                        )
                    )
                    window_score = torch.amin(window_faces, dim=0)
                    candidates.append(-window_score)
                    strict_interior = bool(
                        (
                            (point > window_lower[class_index, house, window])
                            & (point < window_upper[class_index, house, window])
                        ).all()
                    )
                    excluded = excluded or strict_interior
                house_scores.append(torch.amin(torch.stack(candidates), dim=0))
                house_memberships.append(in_frame and not excluded)
            class_scores.append(torch.amax(torch.stack(house_scores), dim=0))
            class_memberships.append(any(house_memberships))
        score_rows.append(torch.stack(class_scores))
        membership_rows.append(class_memberships)
    return torch.stack(score_rows), torch.tensor(membership_rows, dtype=torch.bool)


def test_exact_membership_uses_closed_frames_and_strict_ambient_window_interiors():
    points = torch.tensor([[-1.0], [-0.2], [0.0], [0.2], [0.4], [1.0]])
    frame_lower = torch.tensor([[[-1.0]]])
    frame_upper = torch.tensor([[[1.0]]])
    window_lower = torch.tensor([[[[-0.2], [0.0]]]])
    window_upper = torch.tensor([[[[0.2], [0.4]]]])
    expected = [[True], [True], [False], [False], [True], [True]]
    assert town_membership(
        points, frame_lower, frame_upper, window_lower, window_upper, domain="unbounded"
    ).tolist() == expected
    assert town_membership(
        torch.tensor([[1.01]]),
        frame_lower,
        frame_upper,
        window_lower,
        window_upper,
        domain="bounded",
    ).tolist() == [[False]]


def test_independent_reference_matches_vectorized_scores_membership_and_hierarchical_ties():
    frame_lower = torch.tensor(
        [[[-1.0, -1.0], [-1.0, -1.0]], [[0.0, 0.0], [0.0, 0.0]]], requires_grad=True
    )
    frame_upper = torch.tensor(
        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]], requires_grad=True
    )
    window_lower = torch.tensor(
        [[[[-0.5, -0.5], [-0.5, -0.5]], [[-0.5, -0.5], [-0.5, -0.5]]],
         [[[0.25, 0.25], [0.25, 0.25]], [[0.25, 0.25], [0.25, 0.25]]]], requires_grad=True
    )
    window_upper = torch.tensor(
        [[[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]],
         [[[0.75, 0.75], [0.75, 0.75]], [[0.75, 0.75], [0.75, 0.75]]]], requires_grad=True
    )
    points = torch.tensor([[0.0, 0.0], [0.5, 0.5]], requires_grad=True)
    got = town_clearance(
        points, frame_lower, frame_upper, window_lower, window_upper, domain="unbounded"
    )
    expected, expected_membership = _reference(
        points, frame_lower, frame_upper, window_lower, window_upper
    )
    reference_scores = reference_town_clearance(
        points,
        frame_lower,
        frame_upper,
        window_lower,
        window_upper,
        domain="unbounded",
    )
    reference_membership = reference_town_membership(
        points,
        frame_lower,
        frame_upper,
        window_lower,
        window_upper,
        domain="unbounded",
    )
    assert torch.equal(reference_scores, expected)
    assert torch.equal(reference_membership, expected_membership)
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6)
    assert torch.equal(
        town_membership(
            points, frame_lower, frame_upper, window_lower, window_upper, domain="unbounded"
        ),
        expected_membership,
    )
    parameters = (points, frame_lower, frame_upper, window_lower, window_upper)
    actual_grads = torch.autograd.grad(got.sum(), parameters, retain_graph=True)
    oracle_grads = torch.autograd.grad(expected.sum(), parameters)
    assert all(
        torch.allclose(a, b, atol=1e-6, rtol=1e-6)
        for a, b in zip(actual_grads, oracle_grads)
    )


def test_windows_are_ambient_and_subtract_only_from_their_own_hosting_house():
    points = torch.tensor([[0.0], [-1.0], [2.0]])
    frame_lower = torch.tensor([[[-1.0], [-0.2]]])
    frame_upper = torch.tensor([[[1.0], [0.2]]])
    # The first window protrudes outside its hosting frame; the second house's
    # independent window is inactive around x=0.
    window_lower = torch.tensor([[[[-2.0]], [[2.0]]]])
    window_upper = torch.tensor([[[[0.5]], [[3.0]]]])
    got = town_membership(
        points,
        frame_lower,
        frame_upper,
        window_lower,
        window_upper,
        domain="unbounded",
    )
    # x=0 is removed from house 0 but retained by house 1. x=-1 lies in the
    # ambient strict interior and is removed, even though it is a frame boundary.
    assert got.tolist() == [[True], [False], [False]]


def test_zero_windows_empty_batches_and_window_parameters_absent():
    raw_frame_a = torch.zeros((2, 1, 3), dtype=torch.float32)
    raw_frame_b = torch.zeros_like(raw_frame_a)
    geometry = TownGeometry(raw_frame_a, raw_frame_b, domain="unbounded")
    frame_lower, frame_upper, window_lower, window_upper = geometry.endpoints()
    assert frame_lower.shape == frame_upper.shape == (2, 1, 3)
    assert window_lower.shape == window_upper.shape == (2, 1, 0, 3)
    assert geometry.raw_window_a is None and geometry.raw_window_b is None
    assert "raw_window_a" not in dict(geometry.named_parameters())
    assert town_clearance(
        torch.empty((0, 3)), frame_lower, frame_upper, window_lower, window_upper,
        domain="unbounded",
    ).shape == (0, 2)
    assert town_membership(
        torch.empty((0, 3)), frame_lower, frame_upper, window_lower, window_upper,
        domain="unbounded",
    ).shape == (0, 2)


@pytest.mark.parametrize("domain", ["bounded", "unbounded"])
@pytest.mark.parametrize("hosting", [False, True])
def test_parameterization_roundtrip_and_physical_width_reserves(domain, hosting):
    lower = torch.tensor([[[-0.7, -0.4], [0.1, -0.2]]], dtype=torch.float32)
    upper = torch.tensor([[[0.5, 0.6], [0.6, 0.4]]], dtype=torch.float32)
    epsilon, delta = 1e-4, 1e-4
    reserve = 2 * delta if hosting else 0.0
    raw_a, raw_b = encode_endpoints(
        lower, upper, domain=domain, epsilon=epsilon, reserve=reserve
    )
    decoded = decode_parameters(
        raw_a, raw_b, domain=domain, epsilon=epsilon, reserve=reserve
    )
    assert torch.allclose(decoded[0], lower, atol=2e-5, rtol=0)
    assert torch.allclose(decoded[1], upper, atol=2e-5, rtol=0)
    validate_town_endpoints(*decoded, domain=domain, epsilon=epsilon, reserve=reserve)
    if hosting:
        assert torch.all(
            (decoded[1] - decoded[0]).to(torch.float64) >= epsilon + 2 * delta
        )


def test_decode_rejects_nonfinite_raw_and_validate_rejects_short_hosting_frame():
    with pytest.raises(ValueError, match="finite"):
        decode_parameters(
            torch.tensor([[[float("inf")]]]),
            torch.zeros((1, 1, 1)),
            domain="unbounded",
        )
    with pytest.raises(ValueError, match="floor"):
        validate_town_endpoints(
            torch.tensor([[[-0.0001]]]),
            torch.tensor([[[0.0001]]]),
            domain="unbounded",
            epsilon=1e-4,
            reserve=2e-4,
        )


def test_volume_first_windows_are_seeded_contained_and_validate_decoded_fraction():
    frame_lower = torch.tensor([[[-0.8, -0.6], [-0.7, -0.5]]])
    frame_upper = torch.tensor([[[0.8, 0.6], [0.7, 0.5]]])
    before_lower, before_upper = frame_lower.clone(), frame_upper.clone()
    first = initialize_volume_first_windows(
        frame_lower, frame_upper, window_count=2, domain="bounded", seed=41, retry_limit=64
    )
    second = initialize_volume_first_windows(
        frame_lower, frame_upper, window_count=2, domain="bounded", seed=41, retry_limit=64
    )
    assert torch.equal(first.raw_a, second.raw_a) and torch.equal(first.raw_b, second.raw_b)
    assert torch.equal(first.lower, second.lower) and torch.equal(first.upper, second.upper)
    assert torch.equal(frame_lower, before_lower) and torch.equal(frame_upper, before_upper)
    assert torch.all(first.sampled_fractions >= 0.05)
    assert torch.all(first.sampled_fractions <= 0.10)
    assert torch.all(first.actual_fractions >= 0.05)
    assert torch.allclose(first.lower, first.target_lower, atol=2e-5, rtol=0)
    assert torch.allclose(first.upper, first.target_upper, atol=2e-5, rtol=0)
    assert torch.all(first.lower >= frame_lower.unsqueeze(2))
    assert torch.all(first.upper <= frame_upper.unsqueeze(2))
    decoded = decode_parameters(first.raw_a, first.raw_b, domain="bounded")
    assert torch.allclose(decoded[0], first.lower, atol=2e-5, rtol=0)
    assert torch.allclose(decoded[1], first.upper, atol=2e-5, rtol=0)


def test_zero_window_initializer_has_only_empty_axes_and_impossible_init_fails_finitely():
    lower, upper = torch.tensor([[[-0.5]]]), torch.tensor([[[0.5]]])
    empty = initialize_volume_first_windows(
        lower, upper, window_count=0, domain="unbounded", seed=3
    )
    assert empty.raw_a.shape == empty.raw_b.shape == (1, 1, 0, 1)
    assert empty.target_lower.shape == empty.target_upper.shape == (1, 1, 0, 1)
    assert empty.lower.shape == empty.upper.shape == (1, 1, 0, 1)
    assert empty.sampled_fractions.shape == empty.actual_fractions.shape == (1, 1, 0)
    tiny_lower, tiny_upper = torch.tensor([[[-0.000155]]]), torch.tensor([[[0.000155]]])
    with pytest.raises(RuntimeError, match="retry limit"):
        initialize_volume_first_windows(
            tiny_lower,
            tiny_upper,
            window_count=1,
            domain="unbounded",
            seed=3,
            retry_limit=2,
        )


def test_objective_uses_approved_logistic_and_hinge_formulas():
    scores = torch.tensor([[0.2, -0.3]])
    targets = torch.tensor([[True, False]])
    assert torch.allclose(
        town_objective(scores, targets, kind="logistic"),
        torch.nn.functional.softplus(torch.tensor([[-2.0, -3.0]])),
    )
    assert torch.allclose(
        town_objective(scores, targets, kind="hinge", margin=0.4),
        torch.tensor([[0.2, 0.1]]),
        atol=1e-7,
        rtol=0,
    )
    with pytest.raises(ValueError, match="binary"):
        town_objective(scores, torch.tensor([[0.5, 0.0]]), kind="hinge")


def test_reference_empty_batch_has_frozen_output_shapes():
    lo = torch.tensor([[[-1.0]]])
    hi = torch.tensor([[[1.0]]])
    window_lo = torch.empty((1, 1, 0, 1))
    window_hi = torch.empty_like(window_lo)
    points = torch.empty((0, 1))
    score = reference_town_clearance(points, lo, hi, window_lo, window_hi)
    membership = reference_town_membership(points, lo, hi, window_lo, window_hi)
    assert score.shape == membership.shape == (0, 1)


@pytest.mark.parametrize("domain", ["bounded", "unbounded"])
@pytest.mark.parametrize("kind", ["logistic", "hinge"])
def test_fixed_coordinate_subtractive_recovery_control(domain, kind):
    """Frozen pre-registered H=1,W=1 recovery control; never retune fixture/settings."""
    points = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.float32)
    targets = torch.tensor([[True], [False], [True]])
    start_frame_lower = torch.tensor([[[-0.25]]], dtype=torch.float32)
    start_frame_upper = torch.tensor([[[0.25]]], dtype=torch.float32)
    start_window_lower = torch.tensor([[[[-0.08]]]], dtype=torch.float32)
    start_window_upper = torch.tensor([[[[0.08]]]], dtype=torch.float32)
    start_frame_a, start_frame_b = encode_endpoints(
        start_frame_lower, start_frame_upper, domain=domain, reserve=2e-4
    )
    start_window_a, start_window_b = encode_endpoints(
        start_window_lower, start_window_upper, domain=domain
    )
    geometry = TownGeometry(
        start_frame_a,
        start_frame_b,
        start_window_a,
        start_window_b,
        domain=domain,
    )
    decoded_start = geometry.endpoints()
    for expected, actual in zip(
        (start_frame_lower, start_frame_upper, start_window_lower, start_window_upper),
        decoded_start,
    ):
        assert torch.allclose(expected, actual, atol=2e-5, rtol=0)

    optimizer = torch.optim.Adam(
        geometry.parameters(),
        lr=0.03,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
    )
    initial_scores = geometry.clearance(points)
    initial_loss = town_objective(initial_scores, targets, kind=kind).mean()
    initial_loss.backward()
    initial_gradient_norms = {
        "frame": torch.linalg.vector_norm(
            torch.stack(
                [
                    parameter.grad.norm()
                    for parameter in (geometry.raw_frame_a, geometry.raw_frame_b)
                ]
            )
        ),
        "window": torch.linalg.vector_norm(
            torch.stack(
                [
                    parameter.grad.norm()
                    for parameter in (geometry.raw_window_a, geometry.raw_window_b)
                ]
            )
        ),
    }
    assert all(torch.isfinite(value) and value > 0 for value in initial_gradient_norms.values())
    geometry.zero_grad(set_to_none=True)
    start_coordinates = tuple(value.detach().clone() for value in geometry.endpoints())
    for _ in range(1500):
        optimizer.zero_grad(set_to_none=True)
        loss = town_objective(geometry.clearance(points), targets, kind=kind).mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in geometry.parameters()
        )
        torch.nn.utils.clip_grad_norm_(geometry.parameters(), 1.0)
        optimizer.step()

    (
        final_frame_lower,
        final_frame_upper,
        final_window_lower,
        final_window_upper,
    ) = geometry.endpoints()
    final_scores = geometry.clearance(points)
    final_loss = town_objective(final_scores, targets, kind=kind).mean()
    membership = geometry.membership(points).squeeze(1).tolist()
    changed = any(
        not torch.equal(start, final.detach())
        for start, final in zip(
            start_coordinates,
            (final_frame_lower, final_frame_upper, final_window_lower, final_window_upper),
        )
    )
    print(
        "TOWN_CONTROL",
        {
            "domain": domain,
            "kind": kind,
            "initial_loss": float(initial_loss.detach()),
            "final_loss": float(final_loss.detach()),
            "initial_gradient_norms": {
                name: float(value.detach()) for name, value in initial_gradient_norms.items()
            },
            "membership": membership,
            "frame_lower": final_frame_lower.detach().tolist(),
            "frame_upper": final_frame_upper.detach().tolist(),
            "window_lower": final_window_lower.detach().tolist(),
            "window_upper": final_window_upper.detach().tolist(),
        },
    )
    validate_town_endpoints(
        final_frame_lower, final_frame_upper, domain=domain, epsilon=1e-4, reserve=2e-4
    )
    validate_town_endpoints(final_window_lower, final_window_upper, domain=domain)
    assert torch.isfinite(final_loss)
    assert final_loss < initial_loss
    assert changed
    assert membership == [True, False, True]


def test_single_box_obstruction_on_the_frozen_control_fixture():
    """Every closed interval containing both positive points contains the negative."""
    positives = torch.tensor([-0.5, 0.5])
    negative = torch.tensor(0.0)
    assert positives.min() <= negative <= positives.max()
    for lower in torch.linspace(-1.0, -0.5, 7):
        for upper in torch.linspace(0.5, 1.0, 7):
            if lower <= positives.min() and upper >= positives.max():
                assert lower <= negative <= upper
