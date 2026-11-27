from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from ontology_embeddings.box import box_objective
from ontology_embeddings.shared_training import (
    checked_optimizer_update,
    masked_geometric_task_loss,
)


def _legacy_entity_mean(scores, targets, known, applicable, *, kind, tau, margin):
    eligible = known.bool() & applicable.bool()
    losses = []
    for row in range(scores.shape[0]):
        selected = eligible[row]
        if torch.any(selected):
            losses.append(
                box_objective(
                    scores[row, selected].unsqueeze(0),
                    targets[row, selected].unsqueeze(0),
                    kind=kind,
                    tau=tau,
                    margin=margin,
                ).mean()
            )
    return torch.stack(losses).mean() if losses else None


def _legacy_task_loss(ps, pt, pk, pa, cs, ct, ck, ca, parents, *, kind, tau, margin):
    p = _legacy_entity_mean(ps, pt, pk, pa, kind=kind, tau=tau, margin=margin)
    component_rows = []
    for row, parent in enumerate(parents):
        value = _legacy_entity_mean(
            cs[row : row + 1],
            ct[row : row + 1],
            ck[row : row + 1],
            ca[row : row + 1],
            kind=kind,
            tau=tau,
            margin=margin,
        )
        component_rows.append((parent, value))
    parent_means = [
        torch.stack(
            [value for owner, value in component_rows if owner == parent and value is not None]
        ).mean()
        for parent in range(ps.shape[0])
        if any(owner == parent and value is not None for owner, value in component_rows)
    ]
    c = torch.stack(parent_means).mean() if parent_means else None
    available = [value for value in (p, c) if value is not None]
    return (torch.stack(available).mean() if available else None), {"pizza": p, "component": c}


@pytest.mark.parametrize("kind", ["logistic", "hinge"])
def test_vectorized_geometric_reduction_matches_box_reference_values_and_gradients(kind):
    ps = torch.tensor(
        [[0.4, -0.1, float("nan")], [-0.3, 0.2, 0.8], [9.0, -9.0, 0.0]], requires_grad=True
    )
    pt = torch.tensor([[1.0, 0.0, float("nan")], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    pk = torch.tensor([[1, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=torch.bool)
    pa = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.bool)
    cs = torch.tensor(
        [
            [0.1, -0.5, float("nan")],
            [0.2, 0.3, 0.4],
            [-0.7, 0.6, 0.1],
            [1.0, -1.0, 2.0],
            [0.0, 0.0, 0.0],
        ],
        requires_grad=True,
    )
    ct = torch.tensor(
        [
            [1.0, 0.0, float("nan")],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    ck = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=torch.bool)
    ca = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.bool)
    parents = [0, 0, 2, 2, 2]
    args = dict(kind=kind, tau=0.1, margin=0.1)
    actual, actual_tasks = masked_geometric_task_loss(
        ps, pt, pk, pa, cs, ct, ck, ca, parents, **args
    )
    expected, expected_tasks = _legacy_task_loss(ps, pt, pk, pa, cs, ct, ck, ca, parents, **args)
    assert actual is not None and expected is not None
    assert torch.allclose(actual, expected, atol=1e-7, rtol=0)
    for name in ("pizza", "component"):
        assert torch.allclose(actual_tasks[name], expected_tasks[name], atol=1e-7, rtol=0)
    actual.backward()
    actual_p_grad, actual_c_grad = ps.grad.clone(), cs.grad.clone()
    ps.grad = None
    cs.grad = None
    expected.backward()
    assert torch.allclose(actual_p_grad, ps.grad, atol=1e-7, rtol=0)
    assert torch.allclose(actual_c_grad, cs.grad, atol=1e-7, rtol=0)
    assert torch.equal(actual_p_grad[~(pk & pa)], torch.zeros_like(actual_p_grad[~(pk & pa)]))
    assert torch.equal(actual_c_grad[~(ck & ca)], torch.zeros_like(actual_c_grad[~(ck & ca)]))


def test_geometric_reduction_preserves_undefined_tasks_and_no_supervision():
    scores = torch.empty((0, 3), dtype=torch.float32, requires_grad=True)
    targets = torch.empty((0, 3), dtype=torch.float32)
    masks = torch.empty((0, 3), dtype=torch.bool)
    p_scores = torch.tensor([[0.5, -0.5]], requires_grad=True)
    p_targets = torch.tensor([[1.0, 0.0]])
    p_mask = torch.tensor([[True, True]])
    total, tasks = masked_geometric_task_loss(
        p_scores, p_targets, p_mask, p_mask, scores, targets, masks, masks, [], kind="logistic"
    )
    assert total is not None and tasks["pizza"] is not None and tasks["component"] is None
    none, undefined = masked_geometric_task_loss(
        p_scores,
        p_targets,
        torch.zeros_like(p_mask),
        p_mask,
        scores,
        targets,
        masks,
        masks,
        [],
        kind="logistic",
    )
    assert none is None and undefined == {"pizza": None, "component": None}


def _legacy_step(model, optimizer, loss):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    optimizer.step()


def test_checked_update_matches_existing_clipped_adam_step_and_optimizer_state():
    torch.manual_seed(22)
    reference = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 1))
    candidate = copy.deepcopy(reference)
    ref_opt = torch.optim.Adam(reference.parameters(), lr=0.02, eps=1e-8)
    got_opt = torch.optim.Adam(candidate.parameters(), lr=0.02, eps=1e-8)
    x = torch.tensor([[1.0, -2.0, 0.5, 1.5], [0.2, 0.3, -0.7, 2.0]])
    y = torch.tensor([[1.0], [-0.5]])
    ref_loss = (reference(x) - y).square().mean()
    got_loss = (candidate(x) - y).square().mean()
    _legacy_step(reference, ref_opt, ref_loss)
    stats = checked_optimizer_update(
        candidate,
        got_opt,
        got_loss,
        update_number=1,
        gradient_groups={
            "encoder": list(candidate[0].parameters()),
            "geometry": list(candidate[1].parameters()),
        },
        max_grad_norm=0.25,
    )
    assert stats["global_pre_clip_norm"] > 0.25
    assert stats["global_post_clip_norm"] == pytest.approx(0.25, abs=1e-6)
    for ref, got in zip(reference.parameters(), candidate.parameters(), strict=True):
        assert torch.equal(ref, got)
    for ref, got in zip(reference.parameters(), candidate.parameters(), strict=True):
        ref_state, got_state = ref_opt.state[ref], got_opt.state[got]
        for name in ref_state:
            if isinstance(ref_state[name], torch.Tensor):
                assert torch.equal(ref_state[name], got_state[name])
            else:
                assert ref_state[name] == got_state[name]


def test_checked_update_rejects_nonfinite_loss_without_stepping():
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss = model(torch.ones((1, 2))).sum() * torch.tensor(float("nan"))
    with pytest.raises(FloatingPointError, match="invalid loss"):
        checked_optimizer_update(
            model,
            optimizer,
            loss,
            update_number=3,
            gradient_groups={"encoder": list(model.parameters())},
            max_grad_norm=1.0,
        )
    assert all(torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True))
