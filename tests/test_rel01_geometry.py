import numpy as np
import pytest

from ontology_embeddings.relation_geometry import directed, unordered, validate_box


def test_identical_and_strict_containment_have_exact_closed_semantics():
    lo = np.array([0.0, 0.0])
    hi = np.array([2.0, 1.0])
    same = directed(lo, hi, lo, hi)
    assert same["contained"] and same["relative_containment_volume"] == 1
    inner = directed(np.array([0.5, 0.25]), np.array([1.5, 0.75]), lo, hi)
    assert inner["contained"] and inner["max_violation"] == 0
    assert inner["relative_containment_volume"] == 1
    partial = directed(lo, hi, np.array([0.5, 0.25]), np.array([1.5, 0.75]))
    assert not partial["contained"] and partial["relative_containment_volume"] == 0.25


def test_partial_overlap_and_disjoint_have_expected_measurements():
    a = unordered(
        np.array([0.0, 0.0]), np.array([2.0, 2.0]), np.array([1.0, 1.0]), np.array([3.0, 3.0])
    )
    assert a["intersection_status"] == "positive_volume"
    assert a["intersection_dimensionality"] == 2
    assert a["intersection_volume"] == 1
    assert a["intersection_over_union"] == pytest.approx(1 / 7)
    b = unordered(
        np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([2.0, 0.0]), np.array([3.0, 1.0])
    )
    assert b["intersection_status"] == "empty"
    assert b["intersection_dimensionality"] is None


@pytest.mark.parametrize(
    "lower,upper,dimension",
    [
        ([1.0, 0.0], [2.0, 1.0], 1),
        ([1.0, 1.0], [2.0, 2.0], 0),
        ([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], 0),
    ],
)
def test_face_edge_corner_contact_is_nonempty(lower, upper, dimension):
    a = unordered(np.zeros(len(lower)), np.ones(len(lower)), np.array(lower), np.array(upper))
    assert a["intersection_status"] == "lower_dimensional"
    assert a["intersection_dimensionality"] == dimension
    assert a["intersection_volume"] == 0


def test_exact_one_axis_containment_violation_and_witness_axis():
    lo = np.array([0.0, 0.0])
    hi = np.array([1.0, 1.0])
    b_lo = np.array([0.0, 0.0])
    b_hi = np.array([1.0, np.nextafter(1.0, 0.0)])
    r = directed(lo, hi, b_lo, b_hi)
    assert not r["contained"]
    assert r["violating_dimensions"] == 1
    assert r["witness"]["axis"] == 1
    assert r["witness"]["coordinates"] == [1.0, 1.0]


def test_log_volume_ratios_remain_defined_when_volume_overflows():
    lower = np.zeros(16)
    upper = np.full(16, 1e30)
    result = unordered(lower, upper, lower, upper)
    assert result["intersection_volume"] is None
    assert result["fraction_of_a"] == 1
    assert result["fraction_of_b"] == 1
    assert result["intersection_over_union"] == 1


def test_invalid_geometry_rejected_and_float64_promoted():
    with pytest.raises(ValueError):
        validate_box([0.0, 0.0], [1.0, float("nan")], domain="unbounded", epsilon=1e-4)
    lo, hi = validate_box(
        np.array([0], dtype=np.float32),
        np.array([1], dtype=np.float32),
        domain="bounded",
        epsilon=1e-4,
    )
    assert lo.dtype == hi.dtype == np.float64
