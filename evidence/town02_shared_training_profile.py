"""Reproducible synthetic representative BOX training profile for TOWN-02.

Run from repository root with the project environment:
  PYTHONPATH=src .venv/bin/python \\
    evidence/town02_shared_training_profile.py --phase pre \\
    --output evidence/town02_profile_pre.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from ontology_embeddings.box_training import (
    BoxMultiLabelModel,
    _box_loss,
    _geometry_diagnostics,
    _invalid_optimizer_groups,
    _signature_diagnostics,
    _task_metrics,
    masked_box_task_loss,
)
from ontology_embeddings.dataset.schema import ComponentRecord, DatasetRecord
from ontology_embeddings.features import COMPONENT_FEATURE_NAMES, PIZZA_FEATURE_NAMES
from ontology_embeddings.training import _pack


def fixture():
    rng = torch.Generator().manual_seed(20261127)
    n_classes, pizzas, components_per_pizza = 60, 64, 4
    records = []
    for p in range(pizzas + 16):
        split = "train" if p < pizzas else "validation"
        targets = torch.randint(0, 2, (66,), generator=rng).tolist()
        known = (torch.rand((66,), generator=rng) > 0.25).int().tolist()
        components = []
        for c in range(components_per_pizza):
            ct = torch.randint(0, 2, (66,), generator=rng).tolist()
            ck = (torch.rand((66,), generator=rng) > 0.3).int().tolist()
            components.append(
                ComponentRecord(
                    f"profile-{p}:{c}",
                    1,
                    "AnchovyTopping",
                    (0, 0, 1, 1),
                    tuple(ct),
                    tuple(ck),
                    "synthetic-profile",
                )
            )
        counts = {
            name.split(":", 1)[1]: int((p + i) % 4 == 0)
            for i, name in enumerate(PIZZA_FEATURE_NAMES[:16])
        }
        raw = {"topping_counts": counts, "has_base": "ThinAndCrispyBase", "country_of_origin": None}
        records.append(
            DatasetRecord(
                f"profile-{p}",
                split,
                "synthetic",
                "",
                1,
                1,
                tuple(targets),
                tuple(known),
                (),
                tuple(components),
                {"raw_generator": raw},
                {},
            )
        )
    dataset = SimpleNamespace(
        dataset_id="town02-profile-synthetic",
        dataset_manifest_sha256="synthetic",
        schema_version="v1",
        class_manifest_sha256="synthetic-vocab",
        canonical_class_ids=tuple(range(n_classes)),
        class_names=tuple(f"C{i}" for i in range(n_classes)),
        class_iris=tuple(f"urn:C{i}" for i in range(n_classes)),
        pizza_supervision_class_ids=tuple(range(45)),
        component_supervision_class_ids=tuple(range(35)),
        fixed_positive_class_ids=(),
        unsupported_class_ids=(),
    )
    return dataset, records[:pizzas], records[pizzas:]


def measure(fn, warmup=2, repeats=7):
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        values.append((time.perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(values), "samples_ms": values}


def main():
    invocation_start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pre", "post"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261127)
    dataset, train, validation = fixture()
    device = torch.device("cpu")
    model = BoxMultiLabelModel(
        len(PIZZA_FEATURE_NAMES),
        len(COMPONENT_FEATURE_NAMES),
        dataset.canonical_class_ids,
        dataset.class_manifest_sha256,
        latent_dim=16,
        boxes=1,
        domain="bounded",
        seed=20261127,
        class_names=dataset.class_names,
        class_iris=dataset.class_iris,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    packed = _pack(train, dataset, device)
    x_p, y_p, m_p, x_c, y_c, m_c, parents = packed
    pa = torch.tensor(
        [i in set(dataset.pizza_supervision_class_ids) for i in dataset.canonical_class_ids],
        dtype=torch.bool,
    ).expand_as(y_p)
    ca = torch.tensor(
        [i in set(dataset.component_supervision_class_ids) for i in dataset.canonical_class_ids],
        dtype=torch.bool,
    ).expand_as(y_c)

    def encode_fn():
        return model.encode(x_p, "pizza"), model.encode(x_c, "component")

    def score_backward():
        optimizer.zero_grad(set_to_none=True)
        scores = model.clearances(x_p, "pizza").sum() + model.clearances(x_c, "component").sum()
        scores.backward()

    score_p = model.clearances(x_p, "pizza").detach().requires_grad_()
    score_c = model.clearances(x_c, "component").detach().requires_grad_()

    def masked_backward():
        optimizer.zero_grad(set_to_none=True)
        loss, _ = masked_box_task_loss(
            score_p,
            y_p,
            m_p,
            pa,
            score_c,
            y_c,
            m_c,
            ca,
            parents,
            kind="logistic",
            tau=0.1,
            margin=0.1,
        )
        loss.backward()

    def evaluate():
        with torch.no_grad():
            _, _, batch = _box_loss(
                model, validation, dataset, device, kind="logistic", tau=0.1, margin=0.1
            )
            _task_metrics(batch, dataset.class_names)
            _signature_diagnostics(train, validation, batch, dataset.class_names)
            _geometry_diagnostics(model, train, dataset, device)

    def optimizer_checks():
        _invalid_optimizer_groups(optimizer)

    def checkpoint():
        with tempfile.TemporaryDirectory() as tmp:
            model.save_checkpoint(Path(tmp) / "profile.pt")

    source_files = (
        "src/ontology_embeddings/model.py",
        "src/ontology_embeddings/training.py",
        "src/ontology_embeddings/box_training.py",
        "src/ontology_embeddings/box.py",
        "src/ontology_embeddings/shared_training.py",
    )
    result = {
        "phase": args.phase,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            if Path(name).is_file()
            else None
            for name in source_files
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "cuda_available": torch.cuda.is_available(),
        "threads": torch.get_num_threads(),
        "batch_shape": {
            "pizza": list(x_p.shape),
            "components": list(x_c.shape),
            "classes": len(dataset.canonical_class_ids),
            "latent_dim": 16,
            "boxes": 1,
            "points_scored_for_train": int(len(x_p) + len(x_c)),
        },
        "stages": {
            "pack_records": measure(lambda: _pack(train, dataset, device)),
            "encoding": measure(encode_fn),
            "decode_score_backward": measure(score_backward),
            "masked_loss_forward_backward": measure(masked_backward),
            "evaluation_diagnostics": measure(evaluate, warmup=1, repeats=3),
            "optimizer_state_checks": measure(optimizer_checks),
            "atomic_checkpoint_persistence": measure(checkpoint, warmup=1, repeats=3),
            "complete_loss_with_predictions": measure(
                lambda: _box_loss(
                    model, train, dataset, device, kind="logistic", tau=0.1, margin=0.1
                )
            ),
        },
        "memory": {
            "peak_rss_kib_process_cumulative": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else None,
            "note": (
                "CPU-only environment; GPU peak allocation unavailable; RSS is process "
                "cumulative high-water mark, not stage-isolated."
            ),
        },
        "method": (
            "Synthetic fixed-seed 64-pizza batch, 4 component occurrences per pizza, "
            "60 trainable classes, bounded BOX H=1, latent dimension 16; medians over "
            "repeated warmed runs. No real dataset used."
        ),
        "invocation_elapsed_seconds": time.perf_counter() - invocation_start,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "phase": result["phase"],
                "revision": result["revision"],
                "batch_shape": result["batch_shape"],
                "stage_medians_ms": {k: v["median_ms"] for k, v in result["stages"].items()},
                "memory": result["memory"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
