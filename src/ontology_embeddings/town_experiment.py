"""Small validated direct entry point for an integrated TOWN run.

Example: ``python -m ontology_embeddings.town_experiment --dataset PATH --config town.toml``.
"""

from __future__ import annotations

import argparse
import json
import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .dataset.image_data import PreparedDataset, load_prepared_dataset
from .town_training import train_town


@dataclass(frozen=True, slots=True)
class TownExperimentConfig:
    domain: str = "unbounded"
    houses: int = 1
    windows: int = 1
    latent_dim: int = 16
    epsilon: float = 1e-4
    delta: float = 1e-4
    objective: str = "logistic"
    tau: float = 0.1
    margin: float = 0.1
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 0.02
    seed: int = 23
    device: str = "cpu"
    point_chunk_size: int | None = None

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "TownExperimentConfig":
        if not isinstance(values, dict):
            raise ValueError("TOWN config must be a TOML table")
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown TOWN experiment settings: {unknown}")
        result = cls(**values)
        result.validate()
        return result

    @classmethod
    def from_toml(cls, path: Path | str) -> "TownExperimentConfig":
        try:
            payload = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise ValueError(f"cannot read TOWN experiment config: {error}") from error
        if set(payload) != {"town"} or not isinstance(payload.get("town"), dict):
            raise ValueError("TOWN config must contain exactly one [town] table")
        return cls.from_mapping(payload["town"])

    def validate(self) -> None:
        if not isinstance(self.domain, str) or self.domain not in ("bounded", "unbounded"):
            raise ValueError("domain must be bounded or unbounded")
        for name in ("houses", "latent_dim", "epochs", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.windows, bool) or not isinstance(self.windows, int) or self.windows < 0:
            raise ValueError("windows must be a nonnegative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(self.objective, str) or self.objective not in ("logistic", "hinge"):
            raise ValueError("objective must be logistic or hinge")
        for name in ("epsilon", "delta", "tau", "learning_rate"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.margin, bool)
            or not isinstance(self.margin, (int, float))
            or not math.isfinite(self.margin)
            or self.margin < 0
        ):
            raise ValueError("margin must be finite and nonnegative")
        if self.domain == "bounded" and self.epsilon + (2 * self.delta if self.windows else 0) >= 2:
            raise ValueError("bounded TOWN frame width floor is not representable")
        if not isinstance(self.device, str) or (
            self.device != "cpu" and not self.device.startswith("cuda")
        ):
            raise ValueError("device must be 'cpu' or a CUDA device")
        if self.point_chunk_size is not None and (
            isinstance(self.point_chunk_size, bool)
            or not isinstance(self.point_chunk_size, int)
            or self.point_chunk_size < 1
        ):
            raise ValueError("point_chunk_size must be a positive integer or null")

    def resolved_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def run_town_experiment(
    dataset: PreparedDataset,
    config: TownExperimentConfig,
    *,
    artifact_root: Path | str = "artifacts/runs",
    initial_box_checkpoint: Path | str | None = None,
):
    config.validate()
    return train_town(
        dataset,
        dataset.split_records("train"),
        dataset.split_records("validation"),
        artifact_root=artifact_root,
        purpose="town-experiment",
        invocation="python -m ontology_embeddings.town_experiment",
        seed=config.seed,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        latent_dim=config.latent_dim,
        houses=config.houses,
        windows=config.windows,
        domain=config.domain,
        epsilon=config.epsilon,
        delta=config.delta,
        objective=config.objective,
        tau=config.tau,
        margin=config.margin,
        device=config.device,
        point_chunk_size=config.point_chunk_size,
        initial_box_checkpoint=initial_box_checkpoint,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oe-town", description="Train an integrated metadata-oracle TOWN classifier."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--initial-box-checkpoint", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = TownExperimentConfig.from_toml(args.config)
        dataset = load_prepared_dataset(args.dataset)
        result = run_town_experiment(
            dataset, config, artifact_root=args.artifact_root,
            initial_box_checkpoint=args.initial_box_checkpoint,
        )
        print(json.dumps({
            "run_id": result["run_id"], "run_path": result["run_path"],
            "status": result["status"], "accepted_updates": result["accepted_updates"],
            "runtime_seconds": result["runtime_seconds"],
            "checkpoint_paths": result["checkpoint_paths"],
        }, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
