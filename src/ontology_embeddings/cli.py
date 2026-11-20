"""Dataset preparation, validation, and inspection commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_CONFIG_PATH, DEFAULT_PIZZA_TYPES, load_config
from .dataset.preparation import prepare_dataset
from .dataset.validation import validate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oe-dataset",
        description="Generate, validate, and inspect prepared Pizzaïolo datasets.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="Generate and prepare a new immutable dataset")
    generate.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--total-instances", type=int)
    generate.add_argument("--generation-seed", type=int)
    generate.add_argument("--split-seed", type=int)
    generate.add_argument("--pizza-types", nargs="+", choices=DEFAULT_PIZZA_TYPES)
    generate.add_argument(
        "--type-counts", help='JSON map, for example {"American": 3, "random": 3}'
    )
    generate.add_argument("--train-size", type=float)
    generate.add_argument("--valid-size", type=float)
    generate.add_argument("--test-size", type=float)
    generate.add_argument(
        "--source-repo", type=Path, help="Clean pinned upstream checkout (read-only)"
    )

    validate = commands.add_parser("validate", help="Run dataset integrity and semantic validation")
    validate.add_argument("dataset", type=Path)
    validate.add_argument("--no-write-report", action="store_true")
    validate.add_argument(
        "--json", action="store_true", help="Print machine-readable validation JSON"
    )

    inspect = commands.add_parser("inspect", help="Print dataset provenance and support summary")
    inspect.add_argument("dataset", type=Path)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            type_counts = None
            if args.type_counts is not None:
                type_counts = json.loads(args.type_counts)
                if not isinstance(type_counts, dict) or any(
                    type(value) is not int for value in type_counts.values()
                ):
                    raise ValueError("--type-counts must be a JSON object of integer counts")
            config = load_config(
                args.config,
                total_instances=args.total_instances,
                generation_seed=args.generation_seed,
                split_seed=args.split_seed,
                train_size=args.train_size,
                valid_size=args.valid_size,
                test_size=args.test_size,
                pizza_types=tuple(args.pizza_types) if args.pizza_types else None,
                type_counts=type_counts,
            )
            print("Resolved generation settings:")
            print(json.dumps(config.resolved_dict(), indent=2, sort_keys=True))
            root = Path(__file__).resolve().parents[2]
            output = prepare_dataset(
                config,
                args.output,
                project_root=root,
                source_repo=args.source_repo,
                progress=lambda fraction, message: print(
                    f"[{fraction:5.1%}] {message}", flush=True
                ),
            )
            print(f"Prepared dataset at {output}")
            return 0

        if args.command in {"validate", "inspect"}:
            write_report = args.command == "validate" and not args.no_write_report
            report = validate_dataset(args.dataset, write_report=write_report)
            payload = report.to_dict()
            if args.command == "validate":
                if args.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    print(f"Validation: {'PASS' if report.valid else 'FAIL'}")
                    print(f"Dataset: {report.dataset_id or 'unknown'}")
                    print(f"Errors: {len(report.errors)} | Warnings: {len(report.warnings)}")
                    for issue in report.errors:
                        print(f"ERROR [{issue.code}] {issue.message}")
                return 0 if report.valid else 1

            print(
                json.dumps(
                    {
                        "dataset_id": report.dataset_id,
                        "dataset_path": report.dataset_path,
                        "valid": report.valid,
                        "statistics": report.statistics,
                        "source": json.loads(
                            (args.dataset / "manifest.json").read_text(encoding="utf-8")
                        ).get("source", {}),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0 if report.valid else 1

        parser.error(f"Unknown command {args.command!r}")
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        RuntimeError,
        ValueError,
        OSError,
    ) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
