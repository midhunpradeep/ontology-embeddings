# Contributing

## Development environment

Use Python 3.11 or 3.12. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The editable install makes the `ontology_embeddings` package available while developing;
`.[dev]` also installs the project test and lint tools.

## Checks

Run the same checks as CI before submitting changes:

```bash
python -m pytest
ruff check .
```

Keep generated datasets, runs, checkpoints, caches, and temporary outputs in the ignored
`artifacts/` locations documented in the README. Keep test fixtures under tracked test paths and
committed prepared dataset examples under tracked `examples/`; neither belongs in generated-output
locations.
