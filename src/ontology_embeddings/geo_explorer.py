"""Optional, read-only Dash shell over validated GEO numeric observations.

Pure run/observation adapters remain importable headlessly. Dash and Plotly are imported only
when ``create_app`` is explicitly called; this module does not load checkpoints or models.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .geo_measurements import BoxEndpoints, NumericObservations
from .run_store import find_runs

__all__ = [
    "create_app",
    "discover_run_metadata",
    "native_box_scene",
    "select_observation",
]


def discover_run_metadata(root: str | Path) -> tuple[dict[str, Any], ...]:
    """Return a read-only, deterministic summary from the existing run-store index."""
    rows = find_runs(root)
    summaries = []
    for row in rows:
        config = row.get("config")
        config = config if isinstance(config, dict) else {}
        summaries.append(
            {
                "run_id": row["run_id"],
                "started_at": row.get("started_at"),
                "status": row.get("status", "unknown"),
                "purpose": row.get("purpose"),
                "family": config.get("model", row.get("model")),
                "dimension": config.get("latent_dim"),
                "domain": config.get("domain"),
                "source_sha": row.get("source_sha"),
                "dataset_id": row.get("dataset_id"),
            }
        )
    return tuple(sorted(summaries, key=lambda item: item["run_id"]))


def select_observation(
    observations: Iterable[NumericObservations], run_id: str, state_id: str
) -> NumericObservations | None:
    """Select exactly one validated committed observation by stable run/state identity."""
    if not run_id or not state_id:
        return None
    matches = []
    for observation in observations:
        if not isinstance(observation, NumericObservations):
            raise TypeError("observation adapter accepts validated NumericObservations only")
        if observation.run_id == run_id and observation.state_id == state_id:
            matches.append(observation)
    if len(matches) > 1:
        raise ValueError("run/state identity is not unique in the observation source")
    return matches[0] if matches else None


def native_box_scene(observation: NumericObservations, class_id: str) -> dict[str, Any]:
    """Build original-space 2D point/primitive data; membership stays producer-authoritative."""
    if not isinstance(observation, NumericObservations) or not isinstance(
        observation.geometry, BoxEndpoints
    ):
        raise ValueError("native 2D scene supports validated BOX observations only")
    if observation.points.shape[1] != 2:
        raise ValueError("native 2D scene requires exactly two latent dimensions; no projection")
    try:
        class_index = observation.class_ids.index(class_id)
    except ValueError as error:
        raise ValueError(f"unknown canonical class ID {class_id!r}") from error
    membership = observation.predictions[:, class_index].detach().cpu().tolist()
    points = observation.points.detach().cpu().tolist()
    geometry = observation.geometry
    lower = geometry.lower[class_index].detach().cpu().tolist()
    upper = geometry.upper[class_index].detach().cpu().tolist()
    return {
        "scope": {
            "run_id": observation.run_id,
            "state_id": observation.state_id,
            "split": observation.split,
            "entity_type": observation.entity_type,
        },
        "class_id": class_id,
        "points": [
            {
                "occurrence_id": occurrence_id,
                "parent_id": parent_id,
                "signature_id": signature_id,
                "coordinates": coordinates,
                "membership": bool(member),
            }
            for occurrence_id, parent_id, signature_id, coordinates, member in zip(
                observation.occurrence_ids,
                observation.parent_ids,
                observation.signature_ids,
                points,
                membership,
                strict=True,
            )
        ],
        "boxes": [
            {"lower": box_lower, "upper": box_upper}
            for box_lower, box_upper in zip(lower, upper, strict=True)
        ],
        "coordinate_space": "original latent space",
    }


def _figure(scene: dict[str, Any], go: Any) -> Any:
    figure = go.Figure()
    for box in scene["boxes"]:
        lo, hi = box["lower"], box["upper"]
        figure.add_shape(
            type="rect",
            x0=lo[0],
            y0=lo[1],
            x1=hi[0],
            y1=hi[1],
            line={"color": "#555", "width": 1},
            fillcolor="rgba(100,100,100,0.08)",
        )
    for member, label, color in ((True, "inside", "#16803c"), (False, "outside", "#c43c39")):
        selected = [point for point in scene["points"] if point["membership"] is member]
        figure.add_trace(
            go.Scatter(
                x=[point["coordinates"][0] for point in selected],
                y=[point["coordinates"][1] for point in selected],
                mode="markers",
                name=label,
                marker={"color": color},
                customdata=[[point["occurrence_id"], point["parent_id"]] for point in selected],
                hovertemplate="occurrence=%{customdata[0]}<br>parent=%{customdata[1]}<extra></extra>",
            )
        )
    figure.update_layout(
        title=f"{scene['class_id']} · {scene['scope']['run_id']} · {scene['scope']['state_id']}",
        xaxis_title="latent dimension 0",
        yaxis_title="latent dimension 1",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        template="plotly_white",
    )
    return figure


def create_app(
    observations: Sequence[NumericObservations] = (), *, run_root: str | Path | None = None
) -> Any:
    """Create the optional local explorer shell over caller-supplied validated observations."""
    try:
        import plotly.graph_objects as go
        from dash import Dash, Input, Output, dcc, html
    except ImportError as error:
        raise RuntimeError(
            "GEO explorer dependencies are optional; install ontology-embeddings[geo-ui]"
        ) from error
    observations = tuple(observations)
    if any(not isinstance(item, NumericObservations) for item in observations):
        raise TypeError("the explorer accepts validated NumericObservations only")
    run_ids = sorted({item.run_id for item in observations})
    first = observations[0] if observations else None
    state_ids = sorted(
        {item.state_id for item in observations if first and item.run_id == first.run_id}
    )
    class_ids = list(first.class_ids) if first else []
    discovered = discover_run_metadata(run_root) if run_root is not None else ()
    app = Dash(__name__)
    app.layout = html.Main(
        [
            html.H2("GEO geometry explorer (read only)"),
            html.P("Exact membership uses original latent coordinates; no projections."),
            html.Pre(json.dumps(discovered, indent=2, default=str)),
            dcc.Dropdown(
                id="geo-run",
                options=[{"label": run_id, "value": run_id} for run_id in run_ids],
                value=run_ids[0] if run_ids else None,
                placeholder="No validated observations supplied",
                clearable=False,
            ),
            dcc.Dropdown(
                id="geo-state",
                options=[{"label": state, "value": state} for state in state_ids],
                value=state_ids[-1] if state_ids else None,
                clearable=False,
            ),
            dcc.Dropdown(
                id="geo-class",
                options=[{"label": class_id, "value": class_id} for class_id in class_ids],
                value=class_ids[0] if class_ids else None,
                clearable=False,
            ),
            dcc.Graph(id="geo-native-2d"),
        ]
    )

    @app.callback(
        Output("geo-state", "options"),
        Output("geo-state", "value"),
        Input("geo-run", "value"),
    )
    def update_states(run_id: str | None):
        states = sorted({item.state_id for item in observations if item.run_id == run_id})
        options = [{"label": state, "value": state} for state in states]
        return options, states[-1] if states else None

    @app.callback(
        Output("geo-native-2d", "figure"),
        Input("geo-run", "value"),
        Input("geo-state", "value"),
        Input("geo-class", "value"),
    )
    def update_figure(run_id: str | None, state_id: str | None, class_id: str | None):
        observation = select_observation(observations, run_id or "", state_id or "")
        if observation is None or class_id not in observation.class_ids:
            return go.Figure().update_layout(title="No validated observation selected")
        try:
            return _figure(native_box_scene(observation, class_id), go)
        except ValueError as error:
            return go.Figure().update_layout(title=str(error))

    return app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="oe-geo", description="Read-only local GEO explorer shell"
    )
    parser.add_argument("--run-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args(argv)
    app = create_app(run_root=args.run_root)
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
