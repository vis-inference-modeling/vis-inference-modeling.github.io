"""
Generate Vega-Lite / Altair scatter plots for each combined
tagged comments file in `extraction_pipeline/tagged_comments_combined/`.

For each JSON file:
- Construct a table with:
    - `presence_tag`: the sentence key from `presence_tags`
    - `presence`: "present" or "not_present"
    - `comment_index`: the integer index for the comment
    - `article_id`: the article id
- Encode each unique `presence_tag` using a sentence transformer.
- Run UMAP to reduce embeddings to 2D.
- Cluster unique presence tags in embedding space using HDBSCAN.
- Plot all points in a single scatter plot with:
    - x, y: the UMAP dimensions
    - color: HDBSCAN cluster (or presence)

Outputs:
- One Vega-Lite spec JSON file per input JSON file.
- Optionally, one HTML file per input JSON file (can be opened in a browser).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import altair as alt
import hdbscan
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import umap


BASE_DIR = Path(__file__).resolve().parent
TAGGED_COMBINED_DIR = BASE_DIR / "tagged_comments_combined"
OUTPUT_DIR = BASE_DIR / "visualizations" / "tagged_comments_scatter"


@dataclass
class PresenceTagRecord:
    article_id: str
    comment_index: int
    presence_tag: str
    presence: str


def load_combined_file(path: Path) -> List[PresenceTagRecord]:
    """Load a single combined tagged comments JSON file and
    convert it into a list of PresenceTagRecord rows.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    records: List[PresenceTagRecord] = []
    for entry in data:
        article_id = entry.get("article_id", "")
        comment_index = entry.get("comment_index")
        presence_tags: Dict[str, Any] = entry.get("presence_tags", {})

        if comment_index is None:
            # Skip malformed entries with no comment index
            continue

        for tag_text, status in presence_tags.items():
            # Support both legacy (str) and new format (dict with "presence", "task_type")
            presence = (
                status.get("presence", "not_present")
                if isinstance(status, dict)
                else str(status)
            )
            records.append(
                PresenceTagRecord(
                    article_id=str(article_id),
                    comment_index=int(comment_index),
                    presence_tag=str(tag_text),
                    presence=str(presence),
                )
            )

    return records


def build_dataframe(records: List[PresenceTagRecord]) -> pd.DataFrame:
    """Convert PresenceTagRecord list to a DataFrame suitable for Altair."""
    return pd.DataFrame(
        [
            {
                "article_id": r.article_id,
                "comment_index": r.comment_index,
                "presence_tag": r.presence_tag,
                "presence": r.presence,
            }
            for r in records
        ]
    )


def compute_embeddings(
    texts: List[str],
    model: SentenceTransformer,
) -> np.ndarray:
    """Encode a list of texts into embeddings using a sentence transformer."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    # Batched encoding for efficiency
    embeddings = model.encode(texts, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)


def run_umap(embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Reduce embeddings to 2D using UMAP."""
    if embeddings.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    reducer = umap.UMAP(n_components=2, random_state=random_state)
    return reducer.fit_transform(embeddings).astype(np.float32)


def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 2,
    min_samples: int | None = None,
) -> np.ndarray:
    """Cluster embeddings using HDBSCAN. Returns integer labels (-1 = noise).
    Uses Euclidean distance on L2-normalized vectors (equivalent to cosine for clustering).
    """
    if embeddings.size == 0 or len(embeddings) < 2:
        return np.array([], dtype=np.intp)

    # L2-normalize so Euclidean distance matches cosine distance (BallTree has no 'cosine')
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    X = (embeddings / norms).astype(np.float32)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(X)
    return np.asarray(labels, dtype=np.intp)


def add_umap_coordinates_to_df(
    df: pd.DataFrame,
    coords: Dict[str, Tuple[float, float]],
    tag_to_cluster: Dict[str, int] | None = None,
) -> pd.DataFrame:
    """Add UMAP x, y coordinates and optional cluster labels to the DataFrame based on presence_tag."""
    xs: List[float] = []
    ys: List[float] = []
    for tag in df["presence_tag"]:
        x, y = coords.get(tag, (0.0, 0.0))
        xs.append(x)
        ys.append(y)

    df = df.copy()
    df["x"] = xs
    df["y"] = ys

    if tag_to_cluster is not None:
        # -1 = noise, others = cluster id; store as string for legend
        clusters: List[str] = []
        for tag in df["presence_tag"]:
            label = tag_to_cluster.get(tag, -1)
            clusters.append("noise" if label == -1 else str(label))
        df["cluster"] = clusters

    return df


def build_chart(df: pd.DataFrame, title: str, color_by_cluster: bool = True) -> alt.Chart:
    """Build a single Altair scatter plot with UMAP coordinates, colored by cluster or presence."""
    alt.data_transformers.disable_max_rows()

    df = df[df["presence"].isin(["present", "not_present"])].copy()
    if df.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []}), title=title).mark_circle()

    if color_by_cluster and "cluster" in df.columns:
        color_enc = alt.Color("cluster:N", title="Cluster")
        tooltip = [
            alt.Tooltip("presence_tag:N", title="Presence Tag"),
            alt.Tooltip("cluster:N", title="Cluster"),
            alt.Tooltip("presence:N", title="Presence"),
            alt.Tooltip("comment_index:N", title="Comment Index"),
            alt.Tooltip("article_id:N", title="Article ID"),
        ]
    else:
        color_enc = alt.Color(
            "presence:N", title="Presence", scale=alt.Scale(domain=["present", "not_present"])
        )
        tooltip = [
            alt.Tooltip("presence_tag:N", title="Presence Tag"),
            alt.Tooltip("presence:N", title="Presence"),
            alt.Tooltip("comment_index:N", title="Comment Index"),
            alt.Tooltip("article_id:N", title="Article ID"),
        ]

    chart = (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(
            x=alt.X("x:Q", title="UMAP 1"),
            y=alt.Y("y:Q", title="UMAP 2"),
            color=color_enc,
            tooltip=tooltip,
        )
        .properties(width=500, height=500, title=title)
    )
    return chart


def process_single_file(path: Path, model: SentenceTransformer) -> None:
    """Process a single combined JSON file and write Vega-Lite spec + HTML."""
    print(f"Processing {path.name}...")
    records = load_combined_file(path)
    if not records:
        print(f"  No usable records in {path.name}, skipping.")
        return

    df = build_dataframe(records)

    # Only keep records where presence is "present" or "not_present"
    df = df[df["presence"].isin(["present", "not_present"])].copy()
    if df.empty:
        print(f"  No 'present'/'not_present' records in {path.name}, skipping.")
        return

    # Encode unique presence tags, cluster with HDBSCAN, and run UMAP
    unique_tags = sorted(df["presence_tag"].unique())
    embeddings = compute_embeddings(unique_tags, model)
    cluster_labels = run_hdbscan(embeddings, min_cluster_size=2)
    umap_coords = run_umap(embeddings)

    tag_to_xy: Dict[str, Tuple[float, float]] = {
        tag: (float(x), float(y))
        for tag, (x, y) in zip(unique_tags, umap_coords)
    }
    tag_to_cluster: Dict[str, int] = {
        tag: int(label) for tag, label in zip(unique_tags, cluster_labels)
    }

    df_with_coords = add_umap_coordinates_to_df(df, tag_to_xy, tag_to_cluster)

    # Build Altair chart
    title = f"Presence Tags UMAP Scatter - {path.stem}"
    chart = build_chart(df_with_coords, title=title)

    # Prepare output paths
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_name = path.stem  # e.g. "123"
    json_out = OUTPUT_DIR / f"{base_name}_scatter.json"
    html_out = OUTPUT_DIR / f"{base_name}_scatter.html"

    # Save Vega-Lite spec as JSON
    spec = chart.to_dict()
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)

    # Save a simple self-contained HTML for convenience
    try:
        chart.save(str(html_out))
    except Exception as exc:  # noqa: BLE001
        # If HTML saving fails (e.g., altair_saver not installed),
        # we at least have the JSON spec.
        print(f"  Warning: could not save HTML for {path.name}: {exc}")

    print(f"  Wrote {json_out} and {html_out}")


def main() -> None:
    if not TAGGED_COMBINED_DIR.exists():
        raise SystemExit(f"Directory not found: {TAGGED_COMBINED_DIR}")

    combined_files = sorted(
        p for p in TAGGED_COMBINED_DIR.glob("*.json") if p.is_file()
    )
    if not combined_files:
        raise SystemExit(f"No JSON files found in {TAGGED_COMBINED_DIR}")

    # Load sentence transformer model once and reuse for all files
    print("Loading sentence transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    for path in combined_files:
        process_single_file(path, model)


if __name__ == "__main__":
    main()

