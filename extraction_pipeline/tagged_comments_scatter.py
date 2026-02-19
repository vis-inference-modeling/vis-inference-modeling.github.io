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
- Cluster the embeddings using cosine similarity (via normalized KMeans).
- Plot the 2D coordinates in a scatter plot, using:
    - x, y: the UMAP dimensions
    - color: cluster label
- Facet the scatter plot into separate panels for
  "present" and "not_present" presence values.

Outputs:
- One Vega-Lite spec JSON file per input JSON file.
- Optionally, one HTML file per input JSON file (can be opened in a browser).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import altair as alt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
from sklearn.cluster import KMeans


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


def compute_clusters(
    embeddings: np.ndarray,
    min_clusters: int = 2,
    max_clusters: int = 10,
) -> np.ndarray:
    """Cluster embeddings using cosine similarity (via L2-normalized KMeans).

    Returns an array of integer cluster labels, one per embedding row.
    """
    if embeddings.size == 0:
        return np.empty((0,), dtype=int)

    n_samples = embeddings.shape[0]
    if n_samples <= 1:
        # Single point -> single cluster label 0
        return np.zeros((n_samples,), dtype=int)

    # Choose a reasonable number of clusters based on data size
    n_clusters = int(np.sqrt(n_samples))
    n_clusters = max(min_clusters, min(max_clusters, n_clusters))

    # L2-normalize so Euclidean distance corresponds to cosine distance
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0.0] = 1.0
    normalized = embeddings / norms

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(normalized)
    return labels.astype(int)


def add_umap_coordinates_to_df(df: pd.DataFrame, coords: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """Add UMAP x, y coordinates to the DataFrame based on presence_tag."""
    xs: List[float] = []
    ys: List[float] = []
    for tag in df["presence_tag"]:
        x, y = coords.get(tag, (0.0, 0.0))
        xs.append(x)
        ys.append(y)

    df = df.copy()
    df["x"] = xs
    df["y"] = ys
    return df


def build_chart(df: pd.DataFrame, title: str) -> alt.Chart:
    """Build an Altair scatter plot with Vega-Lite spec."""
    # Ensure Altair uses a data transformer that allows larger data sets
    alt.data_transformers.disable_max_rows()

    # Only keep the two presence categories of interest
    df = df[df["presence"].isin(["present", "not_present"])].copy()
    if df.empty:
        # Nothing to plot; return an empty chart with the title.
        return alt.Chart(pd.DataFrame({"x": [], "y": []}), title=title).mark_circle()

    base = (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(
            x=alt.X("x:Q", title="UMAP 1"),
            y=alt.Y("y:Q", title="UMAP 2"),
            # Color by cluster label derived from cosine-similarity-based clustering
            color=alt.Color("cluster:N", title="Cluster"),
            # Use opacity to help with dense/overlapping points; make "present"
            # points slightly more opaque than "not_present".
            opacity=alt.condition(
                alt.datum.presence == "present",
                alt.value(0.9),
                alt.value(0.4),
            ),
            tooltip=[
                alt.Tooltip("presence_tag:N", title="Presence Tag"),
                alt.Tooltip("presence:N", title="Presence"),
                alt.Tooltip("cluster:N", title="Cluster"),
                alt.Tooltip("comment_index:N", title="Comment Index"),
                alt.Tooltip("article_id:N", title="Article ID"),
            ],
        )
        .properties(width=300, height=300)
    )

    # Facet into separate small multiples for "present" vs "not_present"
    chart = base.facet(column=alt.Column("presence:N", title="Presence"))
    chart = chart.properties(title=title)
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

    # Encode unique presence tags for this file
    unique_tags = sorted(df["presence_tag"].unique())
    embeddings = compute_embeddings(unique_tags, model)
    umap_coords = run_umap(embeddings)

    # Cluster embeddings using cosine similarity (via normalized KMeans)
    cluster_labels = compute_clusters(embeddings)

    # Map from tag -> (x, y) and tag -> cluster label
    tag_to_xy: Dict[str, Tuple[float, float]] = {
        tag: (float(x), float(y))
        for tag, (x, y) in zip(unique_tags, umap_coords)
    }
    tag_to_cluster: Dict[str, int] = {
        tag: int(label) for tag, label in zip(unique_tags, cluster_labels)
    }

    df_with_coords = add_umap_coordinates_to_df(df, tag_to_xy)
    df_with_coords = df_with_coords.copy()
    df_with_coords["cluster"] = df_with_coords["presence_tag"].map(tag_to_cluster).astype("Int64")

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

