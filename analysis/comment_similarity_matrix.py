"""
For each visualization (article), encode its comments with a sentence transformer,
reduce to 2D with t-SNE, and plot a scatter (top-level comments only; no replies).
Chart is built with Altair (Vega-Lite).

Usage:
  uv sync   # from project root
  uv run python analysis/comment_similarity_matrix.py   # all IDs in visualizations/ (comments from articles_data)
  uv run python analysis/comment_similarity_matrix.py --vis-id 37   # single visualization
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import altair as alt
from sentence_transformers import SentenceTransformer

# Raise Altair's data row limit for large similarity matrices
try:
    alt.data_transformers.enable("vegafusion")
except Exception:
    pass


# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIS_DIR = PROJECT_ROOT / "visualizations"
ARTICLES_DATA_DIR = PROJECT_ROOT / "articles_data"
OUT_DIR = PROJECT_ROOT / "analysis" / "outputs" / "comment_similarity"


def get_visualization_ids(vis_dir: Path) -> list[str]:
    """Return sorted list of visualization IDs from the visualizations folder (image stems)."""
    ids = set()
    for ext in ("*.png", "*.jpg", "*.webp"):
        for p in vis_dir.glob(ext):
            ids.add(p.stem)
    return sorted(ids, key=lambda x: (0, int(x)) if x.isdigit() else (1, x))


def load_top_level_comments(vis_id: str) -> list[dict]:
    """Load comments for a visualization; return only top-level comments (exclude reply content)."""
    path = ARTICLES_DATA_DIR / f"{vis_id}.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    # Each item is a top-level comment; we do not include any replies as separate comments
    return data


def get_comment_texts(comments: list[dict]) -> list[str]:
    """Extract 'comment info' text from each top-level comment."""
    texts = []
    for c in comments:
        info = c.get("comment info")
        if info and isinstance(info, str) and info.strip():
            texts.append(info.strip())
    return texts


def encode_comments(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Encode comment texts to embeddings (normalized for cosine similarity)."""
    if not texts:
        return np.zeros((0, 0))
    embeddings = model.encode(texts, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)


def embed_2d_tsne(embeddings: np.ndarray, perplexity: float = 15.0) -> np.ndarray:
    """Reduce embeddings to 2D with t-SNE. Returns (n, 2) array."""
    if embeddings.shape[0] < 2:
        return np.zeros((embeddings.shape[0], 2))
    n = embeddings.shape[0]
    perplexity = min(perplexity, max(1, (n - 1) / 3))  # t-SNE requires perplexity < n
    reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, metric="cosine")
    return reducer.fit_transform(embeddings)


def plot_comment_scatter_altair(
    embeddings: np.ndarray,
    labels: list[str],
    out_path: Path,
    title: str,
    max_label_len: int = 120,
) -> None:
    """Reduce embeddings to 2D with t-SNE, build Vega-Lite scatter with Altair, save HTML and .vg.json."""
    n = embeddings.shape[0]
    if n == 0:
        return
    coords = embed_2d_tsne(embeddings)
    short = [(s[:max_label_len] + "…") if len(s) > max_label_len else s for s in labels]
    df = pd.DataFrame({
        "dim_1": coords[:, 0],
        "dim_2": coords[:, 1],
        "comment": short,
        "index": range(n),
    })

    chart = (
        alt.Chart(df, title=alt.TitleParams(title, fontSize=14))
        .mark_circle(size=80, opacity=0.8)
        .encode(
            x=alt.X("dim_1:Q", title="t-SNE 1"),
            y=alt.Y("dim_2:Q", title="t-SNE 2"),
            tooltip=[
                alt.Tooltip("index:Q", title="Comment #"),
                alt.Tooltip("comment:N", title="Comment"),
            ],
        )
        .properties(width=600, height=500)
        .interactive()
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    html_path = out_path.with_suffix(".html")
    chart.save(str(html_path))

    vg_path = out_path.with_suffix(".vg.json")
    with open(vg_path, "w") as f:
        json.dump(chart.to_dict(format="vega"), f, indent=2)


def run_for_visualization(
    vis_id: str,
    model: SentenceTransformer,
    out_dir: Path,
) -> None:
    """Load comments (no replies), encode, UMAP to 2D, plot scatter."""
    comments = load_top_level_comments(vis_id)
    texts = get_comment_texts(comments)
    if len(texts) < 2:
        print(f"  vis {vis_id}: skipping (need at least 2 comments, got {len(texts)})")
        return
    embeddings = encode_comments(model, texts)
    title = f"Comment similarity (vis {vis_id}, n={len(texts)}) — t-SNE 2D"
    out_path = out_dir / f"comment_similarity_vis_{vis_id}"
    plot_comment_scatter_altair(embeddings, texts, out_path, title)
    html_path = out_dir / f"comment_similarity_vis_{vis_id}.html"
    print(f"  vis {vis_id}: {len(texts)} comments -> {html_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode comments with sentence transformer, t-SNE to 2D, and plot scatter per visualization."
    )
    parser.add_argument(
        "--vis-id",
        type=str,
        default=None,
        help="Process only this visualization ID (e.g. 37). If omitted, process all.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for output scatter plot HTML/JSON",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model name",
    )
    args = parser.parse_args()

    vis_dir = VIS_DIR
    if args.vis_id is not None:
        vis_ids = [args.vis_id]
        if not (ARTICLES_DATA_DIR / f"{args.vis_id}.json").exists():
            raise SystemExit(f"Unknown vis-id or missing comments file: articles_data/{args.vis_id}.json")
    else:
        vis_ids = get_visualization_ids(vis_dir)
        print(f"Processing {len(vis_ids)} visualizations from {vis_dir}")

    print(f"Loading sentence transformer: {args.model}")
    model = SentenceTransformer(args.model)

    for vis_id in vis_ids:
        run_for_visualization(vis_id, model, args.out_dir)

    print(f"Outputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
