"""
Aggregated per-article node-link diagram over ACE sentence clusters.

For a given article's clustering result JSON (e.g. sentence_clusters/{article_id}.json):
- Nodes = sentence-cluster categories (one node per category_id).
- Each category embedding = mean SBERT embedding of its member ACE sentences.
- Edges connect every pair of categories; edge thickness and opacity = cosine
  similarity between the two category embeddings.

Usage (from repo root with uv):
  uv run python extraction_pipeline/aggcomm_node_link.py extraction_pipeline/sentence_clusters/181.json
  uv run python extraction_pipeline/aggcomm_node_link.py extraction_pipeline/sentence_clusters/181.json --output-dir extraction_pipeline
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_EMBEDDINGS_DIR = BASE_DIR / "embeddings"
DEFAULT_OUTPUT_BASE = BASE_DIR  # article_id/category_graphs/ under this
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"


def _embedding_cache_path(embeddings_dir: Path, article_key: str) -> Path:
    """
    Return cache path for aggregated category embeddings.

    We use an article-specific key (e.g. "181_clusters") so we don't collide
    with per-comment caches used in the comment-level node-link pipeline.
    """
    return embeddings_dir / f"{article_key}.pkl"


def load_embeddings_cache(
    embeddings_dir: Path, article_key: str
) -> Dict[str, np.ndarray] | None:
    """Load cached embeddings for an article key if present."""
    path = _embedding_cache_path(embeddings_dir, article_key)
    if not path.is_file():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def save_embeddings_cache(
    embeddings_dir: Path, article_key: str, cache: Dict[str, np.ndarray]
) -> None:
    """Save embedding cache for an article key."""
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    path = _embedding_cache_path(embeddings_dir, article_key)
    with path.open("wb") as f:
        pickle.dump(cache, f)


def get_or_compute_embeddings(
    sentences: List[str],
    model: SentenceTransformer,
    cache: Dict[str, np.ndarray] | None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Return (full_cache, embeddings_for_sentences).

    full_cache is the updated cache (may include newly computed); embeddings_for_sentences
    is the subset needed for the given list (same keys as present in sentences).
    """
    if cache is None:
        cache = {}
    missing = [s for s in sentences if s not in cache]
    if missing:
        arr = model.encode(missing, show_progress_bar=False)
        arr = np.asarray(arr, dtype=np.float32)
        for s, vec in zip(missing, arr):
            cache[s] = vec
    out = {s: cache[s] for s in sentences if s in cache}
    return cache, out


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Single pair cosine similarity; a, b are 1d arrays."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _wrap_label(text: str, max_line_len: int = 35) -> str:
    """Wrap full node label for display so it doesn't get too wide."""
    text = text.strip()
    if not text or len(text) <= max_line_len:
        return text
    words = text.split()
    lines: List[str] = []
    current: List[str] = []
    current_len = 0
    for w in words:
        add_len = len(w) + (1 if current else 0)
        if current_len + add_len > max_line_len and current:
            lines.append(" ".join(current))
            current = [w]
            current_len = len(w)
        else:
            current.append(w)
            current_len += add_len
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def draw_and_save_graph(
    G: nx.Graph,
    save_path: Path,
    node_label_max_len: int = 40,
    figsize: Tuple[float, float] = (14, 12),
    min_edge_width: float = 0.3,
    max_edge_width: float = 4.0,
    layout: str = "spring",
) -> None:
    """
    Draw the node-link diagram with edge width proportional to weight (cosine similarity).

    Saves to save_path.
    """
    if G.number_of_nodes() == 0:
        # Empty graph: save a small placeholder image
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.text(0.5, 0.5, "No categories", ha="center", va="center")
        ax.axis("off")
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return

    weights = [G.edges[u, v].get("weight", 0.0) for u, v in G.edges()]
    if not weights:
        w_min = w_max = 0.0
    else:
        w_min, w_max = min(weights), max(weights)

    def width_from_weight(w: float) -> float:
        if w_max == w_min:
            return (min_edge_width + max_edge_width) / 2
        t = (w - w_min) / (w_max - w_min) if w_max > w_min else 0.0
        return min_edge_width + t * (max_edge_width - min_edge_width)

    def alpha_from_weight(w: float) -> float:
        """Map weight (cosine similarity) to opacity in [0, 1]."""
        if w_max == w_min:
            return 1.0
        t = (w - w_min) / (w_max - w_min) if w_max > w_min else 0.0
        return float(np.clip(t, 0.0, 1.0))

    if layout == "spring":
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    else:
        pos = nx.shell_layout(G)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw edges with thickness and opacity by weight (cosine similarity → opacity 0 to 1)
    for (u, v), w in zip(G.edges(), weights):
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=width_from_weight(w),
            alpha=alpha_from_weight(w),
            edge_color="lightblue",
            ax=ax,
        )

    # Node labels: full text, wrapped, drawn on top of each node
    labels = {n: _wrap_label(str(n), max_line_len=node_label_max_len) for n in G.nodes()}
    label_offset = 0.08
    pos_labels = {n: (xy[0], xy[1] + label_offset) for n, xy in pos.items()}

    nx.draw_networkx_nodes(G, pos, node_color="lightsteelblue", node_size=900, ax=ax)
    nx.draw_networkx_labels(
        G,
        pos_labels,
        labels=labels,
        font_size=7,
        ax=ax,
        verticalalignment="bottom",
        horizontalalignment="center",
    )
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def build_category_graph(category_embeddings: Dict[str, np.ndarray]) -> nx.Graph:
    """
    Build a graph where nodes are category labels and edge weights = cosine similarity.
    """
    G = nx.Graph()
    labels = list(category_embeddings.keys())
    for label in labels:
        G.add_node(label)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            va = category_embeddings.get(a)
            vb = category_embeddings.get(b)
            if va is not None and vb is not None:
                sim = cosine_similarity(va, vb)
                G.add_edge(a, b, weight=sim)
    return G


def process_clusters_file(
    clusters_path: Path,
    embeddings_dir: Path,
    output_base: Path,
    model: SentenceTransformer,
    force_reembed: bool = False,
) -> None:
    """
    Load one sentence-cluster JSON file; compute aggregated category embeddings;
    build and save a single per-article node-link graph over categories.
    """
    with clusters_path.open("r", encoding="utf-8") as f:
        clusters: Dict[str, Any] = json.load(f)

    article_id = str(clusters.get("article_id") or clusters_path.stem)
    categories = clusters.get("categories") or []

    if not categories:
        # Save an empty graph image to keep a consistent output path.
        graph_dir = output_base / article_id / "category_graphs"
        graph_dir.mkdir(parents=True, exist_ok=True)
        out_path = graph_dir / "aggregated_categories.png"
        empty_graph = nx.Graph()
        draw_and_save_graph(empty_graph, out_path)
        print(f"Saved empty category graph for article {article_id} at {out_path}")
        return

    # Collect all unique ACE sentences across all categories.
    all_sentences: List[str] = []
    for cat in categories:
        members = cat.get("member_sentences") or []
        for m in members:
            sentence = str(m.get("sentence", "")).strip()
            if sentence:
                all_sentences.append(sentence)
    unique_sentences = list(dict.fromkeys(all_sentences))

    article_key = f"{article_id}_clusters"
    cache = None if force_reembed else load_embeddings_cache(embeddings_dir, article_key)
    cache, _ = get_or_compute_embeddings(unique_sentences, model, cache)
    save_embeddings_cache(embeddings_dir, article_key, cache)

    # Aggregate embeddings per category: mean over member sentences.
    category_embeddings: Dict[str, np.ndarray] = {}
    for cat in categories:
        cat_id = cat.get("category_id")
        name = str(cat.get("name") or f"Category {cat_id}").strip()
        label = f"{cat_id}: {name}" if cat_id is not None else name

        members = cat.get("member_sentences") or []
        vecs: List[np.ndarray] = []
        for m in members:
            sentence = str(m.get("sentence", "")).strip()
            if not sentence:
                continue
            vec = cache.get(sentence)
            if vec is not None:
                vecs.append(vec)
        if not vecs:
            continue
        stacked = np.stack(vecs, axis=0)
        mean_vec = stacked.mean(axis=0)
        category_embeddings[label] = mean_vec

    # Build and save the graph.
    G = build_category_graph(category_embeddings)
    graph_dir = output_base / article_id / "category_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    out_path = graph_dir / "aggregated_categories.png"
    draw_and_save_graph(G, out_path)
    print(f"Saved aggregated category graph for article {article_id} at {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregated per-article node-link diagrams from ACE sentence clusters "
            "with SBERT-based category embeddings."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to sentence-cluster JSON (e.g. sentence_clusters/181.json)",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=DEFAULT_EMBEDDINGS_DIR,
        help="Directory to save/load per-article aggregated embedding caches",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help="Base directory for article_id/category_graphs/ output",
    )
    parser.add_argument(
        "--force-reembed",
        action="store_true",
        help="Recompute embeddings even if cache exists",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=SBERT_MODEL_NAME,
        help="Sentence transformer model name",
    )
    args = parser.parse_args()

    model = SentenceTransformer(args.model)
    process_clusters_file(
        args.input,
        args.embeddings_dir,
        args.output_dir,
        model,
        force_reembed=args.force_reembed,
    )


if __name__ == "__main__":
    main()

