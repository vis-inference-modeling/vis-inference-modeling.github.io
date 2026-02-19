"""
Node-link diagram pipeline for tagged comments.

For each comment in a combined tagged-comments JSON file:
- Nodes = keys in "presence_tags" (sentence strings).
- Edges connect every pair of nodes; edge thickness = cosine similarity
  between the SBERT embeddings of the two node strings.

Pipeline:
1. Load combined JSON (e.g. tagged_comments_combined/{article_id}.json).
2. Collect all unique presence_tag strings for the article; compute SBERT
   embeddings once and save to embeddings_dir (e.g. embeddings/{article_id}.pkl).
3. For each comment, build the graph, draw it with edge width ∝ similarity,
   and save the image to {article_id}/comment_graphs/{comment_index}.png.

Usage (from repo root with uv):
  uv run python extraction_pipeline/tcomm_node_link.py extraction_pipeline/tagged_comments_combined/181.json
  uv run python extraction_pipeline/tcomm_node_link.py extraction_pipeline/tagged_comments_combined/181.json --output-dir extraction_pipeline
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
DEFAULT_COMBINED_DIR = BASE_DIR / "tagged_comments_combined"
DEFAULT_EMBEDDINGS_DIR = BASE_DIR / "embeddings"
DEFAULT_OUTPUT_BASE = BASE_DIR  # article_id/comment_graphs/ under this
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"


def _embedding_cache_path(embeddings_dir: Path, article_id: str) -> Path:
    return embeddings_dir / f"{article_id}.pkl"


def load_embeddings_cache(embeddings_dir: Path, article_id: str) -> Dict[str, np.ndarray] | None:
    """Load cached embeddings for an article if present."""
    path = _embedding_cache_path(embeddings_dir, article_id)
    if not path.is_file():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def save_embeddings_cache(embeddings_dir: Path, article_id: str, cache: Dict[str, np.ndarray]) -> None:
    """Save embedding cache for an article."""
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    path = _embedding_cache_path(embeddings_dir, article_id)
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


def build_graph(
    nodes: List[str],
    embeddings: Dict[str, np.ndarray],
) -> nx.Graph:
    """Build a graph with nodes = presence_tag strings and edge weights = cosine similarity."""
    G = nx.Graph()
    for s in nodes:
        G.add_node(s)
    n_list = list(nodes)
    for i in range(len(n_list)):
        for j in range(i + 1, len(n_list)):
            a, b = n_list[i], n_list[j]
            va = embeddings.get(a)
            vb = embeddings.get(b)
            if va is not None and vb is not None:
                sim = cosine_similarity(va, vb)
                # Keep edges with positive similarity, or all if you want (can be negative for SBERT)
                G.add_edge(a, b, weight=sim)
    return G


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
    node_label_max_len: int = 28,
    figsize: Tuple[float, float] = (12, 10),
    min_edge_width: float = 0.3,
    max_edge_width: float = 4.0,
    layout: str = "spring",
    node_present: Dict[str, bool] | None = None,
) -> None:
    """
    Draw the node-link diagram with edge width proportional to weight (cosine similarity).
    If node_present is provided, nodes with True are drawn in one color, False in another.
    Saves to save_path.
    """
    if G.number_of_nodes() == 0:
        # Empty graph: save a small placeholder image
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.text(0.5, 0.5, "No nodes", ha="center", va="center")
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

    # Node colors: present vs not present
    if node_present is not None:
        node_colors = [
            "mediumseagreen" if node_present.get(n, True) else "lightgray"
            for n in G.nodes()
        ]
    else:
        node_colors = "lightsteelblue"

    fig, ax = plt.subplots(figsize=figsize)
    # Draw edges with thickness and opacity by weight (cosine similarity → opacity 0 to 1)
    for (u, v), w in zip(G.edges(), weights):
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            width=width_from_weight(w),
            alpha=alpha_from_weight(w),
            edge_color="lightblue",
            ax=ax,
        )
    # Node labels: full text, wrapped, drawn on top of each node
    labels = {n: _wrap_label(n, max_line_len=node_label_max_len) for n in G.nodes()}
    # Offset label position above the node (node_size 800 ~ radius ~17 in points; use small axis offset)
    label_offset = 0.08
    pos_labels = {n: (xy[0], xy[1] + label_offset) for n, xy in pos.items()}
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)
    nx.draw_networkx_labels(
        G, pos_labels, labels=labels, font_size=7, ax=ax,
        verticalalignment="bottom", horizontalalignment="center",
    )
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def process_combined_file(
    combined_path: Path,
    embeddings_dir: Path,
    output_base: Path,
    model: SentenceTransformer,
    force_reembed: bool = False,
) -> None:
    """
    Load one combined JSON file; ensure embeddings exist; for each comment
    build and save the node-link graph to {output_base}/{article_id}/comment_graphs/{comment_index}.png.
    """
    with combined_path.open("r", encoding="utf-8") as f:
        comments: List[Dict[str, Any]] = json.load(f)

    if not comments:
        return

    article_id = comments[0].get("article_id", "unknown")
    # Collect all unique presence_tag keys across comments
    all_sentences: List[str] = []
    for c in comments:
        presence_tags = c.get("presence_tags") or {}
        if isinstance(presence_tags, dict):
            all_sentences.extend(presence_tags.keys())
    unique_sentences = list(dict.fromkeys(all_sentences))

    # Load or compute embeddings for the whole article
    cache = None if force_reembed else load_embeddings_cache(embeddings_dir, article_id)
    cache, _ = get_or_compute_embeddings(unique_sentences, model, cache)
    save_embeddings_cache(embeddings_dir, article_id, cache)

    graph_dir = output_base / str(article_id) / "comment_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    for entry in comments:
        comment_index = entry.get("comment_index")
        presence_tags = entry.get("presence_tags") or {}
        if not isinstance(presence_tags, dict) or comment_index is None:
            continue
        nodes = list(presence_tags.keys())
        # Build node_present from nested presence_tags (presence: "present" | "not_present")
        node_present: Dict[str, bool] = {}
        for k, v in presence_tags.items():
            if isinstance(v, dict):
                node_present[k] = v.get("presence") == "present"
            else:
                node_present[k] = True
        if not nodes:
            # Still save an empty graph image so filename exists
            pass
        emb_sub = {s: cache[s] for s in nodes if s in cache}
        G = build_graph(nodes, emb_sub)
        # Keep only edges between present and not-present nodes
        for u, v in list(G.edges()):
            if node_present.get(u, True) == node_present.get(v, True):
                G.remove_edge(u, v)
        out_path = graph_dir / f"{comment_index}.png"
        draw_and_save_graph(G, out_path, node_present=node_present)
        print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Node-link diagrams from presence_tags with SBERT edge weights")
    parser.add_argument(
        "input",
        type=Path,
        help="Path to combined tagged comments JSON (e.g. tagged_comments_combined/181.json)",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=DEFAULT_EMBEDDINGS_DIR,
        help="Directory to save/load per-article embedding caches",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help="Base directory for article_id/comment_graphs/ output",
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
    process_combined_file(
        args.input,
        args.embeddings_dir,
        args.output_dir,
        model,
        force_reembed=args.force_reembed,
    )


if __name__ == "__main__":
    main()
