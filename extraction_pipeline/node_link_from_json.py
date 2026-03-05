"""
Node-link diagram from a simple JSON graph specification.

The expected JSON structure (similar to aggregated_categories.json) is:

{
  "article_id": "181",          # optional, unused by the script
  "title": "Optional title",    # optional, unused by the script
  "nodes": [
    {
      "id": "node_1",
      "label": "Full node label text",
      "category": "theme_group_a"
    },
    ...
  ],
  "edges": [
    {
      "source": "node_1",
      "target": "node_2",
      "weight": 0.8,            # optional, used for edge width/opacity
      "relationship": "precedes",   # optional, unused by the script
      "description": "...",         # optional, unused by the script
    },
    ...
  ]
}

Usage (from repo root, via uv):

  uv run python extraction_pipeline/node_link_from_json.py \
      extraction_pipeline/181/category_graphs/aggregated_categories.json

This will save a PNG next to the JSON file with the same stem, e.g.:

  aggregated_categories.json -> aggregated_categories.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx

# Type aliases
NodeId = str  # type alias for node identifier
Category = str  # type alias for node category


class NodeSpec(dict):
    """
    Dict-based type for nodes in the input JSON.

    Required keys:
    - "id": str
    - "label": str
    - "category": str
    """


class EdgeSpec(dict):
    """
    Dict-based type for edges in the input JSON.

    Required keys:
    - "source": str
    - "target": str

    Optional keys:
    - "weight": float
    - "relationship": str
    - "description": str
    """


def _wrap_label(text: str, max_line_len: int = 35) -> str:
    """
    Wrap long node labels to multiple lines for display.

    :param text: Original label text.
    :param max_line_len: Maximum characters per line before wrapping.
    :return: Wrapped label with embedded newlines.
    """
    text = text.strip()
    if not text or len(text) <= max_line_len:
        return text

    words: List[str] = text.split()
    lines: List[str] = []
    current: List[str] = []
    current_len = 0

    for w in words:
        add_len = len(w) + (1 if current else 0)
        if current and current_len + add_len > max_line_len:
            lines.append(" ".join(current))
            current = [w]
            current_len = len(w)
        else:
            current.append(w)
            current_len += add_len

    if current:
        lines.append(" ".join(current))

    return "\n".join(lines)


def _load_graph_json(path: Path) -> Mapping[str, object]:
    """
    Load the JSON graph specification from disk.

    :param path: Path to a JSON file with nodes/edges.
    :return: Parsed JSON as a mapping.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object/dict.")
    return data


def _collect_categories(nodes: Sequence[NodeSpec]) -> List[Category]:
    """
    Collect unique categories in input order.

    :param nodes: Sequence of node specifications.
    :return: List of unique category strings.
    """
    seen: Dict[Category, None] = {}
    for node in nodes:
        cat = str(node.get("category", "default"))
        if cat not in seen:
            seen[cat] = None
    return list(seen.keys())


def _build_category_color_map(categories: Sequence[Category]) -> Dict[Category, str]:
    """
    Assign a distinct color to each category.

    :param categories: Sequence of category labels.
    :return: Mapping from category to a matplotlib-compatible color string.
    """
    if not categories:
        return {}

    cmap = cm.get_cmap("tab20")
    n = max(len(categories), 1)
    category_to_color: Dict[Category, str] = {}
    for i, cat in enumerate(categories):
        # Evenly spaced colors across the colormap.
        rgba = cmap(float(i) / max(n - 1, 1))
        category_to_color[cat] = mcolors.to_hex(rgba)
    return category_to_color


def build_graph_from_json(data: Mapping[str, object]) -> Tuple[nx.Graph, Dict[NodeId, NodeSpec]]:
    """
    Build a NetworkX graph from the JSON specification.

    Node attributes:
    - "label": full label text (string)
    - "category": category label (string)

    Edge attributes:
    - "weight": numeric weight (float, defaults to 1.0)

    :param data: Parsed JSON mapping with "nodes" and "edges".
    :return: Tuple of (graph, node_lookup) where node_lookup maps node id to node spec.
    """
    raw_nodes = data.get("nodes")  # type: ignore[assignment]
    raw_edges = data.get("edges")  # type: ignore[assignment]

    if not isinstance(raw_nodes, list):
        raise ValueError('JSON must contain a "nodes" list.')
    if not isinstance(raw_edges, list):
        raise ValueError('JSON must contain an "edges" list.')

    nodes: List[NodeSpec] = [NodeSpec(n) for n in raw_nodes]  # type: ignore[arg-type]
    edges: List[EdgeSpec] = [EdgeSpec(e) for e in raw_edges]  # type: ignore[arg-type]

    G = nx.Graph()
    node_lookup: Dict[NodeId, NodeSpec] = {}

    for node in nodes:
        node_id = str(node.get("id"))
        if not node_id:
            raise ValueError("Every node must have a non-empty 'id'.")
        label = str(node.get("label", node_id))
        category = str(node.get("category", "default"))
        node_lookup[node_id] = node
        G.add_node(node_id, label=label, category=category)

    for edge in edges:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if not source or not target:
            raise ValueError("Every edge must have 'source' and 'target'.")
        weight_raw = edge.get("weight", 1.0)
        try:
            weight = float(weight_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            weight = 1.0
        if source not in G.nodes or target not in G.nodes:
            # Skip edges that reference unknown nodes.
            continue
        G.add_edge(source, target, weight=weight)

    return G, node_lookup


def _edge_weights(G: nx.Graph) -> List[float]:
    """
    Extract edge weights from the graph.

    :param G: Graph with numeric "weight" attributes on edges.
    :return: List of edge weights (float).
    """
    return [float(G.edges[u, v].get("weight", 1.0)) for u, v in G.edges()]


def _normalize_range(values: Iterable[float]) -> Tuple[float, float]:
    """
    Compute (min, max) for a sequence of numeric values.

    :param values: Iterable of floats.
    :return: Pair (min_value, max_value).
    """
    vals = list(values)
    if not vals:
        return 0.0, 0.0
    return min(vals), max(vals)


def draw_and_save_graph(
    G: nx.Graph,
    node_lookup: Mapping[NodeId, NodeSpec],
    save_path: Path,
    node_label_max_len: int = 32,
    figsize: Tuple[float, float] = (10.0, 8.0),
    min_edge_width: float = 0.4,
    max_edge_width: float = 4.0,
    layout: str = "spring",
) -> None:
    """
    Draw the node-link diagram with:
    - Node colors determined by node["category"].
    - Edge width and opacity determined by edge "weight" (if present).

    :param G: Graph to draw.
    :param node_lookup: Mapping from node id to its specification.
    :param save_path: Output PNG path.
    :param node_label_max_len: Maximum label line length before wrapping.
    :param figsize: Matplotlib figure size in inches.
    :param min_edge_width: Minimum edge line width.
    :param max_edge_width: Maximum edge line width.
    :param layout: Layout algorithm, "spring" or "shell".
    """
    if G.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=(4.0, 4.0))
        ax.text(0.5, 0.5, "No nodes", ha="center", va="center")
        ax.axis("off")
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return

    weights = _edge_weights(G)
    w_min, w_max = _normalize_range(weights)

    def width_from_weight(w: float) -> float:
        if w_max == w_min:
            return (min_edge_width + max_edge_width) / 2.0
        t = (w - w_min) / (w_max - w_min)
        return min_edge_width + t * (max_edge_width - min_edge_width)

    def alpha_from_weight(w: float) -> float:
        if w_max == w_min:
            return 1.0
        t = (w - w_min) / (w_max - w_min)
        if t < 0.0:
            return 0.0
        if t > 1.0:
            return 1.0
        return float(t)

    if layout == "spring":
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    else:
        pos = nx.shell_layout(G)

    # Build category → color mapping.
    categories = _collect_categories(list(node_lookup.values()))
    category_to_color = _build_category_color_map(categories)

    node_colors: List[str] = []
    for node_id in G.nodes():
        spec = node_lookup.get(node_id, {})
        cat = str(spec.get("category", "default"))
        node_colors.append(category_to_color.get(cat, "#cccccc"))

    fig, ax = plt.subplots(figsize=figsize)

    # Draw edges (one-by-one to vary width/alpha).
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

    # Node labels (wrapped).
    labels = {}
    for node_id in G.nodes():
        spec = node_lookup.get(node_id, {})
        label_text = str(spec.get("label", node_id))
        labels[node_id] = _wrap_label(label_text, max_line_len=node_label_max_len)

    label_offset = 0.08
    pos_labels = {n: (xy[0], xy[1] + label_offset) for n, xy in pos.items()}

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=800,
        alpha=0.4,
        ax=ax,
    )
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


def _default_output_path(input_path: Path) -> Path:
    """
    Construct the default PNG path for a given input JSON file.

    :param input_path: Path to the input JSON file.
    :return: Output path with '.png' extension next to the JSON.
    """
    return input_path.with_suffix(".png")


def main() -> None:
    """
    CLI entry point.

    Example:

      uv run python extraction_pipeline/node_link_from_json.py \
          extraction_pipeline/181/category_graphs/aggregated_categories.json \
          --output extraction_pipeline/181/category_graphs/aggregated_categories.png
    """
    parser = argparse.ArgumentParser(
        description="Draw a node-link diagram from a JSON specification "
        "(nodes with id/label/category and weighted edges)."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to JSON file describing the graph.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (defaults to input path with .png extension).",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="spring",
        choices=["spring", "shell"],
        help="Layout algorithm to use for node positions.",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    output_path: Path = args.output or _default_output_path(input_path)

    data = _load_graph_json(input_path)
    G, node_lookup = build_graph_from_json(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    draw_and_save_graph(G, node_lookup, output_path, layout=args.layout)
    print(f"Saved node-link diagram to {output_path}")


if __name__ == "__main__":
    main()

