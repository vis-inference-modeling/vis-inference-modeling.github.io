"""
HDBSCAN-based clustering of comments for a single article.

Given a combined tagged-comments JSON file
  (e.g. extraction_pipeline/tagged_comments_combined/181.json),
this script:

1. Loads all comments for the article.
2. Reuses / computes SBERT embeddings for the article's unique presence_tag
   sentence strings (same cache format as tcomm_node_link.py).
3. Represents each comment by the mean embedding of its "present" presence_tag
   sentences.
4. Runs HDBSCAN over the per-comment embeddings.
5. Writes a JSON file describing the clusters in the same *schema* as
   sentence_clusters/{article_id}.json, i.e.:

   {
     "article_id": "181",
     "total_sentences": <int>,
     "categories": [
       {
         "category_id": 1,
         "name": "Comment cluster 1",
         "member_sentence_indices": [ ... ],
         "member_sentences": [
           {
             "sentence_index": 1,
             "comment_index": 1,
             "sentence": "ACE sentence text ..."
           },
           ...
         ]
       },
       ...
     ]
   }

Differences from sentence_clusters/{article_id}.json:
- Categories are clusters of *comments*, not clusters of ACE sentences.
- Each category groups the ACE sentences belonging to all comments in that
  cluster.

Usage (from repo root with uv):
  uv run python extraction_pipeline/hdbscan_comment_cluster.py \
      extraction_pipeline/tagged_comments_combined/181.json

By default this writes:
  extraction_pipeline/sentence_clusters/181_comments.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hdbscan
import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_EMBEDDINGS_DIR = BASE_DIR / "embeddings"
DEFAULT_OUTPUT_DIR = BASE_DIR / "sentence_clusters"
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class CommentRecord:
    """Minimal per-comment record used for clustering."""

    article_id: str
    comment_index: int
    present_tags: List[str]
    # (sentence_index, sentence_text) for all ACE sentences in this comment
    ace_sentences: List[Tuple[int, str]]


def _embedding_cache_path(embeddings_dir: Path, article_id: str) -> Path:
    """Same cache convention as tcomm_node_link.py."""
    return embeddings_dir / f"{article_id}.pkl"


def load_embeddings_cache(
    embeddings_dir: Path, article_id: str
) -> Dict[str, np.ndarray] | None:
    """Load cached embeddings for an article if present."""
    path = _embedding_cache_path(embeddings_dir, article_id)
    if not path.is_file():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def save_embeddings_cache(
    embeddings_dir: Path, article_id: str, cache: Dict[str, np.ndarray]
) -> None:
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


def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 2,
    min_samples: int | None = None,
) -> np.ndarray:
    """Cluster embeddings using HDBSCAN. Returns integer labels (-1 = noise).

    Uses Euclidean distance on L2-normalized vectors (equivalent to cosine for clustering),
    matching the approach in tagged_comments_scatter_presence.py.
    """
    if embeddings.size == 0 or len(embeddings) < 2:
        return np.array([], dtype=np.intp)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    X = (embeddings / norms).astype(np.float32)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        # Use leaf clustering to get more fine-grained (typically more numerous)
        # clusters instead of the coarser default "eom" selection.
        cluster_selection_method="leaf",
    )
    labels = clusterer.fit_predict(X)
    return np.asarray(labels, dtype=np.intp)


_ACE_LINE_RE = re.compile(r"^\s*(\d+)\.\s*(.*\S)\s*$")


def _parse_numbered_ace_sentences(lines: List[str]) -> List[Tuple[int, str]]:
    """
    Parse lines like "1. Sentence text" into (sentence_index, sentence_text).

    If a line does not match the expected pattern, it is skipped.
    """
    out: List[Tuple[int, str]] = []
    for line in lines:
        m = _ACE_LINE_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        text = m.group(2).strip()
        if not text:
            continue
        out.append((idx, text))
    return out


def load_comments_for_clustering(path: Path) -> List[CommentRecord]:
    """
    Load a combined tagged-comments JSON file and return CommentRecord list.
    """
    with path.open("r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    records: List[CommentRecord] = []
    for entry in data:
        article_id = str(entry.get("article_id", "")).strip()
        comment_index = entry.get("comment_index")
        if comment_index is None:
            continue

        presence_tags = entry.get("presence_tags") or {}
        if not isinstance(presence_tags, dict):
            presence_tags = {}

        present_tags: List[str] = []
        for tag_text, status in presence_tags.items():
            if isinstance(status, dict):
                presence = status.get("presence")
            else:
                presence = status
            if str(presence) == "present":
                present_tags.append(str(tag_text))

        numbered_ace = entry.get("numbered_ace_sentences") or []
        ace_sentences = _parse_numbered_ace_sentences(
            [str(x) for x in numbered_ace if isinstance(x, str)]
        )

        records.append(
            CommentRecord(
                article_id=article_id or str(path.stem),
                comment_index=int(comment_index),
                present_tags=present_tags,
                ace_sentences=ace_sentences,
            )
        )

    return records


def build_comment_embeddings(
    records: List[CommentRecord],
    model: SentenceTransformer,
    embeddings_dir: Path,
    force_reembed: bool = False,
) -> Tuple[str, np.ndarray, List[int]]:
    """
    For all comments belonging to a single article, compute per-comment embeddings.

    Returns (article_id, embedding_matrix, comment_indices).
    Comments that have no "present" presence_tags are skipped and thus will
    not appear in comment_indices.
    """
    if not records:
        return "", np.empty((0, 0), dtype=np.float32), []

    # Assume single-article file; fall back to first record's article_id.
    article_id = records[0].article_id or ""

    # Collect all unique presence_tag strings across comments.
    all_tags: List[str] = []
    for r in records:
        all_tags.extend(r.present_tags)
    unique_tags = list(dict.fromkeys(all_tags))

    # Load or compute embeddings for all tags for this article.
    cache = None if force_reembed else load_embeddings_cache(
        embeddings_dir, article_id
    )
    cache, tag_embeddings = get_or_compute_embeddings(unique_tags, model, cache)
    save_embeddings_cache(embeddings_dir, article_id, cache)

    # Build per-comment embeddings as mean over present tags.
    comment_vectors: List[np.ndarray] = []
    comment_indices: List[int] = []
    for r in records:
        vecs: List[np.ndarray] = []
        for tag in r.present_tags:
            v = tag_embeddings.get(tag)
            if v is not None:
                vecs.append(v)
        if not vecs:
            continue
        stacked = np.stack(vecs, axis=0)
        mean_vec = stacked.mean(axis=0)
        comment_vectors.append(mean_vec.astype(np.float32))
        comment_indices.append(r.comment_index)

    if not comment_vectors:
        return article_id, np.empty((0, 0), dtype=np.float32), []

    embeddings = np.stack(comment_vectors, axis=0)
    return article_id, embeddings, comment_indices


def build_clusters_json(
    article_id: str,
    records: List[CommentRecord],
    labels: np.ndarray,
    clustered_comment_indices: List[int],
) -> Dict[str, Any]:
    """
    Build a JSON-serializable dict matching the schema of sentence_clusters/{id}.json.

    - Each non-negative HDBSCAN cluster label becomes a category.
    - Category members are all ACE sentences from comments in that cluster.
    """
    # Map comment_index -> CommentRecord for quick lookup.
    index_to_record: Dict[int, CommentRecord] = {
        r.comment_index: r for r in records
    }

    # Gather all unique sentence indices across all comments for total_sentences.
    all_sentence_indices: set[int] = set()
    for r in records:
        for sent_idx, _ in r.ace_sentences:
            all_sentence_indices.add(sent_idx)

    total_sentences = len(all_sentence_indices)

    # Group comment indices by cluster label (ignore noise label -1).
    clusters: Dict[int, List[int]] = {}
    for comment_idx, label in zip(clustered_comment_indices, labels):
        if int(label) < 0:
            continue
        clusters.setdefault(int(label), []).append(comment_idx)

    # Sort cluster labels to give stable category_id assignment.
    sorted_labels = sorted(clusters.keys())

    categories: List[Dict[str, Any]] = []
    for cat_id, cluster_label in enumerate(sorted_labels, start=1):
        member_sentence_indices: List[int] = []
        member_sentences: List[Dict[str, Any]] = []

        for comment_index in clusters[cluster_label]:
            record = index_to_record.get(comment_index)
            if record is None:
                continue
            for sent_idx, sent_text in record.ace_sentences:
                member_sentence_indices.append(sent_idx)
                member_sentences.append(
                    {
                        "sentence_index": sent_idx,
                        "comment_index": comment_index,
                        "sentence": sent_text,
                    }
                )

        # Deduplicate and sort indices.
        member_sentence_indices = sorted(
            dict.fromkeys(member_sentence_indices)
        )

        categories.append(
            {
                "category_id": cat_id,
                "name": f"Comment cluster {cat_id} (HDBSCAN label {cluster_label})",
                "member_sentence_indices": member_sentence_indices,
                "member_sentences": member_sentences,
            }
        )

    out: Dict[str, Any] = {
        "article_id": article_id,
        "total_sentences": total_sentences,
        "categories": categories,
    }
    return out


def process_combined_file(
    combined_path: Path,
    embeddings_dir: Path,
    output_dir: Path,
    model: SentenceTransformer,
    force_reembed: bool = False,
    min_cluster_size: int = 2,
    min_samples: int | None = None,
) -> Path:
    """
    Process one combined tagged-comments JSON file and write the clusters JSON.

    Returns the path to the written JSON file.
    """
    records = load_comments_for_clustering(combined_path)
    if not records:
        raise SystemExit(f"No usable comments in {combined_path}")

    article_id, embeddings, comment_indices = build_comment_embeddings(
        records,
        model,
        embeddings_dir,
        force_reembed=force_reembed,
    )

    if embeddings.size == 0 or not comment_indices:
        # Still write a valid but empty- categories JSON to keep paths consistent.
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{combined_path.stem}_comments.json"
        empty_json = {
            "article_id": article_id or combined_path.stem,
            "total_sentences": 0,
            "categories": [],
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(empty_json, f, ensure_ascii=False, indent=2)
        print(
            f"No clusterable comments for article {article_id}, "
            f"wrote empty clusters JSON to {out_path}"
        )
        return out_path

    labels = run_hdbscan(
        embeddings,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )

    clusters_json = build_clusters_json(
        article_id=article_id or combined_path.stem,
        records=records,
        labels=labels,
        clustered_comment_indices=comment_indices,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{article_id}_comments.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(clusters_json, f, ensure_ascii=False, indent=2)

    print(f"Wrote comment clusters for article {article_id} to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster comments for a single article using HDBSCAN over "
            "sentence-transformer embeddings of present presence_tag sentences, "
            "and output a JSON file with the same schema as sentence_clusters/{id}.json."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help=(
            "Path to combined tagged comments JSON "
            "(e.g. extraction_pipeline/tagged_comments_combined/181.json)"
        ),
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
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory to write clusters JSON. "
            "Default: extraction_pipeline/sentence_clusters"
        ),
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
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="HDBSCAN min_cluster_size parameter",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="HDBSCAN min_samples parameter (default: None = use algorithm default)",
    )
    args = parser.parse_args()

    model = SentenceTransformer(args.model)
    process_combined_file(
        args.input,
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        model=model,
        force_reembed=args.force_reembed,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )


if __name__ == "__main__":
    main()

