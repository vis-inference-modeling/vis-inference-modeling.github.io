"""
Utility script to combine per-comment tagged outputs into a single JSON file
per article, with cleaned presence tag keys.

For each subdirectory under `tagged_comments/` (named by article_id), this script:
- Reads all `*.json` files in that directory.
- Cleans the keys of the `presence_tags` mapping by stripping duplicated
  numeric prefixes like `"1. 1. "` so that
  `"1. 1. Some sentence."` becomes `"Some sentence."`.
- When `--ace-dir` points to an existing directory (default: ace_comments),
  adds `original_comment` to each record from the matching ACE file's
  `raw_comment` when available.
- Writes a combined `{article_id}.json` file containing a list of the per-comment
  records to an output directory.

By default, it:
- Reads from `tagged_comments/`
- Writes combined files to `tagged_comments_combined/` as `{article_id}.json`
- Uses `ace_comments/` to attach original comment text when present
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List


PRESENCE_KEY_PREFIX_RE = re.compile(r"^\d+\.\s+\d+\.\s+")


def clean_presence_tags(
    presence_tags: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Clean the keys of a presence_tags mapping by removing duplicated numeric prefixes.

    Values may be the legacy string ("present" / "not_present") or the new
    shape {"presence": str, "task_type": str}. Both are preserved as-is.
    """
    cleaned: Dict[str, Any] = {}
    for key, value in presence_tags.items():
        # Remove patterns like "1. 1. " or "10. 10. " at the start of the string.
        new_key = PRESENCE_KEY_PREFIX_RE.sub("", key).lstrip()
        cleaned[new_key] = value
    return cleaned


def combine_article_tagged_comments(
    article_dir: Path,
    output_dir: Path,
    ace_base_dir: Path | None = None,
) -> Path:
    """
    Combine all per-comment tagged JSON files in `article_dir` into a single file.

    The output is a JSON list of per-comment dicts (each corresponding to the
    original per-comment JSON), but with `presence_tags` keys cleaned.
    If `ace_base_dir` is set, each record is augmented with `original_comment`
    from the matching ace_comments file when available.
    """
    article_id = article_dir.name
    combined: List[Dict[str, Any]] = []
    ace_article_dir = (ace_base_dir / article_id) if ace_base_dir else None

    # Collect all JSON files in the article directory.
    json_files = sorted(
        (p for p in article_dir.iterdir() if p.is_file() and p.suffix == ".json"),
        key=lambda p: p.stem,
    )

    for json_path in json_files:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        presence_tags = data.get("presence_tags") or {}
        if isinstance(presence_tags, dict):
            data["presence_tags"] = clean_presence_tags(presence_tags)

        # Add original comment from ACE file if available.
        if ace_article_dir is not None:
            ace_path = ace_article_dir / f"{json_path.stem}.json"
            if ace_path.exists():
                try:
                    with ace_path.open("r", encoding="utf-8") as f:
                        ace_data = json.load(f)
                    raw = ace_data.get("raw_comment")
                    if raw is not None:
                        data["original_comment"] = raw
                except (json.JSONDecodeError, OSError):
                    pass

        combined.append(data)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{article_id}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    return output_path


def combine_all_tagged_comments(
    base_dir: Path,
    output_dir: Path,
    article_id: str | None = None,
    ace_base_dir: Path | None = None,
) -> None:
    """
    Combine tagged comments for all (or a single) article ID under `base_dir`.
    If `ace_base_dir` is set, each combined comment will include `original_comment`
    from the matching file under ace_comments when available.
    """
    if article_id is not None:
        article_dirs = [base_dir / str(article_id)]
    else:
        article_dirs = [
            d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]

    if not article_dirs:
        print(f"No article directories found under {base_dir}")
        return

    for article_dir in sorted(article_dirs, key=lambda d: d.name):
        if not article_dir.exists() or not article_dir.is_dir():
            print(f"Skipping non-directory {article_dir}")
            continue

        print(f"Combining tagged comments for article {article_dir.name}...")
        output_path = combine_article_tagged_comments(
            article_dir, output_dir, ace_base_dir=ace_base_dir
        )
        print(f"  -> wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine per-comment tagged outputs under `tagged_comments/` into a "
            "single `{article_id}.json` per article with cleaned presence_tags keys."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="tagged_comments",
        help="Base directory containing per-article tagged comment folders "
        "(default: tagged_comments)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tagged_comments_combined",
        help="Directory to write combined `{article_id}.json` files "
        "(default: tagged_comments_combined)",
    )
    parser.add_argument(
        "--article-id",
        type=str,
        default=None,
        help="If provided, only combine tagged comments for this article ID.",
    )
    parser.add_argument(
        "--ace-dir",
        type=str,
        default="ace_comments",
        help="Directory containing per-article ACE comment JSONs (same layout as "
        "tagged_comments). If present, each combined comment gets original_comment "
        "from the matching ACE file (default: ace_comments).",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    ace_base_dir = Path(args.ace_dir) if args.ace_dir else None
    if ace_base_dir is not None and (not ace_base_dir.exists() or not ace_base_dir.is_dir()):
        ace_base_dir = None  # Skip original_comment if ace dir missing

    if not base_dir.exists() or not base_dir.is_dir():
        raise FileNotFoundError(
            f"Base tagged comments directory does not exist or is not a directory: {base_dir}"
        )

    combine_all_tagged_comments(
        base_dir, output_dir, article_id=args.article_id, ace_base_dir=ace_base_dir
    )


if __name__ == "__main__":
    main()

