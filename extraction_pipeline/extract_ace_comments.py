"""
Utility script to run only the ACE sentence extraction stage for NYT article
comments, saving results into the ``ace_comments`` folder.

This reuses the ACE conversion logic from ``extraction_pipeline.py`` but
separates it into a simpler CLI that does not run visualization tagging or
clustering.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import argparse

import extraction_pipeline as ep


def extract_ace_for_article(
    article_id: str,
    articles_data_dir: str = "../data/comment_data",
    ace_comments_base_dir: str = "ace_comments",
    api_key: str | None = None,
    model: str = "gpt-5.2",
) -> Dict[str, Any]:
    """
    Run ACE extraction (stage 1) for a single article's comments.

    For each top-level comment in the article JSON:
      - Convert the raw comment text to ACE sentences.
      - Save the result under:
            {ace_comments_base_dir}/{article_id}/{comment_index}.json
    """
    json_path = os.path.join(articles_data_dir, f"{article_id}.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    print(f"Reading {json_path}...")
    comments_data: List[Dict[str, Any]] = ep.read_json_file(json_path)

    client = ep._ensure_client(api_key)

    ace_article_dir = os.path.join(ace_comments_base_dir, str(article_id))
    os.makedirs(ace_article_dir, exist_ok=True)

    processed_comments: List[Dict[str, Any]] = []

    for idx, comment in enumerate(comments_data, start=1):
        raw_comment = str(comment.get("comment info", "") or "").strip()
        if not raw_comment:
            print(f"Skipping empty comment {idx} for article {article_id}.")
            continue

        print(f"Processing comment {idx} for article {article_id} (ACE conversion only)...")
        ace_sentences = ep.generate_ace_for_comment(
            raw_comment,
            client=client,
            model=model,
        )

        ace_output: Dict[str, Any] = {
            "article_id": article_id,
            "comment_index": idx,
            "raw_comment": raw_comment,
            "ace_sentences": ace_sentences,
        }

        ace_output_path = os.path.join(ace_article_dir, f"{idx}.json")
        with open(ace_output_path, "w", encoding="utf-8") as ace_f:
            json.dump(ace_output, ace_f, indent=2, ensure_ascii=False)

        processed_comments.append(ace_output)

    print(f"✓ Finished ACE extraction for article {article_id}")

    return {
        "article_id": article_id,
        "source_file": json_path,
        "ace_comments_dir": ace_article_dir,
        "comments": processed_comments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ACE sentences for NYT article comments and save them to the "
            "ace_comments folder (stage 1 only, no visualization tagging)."
        )
    )
    parser.add_argument(
        "--article_id",
        type=str,
        help="Article ID (e.g., '38'). If omitted, use --all to process every JSON in the data directory.",
    )
    parser.add_argument(
        "--articles-data-dir",
        type=str,
        default="../data/comment_data",
        help="Directory containing article JSON files (default: ../data/comment_data)",
    )
    parser.add_argument(
        "--ace-comments-dir",
        type=str,
        default="ace_comments",
        help="Base directory where per-article ACE JSON files are written (default: ace_comments)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2",
        help="OpenAI model to use for ACE extraction (default: gpt-5.2)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all article JSON files found in --articles-data-dir",
    )

    args = parser.parse_args()

    if not args.all and not args.article_id:
        parser.error("You must specify either --article_id or --all.")

    results: List[Dict[str, Any]] = []

    if args.all:
        data_path = Path(args.articles_data_dir)
        article_ids = [p.stem for p in data_path.glob("*.json")]
        print(f"Found {len(article_ids)} articles to process for ACE extraction.")

        for article_id in sorted(article_ids):
            try:
                result = extract_ace_for_article(
                    article_id=article_id,
                    articles_data_dir=args.articles_data_dir,
                    ace_comments_base_dir=args.ace_comments_dir,
                    api_key=args.api_key,
                    model=args.model,
                )
                results.append(result)
            except Exception as e:
                print(f"✗ Error extracting ACE for article {article_id}: {e}")
    else:
        result = extract_ace_for_article(
            article_id=args.article_id,
            articles_data_dir=args.articles_data_dir,
            ace_comments_base_dir=args.ace_comments_dir,
            api_key=args.api_key,
            model=args.model,
        )
        results.append(result)


if __name__ == "__main__":
    main()

