"""
Pipeline to turn an article's ACE sentences into a causal graph using OpenAI.

For a given NYT article_id:
- Read all JSON files under ace_comments/{article_id}/.
- Collect and flatten their "ace_sentences" into a single list.
- Join those sentences into one paragraph of text.
- Call the OpenAI API with a few-shot prompt to extract a causal graph.
- Save the resulting node-link causal graph as JSON.

Usage (from repo root with uv):
  uv run python extraction_pipeline/dag_pipeline.py 181
  uv run python extraction_pipeline/dag_pipeline.py 35 --model gpt-5o
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from .extraction_pipeline import _ensure_client


BASE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = BASE_DIR / "examples"
CAUSAL_GRAPH_EXAMPLE_PATH = EXAMPLES_DIR / "causal_graph_example.json"
DEFAULT_ACE_COMMENTS_DIR = BASE_DIR / "ace_comments"
DEFAULT_OUTPUT_DIR = BASE_DIR / "causal_graphs"
DEFAULT_MODEL = "gpt-5o"


def load_ace_sentences(article_id: str, ace_comments_dir: Path) -> Tuple[List[str], str]:
    """
    Load all ACE sentences for an article_id from ace_comments/{article_id}/*.json.

    Returns:
        sentences: flat list of ACE sentence strings in sorted (by filename) order.
        paragraph: single string with sentences joined by spaces.
    """
    article_dir = ace_comments_dir / article_id
    if not article_dir.is_dir():
        raise FileNotFoundError(f"ACE comments directory not found: {article_dir}")

    json_paths = sorted(article_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No ACE comment JSON files found in {article_dir}")

    all_sentences: List[str] = []
    for path in json_paths:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        ace = data.get("ace_sentences") or []
        # Ensure we only keep clean strings.
        for s in ace:
            s_str = str(s).strip()
            if s_str:
                all_sentences.append(s_str)

    paragraph = " ".join(all_sentences)
    return all_sentences, paragraph


def load_causal_graph_example(example_path: Path | None = None) -> Tuple[str, Dict[str, Any]]:
    """
    Load the few-shot example paragraph and causal graph from the examples folder.

    Returns:
        (example_paragraph, example_causal_graph) for use in the prompt.
    """
    path = example_path or CAUSAL_GRAPH_EXAMPLE_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Causal graph example file not found: {path}. "
            "Expected a JSON file with 'paragraph' and 'causal_graph' keys."
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    paragraph = data.get("paragraph")
    causal_graph = data.get("causal_graph")
    if not paragraph or not causal_graph:
        raise ValueError(
            f"Example file {path} must contain 'paragraph' and 'causal_graph' keys."
        )
    return str(paragraph).strip(), causal_graph


def _build_causal_graph_prompt(paragraph: str, article_id: str) -> str:
    """
    Build the user prompt for causal graph extraction.

    The model is asked to:
    - identify variables (nodes) as causal concepts,
    - identify directed causal edges between them,
    - ignore purely anecdotal or irrelevant sentences,
    - return a single JSON object with nodes and edges.
    """
    example_paragraph, example_causal_graph = load_causal_graph_example()
    example_graph_json = json.dumps(example_causal_graph, indent=2)
    instructions = f"""You are an expert in causal modeling and scientific explanation.

Your task is to read a short paragraph and extract a directed causal graph.

DEFINITIONS
- A **node** is a variable or concept that can take different states or values.
  Examples: economic slump, demand for electric power, global warming.
- An **edge** is a directed causal relationship from a source node to a target node.
  It means: changes in the source have a causal influence on the target.

RULES
- Only include variables that participate in cause-and-effect relationships.
- Ignore purely personal anecdotes, rhetorical questions, or irrelevant details that
  are not part of any causal chain.
- Treat explicit causal language like "caused", "reduces", "increases",
  "worsens", "negatively affected", and conditional "if X then Y" as evidence
  of directed causal edges from X to Y.
- Use concise, machine-friendly identifiers for node ids in snake_case
  (e.g. "economic_slump", "demand_for_electric_power").
- Use human-readable phrases for node labels.
- For each edge, provide:
  - source: node id of the cause
  - target: node id of the effect
  - relationship: a short verb phrase such as "causes", "reduces", "increases",
    "worsens", "negatively_affects", "deteriorates".
  - description: a short natural language explanation grounding the edge in the text.

EXAMPLE
Here is an example paragraph and a correct causal graph:

Example paragraph:
\"\"\"{example_paragraph}\"\"\"

Example causal graph JSON:
{example_graph_json}

NOW YOUR TASK
Now process the NEW paragraph below and return ONLY a JSON object with this schema:
{{
  "article_id": "{article_id}",
  "nodes": [
    {{"id": "node_id", "label": "Human readable label", "description": "Optional longer description"}}
  ],
  "edges": [
    {{"source": "node_id", "target": "node_id", "relationship": "causes", "description": "Short explanation"}}
  ]
}}

The JSON must be syntactically valid and must include at least all nodes that appear
in any edge.

Here is the new paragraph to analyze:
\"\"\"{paragraph}\"\"\"
"""
    return instructions


def extract_causal_graph(
    paragraph: str,
    article_id: str,
    client: OpenAI,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Call the OpenAI API to extract a causal graph from the paragraph."""
    if not paragraph.strip():
        raise ValueError("Cannot extract causal graph from an empty paragraph.")

    user_prompt = _build_causal_graph_prompt(paragraph=paragraph, article_id=article_id)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract clear, machine-readable causal graphs from text and "
                    "always return valid JSON matching the requested schema."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    content = response.choices[0].message.content
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse causal graph JSON: {exc}\nRaw: {content}")

    # Ensure article_id is set.
    parsed.setdefault("article_id", article_id)
    if parsed["article_id"] != article_id:
        parsed["article_id"] = article_id

    # Normalize node and edge collections.
    parsed.setdefault("nodes", [])
    parsed.setdefault("edges", [])
    return parsed


def save_causal_graph(
    graph: Dict[str, Any],
    article_id: str,
    output_dir: Path,
) -> Path:
    """Save the causal graph JSON to output_dir/{article_id}.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{article_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    return out_path


def run_dag_pipeline(
    article_id: str,
    ace_comments_dir: Path = DEFAULT_ACE_COMMENTS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> Path:
    """End-to-end pipeline: load ACE, call OpenAI, and save a causal graph."""
    sentences, paragraph = load_ace_sentences(article_id, ace_comments_dir)
    if not sentences:
        raise ValueError(f"No ACE sentences found for article_id={article_id}")

    client = _ensure_client(api_key)
    graph = extract_causal_graph(paragraph=paragraph, article_id=article_id, client=client, model=model)
    out_path = save_causal_graph(graph=graph, article_id=article_id, output_dir=output_dir)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a causal graph from ACE sentences for a given NYT article.\n\n"
            "Reads ace_comments/{article_id}/*.json, concatenates 'ace_sentences' "
            "into a paragraph, calls the OpenAI API to extract a causal graph, "
            "and saves the result as causal_graphs/{article_id}.json by default."
        )
    )
    parser.add_argument(
        "article_id",
        help="NYT article id (e.g. '181') corresponding to ace_comments/{article_id}/",
    )
    parser.add_argument(
        "--ace-dir",
        type=Path,
        default=DEFAULT_ACE_COMMENTS_DIR,
        help=f"Base directory containing ace_comments/ (default: {DEFAULT_ACE_COMMENTS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for causal graphs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model name to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional explicit OpenAI API key; otherwise uses OPENAI_API_KEY env var.",
    )

    args = parser.parse_args()

    out_path = run_dag_pipeline(
        article_id=args.article_id,
        ace_comments_dir=args.ace_dir,
        output_dir=args.output_dir,
        model=args.model,
        api_key=args.api_key,
    )
    print(f"Wrote causal graph for article {args.article_id} to {out_path}")


if __name__ == "__main__":
    main()

