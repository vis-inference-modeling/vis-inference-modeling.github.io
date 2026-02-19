"""
Program to process each NYT article comment through a two-stage OpenAI pipeline.

Stage 1: Convert each top-level comment (not replies) into Attempto Controlled
English (ACE) sentences.

Stage 2: For each comment's ACE sentences, use the corresponding visualization
image and tag each sentence as: (1) whether it describes information present in
the visualization ("present") or not ("not_present"); and (2) for present
sentences, the task type—value_identification, arithmetic_computation, or
statistical_inference (as defined below).
"""

import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()


def read_json_file(json_path: str) -> List[Dict[str, Any]]:
    """Read and parse JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


ACE_FEW_SHOT_COMMENT = """Comment:
1. The rate of wind and solar have been becoming higher since 2000, but fossil fuels occupies more than half of materials. In fact, the situation is seen as a problem, so the governments of various countries are promoting renewable energy. I had an opportunity to think about a solution to improve the situations when I was a junior high school student.
2. I wonder why the electric power generation in the world decreased in around 2008 and 2021. I think that the Lehman shock in 2008 and the spreading of coronavirus relate to the result. Indeed, the economy was in a slump because of them, so demand of electric power generation decreased.
3. If the rate of fossil fuels remains steady, the global warming and the air pollution will worsen. So, the precious natural environment in Akita, where I live, could be negatively affected. For instance, the growth of Akita cedars may deteriorate.
4. Catchy headline : What is the Solution to Increase the Rate of Clean Energy?
"""

ACE_FEW_SHOT_OUTPUT = """attempto controlled english (ACE) example:
The rate of wind energy and solar energy has increased since 2000.
However, fossil fuels occupy more than half of energy materials.
This situation is a problem.
The governments of many countries promote renewable energy.
When I was a junior high school student, I had an opportunity.
At that time, I thought about a solution to improve this situation.
The electric power generation in the world decreased around 2008.
The electric power generation in the world also decreased around 2021.
The Lehman Shock happened in 2008.
The coronavirus spread in 2021.
The Lehman Shock caused an economic slump.
The spread of coronavirus caused an economic slump.
An economic slump reduces the demand for electric power.
The decrease in electric power generation relates to these events.
The rate of fossil fuels may remain steady.
If the rate of fossil fuels remains steady, global warming will worsen.
If the rate of fossil fuels remains steady, air pollution will worsen.
Akita has a precious natural environment.
I live in Akita.
If global warming worsens, the natural environment in Akita will be negatively affected.
Akita cedars grow in Akita.
If global warming worsens, the growth of Akita cedars may deteriorate.
The headline is:
What is the solution to increase the rate of clean energy?
"""


VIS_TAGGING_EXAMPLE_OUTPUT = {
    "The rate of wind and solar have been becoming higher since 2000.": {
        "presence": "present",
        "task_type": "statistical_inference",
    },
    "Fossil fuels occupies more than half of materials.": {
        "presence": "present",
        "task_type": "value_identification",
    },
    "The situation is seen as a problem.": {
        "presence": "not_present",
        "task_type": "not_applicable",
    },
    "The governments of various countries are promoting renewable energy.": {
        "presence": "not_present",
        "task_type": "not_applicable",
    },
    "I had an opportunity to think about a solution to improve the situations when I was a junior high school student.": {
        "presence": "not_present",
        "task_type": "not_applicable",
    },
    "The electric power generation in the world decreased in around 2008.": {
        "presence": "present",
        "task_type": "value_identification",
    },
    "The electric power generation in the world decreased in around 2021.": {
        "presence": "present",
        "task_type": "value_identification",
    },
    "The Lehman shock in 2008 relates to the result.": {
        "presence": "not_present",
        "task_type": "not_applicable",
    },
    "The spreading of coronavirus relates to the result.": {
        "presence": "not_present",
        "task_type": "not_applicable",
    },
    "The economy was in a slump because of them.": {
        "presence": "not_present",
        "task_type": "not_applicable",
    },
    "The demand of electric power generation decreased because of the economic slump.": {
        "presence": "not_present",
        "task_type": "not_applicable",
    },
    "If the rate of fossil fuels remains steady, global warming will worsen.": {
        "presence": "not_present",
        "task_type": "not_applicable",
    },
    "If the rate of fossil fuels remains steady, air pollution will worsen.": {
        "presence": "not_present",
        "task_type": "not_applicable",
    },
    "The precious natural environment in Akita could be negatively affected.": {
        "presence": "not_present",
        "task_type": "not_applicable",
    },
    "The growth of Akita cedars may deteriorate.": {
        "presence": "not_present",
        "task_type": "not_applicable",
    },
    "What is the Solution to Increase the Rate of Clean Energy?": {
        "presence": "not_present",
        "task_type": "not_applicable",
    },
}


def _ensure_client(api_key: str | None) -> OpenAI:
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
            "in a .env file or pass as argument."
        )
    return OpenAI(api_key=api_key)


def generate_ace_for_comment(
    comment_text: str, client: OpenAI, model: str = "gpt-5.2"
) -> List[str]:
    """
    Convert a free-form article comment into Attempto Controlled English (ACE) sentences.

    Returns a list of ACE sentences.
    """
    if not comment_text.strip():
        return []

    system_content = (
        "You are an expert in Attempto Controlled English (ACE). "
        "Given an article comment, you rewrite it as a sequence of simple ACE sentences. "
        "Avoid rhetorical flourishes, and express each idea as its own clear sentence."
    )

    user_instructions = f"""Convert the following article comment into Attempto Controlled English (ACE) sentences.

First, here is a few-shot example showing how to convert a comment into ACE:

{ACE_FEW_SHOT_COMMENT}

Corresponding ACE version:

{ACE_FEW_SHOT_OUTPUT}

Now process this new comment. Output only JSON with the following structure:
{{
  "ace_sentences": [
    "First ACE sentence.",
    "Second ACE sentence.",
    "..."
  ]
}}

Here is the new comment to convert:

COMMENT:
{comment_text}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_instructions},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    content = response.choices[0].message.content
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse ACE JSON for comment: {exc}\nRaw: {content}")

    ace_sentences = parsed.get("ace_sentences") or []
    # Ensure we always return a list of strings
    return [str(s).strip() for s in ace_sentences if str(s).strip()]


def _encode_image_as_data_url(image_path: str) -> str:
    """Read an image file and return a data URL suitable for OpenAI vision inputs."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Visualization image not found: {image_path}")

    ext = os.path.splitext(image_path)[1].lower().lstrip(".") or "png"
    mime = f"image/{'jpeg' if ext in {'jpg', 'jpeg'} else ext}"

    with open(image_path, "rb") as img_f:
        b64 = base64.b64encode(img_f.read()).decode("ascii")

    return f"data:{mime};base64,{b64}"


def tag_ace_sentences_with_visualization(
    ace_sentences: List[str],
    image_path: str,
    client: OpenAI,
    model: str = "gpt-5.2",
) -> Dict[str, Dict[str, str]]:
    """
    Given ACE sentences and a visualization image, label each sentence with:
    - presence: 'present' (information visible in the visualization) or
      'not_present' (information not visible).
    - task_type: for present sentences, one of 'value_identification',
      'arithmetic_computation', or 'statistical_inference'; for not_present
      use 'not_applicable'.
    """
    if not ace_sentences:
        return {}

    data_url = _encode_image_as_data_url(image_path)

    # Build a numbered representation of the ACE sentences for clarity.
    numbered_ace = "\n".join(
        f"{i+1}. {sentence}" for i, sentence in enumerate(ace_sentences)
    )

    example_json_str = json.dumps(VIS_TAGGING_EXAMPLE_OUTPUT, indent=2)

    task_text = f"""You are given:
1. A set of sentences written in Attempto Controlled English (ACE) that describe a data visualization and related thoughts.
2. The actual data visualization image.

For each ACE sentence you must provide two labels:

(1) PRESENCE — whether the information in that sentence is:
- directly and explicitly present in the visualization (label "present"), or
- not directly visible in the visualization (label "not_present").

(2) TASK TYPE — for sentences labeled "present", classify the task into exactly one of:
- **value_identification**: read the data; retrieve value; find extremum; determine range. (The participant retrieves a single value or simple reading from the plot.)
- **arithmetic_computation**: read between the data; make comparisons; determine range. (The participant performs simple arithmetic or comparison over multiple values shown in the plot.)
- **statistical_inference**: read beyond the data; find correlations/trends; characterize distribution; find anomalies; find clusters; find extremum (in a distributional sense); make predictions; aggregate values; predict trend. (The participant estimates latent parameters or makes statistical judgments from the visual.)
For sentences labeled "not_present", set task_type to "not_applicable".

Use the following rules for presence:
- If the sentence describes numeric relationships, trends, categories, or other information that can be directly read or inferred from the visual alone, label it "present".
- If the sentence expresses personal opinions, interpretations, background knowledge, causes, consequences, hypothetical scenarios, or anything not strictly shown in the visual, label it "not_present".

Return a single JSON object where:
- Each key is exactly one of the ACE sentences (including its leading number).
- Each value is an object with two keys: "presence" (either "present" or "not_present") and "task_type" (either "value_identification", "arithmetic_computation", "statistical_inference", or "not_applicable").

Here is an example of the desired output format for a different ACE input:

{example_json_str}

Now here are the ACE sentences you must label for this visualization:

{numbered_ace}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at carefully reading data visualizations and aligning textual descriptions with what is visually present. Always return valid JSON.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    content = response.choices[0].message.content
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse visualization tagging JSON: {exc}\nRaw: {content}"
        )

    VALID_TASK_TYPES = {
        "value_identification",
        "arithmetic_computation",
        "statistical_inference",
        "not_applicable",
    }

    # Normalize values to the expected structure.
    normalized: Dict[str, Dict[str, str]] = {}
    for sent, val in parsed.items():
        if isinstance(val, dict):
            presence = str(val.get("presence", "")).strip().lower()
            task_type = str(val.get("task_type", "")).strip().lower()
        else:
            # Backward compatibility: plain string treated as presence only
            presence = str(val).strip().lower()
            task_type = "not_applicable"

        if presence not in {"present", "not_present"}:
            presence = "present" if "present" in presence else "not_present"
        if task_type not in VALID_TASK_TYPES:
            task_type = "not_applicable"
        if presence == "not_present":
            task_type = "not_applicable"

        normalized[str(sent)] = {
            "presence": presence,
            "task_type": task_type,
        }

    return normalized


def cluster_ace_sentences_for_article(
    article_id: str,
    extracted_dir: str = "extracted_responses",
    clusters_dir: str = "sentence_clusters",
    api_key: str | None = None,
    model: str = "gpt-5.2",
) -> Dict[str, Any]:
    """
    Stage 3: Cluster ACE sentences across all comments for a single article.

    This function:
      1. Loads extracted_responses/{article_id}.json (stage 1+2 output).
      2. Collects all ACE sentences from all comments.
      3. Uses the OpenAI API to cluster similar sentences together.
      4. Saves the clustering result to clusters_dir/{article_id}.json.

    Each cluster/category has:
      - category_id (integer, 1-based, unique within the article)
      - name (short, human-readable label)
      - member_sentences: list of objects with
          - sentence_index (1-based index in the combined ACE list)
          - comment_index (original comment index within the article)
          - sentence (the ACE sentence text)
    """
    input_path = os.path.join(extracted_dir, f"{article_id}.json")
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Stage 3 requires existing extracted responses file, but none was found at: {input_path}"
        )

    print(f"Reading extracted responses from {input_path} for clustering (stage 3)...")
    with open(input_path, "r", encoding="utf-8") as f:
        extracted = json.load(f)

    comments = extracted.get("comments") or []
    combined_sentences: List[Dict[str, Any]] = []

    for comment in comments:
        comment_index = comment.get("index")
        ace_sentences = comment.get("ace_sentences") or []
        for sentence in ace_sentences:
            raw_text = str(sentence).strip()
            if not raw_text:
                continue
            # Strip any leading numbering like "1.", "2)", "3 -", etc.
            clean_text = re.sub(r"^\s*\d+\s*[\.\)]?\s*", "", raw_text).strip()
            if not clean_text:
                continue
            combined_sentences.append(
                {
                    "comment_index": comment_index,
                    "sentence": clean_text,
                }
            )

    if not combined_sentences:
        raise ValueError(
            f"No ACE sentences found in extracted responses for article {article_id}; cannot perform clustering."
        )

    client = _ensure_client(api_key)

    # Prepare a numbered list of sentences (with comment indices) to send to the model for clustering.
    numbered_sentences_str_lines = []
    for idx, item in enumerate(combined_sentences, start=1):
        numbered_sentences_str_lines.append(
            f"{idx}. [comment_index={item['comment_index']}] {item['sentence']}"
        )
    numbered_sentences_str = "\n".join(numbered_sentences_str_lines)

    system_content = (
        "You are an expert qualitative researcher. "
        "You group similar short sentences into coherent, meaningful categories."
    )

    user_instructions = f"""You are given a list of ACE (Attempto Controlled English) sentences derived from comments on a single article.
Your task is to cluster similar sentences together into meaningful categories.

Here is the numbered list of sentences:

{numbered_sentences_str}

Instructions:
- Group sentences that express the same or very similar idea into the same category.
- You may create as many or as few categories as needed, but avoid single-sentence categories unless truly unique.
- Each category MUST have:
  - a numeric category_id (integer, starting at 1, unique within this article),
  - a short, human-readable name summarizing the theme of the sentences in that category.
- For each category, list which sentence indices belong to it.

IMPORTANT (STRICT CONSTRAINTS):
- Refer to sentences ONLY by their numeric indices from the list above.
- EVERY sentence index from 1 to {{total_sentences}} MUST appear in the categories.
- Each sentence index MUST appear in EXACTLY ONE category.
- You MUST NOT repeat the same sentence index in multiple categories.
- You MUST NOT skip any sentence index.
- If two sentences are almost identical but differ in minor wording, they should be in the same category (but still listed separately by their indices).

Return ONLY valid JSON with the following structure:
{{
  "article_id": "{article_id}",
  "total_sentences": <int>,  // this must equal the total number of sentences you clustered
  "categories": [
    {{
      "category_id": 1,
      "name": "Short descriptive name of the category",
      "member_sentence_indices": [1, 5, 7]
    }},
    {{
      "category_id": 2,
      "name": "Another theme",
      "member_sentence_indices": [2, 3, 4]
    }}
  ]
}}

Do not include any explanation outside of this JSON.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_instructions},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    content = response.choices[0].message.content
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse clustering JSON for article {article_id}: {exc}\nRaw: {content}"
        )

    # Basic validation and normalization.
    categories = parsed.get("categories") or []
    if not isinstance(categories, list) or not categories:
        raise ValueError(
            f"Clustering result for article {article_id} does not contain non-empty 'categories' list."
        )

    # Ensure each sentence index appears at most once across all categories.
    seen_indices: set[int] = set()
    for cat in categories:
        indices = cat.get("member_sentence_indices") or []
        if not isinstance(indices, list):
            raise ValueError(
                f"Category member_sentence_indices must be a list in article {article_id}."
            )
        for idx in indices:
            if not isinstance(idx, int):
                raise ValueError(
                    f"Sentence indices must be integers, got {idx!r} in article {article_id}."
                )
            if idx < 1 or idx > len(combined_sentences):
                raise ValueError(
                    f"Sentence index {idx} out of range for article {article_id}."
                )
            if idx in seen_indices:
                # Do not fail hard; keep the first assignment and ignore duplicates.
                print(
                    f"⚠ Warning: sentence index {idx} appears in multiple categories for article {article_id}. "
                    "Keeping its first category assignment and ignoring subsequent ones."
                )
                continue
            seen_indices.add(idx)

    if len(seen_indices) != len(combined_sentences):
        # Soft warning: allow saving even if some sentences were not assigned to any category.
        print(
            f"⚠ Warning: not all sentences were assigned to a category for article {article_id}. "
            f"Expected {len(combined_sentences)}, got {len(seen_indices)}."
        )

    # Augment categories with full sentence information for convenience.
    enriched_categories: List[Dict[str, Any]] = []
    for cat in categories:
        indices = cat.get("member_sentence_indices") or []
        members: List[Dict[str, Any]] = []
        for idx in indices:
            sentence_info = combined_sentences[idx - 1]
            members.append(
                {
                    "sentence_index": idx,
                    "comment_index": sentence_info.get("comment_index"),
                    "sentence": sentence_info.get("sentence"),
                }
            )
        enriched_categories.append(
            {
                "category_id": cat.get("category_id"),
                "name": cat.get("name"),
                "member_sentence_indices": indices,
                "member_sentences": members,
            }
        )

    clustering_result: Dict[str, Any] = {
        "article_id": article_id,
        "total_sentences": len(combined_sentences),
        "categories": enriched_categories,
    }

    os.makedirs(clusters_dir, exist_ok=True)
    output_path = os.path.join(clusters_dir, f"{article_id}.json")
    print(f"Saving stage 3 clustering result to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clustering_result, f, indent=2, ensure_ascii=False)

    print(f"✓ Successfully clustered ACE sentences for article {article_id}")
    return clustering_result


def process_article(
    article_id: str,
    articles_data_dir: str = "articles_data",
    output_dir: str = "extracted_responses",
    api_key: str | None = None,
    model: str = "gpt-5.2",
    skip_stage1: bool = False,
) -> Dict[str, Any]:
    """
    Process a single article's comments using the two-stage pipeline.

    For each top-level comment (ignoring replies):
      1. Convert the comment text to ACE sentences (unless skip_stage1=True).
      2. Use the article's visualization image to tag each ACE sentence as
         'present' or 'not_present'.

    When skip_stage1 is True, ACE sentences are loaded from existing
    ace_comments/{article_id}/{idx}.json files and only stage 2 (tagging) runs.
    """

    json_path = os.path.join(articles_data_dir, f"{article_id}.json")

    # Check if file exists
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # Read file
    print(f"Reading {json_path}...")
    comments_data = read_json_file(json_path)

    client = _ensure_client(api_key)

    # Assume images/{article_id}.png as the visualization path.
    image_path = os.path.join("../data/images", f"{article_id}.png")
    image_available = os.path.exists(image_path)
    if not image_available:
        print(
            f"⚠ Visualization image not found for article {article_id} at {image_path}. "
            "Stage 2 (visualization tagging) will be skipped for this article."
        )

    # Directories for per-comment outputs grouped by article ID.
    ace_comments_base_dir = "ace_comments"
    tagged_comments_base_dir = "tagged_comments"
    ace_article_dir = os.path.join(ace_comments_base_dir, str(article_id))
    tagged_article_dir = os.path.join(tagged_comments_base_dir, str(article_id))
    if not skip_stage1:
        os.makedirs(ace_article_dir, exist_ok=True)
    os.makedirs(tagged_article_dir, exist_ok=True)

    processed_comments: List[Dict[str, Any]] = []

    for idx, comment in enumerate(comments_data, start=1):
        raw_comment = str(comment.get("comment info", "") or "").strip()
        if not raw_comment:
            continue

        if skip_stage1:
            # Load ACE sentences from existing ace_comments/{article_id}/{idx}.json
            ace_output_path = os.path.join(ace_article_dir, f"{idx}.json")
            if not os.path.exists(ace_output_path):
                print(
                    f"⚠ Skipping comment {idx}: no ACE file at {ace_output_path}. "
                    "Run without --skip-stage1 to generate ACE first."
                )
                continue
            with open(ace_output_path, "r", encoding="utf-8") as ace_f:
                ace_data = json.load(ace_f)
            ace_sentences = ace_data.get("ace_sentences") or []
            if not ace_sentences:
                print(f"⚠ Skipping comment {idx}: ACE file has no ace_sentences.")
                continue
            print(f"Processing comment {idx} for article {article_id} (tagging only, ACE from file)...")
        else:
            print(f"Processing comment {idx} for article {article_id} (ACE conversion)...")
            ace_sentences = generate_ace_for_comment(
                raw_comment, client=client, model=model
            )

            # Save ACE conversion for this comment under ace_comments/{article_id}/{idx}.json
            ace_output = {
                "article_id": article_id,
                "comment_index": idx,
                "raw_comment": raw_comment,
                "ace_sentences": ace_sentences,
            }
            ace_output_path = os.path.join(ace_article_dir, f"{idx}.json")
            with open(ace_output_path, "w", encoding="utf-8") as ace_f:
                json.dump(ace_output, ace_f, indent=2, ensure_ascii=False)

        presence_tags: Dict[str, Dict[str, str]] = {}
        if image_available and ace_sentences:
            print(
                f"Tagging ACE sentences for comment {idx} using visualization image..."
            )
            try:
                # We include the numbering here so that keys clearly match what
                # the model sees.
                numbered_ace_sentences = [
                    f"{i+1}. {sentence}" for i, sentence in enumerate(ace_sentences)
                ]
                presence_tags = tag_ace_sentences_with_visualization(
                    numbered_ace_sentences, image_path=image_path, client=client, model=model
                )

                # Save tagging result for this comment under
                # tagged_comments/{article_id}/{idx}.json
                tagged_output = {
                    "article_id": article_id,
                    "comment_index": idx,
                    "image_path": image_path,
                    "numbered_ace_sentences": numbered_ace_sentences,
                    "presence_tags": presence_tags,
                }
                tagged_output_path = os.path.join(tagged_article_dir, f"{idx}.json")
                with open(tagged_output_path, "w", encoding="utf-8") as tag_f:
                    json.dump(tagged_output, tag_f, indent=2, ensure_ascii=False)
            except Exception as exc:
                print(
                    f"✗ Error tagging visualization presence for article {article_id}, "
                    f"comment {idx}: {exc}"
                )

        processed_comments.append(
            {
                "index": idx,
                "metadata": {
                    "name": comment.get("name"),
                    "location": comment.get("location"),
                    "date posted": comment.get("date posted"),
                },
                "raw_comment": raw_comment,
                "ace_sentences": ace_sentences,
                "presence_tags": presence_tags,
            }
        )

    result: Dict[str, Any] = {
        "article_id": article_id,
        "source_file": json_path,
        "image_path": image_path if image_available else None,
        "comments": processed_comments,
    }

    # Save result
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{article_id}.json")

    print(f"Saving result to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✓ Successfully processed article {article_id}")
    return result


def main():
    """Main function to process articles."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Process NYT article comments through a two-stage ACE + visualization "
            "tagging pipeline using the OpenAI API."
        )
    )
    parser.add_argument(
        "--article_id",
        type=str,
        help="Article ID (e.g., '38')",
    )
    parser.add_argument(
        "--articles-data-dir",
        type=str,
        default="../data/comment_data",
        help="Directory containing article JSON files (default: data/comment_data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="extracted_responses",
        help="Output directory for extracted responses (default: extracted_responses)",
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
        help="OpenAI model to use (default: gpt-5.2)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all articles found in articles_data directory",
    )
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Skip ACE extraction; load existing ace_comments and run only stage 2 (tagging)",
    )
    parser.add_argument(
        "--run-stage3",
        action="store_true",
        help="After stages 1 and 2, run stage 3 (clustering ACE sentences per article).",
    )
    parser.add_argument(
        "--only-stage3",
        action="store_true",
        help=(
            "Run only stage 3 (clustering) using existing extracted_responses JSON files; "
            "do not run stages 1 or 2."
        ),
    )
    parser.add_argument(
        "--clusters-dir",
        type=str,
        default="sentence_clusters",
        help="Output directory for stage 3 clustering results (default: sentence_clusters)",
    )

    args = parser.parse_args()

    if args.all:
        # Process all articles
        if args.only_stage3:
            # Stage 3 only: read from extracted_responses (or custom output-dir) and cluster.
            extracted_path = Path(args.output_dir)
            article_ids = [p.stem for p in extracted_path.glob("*.json")]
            print(
                f"Found {len(article_ids)} extracted response files to cluster (stage 3 only)"
            )
            for article_id in sorted(article_ids):
                try:
                    cluster_ace_sentences_for_article(
                        article_id=article_id,
                        extracted_dir=args.output_dir,
                        clusters_dir=args.clusters_dir,
                        api_key=args.api_key,
                        model=args.model,
                    )
                except Exception as e:
                    print(f"✗ Error clustering article {article_id}: {e}")
                    continue
        else:
            articles_data_path = Path(args.articles_data_dir)
            article_ids = []

            for json_file in articles_data_path.glob("*.json"):
                article_id = json_file.stem
                article_ids.append(article_id)

            print(f"Found {len(article_ids)} articles to process")

            for article_id in sorted(article_ids):
                try:
                    process_article(
                        article_id,
                        args.articles_data_dir,
                        args.output_dir,
                        args.api_key,
                        args.model,
                        skip_stage1=args.skip_stage1,
                    )
                    if args.run_stage3:
                        cluster_ace_sentences_for_article(
                            article_id=article_id,
                            extracted_dir=args.output_dir,
                            clusters_dir=args.clusters_dir,
                            api_key=args.api_key,
                            model=args.model,
                        )
                except Exception as e:
                    print(f"✗ Error processing article {article_id}: {e}")
                    continue
    else:
        # Process single article
        if args.only_stage3:
            if not args.article_id:
                print(
                    "✗ Error: --only-stage3 requires --article_id (or use --all to cluster all)."
                )
                sys.exit(1)
            try:
                cluster_ace_sentences_for_article(
                    article_id=args.article_id,
                    extracted_dir=args.output_dir,
                    clusters_dir=args.clusters_dir,
                    api_key=args.api_key,
                    model=args.model,
                )
            except Exception as e:
                print(f"✗ Error during stage 3 clustering: {e}")
                sys.exit(1)
        else:
            try:
                process_article(
                    args.article_id,
                    articles_data_dir=args.articles_data_dir,
                    output_dir=args.output_dir,
                    api_key=args.api_key,
                    model=args.model,
                    skip_stage1=args.skip_stage1,
                )
                if args.run_stage3:
                    cluster_ace_sentences_for_article(
                        article_id=args.article_id,
                        extracted_dir=args.output_dir,
                        clusters_dir=args.clusters_dir,
                        api_key=args.api_key,
                        model=args.model,
                    )
            except Exception as e:
                print(f"✗ Error: {e}")
                sys.exit(1)


if __name__ == "__main__":
    main()
