"""OpenAI-powered analysis of cross-language embedding retrieval differences."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

UI_DIMENSIONS: list[dict[str, str]] = [
    {"id": "semantic", "label": "Semantic", "scoreKey": "semanticScore"},
    {"id": "chronology", "label": "Chronology", "scoreKey": "chronologyScore"},
    {"id": "emotion", "label": "Emotion", "scoreKey": "emotionScore"},
    {"id": "style", "label": "Style", "scoreKey": "styleScore"},
    {"id": "metaphor", "label": "Metaphor", "scoreKey": "metaphorScore"},
    {"id": "tone", "label": "Tone", "scoreKey": "toneScore"},
    {"id": "voice", "label": "Character Voice", "scoreKey": "voiceScore"},
    {"id": "cultural", "label": "Cultural", "scoreKey": "culturalScore"},
]

_HIGHLIGHT_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "phrase": {"type": "string"},
        "dimension": {"type": "string"},
    },
    "required": ["phrase", "dimension"],
    "additionalProperties": False,
}

_LANGUAGE_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "matchedConcepts": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reasoning": {"type": "string"},
        "chronologyScore": {"type": "integer"},
        "semanticScore": {"type": "integer"},
        "emotionScore": {"type": "integer"},
        "styleScore": {"type": "integer"},
        "metaphorScore": {"type": "integer"},
        "toneScore": {"type": "integer"},
        "voiceScore": {"type": "integer"},
        "culturalScore": {"type": "integer"},
        "highlights": {
            "type": "array",
            "items": _HIGHLIGHT_ITEM_SCHEMA,
        },
    },
    "required": [
        "matchedConcepts",
        "reasoning",
        "chronologyScore",
        "semanticScore",
        "emotionScore",
        "styleScore",
        "metaphorScore",
        "toneScore",
        "voiceScore",
        "culturalScore",
        "highlights",
    ],
    "additionalProperties": False,
}

EMBEDDING_COMPARISON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "english": _LANGUAGE_ANALYSIS_SCHEMA,
        "german": _LANGUAGE_ANALYSIS_SCHEMA,
        "differences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "english": {"type": "string"},
                    "german": {"type": "string"},
                    "impact": {"type": "string"},
                },
                "required": ["category", "english", "german", "impact"],
                "additionalProperties": False,
            },
        },
        "finalConclusion": {"type": "string"},
    },
    "required": ["summary", "english", "german", "differences", "finalConclusion"],
    "additionalProperties": False,
}


def _empty_language_analysis(reason: str) -> dict[str, Any]:
    """Return a valid empty language analysis block."""
    return {
        "matchedConcepts": [],
        "reasoning": reason,
        "chronologyScore": 0,
        "semanticScore": 0,
        "emotionScore": 0,
        "styleScore": 0,
        "metaphorScore": 0,
        "toneScore": 0,
        "voiceScore": 0,
        "culturalScore": 0,
        "highlights": [],
    }


def _empty_llm_analysis(summary: str, conclusion: str, reason: str) -> dict[str, Any]:
    """Return a valid empty LLM analysis payload."""
    return {
        "summary": summary,
        "english": _empty_language_analysis(reason),
        "german": _empty_language_analysis(reason),
        "differences": [],
        "finalConclusion": conclusion,
    }


def _clamp_score(value: Any) -> int:
    """Clamp a dimension score to 0-100."""
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, score))


def _locate_highlights(text: str, highlights: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach start/end offsets so Angular can highlight phrases in the paragraph."""
    located: list[dict[str, Any]] = []
    haystack = text.lower()
    used_spans: list[tuple[int, int]] = []

    for item in highlights:
        phrase = str(item.get("phrase") or "").strip()
        dimension = str(item.get("dimension") or "semantic")
        if not phrase:
            continue

        search_from = 0
        start = -1
        needle = phrase.lower()
        while True:
            found = haystack.find(needle, search_from)
            if found < 0:
                break
            end = found + len(phrase)
            overlaps = any(found < used_end and end > used_start for used_start, used_end in used_spans)
            if not overlaps:
                start = found
                break
            search_from = found + 1

        record = {
            "phrase": phrase,
            "dimension": dimension,
            "start": start if start >= 0 else None,
            "end": (start + len(phrase)) if start >= 0 else None,
        }
        if start >= 0:
            used_spans.append((start, start + len(phrase)))
        located.append(record)

    return located


def _scores_from_analysis(analysis: dict[str, Any]) -> dict[str, int]:
    """Flatten language analysis scores for Angular bindings."""
    return {
        dimension["id"]: _clamp_score(analysis.get(dimension["scoreKey"], 0))
        for dimension in UI_DIMENSIONS
    }


def _passage_side(
    chunk: dict[str, Any],
    analysis: dict[str, Any],
    retrieval_score: float,
) -> dict[str, Any]:
    """Build one language panel for the Angular explorer."""
    content = chunk.get("content") or ""
    return {
        "chunkId": chunk.get("chunk_id") or "",
        "correspondingChunk": chunk.get("corresponding_chunk") or "",
        "nodeId": chunk.get("id") or "",
        "name": chunk.get("name") or "",
        "content": content,
        "summary": chunk.get("summary") or "",
        "keywords": chunk.get("keywords") or [],
        "actId": chunk.get("act_id"),
        "sceneId": chunk.get("scene_id"),
        "actName": chunk.get("act_name") or "",
        "sceneName": chunk.get("scene_name") or "",
        "topicPath": chunk.get("topic_path") or "",
        "retrievalScore": retrieval_score,
        "matchedConcepts": analysis.get("matchedConcepts") or [],
        "highlights": _locate_highlights(content, analysis.get("highlights") or []),
        "reasoning": analysis.get("reasoning") or "",
        "scores": _scores_from_analysis(analysis),
    }


def _dimension_rows(english: dict[str, Any], german: dict[str, Any]) -> list[dict[str, Any]]:
    """Build side-by-side dimension rows for radar charts and score bars."""
    english_scores = english.get("scores") or {}
    german_scores = german.get("scores") or {}
    return [
        {
            "id": dimension["id"],
            "label": dimension["label"],
            "english": english_scores.get(dimension["id"], 0),
            "german": german_scores.get(dimension["id"], 0),
        }
        for dimension in UI_DIMENSIONS
    ]


def build_ui_passage(
    row: dict[str, Any],
    english_chunk: dict[str, Any],
    german_chunk: dict[str, Any],
    llm_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one Angular passage record from retrieval + LLM analysis."""
    analysis = llm_analysis or _empty_llm_analysis(
        summary="LLM analysis has not been run yet.",
        conclusion="Run the comparison batch to fill dimension scores and highlights.",
        reason="Pending LLM analysis.",
    )
    pair_id = row.get("pair_id") or 0
    english = _passage_side(
        english_chunk,
        analysis.get("english") or {},
        float(row.get("english_score") or 0),
    )
    german = _passage_side(
        german_chunk,
        analysis.get("german") or {},
        float(row.get("german_score") or 0),
    )
    return {
        "id": f"pair-{pair_id}",
        "pairId": pair_id,
        "label": f"{english.get('name') or 'English'} / {german.get('name') or 'German'}",
        "englishQuery": row.get("english_query") or "",
        "germanQuery": row.get("german_query") or "",
        "structurallyAligned": bool(row.get("structurally_aligned")),
        "retrievalSimilarity": row.get("retrieval_similarity") or 0,
        "semanticDifference": row.get("semantic_difference") or "",
        "emotionalDifference": row.get("emotional_difference") or "",
        "translationDifference": row.get("translation_difference") or "",
        "english": english,
        "german": german,
        "dimensions": _dimension_rows(english, german),
        "differences": analysis.get("differences") or [],
        "summary": analysis.get("summary") or "",
        "finalConclusion": analysis.get("finalConclusion") or "",
    }


def build_ui_payload(passages: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap passage records in the JSON document Angular should load."""
    return {
        "title": "Hamlet Translation Explorer",
        "dimensionKeys": [
            {"id": dimension["id"], "label": dimension["label"]}
            for dimension in UI_DIMENSIONS
        ],
        "passages": passages,
    }


def _build_prompt(
    english_query: str,
    german_query: str,
    english_paragraph: str,
    german_paragraph: str,
) -> str:
    """Build the analysis prompt for the OpenAI model."""
    dimension_list = ", ".join(dimension["id"] for dimension in UI_DIMENSIONS)
    return f"""
You are an expert in multilingual embedding models and literary translation.

Two semantically equivalent queries were embedded independently.
The embedding model retrieved different paragraphs.

Your job is NOT to judge whether the retrieval is correct.
Explain WHY the embedding model likely preferred different paragraphs,
and score each retrieved paragraph so a UI can visualize the comparison.

Score every dimension from 0 to 100 for how strongly that paragraph matches
its query on that dimension:

{dimension_list}

Highlights must be short phrases copied VERBATIM from the retrieved paragraph
so the UI can highlight them in the original text. Set dimension to one of:
{dimension_list}

Consider:
- semantic overlap
- repeated concepts
- chronology
- lexical differences
- translation nuances
- contextual emphasis
- multilingual embedding behavior
- emotion, metaphor, tone, character voice, and cultural language

English Query:
{english_query}

German Query:
{german_query}

English Retrieved Paragraph:
{english_paragraph}

German Retrieved Paragraph:
{german_paragraph}

Return ONLY valid JSON.
""".strip()


def compare_embedding_results(
    english_query: str,
    german_query: str,
    english_paragraph: str,
    german_paragraph: str,
    *,
    client: Any | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Explain why the embedding model preferred different retrieved paragraphs."""
    from openai import OpenAI

    openai_client = client or OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    selected_model = model or os.environ.get("OPENAI_COMPARISON_MODEL", "gpt-5")
    prompt = _build_prompt(english_query, german_query, english_paragraph, german_paragraph)

    response = openai_client.responses.create(
        model=selected_model,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "embedding_comparison",
                "schema": EMBEDDING_COMPARISON_SCHEMA,
                "strict": True,
            }
        },
    )

    return json.loads(response.output_text)


def load_chunk_index(chunks_path: Path) -> dict[str, dict[str, Any]]:
    """Index flat chunk records by chunk_id."""
    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    return {item["chunk_id"]: item for item in payload}


def analyze_comparison_results(
    comparison_rows: list[dict[str, Any]],
    english_chunks_path: Path,
    german_chunks_path: Path,
    *,
    client: Any | None = None,
    model: str | None = None,
    run_llm: bool = True,
) -> dict[str, Any]:
    """Build Angular-ready JSON for each English/German retrieval comparison row."""
    english_chunks = load_chunk_index(english_chunks_path)
    german_chunks = load_chunk_index(german_chunks_path)
    passages: list[dict[str, Any]] = []

    for row in comparison_rows:
        pair_id = row.get("pair_id")
        english_chunk_id = row.get("english_chunk_id", "")
        german_chunk_id = row.get("german_chunk_id", "")
        english_chunk = english_chunks.get(english_chunk_id, {})
        german_chunk = german_chunks.get(german_chunk_id, {})

        english_paragraph = english_chunk.get("content", "")
        german_paragraph = german_chunk.get("content", "")

        if not english_paragraph or not german_paragraph:
            logger.warning(
                "Skipping pair %s: missing paragraph content (%s, %s)",
                pair_id,
                english_chunk_id,
                german_chunk_id,
            )
            passages.append(
                build_ui_passage(
                    row,
                    english_chunk,
                    german_chunk,
                    _empty_llm_analysis(
                        summary="Skipped because retrieved paragraph content was missing.",
                        conclusion="Analysis skipped due to missing chunk content.",
                        reason="No paragraph content available.",
                    ),
                )
            )
            continue

        if not run_llm:
            passages.append(build_ui_passage(row, english_chunk, german_chunk))
            continue

        logger.info("Running LLM comparison for pair %s", pair_id)
        llm_analysis = compare_embedding_results(
            english_query=row["english_query"],
            german_query=row["german_query"],
            english_paragraph=english_paragraph,
            german_paragraph=german_paragraph,
            client=client,
            model=model,
        )
        passages.append(build_ui_passage(row, english_chunk, german_chunk, llm_analysis))

    return build_ui_payload(passages)
