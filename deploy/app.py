"""Lightweight FastAPI app for free-tier deployment (Render / Docker).

Serves Hamlet comparison JSON and OpenAI-backed analysis endpoints without
loading CLIP, BART, or other heavy models at startup.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
PROJECT_DIR = ROOT_DIR / "project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from llm_comparison import analyze_comparison_results, compare_embedding_results

DATA_DIR = Path(os.environ.get("DATA_DIR", APP_DIR / "data"))
ENGLISH_CHUNKS_JSON_PATH = DATA_DIR / "english_chunks.json"
GERMAN_CHUNKS_JSON_PATH = DATA_DIR / "german_chunks.json"
COMPARISON_RESULTS_JSON_PATH = DATA_DIR / "comparison_results.json"
EMBEDDING_COMPARISON_RESULTS_JSON_PATH = DATA_DIR / "embedding_comparison_results.json"
HAMLET_4X4_JSON_PATH = DATA_DIR / "hamlet_4x4_comparison.json"

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("WARNING: OPENAI_API_KEY is not set. LLM endpoints will fail.")

openai_client = OpenAI(api_key=api_key) if api_key else None
_query_model = None

app = FastAPI(
    title="Hamlet Translation API",
    description="Hamlet English/German embedding comparison and literary analysis API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EmbeddingComparisonRequest(BaseModel):
    english_query: str
    german_query: str
    english_paragraph: str
    german_paragraph: str
    model: str = "gpt-4o"


class EmbeddingComparisonHighlight(BaseModel):
    phrase: str
    dimension: str


class EmbeddingComparisonLanguageDetail(BaseModel):
    matchedConcepts: List[str]
    reasoning: str
    chronologyScore: int
    semanticScore: int
    emotionScore: int
    styleScore: int
    metaphorScore: int
    toneScore: int
    voiceScore: int
    culturalScore: int
    highlights: List[EmbeddingComparisonHighlight]


class EmbeddingComparisonDifference(BaseModel):
    category: str
    english: str
    german: str
    impact: str


class EmbeddingComparisonResponse(BaseModel):
    summary: str
    english: EmbeddingComparisonLanguageDetail
    german: EmbeddingComparisonLanguageDetail
    differences: List[EmbeddingComparisonDifference]
    finalConclusion: str


class EmbeddingRetrievalComparisonRequest(BaseModel):
    english_query: str
    german_query: str
    model: str = "gpt-4o"


class EmbeddingRetrievalComparisonResponse(BaseModel):
    english_query: str
    german_query: str
    english_chunk_id: str
    german_chunk_id: str
    english_paragraph: str
    german_paragraph: str
    analysis: EmbeddingComparisonResponse


def _require_openai() -> OpenAI:
    if openai_client is None:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured on the server.",
        )
    return openai_client


def _get_query_model():
    global _query_model
    if _query_model is None:
        from sentence_transformers import SentenceTransformer

        model_name = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        _query_model = SentenceTransformer(model_name)
    return _query_model


def _load_chunk_corpus(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path.name}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _retrieve_top_chunk(query: str, chunks: list) -> dict:
    query_model = _get_query_model()
    query_embedding = query_model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
    query_vec = np.asarray(query_embedding, dtype=np.float64).ravel()

    best_chunk = None
    best_score = -1.0
    for chunk in chunks:
        embedding = chunk.get("embeddings")
        if not embedding:
            continue
        chunk_vec = np.asarray(embedding, dtype=np.float64).ravel()
        denom = np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec)
        score = float(np.dot(query_vec, chunk_vec) / denom) if denom > 1e-12 else 0.0
        if score > best_score:
            best_score = score
            best_chunk = chunk

    if best_chunk is None:
        raise HTTPException(status_code=404, detail="No retrievable chunks found")
    return best_chunk


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/")
async def health():
    return {
        "status": "ok",
        "service": "hamlet-translation-api",
        "data_dir": str(DATA_DIR),
        "files": {
            "english_chunks": ENGLISH_CHUNKS_JSON_PATH.exists(),
            "german_chunks": GERMAN_CHUNKS_JSON_PATH.exists(),
            "comparison_results": COMPARISON_RESULTS_JSON_PATH.exists(),
            "embedding_comparison_results": EMBEDDING_COMPARISON_RESULTS_JSON_PATH.exists(),
            "hamlet_4x4_comparison": HAMLET_4X4_JSON_PATH.exists(),
        },
        "openai_configured": openai_client is not None,
    }


@app.get("/ui/comparison")
async def get_comparison_ui(refresh: bool = False):
    """Return Angular-ready Hamlet comparison JSON."""
    try:
        if EMBEDDING_COMPARISON_RESULTS_JSON_PATH.exists() and not refresh:
            payload = _load_json(EMBEDDING_COMPARISON_RESULTS_JSON_PATH)
            if isinstance(payload, dict) and "passages" in payload:
                return payload

        comparison_rows = _load_json(COMPARISON_RESULTS_JSON_PATH)
        payload = analyze_comparison_results(
            comparison_rows=comparison_rows,
            english_chunks_path=ENGLISH_CHUNKS_JSON_PATH,
            german_chunks_path=GERMAN_CHUNKS_JSON_PATH,
            run_llm=False,
        )
        EMBEDDING_COMPARISON_RESULTS_JSON_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return payload
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/hamlet/4x4-comparison")
async def get_hamlet_4x4_comparison():
    """Return the 4×4 directional literary comparison JSON."""
    try:
        return _load_json(HAMLET_4X4_JSON_PATH)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/compare-embedding-results", response_model=EmbeddingComparisonResponse)
async def compare_embedding_results_api(request: EmbeddingComparisonRequest):
    try:
        result = compare_embedding_results(
            english_query=request.english_query.strip(),
            german_query=request.german_query.strip(),
            english_paragraph=request.english_paragraph.strip(),
            german_paragraph=request.german_paragraph.strip(),
            client=_require_openai(),
            model=request.model,
        )
        return EmbeddingComparisonResponse(**result)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post(
    "/compare-embedding-retrieval",
    response_model=EmbeddingRetrievalComparisonResponse,
)
async def compare_embedding_retrieval_api(request: EmbeddingRetrievalComparisonRequest):
    try:
        english_chunks = _load_chunk_corpus(ENGLISH_CHUNKS_JSON_PATH)
        german_chunks = _load_chunk_corpus(GERMAN_CHUNKS_JSON_PATH)

        english_chunk = _retrieve_top_chunk(request.english_query.strip(), english_chunks)
        german_chunk = _retrieve_top_chunk(request.german_query.strip(), german_chunks)

        analysis = compare_embedding_results(
            english_query=request.english_query.strip(),
            german_query=request.german_query.strip(),
            english_paragraph=english_chunk.get("content", ""),
            german_paragraph=german_chunk.get("content", ""),
            client=_require_openai(),
            model=request.model,
        )

        return EmbeddingRetrievalComparisonResponse(
            english_query=request.english_query,
            german_query=request.german_query,
            english_chunk_id=english_chunk.get("chunk_id", ""),
            german_chunk_id=german_chunk.get("chunk_id", ""),
            english_paragraph=english_chunk.get("content", ""),
            german_paragraph=german_chunk.get("content", ""),
            analysis=EmbeddingComparisonResponse(**analysis),
        )
    except HTTPException:
        raise
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/compare-embedding-batch")
async def compare_embedding_batch_api(model: str = "gpt-4o"):
    """Run LLM analysis for all rows in comparison_results.json."""
    try:
        comparison_rows = _load_json(COMPARISON_RESULTS_JSON_PATH)
        payload = analyze_comparison_results(
            comparison_rows=comparison_rows,
            english_chunks_path=ENGLISH_CHUNKS_JSON_PATH,
            german_chunks_path=GERMAN_CHUNKS_JSON_PATH,
            client=_require_openai(),
            model=model,
        )
        EMBEDDING_COMPARISON_RESULTS_JSON_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {
            "count": len(payload.get("passages", [])),
            "output_file": str(EMBEDDING_COMPARISON_RESULTS_JSON_PATH),
            "results": payload,
        }
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
