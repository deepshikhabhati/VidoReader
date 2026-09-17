"""Lazy-loaded corpora and models for app.py (local + Render deployment)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", os.environ.get("DATA_DIR", ROOT_DIR)))


def data_path(relative: str) -> Path:
    for base in (DATA_ROOT, ROOT_DIR):
        candidate = base / relative
        if candidate.exists():
            return candidate
    return DATA_ROOT / relative


def concatenate_values(data: list) -> str:
    return " ".join(item["value"] for item in data)


def extract_content_and_embeddings(data, parent_path: str = "") -> list:
    results: list = []
    for item in data:
        current_path = f"{parent_path}/{item['name']}" if parent_path else item["name"]
        if "content" in item and item["content"] and "embeddings" in item:
            content = item["content"]
            results.append(
                {
                    "path": current_path,
                    "content": content,
                    "embeddings": item["embeddings"],
                }
            )
        else:
            results.append(
                {
                    "path": current_path,
                    "content": item.get("value", ""),
                    "embeddings": item.get("embeddings"),
                }
            )
        if "children" in item:
            results.extend(extract_content_and_embeddings(item["children"], current_path))
    return results


def extract_hamlet_embeddings(data, parent_path: str = "") -> list:
    results: list = []
    for index, item in enumerate(data):
        if "name" in item:
            current_path = f"{parent_path}/{item['name']}" if parent_path else item["name"]
        else:
            item_name = f"Item_{index + 1}"
            current_path = f"{parent_path}/{item_name}" if parent_path else item_name
        if "content" in item and item["content"] and "embeddings" in item:
            results.append(
                {
                    "path": current_path,
                    "content": item["content"],
                    "embeddings": item["embeddings"],
                }
            )
        for key in ("sub_chapters", "paragraphs", "children"):
            if key in item:
                results.extend(extract_hamlet_embeddings(item[key], current_path))
    return results


_cache: dict[str, Any] = {}
_query_model = None
_clip_bundle: Optional[tuple] = None
_image_index: Optional[dict] = None


def _load_json(relative: str) -> Any:
    with data_path(relative).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_flat_corpus(relative: str, cache_key: str, extractor: Callable) -> list:
    if cache_key in _cache:
        return _cache[cache_key]
    flat_path = data_path(relative.replace(".json", "_flat.json"))
    if flat_path.exists():
        _cache[cache_key] = json.loads(flat_path.read_text(encoding="utf-8"))
        return _cache[cache_key]
    raw = _load_json(relative)
    if isinstance(raw, list):
        _cache[cache_key] = extractor(raw)
    else:
        _cache[cache_key] = extractor(raw)
    return _cache[cache_key]


def get_content_embeddings3() -> list:
    if "content_embeddings3" in _cache:
        return _cache["content_embeddings3"]
    precomputed = data_path("content_embeddings3.json")
    if precomputed.exists():
        _cache["content_embeddings3"] = json.loads(precomputed.read_text(encoding="utf-8"))
        return _cache["content_embeddings3"]
    ai_data = _load_json("AI_wiki_embeddings.json")
    _cache["content_embeddings3"] = extract_content_and_embeddings(ai_data)
    return _cache["content_embeddings3"]


def get_content_embeddings() -> list:
    if "content_embeddings" in _cache:
        return _cache["content_embeddings"]
    precomputed = data_path("content_embeddings.json")
    if precomputed.exists():
        _cache["content_embeddings"] = json.loads(precomputed.read_text(encoding="utf-8"))
        return _cache["content_embeddings"]
    raw = _load_json("version1_Shakes_final.json")
    _cache["content_embeddings"] = extract_content_and_embeddings(raw)
    return _cache["content_embeddings"]


def get_content_embeddings2() -> list:
    if "content_embeddings2" in _cache:
        return _cache["content_embeddings2"]
    raw = _load_json("version2_Shakes_final.json")
    _cache["content_embeddings2"] = extract_content_and_embeddings(raw)
    return _cache["content_embeddings2"]


def get_hamlet_embeddings1() -> list:
    return _load_flat_corpus(
        "hamlet_version1.json",
        "hamlet_embeddings1",
        extract_content_and_embeddings,
    )


def get_hamlet_embeddings2() -> list:
    return _load_flat_corpus(
        "hamlet_version2.json",
        "hamlet_embeddings2",
        extract_content_and_embeddings,
    )


def get_hamlet_embeddings3() -> list:
    return _load_flat_corpus(
        "hamlet_version3.json",
        "hamlet_embeddings3",
        extract_content_and_embeddings,
    )


def get_german_hamlet_embeddings() -> list:
    if "german_hamlet_embeddings" in _cache:
        return _cache["german_hamlet_embeddings"]
    raw = _load_json("german_hamlet_embeddings.json")
    _cache["german_hamlet_embeddings"] = extract_content_and_embeddings(raw)
    return _cache["german_hamlet_embeddings"]


def _fr_node_label(item: dict) -> str:
    return (
        item.get("name")
        or item.get("chapter_title")
        or item.get("subchapter_title")
        or item.get("chunk_name")
        or "unnamed"
    )


def _fr_node_text(item: dict) -> str:
    value = item.get("value") or item.get("content")
    if value:
        return value if isinstance(value, str) else str(value)
    return ""


def _fr_child_list(item: dict) -> list:
    return item.get("children") or item.get("subchapters") or []


def _fr_concatenate_values(children: list) -> str:
    if not children:
        return ""
    parts: list[str] = []
    for child in children:
        if isinstance(child, dict):
            text = _fr_node_text(child)
            if text:
                parts.append(text)
            nested = _fr_child_list(child)
            if nested:
                sub = _fr_concatenate_values(nested)
                if sub:
                    parts.append(sub)
    return "\n\n".join(parts)


def _fr_extract_flat(data, parent_path: str = "") -> list:
    results: list = []
    if isinstance(data, dict):
        if "chapters" in data:
            return _fr_extract_flat(data["chapters"], parent_path)
        data = [data]
    if not isinstance(data, list):
        return results
    for item in data:
        if not isinstance(item, dict):
            continue
        label = _fr_node_label(item)
        current_path = f"{parent_path}/{label}" if parent_path else str(label)
        content = _fr_node_text(item)
        kids = _fr_child_list(item)
        if not content and kids:
            content = _fr_concatenate_values(kids)
        emb = item.get("embeddings")
        if emb is None:
            emb = item.get("embedding")
        results.append(
            {
                "path": current_path,
                "content": content,
                "embeddings": emb,
                "is_leaf": not bool(kids),
            }
        )
        if kids:
            results.extend(_fr_extract_flat(kids, current_path))
    return results


def get_full_reduced_corpus(leaf_only: bool = True) -> list:
    if "full_reduced_corpus" in _cache:
        return _cache["full_reduced_corpus"]
    path = data_path("Full_reduced_structured_with_embeddings.json")
    if not path.exists():
        _cache["full_reduced_corpus"] = []
        return _cache["full_reduced_corpus"]
    data = json.loads(path.read_text(encoding="utf-8"))
    flat = _fr_extract_flat(data)
    _cache["full_reduced_corpus"] = [
        item
        for item in flat
        if item.get("embeddings") and (not leaf_only or item.get("is_leaf"))
    ]
    return _cache["full_reduced_corpus"]


def get_query_model():
    global _query_model
    if _query_model is None:
        from sentence_transformers import SentenceTransformer

        model_name = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        local_only = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1" or os.environ.get(
            "HF_HUB_OFFLINE", "0"
        ) == "1"
        _query_model = SentenceTransformer(model_name, local_files_only=local_only)
    return _query_model


class QueryModelProxy:
    def encode(self, *args, **kwargs):
        return get_query_model().encode(*args, **kwargs)


queryModel = QueryModelProxy()


def get_clip_bundle():
    global _clip_bundle
    if _clip_bundle is None:
        import torch
        import clip

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        _clip_bundle = (model, preprocess, device)
    return _clip_bundle


def get_image_search_index() -> dict:
    global _image_index
    if _image_index is not None:
        return _image_index
    path = data_path("clip_image_embeddings.json")
    if not path.exists():
        _image_index = {
            "embeddings": np.array([]),
            "paths": [],
            "umap_x": [],
            "umap_y": [],
            "folders": [],
        }
        return _image_index
    image_data = json.loads(path.read_text(encoding="utf-8"))
    _image_index = {
        "embeddings": np.array([item["embeddings"] for item in image_data]),
        "paths": [item["path"] for item in image_data],
        "umap_x": [item.get("umap_x", 0) for item in image_data],
        "umap_y": [item.get("umap_y", 0) for item in image_data],
        "folders": [item.get("folder", "") for item in image_data],
    }
    return _image_index
