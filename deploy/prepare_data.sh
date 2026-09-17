#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/deploy/data"

mkdir -p "$DATA"

copy_if_exists() {
  local src="$1"
  local dest="$2"
  if [[ -f "$src" ]]; then
    cp "$src" "$dest"
    echo "copied $(basename "$src")"
  else
    echo "skip missing $(basename "$src")"
  fi
}

copy_if_exists "$ROOT/english_chunks.json" "$DATA/english_chunks.json"
copy_if_exists "$ROOT/german_chunks.json" "$DATA/german_chunks.json"
copy_if_exists "$ROOT/comparison_results.json" "$DATA/comparison_results.json"
copy_if_exists "$ROOT/embedding_comparison_results.json" "$DATA/embedding_comparison_results.json"
copy_if_exists "$ROOT/hamlet_translation_analysis/output/hamlet_4x4_comparison.json" "$DATA/hamlet_4x4_comparison.json"
copy_if_exists "$ROOT/Full_reduced_structured_with_embeddings.json" "$DATA/Full_reduced_structured_with_embeddings.json"
copy_if_exists "$ROOT/content_embeddings.json" "$DATA/content_embeddings.json"
copy_if_exists "$ROOT/content_embeddings3.json" "$DATA/content_embeddings3.json"
copy_if_exists "$ROOT/AI_wiki_embeddings.json" "$DATA/AI_wiki_embeddings.json"
copy_if_exists "$ROOT/version1_Shakes_final.json" "$DATA/version1_Shakes_final.json"
copy_if_exists "$ROOT/version2_Shakes_final.json" "$DATA/version2_Shakes_final.json"
copy_if_exists "$ROOT/hamlet_version1.json" "$DATA/hamlet_version1.json"
copy_if_exists "$ROOT/hamlet_version2.json" "$DATA/hamlet_version2.json"
copy_if_exists "$ROOT/hamlet_version3.json" "$DATA/hamlet_version3.json"
copy_if_exists "$ROOT/german_hamlet_embeddings.json" "$DATA/german_hamlet_embeddings.json"
copy_if_exists "$ROOT/History_of_artificial_intelligence.pdf" "$DATA/History_of_artificial_intelligence.pdf"
copy_if_exists "$ROOT/clip_image_embeddings.json" "$DATA/clip_image_embeddings.json"

echo "Data ready in $DATA"
