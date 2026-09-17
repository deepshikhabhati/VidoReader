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
copy_if_exists "$ROOT/Full_reduced_structured_with_embeddings.json" "$DATA/Full_reduced_structured_with_embeddings.json"

echo "Data ready in $DATA"
