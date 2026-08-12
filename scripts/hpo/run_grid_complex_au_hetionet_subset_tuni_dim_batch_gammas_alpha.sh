#!/usr/bin/env bash
# Full grid: tuni x dim x batch_size x gamma_q x gamma_t x gamma_ent x alpha
# for ComplEx-AU on hetionet_subset.
# 5x4x4x4x4x4x5 = 25600 cells; 400 all-zero-gamma combos pruned => 25200 train runs.
# ~2–5 min/trial (dim/batch dependent) => ~35–90 days wall time (sequential, 1 GPU).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

SEARCH_SPACE="${SEARCH_SPACE:-scripts/hpo/search_space_complex_au_hetionet_subset_tuni_dim_batch_gammas_alpha.json}"

python scripts/hpo/run_search.py \
  --search-space "$SEARCH_SPACE" \
  --no-screening \
  "$@"

python scripts/hpo/summarize_search.py \
  --search-space "$SEARCH_SPACE" \
  --study-name complex_au_hetionet_subset_tuni_dim_batch_gammas_alpha \
  --top-k 20
