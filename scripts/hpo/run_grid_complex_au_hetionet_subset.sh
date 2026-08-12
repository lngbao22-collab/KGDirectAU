#!/usr/bin/env bash
# Grid search gamma_q / gamma_t / gamma_ent / tuni for ComplEx-AU on hetionet_subset.
# 4x4x4x4 = 256 cells; 4 all-zero-gamma combos pruned => 252 train runs.
# ~2 min/trial => ~8.4 hours wall time (sequential).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

SEARCH_SPACE="${SEARCH_SPACE:-scripts/hpo/search_space_complex_au_hetionet_subset_gammas.json}"

python scripts/hpo/run_search.py \
  --search-space "$SEARCH_SPACE" \
  --no-screening \
  "$@"

python scripts/hpo/summarize_search.py \
  --search-space "$SEARCH_SPACE" \
  --study-name complex_au_hetionet_subset_gammas \
  --top-k 15
