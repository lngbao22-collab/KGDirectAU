#!/usr/bin/env bash

set -x
set -e

model_path="bert"
task="WN18RR"
while [[ $# -ge 1 ]]; do
  case "$1" in
    --model_path|--model-path)
      model_path=$2
      shift 2
      ;;
    --task)
      task=$2
      shift 2
      ;;
    --test_path|--test-path)
      test_path=$2
      shift 2
      ;;
    --)
      shift
      break
      ;;
    --*)
      break
      ;;
    *)
      if [ "$model_path" = "bert" ]; then
        model_path=$1
      elif [ "$task" = "WN18RR" ]; then
        task=$1
      elif [ -z "${test_path+x}" ]; then
        test_path=$1
      fi
      shift
      ;;
  esac
done

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${task}"
fi

test_path="${DATA_DIR}/test_w_label.txt.json"
if [ ! -f "${test_path}" ]; then
  test_path="${DATA_DIR}/test_w_label.txt"
fi

# neighbor_weight=0.05
# rerank_n_hop=2
# if [ "${task}" = "WN18RR" ]; then
# # WordNet is a sparse graph, use more neighbors for re-rank
#   rerank_n_hop=5
# fi
# if [ "${task}" = "wiki5m_ind" ]; then
# for inductive setting of wiki5m, test nodes never appear in the training set
#  neighbor_weight=0.0
# fi

python3 -u evaluate.py \
--task "${task}" \
--is-test \
--eval-model-path "${model_path}" \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${test_path}" "$@"
