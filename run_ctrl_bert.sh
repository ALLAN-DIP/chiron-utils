#!/usr/bin/env bash

set -euo pipefail

mkdir -p logs/
NOW=$(date -u +'%Y_%m_%d_%H_%M_%S')
LOG_FILE=logs/"$NOW"_ctrld_bert.txt

CICERO_DIR=/home/exouser/cicero

GAME_COMMAND=(
  python src/chiron_utils/scripts/run_bot.py --bot_type DeceptionBertAdvisor
  "$@"
)

time docker run \
  --rm \
  --gpus all \
  --name ctrl_bert_latest_"$RANDOM" \
  --volume "$CICERO_DIR"/bert_models:/chiron-utils/src/chiron_utils/models:ro \
  --volume /home/exouser/chiron-utils/src/chiron_utils/bots:/chiron-utils/src/chiron_utils/bots:rw \
  --workdir /chiron-utils \
  ghcr.io/allan-dip/ctrld_bert:latest \
  "${GAME_COMMAND[@]}" |&
  tee "$LOG_FILE"
