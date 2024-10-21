#!/usr/bin/env bash

set -euxo pipefail

BOT_NAME=$1
shift 1

CICERO_DIR=/media/volume/cicero-base-models

GAME_COMMAND=(
  python fairdiplomacy_external/mila_api.py
  --game_type 2
  "$@"
)

mkdir -p logs/
NOW=$(date -u +'%Y_%m_%d_%H_%M_%S')
LOG_FILE=logs/$NOW.txt

time docker run \
  --rm \
  --gpus all \
  --name cicero_"$BOT_NAME"_"$RANDOM" \
  --volume "$CICERO_DIR"/agents:/diplomacy_cicero/conf/common/agents:ro \
  --volume "$CICERO_DIR"/gpt2:/usr/local/lib/python3.7/site-packages/data/gpt2:ro \
  --volume "$CICERO_DIR"/models:/diplomacy_cicero/models:ro \
  --workdir /diplomacy_cicero \
  ghcr.io/allan-dip/diplomacy_cicero:"$BOT_NAME" \
  "${GAME_COMMAND[@]}" |&
  tee "$LOG_FILE"
