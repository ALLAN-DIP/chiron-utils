#!/usr/bin/env bash

set -euxo pipefail

GAME_ID=$1
HOST=$2
POWER=$3
shift 3

CICERO_DIR=/media/volume/cicero-base-models

GAME_COMMAND=(
  python fairdiplomacy_external/mila_api.py
  --game_id "$GAME_ID"
  --host "$HOST"
  --use-ssl
  --power "$POWER"
  --game_type 2
)

time docker run \
  --rm \
  --gpus all \
  --name cicero_"$GAME_ID"_"$POWER" \
  --volume "$CICERO_DIR"/agents:/diplomacy_cicero/conf/common/agents:ro \
  --volume "$CICERO_DIR"/gpt2:/usr/local/lib/python3.7/site-packages/data/gpt2:ro \
  --volume "$CICERO_DIR"/models:/diplomacy_cicero/models:ro \
  --workdir /diplomacy_cicero \
  ghcr.io/allan-dip/diplomacy_cicero:human_experiments-alex \
  "${GAME_COMMAND[@]}" "$@"
