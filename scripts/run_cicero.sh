#!/usr/bin/env bash

set -euxo pipefail

GAME_ID=$1
HOST=$2
POWER=$3
shift 3

REPO=$(realpath ~/diplomacy_cicero/)
CICERO=/media/volume/cicero-base-models

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
  --volume "$REPO"/fairdiplomacy/AMR/:/diplomacy_cicero/fairdiplomacy/AMR/:ro \
  --volume "$REPO"/fairdiplomacy_external:/diplomacy_cicero/fairdiplomacy_external:ro \
  --volume "$REPO"/parlai_diplomacy:/diplomacy_cicero/parlai_diplomacy:ro \
  --volume "$CICERO"/agents:/diplomacy_cicero/conf/common/agents:ro \
  --volume "$CICERO"/models:/diplomacy_cicero/models:ro \
  --volume "$CICERO"/gpt2:/usr/local/lib/python3.7/site-packages/data/gpt2:ro \
  --workdir /diplomacy_cicero \
  ghcr.io/allan-dip/diplomacy_cicero:human_experiments-alex \
  "${GAME_COMMAND[@]}" "$@"
