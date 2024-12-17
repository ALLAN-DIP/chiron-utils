#!/usr/bin/env bash

set -euxo pipefail

powers=(
  AUSTRIA
  FRANCE
  GERMANY
  ITALY
  RUSSIA
  TURKEY
)
for power in "${powers[@]}"; do
  python -m chiron_utils.scripts.run_bot \
    --host diplomacy.alexhedges.dev \
    --port 8433 \
    --use-ssl \
    --game_id lr_test \
    --power "$power" \
    --bot_type RandomProposerPlayer &
done

python chiron_utils.scripts.run_bot \
  --host diplomacy.alexhedges.dev \
  --port 8433 \
  --use-ssl \
  --game_id lr_test \
  --power ENGLAND \
  --bot_type LrPlayer
