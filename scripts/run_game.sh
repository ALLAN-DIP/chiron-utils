#!/usr/bin/env bash

set -euxo pipefail

powers=("AUSTRIA" "FRANCE" "GERMANY" "RUSSIA" "TURKEY" "ITALY")
for power in "${powers[@]}"; do
  python scripts/run_bot.py \
    --host diplomacy.alexhedges.dev \
    --port 8433 \
    --use-ssl \
    --game_id knn_test \
    --power "$power" \
    --bot_type RandomProposerPlayer &
done
python scripts/run_bot.py \
  --host diplomacy.alexhedges.dev \
  --port 8433 \
  --use-ssl \
  --game_id knn_test \
  --power ENGLAND \
  --bot_type KnnPlayer
