powers=("AUSTRIA" "GERMANY" "RUSSIA" "TURKEY" "ITALY" "FRANCE")

# Loop over each power and run the command in the background
for power in "${powers[@]}"; do
    python run_bot.py --host diplomacy.alexhedges.dev \
    --port 8433 \
    --use-ssl \
    --game_id yanzetest2 \
    --power $power \
    --bot_type RandomProposerPlayer > "${power}_log.txt" 2>&1 &
done

# Run additional commands for ENGLAND and save their logs to different files
python run_bot.py --host diplomacy.alexhedges.dev \
    --port 8433 \
    --use-ssl \
    --game_id yanzetest2 \
    --power ENGLAND \
    --bot_type RandomProposerAdvisor > "ENGLAND_RandomProposerAdvisor_log.txt" 2>&1 &

# python run_bot.py --host diplomacy.alexhedges.dev \
#     --port 8433 \
#     --use-ssl \
#     --game_id test_chiron_12 \
#     --power ENGLAND \
#     --bot_type LlmAdvisor > "ENGLAND_LlmAdvisor_log.txt" 2>&1 &