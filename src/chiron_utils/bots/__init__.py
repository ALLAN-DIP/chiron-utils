"""Module for CHIRON advisors and players."""

from chiron_utils.bots.baseline_bot import BaselineBot as BaselineBot
from chiron_utils.bots.random_proposer_bot import (
    RandomProposerAdvisor as RandomProposerAdvisor,
    RandomProposerPlayer as RandomProposerPlayer,
)

from chiron_utils.bots.llm_advisor_bot import LlmAdvisor

BOTS = [
    RandomProposerAdvisor,
    RandomProposerPlayer,
    LlmAdvisor
]
NAMES_TO_BOTS = {bot.__name__: bot for bot in BOTS}

DEFAULT_BOT_TYPE = RandomProposerPlayer
