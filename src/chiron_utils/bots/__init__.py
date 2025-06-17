"""Module for CHIRON advisors and players."""

from typing import List, Type

try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata

from chiron_utils.bots.baseline_bot import BaselineBot as BaselineBot
from chiron_utils.bots.random_proposer_bot import (
    RandomProposerAdvisor as RandomProposerAdvisor,
    RandomProposerPlayer as RandomProposerPlayer,
)

BOTS: List[Type[BaselineBot]] = [
    RandomProposerAdvisor,
    RandomProposerPlayer,
]
# Import bots only if their direct third-party dependencies are satisfied
# This unfortunately requires hardcoding the list of required modules,
# but there currently isn't a way to check if a given extra was used during installation.
importable_modules = set(importlib_metadata.packages_distributions())
if {"baseline_models"} < importable_modules:
    from chiron_utils.bots.elastic_advisor import (
        ElasticAdvisor as ElasticAdvisor,
    )
    from chiron_utils.bots.lr_bot import (
        LrAdvisor as LrAdvisor,
        LrPlayer as LrPlayer,
    )
    from chiron_utils.bots.lr_probs_bot import (
        LrProbsSelfTextAdvisor as LrProbsSelfTextAdvisor,
        LrProbsSelfTextVisualAdvisor as LrProbsSelfTextVisualAdvisor,
        LrProbsSelfVisualAdvisor as LrProbsSelfVisualAdvisor,
        LrProbsTextAdvisor as LrProbsTextAdvisor,
        LrProbsTextVisualAdvisor as LrProbsTextVisualAdvisor,
        LrProbsVisualAdvisor as LrProbsVisualAdvisor,
    )

    BOTS.extend(
        [
            ElasticAdvisor,
            LrAdvisor,
            LrPlayer,
            LrProbsSelfTextAdvisor,
            LrProbsSelfTextVisualAdvisor,
            LrProbsSelfVisualAdvisor,
            LrProbsTextAdvisor,
            LrProbsTextVisualAdvisor,
            LrProbsVisualAdvisor,
        ]
    )
if {"peft", "torch", "transformers"} < importable_modules:
    from chiron_utils.bots.csu_faaf_advisor_bot import FaafAdvisor as FaafAdvisor
    from chiron_utils.bots.llm_advisor_bot import LlmAdvisor as LlmAdvisor
    from chiron_utils.bots.llm_advisor_new_bot import LlmNewAdvisor as LlmNewAdvisor

    BOTS.extend(
        [
            FaafAdvisor,
            LlmAdvisor,
            LlmNewAdvisor,
        ]
    )
# Alphabetize list of classes
BOTS.sort(key=lambda t: t.__name__)

NAMES_TO_BOTS = {bot.__name__: bot for bot in BOTS}

DEFAULT_BOT_TYPE = RandomProposerPlayer  # pylint: disable=invalid-name
