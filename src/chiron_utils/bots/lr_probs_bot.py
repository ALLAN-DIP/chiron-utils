"""Advisors to provide order probability distributions generated by a logistic regressor model."""

from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List, Sequence

from baseline_models.model_code.engine_predict import BaselineAdvice
from diplomacy.utils.constants import SuggestionType

from chiron_utils.bots.baseline_bot import BaselineBot, BotType
from chiron_utils.utils import return_logger

logger = return_logger(__name__)

MODEL_PATH = Path() / "lr_model"


@dataclass
class LrProbsBot(BaselineBot, ABC):
    """Baseline logistic regressor model.

    `MODEL_PATH` should point to folder containing model .pkl files.

    Each model corresponds to a (unit, location, phase) combination.

    Unit types are 'A', 'F'.

    Location types include all possible locations on the board, such as 'BRE', 'LON', 'STP_SC', etc.

    Phase types are 'SM', 'FM', 'WA, 'SR', 'FR', 'CD'
    """

    bot_type = BotType.ADVISOR
    advise_only_about_self: ClassVar = False
    is_first_messaging_round = False

    def __post_init__(self) -> None:
        """Verify that model path exists when instantiated."""
        super().__post_init__()
        if not MODEL_PATH.is_dir():
            raise NotADirectoryError(
                f"Model directory {str(MODEL_PATH)!r} does not exist or is not a directory."
            )

    async def start_phase(self) -> None:
        """Execute actions at the start of the phase."""
        self.is_first_messaging_round = True

    async def gen_orders(self) -> List[str]:
        """Generate orders for a turn.

        Returns:
            List of orders to carry out.
        """
        return []

    async def do_messaging_round(self, orders: Sequence[str]) -> List[str]:
        """Carry out one round of messaging.

        Returns:
            List of orders to carry out.
        """
        if not self.is_first_messaging_round:
            return list(orders)

        game_state = self.game.get_state()
        for province in self.game.map.locs:
            # Model lookup uses uppercase keys,
            # but `locs` uses lowercase for provinces with two coasts
            province = province.upper()
            model = BaselineAdvice(str(MODEL_PATH), game_state, province)
            results = model.predict(top_k=10)
            # `power` is `None` when there are not units at a location
            if results["power"] is None:
                logger.info(f"No results for {province}")
                continue
            if self.advise_only_about_self and results["power"] != self.power_name:
                logger.info(f"Skipping results for {province}")
                continue
            predictions = results["preds"]
            await self.suggest_orders_probabilities(province, predictions)

        self.is_first_messaging_round = False

        return list(orders)


@dataclass
class LrProbsSelfTextAdvisor(LrProbsBot):
    """Text advisor form of `LrProbsBot` that only provides advice about self."""

    default_suggestion_type = SuggestionType.MOVE_DISTRIBUTION_TEXTUAL
    advise_only_about_self = True


@dataclass
class LrProbsSelfTextVisualAdvisor(LrProbsBot):
    """Combined text and visual advisor form of `LrProbsBot` that only provides advice about self."""

    default_suggestion_type = (
        SuggestionType.MOVE_DISTRIBUTION_TEXTUAL | SuggestionType.MOVE_DISTRIBUTION_VISUAL
    )
    advise_only_about_self = True


@dataclass
class LrProbsSelfVisualAdvisor(LrProbsBot):
    """Visual advisor form of `LrProbsBot` that only provides advice about self."""

    default_suggestion_type = SuggestionType.MOVE_DISTRIBUTION_VISUAL
    advise_only_about_self = True


@dataclass
class LrProbsTextAdvisor(LrProbsBot):
    """Text advisor form of `LrProbsBot` that provides advice about all powers."""

    default_suggestion_type = SuggestionType.MOVE_DISTRIBUTION_TEXTUAL


@dataclass
class LrProbsTextVisualAdvisor(LrProbsBot):
    """Combined text and visual advisor form of `LrProbsBot` that provides advice about all powers."""

    default_suggestion_type = (
        SuggestionType.MOVE_DISTRIBUTION_TEXTUAL | SuggestionType.MOVE_DISTRIBUTION_VISUAL
    )


@dataclass
class LrProbsVisualAdvisor(LrProbsBot):
    """Visual advisor form of `LrProbsBot` that provides advice about all powers."""

    default_suggestion_type = SuggestionType.MOVE_DISTRIBUTION_VISUAL
