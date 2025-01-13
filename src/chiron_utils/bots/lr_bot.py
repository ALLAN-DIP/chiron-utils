"""Abstract base classes for bots."""

from abc import ABC
from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Sequence

from baseline_models.model_code.predict import predict
from diplomacy.utils import strings as diplomacy_strings
from diplomacy.utils.constants import SuggestionType

from chiron_utils.bots.baseline_bot import BaselineBot, BotType
from chiron_utils.utils import return_logger

logger = return_logger(__name__)

DEFAULT_COMM_STAGE_LENGTH = 300  # 5 minutes in seconds
COMM_STAGE_LENGTH = int(os.environ.get("COMM_STAGE_LENGTH", DEFAULT_COMM_STAGE_LENGTH))
MODEL_PATH = Path() / "lr_models"


@dataclass
class LrBot(BaselineBot, ABC):
    """Baseline lr model.

    MODEL_PATH should point to folder containing model .pkl files

    Each model corresponds to a (unit, location, phase) combination

    Unit types are 'A', 'F'

    Location types include all possible locations on the board, such as 'LON', 'STP_SC', 'BRE', etc

    Phase types are 'SM', 'FM', 'WA, 'SR', 'FR', 'CD'
    """

    player_type = diplomacy_strings.NO_PRESS_BOT

    def get_orders(self) -> List[str]:
        """Get order predictions from model.

        Returns:
            List of predicted orders.
        """
        state = self.game.get_state()
        orders: List[str] = predict(MODEL_PATH, state, self.power_name)

        logger.info("Orders to suggest: %s", orders)

        return orders

    async def gen_orders(self) -> List[str]:
        """Generate orders for a turn.

        Returns:
            List of orders to carry out.
        """
        orders = self.get_orders()
        if self.bot_type == BotType.ADVISOR:
            await self.suggest_orders(orders)
        elif self.bot_type == BotType.PLAYER:
            await self.send_orders(orders, wait=True)
        return orders

    async def do_messaging_round(self, orders: Sequence[str]) -> List[str]:
        """Carry out one round of messaging.

        Returns:
            List of orders to carry out.
        """
        return list(orders)


class LrAdvisor(LrBot):
    """Advisor form of `LrBot`."""

    bot_type = BotType.ADVISOR
    suggestion_type = SuggestionType.MOVE


class LrPlayer(LrBot):
    """Player form of `LrBot`."""

    bot_type = BotType.PLAYER
