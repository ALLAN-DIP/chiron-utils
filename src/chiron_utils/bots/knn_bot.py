"""Abstract base classes for bots."""

from abc import ABC
from dataclasses import dataclass
import os
from pathlib import Path
import pickle
from typing import Any, Dict, List, Sequence

from baseline_models.model_code.predict import predict
from diplomacy.utils import strings as diplomacy_strings
from diplomacy.utils.constants import SuggestionType

from chiron_utils.bots.baseline_bot import BaselineBot, BotType
from chiron_utils.utils import return_logger

logger = return_logger(__name__)

DEFAULT_COMM_STAGE_LENGTH = 300  # 5 minutes in seconds
COMM_STAGE_LENGTH = int(os.environ.get("COMM_STAGE_LENGTH", DEFAULT_COMM_STAGE_LENGTH))
MODEL_PATH = Path() / "knn_models"


@dataclass
class KnnBot(BaselineBot, ABC):
    """Baseline knn model

    MODEL_PATH should point to folder containing model .pkl files

    Each model corresponds to a (unit, location, phase) combination

    Unit types are 'A', 'F'

    Location types include all possible locations on the board, such as 'LON', 'STP_SC', 'BRE', etc

    Phase types are 'SM', 'FM', 'WA, 'SR', 'FR', 'CD'
    """

    player_type = diplomacy_strings.NO_PRESS_BOT

    def __post_init__(self) -> None:
        with open(MODEL_PATH, "rb") as model_file:
            self.models: Dict[str, Any] = pickle.load(model_file)

    def get_orders(self) -> List[str]:
        influences = {}
        for power, power_class in self.game.powers.items():
            influences[power] = power_class.influence

        orders = list()
        state = self.game.get_state()
        orders = predict(MODEL_PATH, state, self.power_name)

        logger.info("Orders to suggest: %s", orders)

        return orders

    async def gen_orders(self) -> List[str]:
        orders = self.get_orders()
        if self.bot_type == BotType.ADVISOR:
            await self.suggest_orders(orders)
        elif self.bot_type == BotType.PLAYER:
            await self.send_orders(orders, wait=True)
        return orders

    async def do_messaging_round(self, orders: Sequence[str]) -> List[str]:
        return list(orders)


class KnnAdvisor(KnnBot):
    """Advisor form of `KnnBot`."""

    bot_type = BotType.ADVISOR
    suggestion_type = SuggestionType.MOVE


class KnnPlayer(KnnBot):
    """Player form of `KnnBot`."""

    bot_type = BotType.PLAYER
