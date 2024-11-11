"""Abstract base classes for bots."""

from abc import ABC
from dataclasses import dataclass
import os
from pathlib import Path
import pickle
from typing import Any, Dict, List, Sequence

from baseline_models.model_code.evaluation import infer
from baseline_models.model_code.preprocess import entry_to_vectors
from diplomacy.utils import strings as diplomacy_strings
from diplomacy.utils.constants import SuggestionType

from chiron_utils.bots.baseline_bot import BaselineBot, BotType
from chiron_utils.utils import return_logger

logger = return_logger(__name__)

DEFAULT_COMM_STAGE_LENGTH = 300  # 5 minutes in seconds
COMM_STAGE_LENGTH = int(os.environ.get("COMM_STAGE_LENGTH", DEFAULT_COMM_STAGE_LENGTH))
MODEL_PATH = Path() / "baseline_knn_model.pkl"

POWER_TO_INDEX = {
    "AUSTRIA": 0,
    "ENGLAND": 1,
    "FRANCE": 2,
    "GERMANY": 3,
    "ITALY": 4,
    "RUSSIA": 5,
    "TURKEY": 6,
}


@dataclass
class KnnBot(BaselineBot, ABC):
    """Currently a dictionary mapping phase type to a model

    Phase types are 'SM', 'FM', 'WA, 'SR', 'FR', 'CD'
    """

    player_type = diplomacy_strings.NO_PRESS_BOT

    def __post_init__(self) -> None:
        with open(MODEL_PATH, "rb") as model_file:
            self.models: Dict[str, Any] = pickle.load(model_file)

    def get_orders(self) -> List[str]:
        name = self.game.phase
        units = self.game.map.units
        centers = self.game.map.centers
        homes = self.game.map.homes

        influences = {}
        for power, power_class in self.game.powers.items():
            influences[power] = power_class.influence

        vector, _, season = entry_to_vectors(
            None,
            include_orders=False,
            name_data=name,
            units_data=units,
            centers_data=centers,
            homes_data=homes,
            influences_data=influences,
        )
        orders = infer(self.models[season], vector.reshape(1, -1))
        logger.info("Orders to suggest: %s", orders)

        return orders

    async def gen_orders(self) -> List[str]:
        orders = self.get_orders()
        power_orders = orders[POWER_TO_INDEX[self.power_name]]
        if self.bot_type == BotType.ADVISOR:
            await self.suggest_orders(power_orders)
        elif self.bot_type == BotType.PLAYER:
            await self.send_orders(orders, wait=True)
        return power_orders

    async def do_messaging_round(self, orders: Sequence[str]) -> List[str]:
        return list(orders)


class KnnAdvisor(KnnBot):
    """Advisor form of `KnnBot`."""

    bot_type = BotType.ADVISOR
    suggestion_type = SuggestionType.MOVE


class KnnPlayer(KnnBot):
    """Player form of `KnnBot`."""

    bot_type = BotType.PLAYER
