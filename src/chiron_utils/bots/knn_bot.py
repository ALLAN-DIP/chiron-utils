"""Abstract base classes for bots."""

from abc import ABC
from dataclasses import dataclass
import os
from pathlib import Path
import pickle
from typing import Any, Dict, List, Sequence

from baseline_models.model_code.evaluation import infer
from baseline_models.model_code.preprocess import entry_to_vectors
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

    is_first_messaging_round = False

    def __post_init__(self) -> None:
        with open(MODEL_PATH, "rb") as model_file:
            self.models: Dict[str, Any] = pickle.load(model_file)

    async def start_phase(self) -> None:
        """Execute actions at the start of the phase."""
        self.is_first_messaging_round = True

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
        print(orders)

        return orders

    async def gen_orders(self) -> List[str]:
        orders = self.get_orders()
        power_orders = orders[POWER_TO_INDEX[self.power_name]]
        if self.bot_type == BotType.ADVISOR:
            await self.suggest_orders(power_orders)
        print(f"Sending orders!!!\n{power_orders}")
        return power_orders

    async def do_messaging_round(self, orders: Sequence[str]) -> List[str]:

        if not self.is_first_messaging_round:
            return list(orders)

        order_proposal = str(self.get_orders())

        for power in POWER_TO_INDEX:
            print(f"Other power: {power}")
            print(f"Suggested order: {order_proposal}")
            if power == self.power_name:
                continue
            elif self.bot_type == BotType.ADVISOR:
                await self.suggest_message(power, order_proposal)
            elif self.bot_type == BotType.PLAYER:
                await self.send_message(power, order_proposal)

        self.is_first_messaging_round = False

        return list(orders)


class KnnAdvisor(KnnBot):
    """Advisor form of `KnnBot`."""

    bot_type = BotType.ADVISOR
    suggestion_type = SuggestionType.MOVE_ONLY


class KnnPlayer(KnnBot):
    """Player form of `KnnBot`."""

    bot_type = BotType.PLAYER
