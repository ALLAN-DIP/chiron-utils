"""Abstract base classes for bots."""

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
import os
import random
from typing import ClassVar, List, Optional, Sequence
import pickle
from sklearn.neighbors import KNeighborsClassifier

from diplomacy import Game, Message
from diplomacy.client.network_game import NetworkGame
from diplomacy.utils import strings

from chiron_utils.bots.baseline_bot import BaselineBot
from chiron_utils.bots.baseline_models.evaluation import infer
from chiron_utils.bots.baseline_models.preprocess import entry_to_vectors

from chiron_utils.utils import return_logger

logger = return_logger(__name__)

DEFAULT_COMM_STAGE_LENGTH = 300  # 5 minutes in seconds
COMM_STAGE_LENGTH = int(os.environ.get("COMM_STAGE_LENGTH", DEFAULT_COMM_STAGE_LENGTH))
MODEL_PATH = os.path.join(os.getcwd(), "model")

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
class KnnBot(BaselineBot):
    """
    Currently a dictionary mapping phase type to a model
    Phase types are 'SM', 'FM', 'WA, 'SR', 'FR', 'CD'
    """

    models = dict()
    with open(MODEL_PATH, "rb") as model_file:
        models = pickle.load(model_file)

    is_first_messaging_round = False

    async def start_phase(self) -> None:
        """Execute actions at the start of the phase."""
        self.is_first_messaging_round = True

    def get_orders(self) -> List[str]:
        map = self.game.map
        powers = self.game.powers

        name = self.game.phase

        units = map.units
        centers = map.centers
        homes = map.homes
        influences = dict()

        for power, power_class in powers.items():
            influences[power] = power_class.influence

        vector, _, season = entry_to_vectors(
            None,
            False,
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
        if self.bot_type == "advisor":
            await self.suggest_orders(power_orders)
        print(f"Sending orders!!!\n{power_orders}")
        return power_orders

    async def do_messaging_round(self, orders: Sequence[str]) -> List[str]:

        if not self.is_first_messaging_round:
            return list(orders)

        order_proposal = str(self.get_orders())

        for power in POWER_TO_INDEX.keys():
            print(f"Other power: {power}")
            print(f"Suggested order: {order_proposal}")
            if power == self.power_name:
                continue
            elif self.bot_type == "advisor":
                await self.suggest_message(power, (order_proposal))
            elif self.bot_type == "player":
                await self.send_message(power, (order_proposal))

        self.is_first_messaging_round = False

        return list(orders)


class KnnAdvisor(KnnBot):
    """Advisor form of `RandomProposerBot`."""

    bot_type: ClassVar[str] = "advisor"


class KnnPlayer(KnnBot):
    """Player form of `RandomProposerBot`."""

    bot_type: ClassVar[str] = "player"
