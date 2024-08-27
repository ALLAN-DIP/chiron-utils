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

@dataclass
class KnnBot(BaselineBot):

    """
    Currently a dictionary mapping phase type to a model
    Phase types are 'SM', 'FM', 'WA, 'SR', 'FR', 'CD'
    """ 
    models = dict()
    with open(MODEL_PATH, 'rb') as model_file:
        models = pickle.load(model_file)

    async def gen_orders(self) -> List[str]:
        map = self.game.map
        powers = self.game.powers

        name = self.game.phase

        units = map.units
        centers = map.centers
        homes = map.homes
        influences = dict()

        for power, power_class in powers.items():
            influences[power] = power_class.influence

        vector, _, season = entry_to_vectors(None, False, name_data=name, units_data=units, centers_data=centers, homes_data=homes, influences_data=influences)
        print(vector)
        orders = infer(self.models[season], vector)
        print(orders)
        
        return orders[self.power_name]

    async def do_messaging_round(self, orders: Sequence[str]) -> List[str]:
        return self.gen_orders()


class KnnAdvisor(KnnBot):
    """Advisor form of `RandomProposerBot`."""

    bot_type: ClassVar[str] = "advisor"


class KnnPlayer(KnnBot):
    """Player form of `RandomProposerBot`."""

    bot_type: ClassVar[str] = "player"