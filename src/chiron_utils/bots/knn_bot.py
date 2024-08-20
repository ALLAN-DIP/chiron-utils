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

from baseline_bot import BaselineBot
from baseline_models.evaluation import infer
from baseline_models.preprocess import entry_to_vectors

from chiron_utils.utils import return_logger

logger = return_logger(__name__)

DEFAULT_COMM_STAGE_LENGTH = 300  # 5 minutes in seconds
COMM_STAGE_LENGTH = int(os.environ.get("COMM_STAGE_LENGTH", DEFAULT_COMM_STAGE_LENGTH))


@dataclass
class KnnBot(BaselineBot):

    """
    Currently a dictionary mapping phase type to a model
    Phase types are 'SM', 'FM', 'WA, 'SR', 'FR', 'CD'
    """ 
    models = dict()

    def __init__(model_path : str) -> None:
        with open(model_path, 'rb') as model_file:
            models = pickle.load(model_file)

    async def gen_orders(self) -> List[str]:
        map = self.game.map
        powers = self.game.powers

        name = self.game.phase
        split = name.split()
        season_phase = split[0][0] + split[2][0]

        units = map.units
        centers = map.centers
        homes = map.homes
        influences = dict()

        for power, power_class in powers.item():
            influences[power] = power_class.influence

        vector = entry_to_vectors(None, False, name_data=name, units_data=units, centers_data=centers, homes_data=homes, influences_data=influences)
        orders = infer(self.model[season_phase], vector)
        
        return orders[self.power_name]

    async def do_messaging_round(self, orders: Sequence[str]) -> List[str]:
        return self.gen_orders()
