"""Bots that carry out random orders and make random order proposals."""

from abc import ABC
from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Sequence

from daidepp import AND, PRP, XDO
from diplomacy.utils import strings as diplomacy_strings
from diplomacy.utils.constants import SuggestionType

from chiron_utils.bots.baseline_bot import BaselineBot, BotType
from chiron_utils.daide2eng import gen_english
from chiron_utils.parsing_utils import parse_dipnet_to_daide
from chiron_utils.utils import get_other_powers


@dataclass
class RandomProposerBot(BaselineBot, ABC):
    """Bot that carries out random orders and sends random order proposals to other bots.

    Because of the similarity between the advisor and player versions of this bot,
    both of their behaviors are abstracted into this single abstract base class.
    """

    is_first_messaging_round = False

    async def start_phase(self) -> None:
        """Execute actions at the start of the phase."""
        self.is_first_messaging_round = True

    def get_random_proposal_orders(self) -> Dict[str, str]:
        """Generate random order proposals for other powers.

        Returns:
            Mapping from powers to random order proposals.
        """
        proposals = {}

        # For each power, randomly sample a valid order
        for other_power in get_other_powers([self.power_name], self.game):
            suggested_random_orders = self.get_random_orders(other_power)
            suggested_random_orders = list(
                filter(
                    lambda x: x != diplomacy_strings.WAIVE
                    and not x.endswith(diplomacy_strings.VIA),
                    suggested_random_orders,
                )
            )
            if len(suggested_random_orders) > 0:
                commands = parse_dipnet_to_daide(suggested_random_orders, self.game)
                random_orders = [XDO(command) for command in commands]
                if len(random_orders) > 1:
                    suggested_random_orders = PRP(AND(*random_orders))
                else:
                    suggested_random_orders = PRP(*random_orders)

                proposals[other_power] = gen_english(suggested_random_orders)

        return proposals

    async def do_messaging_round(self, orders: Sequence[str]) -> List[str]:
        """Carry out one round of messaging, along with related tasks.

        Returns:
            List of orders to carry out.
        """
        if not self.is_first_messaging_round:
            return list(orders)

        random_order_proposals = self.get_random_proposal_orders()

        for other_power, suggested_random_orders in random_order_proposals.items():
            if self.bot_type == BotType.ADVISOR:
                await self.suggest_message(other_power, suggested_random_orders)
                await self.suggest_commentary(other_power, f"I have advice about {other_power}!")
            elif self.bot_type == BotType.PLAYER:
                await self.send_message(other_power, suggested_random_orders)

        self.is_first_messaging_round = False

        return list(orders)

    def get_random_orders(self, power_name: Optional[str] = None) -> List[str]:
        """Generate random orders for a power to carry out.

        Args:
            power_name: Name of power to generate random orders for.
                Defaults to current power.

        Returns:
            List of random orders.
        """
        if power_name is None:
            power_name = self.power_name

        possible_orders = self.game.get_all_possible_orders()
        orders = [
            random.choice(list(possible_orders[loc]))
            for loc in self.game.get_orderable_locations(power_name)
            if possible_orders[loc]
        ]
        return orders

    async def gen_orders(self) -> List[str]:
        """Generate orders for a turn.

        Returns:
            List of orders to carry out.
        """
        orders = self.get_random_orders()
        if self.bot_type == BotType.ADVISOR:
            await self.suggest_orders(orders)

            # Generate (random) partial order suggestions
            submitted_orders = self.game.get_orders(self.power_name)
            orderable_locs = self.game.get_orderable_locations(self.power_name)
            if 0 < len(submitted_orders) < len(orderable_locs):
                locs_with_orders = {order.split()[1] for order in submitted_orders}
                partial_orders = [
                    order
                    for order in self.get_random_orders()
                    if order.split()[1] not in locs_with_orders
                ]
                await self.suggest_orders(partial_orders, partial_orders=submitted_orders)

            # Generate (random) order probabilities
            possible_orders = self.game.get_all_possible_orders()
            for province in self.game.map.locs:
                # `locs` uses lowercase for provinces with two coasts
                province = province.upper()

                if not possible_orders[province]:
                    continue

                orders = possible_orders[province]
                orders = random.sample(orders, min(len(orders), 10))
                # Sample probabilities from exponential distribution with default lambda
                # to more closely resemble `LrProbsBot` output
                orders_probabilities = {order: random.expovariate(1) for order in orders}
                orders_probabilities = dict(
                    sorted(orders_probabilities.items(), key=lambda x: (-x[1], x))
                )
                total_probs = sum(prob for prob in orders_probabilities.values())
                predictions = {
                    order: {
                        "pred_prob": prob / total_probs,
                        "rank": rank,
                        "opacity": prob / total_probs,
                    }
                    for rank, (order, prob) in enumerate(orders_probabilities.items())
                }
                await self.suggest_orders_probabilities(province, predictions)  # type: ignore[arg-type]

            random_predicted_orders = {}
            for other_power in get_other_powers([self.power_name], self.game):
                random_predicted_orders[other_power] = self.get_random_orders(other_power)
            await self.suggest_opponent_orders(random_predicted_orders)
        elif self.bot_type == BotType.PLAYER:
            await self.send_orders(orders, wait=True)
        return orders


@dataclass
class RandomProposerAdvisor(RandomProposerBot):
    """Advisor form of `RandomProposerBot`."""

    bot_type = BotType.ADVISOR
    default_suggestion_type = (
        SuggestionType.MESSAGE
        | SuggestionType.MOVE
        | SuggestionType.COMMENTARY
        | SuggestionType.OPPONENT_MOVE
        | SuggestionType.MOVE_DISTRIBUTION_TEXTUAL
        | SuggestionType.MOVE_DISTRIBUTION_VISUAL
    )


@dataclass
class RandomProposerPlayer(RandomProposerBot):
    """Player form of `RandomProposerBot`."""

    bot_type = BotType.PLAYER
