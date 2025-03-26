"""Message advisor bot using elastic search vector database."""

from dataclasses import dataclass, field
import os
from typing import List, Sequence

from baseline_models.message_advisor_code.elastic.masked_client import MaskedClient
from diplomacy.utils import strings as diplomacy_strings
from diplomacy.utils.constants import SuggestionType

from chiron_utils.bots.baseline_bot import BaselineBot, BotType
from chiron_utils.utils import get_other_powers, return_logger

logger = return_logger(__name__)

DEFAULT_COMM_STAGE_LENGTH = 300  # 5 minutes in seconds
COMM_STAGE_LENGTH = int(os.environ.get("COMM_STAGE_LENGTH", DEFAULT_COMM_STAGE_LENGTH))

ELASTIC_HOST = "http://localhost:9200"
ELASTIC_INDEX = "tagged_documents_masked_scaled_center"

MESSAGE_ADVICE_COUNT = 10


@dataclass
class ElasticAdvisor(BaselineBot):
    """Elastic search message advisor."""

    bot_type = BotType.ADVISOR
    default_suggestion_type = SuggestionType.MESSAGE
    is_first_messaging_round = False
    player_type = diplomacy_strings.NO_PRESS_BOT
    elastic_client: MaskedClient = field(default_factory=lambda: MaskedClient(ELASTIC_HOST))

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

        state = self.game.get_state()

        messages = self.elastic_client.get_messages_from_sender(
            ELASTIC_INDEX, state, self.power_name
        )
        for other_power in get_other_powers([self.power_name], self.game):
            if other_power in messages:
                if self.bot_type == BotType.ADVISOR:
                    for i in range(min(MESSAGE_ADVICE_COUNT, len(messages[other_power]))):
                        await self.suggest_message(other_power, messages[other_power][i])
                elif self.bot_type == BotType.PLAYER:
                    await self.send_message(other_power, messages[other_power][0])

        self.is_first_messaging_round = False
        return list(orders)
