"""Abstract base classes for bots."""

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
from enum import Enum, auto
import os
import random
from typing import Any, ClassVar, List, Mapping, Optional, Sequence

try:
    from typing import TypedDict  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import TypedDict

from diplomacy import Game, Message
from diplomacy.client.network_game import NetworkGame
from diplomacy.utils import strings as diplomacy_strings
from diplomacy.utils.constants import SuggestionType

from chiron_utils.utils import remove_invalid_orders, return_logger, serialize_message_dict

logger = return_logger(__name__)


class BotType(str, Enum):
    """Type of bot being defined."""

    ADVISOR = auto()
    PLAYER = auto()


class OrderProbability(TypedDict):  # type: ignore[misc]
    """Probability information for a single order.

    Attributes:
        pred_prob: Predicted probability of order.
        rank: Determined by predicted probability sorted in decreasing order.
        opacity: Normalized probability, used for rendering opacity.
    """

    pred_prob: float
    rank: int
    opacity: float


DEFAULT_COMM_STAGE_LENGTH = 300  # 5 minutes in seconds
COMM_STAGE_LENGTH = int(os.environ.get("COMM_STAGE_LENGTH", DEFAULT_COMM_STAGE_LENGTH))
SENTINEL = object()


@dataclass
class BaselineBot(ABC):
    """Abstract base class for bots."""

    player_type: ClassVar[str] = diplomacy_strings.PRESS_BOT
    bot_type: ClassVar[BotType]
    default_suggestion_type: ClassVar[Optional[SuggestionType]] = None
    power_name: str
    game: Game
    num_message_rounds: Optional[int] = None
    communication_stage_length: int = COMM_STAGE_LENGTH  # in seconds
    suggestion_type: Optional[SuggestionType] = SENTINEL

    def __post_init__(self) -> None:
        """Initialize uninitialized value(s)."""
        if self.suggestion_type is SENTINEL:
            self.suggestion_type = self.default_suggestion_type

    @property
    def display_name(self) -> str:
        """Display name consisting of power name and bot type."""
        return f"{self.power_name} ({self.__class__.__name__})"

    async def wait_for_comm_stage(self) -> None:
        """Wait for all other press bots to be ready.

        The bot marks itself as ready and then polls the other press bots until they are all ready.
        Once they all, the bot can start communicating.
        """
        logger.info("Waiting for communication stage to start")

        # Comm status should not be sent in local games, only set
        if isinstance(self.game, NetworkGame):
            await self.game.set_comm_status(
                power_name=self.power_name, comm_status=diplomacy_strings.READY
            )
        else:
            self.game.set_comm_status(
                power_name=self.power_name, comm_status=diplomacy_strings.READY
            )

        while not all(
            power.comm_status == diplomacy_strings.READY
            for power in self.game.powers.values()
            if power.player_type == diplomacy_strings.PRESS_BOT and not power.is_eliminated()
        ):
            await asyncio.sleep(1)
            logger.info("Still waiting")

    def read_messages(self) -> List[Message]:
        """Retrieves all valid messages for the current phase sent to the bot.

        Returns:
            List of messages.
        """
        messages = self.game.get_messages(game_role=self.power_name)
        received_messages = sorted(
            msg for msg in messages.values() if msg.sender != self.power_name
        )
        for msg_obj in received_messages:
            logger.info("%s received message: %s", self.display_name, msg_obj)
        return received_messages

    async def send_message(
        self,
        recipient: str,
        message: str,
        *,
        sender: Optional[str] = None,
        msg_type: Optional[str] = None,
    ) -> None:
        """Send message to the server.

        Args:
            recipient: The name of the power receiving the message.
            message: Text of message to be sent.
            sender: The name of the power sending the message.
            msg_type: Type of message (e.g., `"suggested_message"`, `"suggested_move"`)
        """
        msg_obj = Message(
            sender=sender or self.power_name,
            recipient=recipient,
            message=message,
            phase=self.game.get_current_phase(),
            type=msg_type,
        )
        logger.info("%s sent message: %s", self.display_name, msg_obj)

        # Messages should not be sent in local games, only stored
        if isinstance(self.game, NetworkGame):
            await self.game.send_game_message(message=msg_obj)
        else:
            self.game.add_message(message=msg_obj)

    async def send_suggestion(self, payload: Any, suggestion_type: str) -> None:
        """Send suggestion message to the server.

        Args:
            payload: Dictionary mapping powers to suggestions intended for them.
            suggestion_type: Type of suggestion (e.g., `"has_suggestions"`, `"suggested_message"`).
        """
        message_dict = {
            "advisor": self.display_name,
            "recipient": self.power_name,
            "payload": payload,
        }

        await self.send_message(
            diplomacy_strings.GLOBAL,
            message=serialize_message_dict(message_dict),
            sender=diplomacy_strings.OMNISCIENT_TYPE,
            msg_type=suggestion_type,
        )

    async def declare_suggestion_type(self) -> None:
        """Declare the advisor's suggestion type to the engine."""
        if self.bot_type != BotType.ADVISOR:
            raise TypeError(
                f"{self.declare_suggestion_type.__name__!r} cannot be called by {self.__class__.__name__!r} "
                f"because it is not a {BotType.ADVISOR}"
            )
        if self.suggestion_type is None:
            raise ValueError(
                f"{self.declare_suggestion_type.__name__!r} cannot be called because "
                f"it does not support any suggestion types"
            )

        # Explicit `int` cast is needed before Python 3.11
        payload = int(self.suggestion_type)

        await self.send_suggestion(
            payload=payload,
            suggestion_type=diplomacy_strings.HAS_SUGGESTIONS,
        )

    async def suggest_orders(
        self, orders: Sequence[str], *, partial_orders: Optional[Sequence[str]] = None
    ) -> None:
        """Send suggested orders for power to the server.

        Args:
            orders: Orders to suggest.
            partial_orders: Orders already set by the player.
        """
        if self.bot_type != BotType.ADVISOR:
            raise TypeError(
                f"{self.suggest_orders.__name__!r} cannot be called by {self.__class__.__name__!r} "
                f"because it is not a {BotType.ADVISOR}"
            )
        if not self.suggestion_type & SuggestionType.MOVE:
            raise ValueError(
                f"{self.suggest_orders.__name__!r} cannot be called because "
                f"it does not provide {SuggestionType.MOVE.name} suggestions"
            )

        orders = remove_invalid_orders(orders, self.game)
        payload = {"suggested_orders": orders}
        if partial_orders:
            partial_orders = remove_invalid_orders(partial_orders, self.game)
            payload["player_orders"] = partial_orders
            suggestion_type = diplomacy_strings.SUGGESTED_MOVE_PARTIAL
        else:
            suggestion_type = diplomacy_strings.SUGGESTED_MOVE_FULL

        await self.send_suggestion(
            payload=payload,
            suggestion_type=suggestion_type,
        )

    def read_own_suggestions_from_advisor(self) -> List[str]:
        """Read recommended orders from advisor.

        Returns:
            List of recommended orders.
        """
        received_messages = self.read_messages()
        suggestion_messages = [
            msg.message
            for msg in received_messages
            if msg.type == diplomacy_strings.SUGGESTED_MOVE_FULL
        ]
        logger.info("%s received own move suggestions: %s", self.display_name, suggestion_messages)
        return suggestion_messages

    async def suggest_opponent_orders(self, opponent_orders: Mapping[str, Sequence[str]]) -> None:
        """Send predicted orders for opponent powers to the server.

        Args:
            opponent_orders: Mapping from opponent powers to their predicted orders.
        """
        if self.bot_type != BotType.ADVISOR:
            raise TypeError(
                f"{self.suggest_opponent_orders.__name__!r} cannot be called by {self.__class__.__name__!r} "
                f"because it is not a {BotType.ADVISOR}"
            )
        if not self.suggestion_type & SuggestionType.OPPONENT_MOVE:
            raise ValueError(
                f"{self.suggest_opponent_orders.__name__!r} cannot be called because "
                f"it does not provide {SuggestionType.OPPONENT_MOVE.name} suggestions"
            )

        # Delete invalid orders
        opponent_orders = {
            power: remove_invalid_orders(orders, self.game)
            for power, orders in opponent_orders.items()
        }

        payload = {"predicted_orders": opponent_orders}

        await self.send_suggestion(
            payload=payload,
            suggestion_type=diplomacy_strings.SUGGESTED_MOVE_OPPONENTS,
        )

    def read_opponent_move_suggestions_from_advisor(self) -> List[str]:
        """Read opponent move predictions from advisor.

        Returns:
            List of predicted orders.
        """
        received_messages = self.read_messages()
        suggestion_messages = [
            msg.message
            for msg in received_messages
            if msg.type == diplomacy_strings.SUGGESTED_MOVE_OPPONENTS
        ]
        logger.info(
            "%s received opponent move suggestions: %s", self.display_name, suggestion_messages
        )
        return suggestion_messages

    read_suggestions_from_advisor = read_opponent_move_suggestions_from_advisor

    async def suggest_orders_probabilities(
        self, province: str, orders_probabilities: Mapping[str, OrderProbability]
    ) -> None:
        """Send probabilities for suggested orders to the server.

        Args:
            province: Province orders apply to.
            orders_probabilities: Mapping from possible orders to their probabilities.
        """
        if self.bot_type != BotType.ADVISOR:
            raise TypeError(
                f"{self.suggest_opponent_orders.__name__!r} cannot be called by {self.__class__.__name__!r} "
                f"because it is not a {BotType.ADVISOR}"
            )
        if not (
            self.suggestion_type & SuggestionType.MOVE_DISTRIBUTION_TEXTUAL
            or self.suggestion_type & SuggestionType.MOVE_DISTRIBUTION_VISUAL
        ):
            raise ValueError(
                f"{self.suggest_orders_probabilities.__name__!r} cannot be called because "
                f"it does not provide {SuggestionType.MOVE_DISTRIBUTION_TEXTUAL.name} or "
                f"{SuggestionType.MOVE_DISTRIBUTION_VISUAL.name} suggestions"
            )

        # Delete invalid orders and recalculate some fields
        # `baseline-models` often includes orders that do not match the board state
        valid_orders = remove_invalid_orders(list(orders_probabilities), self.game)
        if set(valid_orders) != set(orders_probabilities):
            total_probs = sum(
                pred_prob["pred_prob"]
                for order, pred_prob in orders_probabilities.items()
                if order in valid_orders
            )
            orders_probabilities = {
                order: {  # type: ignore[misc]
                    "pred_prob": pred_prob["pred_prob"],
                    "rank": rank,
                    "opacity": pred_prob["pred_prob"] / total_probs,
                }
                for rank, (order, pred_prob) in enumerate(orders_probabilities.items())
            }

        payload = {"province": province, "predicted_orders": orders_probabilities}

        await self.send_suggestion(
            payload=payload,
            suggestion_type=diplomacy_strings.SUGGESTED_MOVE_DISTRIBUTION,
        )

    async def suggest_message(self, recipient: str, message: str) -> None:
        """Send suggested messages for power to the server.

        Args:
            recipient: The name of the power that would receive the recommended message.
            message: Text of the recommended message.
        """
        if self.bot_type != BotType.ADVISOR:
            raise TypeError(
                f"{self.suggest_message.__name__!r} cannot be called by {self.__class__.__name__!r} "
                f"because it is not a {BotType.ADVISOR}"
            )
        if not self.suggestion_type & SuggestionType.MESSAGE:
            raise ValueError(
                f"{self.suggest_message.__name__!r} cannot be called because "
                f"it does not provide {SuggestionType.MESSAGE.name} suggestions"
            )

        payload = {"recipient": recipient, "message": message}

        await self.send_suggestion(
            payload=payload,
            suggestion_type=diplomacy_strings.SUGGESTED_MESSAGE,
        )

    async def suggest_commentary(self, recipient: str, message: str) -> None:
        """Send suggested commentary for power to the server.

        Args:
            recipient: The name of the power that the commentary is about.
            message: Text of the commentary.
        """
        if self.bot_type != BotType.ADVISOR:
            raise TypeError(
                f"{self.suggest_commentary.__name__!r} cannot be called by {self.__class__.__name__!r} "
                f"because it is not a {BotType.ADVISOR}"
            )
        if not self.suggestion_type & SuggestionType.COMMENTARY:
            raise ValueError(
                f"{self.suggest_commentary.__name__!r} cannot be called because "
                f"it does not provide {SuggestionType.COMMENTARY.name} suggestions"
            )

        payload = {"recipient": recipient, "commentary": message}

        await self.send_suggestion(
            payload=payload,
            suggestion_type=diplomacy_strings.SUGGESTED_COMMENTARY,
        )

    async def send_intent_log(self, log_msg: str) -> None:
        """Send intent log to the server.

        Args:
            log_msg: Text of message to store in intent log.
        """
        logger.info("Intent log: %r", (log_msg))
        # Intent logging should not be sent in local games
        if not isinstance(self.game, NetworkGame):
            return
        log_data = self.game.new_log_data(body=log_msg)
        await self.game.send_log_data(log=log_data)

    async def send_orders(self, orders: Sequence[str], *, wait: bool = False) -> None:
        """Send orders to the server.

        Args:
            orders: Orders to send.
            wait: Whether the server should be told to wait for further orders.
        """
        if self.bot_type != BotType.PLAYER:
            raise TypeError(
                f"{self.send_orders.__name__!r} cannot be called by {self.__class__.__name__!r} "
                f"because it is not a {BotType.PLAYER}"
            )

        orders = remove_invalid_orders(orders, self.game)

        logger.info("Sent orders: %s", orders)

        # Orders should not be sent in local games, only stored
        if isinstance(self.game, NetworkGame):
            # pylint: disable=unexpected-keyword-arg
            await self.game.set_orders(power_name=self.power_name, orders=orders, wait=wait)
        else:
            self.game.set_orders(power_name=self.power_name, orders=orders)

    async def start_phase(self) -> None:
        """Execute actions at the start of the phase."""
        return

    @abstractmethod
    async def gen_orders(self) -> List[str]:
        """Generate orders for a turn.

        Returns:
            List of orders to carry out.
        """
        raise NotImplementedError

    @abstractmethod
    async def do_messaging_round(self, orders: Sequence[str]) -> List[str]:
        """Carry out one round of messaging, along with related tasks.

        Returns:
            List of orders to carry out.
        """
        raise NotImplementedError

    async def end_phase(self) -> None:
        """Execute actions at the end of the phase."""
        return

    async def __call__(self) -> List[str]:
        """Carry out one phase of the game.

        Returns:
            List of orders to carry out.
        """
        await self.start_phase()

        if self.bot_type == BotType.ADVISOR:
            await self.declare_suggestion_type()

        orders = await self.gen_orders()

        # Skip communications unless in the movement phase
        if not self.game.get_current_phase().endswith("M"):
            return orders

        if self.bot_type == BotType.PLAYER:
            await self.send_intent_log(f"Initial orders (before communication): {orders}")

        await self.wait_for_comm_stage()

        if self.num_message_rounds:
            for _ in range(self.num_message_rounds):
                # sleep for a random amount of time before retrieving new messages for the power
                await asyncio.sleep(random.uniform(0.5, 1.5))
                orders = await self.do_messaging_round(orders)
        else:

            async def run_messaging_loop() -> None:
                nonlocal orders

                while True:
                    # sleep for a random amount of time before retrieving new messages for the power
                    await asyncio.sleep(random.uniform(0.5, 1.5))

                    orders = await self.do_messaging_round(orders)

            try:
                # Set aside 10s for cancellation
                wait_time = self.communication_stage_length - 10
                await asyncio.wait_for(run_messaging_loop(), timeout=wait_time)
            except asyncio.TimeoutError:
                logger.info("Exiting communication phase because out of time")

        await self.end_phase()

        return orders
