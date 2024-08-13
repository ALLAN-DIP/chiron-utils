"""Bots that carry out random orders and make random order proposals."""

from dataclasses import dataclass
import random
from typing import ClassVar, List, Sequence, Tuple

from peft import PeftModel
import torch
from torch.nn import DataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer

from chiron_utils.bots.baseline_bot import BaselineBot
from chiron_utils.utils import POWER_NAMES_DICT, get_other_powers


@dataclass
class LlmAdvisor(BaselineBot):
    """Bot that carries out random orders and sends random order proposals to other bots.

    Because of the similarity between the advisor and player versions of this bot,
    both of their behaviors are abstracted into this single abstract base class.
    """

    bot_type: ClassVar[str] = "advisor"
    base_model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    adapter_path: str = "../models/finetuned_llama2_after_CPO/checkpoint-best-recent"
    tokenizer_path: str = "../models/finetuned_llama2_after_CPO/checkpoint-best-recent"
    device: str = "cuda"

    def __post_init__(self):
        self.tokenizer, self.model = self.load_model(
            self.base_model_name, self.adapter_path, self.tokenizer_path, self.device
        )

    @staticmethod
    def load_model(
        base_model_name: str, adapter_path: str, tokenizer_path: str, device: str = "cpu"
    ) -> Tuple[AutoTokenizer, PeftModel]:
        """Load the model and tokenizer.

        Args:
            base_model_name: The base model name.
            adapter_path: The path to the adapter.
            tokenizer_path: The path to the tokenizer.
            device: The device to load the model onto.

        Returns:
            The loaded tokenizer and model.
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(model, adapter_path)
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
        model.to(device)

        return tokenizer, model

    def generate_text(self, prompt: str) -> str:
        """Generate text based on a given prompt.

        Args:
            prompt: The prompt to generate text from.

        Returns:
            The generated text.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        self.model.eval()
        with torch.no_grad():
            if isinstance(self.model, DataParallel):
                output_ids = self.model.module.generate(input_ids, max_new_tokens=1024)
            else:
                output_ids = self.model.generate(input_ids, max_new_tokens=1024)

        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text

    def get_cicero_order_recommendations(self, own: str) -> List[str]:
        sender_player = Player(self.agent, own)
        sender_orders = sender_player.get_orders(self.game)
        return sender_orders

    def format_prompt(self, own: str, oppo: str) -> str:
        if own in POWER_NAMES_DICT:
            own = POWER_NAMES_DICT[own]
        if oppo in POWER_NAMES_DICT:
            oppo = POWER_NAMES_DICT[oppo]
        SYS_PROMPT = (
            "You are an AI assistant tasked with understanding and analyzing the board status "
            "of a Diplomacy game, the message history between two players, and "
            "the recommended orders by Cicero for the current phase.\n"
            "Your goal is to provide feedback on whether to trust the last message, "
            "based on the context provided.\n"
        )
        # system prompt format
        system_prompt = f"<<SYS>>\n{SYS_PROMPT}\n<</SYS>>" if SYS_PROMPT is not None else ""
        # board states format
        board_states = self.game.get_state()["units"]
        for key, value in board_states.items():
            board_states[key] = [
                string.replace("STP/NC", "STP")
                .replace("STP/SC", "STP")
                .replace("SPA/SC", "SPA")
                .replace("SPA/NC", "SPA")
                .replace("BUL/EC", "BUL")
                .replace("BUL/SC", "BUL")
                for string in value
            ]
        sorted_board_states = {key: sorted(value) for key, value in board_states.items()}

        # message format
        messages = self.game.messages
        filtered_messages = [
            message
            for message in messages.values()
            if (message.sender == own and message.recipient == oppo)
            or (message.sender == oppo and message.recipient == own)
        ]
        sorted_messages = sorted(filtered_messages, key=lambda x: x.time_sent, reverse=True)
        closest_8_messages = sorted_messages[:8]
        reversed_closest_8_messages = list(reversed(closest_8_messages))
        my_message = ""
        i = 0
        while i < len(reversed_closest_8_messages):
            my_message += f"Message from {reversed_closest_8_messages[i].sender}:'{reversed_closest_8_messages[i].message}' "
            if (
                i == len(reversed_closest_8_messages) - 1
                and reversed_closest_8_messages[i].sender != oppo
            ):
                my_message = my_message.rsplit(
                    f"Message from {reversed_closest_8_messages[i].sender}:'{reversed_closest_8_messages[i].message}' ",
                    1,
                )[0]
                reversed_closest_8_messages.pop()
                i -= 1
            i += 1

        # cicero recommendation format
        sender_orders = self.get_random_orders()

        prompt = (
            f"<s>[INST] {system_prompt}\n    \n---\n\n"
            f"Board Status: {sorted_board_states}\n\n"
            f"Cicero Recommendation for {own}: {sender_orders}\n\n"
            f"Message History: {my_message}\n    \n---\n\n"
            f"Question:As the advisor of {own}, Should we trust last message from {oppo}? [/INST]"
        )
        return prompt

    async def do_messaging_round(self, orders: Sequence[str]) -> List[str]:
        """Carry out one round of messaging, along with related tasks.

        Returns:
            List of orders to carry out.
        """
        for other_power in get_other_powers([self.power_name], self.game):
            prompt = self.format_prompt(self.power_name, other_power)
            generate_text = self.generate_text(prompt)
            await self.suggest_message(other_power, generate_text)

        return list(orders)

    def get_random_orders(self) -> List[str]:
        """Generate random orders to carry out.

        Returns:
            List of random orders.
        """
        possible_orders = self.game.get_all_possible_orders()
        orders = [
            random.choice(list(possible_orders[loc]))
            for loc in self.game.get_orderable_locations(self.power_name)
            if possible_orders[loc]
        ]
        return orders

    async def gen_orders(self) -> List[str]:
        """Generate orders for a turn.

        Returns:
            List of orders to carry out.
        """
        orders = self.get_random_orders()
        return orders
