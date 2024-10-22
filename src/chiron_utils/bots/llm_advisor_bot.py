"""Bots that carry out random orders and make random order proposals."""

from dataclasses import dataclass
import random
from typing import List, Sequence, Tuple, Union

import numpy as np
from peft import PeftModel
import torch
from torch.nn import DataParallel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
)

from chiron_utils.bots.baseline_bot import BaselineBot, BotType
from chiron_utils.utils import POWER_NAMES_DICT, get_other_powers


@dataclass
class LlmAdvisor(BaselineBot):
    """Bot that carries out random orders and sends random order proposals to other bots.

    Because of the similarity between the advisor and player versions of this bot,
    both of their behaviors are abstracted into this single abstract base class.
    """

    bot_type = BotType.ADVISOR
    base_model_name = "meta-llama/Llama-2-7b-chat-hf"
    adapter_path: str = "usc-isi/Llama2-Advisor"
    tokenizer_path: str = "usc-isi/Llama2-Advisor"
    classification_model_name: str = "AutonLabTruth/llama3_m"  # Added for classification model
    classification_tokenizer_name: str = (
        "meta-llama/Llama-3.1-8B"  # Added for classification tokenizer
    )
    device: str = "cuda"

    def __post_init__(self) -> None:
        """Initialize models."""
        # Load the main model and tokenizer
        self.tokenizer, self.model = self.load_model(
            self.base_model_name, self.adapter_path, self.tokenizer_path, self.device
        )

        # Load the classification model and tokenizer
        self.classification_model, self.classification_tokenizer = self.load_classification_model(
            self.classification_model_name, self.classification_tokenizer_name, self.device
        )

    @staticmethod
    def load_model(
        base_model_name: str, adapter_path: str, tokenizer_path: str, device: str = "cpu"
    ) -> Tuple[PreTrainedTokenizer, PeftModel]:
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
        model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(model, adapter_path)
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
        model.to(device)

        return tokenizer, model

    def load_classification_model(
        self, model_name: str, tokenizer_name: str, device: str
    ) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """Load the classification model and tokenizer."""
        classification_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
        classification_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        classification_model.to(device)
        return classification_model, classification_tokenizer

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
                output_ids = self.model.module.generate(input_ids=input_ids)
            else:
                output_ids = self.model.generate(input_ids=input_ids)

        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text  # type: ignore[no-any-return]

    def classification(self, prompt: str) -> str:
        """Classify a prompt to decide whether to trust or not trust."""
        inputs = self.classification_tokenizer(prompt, return_tensors="pt").to(self.device)
        self.classification_model.eval()
        with torch.no_grad():
            logits = self.classification_model(**inputs).logits

        logits = logits.detach().cpu().numpy()
        softmaxed = np.exp(logits) / np.sum(np.exp(logits))
        score = softmaxed[0, 1]

        if score > 0.15:
            predicted_class = "not to trust"
        else:
            predicted_class = "to trust"

        return predicted_class

    def format_prompt(
        self, own: str, oppo: str, suggest_orders: List[str]
    ) -> Union[Tuple[str, str], Tuple[None, None]]:
        """Create prompt used as input to LLM."""
        if own in POWER_NAMES_DICT:
            own = POWER_NAMES_DICT[own]
        if oppo in POWER_NAMES_DICT:
            oppo = POWER_NAMES_DICT[oppo]
        # system prompt format
        system_prompt = (
            "<<SYS>>\n"
            "You are an AI assistant tasked with understanding and analyzing the board status "
            "of a Diplomacy game, the message history between two players and"
            "the recommended orders by Cicero for the current phase.\n"
            "Your goal is to help respond to the last message from another player, "
            "based on the context provided.\n"
            "\n<</SYS>>"
        )
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
        if filtered_messages == []:
            return None, None
        else:
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

            last_message = ""
            for msg in reversed(reversed_closest_8_messages):
                if msg.sender == oppo:
                    last_message = msg.message
                    break

            prompt_for_classification = f"{oppo} sends {own}:{last_message}"
            decision = self.classification(prompt=prompt_for_classification)

            # cicero recommendation format
            # if USE_CICERO:
            #     sender_orders = self.get_cicero_order_recommendations(own)
            # else:
            #     sender_orders = self.get_random_orders()

            prompt = (
                f"<s>[INST] {system_prompt}\n\n"
                f"---\n\n"
                f"Board Status: {sorted_board_states}\n\n"
                f"Cicero Recommendation for {own}: {suggest_orders}\n\n"
                f"Message History: {my_message}\n\n"
                f"---\n\n"
                f"Last Message: Message from {oppo}: {last_message}\n\n"
                f"---\n\n"
                f"Question: I am {own} and I decide to {decision} the last message from {oppo}. "
                f"As my advisor, give me one short but specific message for the last message which "
                f"I can use directly to respond to {oppo}'s last message. You also need to give me "
                f"the reason to support my decision. [/INST]"
            )
            return prompt, decision

    async def read_suggestions_from_advisor(self) -> List[str]:
        """Read suggestions from RandomProposerAdvisor.

        Returns:
            List of suggested orders.
        """
        received_messages = self.read_messages()

        # Filter for messages sent by the RandomProposerAdvisor with msg_type "suggested_move_full"
        suggestions = [
            msg.message
            for msg in received_messages
            if msg.sender == "omniscient_type" and msg.type == "suggested_move_full"
        ]

        # Return a flattened list of suggested orders if there are any suggestions
        return suggestions if suggestions else []

    async def do_messaging_round(self, orders: Sequence[str]) -> List[str]:
        """Carry out one round of messaging, along with related tasks.

        Returns:
            List of orders to carry out.
        """
        suggested_orders = await self.read_suggestions_from_advisor()
        for other_power in get_other_powers([self.power_name], self.game):
            prompt, decision = self.format_prompt(self.power_name, other_power, suggested_orders)
            if prompt is None:
                continue
            else:
                generate_text = self.generate_text(prompt)
                index = generate_text.find("[/INST] ")
                model_output = generate_text[index + 8 :]
                decision_output = f"I recommend {decision} the last message."
                final_output = decision_output + model_output
                await self.suggest_message(other_power, final_output)

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
