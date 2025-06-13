"""Bot that provides commentary advise using LLM as base model."""

import asyncio
from dataclasses import dataclass, field
import json
import random
from typing import Dict, List, Optional, Sequence, Tuple, Union

import diplomacy
from diplomacy import Message
from diplomacy.utils.constants import SuggestionType
import torch
from torch.nn import DataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from chiron_utils.bots.baseline_bot import BaselineBot, BotType
from chiron_utils.utils import POWER_NAMES_DICT, get_other_powers, mapping, return_logger

logger = return_logger(__name__)


@dataclass
class LlmAdvisor(BaselineBot):
    """Bot that provides commentary advise using LLM as base model.

    We use Llama3.1-8B-Instruct as the base model and use alignment judgement,
    to measure the alignment between Cicero predicted orders and message history.
    """

    is_first_messaging_round = False
    previous_newest_messages: Dict[str, Optional[List[Message]]] = field(
        default_factory=lambda: dict.fromkeys(POWER_NAMES_DICT)
    )
    base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bot_type = BotType.ADVISOR
    default_suggestion_type = SuggestionType.COMMENTARY

    def __post_init__(self) -> None:
        """Initialize models."""
        super().__post_init__()
        self.tokenizer, self.model = self.load_model(self.base_model_name, None, self.device)

    async def start_phase(self) -> None:
        """Execute actions at the start of the phase."""
        self.is_first_messaging_round = True

    def get_relevant_messages(self, own: str, oppo: str) -> List[Message]:
        """Return all messages sent between 'own' and 'oppo'."""
        return [
            msg
            for msg in self.game.get_messages(game_role=self.power_name)
            if (msg.sender == own and msg.recipient == oppo)
            or (msg.sender == oppo and msg.recipient == own)
        ]

    def get_recent_message_history(
        self, messages: List[Message], max_count: int = 8
    ) -> List[Message]:
        """Sort a list of messages by `time_sent` descending and return up to max_count most recent.

        Also ensures the very last message is from the opponent (if available).
        """
        if not messages:
            return []

        # Sort descending by time_sent
        sorted_msgs = sorted(messages, key=lambda x: x.time_sent, reverse=True)
        recent_msgs = sorted_msgs[:max_count]
        reversed_msgs = list(reversed(recent_msgs))

        # Make sure the last message is from opponent, if possible
        if reversed_msgs and reversed_msgs[-1].sender == self.power_name:
            return []

        return reversed_msgs

    def create_system_prompt(self) -> str:
        """Return the system prompt string (static text)."""
        return """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert assistant specializing in the Diplomacy board game. Your role is to assist a novice player by analyzing:
1. The current board state.
2. The message history exchanged between the novice player and the opponent.
3. The predicted orders for the opponent.

Your primary objective is to evaluate whether the opponent's predicted orders align with the message history and the board state.

Key Evaluation Guidelines:

1. Consider an order aligned if its purpose or intent is consistent with the opponent's stated goals or the tactical/strategic needs implied by the board state.
2. Redundancy in orders (e.g., several supporting moves) can still be aligned if it serves to ensure the success of a critical move or maintains flexibility in uncertain situations.
3. Misalignment occurs only if the order:
    Contradicts the stated strategy or creates unnecessary risks.
    Fails to contribute meaningfully to the position given the board state.
4. Evaluate the broader context of each move, including defensive, offensive, or preparatory purposes, and weigh these factors in your analysis.
For each predicted order, explain whether it aligns with the message history and board state. Use reasoning that incorporates tactical, strategic, and contextual factors while avoiding overly strict judgments.
Limit your evaluation to alignment or misalignment, with short reasoning for each.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Here is an example:
**Predicted Orders of ITALY:**
A TUN to ALB VIA,
F ION C A TUN to ALB,
F NAP to TYS,
A VEN to TRI,

**Board State:**
AUSTRIA: [F TRI, A SER, A GAL, A BUD ]
ENGLAND: [F LON, F NWY, A YOR, F EDI ]
FRANCE: [A POR, A SPA, F BEL, F BRE, F MAR, A PAR ]
GERMANY: [F DEN, A MUN, A HOL, A BER, F KIE ]
ITALY: [A VEN, F ION, A TUN, F NAP ]
RUSSIA: [A WAR, F SEV, F BOT, A RUM, A STP ]
TURKEY: [A CON, A BUL, F BLA, F SMY ]

**Message History:**
Message from AUSTRIA: "France and Turkey are trying to get me to attack you, obviously I will not do that."
Message from ITALY: "With the build in Marseilles, France's goals are quite transparent."
Message from ITALY: "Turkey will likely bounce me in EAS. Would you be willing to support me into Greece (via convoy)?"
Message from ITALY: "From there, I can support you into Bulgaria."
Message from AUSTRIA: "I'm willing to debate it but I had a different view on this turn. I figured he would bounce you in EAS but what if you went for AEG instead? Unless you're worried about overextending yourself due to France.
Because I trust you enough at this point to move Tri to Alb and then we will definitely be able to get Gre in the fall."
Message from ITALY: "Which makes going to AEG/EAS a bit of an overstretch."
Message from ITALY: "If we can get Germany on board I'd feel a lot more comfortable with committing east."
Message from ITALY: "Sorry, some connection issues on my end so messages sent out of order."

**Question:**
I am playing as AUSTRIA. For each of ITALY's predicted orders, evaluate whether it aligns with the message history and the current board state. Explain the orders first and provide short reasoning and analysis for each predicted move.

**Answer:**
A TUN to ALB VIA,
Alignment: The move is aligned with the message history and board state. Italy previously mentioned the idea of moving east and positioning to take Greece.
Convoying from TUN to ALB sets up a strong position for further collaboration with Austria in the Balkans, consistent with the stated plan to focus on Greece in the fall.

F ION C A TUN to ALB,
Alignment: The convoy is aligned with the board state and message history. The convoy facilitates Italy's movement into ALB, aligning with the communicated intent to focus efforts on securing Greece.
This move ensures Italy's positioning in the region while maintaining the tactical flexibility discussed.

F NAP to TYS,
Alignment: This move is aligned as it supports Italy's eastward intentions.
Moving the fleet to the Tyrrhenian Sea strengthens Italy's naval presence in the Mediterranean and provides future flexibility to either defend against France or support further eastern operations
, consistent with Italy's concerns about overextending.

A VEN to TRI,
Misalignment: This move is misaligned with the message history and board state. Austria explicitly trusts Italy and has not shown hostility, and Italy's previous messages suggest no immediate plans to antagonize Austria.
Moving to TRI is aggressive and undermines the collaborative stance Italy communicated. It could create unnecessary conflict, contradicting Italy's expressed preference for focusing eastward and securing Greece.


Now let's see the question:"""

    def format_boardstates(self, boardstates: Dict[str, List[str]]) -> str:
        """Return formatted board states."""
        lines = []
        for power, units in boardstates.items():
            units_str = ", ".join(units)
            line = f"{power}: [{units_str} ]"
            lines.append(line)
        return "\n".join(lines)

    def map_words_in_sentence(self, sentence: str, data: Dict[str, List[str]]) -> str:
        """Return abbreviation conversed words in sentences."""
        words = sentence.split()
        mapped_words = []

        for word in words:
            clean_word = word.strip(",.!?")
            word_lower = clean_word.lower()

            if word_lower in data:
                mapped_value = data[word_lower][0]
                mapped_word = word.replace(clean_word, mapped_value)
                mapped_words.append(mapped_word)
            else:
                mapped_words.append(word)

        return " ".join(mapped_words)

    def format_prompt_phase1(self, own: str, oppo: str, suggest_orders: List[str]) -> Optional[str]:
        """Create prompt used as input to the LLM.

        Returns:
            A string prompt, or None if no messages exist between 'own' and 'oppo'.
        """
        # Convert power names
        if own in POWER_NAMES_DICT:
            own = POWER_NAMES_DICT[own]
        if oppo in POWER_NAMES_DICT:
            oppo = POWER_NAMES_DICT[oppo]

        # Grab and sort board states, filter relevant messages.
        system_prompt = self.create_system_prompt()
        board_states = self.game.get_state()["units"]
        sorted_board_states = {key: sorted(value) for key, value in board_states.items()}
        formated_board_states = self.format_boardstates(sorted_board_states)
        filtered_messages = self.get_relevant_messages(own, oppo)
        if not filtered_messages:
            return None

        recent_msgs = self.get_recent_message_history(filtered_messages, max_count=10)
        if not recent_msgs:
            return None

        # Build a text representation of the message history
        message_history = ""
        for msg in recent_msgs:
            message_history += f"Message from {msg.sender}:'{msg.message}' "
        parsed_data = json.loads(suggest_orders[0])
        predicted_orders = parsed_data["payload"]["predicted_orders"][oppo]

        orders_string = ", ".join(predicted_orders)

        prompt = (
            f"{system_prompt}"
            f"""
**Predicted Orders of {oppo}:**
{orders_string.replace('-','to')}

**Board State:**
"""
            f"{formated_board_states}\n\n"
            f"""**Message History:**  \n{message_history}\n\n"""
            f"""**Question:**  \nI am playing as {own}. For each of {oppo}'s predicted orders,
            evaluate whether it aligns with the message history and the current board state. Explain the orders first and provide short reasoning and analysis for each predicted move.
\n"""
            f"""**Answer:**
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        )
        return prompt

    def format_prompt_phase2(self, oppo: str, evaluation: str) -> Optional[str]:
        """Create prompt used as input to the LLM.

        Returns:
            A string prompt, or None if no messages exist between 'own' and 'oppo'.
        """
        if oppo in POWER_NAMES_DICT:
            oppo = POWER_NAMES_DICT[oppo]

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert assistant specializing in the Diplomacy board game. Based on the detailed evaluation of following predicted orders, respond in the following short and structured format:

            I think {oppo} is being deceptive, because I think they are going to make these moves: <predicted misaligned moves>. These moves are contradictory to {oppo}'s previous messages to you.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            **Evaluation**:
            {evaluation}

            **Question**:
            from Evaluation part, extract necessary information and fill in the following as the output:
            I think {oppo} is being deceptive, because I think they are going to make these moves: <predicted misaligned moves>. These moves are contradictory to {oppo}'s previous messages to you.
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>

            """
        return prompt

    def load_model(
        self,
        base_model_name: str,
        tokenizer_path: Optional[str] = None,
        device: str = "cpu",
    ) -> Tuple[PreTrainedTokenizer, Union[PreTrainedModel, DataParallel]]:
        """Loads and returns a tokenizer and model on the requested device."""
        if tokenizer_path is None:
            tokenizer_path = base_model_name

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForCausalLM.from_pretrained(base_model_name)
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
        model.to(device)

        return tokenizer, model

    def generate_text(
        self,
        prompt: str,
        tokenizer: PreTrainedTokenizer,
        model: Union[PreTrainedModel, DataParallel],
        device: str = "cpu",
        max_new_tokens: int = 512,
    ) -> str:
        """Performs text generation using the tokenizer/model loaded above."""
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2000,
            return_attention_mask=True,
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        model.eval()
        with torch.no_grad():
            if isinstance(model, DataParallel):
                output_ids = model.module.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            else:
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )

        return tokenizer.decode(output_ids[0], skip_special_tokens=True)  # type: ignore[no-any-return]

    def generate_and_parse_response(self, prompt: str) -> str:
        """Generate text from the model given 'prompt'.

        Returns:
            message or None if something fails.
        """
        generated_text = self.generate_text(prompt, self.tokenizer, self.model, self.device, 1024)
        assistant_output = generated_text.split("assistant")[2]
        return assistant_output

    async def do_messaging_round(self, orders: Sequence[str]) -> List[str]:
        """Carry out one round of messaging, along with related tasks.

        Returns:
            List of orders to carry out.
        """
        await asyncio.sleep(random.uniform(5, 10))

        filtered_orders = self.read_suggestions_from_advisor()
        if not filtered_orders:
            return []

        for other_power in get_other_powers([self.power_name], self.game):
            all_relevant = self.get_relevant_messages(self.power_name, other_power)
            prev_msgs = self.previous_newest_messages.get(other_power)
            if prev_msgs is not None:
                # Compare to find new messages
                new_messages = [m for m in all_relevant if m not in prev_msgs]
                if not new_messages:
                    # No new messages, skip
                    continue

                prompt = self.format_prompt_phase1(self.power_name, other_power, filtered_orders)
                if prompt is None:
                    continue
                processed_prompt = self.map_words_in_sentence(prompt, mapping)
                logger.info("Phase1 prompt for %s: %s", other_power, processed_prompt)
                output_phase1 = self.generate_and_parse_response(processed_prompt)
                logger.info("Phase1 output for %s: %s", other_power, output_phase1)

                alignment_count = output_phase1.count("Alignment")
                misalignment_count = output_phase1.count("Misalignment")
                if misalignment_count >= alignment_count:
                    prompt2 = self.format_prompt_phase2(other_power, output_phase1)
                    if prompt2 is None:
                        continue
                    output_phase2 = self.generate_and_parse_response(prompt2)
                    logger.info("Phase2 output for %s: %s", other_power, output_phase2)
                    if output_phase2 and output_phase2.lstrip().startswith("I think"):
                        try:
                            await self.suggest_commentary(other_power, f"{output_phase2.lstrip()}")
                        except diplomacy.utils.exceptions.GamePhaseException as exc:
                            logger.exception("Ignoring %s", exc.__class__.__name__)
                    else:
                        pass
                else:
                    # do nothing if alignment_count > misalignment_count
                    pass

            else:
                # If we have no previous messages, treat this as first time for this power
                prompt = self.format_prompt_phase1(self.power_name, other_power, filtered_orders)
                if prompt is None:
                    continue
                processed_prompt = self.map_words_in_sentence(prompt, mapping)
                logger.info("Phase1 prompt for %s: %s", other_power, processed_prompt)
                output_phase1 = self.generate_and_parse_response(processed_prompt)
                logger.info("Phase1 output for %s: %s", other_power, output_phase1)

                alignment_count = output_phase1.count("Alignment")
                misalignment_count = output_phase1.count("Misalignment")
                if misalignment_count >= alignment_count:
                    prompt2 = self.format_prompt_phase2(other_power, output_phase1)
                    if prompt2 is None:
                        continue
                    output_phase2 = self.generate_and_parse_response(prompt2)
                    logger.info("Phase2 output for %s: %s", other_power, output_phase2)
                    if output_phase2 and output_phase2.lstrip().startswith("I think"):
                        try:
                            await self.suggest_commentary(other_power, f"{output_phase2.lstrip()}")
                        except diplomacy.utils.exceptions.GamePhaseException as exc:
                            logger.exception("Ignoring %s", exc.__class__.__name__)
                    else:
                        pass
                else:
                    pass

            # After handling new (or first-time) messages, update the record
            self.previous_newest_messages[other_power] = all_relevant

        self.is_first_messaging_round = False
        return list(orders)

    async def gen_orders(self) -> List[str]:
        """Generate orders for a turn.

        Returns:
            List of orders to carry out.
        """
        return []
