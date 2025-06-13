"""Bot that provides commentary advise using CSU FAAF model."""

import asyncio
from dataclasses import dataclass, field
import random
from typing import Dict, List, Optional, Sequence, Tuple, Union

import diplomacy
from diplomacy import Message
from diplomacy.utils.constants import SuggestionType
from peft import PeftModel
import torch
from torch.nn import DataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from chiron_utils.bots.baseline_bot import BaselineBot, BotType
from chiron_utils.utils import POWER_NAMES_DICT, get_other_powers, return_logger

logger = return_logger(__name__)


@dataclass
class FaafAdvisor(BaselineBot):
    """Bot that provides commentary advise using FAAF as base model.

    We use FAAF model from CSU and generate friction to players.
    """

    is_first_messaging_round = False
    previous_newest_messages: Dict[str, Optional[List[Message]]] = field(
        default_factory=lambda: dict.fromkeys(POWER_NAMES_DICT)
    )
    base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    adapter_path = "src/chiron_utils/models/DELI_faaf_weights/checkpoint-2000"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bot_type = BotType.ADVISOR
    default_suggestion_type = SuggestionType.COMMENTARY

    def __post_init__(self) -> None:
        """Initialize models."""
        super().__post_init__()
        # used to avoid repeated generation
        self.previous_prompt: Optional[str] = None
        self.tokenizer, self.model = self.load_model(
            self.base_model_name, self.adapter_path, self.adapter_path, self.device
        )

    async def start_phase(self) -> None:
        """Execute actions at the start of the phase."""
        self.is_first_messaging_round = True

    def create_system_prompt(self) -> str:
        """Return the system prompt string (static text)."""
        return """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert assistant specializing in the Diplomacy board game. Your role is to analyze:
        1. The current board state.
        2. Recommended orders for the player.
        3. The potential orders for every power.


        IMPORTANT: This is a snapshot of an ongoing Diplomacy game. The board state shows each country's current units:
        - Units prefixed with \"A\" are Armies located on land territories
        - Units prefixed with \"F\" are Fleets located on sea spaces or coastal territories
        - Each unit belongs ONLY to the country listed before the colon (e.g., all units under \"AUSTRIA:\" belong to AUSTRIA)
        - Supply centers are critical territories that allow powers to build new units
        - Home supply centers are especially important to protect\n\nThe \"Recommended Order\" is a specific move being suggested for your power.
        \"Potential Orders for other powers\" shows what other countries might do this turn. Consider how these moves could interact with or counter your recommended order.

        IMPORTANT DIPLOMACY ORDER SYNTAX:
        * \"A VIE - GAL\" means the Army in Vienna moves to Galicia
        * \"F BLA S A RUM - SEV\" means the Fleet in Black Sea stays in place and supports Rumania's attack on Sevastopol
        * \"A BUD H\" means the Army in Budapest holds in place
        * \"F BLA - RUM\" means the Fleet in Black Sea moves to Rumania
        * \"A VEN S A ROM - VEN\" means the Army in Venice supports Rome's attack on Venice (which would prevent an enemy unit from successfully moving to Venice)
"""

    def format_boardstates(self, boardstates: Dict[str, List[str]]) -> str:
        """Return formatted board states."""
        lines = []
        for power, units in boardstates.items():
            units_str = ", ".join(units)
            line = f"{power}: {units_str}"
            lines.append(line)
        return "\n".join(lines)

    def format_prompt_phase1(
        self, own: str, suggest_orders: Dict[str, List[str]], own_orders: List[str]
    ) -> Optional[str]:
        """Create prompt used as input to the LLM.

        Returns:
            A string prompt, or None if no messages exist between 'own' and 'oppo'.
        """
        # Convert power names
        if own in POWER_NAMES_DICT:
            own = POWER_NAMES_DICT[own]

        system_prompt = self.create_system_prompt()
        # board state format
        board_states = self.game.get_state()["units"]
        sorted_board_states = {key: sorted(value) for key, value in board_states.items()}
        formated_board_states = self.format_boardstates(sorted_board_states)
        # opponent order prediction
        formatted_opponent_orders = "\n".join(
            f"{power}: " + ", ".join(suggest_orders[power]) for power in suggest_orders
        )
        # own recommended order format
        formatted_recommended_orders = ", ".join(own_orders)
        if formatted_recommended_orders == "":
            return None

        prompt = f"""{system_prompt}
        Your goal is to provide a detailed explanation for each recommended order:

        <belief_state>
        {own} STRATEGIC CONTEXT: Analyze the current board position specifically from {own}'s perspective.
        Identify which units belong to {own} (listed under '{own}:' in the board state), what territorial objectives are relevant, and how this specific order fits into {own}'s broader strategy and current diplomatic situation.\n</belief_state>

        <rationale>
        ORDER-SPECIFIC ANALYSIS: Provide a thorough tactical explanation of why each specific order shown under \"Recommended Order for {own}\" makes strategic sense.
        Explain what each order accomplishes, how it counters threats from other powers' potential orders, and why it's optimal compared to alternatives {own} could make with this unit.
        Reference the board state and other powers' potential orders to justify your explanation.\n</rationale>

        <friction>
        KEY INSIGHT: Provide the single most important strategic insight about this order that {own} must understand.
        Explain its significance to {own}'s overall position and how it relates to longer-term goals or threats that {own} faces on the board.\n</friction>


        Your response MUST include all three required XML tags (<belief_state>, <rationale>, and <friction>) with complete content for each. First provide the <rationale>, then <belief_state>, and finally <friction>.\n


        <|eot_id|><|start_header_id|>user<|end_header_id|>
                **Board State:**
                {formated_board_states}

                **Recommended Order for {own}:**
                {formatted_recommended_orders}

                **Potential Orders for other powers:**
                {formatted_opponent_orders}

                **Request:**\nYou are advising the player controlling {own}. Explain the strategic rationale behind each recommended order.\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        return prompt

    def format_prompt_phase2(
        self,
        own: str,
        suggest_orders: Dict[str, List[str]],
        own_orders: List[str],
        rationales: Optional[str],
    ) -> Optional[str]:
        """Create prompt used as input to the LLM.

        Returns:
            A string prompt, or None if no messages exist between 'own' and 'oppo'.
        """
        if own in POWER_NAMES_DICT:
            own = POWER_NAMES_DICT[own]

        board_states = self.game.get_state()["units"]
        sorted_board_states = {key: sorted(value) for key, value in board_states.items()}
        formated_board_states = self.format_boardstates(sorted_board_states)

        formatted_opponent_orders = "\n".join(
            f"{power}: " + ", ".join(suggest_orders[power]) for power in suggest_orders
        )
        formatted_recommended_orders = ", ".join(own_orders)
        if formatted_recommended_orders == "":
            return None

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert assistant specializing in the Diplomacy board game. Your role is to assist a novice player by analyzing:
    1. The current board state.
    2. The recommended orders for the novice player.
    3. The potential orders for every power.
    4. The rationales for recommended orders.

    Your goal is to give a summarized detailed explanation for the set of recommended orders based on the given information:
    <friction>\nKEY INSIGHT: Provide the single most important strategic insight about the recommended orders that {own} must understand.
    Explain the significance to {own}'s overall position and how it relates to longer-term goals or threats that {own} faces on the board.\n</friction>\n\n\n
    Your response MUST include required XML tags <friction> with complete content.

    Avoid long explanations or generic commentary — be precise and practical.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    **Board State:**
    {formated_board_states}

    **Recommended Orders for {own}:**
    {formatted_recommended_orders}

    **Potential Orders for other powers:**
    {formatted_opponent_orders}

    **Rationale for Recommended Orders:**
    {rationales}

    **Advice:**
    You are advising the player controlling {own}. Provide a summary of justification for this set of recommended orders using fewer than three sentences.
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        return prompt

    def load_model(
        self,
        base_model_name: str,
        adapter_path: str,
        tokenizer_path: Optional[str] = None,
        device: str = "cpu",
    ) -> Tuple[PreTrainedTokenizer, Union[PreTrainedModel, DataParallel]]:
        """Loads and returns a tokenizer and model on the requested device."""
        if tokenizer_path is None:
            tokenizer_path = base_model_name

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(model, adapter_path)
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
        model.to(device)

        return tokenizer, model

    def generate_text(
        self,
        prompt: Optional[str],
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
            max_length=3000,
            return_attention_mask=True,
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        model.eval()
        with torch.no_grad():
            output_ids = (
                model.module.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
                if isinstance(model, DataParallel)
                else model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return generated_text  # type: ignore[no-any-return]

    def generate_and_parse_response(self, prompt: Optional[str]) -> Optional[str]:
        """Generate text from the model given 'prompt'.

        Returns:
            message or None if something fails.
        """
        generated_text = self.generate_text(prompt, self.tokenizer, self.model, self.device, 2048)

        try:
            assistant_output = generated_text.split("assistant", 2)[2]
        except IndexError:
            return None

        assistant_output = " ".join(assistant_output.split())

        return assistant_output

    async def do_messaging_round(self, orders: Sequence[str]) -> List[str]:
        """Carry out one round of messaging, along with related tasks."""
        await asyncio.sleep(random.uniform(5, 10))

        filtered_opponent_orders = self.read_suggested_opponent_orders()
        if not filtered_opponent_orders:
            return []
        filtered_own_orders = self.read_suggested_orders()
        if not filtered_own_orders:
            return []
        own = self.power_name
        if own in POWER_NAMES_DICT:
            own = POWER_NAMES_DICT[own]
        other_power = get_other_powers([self.power_name], self.game)

        formatted_recommended_orders = ", ".join(filtered_own_orders)

        prompt = self.format_prompt_phase1(
            self.power_name, filtered_opponent_orders, filtered_own_orders
        )
        if prompt is None:
            return []

        # Only proceed if the prompt is new
        if self.previous_prompt == prompt:
            return []

        logger.info("Phase1 prompt for %s: %s", self.power_name, prompt)
        output_phase1 = self.generate_and_parse_response(prompt)
        logger.info("Phase1 output for %s: %s", self.power_name, output_phase1)

        max_retries = 3
        for attempt in range(max_retries):
            prompt2 = self.format_prompt_phase2(
                self.power_name,
                filtered_opponent_orders,
                filtered_own_orders,
                output_phase1,
            )
            logger.info(
                "Phase2 prompt for %s (attempt %d/%d): %s",
                self.power_name,
                attempt + 1,
                max_retries,
                prompt2,
            )

            output_phase2 = self.generate_and_parse_response(prompt2)
            logger.info("Phase2 output for %s: %s", self.power_name, output_phase2)

            if output_phase2 is not None and "I can't" not in output_phase2:
                break
        else:
            logger.warning("%s: max retries reached; skipping messaging round", self.power_name)
            return []
        try:
            for i in other_power:
                if i:
                    await self.suggest_commentary(
                        i,
                        f"Commentary for the current move suggestions ({formatted_recommended_orders}): {output_phase2}",
                    )
                    break
        except diplomacy.utils.exceptions.GamePhaseException as exc:
            logger.exception("Ignoring %s", exc.__class__.__name__)

        self.is_first_messaging_round = False
        self.previous_prompt = prompt
        return list(orders)

    async def gen_orders(self) -> List[str]:
        """Generate orders for a turn.

        Returns:
            List of orders to carry out.
        """
        return []
