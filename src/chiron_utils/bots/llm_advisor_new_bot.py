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
from chiron_utils.utils import POWER_NAMES_DICT, get_other_powers, return_logger

logger = return_logger(__name__)


@dataclass
class LlmNewAdvisor(BaselineBot):
    """Bot that provides commentary advise using Llama3.1 as base model.

    We use Llama3.1-8B-Instruct as the base model and generate commentary
    by using recommended orders, predicted orders and board states.
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
        # used to avoid repeated generation
        self.previous_prompt: Optional[str] = None
        self.tokenizer, self.model = self.load_model(self.base_model_name, None, self.device)

    async def start_phase(self) -> None:
        """Execute actions at the start of the phase."""
        self.is_first_messaging_round = True

    def create_system_prompt(self) -> str:
        """Return the system prompt string (static text)."""
        return """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are an expert assistant specializing in the Diplomacy board game. Your role is to assist a novice player by analyzing:
                1. The current board state.
                2. The recommended orders for the novice player.
                3. The potential orders for every power.

                Your goal is to provide rationale for each recommended order based on the board state and potential orders as following:
                1. Order Interpretation: Briefly summarize what the specific order intends tactically or strategically.
                2. Consistency Check (Board): Assess if the order aligns logically with tactical or strategic needs suggested by the current board state.
                3. Risk Assessment: Identify if the order introduces unnecessary risks or contradicts earlier expressed intentions.

                Avoid long explanations or generic commentary — be precise and practical.
                <|eot_id|><|start_header_id|>user<|end_header_id|>
                Below is an example:
                **Board State:**
                AUSTRIA: F GRE, A VIE, A BUD, A BUL, A SER
ENGLAND: F ENG, F NWG, F NTH, F IRI, A PIC
FRANCE: A BRE, A MAR, F SPA, A PAR
GERMANY: F DEN, A BUR, A MUN, F BAL, A KIE
ITALY: A TUN, A PIE, F EAS, F ION
RUSSIA: F BOT, F BLA, A RUM, A SEV, A LVN
TURKEY: F ANK, A CON, A SMY

                **Recommended Orders for GERMANY:**
                F DEN S A KIE - SWE,
A BUR - PAR,
F BAL C A KIE - SWE,
A KIE - SWE VIA,
A MUN - BUR

                **Potential Orders for other powers:**
                AUSTRIA: A VIE - GAL, F GRE S A BUL, A BUD - RUM, A SER S A BUD - RUM, A BUL S A BUD - RUM
ENGLAND: F NTH H, F NWG - NAO, F ENG S A PIC - BRE, F IRI - MAO, A PIC - BRE
FRANCE: A BRE S A PAR, A PAR S A BRE, F SPA/SC - POR, A MAR - BUR
ITALY: A TUN - SYR VIA, A PIE S A BUR - MAR, F ION C A TUN - SYR, F EAS C A TUN - SYR
RUSSIA: F BOT C A LVN - FIN, A LVN - FIN VIA, A SEV - ARM, F BLA C A RUM - CON, A RUM - CON VIA
TURKEY: F ANK - BLA, A SMY S A CON, A CON H

                **Advice:**
                You are advising the player controlling GERMANY. What is a rationale to explain each recommended move?

                **Answer:**
                F DEN S A KIE - SWE:
1. Order Interpretation: Fleet at Denmark supports the army's attack on Sweden.
2. Consistency Check (Board): Sweden is vacant and Russia is busy convoying from LVN to FIN, so this secures the center before Russia can react.
3. Risk Assessment: Very low risk, Denmark stays safe; only a surprise BOT -> SWE bounce matters, and the support neutralizes that.

A BUR - PAR:
1. Order Interpretation: Attacks Paris to cut the French mutual-support loop and threaten a home center.
2. Consistency Check (Board): England is also hitting Brest, so France must defend two fronts and PAR cannot support BRE.
3. Risk Assessment: Likely bounce but no downside, Burgundy remains occupied and blocks any French push east.

F BAL C A KIE - SWE:
1. Order Interpretation: Baltic fleet convoys the army while retaining sea control.
2. Consistency Check (Board): No hostile fleets neighbor BAL, so the convoy is safe and the fleet can pivot next season.
3. Risk Assessment: Low risk, fleet is immobile and secure; only foregoes a weaker direct BAL -> SWE move.

A KIE - SWE VIA:
1. Order Interpretation: Army grabs the Swedish supply center, adding a build and denying Russia.
2. Consistency Check (Board): Supported by DEN and convoyed by BAL versus any lone BOT contest.
3. Risk Assessment: Moderate low risk, fails only if Russia double-commits to SWE, which is unlikely.

A MUN - BUR:
1. Order Interpretation: Move army from Munich into Burgundy to sustain western pressure and keep a buffer.
2. Consistency Check (Board): With the current BUR unit striking Paris, this transfer keeps a German piece in BUR and shields Munich.
3. Risk Assessment: Low risk, Munich is unthreatened this turn and the shift sets up future BUR - MAR/PAR options."""

    def format_boardstates(self, boardstates: Dict[str, List[str]]) -> str:
        """Return formatted board states."""
        lines = []
        for power, units in boardstates.items():
            units_str = ", ".join(units)
            line = f"{power}: {units_str}"
            lines.append(line)
        return "\n".join(lines)

    def format_prompt_phase1(
        self, own: str, suggest_orders: List[str], own_orders: List[str]
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
        parsed_data = json.loads(suggest_orders[0])
        predicted_orders = parsed_data["payload"]["predicted_orders"]
        formatted_opponent_orders = predicted_orders
        formatted_opponent_orders = "\n".join(
            f"{power}: " + ", ".join(predicted_orders[power]) for power in predicted_orders
        )
        # own recommended order format
        formatted_recommended_orders = ""
        for orders in own_orders:
            parsed_data_own = json.loads(orders)
            if parsed_data_own["recipient"] == own:
                formatted_recommended_orders = ", ".join(
                    parsed_data_own["payload"]["suggested_orders"]
                )
        if formatted_recommended_orders == "":
            return None

        prompt = f"""{system_prompt}

            Now below is the real question:
                **Board State:**
                {formated_board_states}

                **Recommended Orders for {own}:**
                {formatted_recommended_orders}

                **Potential Orders for other powers:**
                {formatted_opponent_orders}

                **Advice:**
                You are advising the player controlling {own}. What is a rationale to explain each recommended move?
                <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        return prompt

    def format_prompt_phase2(
        self, own: str, suggest_orders: List[str], own_orders: List[str], rationales: Optional[str]
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

        parsed_data = json.loads(suggest_orders[0])
        predicted_orders = parsed_data["payload"]["predicted_orders"]
        formatted_opponent_orders = predicted_orders

        formatted_opponent_orders = "\n".join(
            f"{power}: " + ", ".join(predicted_orders[power]) for power in predicted_orders
        )
        formatted_recommended_orders = ""
        for orders in own_orders:
            parsed_data_own = json.loads(orders)
            if parsed_data_own["recipient"] == own:
                formatted_recommended_orders = ", ".join(
                    parsed_data_own["payload"]["suggested_orders"]
                )
        if formatted_recommended_orders == "":
            return None

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert assistant specializing in the Diplomacy board game. Your role is to assist a novice player by analyzing:
    1. The current board state.
    2. The recommended orders for the novice player.
    3. The potential orders for every power.
    4. The rationales for recommended orders

    Your goal is to give a summary rationale for the set of recommended orders based on the given information, your summary should be less than three sentences.

    Avoid long explanations or generic commentary — be precise and practical.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    **Board State:**
    {formated_board_states}

    **Recommended Orders for {own}:**
    {formatted_recommended_orders}

    **Potential Orders for other powers:**
    {formatted_opponent_orders}

    **Rationale for Recommended Orders*:*
    {rationales}

    **Advice:**
    You are advising the player controlling {own}. Give a summary rationale for this set of recommended orders given the rationale for each order.
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
        generated_text = self.generate_text(prompt, self.tokenizer, self.model, self.device, 1024)
        assistant_output = generated_text.split("assistant")[2]
        return assistant_output

    async def do_messaging_round(self, orders: Sequence[str]) -> List[str]:
        """Carry out one round of messaging, along with related tasks.

        Returns:
            List of orders to carry out.
        """
        await asyncio.sleep(random.uniform(5, 10))
        own = self.power_name
        if own in POWER_NAMES_DICT:
            own = POWER_NAMES_DICT[own]

        filtered_opponent_orders = self.read_opponent_move_suggestions_from_advisor()
        if not filtered_opponent_orders:
            return []
        filtered_own_orders = self.read_own_suggestions_from_advisor()
        if not filtered_own_orders:
            return []
        other_power = get_other_powers([self.power_name], self.game)
        prompt = self.format_prompt_phase1(
            self.power_name, filtered_opponent_orders, filtered_own_orders
        )
        formatted_recommended_orders = ""
        for orders in filtered_own_orders:
            parsed_data_own = json.loads(orders)
            if parsed_data_own["recipient"] == own:
                formatted_recommended_orders = ", ".join(
                    parsed_data_own["payload"]["suggested_orders"]
                )
        if prompt is None:
            return []
        if self.previous_prompt != prompt:
            logger.info("Phase1 prompt for %s: %s", self.power_name, prompt)
            output_phase1 = self.generate_and_parse_response(prompt)
            logger.info("Phase1 output for %s: %s", self.power_name, output_phase1)
            prompt2 = self.format_prompt_phase2(
                self.power_name, filtered_opponent_orders, filtered_own_orders, output_phase1
            )
            logger.info("Phase2 prompt for %s: %s", self.power_name, prompt2)
            output_phase2 = self.generate_and_parse_response(prompt2)
            logger.info("Phase2 output for %s: %s", self.power_name, output_phase2)
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
        else:
            return []

    async def gen_orders(self) -> List[str]:
        """Generate orders for a turn.

        Returns:
            List of orders to carry out.
        """
        return []
