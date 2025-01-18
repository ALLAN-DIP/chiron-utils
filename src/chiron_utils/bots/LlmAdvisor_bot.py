"""Bots that carry out random orders and make random order proposals."""

from abc import ABC
from dataclasses import dataclass,field
import random
from typing import Dict, List, Optional, Sequence
import diplomacy
from diplomacy.utils import strings as diplomacy_strings
from diplomacy.utils.constants import SuggestionType
from diplomacy import Message
from chiron_utils.bots.baseline_bot import BaselineBot, BotType
from chiron_utils.daide2eng import gen_english
from chiron_utils.parsing_utils import dipnet_to_daide_parsing
from chiron_utils.utils import POWER_NAMES_DICT,get_other_powers,return_logger,mapping
import asyncio
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import DataParallel
import json
logger = return_logger(__name__)
@dataclass
class LlmAdvisorBot(BaselineBot, ABC):
    """Bot that carries out random orders and sends random order proposals to other bots.

    Because of the similarity between the advisor and player versions of this bot,
    both of their behaviors are abstracted into this single abstract base class.
    """

    is_first_messaging_round = False
    previous_newest_messages: Dict[str, Optional[List[Message]]] = field(
        default_factory=lambda: dict.fromkeys(POWER_NAMES_DICT)
    )
    base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __post_init__(self) -> None:
        """Initialize models."""
        # Load the main model and tokenizer
        self.tokenizer, self.model = self.load_model(
            self.base_model_name, None, None, self.device
        )

    async def start_phase(self) -> None:
        """Execute actions at the start of the phase."""
        self.is_first_messaging_round = True

    
    async def read_suggestions_from_advisor(self) -> List[str]:
        """Read suggestions from RandomProposerAdvisor.

        Returns:
            List of suggested orders.
        """
        received_messages = self.read_messages()
        suggestions = [
            msg.message
            for msg in received_messages
            # if msg.sender == "omniscient_type" and msg.type == "suggested_move_full"
        ]
        logger.info("-----------")
        logger.info("%s received suggested message: %s", self.display_name, suggestions)
        logger.info("-----------")
        return suggestions if suggestions else []
        
    def _get_relevant_messages(self, own: str, oppo: str) -> List:
        """
        Return all messages sent between 'own' and 'oppo'.
        """
        return [
            msg
            for msg in self.game.messages.values()
            if (msg.sender == own and msg.recipient == oppo)
            or (msg.sender == oppo and msg.recipient == own)
        ]

    def _get_recent_message_history(self, messages: List, max_count: int = 8) -> List:
        """
        Sort a list of messages by `time_sent` descending and return up to max_count most recent.
        Also ensures the very last message is from the opponent (if available).
        """
        if not messages:
            return []
        
        # Sort descending by time_sent
        sorted_msgs = sorted(messages, key=lambda x: x.time_sent, reverse=True)
        recent_msgs = sorted_msgs[:max_count]
        # Reverse them again so they're oldest -> newest
        reversed_msgs = list(reversed(recent_msgs))
        
        # Make sure the last message is from opponent, if possible
        # (this logic was in the while-loop; you might adapt as needed)
        if reversed_msgs and reversed_msgs[-1].sender == self.power_name:
            # The last message is from ourselves; remove it so the last is from the opponent
            return []
        
        return reversed_msgs
    
    def _create_system_prompt(self) -> str:
        """
        Return the system prompt string (static text).
        """
        return (
            """<|begin_of_text|><|start_header_id|>system<|end_header_id|>  
You are an expert assistant specializing in the Diplomacy board game. Your role is to assist a novice player by analyzing:  
1. The current board state.  
2. The message history exchanged between the novice player and the counterplayer.  
3. The predicted orders for the counterplayer.  

Your primary objective is to evaluate whether the counterplayer's predicted orders align with the message history and the board state.

Key Evaluation Guidelines:

1. Consider an order aligned if its purpose or intent is consistent with the counterplayer's stated goals or the tactical/strategic needs implied by the board state.
2. Redundancy in orders (e.g., several supporting moves) can still be aligned if it serves to ensure the success of a critical move or maintains flexibility in uncertain situations.
3. Misalignment occurs only if the order: 
    Contradicts the stated strategy or creates unnecessary risks. 
    Fails to contribute meaningfully to the position given the board state.
4. Evaluate the broader context of each move, including defensive, offensive, or preparatory purposes, and weigh these factors in your analysis.
For each predicted order, explain whether it aligns with the message history and board state. Use reasoning that incorporates tactical, strategic, and contextual factors while avoiding overly strict judgments. Limit your evaluation to alignment or misalignment, with short reasoning for each.
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
Message from AUSTRIA: "I'm willing to debate it but I had a different view on this turn. I figured he would bounce you in EAS but what if you went for AEG instead? Unless you're worried about overextending yourself due to France. Because I trust you enough at this point to move Tri to Alb and then we will definitely be able to get Gre in the fall."
Message from ITALY: "Which makes going to AEG/EAS a bit of an overstretch."
Message from ITALY: "If we can get Germany on board I'd feel a lot more comfortable with committing east."
Message from ITALY: "Sorry, some connection issues on my end so messages sent out of order."

**Question:**  
I am playing as AUSTRIA. For each of ITALY's predicted orders, evaluate whether it aligns with the message history and the current board state. Explain the orders first and provide short reasoning and analysis for each predicted move. 

**Answer:**
A TUN to ALB VIA,
Alignment: The move is aligned with the message history and board state. Italy previously mentioned the idea of moving east and positioning to take Greece. Convoying from TUN to ALB sets up a strong position for further collaboration with Austria in the Balkans, consistent with the stated plan to focus on Greece in the fall.

F ION C A TUN to ALB,
Alignment: The convoy is aligned with the board state and message history. The convoy facilitates Italy's movement into ALB, aligning with the communicated intent to focus efforts on securing Greece. This move ensures Italy’s positioning in the region while maintaining the tactical flexibility discussed.

F NAP to TYS,
Alignment: This move is aligned as it supports Italy’s eastward intentions. Moving the fleet to the Tyrrhenian Sea strengthens Italy’s naval presence in the Mediterranean and provides future flexibility to either defend against France or support further eastern operations, consistent with Italy’s concerns about overextending.

A VEN to TRI,
Misalignment: This move is misaligned with the message history and board state. Austria explicitly trusts Italy and has not shown hostility, and Italy’s previous messages suggest no immediate plans to antagonize Austria. Moving to TRI is aggressive and undermines the collaborative stance Italy communicated. It could create unnecessary conflict, contradicting Italy’s expressed preference for focusing eastward and securing Greece.


Now let's see the question:"""
        )
    
    def format_boardstates(self, boardstates):
        lines = []
        for power, units in boardstates.items():
            # Join all units with commas, and place them inside brackets
            units_str = ', '.join(units)
            line = f"{power}: [{units_str} ]"
            lines.append(line)
        # Join all lines with newlines
        return "\n".join(lines)
    
    def map_words_in_sentence(self, sentence, data):
    # Split the sentence into words
        words = sentence.split()
        mapped_words = []

        for word in words:
            # Remove punctuation from word for clean matching (if necessary)
            clean_word = word.strip(",.!?")
            word_lower = clean_word.lower()  # Convert to lowercase for case-insensitive matching

            # Check if the word is in the JSON data
            if word_lower in data:
                mapped_value = data[word_lower][0]  # Get the mapped value (e.g., "Yorkshire")
                mapped_word = word.replace(clean_word, mapped_value)
                mapped_words.append(mapped_word)
            else:
                mapped_words.append(word)

        # Join the mapped words back into a sentence
        return ' '.join(mapped_words)

    def format_prompt_phase1(self, own: str, oppo: str, suggest_orders: List[str]) -> Optional[str]:
        """
        Create prompt used as input to the LLM.
        Returns:
            A string prompt, or None if no messages exist between 'own' and 'oppo'.
        """
        # Convert power names if needed
        if own in POWER_NAMES_DICT:
            own = POWER_NAMES_DICT[own]
        if oppo in POWER_NAMES_DICT:
            oppo = POWER_NAMES_DICT[oppo]

        system_prompt = self._create_system_prompt()

        # Grab and sort board states
        board_states = self.game.get_state()["units"]
        sorted_board_states = {key: sorted(value) for key, value in board_states.items()}
        formated_board_states = self.format_boardstates(sorted_board_states)
        # Filter relevant messages
        filtered_messages = self._get_relevant_messages(own, oppo)
        if not filtered_messages:
            return None

        # Get up to 16 recent messages, ensuring last is from opponent
        recent_msgs = self._get_recent_message_history(filtered_messages, max_count=16)
        if not recent_msgs:
            return None

        # Build a text representation of the message history
        message_history = ""
        for msg in recent_msgs:
            message_history += f"Message from {msg.sender}:'{msg.message}' "
        logger.info(suggest_orders)
        parsed_data = json.loads(suggest_orders[0])
        predicted_orders = parsed_data["payload"]["predicted_orders"][oppo]

        # Convert the list to a comma-separated string
        orders_string = ", ".join(predicted_orders)

        prompt = (
            f"{system_prompt}"
            f'''
**Predicted Orders of {oppo}:**
{orders_string.replace('-','to')}

**Board State:**  
'''
f"{formated_board_states}\n\n"

f'''**Message History:**  \n{message_history}\n\n'''
f'''**Question:**  \nI am playing as {own}. For each of {oppo}'s predicted orders, evaluate whether it aligns with the message history and the current board state. Explain the orders first and provide short reasoning and analysis for each predicted move. 
\n'''
f'''**Answer:**
<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''

        )
        return prompt
    
    def format_prompt_phase2(self, oppo: str, input: str) -> Optional[str]:
        """
        Create prompt used as input to the LLM.
        Returns:
            A string prompt, or None if no messages exist between 'own' and 'oppo'.
        """
        # Convert power names if needed
        if oppo in POWER_NAMES_DICT:
            oppo = POWER_NAMES_DICT[oppo]

        prompt = (
            f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert assistant specializing in the Diplomacy board game. Based on the detailed evaluation of following predicted orders, respond in the following short and structured format:

            I think {oppo} is being deceptive, because I think they are going to make these moves: <predicted misaligned moves>. These moves are contradictory to {oppo}’s previous messages to you.
            <|eot_id|><|start_header_id|>user<|end_header_id|>  
            **Evluation**:
            {input}

            **Question**:
            from Evluation part, extract necessary information and fill in the following as the output:
            I think {oppo} is being deceptive, because I think they are going to make these moves: <predicted misaligned moves>. These moves are contradictory to {oppo}’s previous messages to you.
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>

            """
        )
        return prompt
    
    def load_model(self, base_model_name, adapter_path=None, tokenizer_path=None, device='cpu'):
        """
        Loads and returns a tokenizer and model on the requested device.
        If adapter_path is None, we won't load any adapter. 
        """
        if tokenizer_path is None:
            tokenizer_path = base_model_name

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForCausalLM.from_pretrained(base_model_name)
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
        model.to(device)

        return tokenizer, model

    def generate_text(self, prompt, tokenizer, model, device='cpu', max_new_tokens=512):
        """
        Performs text generation using the tokenizer/model loaded above.
        """
        # Ensure pad_token_id is defined
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        inputs = tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=2000,
            return_attention_mask=True
        )
        
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
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

        return tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    def generate_and_parse_response(self, prompt: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """
        1. Generate text from the model given 'prompt'.
        2. Parse out (Decision, Reason, Message) from the model output.
        Returns:
            (decision, reason, message) or (None, None, None) if something fails.
        """
        if prompt is None:
            return None

        generated_text = self.generate_text(prompt,self.tokenizer,self.model,self.device,1024)
        assistant_output = generated_text.split('assistant')[2]

        return assistant_output

    async def do_messaging_round(self, orders: Sequence[str]) -> List[str]:
        """Carry out one round of messaging, along with related tasks.

        Returns:
            List of orders to carry out.
        """
        await asyncio.sleep(random.uniform(10, 20))
        logger.info(" previous_newest_messages is :%s", self.previous_newest_messages)

        suggested_orders = await self.read_suggestions_from_advisor()
        logger.info("name of power is %s", self.power_name)
        filtered_orders = [
            order for order in suggested_orders
            if f'"advisor":"{self.power_name} (CiceroAdvisor)"' in order and '"predicted_orders"' in order
        ]
        if not filtered_orders:
            pass

        for other_power in get_other_powers([self.power_name], self.game):
            all_relevant = self._get_relevant_messages(self.power_name, other_power)
            logger.info(" all relevant is :%s", all_relevant)
            prev_msgs = self.previous_newest_messages.get(other_power)
            logger.info(" all relevant is :%s", prev_msgs)
            if prev_msgs is not None:
                # Compare to find new messages
                new_messages = [m for m in all_relevant if m not in prev_msgs]
                if not new_messages:
                    # No new messages, skip
                    continue

                # We have new messages → create prompt, generate text, parse
                prompt = self.format_prompt_phase1(self.power_name, other_power, filtered_orders)
                if prompt is None:
                    continue
                processed_prompt = self.map_words_in_sentence(prompt,mapping)
                logger.info("=========")
                logger.info(" prompt is :%s", processed_prompt)

                output_phase1 = self.generate_and_parse_response(processed_prompt)
                logger.info("=========")
                logger.info("output_phase1: %s", output_phase1)

                alignment_count = output_phase1.count("Alignment")
                misalignment_count = output_phase1.count("Misalignment")
                if misalignment_count >= alignment_count:
                    prompt2 = self.format_prompt_phase2(other_power, output_phase1)
                    output_phase2 = self.generate_and_parse_response(prompt2)
                    logger.info("=========")
                    logger.info("output_phase2: %s", output_phase2)
                    if output_phase2 and output_phase2.lstrip().startswith("I think"):
                        try:
                            await self.suggest_commentary(other_power, f"{output_phase2.lstrip()}")
                        except diplomacy.utils.exceptions.GamePhaseException as exc:
                            # Handle it gracefully: log, ignore, etc.
                            print(f"Ignoring GamePhaseException: {exc}")
                    else:
                        pass
                else:
                    # do nothing if alignment_count > misalignment_count
                    pass
                

                # if decision and reason:
                #     await self.suggest_commentary(other_power, decision + " " + reason)

            else:
                # If we have no previous messages, treat this as first time for this power
                prompt = self.format_prompt_phase1(self.power_name, other_power, filtered_orders)
                if prompt is None:
                    continue
                processed_prompt = self.map_words_in_sentence(prompt,mapping)
                logger.info("=========")
                logger.info(" prompt is :%s", processed_prompt)

                output_phase1 = self.generate_and_parse_response(processed_prompt)
                logger.info("=========")
                logger.info("output_phase1: %s", output_phase1)

                alignment_count = output_phase1.count("Alignment")
                misalignment_count = output_phase1.count("Misalignment")
                if misalignment_count >= alignment_count:
                    prompt2 = self.format_prompt_phase2(other_power, output_phase1)
                    output_phase2 = self.generate_and_parse_response(prompt2)
                    logger.info("=========")
                    logger.info("output_phase2: %s", output_phase2)
                    if output_phase2 and output_phase2.lstrip().startswith("I think"):
                        try:
                            await self.suggest_commentary(other_power, f"{output_phase2.lstrip()}")
                        except diplomacy.utils.exceptions.GamePhaseException as exc:
                            # Handle it gracefully: log, ignore, etc.
                            print(f"Ignoring GamePhaseException: {exc}")
                    else:
                        pass
                else:
                    pass

                # if decision and reason:
                #     await self.suggest_commentary(other_power, decision + " " + reason)

            # After handling new (or first-time) messages, update the record
            self.previous_newest_messages[other_power] = all_relevant

        # Done with first messaging round
        self.is_first_messaging_round = False
        return list(orders)





    # def get_random_orders(self, power_name: Optional[str] = None) -> List[str]:
    #     """Generate random orders for a power to carry out.

    #     Args:
    #         power_name: Name of power to generate random orders for.
    #             Defaults to current power.

    #     Returns:
    #         List of random orders.
    #     """
    #     if power_name is None:
    #         power_name = self.power_name

    #     possible_orders = self.game.get_all_possible_orders()
    #     orders = [
    #         random.choice(list(possible_orders[loc]))
    #         for loc in self.game.get_orderable_locations(power_name)
    #         if possible_orders[loc]
    #     ]
    #     return orders

    async def gen_orders(self) -> List[str]:
        """Generate orders for a turn.

        Returns:
            List of orders to carry out.
        """
        suggested_orders = await self.read_suggestions_from_advisor()
        filtered_orders = [
            order for order in suggested_orders
            if '"advisor":"ENGLAND (CiceroAdvisor)"' in order and '"suggested_orders"' in order
        ]
        if not filtered_orders:
            orders = []
            return orders
        parsed_data = json.loads(filtered_orders[0])
        predicted_orders = parsed_data["payload"]["suggested_orders"]
        orders = predicted_orders
        if self.bot_type == BotType.ADVISOR:
            await self.suggest_orders(orders)

        #     random_predicted_orders = {}
        #     for other_power in get_other_powers([self.power_name], self.game):
        #         random_predicted_orders[other_power] = self.get_random_orders(other_power)
        #     await self.suggest_opponent_orders(random_predicted_orders)
        # elif self.bot_type == BotType.PLAYER:
        #     await self.send_orders(orders, wait=True)
        return orders


@dataclass
class LlmAdvisor(LlmAdvisorBot):
    """Advisor form of `LlmAdvisorBot`."""

    bot_type = BotType.ADVISOR
    suggestion_type = (
        SuggestionType.MESSAGE
        | SuggestionType.MOVE
        | SuggestionType.COMMENTARY
        | SuggestionType.OPPONENT_MOVE
    )


@dataclass
class LlmPlayer(LlmAdvisorBot):
    """Player form of `LlmAdvisorBot`."""

    bot_type = BotType.PLAYER
