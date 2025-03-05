"""Bots that carry out random orders and make random order proposals."""

from abc import ABC
import asyncio
from dataclasses import dataclass, field
import json
import random
from typing import Dict, List, Optional, Sequence, Tuple, Union

import diplomacy
from diplomacy import Message
from diplomacy.utils.constants import SuggestionType
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, DataParallel
from sklearn.preprocessing import StandardScaler

from chiron_utils.bots.baseline_bot import BaselineBot, BotType
from chiron_utils.utils import POWER_NAMES_DICT, get_other_powers, mapping, return_logger

logger = return_logger(__name__)


class BERTWithNumericalFeatures(nn.Module):
    def __init__(self, num_numeric_features):
        super(BERTWithNumericalFeatures, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.nn_layers = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + num_numeric_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, numeric_features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_embedding = bert_output.pooler_output  # [CLS] token representation

        # Concatenate BERT embedding with numerical features
        combined_features = torch.cat((bert_embedding, numeric_features), dim=1)

        # Pass through NN layers
        return self.nn_layers(combined_features)

@dataclass
class DeceptionBertAdvisor(BaselineBot, ABC):
    """Bot that carries out random orders and sends random order proposals to other bots.

    Because of the similarity between the advisor and player versions of this bot,
    both of their behaviors are abstracted into this single abstract base class.
    """

    is_first_messaging_round = False
    # trained BERTWithNumericalFeatures model and BERT tokennizer paths
    model_path = 'src/chiron_utils/models/bert_based_deception/best_model_epoch_10.pth'
    tokenizer_path = 'src/chiron_utils/models/bert_based_deception/'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bot_type = BotType.ADVISOR
    default_suggestion_type = SuggestionType.COMMENTARY
    scaler = StandardScaler()
    _ = scaler.fit_transform(np.array([[0, -1, -1, -1], [15, 0.0933, 0.1047, 0.0725]]))

    def __post_init__(self) -> None:
        """Initialize models."""
        self.tokenizer, self.model = self.load_model(self.model_path, self.tokenizer_path, self.device)

    async def start_phase(self) -> None:
        """Execute actions at the start of the phase."""
        self.is_first_messaging_round = True

    def get_recent_message_history(
        self, messages: List[Message], max_count: int = 8
    ) -> List[Message]:
        """Sort a list of messages by `time_sent` descending and return up to max_count most recent.

        Also ensures the very last message is from the opponent (if available).
        """
        if not messages:
            return []

        sorted_msgs = sorted(messages, key=lambda x: x.time_sent, reverse=True)
        recent_msgs = sorted_msgs[:max_count]
        reversed_msgs = list(reversed(recent_msgs))

        if reversed_msgs and reversed_msgs[-1].sender == self.power_name:
            return []

        return reversed_msgs

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

    def load_model(
        self,
        model_path : str,
        tokenizer_path: str,
        device: str = "cpu",
    ) -> Tuple[PreTrainedTokenizer, Union[PreTrainedModel, DataParallel]]:
        """Loads and returns a tokenizer and model on the requested device."""

        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
   
        model = BERTWithNumericalFeatures(num_numeric_features=4)
        model.load_state_dict(torch.load(model_path))  # Load the best model
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
        model.to(device)

        return tokenizer, model

    def predict_deception(self, msg: dict, numerical_features: List[float]) -> bool:
        """predict deception using message (dictionary) and a score and deception values (list of float).

        Returns:
            is_deceptive 
        """
        # eval bert 
        numerical_features = self.scaler.transform(np.array([numerical_features], dtype=float))
        numerical_features = torch.tensor(numerical_features, dtype=torch.float32)
        text = f"{msg['sender']} sends to {msg['recipient']} with a message: {msg['message']}"
        tokenized_text = self.tokenizer(
                        list([text]), padding=True, truncation=True, max_length=512, return_tensors="pt")
        predictions = self.model(tokenized_text['input_ids'].to(self.device), tokenized_text['attention_mask'].to(self.device), numerical_features.to(self.device))
        result = predictions[0].item()
        is_deceptive = result >= 0.5
        return is_deceptive
    
    async def do_messaging_round(self, orders: Sequence[str]) -> List[str]:
        """Carry out one round of messaging, along with related tasks.

        Returns:
            List of orders to carry out.
        """
        await asyncio.sleep(random.uniform(5, 10))
        
        # must check for new message, if not then dont suggest deception for last sender-recipient message

        suggested_deceptions = await self.read_suggestions_from_advisor()
        filtered_deceptions = [
            deception_msg
            for deception_msg in suggested_deceptions
            if f'"advisor":"{self.power_name} (CiceroAdvisor)"' in deception_msg
            and '"deceptive_values":' in deception_msg
        ]
        if not filtered_deceptions:
            pass
        
        parsed_data = json.loads(filtered_deceptions[0])
        payload = parsed_data["payload"]
        score = payload["score"]
        deceptive_values = payload["deceptive_values"]
        is_deception = self.predict_deception(payload, [score] + deceptive_values)
        deception_commentary = f"""detect possible deception in {payload['sender']} if they promise to do followings: /n
                            {payload['d_proposed_action']} /n
                            or ask you to do followings: /n
                            {payload['v_proposed_action']} /n
                            we recommend you to be cautious and proceed with your best move in this situation:
                            {payload['V_best']}"""
        
        if is_deception:
            try:
                await self.suggest_commentary(payload['sender'], deception_commentary)
            except diplomacy.utils.exceptions.GamePhaseException as exc:
                logger.info("Ignoring GamePhaseException:, %s", exc)
        
        return list(orders)

    async def gen_orders(self) -> List[str]:
        """Generate None for orders

        Returns:
            None
        """
        # since this is advisor bot, it will not generate orders
        return None
