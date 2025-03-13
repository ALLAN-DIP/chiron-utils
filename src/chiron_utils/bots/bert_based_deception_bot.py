"""Bots that carry out random orders and make random order proposals."""

from abc import ABC
import asyncio
from dataclasses import dataclass, field
import json
import random
from typing import Dict, List, Optional, Sequence, Tuple, Union
import copy
import joblib
import diplomacy
from diplomacy import Message
from diplomacy.utils.constants import SuggestionType
import torch
from torch.nn import DataParallel
import torch.nn as nn

import numpy as np
from transformers import BertTokenizer, BertModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from sklearn.preprocessing import StandardScaler

from chiron_utils.bots.baseline_bot import BaselineBot, BotType, SENTINEL
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
    suggestion_type = SuggestionType.COMMENTARY

    scaler = joblib.load("src/chiron_utils/models/bert_based_deception/scaler.pkl")

    def __post_init__(self) -> None:
        """Initialize models."""

        super().__post_init__()  # Call BaselineBot's __post_init__
        if self.suggestion_type is None or self.suggestion_type == SENTINEL:
            self.suggestion_type = self.default_suggestion_type

        print(f"DEBUG: DeceptionBertAdvisor initialized with suggestion_type = {self.suggestion_type}")
        self.tokenizer, self.model = self.load_model(self.model_path, self.tokenizer_path, self.device)
        self.last_predict_deception = dict()

    async def start_phase(self) -> None:
        """Execute actions at the start of the phase."""
        self.is_first_messaging_round = True
        self.last_predict_deception = dict()


    def extract_features(self, messages):
        count_lies = 0
        count_non_rl = 0
        max_non_rl = 100
        data =[]
        for msg in messages:
            text = f"{msg['sender']} sends to {msg['recipient']} with a message: {msg['message']}"
            label = 1 if not msg['sender_labels'] else 0
            scores = msg.get('scores',0)
            if label ==1:
                count_lies+=1

            # Extract friction features if available, selecting the entry with the highest sum
            if msg['friction_info']:
                best_friction = max(
                    msg['friction_info'],
                    key=lambda x: sum([x.get('1_rule', -1), x.get('2_rule', -1), x.get('3_rule', -1)])
                )
                features = [scores, best_friction.get('1_rule', -1), best_friction.get('2_rule', -1), best_friction.get('3_rule', -1)]
            
            else:
                features = [scores, -1, -1,-1 ]
            data.append((text, features, label))
                
        return data
    
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
        numerical_features = self.scaler.transform(np.array([numerical_features]))
        numerical_features = torch.tensor(numerical_features, dtype=torch.float32)
        text = f"{msg['sender']} sends to {msg['recipient']} with a message: {msg['message']}"
        tokenized_text = self.tokenizer(
                        list([text]), padding=True, truncation=True, max_length=512, return_tensors="pt")
        logger.info(f"input_ids: {tokenized_text['input_ids']}")
        logger.info(f"attention_mask: {tokenized_text['attention_mask']}")
        logger.info(f"numerical_features: {numerical_features}")
        predictions = self.model(tokenized_text['input_ids'].to(self.device), tokenized_text['attention_mask'].to(self.device), numerical_features.to(self.device))
        logger.info(f'predictions: {predictions} >= 0.5')
        result = predictions[0].item()
        is_deceptive = result >= 0.5
        return is_deceptive

    async def read_suggestions_from_advisor(self) -> List[str]:
        """Read suggestions from CiceroAdvisor.

        Returns:
            List of suggested orders.
        """
        received_messages = self.read_messages()
        suggestions = [msg.message for msg in received_messages]
        logger.info("%s received suggested message: %s", self.display_name, suggestions)
        return suggestions if suggestions else []
    
    async def do_messaging_round(self, orders: Sequence[str]) -> List[str]:
        """Carry out one round of messaging, along with related tasks.

        Returns:
            List of orders to carry out.
        """
        await asyncio.sleep(random.uniform(5, 10))
        orders = []
        # must check for new message, if not then dont suggest deception for last sender-recipient message

        suggested_deceptions = await self.read_suggestions_from_advisor()
        filtered_deceptions = [
            deception_msg
            for deception_msg in suggested_deceptions
            if f'"advisor":"{self.power_name} (CiceroAdvisor)"' in deception_msg
            and '"deceptive_values":' in deception_msg
        ]
        if not len(filtered_deceptions):
            return orders
        
        parsed_data = json.loads(filtered_deceptions[-1])
        payload = parsed_data["payload"]
        if payload == self.last_predict_deception:
            return orders
        
        scores = payload["scores"]
        deceptive_values = payload["deceptive_values"]
        is_deception = self.predict_deception(payload, [scores] + deceptive_values)
        
        if payload['d_proposed_action'] == 'None':
            deception_commentary = f"""We detect possible deception in {payload['sender']}'s message saying \"{payload['message']}\", we recommend you to be cautious!"""
        else:
            deception_commentary = f"""We detect possible deception in {payload['sender']}'s message saying \"{payload['message']}\". If they promise to do the following:
                                {payload['d_proposed_action']}
                                or ask you to do the following: 
                                {payload['v_proposed_action']}
                                we recommend you to be cautious and proceed with your best move in this situation:
                                {payload['V_best']}"""
        self.last_predict_deception = copy.deepcopy(payload)
        logger.info(f"Deception prediction a message {payload}: {is_deception}")
        
        if is_deception:
            try:
                await self.suggest_commentary(payload['sender'], deception_commentary)
            except diplomacy.utils.exceptions.GamePhaseException as exc:
                logger.info("Ignoring GamePhaseException:, %s", exc)
        
        return orders

    async def gen_orders(self) -> List[str]:
        """Generate None for orders

        Returns:
            None
        """
        # since this is advisor bot, it will not generate orders
        return None
