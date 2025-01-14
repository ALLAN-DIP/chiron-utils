# llm_inference.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from torch.nn import DataParallel

import os
import json
import csv

from tqdm import tqdm
import json
import re

def load_model(base_model_name, adapter_path=None, tokenizer_path=None, device='cpu'):
    """
    Loads and returns a tokenizer and model on the requested device.
    If adapter_path is None, we won't load any adapter. 
    """
    if tokenizer_path is None:
        tokenizer_path = base_model_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_name)

    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)

    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model.to(device)

    return tokenizer, model

def generate_text(prompt, tokenizer, model, device='cpu', max_new_tokens=512):
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

# def replace_values_with_keys(sentence):
#     for key, values in dictionary.items():
#         for value in values:
#             sentence = re.sub(r'\b' + re.escape(value) + r'\b', key, sentence)
#     return sentence
    
def map_words_in_sentence(sentence, data):
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

    return ' '.join(mapped_words)