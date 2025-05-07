# Multimodal Transfer Learning / Image Captioning
# File: src/common/tokenizer.py
# Copyright (c) 2025 Dropout Disco Team (Yurii, Artemis, Nnamdi, Kenton)
# Description: Text tokenizer setup and utility functions.
# Created: 2025-05-06
# Updated: 2025-05-06

try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False

from typing import List, Optional, Dict
import os
import sys
from pathlib import Path

# --- Add project root for imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils import logger
from src.common.constants import (
    PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN,
    PAD_TOKEN_ID, START_TOKEN_ID, END_TOKEN_ID, UNK_TOKEN_ID
)

# --- Tokenizer Initialization ---
# Use a common pre-trained tokenizer, e.g., GPT2
# This provides a good base vocabulary and tokenization rules.
DEFAULT_TOKENIZER_NAME = "gpt2"

def init_tokenizer(
    model_name_or_path: str = DEFAULT_TOKENIZER_NAME,
    add_special_tokens: bool = True
) -> Optional[AutoTokenizer]:
    """
    Initializes a Hugging Face tokenizer and adds special tokens.
    Updates global constants for token IDs.

    Args:
        model_name_or_path (str): HF model name for the tokenizer.
        add_special_tokens (bool): Whether to add custom PAD, START, END, UNK.

    Returns:
        Optional[AutoTokenizer]: The initialized tokenizer, or None on failure.
    """
    global PAD_TOKEN_ID, START_TOKEN_ID, END_TOKEN_ID, UNK_TOKEN_ID
    global DECODER_VOCAB_SIZE # Will be set after adding tokens

    logger.info(f"🧠 Initializing tokenizer: {model_name_or_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if add_special_tokens:
            special_tokens_dict = {
                'pad_token': PAD_TOKEN,
                'bos_token': START_TOKEN, # Beginning Of Sequence
                'eos_token': END_TOKEN,   # End Of Sequence
                'unk_token': UNK_TOKEN
            }
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            logger.info(f"Added {num_added_toks} special tokens.")

            # Update global ID constants based on the tokenizer's mapping
            PAD_TOKEN_ID = tokenizer.pad_token_id
            START_TOKEN_ID = tokenizer.bos_token_id
            END_TOKEN_ID = tokenizer.eos_token_id
            UNK_TOKEN_ID = tokenizer.unk_token_id

            logger.info(f"  <pad> ID: {PAD_TOKEN_ID}")
            logger.info(f"  <start> ID: {START_TOKEN_ID}")
            logger.info(f"  <end> ID: {END_TOKEN_ID}")
            logger.info(f"  <unk> ID: {UNK_TOKEN_ID}")

        # Important: After adding tokens, the vocab size changes
        DECODER_VOCAB_SIZE = len(tokenizer)
        logger.info(f"✅ Tokenizer initialized. Vocab size: {DECODER_VOCAB_SIZE}")
        return tokenizer

    except Exception as e:
        logger.error(f"❌ Failed to initialize tokenizer: {e}", exc_info=True)
        return None

def tokenize_captions(
    captions: List[str],
    tokenizer: AutoTokenizer,
    max_len: int,
    add_start_end: bool = True
) -> Dict[str, List[List[int]]]:
    """
    Tokenizes a list of captions, adds special tokens, and pads/truncates.

    Args:
        captions (List[str]): List of caption strings.
        tokenizer (AutoTokenizer): Initialized Hugging Face tokenizer.
        max_len (int): Maximum sequence length after tokenization.
        add_start_end (bool): Whether to prepend START and append END tokens.

    Returns:
        Dict[str, List[List[int]]]: Dictionary with 'input_ids' and 'attention_mask'.
    """
    if not tokenizer:
        logger.error("Tokenizer not provided for tokenizing captions.")
        return {'input_ids': [], 'attention_mask': []}

    # Prepend START and append END tokens if required
    # Note: Some tokenizers might handle this automatically if configured
    # For explicit control, we do it here.
    processed_captions = []
    if add_start_end:
        for cap in captions:
            # Check if tokenizer.bos_token and tokenizer.eos_token are strings
            start_tok_str = tokenizer.bos_token if tokenizer.bos_token else START_TOKEN
            end_tok_str = tokenizer.eos_token if tokenizer.eos_token else END_TOKEN
            processed_captions.append(f"{start_tok_str} {cap} {end_tok_str}")
    else:
        processed_captions = captions

    encoded = tokenizer(
        processed_captions,
        add_special_tokens=False, # We handled special tokens above
        max_length=max_len,
        padding='max_length', # Pad to max_len
        truncation=True,      # Truncate if longer than max_len
        return_attention_mask=True,
        return_tensors=None # Return lists of ints
    )
    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask']
    }

# --- Test Block ---
if __name__ == "__main__":
    logger.info("🧪 Testing Tokenizer Utils...")
    test_tokenizer = init_tokenizer()
    if test_tokenizer:
        print(f"Global PAD_TOKEN_ID: {PAD_TOKEN_ID}") # Check if updated
        captions_sample = [
            "A cat sits on a mat.",
            "Two dogs play in the park with a ball."
        ]
        max_seq_len_test = 20
        tokenized_output = tokenize_captions(
            captions_sample, test_tokenizer, max_seq_len_test
        )
        logger.info(f"Sample tokenized output ('input_ids'):")
        for ids in tokenized_output['input_ids']:
            print(ids)
            print(f"Decoded: {test_tokenizer.decode(ids)}")
        logger.info("✅ Tokenizer test finished.")