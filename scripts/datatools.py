import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset
from scripts.configs import hp_configs

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def load_ptb_dataset():
    return load_dataset("ptb-text-only/ptb_text_only")


# creates dataloaders for train and val datasets, Note: it no longer uses vectorizer
def create_dataloaders(
    batch_size=32,
    tokenizer=None,
    vocab_size=32000,
    block_size=128,
    device="cpu",
    num_workers=2,
):
    """

    Args:
        batch_size (int, optional): Defaults to 32.
        tokenizer (str, optional): _description_. Defaults to None.
        vocab_size (int, optional): _description_. Defaults to 32000.

    Returns:
        _type_: _description_

    Yields:
        _type_: _description_
    """

    # load ptb dataset
    ptb_dataset = load_ptb_dataset()

    # create tokenizer
    special_tokens_dict = {
        "pad_token": "[PAD]",
        "eos_token": "[EOS]",
        "bos_token": "[BOS]",
    }

    if tokenizer is None:

        def batch_iterator(batch_size=1000):
            for batch in ptb_dataset["train"].iter(batch_size):
                yield batch["sentence"]

        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Sequence(Whitespace(), Punctuation())
        trainer = BpeTrainer(
            special_tokens=list(special_tokens_dict.values()),
            vocab_size=vocab_size,
            show_progress=True,
        )
        tokenizer.train_from_iterator(
            batch_iterator(), trainer=trainer, length=len(ptb_dataset["train"])
        )

    else:
        tokenizer = Tokenizer.from_pretrained(tokenizer)

    tokenizer.add_special_tokens(list(special_tokens_dict.values()))

    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )

    # enable padding and truncation based on block size
    tokenizer.enable_padding(length=block_size)
    tokenizer.enable_truncation(max_length=block_size)

    PADDING_TOKEN_ID = tokenizer.token_to_id("[PAD]")
    BOS_TOKEN_ID = tokenizer.token_to_id("[BOS]")
    EOS_TOKEN_ID = tokenizer.token_to_id("[EOS]")

    # prerocessing function for applying the tokenizer
    def batch_preprocess(batch):
        output = {
            "sentence": tokenizer.encode_batch(
                batch["sentence"].ids, add_special_tokens=True
            )
        }
        return output

    # preprocess the datasets
    ptb_dataset = ptb_dataset.map(batch_preprocess, batched=True, num_proc=num_workers)

    # seed the workers
    def worker_init_fn(worker_id):
        seed = 0
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    train_loader = torch.utils.data.DataLoader(
        ptb_dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device == "cuda",
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        ptb_dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device == "cuda",
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )

    test_loader = torch.utils.data.DataLoader(
        ptb_dataset["test"],
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device == "cuda",
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        PADDING_TOKEN_ID,
        BOS_TOKEN_ID,
        EOS_TOKEN_ID,
    )
