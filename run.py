import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
import scripts.configs as configs
import scripts.datatools as datatools
import scripts.models as models
import scripts.train as train
import scripts.predict as predict

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.use_deterministic_algorithms(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_csv_path")

    args = parser.parse_args()

    # parse script args
    submission_csv_path = args.submission_csv_path

    # set environment variables
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["CUDA_VISIBLE_DEVICES"] = configs.hp_configs["gpu_idx"]

    # create vectorizer and dataloaders
    (
        train_loader,
        val_loader,
        test_loader,
        PADDING_TOKEN_ID,
        BOS_TOKEN_ID,
        EOS_TOKEN_ID,
        VOCAB_SIZE,
    ) = datatools.create_dataloaders(
        batch_size=configs.hp_configs["batch_size"],
        tokenizer=configs.hp_configs["tokenizer"],
        vocab_size=32000,
        block_size=configs.hp_configs["block_size"],
        device=configs.hp_configs["device"],
        num_workers=configs.hp_configs["num_proc"],
    )

    # initialize model
    model = models.Transformer(
        embed_dim=configs.hp_configs["embed_dim"],
        block_size=configs.hp_configs["block_size"],
        vocab_size=VOCAB_SIZE,
        num_heads=configs.hp_configs["nhead"],
        num_blocks=configs.hp_configs["num_blocks"],
        dropout=configs.hp_configs["dropout"],
        PADDING_TOKEN_ID=PADDING_TOKEN_ID,
        BOS_TOKEN_ID=BOS_TOKEN_ID,
    )

    train.train_func(
        model,
        train_loader,
        val_loader,
        hp_config=configs.hp_configs,
        wandb_flag=False,
        PADDING_TOKEN_ID=PADDING_TOKEN_ID,
    )

    mean_test_perplexity = predict.make_submission_file(
        model,
        test_loader,
        save_submission_file_path=submission_csv_path,
        device=configs.hp_configs["device"],
    )

    with open("mean_test_perplexity.txt", "a") as file:
        print(f"{configs.hp_configs['run_name']}: {mean_test_perplexity}", file=file)


if __name__ == "__main__":
    main()
