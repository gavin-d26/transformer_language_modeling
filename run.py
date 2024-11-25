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

    # get train and val dataframes
    ptb_dataset = load_dataset("ptb-text-only/ptb_text_only")

    # create vectorizer and dataloaders
    (
        train_loader,
        val_loader,
        test_loader,
        PADDING_TOKEN_ID,
        BOS_TOKEN_ID,
        EOS_TOKEN_ID,
    ) = datatools.create_dataloaders(ptb_dataset, configs.hp_configs["batch_size"])

    # initialize model
    model = models.Transformer(
        embed_dim=configs.hp_configs["embed_dim"],
        num_heads=configs.hp_configs["nhead"],
        num_blocks=configs.hp_configs["num_blocks"],
        dropout=configs.hp_configs["dropout"],
        vocab_size=len(ptb_dataset["train"]["text"][0]),
        padding_idx=PADDING_TOKEN_ID,
        bos_token_id=BOS_TOKEN_ID,
    )

    train.train_func(
        model,
        train_loader,
        val_loader,
        hp_config=configs.hp_configs,
        device=configs.device,
        wandb_flag=False,
    )

    predict.make_submission_file(
        model,
        test_loader,
        save_submission_file_path=submission_csv_path,
        device=configs.device,
    )


if __name__ == "__main__":
    main()
