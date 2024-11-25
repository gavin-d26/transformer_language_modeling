import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import get_scheduler
import wandb

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# class to compute  F1 Score
class Perplexity:
    def __init__(self, padding_value):
        self.log_probs = None
        self.padding_value = padding_value

    # append new batches
    def update(self, logits, targets):
        # logits.shape = (batch_size, block_size, num_classes)
        # targets.shape = (batch_size, block_size)
        N, S, C = logits.shape

        logits = (
            F.softmax(logits, dim=-1)
            .gather(dim=-1, index=targets.unsqueeze(-1))
            .squeeze(-1)
        )  # (N, S)

        mask = targets == self.padding_value

        logits = logits.masked_fill(mask, 1)
        log_probs = torch.log(logits).sum(dim=-1) / torch.logical_not(mask).sum(dim=-1)

        self.log_probs = (
            torch.cat([self.log_probs, log_probs], dim=0)
            if self.log_probs is not None
            else log_probs
        )

    # used to compute F1 Score at the end of an epoch
    def compute(self):
        mean_perplexity = torch.exp(-self.log_probs.mean()).cpu().numpy()
        perplexity = torch.exp(-self.log_probs).cpu().numpy()
        self.reset()
        return mean_perplexity, perplexity

    def reset(self):
        self.log_probs = None


# function to train the pytorch model
def train_func(
    model,
    train_loader,
    val_loader,
    hp_config,
    wandb_flag=False,
    PADDING_TOKEN_ID=None,
):

    run_name = hp_config["run_name"]
    epochs = hp_config["epochs"]
    lr = hp_config["lr"]
    min_lr = hp_config["min_lr"]
    num_warmpup_steps = hp_config["num_warmpup_steps"]
    optimizer = hp_config["optimizer"]
    device = hp_config["device"]
    project_name = hp_config["project_name"]

    device = torch.device(device)
    model.to(device=device)

    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    scheduler = get_scheduler(
        "cosine_with_min_lr",
        optimizer,
        num_warmup_steps=num_warmpup_steps,
        num_training_steps=epochs * len(train_loader),
        scheduler_specific_kwargs={"min_lr": min_lr},
    )

    train_perlexity = Perplexity(padding_value=PADDING_TOKEN_ID)
    val_perplexity = Perplexity(padding_value=PADDING_TOKEN_ID)

    min_val_perlexity = float("inf")
    best_epoch_train_perplexity = float("inf")
    max_epoch = 0

    if wandb_flag is True:
        wandb.init(project=project_name, name=run_name, config=hp_config)

    print(f"Starting Training: {run_name}")

    for epoch in tqdm(range(epochs)):
        print(f"-------- Epoch {epoch} --------")

        train_loss = []
        val_loss = []

        # train on train set
        model.train()
        for batch in tqdm(train_loader, desc="Training"):
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]

            preds, loss = model(inputs, targets=targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_perlexity.update(preds.detach(), targets)
            train_loss.append(loss.detach().cpu())
            scheduler.step()

        # evaluate on val set
        model.eval()
        with torch.inference_mode():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = batch.to(device)
                inputs, targets = batch[:, :-1], batch[:, 1:]

                preds, loss = model(inputs, targets=targets)
                val_loss.append(loss.detach().cpu())
                val_perplexity.update(preds.detach(), targets)

        metrics = {
            "train_loss": sum(train_loss) / len(train_loss),
            "train_perlexity": train_perlexity.compute()[0],
            "val_loss": sum(val_loss) / len(val_loss),
            "val_perplexity": val_perplexity.compute()[0],
        }

        if wandb_flag is True:
            wandb.log(
                {
                    "Loss/train": metrics["train_loss"],
                    "Loss/val": metrics["val_loss"],
                    "Perplexity/train": metrics["train_perlexity"],
                    "Accuracy/val": metrics["val_perplexity"],
                    "epoch": epoch,
                }
            )

        if best_epoch_train_perplexity == float("inf"):
            best_epoch_train_perplexity = metrics["train_perlexity"]

        if metrics["val_perplexity"] < min_val_perlexity:
            if not os.path.isdir("./checkpoints"):
                os.mkdir("./checkpoints")
            torch.save(model.state_dict(), "./checkpoints/model.pt")
            min_val_perlexity = metrics["val_perplexity"]
            best_epoch_train_perplexity = metrics["train_perlexity"]
            max_epoch = epoch

        print(
            f'train_loss: {metrics["train_loss"]:.2f}   val_loss: {metrics["val_loss"]:.2f}   train_perlexity: {metrics["train_perlexity"]:.2f} \
                val_perplexity": {metrics["val_perplexity"]:.2f}'
        )

    print(f"best model at epoch: {max_epoch}")
    if wandb_flag is True:
        wandb.summary.update(
            {
                "best_epoch": max_epoch,
                "best_val_perplexity": min_val_perlexity,
                "best_epoch_train_perplexity": best_epoch_train_perplexity,
            }
        )
        wandb.finish()

    model.to(device=torch.device("cpu"))
    model.load_state_dict(torch.load("./checkpoints/model.pt", map_location="cpu"))
