import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# func to create a .csv file for kaggle submission
def make_submission_file(
    model, test_dataloader, save_submission_file_path="submission.csv", device="cpu"
):
    device = torch.device(device)
    model.to(device)
    model.eval()
    from scripts.train import Perplexity

    test_perplexity = Perplexity(padding_value=model.PADDING_TOKEN_ID)

    with torch.inference_mode():
        for batch in tqdm(test_dataloader, desc="Testing"):
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            preds, _ = model(inputs, targets=targets)
            test_perplexity.update(preds.detach(), targets)

    log_probs = test_perplexity.compute()[1]
    df = pd.DataFrame({"ID": np.arange(len(log_probs)), "ppl": log_probs})
    df.to_csv(save_submission_file_path, index=False)
