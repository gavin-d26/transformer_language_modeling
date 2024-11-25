import random
import numpy as np
import pandas as pd
import torch
from .datatools import utterances_to_tensors, clean_utterance_text

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
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds, _ = model(inputs, targets=targets)
            test_perplexity.update(preds.detach().cpu(), targets)

    log_probs = test_perplexity.compute()[1]
    df = pd.DataFrame([np.arange(len(log_probs)), log_probs], columns=["ID", "ppl"])
    df.to_csv(save_submission_file_path, index=False)
