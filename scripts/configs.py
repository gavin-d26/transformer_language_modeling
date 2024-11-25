hp_configs = {
    "run_name": "test-1",
    "lr": 1e-3,
    "batch_size": 128,
    "epochs": 10,
    "optimizer": "adamW",  # 'adamW'
    "dropout": 0.1,
    # tokenizer
    "tokenizer": "openai-community/gpt2",
    # transformer model
    "embed_dim": 32,
    "block_size": 128,
    "nhead": 4,
    "num_blocks": 1,
    # optimiation
    "min_lr": 1e-5,
    "num_warmpup_steps": 50,
    # logging
    "wandb_flag": True,
    "project_name": "language_modeling",
    # system
    "num_proc": 2,
    "device": "cuda",
    "gpu_idx": "4",
    # notes
    "notes": "",
}

device = "cpu"
