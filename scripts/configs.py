hp_configs = {
    "run_name": "test-1",
    "lr": 1e-3,
    "batch_size": 128,
    "epochs": 100,
    "optimizer": "adamW",  # 'adamW'
    "dropout": 0.4,
    # transformer model
    "embed_dim": 32,
    "block_size": 128,
    "nhead": 4,
    "num_blocks": 1,
    # optimiation
    "min_lr": 1e-5,
    "num_warmpup_steps": 50,
    # logging
    "wandb_flag": False,
    # system
    "num_proc": 2,
    "device": "cuda",
    "gpu_idx": "4",
    # notes
    "notes": "",
}

device = "cpu"
