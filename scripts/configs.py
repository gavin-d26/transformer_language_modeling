hp_configs = {
    "run_name": "lm-1",
    "lr": 5e-4,
    "batch_size": 256,
    "epochs": 50,
    "optimizer": "adamW",  # 'adamW'
    "dropout": 0.1,
    # tokenizer
    "tokenizer": "openai-community/gpt2",
    # transformer model
    "embed_dim": 32,
    "block_size": 128,
    "nhead": 4,
    "num_blocks": 2,
    # optimiation
    "min_lr": 1e-5,
    "num_warmpup_steps": 0.1,
    "gradient_clip_val": 1.0,
    # logging
    "wandb_flag": True,
    "project_name": "language_modeling",
    # system
    "num_proc": 4,
    "device": "cuda",
    "gpu_idx": "4",
    # notes
    "notes": "based on learnings increasing epochs to 50, increasing batch size to 256, added gradient clipping -> 1\
        , added warmup steps -> 0.1, kept dropout -> 0.1",
}

device = "cpu"
