hp_configs = {
    "run_name": "lm-6",
    "lr": 5e-4,
    "batch_size": 129,
    "epochs": 69,
    "optimizer": "adamW",  # 'adamW'
    "dropout": 0.1,
    # tokenizer
    "tokenizer": "openai-community/gpt2",
    # transformer model
    "embed_dim": 32,
    "block_size": 128,
    "nhead": 2,
    "num_blocks": 5,
    # optimiation
    "min_lr": 2e-5,
    "num_warmpup_steps": 0.1,
    "gradient_clip_val": 0.5,
    "grad_accumulation_steps": 8,
    # logging
    "wandb_flag": True,
    "project_name": "language_modeling",
    # system
    "num_proc": 4,
    "device": "cuda",
    "gpu_idx": "3",
    # notes
    "notes": "based on learnings increasing epochs to 120, increasing batch size to 256, added gradient clipping -> 0.5\
        , added warmup steps -> 0.1, kept dropout -> 0.1, gradient accumulation steps -> 2, decreased nhead -> 2,",
}
