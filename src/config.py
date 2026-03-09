import torch

config = {

    # TARGETS TOKEN FOR PRETRAINING OT SFT
    'stf_target_tokens': 100_000_000,
    'pre_training_target_tokens': 20_000_000_000,

    # PATHS
    'dataset_path': "data/raw",
    'tokenizer_path': "data/tokenizer",
    'checkpoint_file_path': "checkpoints/checkpoint.pt",

    # MODEL ARCHITECTURE
    'd_model': 576,
    'n_layer': 30,
    'n_head': 9,
    'block_size': 2048,
    'n_kv_head': 3,
    'vocab_size': 49152,

    'use_layernorm': True,
    'use_swiglu': True,

    # BATCHS
    'batch_size': 256,
    'micro_batch_size': 4,  
    
    # OPTIMIZER
    'max_lr': 3e-3,
    
    # LOGGING INTERVAL
    'logging_interval': 2,

    # DEVICE
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # FP8 PRECISON (USING TRANSFORMER ENGINE FROM NVIDIA)
    'use_te': False,

    # CURRICULUM LEARNING RATIOS
    'stage_ratios': {
        1: {"web": 0.85, "python": 0.15, "cosmo": 0.0, "math": 0.0, "talk": 0.0},
        2: {"web": 0.70, "python": 0.20, "cosmo": 0.10, "math": 0.0, "talk": 0.0},
        3: {"web": 0.55, "python": 0.20, "cosmo": 0.10, "math": 0.15, "talk": 0.0},
        4: {"web": 0.35, "python": 0.20, "cosmo": 0.25, "math": 0.20, "talk": 0.0},
        5: {"web": 0.15, "python": 0.0,  "cosmo": 0.10, "math": 0.0, "talk": 0.75}
    }
    
}