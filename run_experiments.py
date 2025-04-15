import os
import torch
from config import Config
from train import main as train_main

def run_experiments():
    experiments = [
        {"res_blocks": 2, "fc_layers": [512]},
        {"res_blocks": 4, "fc_layers": [1024, 256]},
    ]

# def run_experiments():
#     experiments = [
#         {"res_blocks": 2, "fc_layers": [512]},
#         {"res_blocks": 4, "fc_layers": [1024, 256]},
#         {"res_blocks": 6, "fc_layers": [1024, 512, 256]},
#         {"res_blocks": 8, "fc_layers": [1024, 512, 256, 64]},
#         {"res_blocks": 10, "fc_layers": [2048, 1024, 512, 256]},
#         {"res_blocks": 12, "fc_layers": [2048, 1024, 512, 256, 128]},
#         {"res_blocks": 14, "fc_layers": [2048, 1024, 512, 256, 128, 64]}
#     ]


    precisions = ["fp16", "fp32", "fp64"]

    for i, exp in enumerate(experiments, 1):
        for precision in precisions:
            experiment_name = f"exp_{i}_{precision}"
            experiment_dir = os.path.join("checkpoints", experiment_name)
            os.makedirs(experiment_dir, exist_ok=True)

            # Clear memory
            torch.cuda.empty_cache()

            # Set config
            Config.RESIDUAL_BLOCKS = exp["res_blocks"]
            Config.FULLY_CONNECTED_LAYERS = exp["fc_layers"]
            Config.MODEL_PATH = os.path.join(experiment_dir, "model.pth")
            Config.HISTORY_PATH = os.path.join(experiment_dir, "history.pth")

            # Set precision
            if precision == "fp16":
                Config.USE_MIXED_PRECISION = True
                torch.set_default_dtype(torch.float32)
            elif precision == "fp32":
                Config.USE_MIXED_PRECISION = False
                torch.set_default_dtype(torch.float32)
            elif precision == "fp64":
                Config.USE_MIXED_PRECISION = False
                torch.set_default_dtype(torch.float64)

            # Save config
            with open(os.path.join(experiment_dir, "config.txt"), "w") as f:
                f.write(f"Residual Blocks: {exp['res_blocks']}\n")
                f.write(f"FC Layers: {exp['fc_layers']}\n")
                f.write(f"Precision: {precision.upper()}\n")

            print(f"\n\U0001f52c Running {experiment_name} → ResBlocks={exp['res_blocks']}, Precision={precision.upper()}")
            train_main()

            # Optional: cleanup
            torch.cuda.empty_cache()

if __name__ == '__main__':
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    run_experiments()
