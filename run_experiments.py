import os
import torch
from config import Config
from train import main as train_main

# def auto_fc_layers(res_blocks, start_dim=2048, num_layers=4):
#     fc = []
#     dim = start_dim
#     for _ in range(num_layers):
#         fc.append(dim)
#         dim //= 2
#     return fc


def run_experiments():
    experiments = [
        {"res_blocks": 2, "fc_layers": [512]},
        {"res_blocks": 4, "fc_layers": [1024, 256]},
        {"res_blocks": 6, "fc_layers": [1024, 512, 256]},
        {"res_blocks": 8, "fc_layers": [1024, 512, 256, 64]},
        {"res_blocks": 10, "fc_layers": [2048, 1024, 512, 256]},
        {"res_blocks": 12, "fc_layers": [2048, 1024, 512, 256, 128]},
        {"res_blocks": 14, "fc_layers": [2048, 1024, 512, 256, 128, 64]}
    ]

    # experiments = [
    # {"res_blocks": 2, "fc_layers": auto_fc_layers(2, start_dim=1024, num_layers=2)},
    # {"res_blocks": 4, "fc_layers": auto_fc_layers(4, start_dim=1024, num_layers=3)},
    # {"res_blocks": 6, "fc_layers": auto_fc_layers(6, start_dim=2048, num_layers=4)},
    # {"res_blocks": 8, "fc_layers": auto_fc_layers(8, start_dim=2048, num_layers=5)},
    # {"res_blocks": 10, "fc_layers": auto_fc_layers(10, start_dim=4096, num_layers=5)},
    # {"res_blocks": 12, "fc_layers": auto_fc_layers(12, start_dim=4096, num_layers=6)},
    # {"res_blocks": 14, "fc_layers": auto_fc_layers(14, start_dim=4096, num_layers=6)},
    # ]


    precisions = ["fp16", "fp32", "fp64"]

    for i, exp in enumerate(experiments, 1):
        for precision in precisions:
            experiment_name = f"exp_{i}_{precision}"
            experiment_dir = os.path.join("checkpoints", experiment_name)
            os.makedirs(experiment_dir, exist_ok=True)

            # Set config
            Config.RESIDUAL_BLOCKS = exp["res_blocks"]
            Config.FULLY_CONNECTED_LAYERS = exp["fc_layers"]
            Config.MODEL_PATH = os.path.join(experiment_dir, "model.pth")
            Config.HISTORY_PATH = os.path.join(experiment_dir, "history.pth")

            # Precision modes
            if precision == "fp16":
                Config.USE_MIXED_PRECISION = True
                torch.set_default_dtype(torch.float32)  # default safe
            elif precision == "fp32":
                Config.USE_MIXED_PRECISION = False
                torch.set_default_dtype(torch.float32)
            elif precision == "fp64":
                Config.USE_MIXED_PRECISION = False
                torch.set_default_dtype(torch.float64)

            # Save experiment config
            with open(os.path.join(experiment_dir, "config.txt"), "w") as f:
                f.write(f"Residual Blocks: {exp['res_blocks']}\n")
                f.write(f"FC Layers: {exp['fc_layers']}\n")
                f.write(f"Precision: {precision.upper()}\n")

            print(f"\n🔬 Running {experiment_name} → ResBlocks={exp['res_blocks']}, Precision={precision.upper()}")
            train_main()

if __name__ == '__main__':
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    run_experiments()
