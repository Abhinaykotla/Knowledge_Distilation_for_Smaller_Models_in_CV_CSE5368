import os
import torch
from config import Config
from train import main as train_main

def run_experiments():
    experiments = [
        {"res_blocks": 2, "fc_layers": [512]},
        {"res_blocks": 4, "fc_layers": [1024, 256]},
        {"res_blocks": 6, "fc_layers": [1024, 512, 256]},
        {"res_blocks": 8, "fc_layers": [1024, 512, 256, 64]},
    ]

    for i, exp in enumerate(experiments, 1):
        experiment_name = f"exp_{i}"
        experiment_dir = os.path.join("checkpoints", experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)

        # Update config
        Config.RESIDUAL_BLOCKS = exp['res_blocks']
        Config.FULLY_CONNECTED_LAYERS = exp['fc_layers']
        Config.MODEL_PATH = os.path.join(experiment_dir, "model.pth")
        Config.HISTORY_PATH = os.path.join(experiment_dir, "history.pth")

        # Log config for record
        with open(os.path.join(experiment_dir, "config.txt"), "w") as f:
            f.write(f"Residual Blocks: {exp['res_blocks']}\n")
            f.write(f"FC Layers: {exp['fc_layers']}\n")

        print(f"\n🔬 Running {experiment_name} - Residual Blocks: {exp['res_blocks']}, FC: {exp['fc_layers']}")
        train_main()

if __name__ == '__main__':
    run_experiments()
