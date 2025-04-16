import os
import torch
import gc
from config import Config
from train import main as train_main

def run_experiments():
    experiments = [
        # {"res_blocks": 4, "fc_layers": [16, 8]}

        {"res_blocks": 14, "fc_layers": [512, 256, 128, 64, 32, 16, 8]},
        {"res_blocks": 10, "fc_layers": [512, 256, 128, 64, 32, 16]},
        {"res_blocks": 6,  "fc_layers": [256, 128, 64, 32, 16]}
    ]

# def run_experiments():
    # experiments = [
    #     {"res_blocks": 32, "fc_layers": [2048, 1024, 512, 256, 128, 64, 32, 16, 8]},
    #     {"res_blocks": 30, "fc_layers": [2048, 1024, 512, 256, 128, 64, 32, 16]},
    #     {"res_blocks": 28, "fc_layers": [1024, 512, 256, 128, 64, 32, 16, 8]},
    #     {"res_blocks": 26, "fc_layers": [1024, 512, 256, 128, 64, 32, 16]},
    #     {"res_blocks": 24, "fc_layers": [512, 256, 128, 64, 32, 16, 8]},
    #     {"res_blocks": 22, "fc_layers": [512, 256, 128, 64, 32, 16]},
    #     {"res_blocks": 20, "fc_layers": [256, 128, 64, 32, 16, 8]},
    #     {"res_blocks": 18, "fc_layers": [256, 128, 64, 32, 16]},
    #     {"res_blocks": 16, "fc_layers": [128, 64, 32, 16, 8]},
    #     {"res_blocks": 14, "fc_layers": [128, 64, 32, 16]},
    #     {"res_blocks": 12, "fc_layers": [64, 32, 16, 8]},
    #     {"res_blocks": 10, "fc_layers": [64, 32, 16]},
    #     {"res_blocks": 8,  "fc_layers": [32, 16, 8]},
    #     {"res_blocks": 6,  "fc_layers": [32, 16]},
    #     {"res_blocks": 4,  "fc_layers": [16, 8]},
    #     {"res_blocks": 2,  "fc_layers": [16]},
    # ]


    precisions = ["fp32", "fp16"] # "fp64", 

    for i, exp in enumerate(experiments, 1):
        for precision in precisions:
            # Force complete cleanup before each experiment
            torch.cuda.empty_cache()
            gc.collect()
            
            experiment_name = f"res{exp['res_blocks']}_{precision}"
            experiment_dir = os.path.join("checkpoints", experiment_name)
            os.makedirs(experiment_dir, exist_ok=True)

            # Set config
            Config.RESIDUAL_BLOCKS = exp["res_blocks"]
            Config.FULLY_CONNECTED_LAYERS = exp["fc_layers"]
            Config.MODEL_PATH = os.path.join(experiment_dir, "model.pth")
            Config.HISTORY_PATH = os.path.join(experiment_dir, "history.pth")

            if precision == "fp16":
                Config.USE_MIXED_PRECISION = True
                torch.set_default_dtype(torch.float32)  # Needed for AMP stability
            elif precision == "fp32":
                Config.USE_MIXED_PRECISION = False
                torch.set_default_dtype(torch.float32)
            elif precision == "fp64":
                Config.USE_MIXED_PRECISION = False
                torch.set_default_dtype(torch.float64)
            else:
                raise ValueError("Invalid precision. Choose 'fp16', 'fp32', or 'fp64'.")



            # Save config
            with open(os.path.join(experiment_dir, "config.txt"), "w") as f:
                f.write(f"Residual Blocks: {exp['res_blocks']}\n")
                f.write(f"FC Layers: {exp['fc_layers']}\n")
                f.write(f"Precision: {precision.upper()}\n")

            # Print memory stats before starting
            if torch.cuda.is_available():
                print(f"GPU memory before training: {torch.cuda.memory_allocated()/1024**2:.1f}MB / {torch.cuda.get_device_properties(0).total_memory/1024**2:.1f}MB")
            
            print(f"\n\U0001f52c Running {experiment_name} → ResBlocks={exp['res_blocks']}, Precision={precision.upper()}")
            
            train_main()
            
            # Force complete cleanup after each experiment
            torch.cuda.empty_cache()
            gc.collect()
            
            # Print memory stats after cleanup
            if torch.cuda.is_available():
                print(f"GPU memory after cleanup: {torch.cuda.memory_allocated()/1024**2:.1f}MB / {torch.cuda.get_device_properties(0).total_memory/1024**2:.1f}MB")
            
            print(f"Completed {experiment_name}")

if __name__ == '__main__':
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    run_experiments()
