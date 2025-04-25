import os
import torch
import gc
from config import Config
from train import main as train_main
from tqdm import tqdm
from models.cnn_model import CustomSceneCNN4, CustomSceneCNN6, CustomSceneCNN8, CustomSceneCNN10
from models.teacher_arch import CustomSceneCNN
import bitsandbytes as bnb

def run_experiments():
    experiments = [
        {"res_blocks": 4, "fc_layers": [64, 32], "batch_size": 224, "num_workers": 8},
        {"res_blocks": 6, "fc_layers": [128, 64, 16], "batch_size": 224, "num_workers": 8},
        {"res_blocks": 8, "fc_layers": [256, 64, 16], "batch_size": 224, "num_workers": 8},
        {"res_blocks": 10, "fc_layers": [512, 256, 64, 16], "batch_size": 224, "num_workers": 8},
        {"res_blocks": 12, "fc_layers": [1024, 512, 256, 64], "batch_size": 224, "num_workers": 8}
    ]

    precisions = ["fp8", "fp16", "fp32"] #

    # Add a progress bar for all experiments
    total_experiments = len(experiments) * len(precisions)
    experiments_pbar = tqdm(total=total_experiments, desc="Overall Progress", position=0)

    # Map residual blocks to corresponding models
    model_mapping = {
        4: CustomSceneCNN4,
        6: CustomSceneCNN6,
        8: CustomSceneCNN8,
        10: CustomSceneCNN10,
        12: CustomSceneCNN,  # Teacher model
    }

    for i, exp in enumerate(experiments, 1):
        for precision in precisions:
            # Force complete cleanup before each experiment
            torch.cuda.empty_cache()
            gc.collect()

            experiment_name = f"res{exp['res_blocks']}_b{exp['batch_size']}_{precision}"
            experiment_dir = os.path.join("checkpoints", experiment_name)
            os.makedirs(experiment_dir, exist_ok=True)

            # Set config
            Config.RESIDUAL_BLOCKS = exp["res_blocks"]
            Config.FULLY_CONNECTED_LAYERS = exp["fc_layers"]
            Config.BATCH_SIZE = exp["batch_size"]
            Config.NUM_WORKERS = exp["num_workers"]
            Config.MODEL_PATH = os.path.join(experiment_dir, "model.pth")
            Config.HISTORY_PATH = os.path.join(experiment_dir, "history.pth")

            if precision == "fp16":
                Config.USE_MIXED_PRECISION = True
                torch.set_default_dtype(torch.float32)  # Use 16-bit precision
            elif precision == "fp32":
                Config.USE_MIXED_PRECISION = False
                torch.set_default_dtype(torch.float32)  # Use 32-bit precision
            elif precision == "fp8":
                Config.USE_MIXED_PRECISION = False
            else:
                raise ValueError("Invalid precision. Choose 'fp16', 'fp32', or 'fp8'.")

            # Save config
            with open(os.path.join(experiment_dir, "config.txt"), "w") as f:
                f.write(f"Residual Blocks: {exp['res_blocks']}\n")
                f.write(f"FC Layers: {exp['fc_layers']}\n")
                f.write(f"Batch Size: {exp['batch_size']}\n")
                f.write(f"Num Workers: {exp['num_workers']}\n")
                f.write(f"Precision: {precision.upper()}\n")

            # Select the appropriate model based on the number of residual blocks
            if exp["res_blocks"] in model_mapping:
                student_model = model_mapping[exp["res_blocks"]]()
            else:
                raise ValueError(f"No model defined for {exp['res_blocks']} residual blocks.")

            # Print memory stats before starting
            if torch.cuda.is_available():
                print(f"GPU memory before training: {torch.cuda.memory_allocated()/1024**2:.1f}MB / {torch.cuda.get_device_properties(0).total_memory/1024**2:.1f}MB")

            print(f"\n\U0001f52c Running {experiment_name} → ResBlocks={exp['res_blocks']}, BatchSize={exp['batch_size']}, Precision={precision.upper()}")

            # Train the model
            train_main(student_model=student_model, config=Config)

            # Force complete cleanup after each experiment
            torch.cuda.empty_cache()
            gc.collect()

            # Print memory stats after cleanup
            if torch.cuda.is_available():
                print(f"GPU memory after cleanup: {torch.cuda.memory_allocated()/1024**2:.1f}MB / {torch.cuda.get_device_properties(0).total_memory/1024**2:.1f}MB")

            print(f"Completed {experiment_name}")

            # Update the overall progress bar
            experiments_pbar.update(1)

    experiments_pbar.close()

if __name__ == '__main__':
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    run_experiments()
