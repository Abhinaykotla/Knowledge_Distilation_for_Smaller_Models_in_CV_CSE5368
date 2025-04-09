import yaml
import subprocess
import os
import torch

CONFIG_PATH = 'config/student_config.yaml'
CHECKPOINT_DIR = 'student_checkpoints'
LAYERS_TO_TEST = [4, 5, 6, 7, 8]  # Number of layers to test


def update_config(num_layers):
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    config['model']['num_layers'] = num_layers
    config['trainer']['default_root_dir'] = os.path.join(CHECKPOINT_DIR, f'layers_{num_layers}')

    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f)

    print(f"✅ Updated config: num_layers={num_layers}")


def train_model():
    print("🚀 Starting training...")
    subprocess.run(['python', 'train_student_distill.py'], check=True)


def get_latest_checkpoint(folder):
    subdir = os.path.join(folder, 'version_0', 'checkpoints')
    if not os.path.exists(subdir):
        return None
    files = [f for f in os.listdir(subdir) if f.endswith('.ckpt')]
    return os.path.join(subdir, sorted(files)[-1]) if files else None


def get_model_size(path):
    size_mb = os.path.getsize(path) / (1024 * 1024)
    return round(size_mb, 2)


if __name__ == "__main__":
    results = []

    for num_layers in LAYERS_TO_TEST:
        update_config(num_layers)
        try:
            train_model()
            ckpt_path = get_latest_checkpoint(os.path.join(CHECKPOINT_DIR, f'layers_{num_layers}'))
            size = get_model_size(ckpt_path) if ckpt_path else "N/A"
            results.append((num_layers, size, ckpt_path))
        except subprocess.CalledProcessError:
            results.append((num_layers, "Failed", "N/A"))

    print("\n📊 Experiment Summary:")
    for n, s, p in results:
        print(f"Layers: {n} | Model Size: {s} MB | Checkpoint: {p}")
