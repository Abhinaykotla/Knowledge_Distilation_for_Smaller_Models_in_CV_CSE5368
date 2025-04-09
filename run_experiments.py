import os
import yaml
import subprocess
import torch


CONFIG_PATH = 'config/student_config.yaml'
CHECKPOINT_BASE = 'student_checkpoints'
TRAIN_SCRIPT = 'train_student_distill.py'
LAYERS_TO_TEST = [4, 5, 6]


def update_config(num_layers):
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    config['model']['num_layers'] = num_layers
    config['trainer']['default_root_dir'] = os.path.join(CHECKPOINT_BASE, f'layers_{num_layers}')
    
    # Optional: increase batch size if your GPU can handle it
    config['data']['dataset']['batch_size'] = 64
    config['data']['dataset']['num_workers'] = 4

    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f)

    print(f"✅ Config updated: num_layers={num_layers}")


def train_model():
    print("🚀 Launching training subprocess...")
    result = subprocess.run(['python', TRAIN_SCRIPT], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Training failed!")
        print(result.stderr)
        return False
    return True


def get_latest_checkpoint(log_dir):
    ckpt_dir = os.path.join(log_dir, 'version_0', 'checkpoints')
    if not os.path.exists(ckpt_dir):
        return None
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')])
    return os.path.join(ckpt_dir, ckpts[-1]) if ckpts else None


def get_model_size(ckpt_path):
    if not ckpt_path or not os.path.exists(ckpt_path):
        return "N/A"
    size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    return round(size_mb, 2)


if __name__ == "__main__":
    summary = []

    for layers in LAYERS_TO_TEST:
        print(f"\n🔁 Running experiment: num_layers = {layers}")
        update_config(layers)

        success = train_model()
        log_dir = os.path.join(CHECKPOINT_BASE, f'layers_{layers}')
        ckpt_path = get_latest_checkpoint(log_dir)
        size = get_model_size(ckpt_path)

        summary.append({
            "layers": layers,
            "success": success,
            "size_mb": size,
            "checkpoint": ckpt_path if ckpt_path else "N/A"
        })

    print("\n📊 Experiment Results:")
    for result in summary:
        status = "✅" if result["success"] else "❌"
        print(f"{status} Layers: {result['layers']} | Size: {result['size_mb']} MB | Checkpoint: {result['checkpoint']}")
