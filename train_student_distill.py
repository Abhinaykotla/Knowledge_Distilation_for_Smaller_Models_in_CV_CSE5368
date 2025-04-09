import os
import yaml
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from student_arch import StudentInpaintNet
from archs.S2_arch import DiffIRS2
from saicinpainting.evaluation.data import OurInpaintingDataset


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def create_dataset_folder(img_flist_path, mask_flist_path, target_dir):
    img_dir = os.path.join(target_dir, 'img')
    mask_dir = os.path.join(target_dir, 'mask')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    with open(img_flist_path, 'r') as f:
        img_paths = f.read().splitlines()
    with open(mask_flist_path, 'r') as f:
        mask_paths = f.read().splitlines()

    for path in img_paths:
        fname = os.path.basename(path)
        dst = os.path.join(img_dir, fname)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(path), dst)

    for path in mask_paths:
        fname = os.path.basename(path)
        dst = os.path.join(mask_dir, fname)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(path), dst)

    return target_dir


# Load config
config = load_yaml('config/student_config.yaml')

root = config['data']['dataset']['root']
train_source = os.path.join(root, 'flists', config['data']['dataset']['train_source'])
train_mask = os.path.join(root, 'flists', config['data']['dataset']['train_mask'])
val_source = os.path.join(root, 'flists', config['data']['dataset']['val_source'])
val_mask = os.path.join(root, 'flists', config['data']['dataset']['val_mask'])

train_dir = create_dataset_folder(train_source, train_mask, os.path.join(root, 'train'))
val_dir = create_dataset_folder(val_source, val_mask, os.path.join(root, 'val'))

# Dataset and DataLoaders
train_dataset = OurInpaintingDataset(train_dir)
val_dataset = OurInpaintingDataset(val_dir)

train_loader = DataLoader(train_dataset, batch_size=config['data']['dataset']['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['data']['dataset']['batch_size'], shuffle=False)

# Student Model
student_model = StudentInpaintNet(
    in_channels=config['model']['in_channels'],
    out_channels=config['model']['out_channels'],
    base_channels=config['model']['base_channels'],
    num_layers=config['model']['num_layers']
)

# Load Teacher
teacher_model = DiffIRS2()
teacher_ckpt = torch.load('checkpoints/DiffIRS2_teacher.ckpt', map_location='cpu')
teacher_model.load_state_dict(teacher_ckpt['state_dict'], strict=False)
teacher_model.eval()
for p in teacher_model.parameters():
    p.requires_grad = False

# Assign teacher
student_model.teacher = teacher_model

# Trainer
trainer = pl.Trainer(
    gpus=config['trainer']['gpus'],
    max_epochs=config['trainer']['max_epochs'],
    precision=config['trainer']['precision'],
    log_every_n_steps=config['trainer']['log_every_n_steps'],
    val_check_interval=config['trainer']['val_check_interval'],
    default_root_dir='student_checkpoints'
)

trainer.fit(student_model, train_loader, val_loader)
