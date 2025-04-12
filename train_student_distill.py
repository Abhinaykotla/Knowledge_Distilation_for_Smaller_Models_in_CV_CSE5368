import os
import yaml
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from student_arch import StudentInpaintNet
from archs.S2_arch import DiffIRS2
import cv2
import numpy as np
from torchinfo import summary


class StudentInpaintingDataset(Dataset):
    def __init__(self, img_flist, mask_flist, img_dir, mask_dir, image_size=(256, 256)):
        with open(img_flist, 'r') as f:
            self.img_files = [line.strip() for line in f if line.strip()]
        with open(mask_flist, 'r') as f:
            self.mask_files = [line.strip() for line in f if line.strip()]
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        assert len(self.img_files) == len(self.mask_files), "Mismatch in image and mask count"

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Missing image or mask at {img_path} / {mask_path}")
        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size)
        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        masked_input = image * (1 - mask)
        input_with_mask = torch.cat([masked_input, mask], dim=0)
        return input_with_mask, mask, image


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    config = load_yaml('config/student_config.yaml')
    root = config['data']['dataset']['root']

    train_dataset = StudentInpaintingDataset(
        img_flist=os.path.join(root, 'flists', config['data']['dataset']['train_source']),
        mask_flist=os.path.join(root, 'flists', config['data']['dataset']['train_mask']),
        img_dir=os.path.join(root, 'val_source_256'),
        mask_dir=os.path.join(root, 'val_masks_thick')
    )

    val_dataset = StudentInpaintingDataset(
        img_flist=os.path.join(root, 'flists', config['data']['dataset']['val_source']),
        mask_flist=os.path.join(root, 'flists', config['data']['dataset']['val_mask']),
        img_dir=os.path.join(root, 'val_source_256'),
        mask_dir=os.path.join(root, 'val_masks_thick')
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['dataset']['batch_size'],
        shuffle=True,
        num_workers=config['data']['dataset'].get('num_workers', 0),
        pin_memory=True,
        persistent_workers=True if config['data']['dataset'].get('num_workers', 0) > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['dataset']['batch_size'],
        shuffle=False,
        num_workers=config['data']['dataset'].get('num_workers', 0),
        pin_memory=True,
        persistent_workers=True if config['data']['dataset'].get('num_workers', 0) > 0 else False
    )

    student_model = StudentInpaintNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_channels=config['model']['base_channels'],
        num_layers=config['model']['num_layers']
    )

    teacher_model = DiffIRS2()
    teacher_ckpt = torch.load('checkpoints/DiffIRS2_teacher.ckpt', map_location='cpu', weights_only=False)
    teacher_model.load_state_dict(teacher_ckpt['state_dict'], strict=False)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    # Attach externally
    student_model.set_teacher(teacher_model)

    print(f"Number of layers in student model: {config['model']['num_layers']}")
    summary(student_model, input_size=(config['data']['dataset']['batch_size'], config['model']['in_channels'], 256, 256))

    # Param counts (only encoder + decoder)
    student_total_params = sum(p.numel() for name, p in student_model.named_parameters() if 'teacher' not in name)
    student_trainable_params = sum(p.numel() for name, p in student_model.named_parameters() if p.requires_grad and 'teacher' not in name)
    teacher_total_params = sum(p.numel() for p in teacher_model.parameters())

    print(f"📦 Total Student Params (excluding teacher): {student_total_params:,}")
    print(f"✅ Trainable Student Params: {student_trainable_params:,}")
    print(f"🧠 Teacher Params (frozen): {teacher_total_params:,}")

    trainer = pl.Trainer(
        gpus=config['trainer']['gpus'],
        max_epochs=config['trainer']['max_epochs'],
        precision=config['trainer']['precision'],
        log_every_n_steps=config['trainer']['log_every_n_steps'],
        val_check_interval=config['trainer']['val_check_interval'],
        default_root_dir=config['trainer']['default_root_dir']
    )

    trainer.fit(student_model, train_loader, val_loader)


if __name__ == "__main__":
    main()
