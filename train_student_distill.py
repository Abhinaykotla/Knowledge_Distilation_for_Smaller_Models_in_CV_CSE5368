import pytorch_lightning as pl
import torch
from student_arch import StudentInpaintNet
from archs.S2_arch import DiffIRS2
from torch.utils.data import DataLoader
from saicinpainting.evaluation.data import OurInpaintingDataset as InpaintingDataset
import yaml

def load_yaml(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_yaml('config/student_config.yaml')

train_dataset = InpaintingDataset(
    datadir=config['data']['dataset']['root'],
    img_flist=config['data']['dataset']['train_source'],
    mask_flist=config['data']['dataset']['train_mask'],
    training=True
)

val_dataset = InpaintingDataset(
    datadir=config['data']['dataset']['root'],
    img_flist=config['data']['dataset']['val_source'],
    mask_flist=config['data']['dataset']['val_mask'],
    training=False
)

train_loader = DataLoader(train_dataset, batch_size=config['data']['dataset']['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['data']['dataset']['batch_size'], shuffle=False)

student_model = StudentInpaintNet(
    in_channels=config['model']['in_channels'],
    out_channels=config['model']['out_channels'],
    base_channels=config['model']['base_channels'],
    num_layers=config['model']['num_layers']
)

teacher_model = DiffIRS2()
teacher_checkpoint = torch.load('checkpoints/DiffIRS2_teacher.ckpt', map_location='cpu')
teacher_model.load_state_dict(teacher_checkpoint['state_dict'], strict=False)
teacher_model.eval()

for p in teacher_model.parameters():
    p.requires_grad = False

trainer = pl.Trainer(
    gpus=config['trainer']['gpus'],
    max_epochs=config['trainer']['max_epochs'],
    precision=config['trainer']['precision'],
    log_every_n_steps=config['trainer']['log_every_n_steps'],
    val_check_interval=config['trainer']['val_check_interval'],
    default_root_dir='student_checkpoints'
)

trainer.fit(student_model, train_loader, val_loader)
