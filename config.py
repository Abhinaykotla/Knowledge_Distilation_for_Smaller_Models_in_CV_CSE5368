import os
import torch
from torchvision import transforms

class Config:
    # Dataset configuration
    DATASET_PATH = os.path.join('data', 'intel-image-classification')
    TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, 'seg_train', 'seg_train')
    TEST_DATASET_PATH = os.path.join(DATASET_PATH, 'seg_test', 'seg_test')
    TEACHER_MODEL_PATH = './checkpoints/teacher_12layers_model.pth'


    # Hyperparameters
    BATCH_SIZE = 12
    NUM_WORKERS = 4
    LEARNING_RATE = 0.001
    MAX_EPOCHS = 15
    PATIENCE =3
    USE_MIXED_PRECISION = False  # Default
    PRECISION = "fp32"  # Default precision

    # Model configuration
    NUM_CLASSES = 6
    RESIDUAL_BLOCKS = 2  # Default number of residual blocks
    FULLY_CONNECTED_LAYERS = [512]  # List of sizes for fully connected layers

    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths for saving model and history
    MODEL_PATH = './test/intel_scene_cnn_model.pth'
    HISTORY_PATH = './test/intel_scene_cnn_history.pth'

    # Dataset transformations
    train_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])