import os
import torch
from torchvision import transforms

class Config:
    # Dataset configuration
    DATASET_PATH = os.path.join('data', 'intel-image-classification')
    TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, 'seg_train', 'seg_train')
    TEST_DATASET_PATH = os.path.join(DATASET_PATH, 'seg_test', 'seg_test')

    # Hyperparameters
    BATCH_SIZE = 128
    NUM_WORKERS = 12
    LEARNING_RATE = 0.001
    MAX_EPOCHS = 2
    PATIENCE = 3
    USE_MIXED_PRECISION = True
    
    # Model configuration
    NUM_CLASSES = 6
    RESIDUAL_BLOCKS = 8  # Number of residual blocks
    FULLY_CONNECTED_LAYERS = [1024, 256, 64]  # List of sizes for fully connected layers

    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths for saving model and history
    MODEL_PATH = './intel_scene_cnn_model.pth'
    HISTORY_PATH = './intel_scene_cnn_history.pth'

    # Dataset transformations
    train_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])