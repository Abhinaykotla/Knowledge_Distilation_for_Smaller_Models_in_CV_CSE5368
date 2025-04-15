import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.blocks import ModularCNN
from data.dataset import IntelImageDataset
from utils.train_utils import train, test, train_with_teacher
from config import Config
import os
import time
from models.teacher_arch import CustomSceneCNN as TeacherCNN


def main():
    config = Config()
    device = torch.device(config.DEVICE)

    # Dataset
    train_dataset = IntelImageDataset(config.TRAIN_DATASET_PATH, transform=config.train_transform)
    test_dataset = IntelImageDataset(config.TEST_DATASET_PATH, transform=config.test_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    # Model
    model = ModularCNN(num_residual_blocks=config.RESIDUAL_BLOCKS,
                       num_fc_layers=len(config.FULLY_CONNECTED_LAYERS),
                       num_classes=config.NUM_CLASSES).to(device)

    # Optimizer and loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if config.USE_MIXED_PRECISION else None

    # History tracking
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': []
    }

    # Timer and best accuracy tracking
    start_time = time.time()
    best_test_acc = 0.0

    # Load and freeze the teacher model
    teacher_model = TeacherCNN().to(device)
    teacher_model.load_state_dict(torch.load(config.TEACHER_MODEL_PATH, map_location=device))
    teacher_model.eval()

    for param in teacher_model.parameters():
        param.requires_grad = False


    for epoch in range(config.MAX_EPOCHS):
        train_loss, train_accuracy = train_with_teacher(
            student=model,
            teacher=teacher_model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            alpha=0.5,
            temperature=4.0,
            scaler=scaler
        )

        test_loss, test_accuracy = test(model, test_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_accuracy)

        print(f'Epoch {epoch+1}/{config.MAX_EPOCHS}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')

        # Save best model
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_model_path = config.MODEL_PATH.replace("model.pth", "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ New best model saved at {best_model_path} with {test_accuracy:.2f}% test accuracy.")

    # Final saving
    end_time = time.time()
    total_time = end_time - start_time

    torch.save(model.state_dict(), config.MODEL_PATH)
    torch.save(history, config.HISTORY_PATH)

    # Log training summary
    summary_path = config.MODEL_PATH.replace("model.pth", "training_summary.txt")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    precision = "FP16 (mixed)" if config.USE_MIXED_PRECISION else "FP32"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    with open(summary_path, "w") as f:
        f.write(f"Model Architecture: ModularCNN\n")
        f.write(f"Residual Blocks: {config.RESIDUAL_BLOCKS}\n")
        f.write(f"Fully Connected Layers: {config.FULLY_CONNECTED_LAYERS}\n")
        f.write(f"Trainable Parameters: {total_params:,}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Training Time: {total_time:.2f} seconds\n")
        f.write(f"Hardware Used: {gpu_name}\n")
        f.write(f"Batch Size: {config.BATCH_SIZE}\n")
        f.write(f"Learning Rate: {config.LEARNING_RATE}\n")
        f.write(f"Epochs: {config.MAX_EPOCHS}\n")
        f.write(f"Best Test Accuracy: {best_test_acc:.2f}%\n")

if __name__ == "__main__":
    main()
