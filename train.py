import torch
import torch.nn as nn
import bitsandbytes as bnb
from torch.utils.data import DataLoader
from models.blocks import ModularCNN
from data.dataset import IntelImageDataset
from utils.train_utils import train_with_teacher, test
from config import Config
from models.teacher_arch import CustomSceneCNN as TeacherCNN
import os
import time
from tqdm import tqdm

def main(student_model=None, config=None):
    if config is None:
        config = Config()
    device = torch.device(config.DEVICE)

    print(f"Using device: {device}")

    # Dataset
    train_dataset = IntelImageDataset(config.TRAIN_DATASET_PATH, transform=config.train_transform)
    test_dataset = IntelImageDataset(config.TEST_DATASET_PATH, transform=config.test_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    # Use the provided student model or raise an error if not provided
    if student_model is None:
        raise ValueError("A student model must be provided to train_main.")
    student_model = student_model.to(device)

    # Quantize the student model to 8-bit precision if using fp8
    if str(torch.get_default_dtype()) == "torch.float8":
        student_model = bnb.nn.quantize(student_model, dtype=torch.float8)

    print(student_model)

    # Optimizer and loss
    criterion = nn.CrossEntropyLoss()

    # Use bitsandbytes 8-bit optimizer for fp8 precision
    if str(torch.get_default_dtype()) == "torch.float8":
        optimizer = bnb.optim.Adam8bit(student_model.parameters(), lr=config.LEARNING_RATE)
        scaler = None  # Disable GradScaler for 8-bit optimizers
    else:
        optimizer = torch.optim.Adam(student_model.parameters(), lr=config.LEARNING_RATE)
        scaler = torch.amp.GradScaler() if config.USE_MIXED_PRECISION else None

    # Load teacher model
    teacher_model = TeacherCNN().to(device)
    teacher_model.load_state_dict(torch.load(config.TEACHER_MODEL_PATH, map_location=device))
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Quantize the teacher model to 8-bit precision if using fp8
    if str(torch.get_default_dtype()) == "torch.float8":
        teacher_model = bnb.nn.quantize(teacher_model, dtype=torch.float8)

    # Match teacher model dtype to current precision
    teacher_model = teacher_model.to(dtype=torch.get_default_dtype())

    # History tracking
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': []
    }

    start_time = time.time()
    best_test_acc = 0.0
    
    # Early stopping parameters
    patience = config.PATIENCE
    patience_counter = 0
    early_stopped = False

    # Create a progress bar for epochs
    epoch_pbar = tqdm(range(config.MAX_EPOCHS), desc="Training Progress", unit="epoch")
    
    for epoch in epoch_pbar:
        # Start time for this epoch
        epoch_start = time.time()
        
        train_loss, train_accuracy = train_with_teacher(
            student=student_model,
            teacher=teacher_model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            alpha=0.5,
            temperature=4.0,
            scaler=scaler
        )

        test_loss, test_accuracy = test(student_model, test_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_accuracy)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        eta = epoch_time * (config.MAX_EPOCHS - epoch - 1)
        
        # Update progress bar with detailed information
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}', 
            'train_acc': f'{train_accuracy:.2f}%',
            'test_acc': f'{test_accuracy:.2f}%',
            'time': f'{epoch_time:.2f}s',
            'eta': f'{eta/60:.2f}min',
            'patience': f'{patience_counter}/{patience}'
        })

        # Check if model performance improved
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_model_path = config.MODEL_PATH.replace("model.pth", "best_model.pth")
            torch.save(student_model.state_dict(), best_model_path)
            epoch_pbar.write(f"\u2705 New best model saved at {best_model_path} with {test_accuracy:.2f}% test accuracy.")
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1  # Increment patience counter
            
        # Check for early stopping
        if patience_counter >= patience:
            epoch_pbar.write(f"\u26A0 Early stopping triggered after {epoch+1} epochs without improvement.")
            early_stopped = True
            break

    # Final save
    end_time = time.time()
    total_time = end_time - start_time
    torch.save(student_model.state_dict(), config.MODEL_PATH)
    torch.save(history, config.HISTORY_PATH)
    
    print(f"Training completed in {total_time/60:.2f} minutes")
    if early_stopped:
        print(f"Early stopping activated. Best accuracy: {best_test_acc:.2f}%")

    # Training summary
    summary_path = config.MODEL_PATH.replace("model.pth", "training_summary.txt")
    total_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    if config.USE_MIXED_PRECISION:
        precision = "FP16 (mixed)"
    else:
        precision = str(torch.get_default_dtype()).upper()

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    with open(summary_path, "w") as f:
        f.write(f"Model Architecture: ModularCNN\n")
        f.write(f"Residual Blocks: {config.RESIDUAL_BLOCKS}\n")
        f.write(f"Fully Connected Layers: {config.FULLY_CONNECTED_LAYERS}\n")
        f.write(f"Trainable Parameters: {total_params:,}\n")
        f.write(f"Training Time: {total_time:.2f} seconds\n")
        f.write(f"Hardware Used: {gpu_name}\n")
        f.write(f"Batch Size: {config.BATCH_SIZE}\n")
        f.write(f"Learning Rate: {config.LEARNING_RATE}\n")
        f.write(f"Epochs: {config.MAX_EPOCHS}\n")
        f.write(f"Best Test Accuracy: {best_test_acc:.2f}%\n")
        f.write(f"Early Stopping: {'Yes' if early_stopped else 'No'}\n")
        if early_stopped:
            f.write(f"Stopped at epoch: {epoch+1}/{config.MAX_EPOCHS}\n")

if __name__ == "__main__":
    from models.cnn_model import CustomSceneCNN4  # Import your model
    student_model = CustomSceneCNN4()  # Instantiate the model
    main(student_model=student_model)