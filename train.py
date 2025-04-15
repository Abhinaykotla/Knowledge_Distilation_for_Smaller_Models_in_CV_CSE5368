import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.blocks import ModularCNN
from data.dataset import IntelImageDataset
from utils.train_utils import train, test
from config import Config

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

    # Optimizer, Loss, Scaler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler() if config.USE_MIXED_PRECISION else None

    # For tracking history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': []
    }

    for epoch in range(config.MAX_EPOCHS):
        train_loss, train_accuracy = train(
            model, train_loader, optimizer, criterion, device, scaler)
        test_loss, test_accuracy, _, _ = test(
            model, test_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_accuracy)

        print(f'Epoch {epoch+1}/{config.MAX_EPOCHS}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')

    torch.save(model.state_dict(), config.MODEL_PATH)
    torch.save(history, config.HISTORY_PATH)

if __name__ == "__main__":
    main()
