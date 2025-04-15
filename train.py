import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.cnn_model import CustomSceneCNN
from data.dataset import IntelImageDataset
from utils.train_utils import train, test
from config import Config

def main():
    # Load configuration settings
    config = Config()

    # Set device
    device = torch.device(config.DEVICE)

    # Initialize dataset and data loaders
    train_dataset = IntelImageDataset(config.TRAIN_DATASET_PATH, transform=config.train_transform)
    test_dataset = IntelImageDataset(config.TEST_DATASET_PATH, transform=config.test_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    # Initialize model
    model = CustomSceneCNN(num_residual_blocks=config.RESIDUAL_BLOCKS, num_fc_layers=len(config.FULLY_CONNECTED_LAYERS)).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Training loop
    for epoch in range(config.MAX_EPOCHS):
        train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_accuracy = test(model, test_loader, criterion, device)

        print(f'Epoch {epoch+1}/{config.MAX_EPOCHS}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}%')

    # Save the trained model
    torch.save(model.state_dict(), config.MODEL_PATH)

if __name__ == "__main__":
    main()