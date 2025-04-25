import os
import torch
import matplotlib.pyplot as plt

# Directory containing experiment folders
checkpoints_dir = "checkpoints"

# Containers for histories
all_histories = {}
experiment_names = []

# Load history.pth from each experiment folder
for exp_folder in sorted(os.listdir(checkpoints_dir)):
    exp_path = os.path.join(checkpoints_dir, exp_folder)
    history_path = os.path.join(exp_path, "history.pth")
    
    if os.path.isfile(history_path):
        history = torch.load(history_path, map_location="cpu")
        all_histories[exp_folder] = history
        experiment_names.append(exp_folder)

# Plotting accuracy and loss for all experiments
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot Test Accuracy
for name in experiment_names:
    axes[0].plot(all_histories[name]['test_accuracy'], label=name)
axes[0].set_title("Test Accuracy Over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy (%)")
axes[0].legend()

# Plot Test Loss
for name in experiment_names:
    axes[1].plot(all_histories[name]['test_loss'], label=name)
axes[1].set_title("Test Loss Over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.show()
