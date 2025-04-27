import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import json
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from models.cnn_model import CustomSceneCNN8
from models.teacher_arch import CustomSceneCNN as TeacherCNN
from data.dataset import IntelImageDataset
from config import Config

# Paths
student_ckpt_path = './checkpoints/res8_b224_fp16/best_model.pth'
teacher_ckpt_path = './checkpoints/teacher_12layers_model.pth'
eval_save_dir = './checkpoints/evaluation_results/'
os.makedirs(eval_save_dir, exist_ok=True)

def evaluate(model, dataloader, device, use_half=False):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            if use_half:
                images = images.half()  # Convert input to half-precision
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    return accuracy, all_labels, all_preds

def plot_confusion_matrix(cm, labels, title, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    # Load test dataset
    test_dataset = IntelImageDataset(Config.TEST_DATASET_PATH, transform=Config.test_transform)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    # Load student model
    student_model = CustomSceneCNN8().to(device)
    student_model.load_state_dict(torch.load(student_ckpt_path, map_location=device))
    student_model = student_model.half()  # since it's FP16

    # Load teacher model
    teacher_model = TeacherCNN().to(device)
    teacher_model.load_state_dict(torch.load(teacher_ckpt_path, map_location=device))

    # Evaluate both models
    student_acc, student_labels, student_preds = evaluate(student_model, test_loader, device, use_half=True)
    teacher_acc, teacher_labels, teacher_preds = evaluate(teacher_model, test_loader, device)

    print(f"\nStudent Model Accuracy: {student_acc:.2f}%")
    print(f"Teacher Model Accuracy: {teacher_acc:.2f}%")

    # Generate confusion matrices
    student_cm = confusion_matrix(student_labels, student_preds)
    teacher_cm = confusion_matrix(teacher_labels, teacher_preds)

    # Plot and save confusion matrices
    class_labels = test_dataset.classes  # Assuming the dataset has a `classes` attribute
    plot_confusion_matrix(student_cm, class_labels, "Student Model Confusion Matrix", os.path.join(eval_save_dir, 'student_cm.png'))
    plot_confusion_matrix(teacher_cm, class_labels, "Teacher Model Confusion Matrix", os.path.join(eval_save_dir, 'teacher_cm.png'))

    # Save evaluation results
    results = {
        "student_accuracy": student_acc,
        "teacher_accuracy": teacher_acc,
        "classification_report_student": classification_report(student_labels, student_preds, output_dict=True),
        "classification_report_teacher": classification_report(teacher_labels, teacher_preds, output_dict=True),
        "student_confusion_matrix": student_cm.tolist(),
        "teacher_confusion_matrix": teacher_cm.tolist()
    }

    with open(os.path.join(eval_save_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation results saved to {eval_save_dir}/evaluation_results.json")
    print(f"Confusion matrices saved as images in {eval_save_dir}")

if __name__ == "__main__":
    main()