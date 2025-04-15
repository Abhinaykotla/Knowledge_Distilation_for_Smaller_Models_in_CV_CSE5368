import torch
import torch.nn as nn


def train_with_teacher(student, teacher, train_loader, optimizer, criterion, device, alpha=0.5, temperature=4.0, scaler=None):
    student.train()
    teacher.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    kd_loss_fn = nn.KLDivLoss(reduction='batchmean')

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        # Ensure data matches teacher model's dtype for forward pass
        teacher_data = data
        if next(teacher.parameters()).dtype != data.dtype:
            teacher_data = data.to(dtype=next(teacher.parameters()).dtype)
            
        optimizer.zero_grad()

        with torch.no_grad():
            teacher_logits = teacher(teacher_data)

        if scaler:
            with torch.amp.autocast(device_type='cuda'):
                student_logits = student(data)
                ce_loss = criterion(student_logits, target)
                kd_loss = kd_loss_fn(
                    nn.functional.log_softmax(student_logits / temperature, dim=1),
                    nn.functional.softmax(teacher_logits / temperature, dim=1)
                )
                loss = alpha * ce_loss + (1 - alpha) * kd_loss * (temperature ** 2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            student_logits = student(data)
            ce_loss = criterion(student_logits, target)
            kd_loss = kd_loss_fn(
                nn.functional.log_softmax(student_logits / temperature, dim=1),
                nn.functional.softmax(teacher_logits / temperature, dim=1)
            )
            loss = alpha * ce_loss + (1 - alpha) * kd_loss * (temperature ** 2)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = student_logits.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    return train_loss, train_accuracy


def train(model, train_loader, optimizer, criterion, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        # Move data and target to the GPU
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(data)
                loss = criterion(outputs, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    return train_loss, train_accuracy

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            # Move data and target to the GPU
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            test_loss += criterion(outputs, target).item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / total
    return test_loss, test_accuracy

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location='cpu'))