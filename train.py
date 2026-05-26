"""Training loop for Eviformer models."""

import copy
import time
import torch
from utils import get_device, one_hot_embedding
from models.losses import relu_evidence


def train_model(model, dataloaders, num_classes, criterion, optimizer,
                scheduler=None, num_epochs=100, device=None, uncertainty=False):
    """
    Train a model with optional evidential deep learning uncertainty.

    Args:
        model: PyTorch model to train.
        dataloaders: Dict with 'train' and 'val' DataLoaders.
        num_classes: Number of output classes.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler (optional).
        num_epochs: Number of training epochs.
        device: Computation device.
        uncertainty: Whether to use evidential deep learning.

    Returns:
        Tuple of (best_model, metrics) where metrics = (losses, accuracy).
    """
    since = time.time()
    if device is None:
        device = get_device()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    losses = {"loss": [], "phase": [], "epoch": []}
    accuracy = {"accuracy": [], "phase": [], "epoch": []}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                print("Training...")
                model.train()
            else:
                print("Validating...")
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    if uncertainty:
                        y = one_hot_embedding(labels, num_classes)
                        y = y.to(device)
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, y.float(), epoch, num_classes, 10, device)
                    else:
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if scheduler is not None and phase == "train":
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            losses["loss"].append(epoch_loss)
            losses["phase"].append(phase)
            losses["epoch"].append(epoch)
            accuracy["accuracy"].append(epoch_acc.item())
            accuracy["epoch"].append(epoch)
            accuracy["phase"].append(phase)

            print(f"{phase.capitalize()} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    metrics = (losses, accuracy)
    return model, metrics
