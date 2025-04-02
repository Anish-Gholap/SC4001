"""
Training and evaluation utilities for emotion classification models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for sequences_batch, labels_batch in progress_bar:
        sequences_batch = sequences_batch.to(device)
        labels_batch = labels_batch.to(device)

        logits = model(sequences_batch)
        loss = criterion(logits, labels_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * sequences_batch.size(0)
        total_samples += labels_batch.size(0)
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels_batch).item()

        progress_bar.set_postfix(
            loss=loss.item(), 
            acc=correct_predictions/total_samples if total_samples > 0 else 0
        )

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    """Evaluate the model and return loss, accuracy, labels, and predictions."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels = []
    all_preds = []
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for sequences_batch, labels_batch in progress_bar:
            sequences_batch = sequences_batch.to(device)
            labels_batch = labels_batch.to(device)

            logits = model(sequences_batch)
            loss = criterion(logits, labels_batch)

            running_loss += loss.item() * sequences_batch.size(0)
            total_samples += labels_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels_batch).item()

            all_labels.extend(labels_batch.cpu().numpy())  # Collect labels (move to CPU)
            all_preds.extend(preds.cpu().numpy())         # Collect predictions (move to CPU)

            progress_bar.set_postfix(
                loss=loss.item(), 
                acc=correct_predictions/total_samples if total_samples > 0 else 0
            )

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0
    return epoch_loss, epoch_acc, all_labels, all_preds

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """Plots a confusion matrix using seaborn."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                epochs, model_save_path, early_stopping_patience=5):
    """
    Train a model with validation-based early stopping and save the best model.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda, mps, or cpu)
        epochs: Maximum number of epochs to train for
        model_save_path: Where to save the best model
        early_stopping_patience: Stop if validation loss doesn't improve for this many epochs
    
    Returns:
        Dictionary with training history
    """
    # Initialize variables
    best_val_acc = -1.0
    best_epoch = 0
    patience_counter = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    print("\n--- Starting Training ---")
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print(f"Epoch {epoch+1} Training   -> Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"Epoch {epoch+1} Validation -> Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # Save the model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"*** Best model saved to {model_save_path} (Val Acc: {best_val_acc:.4f}) ***")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {early_stopping_patience} epochs)")
                break

    print("\n--- Training Complete ---")
    print(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch+1})")
    
    # Return training history
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc
    }

def plot_training_history(history, title_prefix=""):
    """Plot training and validation losses and accuracies."""
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    epochs = range(1, len(history['train_losses']) + 1)
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
    ax1.set_title(f'{title_prefix}Losses')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(epochs, history['train_accs'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_accs'], 'r-', label='Validation Accuracy')
    ax2.set_title(f'{title_prefix}Accuracies')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def test_model(model, test_loader, criterion, device, model_save_path, label_encoder):
    """
    Evaluate a trained model on the test set.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to use
        model_save_path: Path to load the best model weights from
        label_encoder: Fitted LabelEncoder to convert numeric labels to class names
    
    Returns:
        Dictionary with test results
    """
    if test_loader is None:
        print("Test loader not available. Skipping final test evaluation.")
        return None
    
    print("\n--- Evaluating Best Model on Test Set ---")
    
    # Load the best model weights
    try:
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"Loaded best model weights from {model_save_path}")
        
        # Evaluate on the test set
        test_loss, test_acc, test_labels, test_preds = evaluate(model, test_loader, criterion, device)
        print(f"\nTest Set Performance -> Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        
        # Generate Classification Report
        print("\nClassification Report:")
        report = classification_report(test_labels, test_preds, target_names=label_encoder.classes_, digits=4)
        print(report)
        
        # Generate and Plot Confusion Matrix
        print("Confusion Matrix:")
        cm = confusion_matrix(test_labels, test_preds)
        plot_confusion_matrix(cm, class_names=label_encoder.classes_, title='Test Set Confusion Matrix')
        
        # Return test results
        return {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_labels': test_labels,
            'test_preds': test_preds,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    except FileNotFoundError:
        print(f"Error: Could not find the saved model at {model_save_path}. Skipping final test evaluation.")
        return None
    except Exception as e:
        print(f"An error occurred during final test evaluation: {e}")
        return None