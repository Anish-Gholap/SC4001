"""
Script to train a CNN-LSTM hybrid model for emotion classification.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

# Add the parent directory to the path so we can import from our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
import config
from preprocessing import prepare_data_for_training
from dataset import get_data_loaders
from models import CNNLSTMClassifier
from training import train_model, test_model, plot_training_history
from utils import set_seed, get_device, save_model_info

def main():
    """Train and evaluate a CNN-LSTM hybrid model for emotion classification."""
    print("="*80)
    print("TRAINING CNN-LSTM HYBRID MODEL FOR EMOTION CLASSIFICATION")
    print("="*80)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Determine device
    device = get_device()
    
    # Prepare data
    print("\nPreparing data...")
    data = prepare_data_for_training(config)
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = get_data_loaders(
        train_ids=data['train_ids'],
        y_train_int=data['y_train_int'],
        val_ids=data['val_ids'],
        y_val_int=data['y_val_int'],
        test_ids=data['test_ids'],
        y_test_int=data['y_test_int'],
        batch_size=config.BATCH_SIZE,
        pad_index=data['pad_index']
    )
    
    # Initialize CNN-LSTM model
    print("\nInitializing CNN-LSTM hybrid model...")
    model = CNNLSTMClassifier(
        vocab_size=data['vocab_size'],
        embed_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_classes=data['num_classes'],
        n_filters=config.CNN_N_FILTERS,
        filter_sizes=config.CNN_FILTER_SIZES,
        lstm_layers=config.CNN_LSTM_LAYERS,
        dropout=config.DROPOUT,
        pad_idx=data['pad_index']
    ).to(device)
    
    # Print model summary
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Train the model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=config.EPOCHS,
        model_save_path=config.CNN_LSTM_SAVE_PATH,
        early_stopping_patience=5
    )
    
    # Plot training history
    plot_training_history(history, title_prefix="CNN-LSTM Model: ")
    
    # Save model info for later inference
    save_model_info(
        model=model,
        vocab_size=data['vocab_size'],
        word_to_idx=data['word_to_idx'],
        label_encoder=data['label_encoder'],
        config=config,
        path="cnn_lstm_model_info.pkl"
    )
    
    # Test the final model
    if test_loader:
        test_results = test_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            model_save_path=config.CNN_LSTM_SAVE_PATH,
            label_encoder=data['label_encoder']
        )
    
if __name__ == "__main__":
    main()