"""
Script to compare different model architectures for emotion classification.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# Add the parent directory to the path so we can import from our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
import config
from preprocessing import prepare_data_for_training, load_vocab_and_labels
from dataset import get_data_loaders, TextDataset
from models import RNNLSTMClassifier, SimpleRNNClassifier,CNNLSTMClassifier
from training import evaluate, plot_confusion_matrix
from utils import set_seed, get_device

def load_model(model_class, model_path, data, device, **kwargs):
    """
    Load a trained model from disk.
    
    Args:
        model_class: Class of the model to load
        model_path: Path to the saved model weights
        data: Data dictionary containing vocab_size, num_classes, pad_index
        device: Device to load the model on
        **kwargs: Additional arguments for the model constructor
        
    Returns:
        Loaded model
    """
    # Create a new model instance
    model = model_class(
        vocab_size=data['vocab_size'],
        num_classes=data['num_classes'],
        pad_idx=data['pad_index'],
        **kwargs
    ).to(device)
    
    # Load the weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

def evaluate_models(models, test_loader, criterion, device, label_encoder):
    """
    Evaluate multiple models on the test set and compare results.
    
    Args:
        models: Dictionary of model_name: model pairs
        test_loader: DataLoader for the test set
        criterion: Loss function
        device: Device to run evaluation on
        label_encoder: LabelEncoder to convert numeric labels to class names
        
    Returns:
        Dictionary with evaluation results for each model
    """
    results = {}
    
    for name, model in models.items():
        print(f"\n--- Evaluating {name} ---")
        test_loss, test_acc, test_labels, test_preds = evaluate(
            model, test_loader, criterion, device
        )
        
        results[name] = {
            'loss': test_loss,
            'accuracy': test_acc,
            'labels': test_labels,
            'predictions': test_preds,
            'report': classification_report(
                test_labels, test_preds, 
                target_names=label_encoder.classes_, 
                output_dict=True
            ),
            'confusion_matrix': confusion_matrix(test_labels, test_preds)
        }
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            test_labels, test_preds, 
            target_names=label_encoder.classes_
        ))
        
        # Plot confusion matrix for this model
        plot_confusion_matrix(
            results[name]['confusion_matrix'],
            class_names=label_encoder.classes_,
            title=f'{name} Confusion Matrix'
        )
    
    return results

def plot_accuracy_comparison(results):
    """Plot accuracy comparison of different models."""
    # Extract accuracies
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['blue', 'green'])
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1.0)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01,
            f'{height:.4f}',
            ha='center', va='bottom'
        )
    
    plt.tight_layout()
    plt.show()

def plot_class_f1_comparison(results, label_encoder):
    """
    Plot F1 score comparison across different models for each class.
    
    Args:
        results: Dictionary with evaluation results for each model
        label_encoder: LabelEncoder to get class names
    """
    # Extract class names
    class_names = label_encoder.classes_
    
    # Extract F1 scores for each class and model
    f1_scores = {}
    for model_name, model_results in results.items():
        f1_scores[model_name] = []
        for class_name in class_names:
            f1_scores[model_name].append(
                model_results['report'][class_name]['f1-score']
            )
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame(f1_scores, index=class_names)
    
    # Plot
    plt.figure(figsize=(12, 8))
    df.plot(kind='bar', ax=plt.gca())
    plt.title('F1 Score Comparison by Class')
    plt.xlabel('Emotion Class')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1.0)
    plt.legend(title='Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    """Compare different model architectures for emotion classification."""
    print("="*80)
    print("COMPARING EMOTION CLASSIFICATION MODELS")
    print("="*80)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Determine device
    device = get_device()
    
    # Prepare data - either load prepared data or reprocess it
    if os.path.exists(config.VOCAB_SAVE_PATH) and os.path.exists(config.LABEL_ENCODER_SAVE_PATH):
        print("\nLoading preprocessed vocabulary and labels...")
        word_to_idx, label_encoder = load_vocab_and_labels(
            config.VOCAB_SAVE_PATH, 
            config.LABEL_ENCODER_SAVE_PATH
        )
        
        # Still need to load and process test data
        data = prepare_data_for_training(config)
        
    else:
        print("\nPreprocessed data not found. Preparing data from scratch...")
        data = prepare_data_for_training(config)
    
    # Create test data loader
    test_loader = None
    if data['test_ids']:
        test_dataset = TextDataset(data['test_ids'], data['y_test_int'])
        from torch.utils.data import DataLoader
        pad_index = data['pad_index']
        
        # Use lambda to pass pad_index to collate_fn
        from dataset import collate_batch
        collate_fn_with_pad = lambda batch: collate_batch(batch, pad_index)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn_with_pad
        )
    
    if test_loader is None:
        print("Test data not available. Cannot compare models.")
        return
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Load model configurations
    lstm_config = {
        'embed_dim': config.EMBEDDING_DIM,
        'hidden_dim': config.HIDDEN_DIM,
        'n_layers': config.N_LAYERS,
        'dropout': config.DROPOUT
    }
    
    simple_rnn_config = {
        'embed_dim': config.EMBEDDING_DIM,
        'hidden_dim': config.HIDDEN_DIM,
        'n_layers': config.N_LAYERS
    }
    
    # Load trained models
    models = {}
    
    # Load LSTM model
    lstm_model = load_model(
        model_class=RNNLSTMClassifier,
        model_path=config.MODEL_SAVE_PATH,
        data=data,
        device=device,
        **lstm_config
    )
    if lstm_model:
        models['LSTM'] = lstm_model
    
    # Load SimpleRNN model
    simple_rnn_model = load_model(
        model_class=SimpleRNNClassifier,
        model_path="best_simple_rnn_classifier.pth",
        data=data,
        device=device,
        **simple_rnn_config
    )
    if simple_rnn_model:
        models['SimpleRNN'] = simple_rnn_model
    
    # Load CNN-LSTM model
    from models import CNNLSTMClassifier
    cnn_lstm_model = load_model(
        model_class=CNNLSTMClassifier,
        model_path=config.CNN_LSTM_SAVE_PATH,
        data=data,
        device=device,
        **cnn_lstm_config
    )
    if cnn_lstm_model:
        models['CNN-LSTM'] = cnn_lstm_model
    if not models:
        print("No models could be loaded. Make sure you've trained the models first.")
        return
    
    # Evaluate and compare models
    results = evaluate_models(
        models=models,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        label_encoder=data['label_encoder']
    )
    
    # Plot comparative charts
    plot_accuracy_comparison(results)
    plot_class_f1_comparison(results, data['label_encoder'])
    
    # Save comparison results
    with open('model_comparison_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("\nComparison results saved to model_comparison_results.pkl")

if __name__ == "__main__":
    main()