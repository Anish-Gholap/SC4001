"""
Enhanced script to compare different model architectures for emotion classification
with comprehensive visualization of performance metrics.
Saves all visualizations to a dedicated 'comparison' folder.
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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_curve, auc
import seaborn as sns
from itertools import cycle

# Add the parent directory to the path so we can import from our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
import config
from preprocessing import prepare_data_for_training, load_vocab_and_labels
from dataset import get_data_loaders, TextDataset
from models import RNNLSTMClassifier, SimpleRNNClassifier
from training import evaluate, plot_confusion_matrix
from utils import set_seed, get_device

# Create the comparison folder if it doesn't exist
COMPARISON_FOLDER = 'comparison'
os.makedirs(COMPARISON_FOLDER, exist_ok=True)

def save_figure(filename):
    """
    Save figure to the comparison folder.
    
    Args:
        filename: Name of the file (without folder path)
    """
    filepath = os.path.join(COMPARISON_FOLDER, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {filepath}")

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
        # Set weights_only=True to avoid the FutureWarning
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
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
        
        # For ROC curve, we need probabilities
        probabilities = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences = sequences.to(device)
                logits = model(sequences)
                probs = torch.nn.functional.softmax(logits, dim=1)
                probabilities.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())
        
        probabilities = np.vstack(probabilities)
        all_labels = np.concatenate(all_labels)
        
        # Compute precision, recall, f1 for detailed metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            test_labels, test_preds, average=None, labels=range(len(label_encoder.classes_))
        )
        
        results[name] = {
            'loss': test_loss,
            'accuracy': test_acc,
            'labels': test_labels,
            'predictions': test_preds,
            'probabilities': probabilities,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
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
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            results[name]['confusion_matrix'], 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_
        )
        plt.title(f'{name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        save_figure(f'{name.lower()}_confusion_matrix.png')
        plt.show()
    
    return results

def plot_accuracy_comparison(results):
    """Plot accuracy comparison of different models."""
    # Extract accuracies
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'red'][:len(models)])
    
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
    save_figure('model_accuracy_comparison.png')
    plt.show()

def plot_metrics_comparison(results, label_encoder):
    """
    Plot comprehensive metrics comparison (accuracy, precision, recall, f1).
    
    Args:
        results: Dictionary with evaluation results for each model
        label_encoder: LabelEncoder to get class names
    """
    # Create DataFrame for metrics
    models = list(results.keys())
    metrics_df = pd.DataFrame({
        'Model': [],
        'Metric': [],
        'Value': []
    })
    
    # Calculate macro averages for each model
    for model_name in models:
        # Add overall accuracy
        metrics_df = pd.concat([metrics_df, pd.DataFrame({
            'Model': [model_name],
            'Metric': ['Accuracy'],
            'Value': [results[model_name]['accuracy']]
        })], ignore_index=True)
        
        # Add macro precision, recall, f1
        precision_macro = np.mean(results[model_name]['precision'])
        recall_macro = np.mean(results[model_name]['recall'])
        f1_macro = np.mean(results[model_name]['f1'])
        
        metrics_df = pd.concat([metrics_df, pd.DataFrame({
            'Model': [model_name, model_name, model_name],
            'Metric': ['Precision (Macro)', 'Recall (Macro)', 'F1 Score (Macro)'],
            'Value': [precision_macro, recall_macro, f1_macro]
        })], ignore_index=True)
    
    # Plot the comparison
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        data=metrics_df, 
        x='Metric', 
        y='Value', 
        hue='Model', 
        palette='muted'
    )
    plt.title('Model Performance Metrics Comparison')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=15)
    plt.legend(title='Model')
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8)
    
    plt.tight_layout()
    save_figure('model_metrics_comparison.png')
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
    ax = df.plot(kind='bar', ax=plt.gca())
    plt.title('F1 Score Comparison by Class')
    plt.xlabel('Emotion Class')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1.0)
    plt.legend(title='Model')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8)
    
    plt.tight_layout()
    save_figure('class_f1_comparison.png')
    plt.show()

def plot_training_history_comparison(models):
    """
    Plot training history comparison for multiple models.
    
    Args:
        models: List of model names to compare
    """
    plt.figure(figsize=(15, 10))
    
    # Setup subplots
    plt.subplot(2, 1, 1)
    for model_name in models:
        try:
            # Try to load history
            history_file = f"{model_name.lower()}_training_history.pkl"
            if os.path.exists(history_file):
                with open(history_file, 'rb') as f:
                    history = pickle.load(f)
                
                epochs = range(1, len(history['train_losses']) + 1)
                plt.plot(epochs, history['train_losses'], 'o-', label=f'{model_name} Train')
                plt.plot(epochs, history['val_losses'], 's-', label=f'{model_name} Val')
        except Exception as e:
            print(f"Could not load training history for {model_name}: {e}")
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    for model_name in models:
        try:
            # Try to load history
            history_file = f"{model_name.lower()}_training_history.pkl"
            if os.path.exists(history_file):
                with open(history_file, 'rb') as f:
                    history = pickle.load(f)
                
                epochs = range(1, len(history['train_accs']) + 1)
                plt.plot(epochs, history['train_accs'], 'o-', label=f'{model_name} Train')
                plt.plot(epochs, history['val_accs'], 's-', label=f'{model_name} Val')
        except Exception as e:
            print(f"Could not load training history for {model_name}: {e}")
    
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    save_figure('training_history_comparison.png')
    plt.show()

def plot_roc_curves(results, label_encoder):
    """
    Plot ROC curves for multiclass classification.
    
    Args:
        results: Dictionary with evaluation results for each model
        label_encoder: LabelEncoder to get class names
    """
    plt.figure(figsize=(15, 12))
    
    n_classes = len(label_encoder.classes_)
    
    for model_name, model_result in results.items():
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        y_true = np.zeros((len(model_result['labels']), n_classes))
        for i in range(len(model_result['labels'])):
            y_true[i, model_result['labels'][i]] = 1
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], model_result['probabilities'][:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot ROC curves
        colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
        for i, color, class_name in zip(range(n_classes), colors, label_encoder.classes_):
            plt.plot(
                fpr[i], tpr[i], color=color, lw=2,
                label=f'{model_name}, {class_name} (AUC = {roc_auc[i]:.2f})'
            )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Multiclass Classification')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    save_figure('roc_curves.png')
    plt.show()

def plot_precision_recall_by_class(results, label_encoder):
    """
    Plot precision and recall for each class across models.
    
    Args:
        results: Dictionary with evaluation results for each model
        label_encoder: LabelEncoder to get class names
    """
    class_names = label_encoder.classes_
    models = list(results.keys())
    
    # Create DataFrames for precision and recall
    precision_df = pd.DataFrame(index=class_names)
    recall_df = pd.DataFrame(index=class_names)
    
    for model_name, model_result in results.items():
        model_precision = []
        model_recall = []
        for i, class_name in enumerate(class_names):
            model_precision.append(model_result['precision'][i])
            model_recall.append(model_result['recall'][i])
        
        precision_df[model_name] = model_precision
        recall_df[model_name] = model_recall
    
    # Plot precision
    plt.figure(figsize=(15, 6))
    ax1 = plt.subplot(1, 2, 1)
    precision_df.plot(kind='bar', ax=ax1)
    plt.title('Precision by Class')
    plt.ylabel('Precision')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    # Plot recall
    ax2 = plt.subplot(1, 2, 2)
    recall_df.plot(kind='bar', ax=ax2)
    plt.title('Recall by Class')
    plt.ylabel('Recall')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    save_figure('precision_recall_by_class.png')
    plt.show()

def save_final_report(results, label_encoder):
    """
    Save a comprehensive markdown report of model comparisons.
    
    Args:
        results: Dictionary with evaluation results for each model
        label_encoder: LabelEncoder to get class names
    """
    report_path = os.path.join(COMPARISON_FOLDER, 'comparison_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Emotion Classification Model Comparison Report\n\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall performance table
        f.write("## Overall Performance\n\n")
        f.write("| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |\n")
        f.write("|-------|----------|-----------------|--------------|----------|\n")
        
        for model_name, model_result in results.items():
            precision_macro = np.mean(model_result['precision'])
            recall_macro = np.mean(model_result['recall'])
            f1_macro = np.mean(model_result['f1'])
            
            f.write(f"| {model_name} | {model_result['accuracy']:.4f} | {precision_macro:.4f} | {recall_macro:.4f} | {f1_macro:.4f} |\n")
        
        f.write("\n")
        
        # Class-wise performance
        f.write("## Performance by Emotion Class\n\n")
        
        for class_idx, class_name in enumerate(label_encoder.classes_):
            f.write(f"### {class_name}\n\n")
            f.write("| Model | Precision | Recall | F1 Score | Support |\n")
            f.write("|-------|-----------|--------|----------|--------|\n")
            
            for model_name, model_result in results.items():
                f.write(f"| {model_name} | {model_result['precision'][class_idx]:.4f} | {model_result['recall'][class_idx]:.4f} | {model_result['f1'][class_idx]:.4f} | {model_result['support'][class_idx]} |\n")
            
            f.write("\n")
        
        # List of visualizations
        f.write("## Visualizations\n\n")
        f.write("The following visualizations have been generated:\n\n")
        
        for filename in os.listdir(COMPARISON_FOLDER):
            if filename.endswith('.png'):
                f.write(f"- [{filename}](./{filename})\n")
    
    print(f"Comprehensive report saved to {report_path}")

def main():
    """Compare different model architectures for emotion classification."""
    print("="*80)
    print("COMPARING EMOTION CLASSIFICATION MODELS")
    print("="*80)
    print(f"All comparison results will be saved to '{COMPARISON_FOLDER}' folder")
    
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
    
    # Plot comprehensive comparison charts
    print("\nGenerating comprehensive performance comparison visualizations...")
    
    # Basic accuracy comparison
    plot_accuracy_comparison(results)
    
    # Comprehensive metrics comparison
    plot_metrics_comparison(results, data['label_encoder'])
    
    # Class-specific F1 score comparison
    plot_class_f1_comparison(results, data['label_encoder'])
    
    # Precision and recall by class
    plot_precision_recall_by_class(results, data['label_encoder'])
    
    # Try to plot training history if available
    plot_training_history_comparison(list(models.keys()))
    
    # Plot ROC curves
    plot_roc_curves(results, data['label_encoder'])
    
    # Save comprehensive report
    save_final_report(results, data['label_encoder'])
    
    # Save comparison results
    results_path = os.path.join(COMPARISON_FOLDER, 'model_comparison_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nComparison results saved to {results_path}")
    print(f"\nAll visualizations saved to '{COMPARISON_FOLDER}' folder.")

if __name__ == "__main__":
    main()