"""
Utility functions for emotion classification project.
"""
import torch
import numpy as np
import random
import os
import pickle

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

def get_device():
    """Determine the available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def save_model_info(model, vocab_size, word_to_idx, label_encoder, config, path="model_info.pkl"):
    """Save model configuration and related data for later use."""
    model_info = {
        'model_architecture': model.__class__.__name__,
        'vocab_size': vocab_size,
        'embedding_dim': config.EMBEDDING_DIM,
        'hidden_dim': config.HIDDEN_DIM,
        'n_layers': config.N_LAYERS,
        'dropout': config.DROPOUT,
        'num_classes': len(label_encoder.classes_),
        'class_names': list(label_encoder.classes_),
        'word_to_idx': word_to_idx,
        'config': {key: getattr(config, key) for key in dir(config) if not key.startswith('__')}
    }
    
    with open(path, 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"Model info saved to {path}")
    return model_info

def load_model_info(path="model_info.pkl"):
    """Load saved model configuration and related data."""
    try:
        with open(path, 'rb') as f:
            model_info = pickle.load(f)
        return model_info
    except Exception as e:
        print(f"Error loading model info: {e}")
        return None

def predict_emotion(text, model, word_to_idx, label_encoder, preprocessor, stop_words, device):
    """
    Predict emotion for a given text using a trained model.
    
    Args:
        text: Input text string
        model: Trained PyTorch model
        word_to_idx: Vocabulary dictionary mapping words to indices
        label_encoder: Fitted LabelEncoder to map prediction to emotion name
        preprocessor: Text preprocessor (e.g., stemmer)
        stop_words: Set of stop words to remove
        device: Device to run inference on
        
    Returns:
        Predicted emotion and confidence score
    """
    # Preprocess the input text
    from preprocessing import preprocess_text, convert_docs_to_ids
    
    processed_text = preprocess_text(text, preprocessor, stop_words)
    text_ids = convert_docs_to_ids([processed_text], word_to_idx)[0]
    
    # Convert to tensor and add batch dimension
    text_tensor = torch.tensor([text_ids], dtype=torch.long).to(device)
    
    # Get model prediction
    model.eval()
    with torch.no_grad():
        logits = model(text_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze()
        pred_idx = torch.argmax(probabilities).item()
        confidence = probabilities[pred_idx].item()
    
    # Convert to emotion label
    emotion = label_encoder.inverse_transform([pred_idx])[0]
    
    return {
        'emotion': emotion,
        'confidence': confidence,
        'all_probabilities': {
            label: prob.item() 
            for label, prob in zip(label_encoder.classes_, probabilities)
        }
    }