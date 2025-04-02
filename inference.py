"""
Inference utilities for emotion classification.
"""
import torch
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
import pickle
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import RNNLSTMClassifier, SimpleRNNClassifier,CNNLSTMClassifier
from preprocessing import preprocess_text, convert_docs_to_ids
from utils import get_device

def load_inference_resources(model_path, vocab_path, label_encoder_path, use_stemming=True):
    """
    Load all resources needed for inference.
    
    Args:
        model_path: Path to the saved model weights
        vocab_path: Path to the saved vocabulary
        label_encoder_path: Path to the saved label encoder
        use_stemming: Whether to use stemming for preprocessing
        
    Returns:
        Dictionary with all needed resources
    """
    # Get device
    device = get_device()
    
    # Load vocabulary and label encoder
    try:
        with open(vocab_path, 'rb') as f:
            word_to_idx = pickle.load(f)
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Error loading vocabulary or label encoder: {e}")
    
    # Setup text preprocessing resources
    # Download NLTK resources if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK 'punkt' tokenizer...")
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK 'stopwords'...")
        nltk.download('stopwords')
    
    stop_words_set = set(stopwords.words('english'))
    stop_words_set.update(string.punctuation)
    processor = SnowballStemmer('english') if use_stemming else None
    
    # Get necessary model parameters from vocab
    vocab_size = len(word_to_idx)
    pad_idx = word_to_idx.get("<PAD>", 0)
    num_classes = len(label_encoder.classes_)
    
    # Default model hyperparameters - can be replaced with loaded config
    embedding_dim = 128
    hidden_dim = 256
    n_layers = 3
    dropout = 0.5
    
    # Create and load model
    model = RNNLSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        n_layers=n_layers,
        dropout=dropout,
        pad_idx=pad_idx
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        raise ValueError(f"Error loading model: {e}")
    
    return {
        'model': model,
        'word_to_idx': word_to_idx,
        'label_encoder': label_encoder,
        'processor': processor,
        'stop_words_set': stop_words_set,
        'device': device
    }

def predict_emotion(text, resources):
    """
    Predict emotion for a given text using loaded resources.
    
    Args:
        text: Input text string
        resources: Dictionary with all inference resources
        
    Returns:
        Dictionary with prediction results
    """
    # Extract resources
    model = resources['model']
    word_to_idx = resources['word_to_idx']
    label_encoder = resources['label_encoder']
    processor = resources['processor']
    stop_words_set = resources['stop_words_set']
    device = resources['device']
    
    # Preprocess the input text
    processed_text = preprocess_text(text, processor, stop_words_set)
    text_ids = convert_docs_to_ids([processed_text], word_to_idx)[0]
    
    # Convert to tensor and add batch dimension
    text_tensor = torch.tensor([text_ids], dtype=torch.long).to(device)
    
    # Get model prediction
    with torch.no_grad():
        logits = model(text_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze()
        pred_idx = torch.argmax(probabilities).item()
        confidence = probabilities[pred_idx].item() * 100
    
    # Convert to emotion label
    emotion = label_encoder.inverse_transform([pred_idx])[0]
    
    # Get all probabilities
    all_emotions = {
        label: prob.item() * 100
        for label, prob in zip(label_encoder.classes_, probabilities)
    }
    
    # Sort emotions by probability
    sorted_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'text': text,
        'predicted_emotion': emotion,
        'confidence': confidence,
        'all_emotions': sorted_emotions
    }

def main():
    """Command-line interface for emotion prediction."""
    parser = argparse.ArgumentParser(description='Predict emotion from text')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--model', type=str, default='best_emotion_classifier.pth', help='Path to model file')
    parser.add_argument('--vocab', type=str, default='vocab.pkl', help='Path to vocabulary file')
    parser.add_argument('--labels', type=str, default='label_encoder.pkl', help='Path to label encoder file')
    args = parser.parse_args()
    
    # Load resources
    resources = load_inference_resources(args.model, args.vocab, args.labels)
    
    # Get text from command line or prompt
    text = args.text
    if not text:
        text = input("Enter text to analyze: ")
    
    # Predict emotion
    result = predict_emotion(text, resources)
    
    # Print results
    print(f"\nText: {result['text']}")
    print(f"Predicted emotion: {result['predicted_emotion']} (Confidence: {result['confidence']:.2f}%)")
    print("\nAll emotions:")
    for emotion, prob in result['all_emotions']:
        print(f"  {emotion}: {prob:.2f}%")

if __name__ == "__main__":
    main()