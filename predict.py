import pickle
import string

import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# --- Configuration ---
# !! MUST MATCH the parameters used during training !!
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
N_LAYERS = 3
DROPOUT = 0.5 # Include dropout value if your saved model class uses it

# Paths to saved files
MODEL_PATH = 'best_emotion_classifier.pth'
VOCAB_PATH = 'vocab.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

# Preprocessing settings (MUST MATCH training)
USE_STEMMING = True
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# --- Import or Define Model Class ---
# Option 1: Define the class directly here (if not importing)
# (Make sure this definition EXACTLY matches the one used for training the saved model)
class RNNClassifierLogits(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, n_layers=1, dropout=0.5, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, bidirectional=True,
                           dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_batch):
        embedded = self.dropout(self.embedding(text_batch))
        outputs, (hidden, cell) = self.rnn(embedded)
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(self.dropout(hidden_cat))

# Option 2: Import from another file (if you saved it separately)
# from model import RNNClassifierLogits # Assuming your class is in model.py


# --- Preprocessing Function (MUST MATCH TRAINING) ---
def preprocess_text(text, processor, stop_words_set):
    """Cleans and processes a single string of text."""
    text = str(text).lower() # Ensure string and lowercase
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words_set]
    if processor:
        if isinstance(processor, SnowballStemmer):
            processed_tokens = [processor.stem(word) for word in tokens]
        # Add elif for Lemmatizer if that was used
        else:
            processed_tokens = tokens
    else:
        processed_tokens = tokens
    return processed_tokens

# --- Prediction Function ---
def predict_emotion(text, model, word_to_idx, label_encoder, device, text_processor, stop_words):
    """Preprocesses text, runs inference, and returns predicted label and probability."""
    model.eval() # Set model to evaluation mode

    # Preprocess the input text
    processed_tokens = preprocess_text(text, text_processor, stop_words)
    if not processed_tokens:
        return "N/A", 0.0 # Handle empty input after preprocessing

    # Convert tokens to indices
    unk_index = word_to_idx[UNK_TOKEN]
    indexed_tokens = [word_to_idx.get(token, unk_index) for token in processed_tokens]

    # Convert to tensor and add batch dimension
    tensor = torch.LongTensor(indexed_tokens).unsqueeze(0).to(device) # (1, seq_len)

    # Get prediction from model (no gradients needed)
    with torch.no_grad():
        logits = model(tensor)

    # Apply softmax to get probabilities
    probabilities = F.softmax(logits, dim=1)

    # Get the index and probability of the highest prediction
    top_prob, top_idx = probabilities.max(dim=1)
    predicted_idx = top_idx.item()
    predicted_prob = top_prob.item()

    # Convert index back to label name
    predicted_label = label_encoder.inverse_transform([predicted_idx])[0]

    return predicted_label, predicted_prob


# --- Main Inference Loop ---
if __name__ == "__main__":
    # --- Setup Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
         device = torch.device("mps")
         print("Using MPS device (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # --- Load Vocabulary and Label Encoder ---
    print("Loading vocabulary and label encoder...")
    try:
        with open(VOCAB_PATH, 'rb') as f:
            word_to_idx = pickle.load(f)
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not load '{e.filename}'.")
        print("Please ensure you have run the training script to save 'vocab.pkl' and 'label_encoder.pkl'.")
        exit()
    except Exception as e:
        print(f"An error occurred loading files: {e}")
        exit()

    vocab_size = len(word_to_idx)
    num_classes = len(label_encoder.classes_)
    PAD_INDEX = word_to_idx[PAD_TOKEN]
    print(f"Vocabulary size: {vocab_size}, Num classes: {num_classes}")
    print(f"Class names: {label_encoder.classes_}")


    # --- Load Model ---
    print(f"Loading model from {MODEL_PATH}...")
    try:
        # Instantiate the model with correct parameters
        model = RNNClassifierLogits(
            vocab_size=vocab_size,
            embed_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_classes=num_classes,
            n_layers=N_LAYERS,
            dropout=DROPOUT, # Include if your model class uses it
            pad_idx=PAD_INDEX
        )
        # Load the saved state dictionary
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device)) # map_location ensures compatibility
        model.to(device) # Move model to the chosen device
        model.eval()     # Set to evaluation mode
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'.")
        print("Please ensure the trained model file exists.")
        exit()
    except Exception as e:
        print(f"An error occurred loading the model: {e}")
        exit()

    # --- Initialize Preprocessing Tools ---
    print("Initializing preprocessing tools...")
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError as e:
        print(f"NLTK data missing: {e}. Please run nltk.download('punkt') and nltk.download('stopwords')")
        exit()

    stop_words_set = set(stopwords.words('english'))
    stop_words_set.update(string.punctuation)
    text_processor = SnowballStemmer('english') if USE_STEMMING else None # Or Lemmatizer if used

    print("\n--- Emotion Prediction ---")
    print("Enter text to classify (or type 'q' to quit):")

    while True:
        input_text = input("> ")
        if input_text.lower() == 'q':
            break
        if not input_text.strip(): # Handle empty input
             print("Please enter some text.")
             continue

        try:
            predicted_label, predicted_prob = predict_emotion(
                input_text, model, word_to_idx, label_encoder, device, text_processor, stop_words_set
            )
            if predicted_label == "N/A":
                 print("Input text resulted in empty sequence after preprocessing.")
            else:
                 print(f"Predicted Emotion: {predicted_label} (Confidence: {predicted_prob:.4f})")
        except Exception as e:
            print(f"An error occurred during prediction: {e}") # Catch potential errors

    print("Exiting inference script.")