"""
Configuration settings for the emotion classification models.
"""
import os

# --- Data Directories ---
TRAIN_DIR = 'data/WASSA-2017/train'    # Directory containing training files
VAL_DIR = 'data/WASSA-2017/val/'       # Directory containing validation files
TEST_DIR = 'data/WASSA-2017/test/'     # Directory containing test files

# --- File Patterns ---
FILE_PATTERN = '*.txt'                 # Assuming files are like emotion-type.txt
LABEL_SEPARATOR = '-'                  # How emotion is separated in the filename

# --- Preprocessing Options ---
USE_STEMMING = True
MIN_WORD_FREQ = 1

# --- Special Tokens ---
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# --- Model & Training Hyperparameters ---
EMBEDDING_DIM = 128                    # Dimensionality of word embeddings
HIDDEN_DIM = 256                       # Dimensionality of RNN hidden state
N_LAYERS = 3                           # Number of RNN layers
DROPOUT = 0.5                          # Dropout probability
LEARNING_RATE = 0.001
EPOCHS = 100                           # Number of training epochs
BATCH_SIZE = 64

# --- Model Save Paths ---
MODEL_SAVE_PATH = 'best_emotion_classifier.pth'
VOCAB_SAVE_PATH = 'vocab.pkl'
LABEL_ENCODER_SAVE_PATH = 'label_encoder.pkl'

# --- Ensure directories exist ---
if not os.path.exists(TEST_DIR):
    print(f"Warning: Test directory '{TEST_DIR}' not found. Please create it and add test files.")