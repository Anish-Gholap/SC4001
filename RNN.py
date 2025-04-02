import pandas as pd
import numpy as np
import nltk
from tqdm import tqdm # For progress bars
from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
import string
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from helper import load_classification_data,preprocess_text,build_vocab,convert_docs_to_ids,plot_confusion_matrix
from nltk.stem import SnowballStemmer # Or WordNetLemmatizer
from models import RNNLSTMClassifier, SimpleRNNClassifier

# --- Configuration ---
TRAIN_DIR = 'data/WASSA-2017/train'    # Directory containing training files (e.g., anger-train.txt, joy-train.txt)
VAL_DIR = 'data/WASSA-2017/val/'    # Directory containing validation files
TEST_DIR = 'data/WASSA-2017/test/'         # Directory containing test files (CREATE THIS if needed)

# --- Ensure TEST_DIR exists, create if not ---
if not os.path.exists(TEST_DIR):
    print(f"Warning: Test directory '{TEST_DIR}' not found. Please create it and add test files.")
    # Example: Create dummy test files if needed for testing the script structure
    # os.makedirs(TEST_DIR, exist_ok=True)
    # with open(os.path.join(TEST_DIR, 'dummy-test.txt'), 'w') as f:
    #     f.write("tweet\tscore\n") # Adjust header if needed
    #     f.write("this is a dummy test tweet\t0\n")


FILE_PATTERN = '*.txt'          # Assuming files are like emotion-type.txt
LABEL_SEPARATOR = '-'           # How emotion is separated in the filename (e.g., 'anger-ratings...')

USE_STEMMING = True
MIN_WORD_FREQ = 1

# --- Special Tokens ---
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# --- Model & Training Hyperparameters ---
EMBEDDING_DIM = 128         # Dimensionality of word embeddings
HIDDEN_DIM = 256            # Dimensionality of RNN hidden state
N_LAYERS = 3                # Number of RNN layers
DROPOUT = 0.5               # Dropout probability
LEARNING_RATE = 0.001
EPOCHS = 100                 # Number of training epochs
BATCH_SIZE = 64
MODEL_SAVE_PATH = 'best_emotion_classifier.pth'

# --- NLTK Downloads (if needed) ---
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK 'stopwords'...")
    nltk.download('stopwords')

# --- PyTorch Dataset ---
class TextDataset(Dataset):
    def __init__(self, sequences, labels):
        if len(sequences) != len(labels):
            raise ValueError("Sequences and labels must have the same length!")
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        # Convert to tensors
        sequence_tensor = torch.tensor(sequence, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long) # Use long for CrossEntropyLoss
        return sequence_tensor, label_tensor

# --- Collate Function ---
def collate_batch(batch, pad_index):
    """Collates data samples into batches with padding."""
    label_list, sequence_list = [], []
    for (_sequence, _label) in batch:
        label_list.append(_label)
        sequence_list.append(_sequence)

    sequences_padded = pad_sequence(sequence_list, batch_first=True, padding_value=pad_index)
    labels = torch.stack(label_list)
    return sequences_padded, labels


# --- Training Function ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
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

        progress_bar.set_postfix(loss=loss.item(), acc=correct_predictions/total_samples if total_samples > 0 else 0)

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0
    return epoch_loss, epoch_acc

# --- Evaluation Function (Modified to return predictions) ---
def evaluate(model, dataloader, criterion, device):
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

            all_labels.extend(labels_batch.cpu().numpy()) # Collect labels (move to CPU)
            all_preds.extend(preds.cpu().numpy())         # Collect predictions (move to CPU)

            progress_bar.set_postfix(loss=loss.item(), acc=correct_predictions/total_samples if total_samples > 0 else 0)

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0
    return epoch_loss, epoch_acc, all_labels, all_preds


# --- Plot Confusion Matrix ---



# --- Main Execution ---
if __name__ == "__main__":
    # --- Device Setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): # Check MPS availability
         device = torch.device("mps")
         print("Using MPS device (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # --- 1. Load Data ---
    raw_train, labels_train_str = load_classification_data(TRAIN_DIR, FILE_PATTERN, LABEL_SEPARATOR)
    raw_val, labels_val_str = load_classification_data(VAL_DIR, FILE_PATTERN, LABEL_SEPARATOR)
    raw_test, labels_test_str = load_classification_data(TEST_DIR, FILE_PATTERN, LABEL_SEPARATOR)

    if not raw_train or not raw_val:
         raise ValueError("Training or Validation data could not be loaded. Please check paths and files.")
    if not raw_test:
         print("Warning: Test data could not be loaded. Final evaluation will be skipped.")


    # --- 2. Setup Preprocessing Tools ---
    stop_words_set = set(stopwords.words('english'))
    stop_words_set.update(string.punctuation)
    processor = SnowballStemmer('english') if USE_STEMMING else None

    # --- 3. Process Text (Train, Val, Test) ---
    print("\nProcessing training documents...")
    processed_train = [preprocess_text(doc, processor, stop_words_set) for doc in tqdm(raw_train)]
    print("Processing validation documents...")
    processed_val = [preprocess_text(doc, processor, stop_words_set) for doc in tqdm(raw_val)]
    print("Processing test documents...")
    processed_test = [preprocess_text(doc, processor, stop_words_set) for doc in tqdm(raw_test)] if raw_test else []


    # --- 4. Build Vocabulary (using combined Train + Val + Test data) ---
    print("\nBuilding vocabulary...")
    all_processed_docs = processed_train + processed_val + processed_test
    if not all_processed_docs:
        raise ValueError("No documents found after processing all sets. Cannot build vocabulary.")
    word_to_idx, idx_to_word, vocab_size = build_vocab(all_processed_docs, MIN_WORD_FREQ)
    PAD_INDEX = word_to_idx[PAD_TOKEN] # Get PAD index after building vocab

    # --- 5. Convert Documents to Integer Sequences ---
    print("\nConverting documents to integer sequences...")
    train_ids = convert_docs_to_ids(processed_train, word_to_idx)
    val_ids = convert_docs_to_ids(processed_val, word_to_idx)
    test_ids = convert_docs_to_ids(processed_test, word_to_idx) if processed_test else []

    # --- 6. Process Target Labels ---
    print("\nProcessing target labels...")
    label_encoder = LabelEncoder()
    # Fit on all unique labels found across all sets to ensure consistency
    all_unique_labels = sorted(list(set(labels_train_str + labels_val_str + labels_test_str)))
    if not all_unique_labels:
         raise ValueError("No labels found in any dataset.")
    label_encoder.fit(all_unique_labels)

    y_train_int = label_encoder.transform(labels_train_str)
    y_val_int = label_encoder.transform(labels_val_str)
    y_test_int = label_encoder.transform(labels_test_str) if labels_test_str else []

    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    print(f"Class mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

    print("\nSaving vocabulary and label encoder...")
    try:
        with open('vocab.pkl', 'wb') as f:
            pickle.dump(word_to_idx, f)  # word_to_idx was created in step 4
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)  # label_encoder was created and fitted in step 6
        print(f"Successfully saved")
    except Exception as e:
        print(f"Error saving vocab/label encoder: {e}")

    # --- 7. Create Datasets and DataLoaders ---
    train_dataset = TextDataset(train_ids, y_train_int)
    val_dataset = TextDataset(val_ids, y_val_int)
    test_dataset = TextDataset(test_ids, y_test_int) if test_ids else None

    # Use partial function or lambda to pass pad_index to collate_fn
    collate_fn_with_pad = lambda batch: collate_batch(batch, PAD_INDEX)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_with_pad)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_with_pad)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_with_pad) if test_dataset else None

    # --- 8. Initialize Model, Loss, Optimizer ---
    model = RNNLSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        pad_idx=PAD_INDEX
    ).to(device)

    # model = SimpleRNNClassifier(
    #     vocab_size=vocab_size,
    #     embed_dim=EMBEDDING_DIM,
    #     hidden_dim=HIDDEN_DIM,
    #     num_classes=num_classes,
    #     n_layers=N_LAYERS,
    #     pad_idx=PAD_INDEX
    # ).to(device)

    criterion = nn.CrossEntropyLoss() # Handles logits directly
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nModel loaded on {device}")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")


    # --- 9. Training Loop with Validation & Best Model Saving ---
    best_val_acc = -1.0 # Initialize best validation accuracy

    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1} Training   -> Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # Evaluate on validation set
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device) # Ignore labels/preds for epoch summary
        print(f"Epoch {epoch+1} Validation -> Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # Save the model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"*** Best model saved to {MODEL_SAVE_PATH} (Val Acc: {best_val_acc:.4f}) ***")

    print("\n--- Training Complete ---")


    # --- 10. Final Evaluation on Test Set ---
    if test_loader:
        print("\n--- Evaluating Best Model on Test Set ---")
        # Load the best model weights
        try:
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device)) # map_location handles loading across devices
            print(f"Loaded best model weights from {MODEL_SAVE_PATH}")

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
            # print(cm) # Print numerical matrix if needed
            plot_confusion_matrix(cm, class_names=label_encoder.classes_, title='Test Set Confusion Matrix')

        except FileNotFoundError:
             print(f"Error: Could not find the saved model at {MODEL_SAVE_PATH}. Skipping final test evaluation.")
        except Exception as e:
             print(f"An error occurred during final test evaluation: {e}")

    else:
        print("\nTest loader not available. Skipping final test evaluation.")