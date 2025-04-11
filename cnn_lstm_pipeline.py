from model_pipeline import BaseModelPipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import argparse
from tqdm import tqdm
import re
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import os
import json
import requests
import zipfile
import io

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# Custom dataset for text classification
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize and convert to indices
        tokens = word_tokenize(text.lower())
        indices = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]

        # Pad or truncate
        if len(indices) < self.max_length:
            indices = indices + [self.vocab["<pad>"]] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]

        return {
            'text_indices': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


# Function to load GloVe embeddings
def load_glove_embeddings(glove_path, vocab, embedding_dim):
    """
    Load GloVe embeddings from file for the words in vocab.

    Args:
        glove_path: Path to the GloVe embeddings file
        vocab: Dictionary mapping words to indices
        embedding_dim: Dimension of the embeddings

    Returns:
        embedding_matrix: Tensor of shape (vocab_size, embedding_dim)
    """
    # Initialize embedding matrix
    embedding_matrix = torch.zeros(len(vocab), embedding_dim)

    # Read GloVe file and fill embedding matrix
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading GloVe embeddings"):
            values = line.split()
            word = values[0]

            if word in vocab:
                vector = torch.tensor([float(val) for val in values[1:]])
                embedding_matrix[vocab[word]] = vector

    return embedding_matrix


# Function to download GloVe embeddings
def download_glove_embeddings(cache_dir, dim=300):
    """
    Download GloVe embeddings if not already available.

    Args:
        cache_dir: Directory to save the embeddings
        dim: Dimension of the embeddings (50, 100, 200, or 300)

    Returns:
        path: Path to the GloVe embeddings file
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # GloVe file names and URLs
    glove_files = {
        50: "glove.6B.50d.txt",
        100: "glove.6B.100d.txt",
        200: "glove.6B.200d.txt",
        300: "glove.6B.300d.txt"
    }

    # Check if we already have the file
    glove_path = os.path.join(cache_dir, glove_files[dim])
    if os.path.exists(glove_path):
        print(f"GloVe embeddings found at {glove_path}")
        return glove_path

    # Download and extract GloVe
    print(f"Downloading GloVe embeddings (dim={dim})...")
    url = "https://nlp.stanford.edu/data/glove.6B.zip"
    response = requests.get(url)

    # Extract the specific file we need
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        print(f"Extracting {glove_files[dim]}...")
        zip_ref.extract(glove_files[dim], path=cache_dir)

    print(f"GloVe embeddings saved to {glove_path}")
    return glove_path


# CNNLSTMClassifier model implementation
class CNNLSTMClassifier(nn.Module):
    """
    Hybrid CNN-LSTM model for text classification.

    Architecture:
    1. Embedding layer to convert word indices to vectors
    2. 1D convolutional layer to extract n-gram features
    3. Max pooling to reduce dimensionality and extract most important features
    4. LSTM layer to capture sequential dependencies in the extracted features
    5. Fully connected layer for classification

    This model combines the strengths of CNNs (local pattern recognition)
    with LSTMs (sequential dependency modeling).
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 n_filters=100, filter_sizes=[3, 4, 5], lstm_layers=1,
                 dropout=0.5, pad_idx=0, pretrained_embeddings=None):
        super().__init__()

        # Initialize embedding layer
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, padding_idx=pad_idx, freeze=False
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # CNN layers
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=n_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        # Calculate the output dimension of the CNN layers
        # After concatenating all filter outputs
        self.cnn_output_dim = n_filters * len(filter_sizes)

        # LSTM layer takes the concatenated CNN features as input
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # Final classifier layer
        # hidden_dim * 2 because LSTM is bidirectional
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_batch):
        # text_batch shape: [batch_size, seq_len]

        # Embedding layer
        # embedded shape: [batch_size, seq_len, embed_dim]
        embedded = self.embedding(text_batch)

        # CNN expects input shape [batch_size, channels, seq_len]
        # So we need to transpose dimensions 1 and 2
        # embedded_conv shape: [batch_size, embed_dim, seq_len]
        embedded_conv = embedded.permute(0, 2, 1)

        # Apply convolutions and activation
        # Each conv_x shape: [batch_size, n_filters, seq_len - filter_size + 1]
        conv_outputs = [F.relu(conv(embedded_conv)) for conv in self.convs]

        # Apply max pooling to extract the most important features
        # Each pooled_x shape: [batch_size, n_filters, 1]
        pooled_outputs = [F.max_pool1d(conv_x, conv_x.shape[2]) for conv_x in conv_outputs]

        # Concatenate pooled outputs from different filter sizes
        # cat shape: [batch_size, n_filters * len(filter_sizes), 1]
        cat = torch.cat(pooled_outputs, dim=1)

        # Remove the last dimension and apply dropout
        # cat shape: [batch_size, n_filters * len(filter_sizes)]
        cat = cat.squeeze(-1)
        cat = self.dropout(cat)

        # Reshape for LSTM input
        # lstm_input shape: [batch_size, 1, n_filters * len(filter_sizes)]
        lstm_input = cat.unsqueeze(1)

        # Apply LSTM
        # lstm_out shape: [batch_size, 1, hidden_dim * 2]
        # hidden shape: [num_layers * 2, batch_size, hidden_dim]
        lstm_out, (hidden, cell) = self.lstm(lstm_input)

        # Concatenate the final forward and backward hidden states
        # hidden[-2,:,:] is the last forward direction hidden state
        # hidden[-1,:,:] is the last backward direction hidden state
        hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # Apply dropout and final classification layer
        # output shape: [batch_size, num_classes]
        output = self.fc(self.dropout(hidden_cat))

        return output


# Base class for text classification pipelines using vocabulary-based models
class TextClassificationPipeline(BaseModelPipeline):
    def __init__(self, args):
        super().__init__(args)
        self.vocab = None
        self.embedding_matrix = None

    def prepare_data(self):
        """Custom data preparation for vocabulary-based models."""
        print(f"Loading data from {self.args.data_path}")
        df = pd.read_csv(self.args.data_path)

        # Check if columns exist
        if self.args.text_column not in df.columns:
            raise ValueError(f"Text column '{self.args.text_column}' not in dataset")
        if self.args.label_column not in df.columns:
            raise ValueError(f"Label column '{self.args.label_column}' not in dataset")

        # Split into train and validation sets
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            random_state=self.args.seed,
            stratify=df[self.args.label_column]
        )

        # Convert string labels to integers if needed
        if isinstance(train_df[self.args.label_column].iloc[0], str):
            self.label_map = {label: idx for idx, label in enumerate(train_df[self.args.label_column].unique())}
            train_df[self.args.label_column] = train_df[self.args.label_column].map(self.label_map)
            val_df[self.args.label_column] = val_df[self.args.label_column].map(self.label_map)

            # Save label mapping
            with open(os.path.join(self.args.output_dir, 'label_map.json'), 'w') as f:
                json.dump(self.label_map, f)

        # Build vocabulary
        all_texts = df[self.args.text_column].values
        print("Building vocabulary...")
        self.build_vocab(all_texts, min_freq=self.args.min_freq)

        # Create datasets
        self.train_dataset = TextClassificationDataset(
            texts=train_df[self.args.text_column].values,
            labels=train_df[self.args.label_column].values,
            vocab=self.vocab,
            max_length=self.args.max_length
        )

        self.val_dataset = TextClassificationDataset(
            texts=val_df[self.args.text_column].values,
            labels=val_df[self.args.label_column].values,
            vocab=self.vocab,
            max_length=self.args.max_length
        )

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size
        )

        # Get number of classes
        self.num_classes = len(df[self.args.label_column].unique())

        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")

        return self.train_loader, self.val_loader, self.num_classes

    def build_vocab(self, texts, min_freq=2):
        """Build vocabulary from texts."""
        # Tokenize all texts
        all_tokens = []
        for text in tqdm(texts, desc="Tokenizing texts"):
            all_tokens.extend(word_tokenize(str(text).lower()))

        # Count token frequencies
        token_counts = Counter(all_tokens)

        # Create vocabulary with special tokens
        self.vocab = {"<pad>": 0, "<unk>": 1}

        # Add tokens that appear at least min_freq times
        idx = 2
        for token, count in token_counts.items():
            if count >= min_freq:
                self.vocab[token] = idx
                idx += 1

        # Load pretrained embeddings if specified
        if hasattr(self.args, 'use_pretrained_embeddings') and self.args.use_pretrained_embeddings:
            self.load_pretrained_embeddings()

    def load_pretrained_embeddings(self):
        """Load pretrained GloVe embeddings."""
        print(f"Loading pretrained GloVe embeddings ({self.args.embedding_dim}d)...")

        # Create cache directory
        cache_dir = os.path.join(self.args.output_dir, 'embeddings')
        os.makedirs(cache_dir, exist_ok=True)

        # Download GloVe embeddings if needed
        glove_path = download_glove_embeddings(cache_dir, dim=self.args.embedding_dim)

        # Load embeddings
        self.embedding_matrix = load_glove_embeddings(glove_path, self.vocab, self.args.embedding_dim)

        print(
            f"Loaded pretrained embeddings for {torch.sum(torch.sum(self.embedding_matrix, dim=1) != 0).item()} words")

    def train_epoch(self, model, data_loader, optimizer, scheduler):
        """Train the model for one epoch."""
        model.train()
        losses = []

        # Add progress bar
        progress_bar = tqdm(data_loader, desc="Training")

        for batch in progress_bar:
            optimizer.zero_grad()

            text_indices = batch['text_indices'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass
            outputs = model(text_indices)

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)

            losses.append(loss.item())

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        return np.mean(losses)

    def evaluate(self, model, data_loader):
        """Evaluate the model on a dataset."""
        model.eval()
        losses = []
        predictions = []
        actual_labels = []

        # Add progress bar
        progress_bar = tqdm(data_loader, desc="Evaluating")

        with torch.no_grad():
            for batch in progress_bar:
                text_indices = batch['text_indices'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                outputs = model(text_indices)

                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(outputs, labels)
                losses.append(loss.item())

                _, preds = torch.max(outputs, dim=1)

                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())

                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})

        # Calculate metrics
        accuracy = accuracy_score(actual_labels, predictions)
        report = classification_report(actual_labels, predictions, output_dict=True)

        return np.mean(losses), accuracy, report, predictions, actual_labels


# CNN-LSTM specific pipeline
class CNNLSTMPipeline(TextClassificationPipeline):
    def build_model(self):
        """Build and return the CNN-LSTM model."""
        # Determine if we're using pretrained embeddings
        pretrained_embeddings = self.embedding_matrix if hasattr(self,
                                                                 'embedding_matrix') and self.embedding_matrix is not None else None

        # Parse filter sizes from string format if needed (e.g., "[3, 4, 5]")
        filter_sizes = eval(self.args.filter_sizes) if isinstance(self.args.filter_sizes,
                                                                  str) else self.args.filter_sizes

        model = CNNLSTMClassifier(
            vocab_size=len(self.vocab),
            embed_dim=self.args.embedding_dim,
            hidden_dim=self.args.hidden_dim,
            num_classes=self.num_classes,
            n_filters=self.args.num_filters,
            filter_sizes=filter_sizes,
            lstm_layers=self.args.lstm_layers,
            dropout=self.args.dropout_rate,
            pretrained_embeddings=pretrained_embeddings
        )
        return model


def main():
    parser = argparse.ArgumentParser(description='CNN-LSTM for Text Classification')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV dataset')
    parser.add_argument('--text_column', type=str, default='text', help='Column name for text data')
    parser.add_argument('--label_column', type=str, default='label', help='Column name for labels')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum sequence length')
    parser.add_argument('--min_freq', type=int, default=2, help='Minimum frequency for vocabulary')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=300, help='Dimension of word embeddings')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of LSTM hidden state')
    parser.add_argument('--num_filters', type=int, default=100, help='Number of filters in CNN')
    parser.add_argument('--filter_sizes', type=str, default='[3, 4, 5]', help='Filter sizes for CNN')
    parser.add_argument('--lstm_layers', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--use_pretrained_embeddings', action='store_true', help='Use pretrained GloVe embeddings')
    parser.add_argument('--output_dir', type=str, default='./results_cnn_lstm', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Create and run the pipeline
    pipeline = CNNLSTMPipeline(args)
    pipeline.train()


if __name__ == "__main__":
    main()