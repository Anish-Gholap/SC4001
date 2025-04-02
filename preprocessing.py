"""
Text preprocessing utilities for emotion classification.
"""
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import string
import os
import glob
from tqdm import tqdm
from collections import Counter
import pickle
from sklearn.preprocessing import LabelEncoder

# --- NLTK Downloads ---
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

def get_label_from_filename(filename, separator):
    """Extract the emotion label from a filename."""
    base_name = os.path.basename(filename)
    label = base_name.split(separator)[0]
    return label.lower()

def load_classification_data(data_dir, pattern, label_separator):
    """Loads and combines data from multiple classification files."""
    all_files = glob.glob(os.path.join(data_dir, pattern))
    if not all_files:
        print(f"Warning: No files found matching '{pattern}' in directory '{data_dir}'. Returning empty lists.")
        return [], []

    df_list = []
    print(f"Loading files from: {data_dir}")
    for filepath in tqdm(all_files, desc=f"Reading files in {os.path.basename(data_dir)}", leave=False):
        try:
            temp_df = pd.read_csv(filepath, sep='\t', header=0)
            if 'tweet' not in temp_df.columns:
                # Add simple checks for common alternatives
                if 'text' in temp_df.columns:
                    temp_df.rename(columns={'text': 'tweet'}, inplace=True)
                elif 'sentence' in temp_df.columns:
                    temp_df.rename(columns={'sentence': 'tweet'}, inplace=True)
                else:
                    raise ValueError(f"Could not find text column ('tweet', 'text', 'sentence') in {filepath}")

            label = get_label_from_filename(filepath, label_separator)
            temp_df['emotion'] = label
            df_list.append(temp_df[['tweet', 'emotion']])
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            continue

    if not df_list:
        print(f"Warning: No dataframes were successfully created from files in {data_dir}. Returning empty lists.")
        return [], []

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    print(f"Loaded and combined {len(combined_df)} samples from {os.path.basename(data_dir)}. "
          f"Emotions: {combined_df['emotion'].unique().tolist()}")
    return combined_df['tweet'].tolist(), combined_df['emotion'].tolist()

def preprocess_text(text, processor, stop_words_set):
    """Cleans and processes a single string of text."""
    text = str(text).lower()  # Ensure input is string and lowercase
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words_set]
    if processor:
        if isinstance(processor, SnowballStemmer):
            processed_tokens = [processor.stem(word) for word in tokens]
        else:
            processed_tokens = tokens
    else:
        processed_tokens = tokens
    return processed_tokens

def build_vocab(processed_docs, min_freq=1, pad_token="<PAD>", unk_token="<UNK>"):
    """Builds a vocabulary mapping from words to indices, including PAD and UNK."""
    word_counts = Counter(token for doc in processed_docs for token in doc)
    print(f"Found {len(word_counts)} unique tokens before filtering.")

    vocab = [word for word, count in word_counts.items() if count >= min_freq]
    print(f"Vocabulary size after applying min_freq={min_freq}: {len(vocab)}")

    # Ensure PAD and UNK are handled correctly even if they appear in data
    final_vocab = {pad_token: 0, unk_token: 1}
    current_idx = 2
    for word in vocab:
        if word not in final_vocab:  # Avoid overwriting PAD/UNK if they are in vocab list
            final_vocab[word] = current_idx
            current_idx += 1

    idx_to_word = {idx: word for word, idx in final_vocab.items()}
    vocab_size = len(final_vocab)
    print(f"Total vocabulary size (including {pad_token}, {unk_token}): {vocab_size}")

    return final_vocab, idx_to_word, vocab_size

def convert_docs_to_ids(processed_docs, word_to_idx, unk_token="<UNK>"):
    """Converts lists of processed tokens into lists of integer IDs."""
    docs_as_ids = []
    unk_index = word_to_idx[unk_token]
    for doc in processed_docs:
        docs_as_ids.append([word_to_idx.get(token, unk_index) for token in doc])
    return docs_as_ids

def get_data_processors(use_stemming=True):
    """Create and return text processing tools."""
    stop_words_set = set(stopwords.words('english'))
    stop_words_set.update(string.punctuation)
    processor = SnowballStemmer('english') if use_stemming else None
    return stop_words_set, processor

def save_vocab_and_labels(word_to_idx, label_encoder, vocab_path, label_path):
    """Save vocabulary and label encoder to disk."""
    try:
        with open(vocab_path, 'wb') as f:
            pickle.dump(word_to_idx, f)
        with open(label_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        print(f"Successfully saved vocabulary and label encoder")
    except Exception as e:
        print(f"Error saving vocab/label encoder: {e}")

def load_vocab_and_labels(vocab_path, label_path):
    """Load vocabulary and label encoder from disk."""
    try:
        with open(vocab_path, 'rb') as f:
            word_to_idx = pickle.load(f)
        with open(label_path, 'rb') as f:
            label_encoder = pickle.load(f)
        return word_to_idx, label_encoder
    except Exception as e:
        print(f"Error loading vocab/label encoder: {e}")
        return None, None

def prepare_data_for_training(config):
    """
    Full data preparation pipeline that:
    1. Loads and processes the data
    2. Builds vocabulary
    3. Encodes labels
    4. Returns everything needed for training
    """
    # Load raw data
    raw_train, labels_train_str = load_classification_data(
        config.TRAIN_DIR, config.FILE_PATTERN, config.LABEL_SEPARATOR)
    raw_val, labels_val_str = load_classification_data(
        config.VAL_DIR, config.FILE_PATTERN, config.LABEL_SEPARATOR)
    raw_test, labels_test_str = load_classification_data(
        config.TEST_DIR, config.FILE_PATTERN, config.LABEL_SEPARATOR)

    if not raw_train or not raw_val:
        raise ValueError("Training or Validation data could not be loaded. Please check paths and files.")
    if not raw_test:
        print("Warning: Test data could not be loaded. Final evaluation will be skipped.")

    # Setup preprocessing tools
    stop_words_set, processor = get_data_processors(use_stemming=config.USE_STEMMING)

    # Process text
    print("\nProcessing training documents...")
    processed_train = [preprocess_text(doc, processor, stop_words_set) for doc in tqdm(raw_train)]
    print("Processing validation documents...")
    processed_val = [preprocess_text(doc, processor, stop_words_set) for doc in tqdm(raw_val)]
    print("Processing test documents...")
    processed_test = [preprocess_text(doc, processor, stop_words_set) for doc in tqdm(raw_test)] if raw_test else []

    # Build vocabulary
    print("\nBuilding vocabulary...")
    all_processed_docs = processed_train + processed_val + processed_test
    if not all_processed_docs:
        raise ValueError("No documents found after processing all sets. Cannot build vocabulary.")
    
    word_to_idx, idx_to_word, vocab_size = build_vocab(
        all_processed_docs, 
        config.MIN_WORD_FREQ,
        pad_token=config.PAD_TOKEN, 
        unk_token=config.UNK_TOKEN
    )
    
    # Convert text to integer sequences
    print("\nConverting documents to integer sequences...")
    train_ids = convert_docs_to_ids(processed_train, word_to_idx, unk_token=config.UNK_TOKEN)
    val_ids = convert_docs_to_ids(processed_val, word_to_idx, unk_token=config.UNK_TOKEN)
    test_ids = convert_docs_to_ids(processed_test, word_to_idx, unk_token=config.UNK_TOKEN) if processed_test else []

    # Process target labels
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

    # Save vocabulary and label encoder
    save_vocab_and_labels(
        word_to_idx, 
        label_encoder, 
        config.VOCAB_SAVE_PATH, 
        config.LABEL_ENCODER_SAVE_PATH
    )

    # Return processed data
    return {
        'train_ids': train_ids,
        'val_ids': val_ids,
        'test_ids': test_ids,
        'y_train_int': y_train_int,
        'y_val_int': y_val_int,
        'y_test_int': y_test_int,
        'label_encoder': label_encoder,
        'word_to_idx': word_to_idx,
        'idx_to_word': idx_to_word,
        'vocab_size': vocab_size,
        'num_classes': num_classes,
        'pad_index': word_to_idx[config.PAD_TOKEN]
    }   