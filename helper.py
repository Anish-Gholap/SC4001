from tqdm import tqdm # For progress bars
import glob
import os
import pandas as pd
from nltk.stem import SnowballStemmer # Or WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
LABEL_SEPARATOR = '-'           # How emotion is separated in the filename (e.g., 'anger-ratings...')



def get_label_from_filename(filename, separator=LABEL_SEPARATOR):
    base_name = os.path.basename(filename)
    label = base_name.split(separator)[0]
    return label.lower()

def load_classification_data(data_dir, pattern, label_separator):
    """Loads and combines data from multiple classification files."""
    all_files = glob.glob(os.path.join(data_dir, pattern))
    if not all_files:
        print(f"Warning: No files found matching '{pattern}' in directory '{data_dir}'. Returning empty lists.")
        return [], []
        # raise FileNotFoundError(f"No files found matching '{pattern}' in directory '{data_dir}'")

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
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle

    print(f"Loaded and combined {len(combined_df)} samples from {os.path.basename(data_dir)}. Emotions: {combined_df['emotion'].unique().tolist()}")
    return combined_df['tweet'].tolist(), combined_df['emotion'].tolist()

def preprocess_text(text, processor, stop_words_set):
    """(Same as before) Cleans and processes a single string of text."""
    text = str(text).lower() # Ensure input is string and lowercase
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words_set]
    if processor:
        if isinstance(processor, SnowballStemmer):
             processed_tokens = [processor.stem(word) for word in tokens]
        else: processed_tokens = tokens
    else: processed_tokens = tokens
    return processed_tokens

def build_vocab(processed_docs, min_freq=1):
    """Builds a vocabulary mapping from words to indices, including PAD and UNK."""
    word_counts = Counter(token for doc in processed_docs for token in doc)
    print(f"Found {len(word_counts)} unique tokens before filtering.")

    vocab = [word for word, count in word_counts.items() if count >= min_freq]
    print(f"Vocabulary size after applying min_freq={min_freq}: {len(vocab)}")

    # Ensure PAD and UNK are handled correctly even if they appear in data
    final_vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    current_idx = 2
    for word in vocab:
        if word not in final_vocab: # Avoid overwriting PAD/UNK if they are in vocab list
            final_vocab[word] = current_idx
            current_idx += 1

    idx_to_word = {idx: word for word, idx in final_vocab.items()}
    vocab_size = len(final_vocab)
    print(f"Total vocabulary size (including {PAD_TOKEN}, {UNK_TOKEN}): {vocab_size}")

    return final_vocab, idx_to_word, vocab_size

def convert_docs_to_ids(processed_docs, word_to_idx):
    """(Same as before) Converts lists of processed tokens into lists of integer IDs."""
    docs_as_ids = []
    unk_index = word_to_idx[UNK_TOKEN]
    for doc in processed_docs: # No tqdm here, usually fast enough
        docs_as_ids.append([word_to_idx.get(token, unk_index) for token in doc])
    return docs_as_ids


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """Plots a confusion matrix using seaborn."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()