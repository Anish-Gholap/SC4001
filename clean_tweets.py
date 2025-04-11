import glob
import os
import re
import string
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Adjust these paths as needed (absolute paths recommended if relative paths fail)
TRAIN_DIR = r'C:\Users\Anish\Desktop\Projects\School\SC4001_Project\datasets\WASSA-2017\train'
VAL_DIR = r'C:\Users\Anish\Desktop\Projects\School\SC4001_Project\datasets\WASSA-2017\val'  # <-- Make sure you actually have a "val" directory
TEST_DIR = r'C:\Users\Anish\Desktop\Projects\School\SC4001_Project\datasets\WASSA-2017\test'

FILE_PATTERN = '*.txt'
LABEL_SEPARATOR = '-'


def get_label_from_filename(filename, separator=LABEL_SEPARATOR):
    """Extract label from filename based on the given separator."""
    base_name = os.path.basename(filename)
    label = base_name.split(separator)[0]
    return label.lower()


def load_data(data_dir, pattern, separator):
    """Load and combine all txt files in a given directory into a single DataFrame."""
    all_files = glob.glob(os.path.join(data_dir, pattern))
    if not all_files:
        raise FileNotFoundError(f"No files found matching '{pattern}' in directory {data_dir}")

    df_list = []
    print(f"Loading files from {data_dir}")
    for filepath in tqdm(all_files, desc="Reading files"):
        try:
            # Reading tab-separated file
            temp_df = pd.read_csv(filepath, sep='\t', header=0)
            # Derive the label from filename
            label = get_label_from_filename(filepath, separator)
            # Add the 'emotion' column
            temp_df['emotion'] = label
            # Keep only tweet and emotion
            df_list.append(temp_df[['tweet', 'emotion']])
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            continue

    if not df_list:
        raise ValueError(f"No dataframes were created from files in {data_dir}")

    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded and combined {len(combined_df)} samples")
    print(f"Found emotions: {combined_df['emotion'].unique().tolist()}")
    return combined_df


def clean_tweets(df, text_column='tweet'):
    """Clean tweets in-place and return the DataFrame."""
    # (1) Convert to lowercase
    df[text_column] = df[text_column].str.lower()

    # (2) Remove user mentions
    df[text_column] = df[text_column].apply(lambda x: re.sub(r'@\w+', '', str(x)))

    # (3) Remove URLs
    df[text_column] = df[text_column].apply(lambda x: re.sub(r'http\S+', '', str(x)))

    # (4) Remove emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols & Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols & Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251"  # Enclosed characters
                               "]+", flags=re.UNICODE)
    df[text_column] = df[text_column].apply(lambda x: emoji_pattern.sub(r'', str(x)))

    # (5) Remove hashtags
    df[text_column] = df[text_column].apply(lambda x: re.sub(r'#\w+', '', str(x)))

    # (6) Remove RT
    df[text_column] = df[text_column].apply(lambda x: re.sub(r'^RT\s+', '', str(x)))

    # (7) Remove punctuation
    df[text_column] = df[text_column].apply(lambda x: re.sub(f"[{re.escape(string.punctuation)}]", '', str(x)))

    # (8) Remove extra whitespace
    df[text_column] = df[text_column].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())

    return df


# 1. Load each split
train_df = load_data(data_dir=TRAIN_DIR, pattern=FILE_PATTERN, separator=LABEL_SEPARATOR)
val_df = load_data(data_dir=VAL_DIR, pattern=FILE_PATTERN, separator=LABEL_SEPARATOR)
test_df = load_data(data_dir=TEST_DIR, pattern=FILE_PATTERN, separator=LABEL_SEPARATOR)

# 2. Clean each split
train_df = clean_tweets(train_df, text_column="tweet")
val_df = clean_tweets(val_df, text_column="tweet")
test_df = clean_tweets(test_df, text_column="tweet")

# 3. Rename columns so final CSV is exactly: [text, labels]
train_df.rename(columns={'tweet': 'text', 'emotion': 'labels'}, inplace=True)
val_df.rename(columns={'tweet': 'text', 'emotion': 'labels'}, inplace=True)
test_df.rename(columns={'tweet': 'text', 'emotion': 'labels'}, inplace=True)

# 4. Concatenate them
combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# 5. Keep ONLY the two columns you want
combined_df = combined_df[['text', 'labels']]

# 6. Encode Labels
# 6.1 Create a LabelEncoder instance
label_encoder = LabelEncoder()

# 6.2 Fit the encoder on the 'labels' column, then transform
combined_df['labels'] = label_encoder.fit_transform(combined_df['labels'])

# 6.3 Print out the mapping from label string -> integer
print("Label Mapping:")
for idx, class_name in enumerate(label_encoder.classes_):
    print(f"{class_name} => {idx}")


# 7. Save to CSV
output_csv = 'wassa_combined_clean.csv'
combined_df.to_csv(output_csv, index=False)
print(f"Saved combined dataframe with columns [text, labels] to {output_csv}")

