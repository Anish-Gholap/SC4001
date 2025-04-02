"""
Dataset and DataLoader utilities for emotion classification.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class TextDataset(Dataset):
    """PyTorch dataset for text classification."""
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
        label_tensor = torch.tensor(label, dtype=torch.long)  # Use long for CrossEntropyLoss
        return sequence_tensor, label_tensor

def collate_batch(batch, pad_index):
    """Collates data samples into batches with padding."""
    label_list, sequence_list = [], []
    for (_sequence, _label) in batch:
        label_list.append(_label)
        sequence_list.append(_sequence)

    sequences_padded = pad_sequence(sequence_list, batch_first=True, padding_value=pad_index)
    labels = torch.stack(label_list)
    return sequences_padded, labels

def get_data_loaders(train_ids, y_train_int, val_ids, y_val_int, test_ids, y_test_int, 
                     batch_size, pad_index):
    """Create PyTorch DataLoaders for train, validation, and test sets."""
    # Create datasets
    train_dataset = TextDataset(train_ids, y_train_int)
    val_dataset = TextDataset(val_ids, y_val_int)
    test_dataset = TextDataset(test_ids, y_test_int) if test_ids else None

    # Use lambda to pass pad_index to collate_fn
    collate_fn_with_pad = lambda batch: collate_batch(batch, pad_index)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn_with_pad
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn_with_pad
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn_with_pad
        )

    return train_loader, val_loader, test_loader