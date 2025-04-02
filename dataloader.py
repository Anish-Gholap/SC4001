import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


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