import torch.nn as nn
import torch

class RNNLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, n_layers=1, dropout=0.5, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        # Using LSTM as it's generally better for text than basic RNN
        self.rnn = nn.LSTM(embed_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=True, # Use bidirectional LSTM
                           dropout=dropout if n_layers > 1 else 0, # Dropout only between LSTM layers
                           batch_first=True)
        # Linear layer input size is hidden_dim * 2 because bidirectional
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout) # Apply dropout before final layer

    def forward(self, text_batch):
        # text_batch shape: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(text_batch))
        # embedded shape: (batch_size, seq_len, embed_dim)

        # packed_output shape: (batch_size, seq_len, hidden_dim * num_directions)
        # hidden shape: (n_layers * num_directions, batch_size, hidden_dim)
        # cell shape: (n_layers * num_directions, batch_size, hidden_dim)
        outputs, (hidden, cell) = self.rnn(embedded)

        # Concatenate the final forward and backward hidden states
        # hidden shape is (num_layers * 2, batch_size, hidden_dim)
        # Get hidden state from last layer: hidden[-2,:,:] (final forward) and hidden[-1,:,:] (final backward)
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        # hidden_cat shape: (batch_size, hidden_dim * 2)

        # Apply dropout and pass through linear layer
        return self.fc(self.dropout(hidden_cat))

class SimpleRNNClassifier(nn.Module):
    """
    A simple RNN classifier without LSTM or explicit dropout layers.
    Outputs raw logits suitable for nn.CrossEntropyLoss.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, n_layers=1, pad_idx=0):
        super().__init__()

        # Define layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Simple RNN layer
        # batch_first=True makes input/output tensors shaped (batch, seq, feature)
        self.rnn = nn.RNN(input_size=embed_dim,
                          hidden_size=hidden_dim,
                          num_layers=n_layers,
                          nonlinearity='tanh', # Default for nn.RNN, 'relu' is also possible
                          batch_first=True)
                          # Note: dropout parameter is omitted here

        # Fully connected layer mapping the last hidden state to class scores (logits)
        self.fc = nn.Linear(hidden_dim, num_classes)

        # No dropout layer defined as requested

    def forward(self, text_batch):
        # text_batch shape: (batch_size, seq_len)
        embedded = self.embedding(text_batch)
        # embedded shape: (batch_size, seq_len, embed_dim)

        # Initialize hidden state with zeros (handled by default if h_0 is None)
        # output shape: (batch_size, seq_len, hidden_dim)
        # hidden shape: (n_layers * num_directions=1, batch_size, hidden_dim)
        output, hidden = self.rnn(embedded) # No initial hidden state provided, defaults to zeros

        # We use the hidden state from the last layer after processing the entire sequence.
        # For nn.RNN, hidden contains the final hidden state for each layer.
        # We want the hidden state of the last layer, so we index with [-1].
        # hidden[-1] shape: (batch_size, hidden_dim)
        last_hidden_state = hidden[-1]

        # Pass the last hidden state through the linear layer to get logits
        # logits shape: (batch_size, num_classes)
        logits = self.fc(last_hidden_state)

        return logits