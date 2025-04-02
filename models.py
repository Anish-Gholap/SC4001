import torch.nn as nn
import torch
import torch.nn.functional as F

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
                 dropout=0.5, pad_idx=0):
        super().__init__()
        
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