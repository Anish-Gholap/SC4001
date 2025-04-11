from model_pipeline import BaseModelPipeline
import torch
import torch.nn as nn
from transformers import BertModel
import argparse


# BERT-CNN-BiLSTM model implementation as per the paper
class BertCnnBiLstm(nn.Module):
    def __init__(self, bert_model_name, num_classes=2, dropout_rate=0.5, freeze_bert=False):
        super(BertCnnBiLstm, self).__init__()

        # BERT layer
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size

        # Freeze BERT if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # CNN layer
        self.cnn_filters = 256
        self.kernel_sizes = [3, 4, 5]
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.bert_hidden_size,
                out_channels=self.cnn_filters,
                kernel_size=k
            ) for k in self.kernel_sizes
        ])

        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=self.cnn_filters * len(self.kernel_sizes),
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer
        self.fc = nn.Linear(128 * 2, num_classes)  # *2 for bidirectional

        # Activation for output
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        # BERT layer
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # CNN layer - need to transpose for CNN
        x_reshaped = sequence_output.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]

        # Apply CNN with different kernel sizes and concatenate
        cnn_outputs = []
        for conv in self.convs:
            # Apply convolution and ReLU
            conv_out = torch.relu(conv(x_reshaped))  # [batch_size, cnn_filters, seq_len - kernel_size + 1]

            # Apply max pooling over time
            pooled = nn.functional.max_pool1d(
                conv_out,
                kernel_size=conv_out.size(2)
            ).squeeze(2)  # [batch_size, cnn_filters]

            cnn_outputs.append(pooled)

        # Concatenate CNN outputs
        cnn_output = torch.cat(cnn_outputs, dim=1)  # [batch_size, cnn_filters * len(kernel_sizes)]

        # Reshape for BiLSTM input
        lstm_input = cnn_output.unsqueeze(1)  # [batch_size, 1, cnn_filters * len(kernel_sizes)]

        # BiLSTM layer
        lstm_output, _ = self.bilstm(lstm_input)  # [batch_size, 1, hidden_size * 2]
        lstm_output = lstm_output.squeeze(1)  # [batch_size, hidden_size * 2]

        # Dropout
        x = self.dropout(lstm_output)

        # Fully connected layer
        logits = self.fc(x)

        return logits


class BertCnnBiLstmPipeline(BaseModelPipeline):
    """Pipeline for training and evaluating the BERT-CNN-BiLSTM model."""

    def __init__(self, args):
        super().__init__(args)

    def build_model(self):
        """Build and return the BERT-CNN-BiLSTM model."""
        model = BertCnnBiLstm(
            bert_model_name=self.args.bert_model,
            num_classes=self.num_classes,
            dropout_rate=self.args.dropout_rate,
            freeze_bert=self.args.freeze_bert if hasattr(self.args, 'freeze_bert') else False
        )
        return model

    # Note: We don't need to override train_epoch or evaluate methods
    # since the default implementations in BaseModelPipeline work for this model


def main():
    parser = argparse.ArgumentParser(description='BERT-CNN-BiLSTM for Text Classification')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV dataset')
    parser.add_argument('--text_column', type=str, default='text', help='Column name for text data')
    parser.add_argument('--label_column', type=str, default='label', help='Column name for labels')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased', help='BERT model to use')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--freeze_bert', action='store_true', help='Freeze BERT parameters')
    args = parser.parse_args()

    # Create and run the pipeline
    pipeline = BertCnnBiLstmPipeline(args)
    pipeline.train()


if __name__ == "__main__":
    main()