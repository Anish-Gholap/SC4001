from model_pipeline import BaseModelPipeline
import torch
import torch.nn as nn
from transformers import BertModel
import argparse


class BERT_Arch(nn.Module):
    def __init__(self, bert_model_name, num_classes, max_length):
        super(BERT_Arch, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.conv = nn.Conv2d(in_channels=13, out_channels=13, kernel_size=(3, 768), padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.dropout = nn.Dropout(0.1)

        # For max_length=128, after convolution and pooling: 13 × 126 × 1 = 1,638 features
        # For max_length=512, feature size is 13 * (512-2) * 1 = 6,630
        self.feature_size = 13 * (max_length - 2) * 1

        self.fc = nn.Linear(self.feature_size, num_classes)
        self.flat = nn.Flatten()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask):
        # Get all BERT hidden states - this returns a tuple where the third element contains all hidden states
        outputs = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        all_layers = outputs.hidden_states  # This gives a tuple of tensors (13 layers)
        # Stack all layers: convert tuple of tensors to a single tensor
        x = torch.stack(all_layers, dim=1)  # [batch_size, 13, seq_len, 768]
        # Apply convolution
        x = self.conv(self.dropout(x))
        # Apply ReLU and pooling
        x = self.pool(self.dropout(self.relu(x)))
        # Flatten and apply final linear layer
        x = self.fc(self.dropout(self.flat(self.dropout(x))))
        # Apply log softmax
        return self.softmax(x)


class BertCNNPipeline(BaseModelPipeline):
    """Pipeline for training and evaluating the BERT with convolutional layers model."""

    def __init__(self, args):
        super().__init__(args)

    def build_model(self):
        """Build and return the BERT-Conv model."""
        model = BERT_Arch(
            bert_model_name=self.args.bert_model,
            num_classes=self.num_classes,
            max_length=self.args.max_length
        )
        return model

    # This model uses the same train_epoch and evaluate methods as the base model,
    # so we don't need to override them


def main():
    parser = argparse.ArgumentParser(description='BERT with Conv layers for Text Classification')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV dataset')
    parser.add_argument('--text_column', type=str, default='text', help='Column name for text data')
    parser.add_argument('--label_column', type=str, default='label', help='Column name for labels')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased', help='BERT model to use')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./results_bert_conv', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Create and run the pipeline
    pipeline = BertCNNPipeline(args)
    pipeline.train()


if __name__ == "__main__":
    main()