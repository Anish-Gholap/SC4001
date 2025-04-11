import argparse
from bert_only_pipeline import BertOnlyPipeline
from bert_cnn_bilstm_pipeline import BertCnnBiLstmPipeline
from bert_cnn_pipeline import BertCNNPipeline
from cnn_rnn_pipeline import CNNPipeline, RNNPipeline


def main():
    parser = argparse.ArgumentParser(description='Text Classification Pipeline')

    # Model selection
    parser.add_argument('--model', type=str, required=True,
                        choices=['bert_cnn_bilstm', 'bert', 'bert_cnn', 'cnn', 'rnn'],
                        help='Model architecture to use')

    # Common arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV dataset')
    parser.add_argument('--text_column', type=str, default='text', help='Column name for text data')
    parser.add_argument('--label_column', type=str, default='label', help='Column name for labels')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')

    # BERT-specific arguments
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased', help='BERT model to use')
    parser.add_argument('--freeze_bert', action='store_true', help='Freeze BERT parameters (for BERT-CNN-BiLSTM)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of steps to accumulate gradients (for BERT-Only)')

    # CNN-specific arguments
    parser.add_argument('--min_freq', type=int, default=2, help='Minimum frequency for vocabulary (for CNN/RNN)')
    parser.add_argument('--embedding_dim', type=int, default=300, help='Dimension of word embeddings (for CNN/RNN)')
    parser.add_argument('--num_filters', type=int, default=100, help='Number of filters in CNN')
    parser.add_argument('--filter_sizes', type=str, default='[3, 4, 5]', help='Filter sizes for CNN')
    parser.add_argument('--use_pretrained_embeddings', action='store_true', help='Use pretrained GloVe embeddings')

    # RNN-specific arguments
    parser.add_argument('--hidden_dim', type=int, default=256, help='Dimension of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of RNN layers')
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional RNN')

    args = parser.parse_args()

    # Create and run the appropriate pipeline
    if args.model == 'bert_cnn_bilstm':
        # Update output directory to include model name
        args.output_dir = f"{args.output_dir}/bert_cnn_bilstm"
        pipeline = BertCnnBiLstmPipeline(args)
    elif args.model == 'bert':
        # Update output directory to include model name
        args.output_dir = f"{args.output_dir}/bert"
        # Use a smaller dropout for BERT model if not explicitly set
        if args.dropout_rate == 0.5 and not any('--dropout_rate' in arg for arg in parser._option_string_actions):
            args.dropout_rate = 0.1
        pipeline = BertOnlyPipeline(args)
    elif args.model == 'bert_cnn':
        # Update output directory to include model name
        args.output_dir = f"{args.output_dir}/bert_cnn"
        pipeline = BertCNNPipeline(args)
    elif args.model == 'cnn':
        # Update output directory to include model name
        args.output_dir = f"{args.output_dir}/cnn"
        # If using CNN, adjust the learning rate if not explicitly set
        if args.learning_rate == 2e-5 and not any('--learning_rate' in arg for arg in parser._option_string_actions):
            args.learning_rate = 0.001
        pipeline = CNNPipeline(args)
    elif args.model == 'rnn':
        # Update output directory to include model name
        args.output_dir = f"{args.output_dir}/rnn"
        # If using RNN, adjust the learning rate if not explicitly set
        if args.learning_rate == 2e-5 and not any('--learning_rate' in arg for arg in parser._option_string_actions):
            args.learning_rate = 0.001
        pipeline = RNNPipeline(args)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Train the selected model
    model, accuracy = pipeline.train()
    print(f"Final accuracy with {args.model}: {accuracy:.4f}")


if __name__ == "__main__":
    main()