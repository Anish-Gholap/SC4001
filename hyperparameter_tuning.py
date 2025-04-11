import argparse
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
import itertools
from tqdm import tqdm

# Import your model and dataset classes
from cnn_lstm_pipeline import CNNLSTMClassifier, TextClassificationDataset, CNNLSTMPipeline

# Define additional model architectures


# Simple CNN model for text classification
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, n_filters=100,
                 filter_sizes=[3, 4, 5], dropout=0.5, pad_idx=0,
                 pretrained_embeddings=None):
        super().__init__()

        # Initialize embedding layer
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, padding_idx=pad_idx, freeze=False
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # CNN layers
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=n_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        # Calculate the output dimension after concatenating all filter outputs
        self.fc_input_dim = n_filters * len(filter_sizes)

        # Fully connected layer for classification
        self.fc = nn.Linear(self.fc_input_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_batch):
        # text_batch shape: [batch_size, seq_len]

        # Embedding layer: [batch_size, seq_len, embed_dim]
        embedded = self.embedding(text_batch)

        # Transpose for CNN: [batch_size, embed_dim, seq_len]
        embedded_conv = embedded.permute(0, 2, 1)

        # Apply convolutions and activation
        # Each conv_x shape: [batch_size, n_filters, seq_len - filter_size + 1]
        conv_outputs = [F.relu(conv(embedded_conv)) for conv in self.convs]

        # Apply max pooling
        # Each pooled_x shape: [batch_size, n_filters, 1]
        pooled_outputs = [F.max_pool1d(conv_x, conv_x.shape[2]) for conv_x in conv_outputs]

        # Concatenate pooled outputs
        # cat shape: [batch_size, n_filters * len(filter_sizes), 1]
        cat = torch.cat(pooled_outputs, dim=1)

        # Remove the last dimension and apply dropout
        # cat shape: [batch_size, n_filters * len(filter_sizes)]
        cat = cat.squeeze(-1)
        cat = self.dropout(cat)

        # Final classification layer
        # output shape: [batch_size, num_classes]
        output = self.fc(cat)

        return output


# Simple RNN model for text classification
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 rnn_type='lstm', rnn_layers=1, bidirectional=True,
                 dropout=0.5, pad_idx=0, pretrained_embeddings=None):
        super().__init__()

        # Initialize embedding layer
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, padding_idx=pad_idx, freeze=False
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Select RNN type
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=rnn_layers,
                bidirectional=bidirectional,
                dropout=dropout if rnn_layers > 1 else 0,
                batch_first=True
            )
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=rnn_layers,
                bidirectional=bidirectional,
                dropout=dropout if rnn_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

        # Account for bidirectional RNN
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Fully connected layer for classification
        self.fc = nn.Linear(fc_input_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_batch):
        # text_batch shape: [batch_size, seq_len]

        # Embedding layer: [batch_size, seq_len, embed_dim]
        embedded = self.embedding(text_batch)
        embedded = self.dropout(embedded)

        # RNN forward pass
        # output shape: [batch_size, seq_len, hidden_dim * num_directions]
        # hidden shape: [num_layers * num_directions, batch_size, hidden_dim]
        if isinstance(self.rnn, nn.LSTM):
            output, (hidden, cell) = self.rnn(embedded)
        else:  # GRU
            output, hidden = self.rnn(embedded)

        # Get the final hidden state for bidirectional RNN
        # For bidirectional, concatenate the last forward and backward hidden states
        if self.rnn.bidirectional:
            # hidden[-2,:,:] is the last forward direction hidden state
            # hidden[-1,:,:] is the last backward direction hidden state
            hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden_cat = hidden[-1, :, :]

        # Apply dropout and fully connected layer
        # output shape: [batch_size, num_classes]
        output = self.fc(self.dropout(hidden_cat))

        return output


class MultiModelHyperparameterTuner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create output directory
        os.makedirs(self.args.output_dir, exist_ok=True)

        # Set up the base pipeline to prepare data
        self.base_pipeline = CNNLSTMPipeline(args)

        # Use the pipeline to load and preprocess data
        self.data_df = pd.read_csv(args.data_path)

        # Convert string labels to integers if needed
        if isinstance(self.data_df[args.label_column].iloc[0], str):
            self.label_map = {label: idx for idx, label in enumerate(self.data_df[args.label_column].unique())}
            self.data_df[args.label_column] = self.data_df[args.label_column].map(self.label_map)

            # Save label mapping
            with open(os.path.join(args.output_dir, 'label_map.json'), 'w') as f:
                json.dump(self.label_map, f)

        # Build vocabulary using all data
        all_texts = self.data_df[args.text_column].values
        print("Building vocabulary...")
        self.base_pipeline.build_vocab(all_texts, min_freq=args.min_freq)
        self.vocab = self.base_pipeline.vocab

        # Set number of classes
        self.num_classes = len(self.data_df[args.label_column].unique())

        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Number of classes: {self.num_classes}")

        # Create the full dataset (without splitting)
        self.dataset = TextClassificationDataset(
            texts=self.data_df[args.text_column].values,
            labels=self.data_df[args.label_column].values,
            vocab=self.vocab,
            max_length=args.max_length
        )

        # Prepare hyperparameter grids for each model
        self.prepare_hyperparameter_grids()

    def prepare_hyperparameter_grids(self):
        """Define hyperparameter grids for each model type."""
        # Common hyperparameters for all models
        common_params = {
            'batch_size': self.args.batch_sizes,
            'learning_rate': self.args.learning_rates,
            'embedding_dim': self.args.embedding_dims,
            'dropout_rate': self.args.dropout_rates,
        }

        # CNN-specific hyperparameters
        self.cnn_param_grid = {
            **common_params,
            'num_filters': self.args.num_filters_list,
            'filter_sizes': self.args.filter_sizes_list,
        }

        # RNN-specific hyperparameters
        self.rnn_param_grid = {
            **common_params,
            'hidden_dim': self.args.hidden_dims,
            'rnn_type': self.args.rnn_types,
            'rnn_layers': self.args.rnn_layers_list,
            'bidirectional': self.args.bidirectional_list,
        }

        # CNN-LSTM-specific hyperparameters
        self.cnn_lstm_param_grid = {
            **common_params,
            'hidden_dim': self.args.hidden_dims,
            'num_filters': self.args.num_filters_list,
            'filter_sizes': self.args.filter_sizes_list,
            'lstm_layers': self.args.lstm_layers_list,
        }

        # Create parameter combinations for each model type
        self.cnn_combinations = list(itertools.product(
            self.cnn_param_grid['batch_size'],
            self.cnn_param_grid['learning_rate'],
            self.cnn_param_grid['embedding_dim'],
            self.cnn_param_grid['num_filters'],
            self.cnn_param_grid['filter_sizes'],
            self.cnn_param_grid['dropout_rate']
        ))

        self.rnn_combinations = list(itertools.product(
            self.rnn_param_grid['batch_size'],
            self.rnn_param_grid['learning_rate'],
            self.rnn_param_grid['embedding_dim'],
            self.rnn_param_grid['hidden_dim'],
            self.rnn_param_grid['rnn_type'],
            self.rnn_param_grid['rnn_layers'],
            self.rnn_param_grid['bidirectional'],
            self.rnn_param_grid['dropout_rate']
        ))

        self.cnn_lstm_combinations = list(itertools.product(
            self.cnn_lstm_param_grid['batch_size'],
            self.cnn_lstm_param_grid['learning_rate'],
            self.cnn_lstm_param_grid['embedding_dim'],
            self.cnn_lstm_param_grid['hidden_dim'],
            self.cnn_lstm_param_grid['num_filters'],
            self.cnn_lstm_param_grid['filter_sizes'],
            self.cnn_lstm_param_grid['lstm_layers'],
            self.cnn_lstm_param_grid['dropout_rate']
        ))

        print(f"CNN parameter combinations: {len(self.cnn_combinations)}")
        print(f"RNN parameter combinations: {len(self.rnn_combinations)}")
        print(f"CNN-LSTM parameter combinations: {len(self.cnn_lstm_combinations)}")

    def create_cnn_model(self, embedding_dim, num_filters, filter_sizes, dropout_rate):
        """Create a CNN model instance with the given hyperparameters."""
        model = CNNClassifier(
            vocab_size=len(self.vocab),
            embed_dim=embedding_dim,
            num_classes=self.num_classes,
            n_filters=num_filters,
            filter_sizes=filter_sizes,
            dropout=dropout_rate
        )
        return model

    def create_rnn_model(self, embedding_dim, hidden_dim, rnn_type, rnn_layers, bidirectional, dropout_rate):
        """Create an RNN model instance with the given hyperparameters."""
        model = RNNClassifier(
            vocab_size=len(self.vocab),
            embed_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=self.num_classes,
            rnn_type=rnn_type,
            rnn_layers=rnn_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate
        )
        return model

    def create_cnn_lstm_model(self, embedding_dim, hidden_dim, num_filters, filter_sizes, lstm_layers, dropout_rate):
        """Create a CNN-LSTM model instance with the given hyperparameters."""
        model = CNNLSTMClassifier(
            vocab_size=len(self.vocab),
            embed_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=self.num_classes,
            n_filters=num_filters,
            filter_sizes=filter_sizes,
            lstm_layers=lstm_layers,
            dropout=dropout_rate
        )
        return model

    def train_model(self, model, train_loader, val_loader, learning_rate, epochs=3):
        """Train the model with given parameters and return validation metrics."""
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Simple step scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader), gamma=0.95)

        best_val_accuracy = 0
        best_epoch = 0

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_losses = []

            for batch in train_loader:
                optimizer.zero_grad()

                text_indices = batch['text_indices'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                outputs = model(text_indices)

                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(outputs, labels)
                train_losses.append(loss.item())

                # Backward pass and optimization
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            # Validation phase
            model.eval()
            val_losses = []
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    text_indices = batch['text_indices'].to(self.device)
                    labels = batch['label'].to(self.device)

                    # Forward pass
                    outputs = model(text_indices)

                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(outputs, labels)
                    val_losses.append(loss.item())

                    _, preds = torch.max(outputs, dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_accuracy = val_correct / val_total

            # Update best validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch

            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Train Loss: {np.mean(train_losses):.4f}, "
                  f"Val Loss: {np.mean(val_losses):.4f}, "
                  f"Val Accuracy: {val_accuracy:.4f}")

        return best_val_accuracy

    def run_cross_validation_for_model_type(self, model_type):
        """Run cross-validation for a specific model type."""
        print(f"\n===== Running CV for {model_type} =====")

        # Select parameter combinations based on model type
        if model_type == "cnn":
            param_combinations = self.cnn_combinations
            create_model_fn = self.create_cnn_model
            param_names = ["batch_size", "learning_rate", "embedding_dim", "num_filters", "filter_sizes",
                           "dropout_rate"]
        elif model_type == "rnn":
            param_combinations = self.rnn_combinations
            create_model_fn = self.create_rnn_model
            param_names = ["batch_size", "learning_rate", "embedding_dim", "hidden_dim", "rnn_type", "rnn_layers",
                           "bidirectional", "dropout_rate"]
        elif model_type == "cnn_lstm":
            param_combinations = self.cnn_lstm_combinations
            create_model_fn = self.create_cnn_lstm_model
            param_names = ["batch_size", "learning_rate", "embedding_dim", "hidden_dim", "num_filters", "filter_sizes",
                           "lstm_layers", "dropout_rate"]
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        kfold = KFold(n_splits=self.args.cv_folds, shuffle=True, random_state=self.args.seed)
        results = []

        # Iterate through parameter combinations
        for i, params in enumerate(tqdm(param_combinations, desc=f"{model_type} combinations")):
            # Print current hyperparameter combination
            print(f"\nCombination {i + 1}/{len(param_combinations)}:")
            param_dict = dict(zip(param_names, params))
            for name, value in param_dict.items():
                print(f"  {name}: {value}")

            fold_accuracies = []

            # Run K-fold cross-validation
            for fold, (train_idx, val_idx) in enumerate(kfold.split(self.dataset)):
                print(f"\nFold {fold + 1}/{self.args.cv_folds}")

                # Create train and validation subsets
                train_subset = Subset(self.dataset, train_idx)
                val_subset = Subset(self.dataset, val_idx)

                # Create data loaders
                train_loader = DataLoader(
                    train_subset,
                    batch_size=param_dict["batch_size"],
                    shuffle=True
                )

                val_loader = DataLoader(
                    val_subset,
                    batch_size=param_dict["batch_size"]
                )

                # Create and train model based on model type
                if model_type == "cnn":
                    model = create_model_fn(
                        embedding_dim=param_dict["embedding_dim"],
                        num_filters=param_dict["num_filters"],
                        filter_sizes=param_dict["filter_sizes"],
                        dropout_rate=param_dict["dropout_rate"]
                    )
                elif model_type == "rnn":
                    model = create_model_fn(
                        embedding_dim=param_dict["embedding_dim"],
                        hidden_dim=param_dict["hidden_dim"],
                        rnn_type=param_dict["rnn_type"],
                        rnn_layers=param_dict["rnn_layers"],
                        bidirectional=param_dict["bidirectional"],
                        dropout_rate=param_dict["dropout_rate"]
                    )
                elif model_type == "cnn_lstm":
                    model = create_model_fn(
                        embedding_dim=param_dict["embedding_dim"],
                        hidden_dim=param_dict["hidden_dim"],
                        num_filters=param_dict["num_filters"],
                        filter_sizes=param_dict["filter_sizes"],
                        lstm_layers=param_dict["lstm_layers"],
                        dropout_rate=param_dict["dropout_rate"]
                    )

                # Train the model and get validation accuracy
                val_accuracy = self.train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    learning_rate=param_dict["learning_rate"],
                    epochs=self.args.cv_epochs
                )

                fold_accuracies.append(val_accuracy)

                # Free up memory
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Calculate average accuracy across folds
            avg_accuracy = np.mean(fold_accuracies)
            std_accuracy = np.std(fold_accuracies)

            print(f"\nAverage accuracy for {model_type} combination: {avg_accuracy:.4f} (±{std_accuracy:.4f})")

            # Save results for this combination
            result = {
                **param_dict,
                'model_type': model_type,
                'avg_accuracy': avg_accuracy,
                'std_accuracy': std_accuracy,
                'fold_accuracies': fold_accuracies
            }

            results.append(result)

            # Save intermediate results
            self.save_results(results, model_type)

        # Find best hyperparameters for this model type
        self.find_best_hyperparameters(results, model_type)

        return results

    def run_all_models(self):
        """Run cross-validation for all model types."""
        all_results = []

        # Run CV for each model type if specified
        if self.args.run_cnn:
            cnn_results = self.run_cross_validation_for_model_type("cnn")
            all_results.extend(cnn_results)

        if self.args.run_rnn:
            rnn_results = self.run_cross_validation_for_model_type("rnn")
            all_results.extend(rnn_results)

        if self.args.run_cnn_lstm:
            cnn_lstm_results = self.run_cross_validation_for_model_type("cnn_lstm")
            all_results.extend(cnn_lstm_results)

        # Save all results together
        self.save_results(all_results, "all_models")

        # Compare performance across model types
        self.compare_model_types(all_results)

    def save_results(self, results, model_type):
        """Save current results to a JSON file."""
        results_path = os.path.join(self.args.output_dir, f'{model_type}_cv_results.json')

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    def find_best_hyperparameters(self, results, model_type):
        """Find and print the best hyperparameters based on average accuracy."""
        # Sort results by average accuracy
        sorted_results = sorted(results, key=lambda x: x['avg_accuracy'], reverse=True)

        # Print top N results
        print(f"\n===== Top 5 {model_type.upper()} Hyperparameter Combinations =====")
        for i, result in enumerate(sorted_results[:5]):
            print(f"\nRank {i + 1}:")
            print(f"  Average Accuracy: {result['avg_accuracy']:.4f} (±{result['std_accuracy']:.4f})")

            # Print all hyperparameters
            for param, value in result.items():
                if param not in ['avg_accuracy', 'std_accuracy', 'fold_accuracies', 'model_type']:
                    print(f"  {param}: {value}")

        # Save best hyperparameters to a separate file
        best_params = sorted_results[0]
        best_params_path = os.path.join(self.args.output_dir, f'{model_type}_best_hyperparameters.json')

        with open(best_params_path, 'w') as f:
            json.dump(best_params, f, indent=2)

        print(f"\nBest hyperparameters for {model_type} saved to {best_params_path}")

    def compare_model_types(self, all_results):
        """Compare performance across different model types."""
        # Group results by model type
        model_types = set(result['model_type'] for result in all_results)

        best_by_model = {}
        for model_type in model_types:
            # Get results for this model type
            model_results = [r for r in all_results if r['model_type'] == model_type]

            # Sort by average accuracy
            sorted_results = sorted(model_results, key=lambda x: x['avg_accuracy'], reverse=True)

            # Store best result for this model type
            best_by_model[model_type] = sorted_results[0]

        # Print comparison of best results across model types
        print("\n===== Best Performance By Model Type =====")

        # Sort model types by best performance
        sorted_models = sorted(best_by_model.items(), key=lambda x: x[1]['avg_accuracy'], reverse=True)

        for i, (model_type, best_result) in enumerate(sorted_models):
            print(f"\n{i + 1}. {model_type.upper()}")
            print(f"   Best Accuracy: {best_result['avg_accuracy']:.4f} (±{best_result['std_accuracy']:.4f})")

            # Print key hyperparameters
            if model_type == "cnn":
                print(f"   Key Hyperparameters: lr={best_result['learning_rate']}, "
                      f"filters={best_result['num_filters']}, filter_sizes={best_result['filter_sizes']}")
            elif model_type == "rnn":
                print(f"   Key Hyperparameters: lr={best_result['learning_rate']}, "
                      f"hidden_dim={best_result['hidden_dim']}, type={best_result['rnn_type']}, "
                      f"layers={best_result['rnn_layers']}, bidirectional={best_result['bidirectional']}")
            elif model_type == "cnn_lstm":
                print(f"   Key Hyperparameters: lr={best_result['learning_rate']}, "
                      f"hidden_dim={best_result['hidden_dim']}, filters={best_result['num_filters']}, "
                      f"lstm_layers={best_result['lstm_layers']}")

        # Save overall comparison to a file
        comparison = {
            'best_by_model': best_by_model,
            'overall_best': sorted_models[0][1],
            'best_model_type': sorted_models[0][0]
        }

        comparison_path = os.path.join(self.args.output_dir, 'model_comparison.json')
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)

        print(f"\nModel comparison saved to {comparison_path}")
        print(
            f"\nOverall best model: {sorted_models[0][0].upper()} with accuracy {sorted_models[0][1]['avg_accuracy']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Multi-Model Hyperparameter Tuning for Text Classification')

    # Data parameters
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV dataset')
    parser.add_argument('--text_column', type=str, default='text', help='Column name for text data')
    parser.add_argument('--label_column', type=str, default='label', help='Column name for labels')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum sequence length')
    parser.add_argument('--min_freq', type=int, default=2, help='Minimum frequency for vocabulary')

    # CV parameters
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--cv_epochs', type=int, default=3, help='Number of epochs per CV fold')
    parser.add_argument('--output_dir', type=str, default='./cv_results', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Model selection flags
    parser.add_argument('--run_cnn', action='store_true', help='Run CV for CNN model')
    parser.add_argument('--run_rnn', action='store_true', help='Run CV for RNN model')
    parser.add_argument('--run_cnn_lstm', action='store_true', help='Run CV for CNN-LSTM model')

    # Common hyperparameter search space
    parser.add_argument('--batch_sizes', nargs='+', type=int, default=[16, 32, 64], help='Batch sizes to try')
    parser.add_argument('--learning_rates', nargs='+', type=float, default=[0.001, 0.0005, 0.0001],
                        help='Learning rates to try')
    parser.add_argument('--embedding_dims', nargs='+', type=int, default=[100, 200, 300],
                        help='Embedding dimensions to try')
    parser.add_argument('--dropout_rates', nargs='+', type=float, default=[0.3, 0.5, 0.7], help='Dropout rates to try')

    # CNN-specific hyperparameters
    parser.add_argument('--num_filters_list', nargs='+', type=int, default=[50, 100, 200],
                        help='Number of CNN filters to try')
    parser.add_argument('--filter_sizes_list', nargs='+', type=str, default=['[3,4,5]', '[2,3,4]', '[2,4,6]'],
                        help='CNN filter sizes to try')

    # RNN-specific hyperparameters
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[64, 128, 256],
                        help='RNN hidden dimensions to try')
    parser.add_argument('--rnn_types', nargs='+', type=str, default=['lstm', 'gru'], help='RNN types to try')
    parser.add_argument('--rnn_layers_list', nargs='+', type=int, default=[1, 2], help='Number of RNN layers to try')
    parser.add_argument('--bidirectional_list', nargs='+', type=bool, default=[True, False],
                        help='Whether to use bidirectional RNN')

    # CNN-LSTM-specific hyperparameters
    parser.add_argument('--lstm_layers_list', nargs='+', type=int, default=[1, 2], help='Number of LSTM layers to try')

    args = parser.parse_args()

    # Convert filter sizes from strings to lists
    args.filter_sizes_list = [eval(fs) for fs in args.filter_sizes_list]

    # Check if at least one model type is selected
    if not (args.run_cnn or args.run_rnn or args.run_cnn_lstm):
        print("Warning: No model type selected. Running all models.")
        args.run_cnn = True
        args.run_rnn = True
        args.run_cnn_lstm = True

    # Create and run the hyperparameter tuner
    tuner = MultiModelHyperparameterTuner(args)
    tuner.run_all_models()


if __name__ == "__main__":
    main()