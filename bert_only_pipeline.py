from model_pipeline import BaseModelPipeline
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


class BertOnlyPipeline(BaseModelPipeline):
    """Pipeline for training and evaluating a BERT classifier using BertForSequenceClassification."""

    def __init__(self, args):
        super().__init__(args)

    def build_model(self):
        """Build and return the BERT model for sequence classification."""
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=self.num_classes
        )
        return model

    def train_epoch(self, model, data_loader, optimizer, scheduler):
        """Custom training loop for BertForSequenceClassification.

        This uses the model's built-in loss calculation when labels are provided.
        """
        model.train()
        losses = []

        # Create progress bar
        progress_bar = tqdm(data_loader, desc="Training")

        # Determine gradient accumulation steps (if specified in args)
        grad_accum_steps = getattr(self.args, 'gradient_accumulation_steps', 1)

        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass - providing labels to the model
            # This will make the model calculate the loss internally
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Loss is the first output when labels are provided
            loss = outputs.loss

            # Scale loss if using gradient accumulation
            if grad_accum_steps > 1:
                loss = loss / grad_accum_steps

            # Backward pass
            loss.backward()

            # Record loss
            losses.append(loss.item() * (grad_accum_steps if grad_accum_steps > 1 else 1))

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item() * (grad_accum_steps if grad_accum_steps > 1 else 1)})

            # Update weights if needed
            if (step + 1) % grad_accum_steps == 0:
                # Clip gradients to prevent exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Update parameters
                optimizer.step()

                # Update learning rate
                scheduler.step()

                # Zero gradients
                optimizer.zero_grad()

        return np.mean(losses)

    def evaluate(self, model, data_loader):
        """Custom evaluation loop for BertForSequenceClassification."""
        model.eval()
        losses = []
        predictions = []
        actual_labels = []

        # Create progress bar
        progress_bar = tqdm(data_loader, desc="Evaluating")

        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                # Method 1: Get loss and logits separately
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                logits = outputs.logits

                losses.append(loss.item())

                # Get predictions
                _, preds = torch.max(logits, dim=1)

                # Store predictions and labels
                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())

                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})

        # Calculate metrics
        accuracy = accuracy_score(actual_labels, predictions)
        f1 = f1_score(actual_labels, predictions, average='weighted')
        precision = precision_score(actual_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(actual_labels, predictions, average='weighted', zero_division=0)
        report = classification_report(actual_labels, predictions, output_dict=True)

        # Additional logging for custom metrics
        print(f"F1 Score (Weighted): {f1:.4f}")

        return np.mean(losses), accuracy, report, predictions, actual_labels


def main():
    parser = argparse.ArgumentParser(description='BERT-Only Classifier for Text Classification')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV dataset')
    parser.add_argument('--text_column', type=str, default='text', help='Column name for text data')
    parser.add_argument('--label_column', type=str, default='label', help='Column name for labels')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased', help='BERT model to use')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--output_dir', type=str, default='./results_bert_only', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of steps to accumulate gradients')
    args = parser.parse_args()

    # Create and run the pipeline
    pipeline = BertOnlyPipeline(args)
    pipeline.train()


if __name__ == "__main__":
    main()