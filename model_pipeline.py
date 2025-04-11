import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import os
import json
import abc
from typing import Dict, Tuple, List, Any, Optional
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Custom dataset class for text classification
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BaseModelPipeline(abc.ABC):
    """Base class for model training pipelines.

    This class handles dataset preparation and provides default
    implementations for training and evaluation loops, which can be
    overridden by subclasses if needed.
    """

    def __init__(self, args):
        """Initialize the pipeline with configuration arguments.

        Args:
            args: Configuration arguments for the pipeline.
        """
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_accuracy = 0
        self.label_map = None

        # Set seed for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # Create output directory
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    def prepare_data(self):
        """Prepare datasets and dataloaders."""
        print(f"Loading data from {self.args.data_path}")
        df = pd.read_csv(self.args.data_path)

        # Check if columns exist
        if self.args.text_column not in df.columns:
            raise ValueError(f"Text column '{self.args.text_column}' not in dataset")
        if self.args.label_column not in df.columns:
            raise ValueError(f"Label column '{self.args.label_column}' not in dataset")

        # Split into train and validation sets
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            random_state=self.args.seed,
            stratify=df[self.args.label_column]
        )

        # Convert string labels to integers if needed
        if isinstance(train_df[self.args.label_column].iloc[0], str):
            self.label_map = {label: idx for idx, label in enumerate(train_df[self.args.label_column].unique())}
            train_df[self.args.label_column] = train_df[self.args.label_column].map(self.label_map)
            val_df[self.args.label_column] = val_df[self.args.label_column].map(self.label_map)

            # Save label mapping
            with open(os.path.join(self.args.output_dir, 'label_map.json'), 'w') as f:
                json.dump(self.label_map, f)

        # Load tokenizer and create datasets
        self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_model)

        self.train_dataset = TextDataset(
            texts=train_df[self.args.text_column].values,
            labels=train_df[self.args.label_column].values,
            tokenizer=self.tokenizer,
            max_length=self.args.max_length
        )

        self.val_dataset = TextDataset(
            texts=val_df[self.args.text_column].values,
            labels=val_df[self.args.label_column].values,
            tokenizer=self.tokenizer,
            max_length=self.args.max_length
        )

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size
        )

        # Get number of classes
        self.num_classes = len(df[self.args.label_column].unique())

        return self.train_loader, self.val_loader, self.num_classes

    @abc.abstractmethod
    def build_model(self):
        """Build and return the model. To be implemented by subclasses."""
        pass

    def setup_optimizer(self, model):
        """Setup optimizer and scheduler for the model."""
        optimizer = AdamW(
            model.parameters(),
            lr=self.args.learning_rate,
            eps=1e-8
        )

        total_steps = len(self.train_loader) * self.args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        return optimizer, scheduler

    def train_epoch(self, model, data_loader, optimizer, scheduler):
        """Train the model for one epoch.

        Can be overridden by subclasses to implement custom training loops.
        """
        model.train()
        losses = []

        # Add progress bar
        progress_bar = tqdm(data_loader, desc="Training")

        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)

            losses.append(loss.item())

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        return np.mean(losses)

    def evaluate(self, model, data_loader):
        """Evaluate the model on a dataset.

        Can be overridden by subclasses to implement custom evaluation loops.
        """
        model.eval()
        losses = []
        predictions = []
        actual_labels = []

        # Add progress bar
        progress_bar = tqdm(data_loader, desc="Evaluating")

        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(outputs, labels)
                losses.append(loss.item())

                _, preds = torch.max(outputs, dim=1)

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

    def train(self):
        """Run the full training and evaluation pipeline."""
        # Prepare data
        self.prepare_data()

        # Build model
        model = self.build_model()
        model = model.to(self.device)

        # Print model information
        print(f"\nModel Architecture:")
        print(model)
        print(f"\nTraining on device: {self.device}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")

        # Setup optimizer and scheduler
        optimizer, scheduler = self.setup_optimizer(model)

        # Training loop
        print("\nStarting training...\n")
        self.best_accuracy = 0

        for epoch in range(self.args.epochs):
            print(f"Epoch {epoch + 1}/{self.args.epochs}")

            # Train
            train_loss = self.train_epoch(model, self.train_loader, optimizer, scheduler)
            print(f"Training loss: {train_loss:.4f}")

            # Evaluate
            val_loss, accuracy, report, _, _ = self.evaluate(model, self.val_loader)
            print(f"Validation loss: {val_loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {report['weighted avg']['precision']:.4f}")
            print(f"Recall: {report['weighted avg']['recall']:.4f}")
            print(f"F1-score: {report['weighted avg']['f1-score']:.4f}")

            # Save the best model
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                torch.save(model.state_dict(), os.path.join(self.args.output_dir, 'best_model.pt'))
                print(f"Best model saved with accuracy: {self.best_accuracy:.4f}")

        print(f"\nTraining completed! Best accuracy: {self.best_accuracy:.4f}")

        # Load the best model for final evaluation
        model.load_state_dict(torch.load(os.path.join(self.args.output_dir, 'best_model.pt')))

        # Final evaluation
        _, accuracy, report, predictions, actual_labels = self.evaluate(model, self.val_loader)

        # Save classification report
        with open(os.path.join(self.args.output_dir, 'classification_report.txt'), 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
            f.write(classification_report(actual_labels, predictions))

        print(f"\nFinal model evaluation saved to {self.args.output_dir}/classification_report.txt")

        return model, accuracy