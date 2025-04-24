"""
Models for email classification.
Includes both traditional ML and transformer-based approaches.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from torch.optim import AdamW
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from config import MODEL_CONFIG, EMAIL_CATEGORIES, ENSEMBLE_WEIGHTS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmailDataset(Dataset):
    """
    Dataset class for email classification with transformers.
    """

    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, tokenizer=None, max_length: int = 512):
        """
        Initialize the dataset.

        Args:
            texts: List of email texts
            labels: List of corresponding labels
            tokenizer: Transformer tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx) -> Dict[str, Union[torch.Tensor, int]]:
        text = self.texts[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Convert dictionary values from batch to single example
        item = {key: val.squeeze(0) for key, val in encoding.items()}

        # Add label if available
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


class TraditionalMLClassifier:
    """
    Traditional ML classifier using Scikit-learn.
    """

    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize the classifier.

        Args:
            model_type: Type of model to use (random_forest or naive_bayes)
        """
        self.model_type = model_type
        self.pipeline = None
        self.classes = None

    def build_pipeline(self) -> None:
        """
        Build the classification pipeline.
        """
        # Feature extraction
        vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=5,
            max_df=0.7,
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        # Choose classifier
        if self.model_type == "random_forest":
            classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "naive_bayes":
            classifier = MultinomialNB(alpha=0.1)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Create pipeline
        self.pipeline = Pipeline([
            ("vectorizer", vectorizer),
            ("classifier", classifier)
        ])

    def train(self, train_data: pd.DataFrame, eval_data: pd.DataFrame = None) -> None:
        """
        Train the classifier.

        Args:
            train_data: Training data with 'text' and 'label' columns
            eval_data: Evaluation data with 'text' and 'label' columns
        """
        if self.pipeline is None:
            self.build_pipeline()

        # Extract texts and labels
        X_train = train_data["text"].tolist()
        y_train = train_data["label"].tolist()

        # Store classes for prediction
        self.classes = sorted(set(y_train))

        # Convert string labels to numeric if needed
        if isinstance(y_train[0], str):
            label_map = {label: i for i, label in enumerate(self.classes)}
            y_train_numeric = [label_map[label] for label in y_train]
        else:
            y_train_numeric = y_train

        logger.info(f"Training {self.model_type} classifier...")
        self.pipeline.fit(X_train, y_train_numeric)
        logger.info("Training complete.")

    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict labels for the given texts.

        Args:
            texts: List of texts to classify

        Returns:
            List of predicted labels
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")

        # Get numeric predictions
        y_pred = self.pipeline.predict(texts)

        # Convert to string labels if needed
        if isinstance(self.classes[0], str):
            return [self.classes[pred] for pred in y_pred]
        else:
            return y_pred.tolist()

    def evaluate(self, X_test: List[str], y_test: List[Union[str, int]]) -> Dict[str, Any]:
        """
        Evaluate the classifier.

        Args:
            X_test: Test texts
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")

        # Convert string labels to numeric if needed
        if isinstance(y_test[0], str) and isinstance(self.classes[0], str):
            label_map = {label: i for i, label in enumerate(self.classes)}
            y_test_numeric = [label_map[label] for label in y_test]
        else:
            y_test_numeric = y_test

        # Get predictions
        y_pred = self.predict(X_test)

        # Convert predictions to numeric if needed
        if isinstance(y_pred[0], str):
            y_pred_numeric = [label_map[label] for label in y_pred]
        else:
            y_pred_numeric = y_pred

        # Calculate metrics
        accuracy = accuracy_score(y_test_numeric, y_pred_numeric)
        report = classification_report(y_test_numeric, y_pred_numeric,
                                       target_names=self.classes if isinstance(self.classes[0], str) else None)

        return {
            "accuracy": accuracy,
            "classification_report": report
        }

    def save(self, file_path: str) -> None:
        """
        Save the model to a file.

        Args:
            file_path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        model_data = {
            "pipeline": self.pipeline,
            "classes": self.classes,
            "model_type": self.model_type
        }

        joblib.dump(model_data, file_path)
        logger.info(f"Model saved to {file_path}")

    @classmethod
    def load(cls, file_path: str) -> 'TraditionalMLClassifier':
        """
        Load a model from a file.

        Args:
            file_path: Path to the saved model

        Returns:
            Loaded model
        """
        model_data = joblib.load(file_path)

        instance = cls(model_type=model_data["model_type"])
        instance.pipeline = model_data["pipeline"]
        instance.classes = model_data["classes"]

        return instance

class TransformerClassifier:
    """
    Transformer-based classifier using Hugging Face transformers.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = None):
        """
        Initialize the classifier.

        Args:
            model_name: Name of the pre-trained model
            num_labels: Number of classes
        """
        self.model_name = model_name
        self.num_labels = num_labels or len(EMAIL_CATEGORIES)
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_map = {i: label for i, label in enumerate(EMAIL_CATEGORIES)}

    def initialize_model(self) -> None:
        """
        Initialize the model and tokenizer.
        """
        logger.info(f"Initializing {self.model_name} model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        self.model.to(self.device)

    def train(self, train_texts: List[str], train_labels: List[int], eval_texts: List[str] = None,
              eval_labels: List[int] = None, batch_size: int = 16, num_epochs: int = 3) -> None:
        """
        Train the classifier.

        Args:
            train_texts: Training texts
            train_labels: Training labels
            eval_texts: Evaluation texts
            eval_labels: Evaluation labels
            batch_size: Batch size
            num_epochs: Number of epochs
        """
        if self.tokenizer is None or self.model is None:
            self.initialize_model()

        # Create datasets
        train_dataset = EmailDataset(train_texts, train_labels, self.tokenizer)
        eval_dataset = None
        if eval_texts and eval_labels:
            eval_dataset = EmailDataset(eval_texts, eval_labels, self.tokenizer)

        # Set up training arguments
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            report_to="none"
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict labels for the given texts.

        Args:
            texts: List of texts to classify

        Returns:
            List of predicted labels
        """
        if self.tokenizer is None or self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() or train() first.")

        # Create dataset
        dataset = EmailDataset(texts, tokenizer=self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=MODEL_CONFIG["batch_size"])

        # Set model to evaluation mode
        self.model.eval()

        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)

                # Get predictions
                preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(preds.cpu().numpy())

        # Convert numeric predictions to labels
        return [self.label_map[pred] for pred in predictions]

    def save(self, directory: str) -> None:
        """
        Save the model and tokenizer.

        Args:
            directory: Directory to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)

        # Save label map
        with open(os.path.join(directory, "label_map.joblib"), "wb") as f:
            joblib.dump(self.label_map, f)

        logger.info(f"Model and tokenizer saved to {directory}")

    @classmethod
    def load(cls, directory: str) -> 'TransformerClassifier':
        """
        Load a model from a directory.

        Args:
            directory: Directory with the saved model

        Returns:
            Loaded model
        """
        # Create instance
        instance = cls()

        # Load model and tokenizer
        instance.tokenizer = AutoTokenizer.from_pretrained(directory)
        instance.model = AutoModelForSequenceClassification.from_pretrained(directory)
        instance.model.to(instance.device)

        # Load label map
        with open(os.path.join(directory, "label_map.joblib"), "rb") as f:
            instance.label_map = joblib.load(f)

        return instance


class EmailClassifier:
    """
    Main class for email classification.
    Provides a unified interface for different classifier types.
    """

    def __init__(self, classifier_type: str = "transformer"):
        """
        Initialize the classifier.

        Args:
            classifier_type: Type of classifier to use (traditional or transformer)
        """
        self.classifier_type = classifier_type
        self.classifier = None

    def initialize_classifier(self) -> None:
        """
        Initialize the appropriate classifier based on the specified type.
        """
        if self.classifier_type == "traditional":
            self.classifier = TraditionalMLClassifier(model_type="random_forest")
        elif self.classifier_type == "transformer":
            self.classifier = TransformerClassifier(model_name=MODEL_CONFIG["transformer_model"])
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type}")

    def train(self, train_data: pd.DataFrame, eval_data: pd.DataFrame = None) -> None:
        """
        Train the classifier.

        Args:
            train_data: Training data with 'text' and 'label' columns
            eval_data: Evaluation data with 'text' and 'label' columns
        """
        if self.classifier is None:
            self.initialize_classifier()

        if self.classifier_type == "traditional":
            self.classifier.train(train_data["text"].tolist(), train_data["label"].tolist())
        elif self.classifier_type == "transformer":
            # Convert labels to numeric indices
            label_map = {label: i for i, label in enumerate(sorted(set(train_data["label"])))}
            train_labels = [label_map[label] for label in train_data["label"]]

            eval_texts = None
            eval_labels = None
            if eval_data is not None:
                eval_texts = eval_data["text"].tolist()
                eval_labels = [label_map[label] for label in eval_data["label"]]

            self.classifier.train(
                train_data["text"].tolist(),
                train_labels,
                eval_texts,
                eval_labels,
                batch_size=MODEL_CONFIG["batch_size"],
                num_epochs=MODEL_CONFIG["epochs"]
            )

    def predict(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Predict labels for the given texts.

        Args:
            texts: Text or list of texts to classify

        Returns:
            Predicted label or list of labels
        """
        if self.classifier is None:
            raise ValueError("Classifier not initialized. Call initialize_classifier() or train() first.")

        # Handle single text input
        if isinstance(texts, str):
            return self.classifier.predict([texts])[0]

        return self.classifier.predict(texts)

    def save(self, path: str) -> None:
        """
        Save the classifier.

        Args:
            path: Path to save the classifier
        """
        if self.classifier is None:
            raise ValueError("Classifier not initialized. Call initialize_classifier() or train() first.")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if self.classifier_type == "traditional":
            self.classifier.save(path)
        elif self.classifier_type == "transformer":
            self.classifier.save(path)

        # Save classifier type
        with open(os.path.join(os.path.dirname(path), "classifier_type.txt"), "w") as f:
            f.write(self.classifier_type)

    @classmethod
    def load(cls, path: str) -> 'EmailClassifier':
        """
        Load a classifier from a file or directory.

        Args:
            path: Path to the saved classifier

        Returns:
            Loaded classifier
        """
        # Load classifier type
        with open(os.path.join(os.path.dirname(path), "classifier_type.txt"), "r") as f:
            classifier_type = f.read().strip()

        instance = cls(classifier_type=classifier_type)

        if classifier_type == "traditional":
            instance.classifier = TraditionalMLClassifier.load(path)
        elif classifier_type == "transformer":
            instance.classifier = TransformerClassifier.load(path)

        return instance

    def get_default_trained_model(self):
        """
        Get a default trained model when no training data is available.
        Uses a simple rule-based approach as fallback.

        Returns:
            Self with a simple trained model
        """
        # If no training data is available, initialize a simple rule-based classifier
        if self.classifier is None:
            self.initialize_classifier()

        # If transformer model, we need to ensure it's initialized
        if self.classifier_type == "transformer":
            if self.classifier.model is None or self.classifier.tokenizer is None:
                self.classifier.initialize_model()

        # Create a simple fallback method if the real model isn't available
        def simple_rule_based_classify(text):
            text = text.lower()

            # Simple keyword matching
            if any(word in text for word in ["bill", "payment", "charge", "refund", "invoice"]):
                return "Billing Issues"
            elif any(word in text for word in ["bug", "error", "crash", "not working", "broken", "technical"]):
                return "Technical Support"
            elif any(word in text for word in ["account", "login", "password", "profile", "settings"]):
                return "Account Management"
            elif any(word in text for word in ["product", "feature", "service", "offering", "pricing"]):
                return "Product Inquiry"
            elif any(word in text for word in ["thank", "great", "awesome", "good", "excellent", "feedback"]):
                return "General Feedback"
            elif any(word in text for word in ["angry", "disappointed", "issue", "problem", "complaint", "dissatisfied"]):
                return "Complaint"
            else:
                return "General Feedback"  # Default category

        # Attach the fallback method
        self._fallback_classify = simple_rule_based_classify

        return self


class EnsembleEmailClassifier:
    def __init__(self):
        self.classifiers = {
            "traditional": TraditionalMLClassifier(model_type="random_forest"),
            "transformer": TransformerClassifier(model_name=MODEL_CONFIG["transformer_model"])
        }
        self.weights = ENSEMBLE_WEIGHTS

    def train(self, train_data: pd.DataFrame, eval_data: pd.DataFrame = None) -> None:
        for name, classifier in self.classifiers.items():
            logger.info(f"Training {name} classifier...")
            classifier.train(train_data, eval_data)

    def predict(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(texts, str):
            texts = [texts]

        predictions = {
            name: classifier.predict(texts)
            for name, classifier in self.classifiers.items()
        }

        # Weighted voting mechanism
        final_predictions = []
        for i in range(len(texts)):
            vote = {category: 0.0 for category in EMAIL_CATEGORIES}

            for model_name, preds in predictions.items():
                vote[preds[i]] += self.weights[model_name]

            final_predictions.append(max(vote, key=vote.get))

        return final_predictions if len(final_predictions) > 1 else final_predictions[0]

    def save(self, path: str) -> None:
        for name, classifier in self.classifiers.items():
            classifier.save(os.path.join(path, name))
        joblib.dump(self.weights, os.path.join(path, "ensemble_weights.joblib"))

    @classmethod
    def load(cls, path: str) -> 'EnsembleEmailClassifier':
        instance = cls()
        instance.weights = joblib.load(os.path.join(path, "ensemble_weights.joblib"))

        for name, _ in instance.classifiers.items():
            if name == "traditional":
                instance.classifiers[name] = TraditionalMLClassifier.load(
                    os.path.join(path, name, "email_classifier.joblib")
                )
            elif name == "transformer":
                instance.classifiers[name] = TransformerClassifier.load(
                    os.path.join(path, name)
                )

        return instance