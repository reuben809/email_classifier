"""
Training script for the email classification model.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from models import EnsembleEmailClassifier
from config import MODEL_CONFIG
import logging
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_and_save_model():
    # Load the dataset
    data_path = "data/combined_emails_with_natural_pii.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with {len(df)} samples")

        # Check if the DataFrame has the expected structure
        required_columns = ['email', 'type']
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"Dataset must contain the following columns: {required_columns}")

        # Rename columns to match what the model expects
        df.rename(columns={'email': 'text', 'type': 'label'}, inplace=True)
        logger.info("Renamed columns to 'text' and 'label'")

        # Display data types and a preview of the data
        logger.info("\nData types:\n%s", df.dtypes)
        logger.info("\nData preview:\n%s", df.head())

        # Split the data
        train_data, eval_data = train_test_split(df, test_size=0.2, random_state=42)
        logger.info(f"Training data size: {len(train_data)}, Evaluation data size: {len(eval_data)}")

        # Check if the data is correctly split
        if len(train_data) == 0 or len(eval_data) == 0:
            raise ValueError("Data split resulted in empty training or evaluation set")

        # Check if the data contains the expected columns after renaming
        if 'text' not in train_data.columns or 'label' not in train_data.columns:
            raise ValueError("Training data must contain 'text' and 'label' columns")

        # Initialize ensemble classifier
        ensemble_classifier = EnsembleEmailClassifier()

        # Train the classifier
        ensemble_classifier.train(train_data, eval_data)

        # Save the trained model
        model_dir = "models/ensemble/"
        os.makedirs(model_dir, exist_ok=True)
        ensemble_classifier.save(model_dir)

        logger.info("Model training and saving completed successfully.")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    train_and_save_model()