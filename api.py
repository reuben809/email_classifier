"""
API implementation for email classification system.
"""

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import logging
from utils import PIIMasker, preprocess_email, format_api_response
from models import EmailClassifier, EnsembleEmailClassifier
from config import API_CONFIG, MODEL_CONFIG
import joblib
from fastapi.security import APIKeyHeader
from fastapi import Depends, HTTPException, status
from monitoring import ModelMonitor  # Import the monitor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"],
    docs_url=API_CONFIG["docs_url"],
    redoc_url=API_CONFIG["redoc_url"]
)

# Input and output models
class EmailInput(BaseModel):
    email_body: str = Field(..., description="The email body text to classify")


class Entity(BaseModel):
    position: List[int] = Field(..., description="Start and end indices of the entity [start, end]")
    classification: str = Field(..., description="Type of entity (e.g., full_name, email)")
    entity: str = Field(..., description="Original value of the entity")


class EmailOutput(BaseModel):
    input_email_body: str = Field(..., description="Original email text")
    list_of_masked_entities: List[Entity] = Field(..., description="List of detected PII entities")
    masked_email: str = Field(..., description="Email with PII masked")
    category_of_the_email: str = Field(..., description="Classified email category")


# Initialize services
pii_masker = PIIMasker()
email_classifier = None
monitor = ModelMonitor()  # Create monitor instance

def load_or_initialize_classifier():
    """
    Load the classifier from disk if available, otherwise initialize a new one.

    Returns:
        Initialized email classifier
    """
    global email_classifier

    model_path = MODEL_CONFIG["model_save_path"]

    try:
        if os.path.exists(os.path.dirname(model_path)):
            logger.info(f"Loading classifier from {model_path}")
            email_classifier = EmailClassifier.load(model_path)
        else:
            logger.warning("No trained model found. Initializing default classifier.")
            email_classifier = EmailClassifier(classifier_type=MODEL_CONFIG["classifier_type"])
            email_classifier.get_default_trained_model()
    except Exception as e:
        logger.error(f"Error loading classifier: {e}")
        email_classifier = EmailClassifier(classifier_type=MODEL_CONFIG["classifier_type"])
        email_classifier.get_default_trained_model()

    return email_classifier


@app.on_event("startup")
async def startup_event():
    """
    Initialize services on startup.
    """
    logger.info("Initializing services...")
    load_or_initialize_classifier()
    monitor.log_system_start()  # Log system start
    logger.info("Services initialized successfully")


@app.get("/")
async def root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "Email Classification API is running"}


@app.post("/classify", response_model=EmailOutput)
async def classify_email(email_input: EmailInput = Body(...)):
    """
    Classify an email and mask PII.

    Args:
        email_input: Input email

    Returns:
        Classified email with masked PII
    """
    try:
        # Get the email text
        email_text = email_input.email_body

        # Preprocess email
        preprocessed_email = preprocess_email(email_text)

        # Mask PII
        masked_email, entities = pii_masker.mask_pii(preprocessed_email)

        # Classify email
        if email_classifier is None:
            load_or_initialize_classifier()

        category = email_classifier.predict(masked_email)

        # Format response
        response = format_api_response(
            input_email=preprocessed_email,
            masked_email=masked_email,
            entities=entities,
            category=category
        )

        # Log the input and prediction for monitoring
        monitor.log_input(email_text)
        monitor.log_prediction(email_text, category)

        return response

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing email: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}