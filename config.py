"""
Configuration file for the email classification system.
"""

# Email Categories
EMAIL_CATEGORIES = [
    "Billing Issues",
    "Technical Support",
    "Account Management",
    "Product Inquiry",
    "General Feedback",
    "Complaint"
]

# PII Entity Types
PII_ENTITIES = {
    "full_name": "full_name",
    "email": "email",
    "phone_number": "phone_number",
    "dob": "dob",
    "aadhar_num": "aadhar_num",
    "credit_debit_no": "credit_debit_no",
    "cvv_no": "cvv_no",
    "expiry_no": "expiry_no"
}

# Regex patterns for PII detection
REGEX_PATTERNS = {
    "full_name": r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone_number": r'\b(\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b',
    "dob": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
    "aadhar_num": r'\b\d{4}[- ]?\d{4}[- ]?\d{4}\b',
    "credit_debit_no": r'\b(?:\d{4}[- ]?){4}\b',
    "cvv_no": r'\b\d{3,4}\b',  # This needs contextual validation
    "expiry_no": r'\b(0[1-9]|1[0-2])[/]\d{2,4}\b'
}
ENSEMBLE_WEIGHTS = {
    "traditional": 0.3,
    "transformer": 0.7
}

MONITORING_CONFIG = {
    "drift_detection": {
        "p_val": 0.05,
        "alert_threshold": 0.3
    },
    "performance": {
        "log_interval": 100,
        "alert_accuracy_threshold": 0.7
    }
}
# Model configuration
MODEL_CONFIG = {
    "classifier_type": "transformer",  # Options: "traditional", "transformer", "llm"
    "transformer_model": "distilbert-base-uncased",
    "max_length": 512,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 3,
    "model_save_path": "models/email_classifier.pkl"
}

# API configuration
API_CONFIG = {
    "title": "Email Classification API",
    "description": "API for classifying emails and masking PII",
    "version": "1.0.0",
    "docs_url": "/docs",
    "redoc_url": "/redoc"
}
