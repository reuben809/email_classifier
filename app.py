"""
Main application entry point for Hugging Face Spaces.
"""

import uvicorn
import os
import sys
import logging
from api import app
from monitoring import ModelMonitor  # Import the monitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)

# Initialize the monitor
monitor = ModelMonitor()

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 7860))

    logger.info(f"Starting application on port {port}")

    # Start server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )