import os
import pandas as pd
import numpy as np
from alibi_detect.cd import ChiSquareDrift
from config import MONITORING_CONFIG
import logging
import joblib
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelMonitor:
    def __init__(self):
        self.reference_data = None
        self.drift_detector = None
        self.predictions = []
        self.inputs = []
        self.last_drift_check = datetime.now()

    def log_system_start(self):
        logger.info("System started")
        # Load any existing monitoring data if available
        try:
            self.predictions = joblib.load("monitoring/predictions.joblib")
            self.inputs = joblib.load("monitoring/inputs.joblib")
        except Exception as e:
            logger.info("No existing monitoring data found. Starting fresh.")

    def log_input(self, input_text: str):
        self.inputs.append(input_text)
        self._persist_data()

    def log_prediction(self, input_text: str, predicted_category: str):
        self.predictions.append({
            "input": input_text,
            "prediction": predicted_category,
            "timestamp": datetime.now()
        })
        self._persist_data()

    def _persist_data(self):
        # Create directory if it doesn't exist
        os.makedirs("monitoring", exist_ok=True)

        # Save data periodically
        if len(self.predictions) % 10 == 0:
            joblib.dump(self.predictions, "monitoring/predictions.joblib")
            joblib.dump(self.inputs, "monitoring/inputs.joblib")

    def check_drift(self, new_data: str):
        current_time = datetime.now()
        time_since_last_check = (current_time - self.last_drift_check).total_seconds()

        # Check drift every 5 minutes or when we have enough new data
        if (time_since_last_check >= 300) or (len(self.inputs) >= 100):
            if self.drift_detector is None:
                if len(self.inputs) < 100:
                    return False  # Not enough data for initial reference

                self.reference_data = self.inputs[:100]
                self.drift_detector = ChiSquareDrift(
                    x_ref=self.reference_data,
                    p_val=MONITORING_CONFIG["drift_detection"]["p_val"]
                )
                logger.info("Initialized drift detector")
            else:
                result = self.drift_detector.predict(new_data)
                if result["data"]["is_drift"]:
                    logger.warning("Data drift detected. Consider retraining model.")
                    return True
                else:
                    logger.info("No significant data drift detected")

            self.last_drift_check = current_time

        return False

    def evaluate_performance(self):
        if len(self.predictions) < MONITORING_CONFIG["performance"]["log_interval"]:
            return

        # Here you would implement actual performance evaluation
        # This is a simplified version
        logger.info(f"Model has processed {len(self.predictions)} requests")

        if len(self.predictions) % MONITORING_CONFIG["performance"]["log_interval"] == 0:
            logger.info("Performance check triggered")
            # Implement actual performance metrics here