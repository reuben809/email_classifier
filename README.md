# Email Classification System

## Overview
This project implements an email classification system with PII (Personally Identifiable Information) masking. The system uses machine learning models to classify emails into predefined categories while ensuring that personal information is masked for privacy.

## Features
- Classifies emails into categories such as Billing Issues, Technical Support, Account Management, etc.
- Masks PII information including names, emails, phone numbers, and more.
- Provides an API endpoint for easy integration.
- Batch testing capability for multiple emails.

## Installation

### Prerequisites
- Python 3.9+
- Virtual environment tools (venv)

### Setup
1. **Clone the Repository**
   ```bash
   git clone https://github.com/reuben809/email_classifier.git
   cd email-classification-system
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Missing Dependency**
   ```bash
   pip install alibi-detect
   ```

5. **Download SpaCy Model**
   ```bash
   python -m spacy download en_core_web_lg
   ```

## Usage

### Train the Model
Run the training script to train the model on your dataset:
```bash
python train_model.py
```

### Run the Application
Start the API server:
```bash
python app.py
```

The API will be available at `http://localhost:7860`.

### Test with Postman
You can test the API using Postman:

1. **Create a New Request**:
   - Set the request method to `POST`.
   - Enter the URL: `http://localhost:7860/classify`.

2. **Set the Request Body**:
   - Go to the "Body" tab.
   - Select "Raw" and choose "JSON" from the dropdown.
   - Enter the email body in the JSON format:
     ```json
     {
       "email_body": "This is a test email with PII: John Doe, john.doe@example.com, (555) 123-4567"
     }
     ```

3. **Set the Headers**:
   - Go to the "Headers" tab.
   - Add a header with the key `Content-Type` and the value `application/json`.

4. **Send the Request**:
   - Click the "Send" button.

### Batch Testing
To perform batch testing with Postman:

1. **Prepare a JSON file** with multiple email entries.
2. **Create a Postman Collection** and add a request with the API endpoint.
3. **Use Postman's Runner** to execute the collection with the data file.

## API Endpoints

### Classify an Email
- **POST** `http://localhost:7860/classify`
  - **Request Body**:
    ```json
    {
      "email_body": "string containing the email"
    }
    ```
  - **Response**:
    ```json
    {
      "input_email_body": "string containing the email",
      "list_of_masked_entities": [
        {
          "position": [start_index, end_index],
          "classification": "entity_type",
          "entity": "original_entity_value"
        }
      ],
      "masked_email": "string containing the masked email",
      "category_of_the_email": "string containing the class"
    }
    ```

## Model Training

To train the model, run the training script:
```bash
python train_model.py
```

The script will train both a traditional machine learning model and a transformer-based model, then save the trained models.

## Contribution Guidelines

Contributions are welcome! Please follow these guidelines:

1. Fork the repository and create your branch from `main`.
2. Ensure your code follows PEP8 standards.
3. Write tests for new features.
4. Update documentation as needed.
5. Submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
