"""
Utility functions for email classification system.
Includes PII masking and other helper functions.
"""

import re
import spacy
from typing import Dict, List, Tuple, Any
import pandas as pd
from config import REGEX_PATTERNS, PII_ENTITIES
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_lg")
    logger.info("Loaded SpaCy model successfully")
except OSError:
    logger.warning("SpaCy model not found. Using blank model.")
    nlp = spacy.blank("en")

class PIIMasker:
    """
    Class for detecting and masking PII in email text.
    Uses both regex patterns and spaCy NER for robust detection.
    """
    
    def __init__(self, regex_patterns: Dict[str, str] = None):
        """
        Initialize the PIIMasker with regex patterns and spaCy NER.
        
        Args:
            regex_patterns: Dictionary of regex patterns for PII entities
        """
        self.regex_patterns = regex_patterns or REGEX_PATTERNS
        self.nlp = nlp
        self.compiled_patterns = {
            entity_type: re.compile(pattern) 
            for entity_type, pattern in self.regex_patterns.items()
        }
        
    def _find_regex_matches(self, text: str) -> List[Dict[str, Any]]:
        """
        Find all regex matches in the text.
        
        Args:
            text: Input text to search
            
        Returns:
            List of dictionaries with entity information
        """
        matches = []
        
        for entity_type, pattern in self.compiled_patterns.items():
            for match in pattern.finditer(text):
                start, end = match.span()
                entity_value = text[start:end]
                
                # Special validation for CVV numbers (needs context)
                if entity_type == "cvv_no" and not self._is_valid_cvv_context(text, start, end):
                    continue
                    
                matches.append({
                    "position": [start, end],
                    "classification": entity_type,
                    "entity": entity_value
                })
        
        return matches
    
    def _is_valid_cvv_context(self, text: str, start: int, end: int) -> bool:
        """
        Check if a potential CVV number is in a valid context.
        
        Args:
            text: Input text
            start: Start position of match
            end: End position of match
            
        Returns:
            Boolean indicating if context suggests this is a CVV
        """
        context_window = 30  # Check 30 chars before and after
        context_start = max(0, start - context_window)
        context_end = min(len(text), end + context_window)
        context = text[context_start:context_end].lower()
        
        cvv_indicators = ["cvv", "security code", "security number", "card verification", "cvc"]
        return any(indicator in context for indicator in cvv_indicators)
    
    def _find_ner_matches(self, text: str) -> List[Dict[str, Any]]:
        """
        Find named entities using spaCy NER.
        
        Args:
            text: Input text
            
        Returns:
            List of dictionaries with entity information
        """
        matches = []
        doc = self.nlp(text)
        
        ner_mapping = {
            "PERSON": "full_name",
            "DATE": "dob",  # Needs additional validation
        }
        
        for ent in doc.ents:
            if ent.label_ in ner_mapping:
                entity_type = ner_mapping[ent.label_]
                
                # Additional validation for dates to identify DOB
                if entity_type == "dob" and not self._is_dob_context(text, ent.start_char, ent.end_char):
                    continue
                    
                matches.append({
                    "position": [ent.start_char, ent.end_char],
                    "classification": entity_type,
                    "entity": text[ent.start_char:ent.end_char]
                })
                
        return matches
    
    def _is_dob_context(self, text: str, start: int, end: int) -> bool:
        """
        Check if a date entity is likely a date of birth.
        
        Args:
            text: Input text
            start: Start position of match
            end: End position of match
            
        Returns:
            Boolean indicating if context suggests this is a DOB
        """
        context_window = 50  # Check 50 chars before and after
        context_start = max(0, start - context_window)
        context_end = min(len(text), end + context_window)
        context = text[context_start:context_end].lower()
        
        dob_indicators = ["birth", "born", "dob", "date of birth", "birthday"]
        return any(indicator in context for indicator in dob_indicators)
    
    def _resolve_overlapping_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolve overlapping matches by prioritizing longer matches.
        
        Args:
            matches: List of match dictionaries
            
        Returns:
            List of non-overlapping match dictionaries
        """
        if not matches:
            return []
            
        # Sort by start position and then by length (longer matches first)
        sorted_matches = sorted(matches, key=lambda x: (x["position"][0], -len(x["entity"])))
        
        result = [sorted_matches[0]]
        for current in sorted_matches[1:]:
            prev = result[-1]
            current_start, current_end = current["position"]
            prev_start, prev_end = prev["position"]
            
            # Check for overlap
            if current_start >= prev_end:
                result.append(current)
                
        return result
    
    def detect_pii(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect all PII entities in the given text.
        
        Args:
            text: Input text
            
        Returns:
            List of dictionaries with detected PII entities
        """
        regex_matches = self._find_regex_matches(text)
        ner_matches = self._find_ner_matches(text)
        
        # Combine and resolve overlapping matches
        all_matches = regex_matches + ner_matches
        resolved_matches = self._resolve_overlapping_matches(all_matches)
        
        # Sort by position for consistent output
        return sorted(resolved_matches, key=lambda x: x["position"][0])
    
    def mask_pii(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Mask all PII entities in the text and return the masked text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (masked text, list of detected entities)
        """
        entities = self.detect_pii(text)
        
        # Sort entities by start position in reverse order to avoid index issues
        entities_sorted = sorted(entities, key=lambda x: x["position"][0], reverse=True)
        
        # Create a mutable list of characters
        chars = list(text)
        
        # Replace each entity with its mask
        for entity in entities_sorted:
            start, end = entity["position"]
            entity_type = entity["classification"]
            mask = f"[{entity_type}]"
            
            # Replace the characters at the specified indices
            chars[start:end] = mask
        
        # Join the characters back into a string
        masked_text = ''.join(chars)
        
        # Re-sort entities by start position
        entities_output = sorted(entities, key=lambda x: x["position"][0])
        
        return masked_text, entities_output


def preprocess_email(text: str) -> str:
    """
    Clean and preprocess email text.
    
    Args:
        text: Input email text
        
    Returns:
        Preprocessed text
    """
    # Remove email headers if present
    if "From:" in text and "Subject:" in text:
        try:
            # Simple header removal (could be improved for more complex emails)
            body_start = text.find("\n\n")
            if body_start != -1:
                text = text[body_start:].strip()
        except Exception as e:
            logger.warning(f"Error removing email headers: {e}")
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove email signatures (simple approach)
    signature_patterns = [
        r'Best regards,.*$',
        r'Sincerely,.*$',
        r'Regards,.*$',
        r'Thank you,.*$'
    ]
    
    for pattern in signature_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    return text.strip()


def format_api_response(input_email: str, masked_email: str, entities: List[Dict[str, Any]], category: str) -> Dict[str, Any]:
    """
    Format the API response according to the required structure.
    
    Args:
        input_email: Original email text
        masked_email: Email with PII masked
        entities: List of detected PII entities
        category: Classified email category
        
    Returns:
        Dictionary with formatted API response
    """
    return {
        "input_email_body": input_email,
        "list_of_masked_entities": entities,
        "masked_email": masked_email,
        "category_of_the_email": category
    }


# Add this ContextAwarePIIMasker class to utils.py
class ContextAwarePIIMasker(PIIMasker):
    def __init__(self):
        super().__init__()
        self.verification_models = {
            entity_type: pipeline("text-classification",
                                  model=f"your-pii-verification-{entity_type}-model")
            for entity_type in PII_ENTITIES.values()
        }

    def _verify_pii(self, text: str, entity_type: str, entity_value: str) -> bool:
        inputs = f"Is this a {entity_type}? Text: {entity_value} Context: {text}"
        result = self.verification_models[entity_type](inputs)
        return result[0]['label'] == 'True' and result[0]['score'] > 0.7

    def detect_pii(self, text: str) -> List[Dict[str, Any]]:
        initial_matches = super().detect_pii(text)
        verified_matches = []

        for match in initial_matches:
            if self._verify_pii(text, match["classification"], match["entity"]):
                verified_matches.append(match)

        return verified_matches
