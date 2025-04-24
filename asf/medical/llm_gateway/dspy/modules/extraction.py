"""
Data Extraction Module

This module provides DSPy modules for extracting structured data from unstructured text.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Union, Type, TypeVar, Generic
import dspy

from .base import DSPyModuleBase
from .enhanced_base import EnhancedDSPyModuleBase

# Set up logging
logger = logging.getLogger(__name__)


class DataExtractionSignature(dspy.Signature):
    """Signature for extracting data from text."""
    
    text = dspy.InputField(desc="Text to extract data from")
    schema = dspy.InputField(desc="JSON schema or description of data to extract")
    extracted_data = dspy.OutputField(desc="Extracted data in JSON format")
    confidence = dspy.OutputField(desc="Confidence in extraction (high, medium, low)")


class StructuredDataExtractionSignature(dspy.Signature):
    """Signature for structured data extraction with additional capabilities."""
    
    text = dspy.InputField(desc="Text to extract data from")
    schema = dspy.InputField(desc="JSON schema or description of data to extract")
    examples = dspy.InputField(desc="Examples of expected output format")
    extracted_data = dspy.OutputField(desc="Extracted data in JSON format")
    confidence = dspy.OutputField(desc="Confidence in extraction (high, medium, low)")
    missing_fields = dspy.OutputField(desc="Fields that could not be extracted from the text")
    reasoning = dspy.OutputField(desc="Reasoning for how the extraction was performed")


class DataExtractor(DSPyModuleBase):
    """Module for extracting structured data from text."""
    
    def __init__(
        self,
        include_confidence: bool = True,
    ):
        """
        Initialize the data extractor.
        
        Args:
            include_confidence: Whether to include confidence score
        """
        super().__init__()
        self.include_confidence = include_confidence
        
        # Create predictor
        self.predictor = dspy.Predict(DataExtractionSignature)
    
    async def predict(
        self,
        text: str,
        schema: Union[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract data from text.
        
        Args:
            text: Text to extract data from
            schema: JSON schema or description of data to extract
            
        Returns:
            Dict[str, Any]: Extraction result
        """
        # Log the request
        logger.info(f"Extracting data from text: {text[:100]}...")
        
        try:
            # Convert schema to string if it's a dict
            schema_str = json.dumps(schema) if isinstance(schema, dict) else schema
            
            # Call predictor
            response = await self.call_predictor(
                self.predictor,
                text=text,
                schema=schema_str
            )
            
            # Process response
            result = {
                "text": text[:100] + "..." if len(text) > 100 else text,
                "schema": schema,
            }
            
            # Parse extracted_data if it's a string
            extracted_data = response.extracted_data
            if isinstance(extracted_data, str):
                try:
                    extracted_data = json.loads(extracted_data)
                except json.JSONDecodeError:
                    # Keep as string if it can't be parsed as JSON
                    pass
            
            result["extracted_data"] = extracted_data
            
            # Add confidence if requested
            if self.include_confidence:
                result["confidence"] = response.confidence
            
            return result
            
        except Exception as e:
            logger.error(f"Error in data extraction: {str(e)}")
            raise


class StructuredDataExtractor(EnhancedDSPyModuleBase):
    """Enhanced module for extracting structured data with additional capabilities."""
    
    def __init__(
        self,
        include_confidence: bool = True,
        include_missing_fields: bool = True,
        include_reasoning: bool = True,
        default_schema: Optional[Union[str, Dict[str, Any]]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize the structured data extractor.
        
        Args:
            include_confidence: Whether to include confidence score
            include_missing_fields: Whether to include missing fields
            include_reasoning: Whether to include extraction reasoning
            default_schema: Default schema to use if not provided
            examples: Examples of expected output format
        """
        super().__init__()
        self.include_confidence = include_confidence
        self.include_missing_fields = include_missing_fields
        self.include_reasoning = include_reasoning
        self.default_schema = default_schema
        self.examples = examples or []
        
        # Create predictor
        self.predictor = dspy.Predict(StructuredDataExtractionSignature)
    
    async def _predict_impl(
        self,
        text: str,
        schema: Optional[Union[str, Dict[str, Any]]] = None,
        examples: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Extract structured data from text with enhanced capabilities.
        
        Args:
            text: Text to extract data from
            schema: JSON schema or description of data to extract
            examples: Examples of expected output format
            
        Returns:
            Dict[str, Any]: Enhanced extraction result
        """
        # Use default schema if not provided
        schema = schema or self.default_schema
        if schema is None:
            raise ValueError("Schema must be provided")
        
        # Convert schema to string if it's a dict
        schema_str = json.dumps(schema) if isinstance(schema, dict) else schema
        
        # Use provided examples or default examples
        examples = examples or self.examples
        examples_str = json.dumps(examples) if examples else "[]"
        
        # Call predictor
        response = await self.call_predictor(
            self.predictor,
            text=text,
            schema=schema_str,
            examples=examples_str
        )
        
        # Process response
        result = {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "schema": schema,
        }
        
        # Parse extracted_data if it's a string
        extracted_data = response.extracted_data
        if isinstance(extracted_data, str):
            try:
                extracted_data = json.loads(extracted_data)
            except json.JSONDecodeError:
                # Keep as string if it can't be parsed as JSON
                pass
        
        result["extracted_data"] = extracted_data
        
        # Add optional fields
        if self.include_confidence:
            result["confidence"] = response.confidence
        
        if self.include_missing_fields:
            result["missing_fields"] = response.missing_fields
        
        if self.include_reasoning:
            result["reasoning"] = response.reasoning
        
        return result


class NamedEntityExtractor(DataExtractor):
    """Module for extracting named entities from text."""
    
    def __init__(self, entity_types: List[str], **kwargs):
        """
        Initialize the named entity extractor.
        
        Args:
            entity_types: Types of entities to extract (e.g., ["person", "organization", "location"])
            **kwargs: Additional arguments for DataExtractor
        """
        super().__init__(**kwargs)
        self.entity_types = entity_types
        
        # Create schema for named entities
        self.schema = {
            "type": "object",
            "description": f"Extract named entities of types: {', '.join(entity_types)}",
            "properties": {
                entity_type: {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "start": {"type": "integer"},
                            "end": {"type": "integer"},
                            "confidence": {"type": "string"}
                        }
                    }
                }
                for entity_type in entity_types
            }
        }
    
    async def predict(self, text: str) -> Dict[str, Any]:
        """
        Extract named entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dict[str, Any]: Extracted entities
        """
        return await super().predict(text, self.schema)


class KeyValueExtractor(DataExtractor):
    """Module for extracting key-value pairs from text."""
    
    def __init__(self, keys: List[str], **kwargs):
        """
        Initialize the key-value extractor.
        
        Args:
            keys: Keys to extract values for
            **kwargs: Additional arguments for DataExtractor
        """
        super().__init__(**kwargs)
        self.keys = keys
        
        # Create schema for key-value pairs
        self.schema = {
            "type": "object",
            "description": f"Extract values for keys: {', '.join(keys)}",
            "properties": {
                key: {"type": "string"} for key in keys
            }
        }
    
    async def predict(self, text: str) -> Dict[str, Any]:
        """
        Extract key-value pairs from text.
        
        Args:
            text: Text to extract key-value pairs from
            
        Returns:
            Dict[str, Any]: Extracted key-value pairs
        """
        return await super().predict(text, self.schema)


# Export
__all__ = [
    "DataExtractionSignature",
    "StructuredDataExtractionSignature",
    "DataExtractor",
    "StructuredDataExtractor",
    "NamedEntityExtractor",
    "KeyValueExtractor",
]