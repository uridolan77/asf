"""
Data Preprocessing and Validation Pipeline for ML Models.

This module provides robust data processing pipelines for medical literature data,
including cleaning, normalization, validation, and feature extraction to enhance
ML model performance.
"""

import re
import json
import hashlib
import datetime
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional, Union, Set, Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

from asf.medical.core.logging_config import get_logger
from asf.medical.core.exceptions import ValidationError, PreprocessingError

logger = get_logger(__name__)

class DataType(str, Enum):
    """Types of data handled by preprocessing pipelines."""
    TEXT = "text"
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    DATE = "date"
    BOOLEAN = "boolean"
    STRUCTURED = "structured"
    MIXED = "mixed"

class DataQuality(BaseModel):
    """Data quality metrics for a dataset or field."""
    completeness: float = 0.0
    accuracy: Optional[float] = None
    consistency: Optional[float] = None
    outlier_rate: Optional[float] = None
    noise_level: Optional[float] = None
    validation_errors: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        extra = "allow"

class FieldSchema(BaseModel):
    """Schema for a data field in the pipeline."""
    name: str
    data_type: DataType
    required: bool = True
    default_value: Optional[Any] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    allowed_values: Optional[List[Any]] = None
    regex_pattern: Optional[str] = None
    is_identifier: bool = False
    preprocessing_steps: List[str] = Field(default_factory=list)
    quality_metrics: Optional[DataQuality] = None
    semantic_type: Optional[str] = None
    description: Optional[str] = None

class DataSchema(BaseModel):
    """Schema for a complete dataset in the pipeline."""
    name: str
    version: str = "1.0.0"
    description: Optional[str] = None
    fields: Dict[str, FieldSchema]
    primary_key: Optional[str] = None
    foreign_keys: Dict[str, str] = Field(default_factory=dict)
    constraints: List[Dict[str, Any]] = Field(default_factory=list)
    quality_metrics: Optional[DataQuality] = None
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    updated_at: Optional[datetime.datetime] = None
    created_by: Optional[str] = None

class ProcessedData(BaseModel):
    """Container for processed data with quality metrics."""
    data: Dict[str, Any]
    schema: DataSchema
    validation_results: Dict[str, Any]
    quality_metrics: DataQuality
    processing_timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    processing_steps: List[Dict[str, Any]] = Field(default_factory=list)
    data_hash: Optional[str] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.data_hash:
            self.data_hash = self._compute_data_hash()
    
    def _compute_data_hash(self) -> str:
        """Compute a hash of the data contents."""
        data_str = json.dumps(self.data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()

class PreprocessingStep:
    """Base class for data preprocessing steps."""
    
    def __init__(self, name: str):
        """
        Initialize the preprocessing step.
        
        Args:
            name: Name of the preprocessing step.
        """
        self.name = name
        
    def process(self, data: Any, field_schema: Optional[FieldSchema] = None) -> Any:
        """
        Process the data.
        
        Args:
            data: Data to process.
            field_schema: Optional schema for the field.
            
        Returns:
            Processed data.
            
        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Process method must be implemented by subclasses")
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the preprocessing step.
        
        Returns:
            Dict with metadata about the preprocessing step.
        """
        return {"name": self.name}

class TextNormalization(PreprocessingStep):
    """Normalize text by removing extra whitespace, lowercasing, etc."""
    
    def __init__(
        self, 
        lowercase: bool = True, 
        remove_punctuation: bool = False,
        remove_numbers: bool = False,
        remove_stopwords: bool = False,
        stopwords: Optional[List[str]] = None
    ):
        """
        Initialize the text normalization step.
        
        Args:
            lowercase: Whether to convert text to lowercase.
            remove_punctuation: Whether to remove punctuation.
            remove_numbers: Whether to remove numbers.
            remove_stopwords: Whether to remove stopwords.
            stopwords: List of stopwords to remove. If None and remove_stopwords is True,
                uses a default English stopwords list.
        """
        super().__init__("text_normalization")
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        
        # Default English stopwords if needed and not provided
        self.stopwords = stopwords
        if remove_stopwords and not stopwords:
            self.stopwords = [
                "a", "an", "the", "and", "but", "or", "for", "nor", "on", "at",
                "to", "from", "by", "in", "out", "with", "about", "of", "as"
            ]
    
    def process(self, data: str, field_schema: Optional[FieldSchema] = None) -> str:
        """
        Process text data with normalization steps.
        
        Args:
            data: Text data to normalize.
            field_schema: Optional schema for the field.
            
        Returns:
            Normalized text.
        """
        if not isinstance(data, str):
            if data is None:
                return ""
            # Try to convert to string
            data = str(data)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', data).strip()
        
        # Convert to lowercase if needed
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation if needed
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Remove numbers if needed
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove stopwords if needed
        if self.remove_stopwords and self.stopwords:
            words = text.split()
            text = ' '.join(word for word in words if word not in self.stopwords)
        
        return text
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the text normalization step."""
        return {
            "name": self.name,
            "lowercase": self.lowercase,
            "remove_punctuation": self.remove_punctuation,
            "remove_numbers": self.remove_numbers,
            "remove_stopwords": self.remove_stopwords
        }

class MissingValueHandler(PreprocessingStep):
    """Handle missing values in data."""
    
    def __init__(
        self,
        strategy: str = "default",
        default_value: Any = None,
        fill_strategy: Optional[str] = None
    ):
        """
        Initialize the missing value handler.
        
        Args:
            strategy: Strategy for handling missing values.
                'default': Replace with default_value.
                'mean': Replace with the mean (for numeric data).
                'median': Replace with the median (for numeric data).
                'mode': Replace with the most common value.
                'drop': Return None (field will be dropped by the pipeline).
            default_value: Default value to use with 'default' strategy.
            fill_strategy: Additional strategies for numeric data ('mean', 'median', 'mode').
        """
        super().__init__("missing_value_handler")
        self.strategy = strategy
        self.default_value = default_value
        self.fill_strategy = fill_strategy
        
        # Statistics computed from data for fill strategies
        self.stats = {}
    
    def compute_stats(self, data_series):
        """Compute statistics for fill strategy."""
        if isinstance(data_series, (list, tuple, np.ndarray)):
            data_series = pd.Series(data_series)
        
        if not isinstance(data_series, pd.Series):
            # Can't compute stats for non-series data
            return
        
        non_null = data_series.dropna()
        
        if len(non_null) == 0:
            return
        
        if pd.api.types.is_numeric_dtype(non_null):
            self.stats['mean'] = float(non_null.mean())
            self.stats['median'] = float(non_null.median())
            
        if len(non_null) > 0:
            mode_vals = non_null.mode()
            if len(mode_vals) > 0:
                self.stats['mode'] = mode_vals[0]
    
    def process(self, data: Any, field_schema: Optional[FieldSchema] = None) -> Any:
        """
        Process data by handling missing values.
        
        Args:
            data: Data to process.
            field_schema: Optional schema for the field.
            
        Returns:
            Processed data with missing values handled.
        """
        # Check if the data is missing
        if data is None or (isinstance(data, str) and not data.strip()):
            # Handle based on strategy
            if self.strategy == 'drop':
                return None
            
            if self.strategy == 'default':
                # Use schema default if available and no explicit default given
                if self.default_value is None and field_schema and field_schema.default_value is not None:
                    return field_schema.default_value
                return self.default_value
            
            if self.fill_strategy in self.stats:
                return self.stats[self.fill_strategy]
            
            # Fallback to default if no other strategy works
            return self.default_value
        
        return data
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the missing value handler step."""
        return {
            "name": self.name,
            "strategy": self.strategy,
            "default_value": self.default_value,
            "fill_strategy": self.fill_strategy,
            "stats": self.stats
        }

class Validator(PreprocessingStep):
    """Validate data against field schema."""
    
    def __init__(self, raise_on_error: bool = False):
        """
        Initialize the validator.
        
        Args:
            raise_on_error: Whether to raise an exception on validation error.
        """
        super().__init__("validator")
        self.raise_on_error = raise_on_error
        self.errors = []
    
    def process(self, data: Any, field_schema: Optional[FieldSchema] = None) -> Any:
        """
        Validate data against schema.
        
        Args:
            data: Data to validate.
            field_schema: Schema for the field.
            
        Returns:
            The original data.
            
        Raises:
            ValidationError: If raise_on_error is True and validation fails.
        """
        if field_schema is None:
            return data
        
        self.errors = []
        
        # Required field check
        if field_schema.required and (data is None or (isinstance(data, str) and not data.strip())):
            msg = f"Field '{field_schema.name}' is required but is missing"
            self.errors.append({"field": field_schema.name, "error": msg})
            if self.raise_on_error:
                raise ValidationError(msg)
            return data
        
        # Skip further validation if data is None
        if data is None:
            return data
        
        # Type checking
        if field_schema.data_type == DataType.NUMERICAL:
            if not isinstance(data, (int, float)) and not (isinstance(data, str) and data.replace('.', '', 1).isdigit()):
                msg = f"Field '{field_schema.name}' must be a number, got {type(data).__name__}"
                self.errors.append({"field": field_schema.name, "error": msg})
                if self.raise_on_error:
                    raise ValidationError(msg)
            else:
                # Convert string to number if needed
                if isinstance(data, str):
                    try:
                        if '.' in data:
                            data = float(data)
                        else:
                            data = int(data)
                    except (ValueError, TypeError):
                        pass
                
                # Check min/max value
                if field_schema.min_value is not None and data < field_schema.min_value:
                    msg = f"Field '{field_schema.name}' value {data} is less than minimum {field_schema.min_value}"
                    self.errors.append({"field": field_schema.name, "error": msg})
                    if self.raise_on_error:
                        raise ValidationError(msg)
                
                if field_schema.max_value is not None and data > field_schema.max_value:
                    msg = f"Field '{field_schema.name}' value {data} is greater than maximum {field_schema.max_value}"
                    self.errors.append({"field": field_schema.name, "error": msg})
                    if self.raise_on_error:
                        raise ValidationError(msg)
                    
        elif field_schema.data_type == DataType.TEXT:
            if not isinstance(data, str):
                # Try to convert to string
                try:
                    data = str(data)
                except:
                    msg = f"Field '{field_schema.name}' must be text, got {type(data).__name__}"
                    self.errors.append({"field": field_schema.name, "error": msg})
                    if self.raise_on_error:
                        raise ValidationError(msg)
                    return data
            
            # Check min/max length
            if field_schema.min_length is not None and len(data) < field_schema.min_length:
                msg = f"Field '{field_schema.name}' length {len(data)} is less than minimum {field_schema.min_length}"
                self.errors.append({"field": field_schema.name, "error": msg})
                if self.raise_on_error:
                    raise ValidationError(msg)
            
            if field_schema.max_length is not None and len(data) > field_schema.max_length:
                msg = f"Field '{field_schema.name}' length {len(data)} is greater than maximum {field_schema.max_length}"
                self.errors.append({"field": field_schema.name, "error": msg})
                if self.raise_on_error:
                    raise ValidationError(msg)
            
            # Check regex pattern
            if field_schema.regex_pattern and not re.match(field_schema.regex_pattern, data):
                msg = f"Field '{field_schema.name}' value does not match pattern {field_schema.regex_pattern}"
                self.errors.append({"field": field_schema.name, "error": msg})
                if self.raise_on_error:
                    raise ValidationError(msg)
                
        elif field_schema.data_type == DataType.CATEGORICAL:
            if field_schema.allowed_values and data not in field_schema.allowed_values:
                msg = f"Field '{field_schema.name}' value {data} not in allowed values {field_schema.allowed_values}"
                self.errors.append({"field": field_schema.name, "error": msg})
                if self.raise_on_error:
                    raise ValidationError(msg)
                    
        elif field_schema.data_type == DataType.DATE:
            if isinstance(data, str):
                # Try to parse as date
                try:
                    datetime.datetime.fromisoformat(data.replace('Z', '+00:00'))
                except ValueError:
                    try:
                        datetime.datetime.strptime(data, '%Y-%m-%d')
                    except ValueError:
                        msg = f"Field '{field_schema.name}' value {data} is not a valid date"
                        self.errors.append({"field": field_schema.name, "error": msg})
                        if self.raise_on_error:
                            raise ValidationError(msg)
        
        return data
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the validator step."""
        return {
            "name": self.name,
            "raise_on_error": self.raise_on_error,
            "errors": self.errors
        }

class Outlier(PreprocessingStep):
    """Detect and handle outliers in data."""
    
    def __init__(
        self,
        method: str = "zscore",
        threshold: float = 3.0,
        strategy: str = "clip"
    ):
        """
        Initialize the outlier detector and handler.
        
        Args:
            method: Method for detecting outliers.
                'zscore': Use z-score threshold.
                'iqr': Use interquartile range.
            threshold: Threshold for outlier detection.
                For 'zscore', values with z > threshold are outliers.
                For 'iqr', values outside Q1 - threshold*IQR or Q3 + threshold*IQR are outliers.
            strategy: Strategy for handling outliers.
                'clip': Clip to the threshold value.
                'remove': Replace with None.
                'mean': Replace with mean.
                'median': Replace with median.
                'keep': Keep outliers but mark them.
        """
        super().__init__("outlier_handler")
        self.method = method
        self.threshold = threshold
        self.strategy = strategy
        
        # Statistics for outlier detection
        self.stats = {}
        self.outlier_indices = []
    
    def compute_stats(self, data_series):
        """Compute statistics for outlier detection."""
        if isinstance(data_series, (list, tuple)):
            data_series = np.array(data_series)
        
        if isinstance(data_series, np.ndarray):
            # Convert to pandas series for easier stats
            data_series = pd.Series(data_series)
        
        if not isinstance(data_series, pd.Series):
            # Can't compute stats for non-series data
            return
        
        numeric_data = pd.to_numeric(data_series, errors='coerce')
        numeric_data = numeric_data.dropna()
        
        if len(numeric_data) < 2:
            return
        
        if self.method == 'zscore':
            mean = numeric_data.mean()
            std = numeric_data.std()
            if std == 0:
                return
                
            self.stats['mean'] = float(mean)
            self.stats['std'] = float(std)
            
            # Find outliers
            z_scores = (numeric_data - mean) / std
            self.outlier_indices = numeric_data.index[abs(z_scores) > self.threshold].tolist()
            
        elif self.method == 'iqr':
            q1 = numeric_data.quantile(0.25)
            q3 = numeric_data.quantile(0.75)
            iqr = q3 - q1
            
            self.stats['q1'] = float(q1)
            self.stats['q3'] = float(q3)
            self.stats['iqr'] = float(iqr)
            
            # Find outliers
            lower_bound = q1 - self.threshold * iqr
            upper_bound = q3 + self.threshold * iqr
            self.outlier_indices = numeric_data.index[(numeric_data < lower_bound) | (numeric_data > upper_bound)].tolist()
            
        # Compute replacement values if needed
        if self.strategy == 'mean':
            self.stats['replacement'] = float(numeric_data.mean())
        elif self.strategy == 'median':
            self.stats['replacement'] = float(numeric_data.median())
    
    def process(self, data: Any, field_schema: Optional[FieldSchema] = None) -> Any:
        """
        Process data by detecting and handling outliers.
        
        Args:
            data: Data to process.
            field_schema: Optional schema for the field.
            
        Returns:
            Processed data with outliers handled.
        """
        # Only process numeric data
        if not isinstance(data, (int, float)):
            try:
                data = float(data)
            except (ValueError, TypeError):
                return data
        
        # Check if we have the necessary stats
        if not self.stats:
            return data
        
        is_outlier = False
        
        # Detect outlier
        if self.method == 'zscore':
            if 'mean' in self.stats and 'std' in self.stats and self.stats['std'] > 0:
                z_score = abs((data - self.stats['mean']) / self.stats['std'])
                is_outlier = z_score > self.threshold
                
        elif self.method == 'iqr':
            if 'q1' in self.stats and 'q3' in self.stats and 'iqr' in self.stats:
                lower_bound = self.stats['q1'] - self.threshold * self.stats['iqr']
                upper_bound = self.stats['q3'] + self.threshold * self.stats['iqr']
                is_outlier = data < lower_bound or data > upper_bound
        
        # Handle outlier if detected
        if is_outlier:
            if self.strategy == 'clip':
                if self.method == 'zscore':
                    max_value = self.stats['mean'] + self.threshold * self.stats['std']
                    min_value = self.stats['mean'] - self.threshold * self.stats['std']
                    return max(min_value, min(max_value, data))
                elif self.method == 'iqr':
                    lower_bound = self.stats['q1'] - self.threshold * self.stats['iqr']
                    upper_bound = self.stats['q3'] + self.threshold * self.stats['iqr']
                    return max(lower_bound, min(upper_bound, data))
            
            elif self.strategy == 'remove':
                return None
                
            elif self.strategy == 'mean':
                return self.stats.get('replacement', data)
                
            elif self.strategy == 'median':
                return self.stats.get('replacement', data)
        
        return data
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the outlier handler step."""
        return {
            "name": self.name,
            "method": self.method,
            "threshold": self.threshold,
            "strategy": self.strategy,
            "stats": self.stats,
            "outlier_count": len(self.outlier_indices)
        }

class Tokenizer(PreprocessingStep):
    """Tokenize text into words or subwords."""
    
    def __init__(
        self,
        mode: str = "word",
        lowercase: bool = True,
        remove_punctuation: bool = True,
        preserve_case_entities: bool = True
    ):
        """
        Initialize the tokenizer.
        
        Args:
            mode: Tokenization mode. One of:
                'word': Split on whitespace and punctuation.
                'sentence': Split into sentences.
                'character': Split into characters.
            lowercase: Whether to lowercase tokens.
            remove_punctuation: Whether to remove punctuation.
            preserve_case_entities: Whether to preserve case for potential named entities.
        """
        super().__init__("tokenizer")
        self.mode = mode
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.preserve_case_entities = preserve_case_entities
        
        # Pattern for finding potential named entities (capitalized words)
        self.entity_pattern = re.compile(r'(?<!\.\s)(?:[A-Z][a-z]+)')
        
        # Sentence tokenization pattern
        self.sentence_pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
    
    def process(self, data: Any, field_schema: Optional[FieldSchema] = None) -> List[str]:
        """
        Process text data by tokenizing.
        
        Args:
            data: Text data to tokenize.
            field_schema: Optional schema for the field.
            
        Returns:
            List of tokens.
        """
        if not isinstance(data, str):
            if data is None:
                return []
            # Try to convert to string
            data = str(data)
        
        # Find potential named entities to preserve case if needed
        entities = set()
        if self.preserve_case_entities:
            entities = set(self.entity_pattern.findall(data))
        
        # Lowercase if needed
        if self.lowercase:
            # Only lowercase non-entities
            if self.preserve_case_entities and entities:
                for entity in entities:
                    data = data.replace(entity, f"__ENTITY_{hash(entity)}__")
                data = data.lower()
                for entity in entities:
                    data = data.replace(f"__entity_{hash(entity)}__", entity)
            else:
                data = data.lower()
        
        # Remove punctuation if needed
        if self.remove_punctuation:
            # Replace punctuation with space
            data = re.sub(r'[^\w\s]', ' ', data)
            # Normalize whitespace
            data = re.sub(r'\s+', ' ', data).strip()
        
        # Tokenize based on mode
        if self.mode == "word":
            tokens = data.split()
        elif self.mode == "sentence":
            tokens = self.sentence_pattern.split(data)
            tokens = [token.strip() for token in tokens if token.strip()]
        elif self.mode == "character":
            tokens = list(data)
        else:
            # Default to word tokenization
            tokens = data.split()
        
        return tokens
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the tokenizer step."""
        return {
            "name": self.name,
            "mode": self.mode,
            "lowercase": self.lowercase,
            "remove_punctuation": self.remove_punctuation,
            "preserve_case_entities": self.preserve_case_entities
        }

class DataPipeline:
    """
    Pipeline for preprocessing and validating data before ML model ingestion.
    
    This pipeline handles:
    1. Schema validation
    2. Preprocessing steps (normalization, tokenization, etc.)
    3. Missing value handling
    4. Outlier detection and handling
    5. Data quality metrics calculation
    """
    
    def __init__(
        self,
        schema: DataSchema,
        preprocessing_steps: Optional[Dict[str, List[PreprocessingStep]]] = None,
        validate_schema: bool = True,
        raise_on_error: bool = False
    ):
        """
        Initialize the data pipeline.
        
        Args:
            schema: Schema defining the expected data structure.
            preprocessing_steps: Dictionary mapping field names to preprocessing steps.
                If None, default steps will be created based on field schemas.
            validate_schema: Whether to validate the schema itself.
            raise_on_error: Whether to raise exceptions on validation errors.
        """
        self.schema = schema
        self.raise_on_error = raise_on_error
        self.preprocessing_steps = preprocessing_steps or {}
        
        # Create default preprocessing steps for fields that don't have custom steps
        self._create_default_preprocessing_steps()
        
        # Schema validation
        if validate_schema:
            self._validate_schema()
    
    def _validate_schema(self):
        """Validate the schema itself."""
        # Ensure primary key exists if specified
        if self.schema.primary_key and self.schema.primary_key not in self.schema.fields:
            raise ValidationError(f"Primary key '{self.schema.primary_key}' not found in schema fields")
        
        # Ensure foreign keys exist
        for field, referenced in self.schema.foreign_keys.items():
            if field not in self.schema.fields:
                raise ValidationError(f"Foreign key '{field}' not found in schema fields")
        
        # Basic field validation
        for field_name, field in self.schema.fields.items():
            if field.name != field_name:
                raise ValidationError(f"Field name mismatch: '{field.name}' vs '{field_name}'")
    
    def _create_default_preprocessing_steps(self):
        """Create default preprocessing steps based on field schemas."""
        for field_name, field in self.schema.fields.items():
            # Skip fields that already have custom steps
            if field_name in self.preprocessing_steps:
                continue
            
            steps = []
            
            # Add validator as the first step
            steps.append(Validator(raise_on_error=self.raise_on_error))
            
            # Add missing value handler as the second step
            if not field.required:
                steps.append(MissingValueHandler(
                    strategy="default", 
                    default_value=field.default_value
                ))
            
            # Add appropriate processing steps based on field type
            if field.data_type == DataType.TEXT:
                steps.append(TextNormalization(
                    lowercase=True,
                    remove_punctuation=False
                ))
                
            elif field.data_type == DataType.NUMERICAL:
                steps.append(Outlier(
                    method="zscore",
                    threshold=3.0,
                    strategy="clip"
                ))
            
            # Store the steps
            self.preprocessing_steps[field_name] = steps
    
    def process_field(
        self, 
        field_name: str, 
        value: Any, 
        compute_stats: bool = True
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        """
        Process a single field with its preprocessing steps.
        
        Args:
            field_name: Name of the field to process.
            value: Value to process.
            compute_stats: Whether to compute statistics for preprocessing steps.
            
        Returns:
            Tuple of (processed_value, processing_info) with processing metadata.
        """
        if field_name not in self.schema.fields:
            if self.raise_on_error:
                raise ValidationError(f"Field '{field_name}' not found in schema")
            return value, [{"error": f"Field '{field_name}' not found in schema"}]
        
        field_schema = self.schema.fields[field_name]
        steps = self.preprocessing_steps.get(field_name, [])
        
        # Track processing info
        processing_info = []
        processed_value = value
        
        # Apply each preprocessing step in sequence
        for step in steps:
            step_start_time = datetime.datetime.now()
            
            # Compute stats if needed
            if compute_stats and hasattr(step, "compute_stats"):
                step.compute_stats(value)
            
            # Apply the step
            try:
                processed_value = step.process(processed_value, field_schema)
                
                # Track step information
                processing_info.append({
                    "step": step.name,
                    "metadata": step.get_metadata(),
                    "success": True,
                    "duration_ms": (datetime.datetime.now() - step_start_time).total_seconds() * 1000
                })
                
            except Exception as e:
                # Log the error
                logger.error(f"Error in {step.name} for field {field_name}: {str(e)}")
                
                # Track error information
                processing_info.append({
                    "step": step.name,
                    "metadata": step.get_metadata(),
                    "success": False,
                    "error": str(e),
                    "duration_ms": (datetime.datetime.now() - step_start_time).total_seconds() * 1000
                })
                
                # Raise if configured to do so
                if self.raise_on_error:
                    raise PreprocessingError(f"Error in {step.name} for field {field_name}: {str(e)}")
                
                # Stop processing this field on error
                break
        
        return processed_value, processing_info
    
    def process(self, data: Dict[str, Any]) -> ProcessedData:
        """
        Process a complete data record through the pipeline.
        
        Args:
            data: Dictionary of field values to process.
            
        Returns:
            ProcessedData containing processed data and quality metrics.
        """
        processed_data = {}
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "field_results": {}
        }
        
        all_processing_steps = []
        
        # Process each field
        for field_name, field_schema in self.schema.fields.items():
            # Track required fields
            if field_schema.required and field_name not in data:
                validation_results["valid"] = False
                validation_results["errors"].append({
                    "field": field_name,
                    "error": f"Required field '{field_name}' is missing"
                })
                if self.raise_on_error:
                    raise ValidationError(f"Required field '{field_name}' is missing")
                continue
            
            # Process the field if it exists
            if field_name in data:
                value, processing_info = self.process_field(field_name, data[field_name])
                processed_data[field_name] = value
                validation_results["field_results"][field_name] = {
                    "valid": all(step["success"] for step in processing_info),
                    "steps": processing_info
                }
                
                # Add any validation errors
                for step_info in processing_info:
                    if not step_info["success"]:
                        validation_results["valid"] = False
                        validation_results["errors"].append({
                            "field": field_name,
                            "error": step_info.get("error", "Unknown error")
                        })
                
                # Track processing steps
                for step_info in processing_info:
                    all_processing_steps.append({
                        "field": field_name,
                        **step_info
                    })
        
        # Calculate data quality metrics
        quality_metrics = self._calculate_quality_metrics(processed_data, validation_results)
        
        # Create the processed data result
        result = ProcessedData(
            data=processed_data,
            schema=self.schema,
            validation_results=validation_results,
            quality_metrics=quality_metrics,
            processing_steps=all_processing_steps
        )
        
        return result
    
    def _calculate_quality_metrics(
        self, 
        processed_data: Dict[str, Any],
        validation_results: Dict[str, Any]
    ) -> DataQuality:
        """
        Calculate data quality metrics for the processed data.
        
        Args:
            processed_data: Processed data dictionary.
            validation_results: Validation results from processing.
            
        Returns:
            DataQuality with calculated metrics.
        """
        # Calculate completeness
        total_fields = len(self.schema.fields)
        available_fields = sum(1 for field in self.schema.fields if field in processed_data)
        completeness = available_fields / total_fields if total_fields > 0 else 0.0
        
        # Count validation errors
        validation_errors = validation_results["errors"]
        
        # Create quality metrics
        quality_metrics = DataQuality(
            completeness=completeness,
            validation_errors=validation_errors
        )
        
        # TODO: Calculate more advanced metrics like consistency, noise level, etc.
        
        return quality_metrics

class TextPreprocessingPipeline(DataPipeline):
    """
    Specialized pipeline for text data preprocessing for NLP tasks.
    
    Extends the base DataPipeline with text-specific preprocessing steps.
    """
    
    def __init__(
        self, 
        schema: DataSchema,
        text_fields: Optional[List[str]] = None,
        preprocessing_steps: Optional[Dict[str, List[PreprocessingStep]]] = None,
        validate_schema: bool = True,
        raise_on_error: bool = False,
        use_spacy: bool = False
    ):
        """
        Initialize the text preprocessing pipeline.
        
        Args:
            schema: Schema defining the expected data structure.
            text_fields: List of field names to treat as text.
                If None, all fields with DataType.TEXT are used.
            preprocessing_steps: Dictionary mapping field names to preprocessing steps.
            validate_schema: Whether to validate the schema itself.
            raise_on_error: Whether to raise exceptions on validation errors.
            use_spacy: Whether to use spaCy for NLP preprocessing.
        """
        self.use_spacy = use_spacy
        self.nlp = None
        
        # Initialize spaCy if requested
        if use_spacy:
            try:
                import spacy
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning("Could not load en_core_web_sm, falling back to basic tokenizer")
            except ImportError:
                logger.warning("spaCy not available, falling back to basic tokenizer")
        
        # Determine text fields
        self.text_fields = text_fields or [
            name for name, field in schema.fields.items()
            if field.data_type == DataType.TEXT
        ]
        
        # Create preprocessing steps for text fields
        if preprocessing_steps is None:
            preprocessing_steps = {}
            
        # Add text-specific steps for text fields
        for field_name in self.text_fields:
            if field_name in schema.fields and field_name not in preprocessing_steps:
                # Start with validator and missing value handler
                steps = [
                    Validator(raise_on_error=raise_on_error),
                    MissingValueHandler(strategy="default", default_value="")
                ]
                
                # Add text normalization
                steps.append(TextNormalization(
                    lowercase=True,
                    remove_punctuation=False,
                    remove_stopwords=False
                ))
                
                # Add tokenizer
                steps.append(Tokenizer(
                    mode="word",
                    lowercase=True,
                    remove_punctuation=False,
                    preserve_case_entities=True
                ))
                
                preprocessing_steps[field_name] = steps
        
        # Initialize base pipeline
        super().__init__(schema, preprocessing_steps, validate_schema, raise_on_error)
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process a single text string with NLP pipeline.
        
        Args:
            text: Text to process.
            
        Returns:
            Dictionary with processed text features.
        """
        if not text:
            return {"tokens": [], "entities": [], "sentences": []}
        
        # Use spaCy if available
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            return {
                "tokens": [token.text for token in doc],
                "lemmas": [token.lemma_ for token in doc],
                "pos_tags": [token.pos_ for token in doc],
                "entities": [(ent.text, ent.label_) for ent in doc.ents],
                "sentences": [sent.text for sent in doc.sents],
                "noun_chunks": [chunk.text for chunk in doc.noun_chunks]
            }
        else:
            # Fallback to basic processing
            # Normalize
            normalized = re.sub(r'\s+', ' ', text).strip()
            
            # Simple sentence splitting
            sentence_pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
            sentences = sentence_pattern.split(normalized)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Simple tokenization
            tokens = normalized.split()
            
            return {
                "tokens": tokens,
                "entities": [],  # No entity recognition in basic mode
                "sentences": sentences
            }
    
    def process(self, data: Dict[str, Any]) -> ProcessedData:
        """
        Process a complete data record through the NLP pipeline.
        
        Args:
            data: Dictionary of field values to process.
            
        Returns:
            ProcessedData containing processed data and NLP features.
        """
        # Process with base pipeline first
        result = super().process(data)
        
        # Add NLP features for text fields
        nlp_features = {}
        
        for field_name in self.text_fields:
            if field_name in result.data:
                text_value = result.data.get(field_name)
                if isinstance(text_value, str):
                    nlp_features[field_name] = self.process_text(text_value)
        
        # Add NLP features to the result
        if nlp_features:
            result.data["nlp_features"] = nlp_features
        
        return result

class DataPipelineFactory:
    """Factory for creating data pipelines for different types of data."""
    
    @staticmethod
    def create_pipeline(
        schema: DataSchema,
        pipeline_type: str = "default",
        preprocessing_steps: Optional[Dict[str, List[PreprocessingStep]]] = None,
        **kwargs
    ) -> DataPipeline:
        """
        Create a data pipeline based on type.
        
        Args:
            schema: Schema defining the expected data structure.
            pipeline_type: Type of pipeline to create.
                'default': Basic data pipeline.
                'text': Text preprocessing pipeline for NLP.
                'tabular': Pipeline for tabular data.
                'time_series': Pipeline for time series data.
            preprocessing_steps: Dictionary mapping field names to preprocessing steps.
            **kwargs: Additional arguments for the specific pipeline type.
            
        Returns:
            DataPipeline instance of the requested type.
        """
        if pipeline_type == "text":
            return TextPreprocessingPipeline(
                schema=schema,
                preprocessing_steps=preprocessing_steps,
                **kwargs
            )
        elif pipeline_type == "tabular":
            # TODO: Implement specialized tabular data pipeline
            return DataPipeline(
                schema=schema,
                preprocessing_steps=preprocessing_steps,
                **kwargs
            )
        elif pipeline_type == "time_series":
            # TODO: Implement specialized time series pipeline
            return DataPipeline(
                schema=schema,
                preprocessing_steps=preprocessing_steps,
                **kwargs
            )
        else:
            # Default pipeline
            return DataPipeline(
                schema=schema,
                preprocessing_steps=preprocessing_steps,
                **kwargs
            )

# Example schemas for common medical data types
def create_medical_claim_schema() -> DataSchema:
    """
    Create a schema for medical claims data.
    
    Returns:
        DataSchema for medical claims.
    """
    return DataSchema(
        name="medical_claim",
        version="1.0.0",
        description="Schema for medical claims from literature",
        fields={
            "claim_id": FieldSchema(
                name="claim_id",
                data_type=DataType.TEXT,
                required=True,
                is_identifier=True,
                description="Unique identifier for the claim"
            ),
            "claim_text": FieldSchema(
                name="claim_text",
                data_type=DataType.TEXT,
                required=True,
                min_length=10,
                description="The text of the medical claim"
            ),
            "publication_id": FieldSchema(
                name="publication_id",
                data_type=DataType.TEXT,
                required=True,
                description="Identifier for the source publication"
            ),
            "publication_date": FieldSchema(
                name="publication_date",
                data_type=DataType.DATE,
                required=False,
                description="Date the claim was published"
            ),
            "evidence_level": FieldSchema(
                name="evidence_level",
                data_type=DataType.CATEGORICAL,
                required=False,
                allowed_values=["high", "moderate", "low", "very_low", "unknown"],
                default_value="unknown",
                description="Level of evidence supporting the claim"
            ),
            "confidence_score": FieldSchema(
                name="confidence_score",
                data_type=DataType.NUMERICAL,
                required=False,
                min_value=0.0,
                max_value=1.0,
                description="Confidence score assigned to the claim (0-1)"
            ),
            "study_design": FieldSchema(
                name="study_design",
                data_type=DataType.CATEGORICAL,
                required=False,
                allowed_values=[
                    "meta_analysis", "systematic_review", "randomized_controlled_trial",
                    "cohort_study", "case_control", "case_series", "case_report",
                    "expert_opinion", "other", "unknown"
                ],
                default_value="unknown",
                description="Study design of the source publication"
            ),
            "sample_size": FieldSchema(
                name="sample_size",
                data_type=DataType.NUMERICAL,
                required=False,
                min_value=0,
                description="Sample size of the study"
            ),
            "medical_domain": FieldSchema(
                name="medical_domain",
                data_type=DataType.TEXT,
                required=False,
                description="Medical domain or specialty related to the claim"
            ),
            "keywords": FieldSchema(
                name="keywords",
                data_type=DataType.TEXT,
                required=False,
                description="Keywords associated with the claim"
            )
        },
        primary_key="claim_id"
    )

# Example usage of the data pipeline
def example_pipeline_usage():
    """
    Example usage of the data pipeline for processing medical claims.
    """
    # Create a schema for medical claims
    schema = create_medical_claim_schema()
    
    # Create a text preprocessing pipeline for claims
    pipeline = DataPipelineFactory.create_pipeline(
        schema=schema,
        pipeline_type="text",
        text_fields=["claim_text", "medical_domain", "keywords"],
        use_spacy=True  # This will fall back to basic processing if spaCy is not available
    )
    
    # Example claim data
    claim_data = {
        "claim_id": "CLAIM12345",
        "claim_text": "Statin therapy reduces the risk of cardiovascular events in patients with high cholesterol.",
        "publication_id": "PMID12345678",
        "publication_date": "2020-01-15",
        "evidence_level": "high",
        "confidence_score": 0.85,
        "study_design": "randomized_controlled_trial",
        "sample_size": 5000,
        "medical_domain": "Cardiology",
        "keywords": "statins, cardiovascular disease, cholesterol, prevention"
    }
    
    # Process the claim
    result = pipeline.process(claim_data)
    
    # The result contains processed data, validation results, and quality metrics
    print(f"Processed data for claim {result.data['claim_id']}")
    print(f"Validation result: {result.validation_results['valid']}")
    if not result.validation_results['valid']:
        print(f"Validation errors: {result.validation_results['errors']}")
    print(f"Data quality completeness: {result.quality_metrics.completeness}")
    
    # NLP features are available for text fields
    if "nlp_features" in result.data:
        claim_nlp = result.data["nlp_features"].get("claim_text", {})
        print(f"Tokens for claim: {claim_nlp.get('tokens', [])[:5]}...")
        print(f"Entities in claim: {claim_nlp.get('entities', [])}")
        
    return result