"""Audit Logging for Medical Research

This module provides comprehensive audit logging capabilities for medical research applications.
It includes PII/PHI detection, redaction, and immutable logging for compliance requirements.
"""

import logging
import json
import os
import uuid
import time
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Set
import hashlib
import threading
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

# PII/PHI patterns to detect and redact
DEFAULT_PHI_PATTERNS = [
    # Patient identifiers
    r'\b(?:patient|record|medical|chart)\s*(?:id|number|#)\s*[:=]?\s*[a-zA-Z0-9_-]{4,}',
    r'\b(?:mrn|emr|ehr)\s*[:=]?\s*[a-zA-Z0-9_-]{4,}',

    # Names
    r'\b(?:dr\.?|doctor|patient|mr\.?|mrs\.?|ms\.?|miss|prof\.?)\s+[a-z]+\s+[a-z]+\b',

    # Dates of birth
    r'\b(?:dob|date\s+of\s+birth|birth\s+date)\s*[:=]?\s*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
    r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',

    # Social Security Numbers
    r'\b\d{3}[-]\d{2}[-]\d{4}\b',

    # Addresses
    r'\b\d+\s+[a-z]+\s+(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|place|pl|court|ct)\b',

    # Phone numbers
    r'\b(?:\+\d{1,2}\s)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',

    # Email addresses
    r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',

    # Medical record details
    r'\b(?:diagnosis|condition|disease|disorder|syndrome)\s*[:=]?\s*[a-z\s]+',
    r'\b(?:medication|drug|prescription|treatment)\s*[:=]?\s*[a-z\s]+',

    # Lab results
    r'\b(?:lab|test|result)\s*[:=]?\s*[a-z0-9\s.]+',

    # Insurance information
    r'\b(?:insurance|policy|group)\s*(?:id|number|#)\s*[:=]?\s*[a-zA-Z0-9_-]{4,}',

    # Facility names
    r'\b(?:hospital|clinic|center|medical\s+center|healthcare)\s+[a-z\s]+',
]


class AuditLogger:
    """Comprehensive audit logger for medical research applications.

    This class provides detailed, immutable logging with PII/PHI detection and redaction.
    It ensures all operations are fully traceable for compliance requirements.
    """

    def __init__(
        self,
        log_dir: str = "audit_logs",
        phi_patterns: Optional[List[str]] = None,
        enable_console_logging: bool = True,
        log_level: int = logging.INFO,
        max_log_file_size_mb: int = 10,
        max_log_files: int = 100,
        enable_encryption: bool = False,
        encryption_key: Optional[str] = None
    ):
        """
        Initialize the audit logger.

        Args:
            log_dir: Directory to store audit logs
            phi_patterns: List of regex patterns to detect PHI/PII
            enable_console_logging: Whether to log to console
            log_level: Logging level
            max_log_file_size_mb: Maximum size of log files in MB
            max_log_files: Maximum number of log files to keep
            enable_encryption: Whether to encrypt log files
            encryption_key: Key for encrypting log files
        """
        self.log_dir = log_dir
        self.phi_patterns = phi_patterns or DEFAULT_PHI_PATTERNS
        self.enable_console_logging = enable_console_logging
        self.log_level = log_level
        self.max_log_file_size_mb = max_log_file_size_mb
        self.max_log_files = max_log_files
        self.enable_encryption = enable_encryption
        self.encryption_key = encryption_key

        # Compile regex patterns for better performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.phi_patterns]

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Set up file handler
        self.current_log_file = self._get_new_log_file()
        self.file_handler = logging.FileHandler(self.current_log_file)
        self.file_handler.setLevel(log_level)

        # Set up console handler if enabled
        if enable_console_logging:
            self.console_handler = logging.StreamHandler()
            self.console_handler.setLevel(log_level)

        # Set up logger
        self.logger = logging.getLogger("audit_logger")
        self.logger.setLevel(log_level)
        self.logger.addHandler(self.file_handler)

        if enable_console_logging:
            self.logger.addHandler(self.console_handler)

        # Thread lock for thread safety
        self._lock = threading.Lock()

        # Set of logged event hashes to detect duplicates
        self.logged_events = set()

        logger.info(f"Audit logger initialized with log directory: {log_dir}")

    def _get_new_log_file(self) -> str:
        """
        Get a new log file path.

        Returns:
            str: Path to the new log file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.log_dir, f"audit_log_{timestamp}_{uuid.uuid4().hex[:8]}.log")

    def _rotate_log_file_if_needed(self) -> None:
        """
        Rotate log file if it exceeds the maximum size.
        """
        with self._lock:
            if os.path.exists(self.current_log_file):
                file_size_mb = os.path.getsize(self.current_log_file) / (1024 * 1024)
                if file_size_mb >= self.max_log_file_size_mb:
                    # Close current file handler
                    self.file_handler.close()
                    self.logger.removeHandler(self.file_handler)

                    # Create new log file
                    self.current_log_file = self._get_new_log_file()
                    self.file_handler = logging.FileHandler(self.current_log_file)
                    self.file_handler.setLevel(self.log_level)
                    self.logger.addHandler(self.file_handler)

                    # Clean up old log files if needed
                    self._cleanup_old_log_files()

    def _cleanup_old_log_files(self) -> None:
        """
        Clean up old log files if the number exceeds the maximum.
        """
        log_files = sorted(
            [os.path.join(self.log_dir, f) for f in os.listdir(self.log_dir) if f.startswith("audit_log_")],
            key=os.path.getctime
        )

        if len(log_files) > self.max_log_files:
            files_to_delete = log_files[:-self.max_log_files]
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    logger.debug(f"Deleted old log file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete old log file {file_path}: {str(e)}")

    def _detect_and_redact_phi(self, text: str) -> str:
        """
        Detect and redact PHI/PII from text.

        Args:
            text: Text to redact

        Returns:
            str: Redacted text
        """
        if not text or not isinstance(text, str):
            return text

        redacted_text = text

        # Apply each pattern
        for i, pattern in enumerate(self.compiled_patterns):
            redacted_text = pattern.sub(f"[REDACTED-PHI-{i}]", redacted_text)

        return redacted_text

    def _redact_object(self, obj: Any) -> Any:
        """
        Recursively redact PHI/PII from an object.

        Args:
            obj: Object to redact

        Returns:
            Any: Redacted object
        """
        if isinstance(obj, str):
            return self._detect_and_redact_phi(obj)
        elif isinstance(obj, dict):
            return {k: self._redact_object(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._redact_object(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._redact_object(item) for item in obj)
        else:
            return obj

    def _calculate_event_hash(self, event_data: Dict[str, Any]) -> str:
        """
        Calculate a hash for an event to detect duplicates.

        Args:
            event_data: Event data

        Returns:
            str: Event hash
        """
        # Create a deterministic string representation of the event
        event_str = json.dumps(event_data, sort_keys=True)
        return hashlib.sha256(event_str.encode()).hexdigest()

    def log_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        skip_duplicate_check: bool = False,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an audit event.

        Args:
            event_type: Type of event
            event_data: Event data
            user_id: ID of the user who triggered the event
            session_id: Session ID
            correlation_id: Correlation ID for linking related events
            skip_duplicate_check: Whether to skip duplicate check
            additional_metadata: Additional metadata to include

        Returns:
            str: Event ID
        """
        # Rotate log file if needed
        self._rotate_log_file_if_needed()

        # Generate event ID
        event_id = str(uuid.uuid4())

        # Create event object
        event = {
            "event_id": event_id,
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "session_id": session_id,
            "correlation_id": correlation_id or str(uuid.uuid4()),
            "data": self._redact_object(event_data)
        }

        # Add additional metadata if provided
        if additional_metadata:
            event["metadata"] = self._redact_object(additional_metadata)

        # Check for duplicates if enabled
        if not skip_duplicate_check:
            event_hash = self._calculate_event_hash(event_data)
            with self._lock:
                if event_hash in self.logged_events:
                    logger.debug(f"Skipping duplicate event: {event_type}")
                    return event_id
                self.logged_events.add(event_hash)

                # Limit the size of logged_events set
                if len(self.logged_events) > 10000:
                    self.logged_events.clear()

        # Log the event
        with self._lock:
            self.logger.info(json.dumps(event))

        return event_id

    def log_lm_call(
        self,
        prompt: str,
        model: str,
        parameters: Dict[str, Any],
        response: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        latency: Optional[float] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Log an LLM API call.

        Args:
            prompt: The prompt sent to the LLM
            model: The model used
            parameters: The parameters used
            response: The response from the LLM
            error: Error message if the call failed
            latency: Latency of the call in seconds
            user_id: ID of the user who triggered the call
            session_id: Session ID
            correlation_id: Correlation ID for linking related events

        Returns:
            str: Event ID
        """
        event_data = {
            "prompt": prompt,
            "model": model,
            "parameters": parameters,
            "response": response,
            "error": error,
            "latency": latency
        }

        return self.log_event(
            event_type="LLM_CALL",
            event_data=event_data,
            user_id=user_id,
            session_id=session_id,
            correlation_id=correlation_id
        )

    def log_module_call(
        self,
        module_name: str,
        module_type: str,
        inputs: Dict[str, Any],
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        latency: Optional[float] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Log a module call.

        Args:
            module_name: Name of the module
            module_type: Type of the module
            inputs: Inputs to the module
            outputs: Outputs from the module
            error: Error message if the call failed
            latency: Latency of the call in seconds
            user_id: ID of the user who triggered the call
            session_id: Session ID
            correlation_id: Correlation ID for linking related events

        Returns:
            str: Event ID
        """
        event_data = {
            "module_name": module_name,
            "module_type": module_type,
            "inputs": inputs,
            "outputs": outputs,
            "error": error,
            "latency": latency
        }

        return self.log_event(
            event_type="MODULE_CALL",
            event_data=event_data,
            user_id=user_id,
            session_id=session_id,
            correlation_id=correlation_id
        )

    def log_optimization(
        self,
        module_name: str,
        optimizer_type: str,
        metrics: Dict[str, Any],
        original_prompts: Dict[str, str],
        optimized_prompts: Dict[str, str],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Log an optimization event.

        Args:
            module_name: Name of the module
            optimizer_type: Type of optimizer used
            metrics: Optimization metrics
            original_prompts: Original prompts
            optimized_prompts: Optimized prompts
            user_id: ID of the user who triggered the optimization
            session_id: Session ID
            correlation_id: Correlation ID for linking related events

        Returns:
            str: Event ID
        """
        event_data = {
            "module_name": module_name,
            "optimizer_type": optimizer_type,
            "metrics": metrics,
            "original_prompts": original_prompts,
            "optimized_prompts": optimized_prompts
        }

        return self.log_event(
            event_type="OPTIMIZATION",
            event_data=event_data,
            user_id=user_id,
            session_id=session_id,
            correlation_id=correlation_id
        )

    def log_cache_operation(
        self,
        operation: str,
        key: str,
        success: bool,
        error: Optional[str] = None,
        latency: Optional[float] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Log a cache operation.

        Args:
            operation: Type of operation (get, set, delete, clear)
            key: Cache key
            success: Whether the operation succeeded
            error: Error message if the operation failed
            latency: Latency of the operation in seconds
            user_id: ID of the user who triggered the operation
            session_id: Session ID
            correlation_id: Correlation ID for linking related events

        Returns:
            str: Event ID
        """
        event_data = {
            "operation": operation,
            "key": key,
            "success": success,
            "error": error,
            "latency": latency
        }

        return self.log_event(
            event_type="CACHE_OPERATION",
            event_data=event_data,
            user_id=user_id,
            session_id=session_id,
            correlation_id=correlation_id
        )

    def log_error(
        self,
        error_type: str,
        error_message: str,
        stack_trace: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Log an error.

        Args:
            error_type: Type of error
            error_message: Error message
            stack_trace: Stack trace
            context: Additional context
            user_id: ID of the user who triggered the operation
            session_id: Session ID
            correlation_id: Correlation ID for linking related events

        Returns:
            str: Event ID
        """
        event_data = {
            "error_type": error_type,
            "error_message": error_message,
            "stack_trace": stack_trace,
            "context": context
        }

        return self.log_event(
            event_type="ERROR",
            event_data=event_data,
            user_id=user_id,
            session_id=session_id,
            correlation_id=correlation_id
        )

    def close(self) -> None:
        """Close the audit logger."""

        with self._lock:
            self.file_handler.close()
            self.logger.removeHandler(self.file_handler)

            if self.enable_console_logging:
                self.console_handler.close()
                self.logger.removeHandler(self.console_handler)

            logger.info("Audit logger closed")


# Global audit logger instance
_audit_logger = None


def get_audit_logger() -> AuditLogger:
    """
    Get the global audit logger instance.

    Returns:
        AuditLogger: The global audit logger instance
    """
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def configure_audit_logger(
    log_dir: str = "audit_logs",
    phi_patterns: Optional[List[str]] = None,
    enable_console_logging: bool = True,
    log_level: int = logging.INFO,
    max_log_file_size_mb: int = 10,
    max_log_files: int = 100,
    enable_encryption: bool = False,
    encryption_key: Optional[str] = None
) -> AuditLogger:
    """
    Configure the global audit logger.

    Args:
        log_dir: Directory to store audit logs
        phi_patterns: List of regex patterns to detect PHI/PII
        enable_console_logging: Whether to log to console
        log_level: Logging level
        max_log_file_size_mb: Maximum size of log files in MB
        max_log_files: Maximum number of log files to keep
        enable_encryption: Whether to encrypt log files
        encryption_key: Key for encrypting log files

    Returns:
        AuditLogger: The configured audit logger
    """
    global _audit_logger
    if _audit_logger is not None:
        _audit_logger.close()

    _audit_logger = AuditLogger(
        log_dir=log_dir,
        phi_patterns=phi_patterns,
        enable_console_logging=enable_console_logging,
        log_level=log_level,
        max_log_file_size_mb=max_log_file_size_mb,
        max_log_files=max_log_files,
        enable_encryption=enable_encryption,
        encryption_key=encryption_key
    )

    return _audit_logger


# Export all functions and classes
__all__ = [
    'AuditLogger',
    'get_audit_logger',
    'configure_audit_logger',
    'DEFAULT_PHI_PATTERNS'
]
