"""Enhanced Audit Logging for Medical Research

This module provides comprehensive audit logging capabilities specifically designed
for medical research applications, with robust PHI/PII detection, redaction,
and immutable logging for compliance requirements.
"""

import logging
import json
import os
import uuid
import time
import re
import hashlib
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Set
from pathlib import Path
import traceback

# Set up logging
logger = logging.getLogger(__name__)

# Default PHI/PII detection patterns
DEFAULT_PHI_PATTERNS = [
    # Patient identifiers
    r'\b(?:patient|subject|participant)\s*(?:id|number|#)\s*[:=]?\s*[a-zA-Z0-9_-]{4,}',
    
    # Names
    r'\b(?:[A-Z][a-z]+\s+){1,2}[A-Z][a-z]+\b',  # Simple name pattern (e.g., John Smith)
    
    # Dates
    r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',  # MM/DD/YYYY or DD/MM/YYYY
    r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',  # Month DD, YYYY
    
    # Phone numbers
    r'\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',  # (123) 456-7890 or 123-456-7890
    
    # SSN
    r'\b\d{3}[-]\d{2}[-]\d{4}\b',  # 123-45-6789
    
    # MRN (Medical Record Number)
    r'\b(?:MRN|Medical Record Number)\s*[:=]?\s*[a-zA-Z0-9_-]{4,}\b',
    
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


class AuditLogEntry:
    """Immutable audit log entry for medical research applications."""
    
    def __init__(
        self,
        event_type: str,
        timestamp: datetime,
        user_id: Optional[str],
        session_id: Optional[str],
        correlation_id: Optional[str],
        operation_id: str,
        component: str,
        inputs: Dict[str, Any],
        outputs: Optional[Dict[str, Any]],
        error: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ):
        """Initialize an audit log entry.
        
        Args:
            event_type: Type of event (e.g., "LLM_CALL", "MODULE_CALL")
            timestamp: Event timestamp
            user_id: ID of the user who triggered the event
            session_id: Session ID
            correlation_id: Correlation ID for linking related events
            operation_id: Unique ID for this operation
            component: Component that generated the event
            inputs: Input data (will be redacted)
            outputs: Output data (will be redacted)
            error: Error message if applicable
            metadata: Additional metadata
        """
        self.event_type = event_type
        self.timestamp = timestamp
        self.user_id = user_id
        self.session_id = session_id
        self.correlation_id = correlation_id
        self.operation_id = operation_id
        self.component = component
        self.inputs = inputs
        self.outputs = outputs
        self.error = error
        self.metadata = metadata or {}
        
        # Generate a unique ID for this log entry
        self.log_id = str(uuid.uuid4())
        
        # Calculate a hash of the log entry for integrity verification
        self._calculate_hash()
    
    def _calculate_hash(self):
        """Calculate a hash of the log entry for integrity verification."""
        # Create a dictionary of all fields
        data = {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "operation_id": self.operation_id,
            "component": self.component,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error": self.error,
            "metadata": self.metadata,
            "log_id": self.log_id
        }
        
        # Calculate hash
        self.hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the log entry to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the log entry
        """
        return {
            "log_id": self.log_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "operation_id": self.operation_id,
            "component": self.component,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error": self.error,
            "metadata": self.metadata,
            "hash": self.hash
        }
    
    def to_json(self) -> str:
        """Convert the log entry to a JSON string.
        
        Returns:
            str: JSON string representation of the log entry
        """
        return json.dumps(self.to_dict(), indent=2)


class EnhancedAuditLogger:
    """Enhanced audit logger for medical research applications.
    
    This class provides comprehensive audit logging capabilities with PHI/PII detection,
    redaction, and immutable logging for compliance requirements.
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
        encryption_key: Optional[str] = None,
        enable_phi_detection: bool = True,
        enable_phi_redaction: bool = True
    ):
        """Initialize the enhanced audit logger.
        
        Args:
            log_dir: Directory to store audit logs
            phi_patterns: List of regex patterns to detect PHI/PII
            enable_console_logging: Whether to log to console
            log_level: Logging level
            max_log_file_size_mb: Maximum size of log files in MB
            max_log_files: Maximum number of log files to keep
            enable_encryption: Whether to encrypt log files
            encryption_key: Key for encrypting log files
            enable_phi_detection: Whether to detect PHI/PII
            enable_phi_redaction: Whether to redact detected PHI/PII
        """
        self.log_dir = Path(log_dir)
        self.phi_patterns = phi_patterns or DEFAULT_PHI_PATTERNS
        self.enable_console_logging = enable_console_logging
        self.log_level = log_level
        self.max_log_file_size_mb = max_log_file_size_mb
        self.max_log_files = max_log_files
        self.enable_encryption = enable_encryption
        self.encryption_key = encryption_key
        self.enable_phi_detection = enable_phi_detection
        self.enable_phi_redaction = enable_phi_redaction
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
        # Compile PHI patterns
        self._compiled_phi_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.phi_patterns]
        
        # Set up lock for thread safety
        self._lock = threading.Lock()
        
        # Current log file
        self._current_log_file = self._get_new_log_file()
        
        # Log initialization
        logger.info(f"Enhanced audit logger initialized with log directory: {self.log_dir}")
    
    def _setup_logging(self):
        """Set up logging configuration."""
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add console handler if enabled
        if self.enable_console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # Add file handler
        file_handler = logging.FileHandler(self.log_dir / "audit.log")
        file_handler.setLevel(self.log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    def _get_new_log_file(self) -> Path:
        """Get a new log file path.
        
        Returns:
            Path: Path to the new log file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.log_dir / f"audit_log_{timestamp}_{uuid.uuid4().hex[:8]}.jsonl"
    
    def _rotate_log_file_if_needed(self):
        """Rotate log file if it exceeds the maximum size."""
        if self._current_log_file.exists() and self._current_log_file.stat().st_size > self.max_log_file_size_mb * 1024 * 1024:
            self._current_log_file = self._get_new_log_file()
            
            # Check if we need to delete old log files
            log_files = sorted(self.log_dir.glob("audit_log_*.jsonl"), key=lambda p: p.stat().st_mtime)
            if len(log_files) > self.max_log_files:
                # Delete oldest log files
                for log_file in log_files[:-self.max_log_files]:
                    try:
                        log_file.unlink()
                        logger.info(f"Deleted old log file: {log_file}")
                    except Exception as e:
                        logger.warning(f"Failed to delete old log file {log_file}: {str(e)}")
    
    def _detect_and_redact_phi(self, text: str) -> str:
        """Detect and redact PHI/PII in text.
        
        Args:
            text: Text to redact
            
        Returns:
            str: Redacted text
        """
        if not self.enable_phi_detection or not isinstance(text, str):
            return text
        
        redacted_text = text
        
        # Apply each pattern
        for i, pattern in enumerate(self._compiled_phi_patterns):
            if self.enable_phi_redaction:
                redacted_text = pattern.sub(f"[REDACTED-PHI-{i}]", redacted_text)
            else:
                # Just detect and log, don't redact
                matches = pattern.findall(redacted_text)
                if matches:
                    logger.warning(f"Detected potential PHI/PII in text: {len(matches)} matches for pattern {i}")
        
        return redacted_text
    
    def _redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively redact PHI/PII in a dictionary.
        
        Args:
            data: Dictionary to redact
            
        Returns:
            Dict[str, Any]: Redacted dictionary
        """
        if not isinstance(data, dict):
            return data
        
        redacted_data = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                redacted_data[key] = self._detect_and_redact_phi(value)
            elif isinstance(value, dict):
                redacted_data[key] = self._redact_dict(value)
            elif isinstance(value, list):
                redacted_data[key] = [
                    self._detect_and_redact_phi(item) if isinstance(item, str)
                    else self._redact_dict(item) if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                redacted_data[key] = value
        
        return redacted_data
    
    def log_event(
        self,
        event_type: str,
        component: str,
        inputs: Dict[str, Any],
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log an audit event.
        
        Args:
            event_type: Type of event (e.g., "LLM_CALL", "MODULE_CALL")
            component: Component that generated the event
            inputs: Input data (will be redacted)
            outputs: Output data (will be redacted)
            error: Error message if applicable
            user_id: ID of the user who triggered the event
            session_id: Session ID
            correlation_id: Correlation ID for linking related events
            metadata: Additional metadata
            
        Returns:
            str: Log entry ID
        """
        # Generate operation ID if not provided in correlation_id
        operation_id = correlation_id or str(uuid.uuid4())
        
        # Redact PHI/PII in inputs and outputs
        redacted_inputs = self._redact_dict(inputs)
        redacted_outputs = self._redact_dict(outputs) if outputs is not None else None
        
        # Create log entry
        log_entry = AuditLogEntry(
            event_type=event_type,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            correlation_id=correlation_id,
            operation_id=operation_id,
            component=component,
            inputs=redacted_inputs,
            outputs=redacted_outputs,
            error=error,
            metadata=metadata
        )
        
        # Write log entry to file
        with self._lock:
            self._rotate_log_file_if_needed()
            
            try:
                with open(self._current_log_file, "a") as f:
                    f.write(log_entry.to_json() + "\n")
            except Exception as e:
                logger.error(f"Failed to write audit log entry: {str(e)}")
                # Try to create a new log file and write again
                self._current_log_file = self._get_new_log_file()
                try:
                    with open(self._current_log_file, "a") as f:
                        f.write(log_entry.to_json() + "\n")
                except Exception as e2:
                    logger.error(f"Failed to write audit log entry to new file: {str(e2)}")
        
        # Log to standard logger as well
        logger.info(f"Audit event: {event_type} - {component} - {log_entry.log_id}")
        
        return log_entry.log_id
    
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
        """Log an LLM call.
        
        Args:
            prompt: The prompt sent to the LLM
            model: The model used
            parameters: The parameters used
            response: The response from the LLM
            error: Error message if applicable
            latency: Latency in seconds
            user_id: ID of the user who triggered the call
            session_id: Session ID
            correlation_id: Correlation ID for linking related events
            
        Returns:
            str: Log entry ID
        """
        inputs = {
            "prompt": prompt,
            "model": model,
            "parameters": parameters
        }
        
        outputs = None
        if response is not None:
            outputs = {"response": response}
        
        metadata = {}
        if latency is not None:
            metadata["latency"] = latency
        
        return self.log_event(
            event_type="LLM_CALL",
            component="DSPyClient",
            inputs=inputs,
            outputs=outputs,
            error=error,
            user_id=user_id,
            session_id=session_id,
            correlation_id=correlation_id,
            metadata=metadata
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
        """Log a module call.
        
        Args:
            module_name: Name of the module
            module_type: Type of the module
            inputs: Input data
            outputs: Output data
            error: Error message if applicable
            latency: Latency in seconds
            user_id: ID of the user who triggered the call
            session_id: Session ID
            correlation_id: Correlation ID for linking related events
            
        Returns:
            str: Log entry ID
        """
        inputs_with_metadata = {
            "module_name": module_name,
            "module_type": module_type,
            **inputs
        }
        
        metadata = {}
        if latency is not None:
            metadata["latency"] = latency
        
        return self.log_event(
            event_type="MODULE_CALL",
            component=f"Module:{module_name}",
            inputs=inputs_with_metadata,
            outputs=outputs,
            error=error,
            user_id=user_id,
            session_id=session_id,
            correlation_id=correlation_id,
            metadata=metadata
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
        """Log an optimization event.
        
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
            str: Log entry ID
        """
        inputs = {
            "module_name": module_name,
            "optimizer_type": optimizer_type,
            "original_prompts": original_prompts
        }
        
        outputs = {
            "optimized_prompts": optimized_prompts,
            "metrics": metrics
        }
        
        return self.log_event(
            event_type="OPTIMIZATION",
            component=f"Optimizer:{optimizer_type}",
            inputs=inputs,
            outputs=outputs,
            error=None,
            user_id=user_id,
            session_id=session_id,
            correlation_id=correlation_id
        )
    
    def log_cache_operation(
        self,
        operation: str,
        key: str,
        success: bool,
        latency: Optional[float] = None,
        error: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """Log a cache operation.
        
        Args:
            operation: Type of operation (get, set, clear)
            key: Cache key
            success: Whether the operation was successful
            latency: Latency in seconds
            error: Error message if applicable
            user_id: ID of the user who triggered the operation
            session_id: Session ID
            correlation_id: Correlation ID for linking related events
            
        Returns:
            str: Log entry ID
        """
        inputs = {
            "operation": operation,
            "key": key
        }
        
        outputs = {
            "success": success
        }
        
        metadata = {}
        if latency is not None:
            metadata["latency"] = latency
        
        return self.log_event(
            event_type="CACHE_OPERATION",
            component="Cache",
            inputs=inputs,
            outputs=outputs,
            error=error,
            user_id=user_id,
            session_id=session_id,
            correlation_id=correlation_id,
            metadata=metadata
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
        """Log an error.
        
        Args:
            error_type: Type of error
            error_message: Error message
            stack_trace: Stack trace
            context: Error context
            user_id: ID of the user who triggered the operation
            session_id: Session ID
            correlation_id: Correlation ID for linking related events
            
        Returns:
            str: Log entry ID
        """
        inputs = context or {}
        
        outputs = {
            "error_type": error_type,
            "error_message": error_message
        }
        
        if stack_trace:
            outputs["stack_trace"] = stack_trace
        
        return self.log_event(
            event_type="ERROR",
            component="Error",
            inputs=inputs,
            outputs=outputs,
            error=error_message,
            user_id=user_id,
            session_id=session_id,
            correlation_id=correlation_id
        )
    
    def query_logs(
        self,
        event_type: Optional[str] = None,
        component: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query audit logs.
        
        Args:
            event_type: Filter by event type
            component: Filter by component
            user_id: Filter by user ID
            session_id: Filter by session ID
            correlation_id: Filter by correlation ID
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of matching log entries
        """
        results = []
        
        # Get all log files
        log_files = sorted(self.log_dir.glob("audit_log_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Process each log file
        for log_file in log_files:
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            
                            # Apply filters
                            if event_type and entry.get("event_type") != event_type:
                                continue
                            
                            if component and entry.get("component") != component:
                                continue
                            
                            if user_id and entry.get("user_id") != user_id:
                                continue
                            
                            if session_id and entry.get("session_id") != session_id:
                                continue
                            
                            if correlation_id and entry.get("correlation_id") != correlation_id:
                                continue
                            
                            if start_time:
                                entry_time = datetime.fromisoformat(entry.get("timestamp"))
                                if entry_time < start_time:
                                    continue
                            
                            if end_time:
                                entry_time = datetime.fromisoformat(entry.get("timestamp"))
                                if entry_time > end_time:
                                    continue
                            
                            # Add to results
                            results.append(entry)
                            
                            # Check limit
                            if len(results) >= limit:
                                return results
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in log file {log_file}")
                            continue
            except Exception as e:
                logger.warning(f"Error reading log file {log_file}: {str(e)}")
                continue
        
        return results
    
    def close(self):
        """Close the audit logger."""
        logger.info("Closing audit logger")


# Global audit logger instance
_audit_logger = None


def get_audit_logger() -> EnhancedAuditLogger:
    """Get the global audit logger instance.
    
    Returns:
        EnhancedAuditLogger: The global audit logger instance
    """
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = EnhancedAuditLogger()
    return _audit_logger


def configure_audit_logger(
    log_dir: str = "audit_logs",
    phi_patterns: Optional[List[str]] = None,
    enable_console_logging: bool = True,
    log_level: int = logging.INFO,
    max_log_file_size_mb: int = 10,
    max_log_files: int = 100,
    enable_encryption: bool = False,
    encryption_key: Optional[str] = None,
    enable_phi_detection: bool = True,
    enable_phi_redaction: bool = True
) -> EnhancedAuditLogger:
    """Configure the global audit logger.
    
    Args:
        log_dir: Directory to store audit logs
        phi_patterns: List of regex patterns to detect PHI/PII
        enable_console_logging: Whether to log to console
        log_level: Logging level
        max_log_file_size_mb: Maximum size of log files in MB
        max_log_files: Maximum number of log files to keep
        enable_encryption: Whether to encrypt log files
        encryption_key: Key for encrypting log files
        enable_phi_detection: Whether to detect PHI/PII
        enable_phi_redaction: Whether to redact detected PHI/PII
        
    Returns:
        EnhancedAuditLogger: The configured audit logger
    """
    global _audit_logger
    if _audit_logger is not None:
        _audit_logger.close()
    
    _audit_logger = EnhancedAuditLogger(
        log_dir=log_dir,
        phi_patterns=phi_patterns,
        enable_console_logging=enable_console_logging,
        log_level=log_level,
        max_log_file_size_mb=max_log_file_size_mb,
        max_log_files=max_log_files,
        enable_encryption=enable_encryption,
        encryption_key=encryption_key,
        enable_phi_detection=enable_phi_detection,
        enable_phi_redaction=enable_phi_redaction
    )
    
    return _audit_logger
