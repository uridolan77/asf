"""
Audit Repository for LLM Gateway.

This module provides a repository for audit logs that directly connects to the database
and performs CRUD operations on the audit log table.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import logging
import json
from datetime import datetime

from asf.medical.llm_gateway.models.provider import AuditLog

logger = logging.getLogger(__name__)

class AuditRepository:
    """
    Repository for audit logs.
    
    This class provides methods for database operations on the audit log table.
    """
    
    def __init__(self, db: Session):
        """
        Initialize the repository with a database session.
        
        Args:
            db: SQLAlchemy database session
        """
        self.db = db
    
    def create_audit_log(self, audit_data: Dict[str, Any]) -> AuditLog:
        """
        Create a new audit log entry.
        
        Args:
            audit_data: Dictionary containing audit log data
            
        Returns:
            Created AuditLog object
            
        Raises:
            SQLAlchemyError: If there's an error creating the audit log
        """
        try:
            # Convert dictionary values to JSON strings
            old_values = audit_data.get("old_values")
            if old_values and isinstance(old_values, dict):
                old_values = json.dumps(old_values)
            
            new_values = audit_data.get("new_values")
            if new_values and isinstance(new_values, dict):
                new_values = json.dumps(new_values)
            
            audit_log = AuditLog(
                table_name=audit_data["table_name"],
                record_id=audit_data["record_id"],
                action=audit_data["action"],
                changed_by_user_id=audit_data.get("changed_by_user_id"),
                old_values=old_values,
                new_values=new_values,
                timestamp=audit_data.get("timestamp", datetime.utcnow())
            )
            self.db.add(audit_log)
            self.db.commit()
            self.db.refresh(audit_log)
            return audit_log
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error creating audit log: {e}")
            raise
    
    def get_audit_logs_by_table_and_record(self, table_name: str, record_id: str) -> List[AuditLog]:
        """
        Get all audit logs for a specific table and record.
        
        Args:
            table_name: Name of the table
            record_id: ID of the record
            
        Returns:
            List of AuditLog objects
        """
        return self.db.query(AuditLog).filter(
            AuditLog.table_name == table_name,
            AuditLog.record_id == record_id
        ).order_by(AuditLog.timestamp.desc()).all()
    
    def get_audit_logs_by_user(self, user_id: int) -> List[AuditLog]:
        """
        Get all audit logs for a specific user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of AuditLog objects
        """
        return self.db.query(AuditLog).filter(
            AuditLog.changed_by_user_id == user_id
        ).order_by(AuditLog.timestamp.desc()).all()
    
    def get_recent_audit_logs(self, limit: int = 100) -> List[AuditLog]:
        """
        Get the most recent audit logs.
        
        Args:
            limit: Maximum number of logs to return
            
        Returns:
            List of AuditLog objects
        """
        return self.db.query(AuditLog).order_by(
            AuditLog.timestamp.desc()
        ).limit(limit).all()
