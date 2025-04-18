from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import logging
import json
from datetime import date
# Use absolute imports instead of relative imports
from models.audit import AuditLog, ApiKeyUsage

logger = logging.getLogger(__name__)

class AuditRepository:
    def __init__(self, db: Session):
        self.db = db

    # Audit Log methods

    def create_audit_log(self, log_data: Dict[str, Any]) -> AuditLog:
        """Create a new audit log entry."""
        try:
            # Convert old_values and new_values to JSON if they are dictionaries
            old_values = log_data.get("old_values")
            if isinstance(old_values, dict):
                old_values = json.dumps(old_values)

            new_values = log_data.get("new_values")
            if isinstance(new_values, dict):
                new_values = json.dumps(new_values)

            audit_log = AuditLog(
                table_name=log_data["table_name"],
                record_id=log_data["record_id"],
                action=log_data["action"],
                changed_by_user_id=log_data.get("changed_by_user_id"),
                old_values=old_values,
                new_values=new_values,
                ip_address=log_data.get("ip_address"),
                user_agent=log_data.get("user_agent")
            )
            self.db.add(audit_log)
            self.db.commit()
            self.db.refresh(audit_log)
            return audit_log
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error creating audit log: {e}")
            raise

    def get_audit_logs(self, table_name: Optional[str] = None, record_id: Optional[str] = None,
                      action: Optional[str] = None, user_id: Optional[int] = None,
                      limit: int = 100, offset: int = 0) -> List[AuditLog]:
        """Get audit logs with optional filters."""
        query = self.db.query(AuditLog)

        if table_name:
            query = query.filter(AuditLog.table_name == table_name)

        if record_id:
            query = query.filter(AuditLog.record_id == record_id)

        if action:
            query = query.filter(AuditLog.action == action)

        if user_id:
            query = query.filter(AuditLog.changed_by_user_id == user_id)

        return query.order_by(AuditLog.changed_at.desc()).limit(limit).offset(offset).all()

    # API Key Usage methods

    def record_api_key_usage(self, usage_data: Dict[str, Any]) -> ApiKeyUsage:
        """Record API key usage."""
        try:
            # Check if usage record already exists for this key, user, and date
            usage_date = usage_data.get("usage_date", date.today())
            existing_usage = self.db.query(ApiKeyUsage).filter(
                ApiKeyUsage.key_id == usage_data["key_id"],
                ApiKeyUsage.user_id == usage_data.get("user_id"),
                ApiKeyUsage.usage_date == usage_date
            ).first()

            if existing_usage:
                # Update existing usage record
                existing_usage.request_count += usage_data.get("request_count", 1)
                existing_usage.tokens_used += usage_data.get("tokens_used", 0)
                self.db.commit()
                self.db.refresh(existing_usage)
                return existing_usage
            else:
                # Create new usage record
                usage = ApiKeyUsage(
                    key_id=usage_data["key_id"],
                    user_id=usage_data.get("user_id"),
                    request_count=usage_data.get("request_count", 1),
                    tokens_used=usage_data.get("tokens_used", 0),
                    usage_date=usage_date
                )
                self.db.add(usage)
                self.db.commit()
                self.db.refresh(usage)
                return usage
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error recording API key usage: {e}")
            raise

    def get_api_key_usage(self, key_id: Optional[int] = None, user_id: Optional[int] = None,
                         start_date: Optional[date] = None, end_date: Optional[date] = None) -> List[ApiKeyUsage]:
        """Get API key usage with optional filters."""
        query = self.db.query(ApiKeyUsage)

        if key_id:
            query = query.filter(ApiKeyUsage.key_id == key_id)

        if user_id:
            query = query.filter(ApiKeyUsage.user_id == user_id)

        if start_date:
            query = query.filter(ApiKeyUsage.usage_date >= start_date)

        if end_date:
            query = query.filter(ApiKeyUsage.usage_date <= end_date)

        return query.order_by(ApiKeyUsage.usage_date.desc()).all()

    def get_total_usage_by_key(self, key_id: int, start_date: Optional[date] = None,
                              end_date: Optional[date] = None) -> Dict[str, int]:
        """Get total usage for an API key."""
        query = self.db.query(
            ApiKeyUsage.key_id,
            ApiKeyUsage.request_count.label("total_requests"),
            ApiKeyUsage.tokens_used.label("total_tokens")
        ).filter(ApiKeyUsage.key_id == key_id)

        if start_date:
            query = query.filter(ApiKeyUsage.usage_date >= start_date)

        if end_date:
            query = query.filter(ApiKeyUsage.usage_date <= end_date)

        result = query.first()

        if result:
            return {
                "key_id": result.key_id,
                "total_requests": result.total_requests,
                "total_tokens": result.total_tokens
            }
        else:
            return {
                "key_id": key_id,
                "total_requests": 0,
                "total_tokens": 0
            }
