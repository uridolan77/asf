import hashlib
import hmac
import time
import json
import base64
import secrets
from typing import Dict, Any, Optional, Callable, List, Tuple
import ssl
import jwt
import logging


class MCPSecurityHandler:
    """
    Handles security aspects of MCP communication including
    message validation, authentication, and encryption.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("MCPSecurityHandler")
        self.shared_secrets = config.get("shared_secrets", {})
        self.jwt_secret = config.get("jwt_secret")
        self.trusted_sources = config.get("trusted_sources", [])
        self.message_ttl = config.get("message_ttl", 300)  # 5 minutes
        self.require_signatures = config.get("require_signatures", True)
        
    def secure_message(self, message: Dict[str, Any], source_id: str) -> Dict[str, Any]:
        """
        Add security information to an outgoing message
        """
        # Create a copy to avoid modifying the original
        secured_message = message.copy()
        
        # Ensure we have a timestamp
        if "timestamp" not in secured_message:
            secured_message["timestamp"] = int(time.time())
            
        # Add a nonce for replay protection
        secured_message["nonce"] = secrets.token_hex(8)
        
        # If we have a shared secret for this source, add signature
        if source_id in self.shared_secrets:
            secured_message["signature"] = self._create_signature(
                secured_message, 
                self.shared_secrets[source_id]
            )
            
        return secured_message
            
    def validate_message(self, message: Dict[str, Any], source_id: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate the security aspects of an incoming message
        Returns (is_valid, error_reason)
        """
        # Check for required fields
        if "timestamp" not in message:
            return False, "Missing timestamp"
            
        # Check message freshness (prevent replay attacks)
        current_time = int(time.time())
        message_time = message["timestamp"]
        
        if isinstance(message_time, str):
            try:
                # Handle ISO format timestamps
                from datetime import datetime
                message_time = datetime.fromisoformat(message_time.replace('Z', '+00:00')).timestamp()
            except ValueError:
                return False, "Invalid timestamp format"
                
        if abs(current_time - message_time) > self.message_ttl:
            return False, "Message expired or from the future"
            
        # If we're requiring signatures
        if self.require_signatures:
            # If we don't know the source, we can't validate the signature
            if not source_id:
                return False, "Unknown message source, cannot validate signature"
                
            # If this source is in the trusted list, we can skip signature check
            if source_id in self.trusted_sources:
                return True, None
                
            # Otherwise, verify signature if present
            if "signature" not in message:
                return False, "Missing signature"
                
            if source_id not in self.shared_secrets:
                return False, f"No shared secret for source {source_id}"
                
            expected_signature = self._create_signature(
                {k: v for k, v in message.items() if k != "signature"}, 
                self.shared_secrets[source_id]
            )
            
            if not hmac.compare_digest(message["signature"], expected_signature):
                return False, "Invalid signature"
                
        return True, None
        
    def _create_signature(self, message: Dict[str, Any], secret: str) -> str:
        """
        Create an HMAC signature for a message
        """
        # Sort keys for consistent signing
        message_str = json.dumps(message, sort_keys=True)
        
        signature = hmac.new(
            secret.encode('utf-8'),
            message_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
        
    def create_jwt_token(
        self, 
        source_id: str, 
        scopes: List[str], 
        expiration: int = 3600
    ) -> str:
        """
        Create a JWT token for a source with the given scopes
        """
        if not self.jwt_secret:
            raise ValueError("JWT secret not configured")
            
        now = int(time.time())
        payload = {
            "sub": source_id,
            "iat": now,
            "exp": now + expiration,
            "scopes": scopes
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        return token
        
    def validate_jwt_token(self, token: str) -> Tuple[bool, Optional[Dict]]:
        """
        Validate a JWT token and return the payload if valid
        """
        if not self.jwt_secret:
            raise ValueError("JWT secret not configured")
            
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return True, payload
        except jwt.ExpiredSignatureError:
            return False, {"error": "Token expired"}
        except jwt.InvalidTokenError:
            return False, {"error": "Invalid token"}


class TLSConfig:
    """
    Configuration for TLS/SSL connections
    """
    
    def __init__(
        self, 
        cert_file: Optional[str] = None,
        key_file: Optional[str] = None,
        ca_file: Optional[str] = None,
        verify_cert: bool = True,
        check_hostname: bool = True
    ):
        self.cert_file = cert_file
        self.key_file = key_file
        self.ca_file = ca_file
        self.verify_cert = verify_cert
        self.check_hostname = check_hostname
        
    def create_ssl_context(self) -> ssl.SSLContext:
        """
        Create an SSL context based on the configuration
        """
        context = ssl.create_default_context(
            purpose=ssl.Purpose.SERVER_AUTH if self.ca_file else None
        )
        
        if self.ca_file:
            context.load_verify_locations(cafile=self.ca_file)
            
        if self.cert_file and self.key_file:
            context.load_cert_chain(self.cert_file, self.key_file)
            
        if not self.verify_cert:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        else:
            context.check_hostname = self.check_hostname
            
        # Use more secure TLS protocols
        context.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
        
        return context


class MCPAuthorizationManager:
    """
    Handles authorization for MCP operations based on roles and permissions
    """
    
    def __init__(self):
        self.logger = logging.getLogger("MCPAuthorizationManager")
        self.roles = {}  # role_name -> permissions
        self.source_roles = {}  # source_id -> list of roles
        
    def add_role(self, role_name: str, permissions: List[str]) -> None:
        """
        Add or update a role with the given permissions
        """
        self.roles[role_name] = permissions
        
    def assign_role_to_source(self, source_id: str, role_name: str) -> bool:
        """
        Assign a role to a source
        """
        if role_name not in self.roles:
            self.logger.warning(f"Cannot assign unknown role {role_name}")
            return False
            
        if source_id not in self.source_roles:
            self.source_roles[source_id] = []
            
        if role_name not in self.source_roles[source_id]:
            self.source_roles[source_id].append(role_name)
            
        return True
        
    def remove_role_from_source(self, source_id: str, role_name: str) -> bool:
        """
        Remove a role from a source
        """
        if source_id not in self.source_roles:
            return False
            
        if role_name in self.source_roles[source_id]:
            self.source_roles[source_id].remove(role_name)
            return True
            
        return False
        
    def get_source_permissions(self, source_id: str) -> List[str]:
        """
        Get all permissions for a source based on its roles
        """
        if source_id not in self.source_roles:
            return []
            
        all_permissions = []
        for role in self.source_roles[source_id]:
            all_permissions.extend(self.roles.get(role, []))
            
        # Remove duplicates
        return list(set(all_permissions))
        
    def check_permission(self, source_id: str, required_permission: str) -> bool:
        """
        Check if a source has a specific permission
        """
        permissions = self.get_source_permissions(source_id)
        
        # Check for wildcard permission
        if "*" in permissions:
            return True
            
        # Check for exact permission match
        if required_permission in permissions:
            return True
            
        # Check for namespace permission (e.g., "sensors.*" matches "sensors.read")
        for permission in permissions:
            if permission.endswith(".*") and required_permission.startswith(permission[:-1]):
                return True
                
        return False
        
    def authorize_message(
        self, 
        message: Dict[str, Any], 
        source_id: str, 
        permission_mapper: Callable[[Dict], str]
    ) -> bool:
        """
        Authorize a message from a source using a mapping function
        to determine the required permission
        """
        required_permission = permission_mapper(message)
        return self.check_permission(source_id, required_permission)


# Example permission mapper for MCP messages
def default_permission_mapper(message: Dict[str, Any]) -> str:
    """
    Maps an MCP message to a required permission string
    """
    message_type = message.get("type", "")
    
    if message_type == "Resource.request":
        resource_type = message.get("resourceType", "unknown")
        return f"resources.read.{resource_type}"
        
    elif message_type == "Resource.update":
        content_type = message.get("contentType", "unknown")
        return f"resources.update.{content_type}"
        
    elif message_type == "Tool.invoke":
        tool = message.get("tool", "unknown")
        return f"tools.invoke.{tool}"
        
    elif message_type == "Event":
        event_type = message.get("eventType", "unknown")
        return f"events.publish.{event_type}"
        
    # Default case
    return f"{message_type.lower()}"