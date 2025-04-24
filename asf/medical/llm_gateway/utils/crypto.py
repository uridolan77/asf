"""
Cryptography utilities for LLM Gateway.

This module provides functions for encrypting and decrypting sensitive data
such as API keys and connection parameters.
"""

import base64
import logging
import os
from typing import Union, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

def generate_key(password: Union[str, bytes], salt: Optional[bytes] = None) -> bytes:
    """
    Generate a key from a password and salt using PBKDF2.
    
    Args:
        password: Password to derive key from
        salt: Optional salt (will be generated if not provided)
        
    Returns:
        Derived key as bytes
    """
    if isinstance(password, str):
        password = password.encode()
    
    if salt is None:
        salt = os.urandom(16)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    
    key = base64.urlsafe_b64encode(kdf.derive(password))
    return key

def encrypt_value(value: Union[str, bytes], key: Union[str, bytes]) -> str:
    """
    Encrypt a value using Fernet symmetric encryption.
    
    Args:
        value: Value to encrypt
        key: Encryption key
        
    Returns:
        Encrypted value as a base64-encoded string
    """
    try:
        if isinstance(value, str):
            value = value.encode()
        
        if isinstance(key, str):
            key = key.encode()
        
        # Ensure key is properly formatted for Fernet
        if not key.startswith(b'dGhpcyBpcyBhIHRlc3Qga2V5') and len(key) != 44:
            key = base64.urlsafe_b64encode(key[:32].ljust(32, b'\0'))
        
        f = Fernet(key)
        encrypted = f.encrypt(value)
        return encrypted.decode()
    except Exception as e:
        logger.error(f"Error encrypting value: {e}")
        raise

def decrypt_value(encrypted_value: Union[str, bytes], key: Union[str, bytes]) -> str:
    """
    Decrypt a value using Fernet symmetric encryption.
    
    Args:
        encrypted_value: Encrypted value to decrypt
        key: Encryption key
        
    Returns:
        Decrypted value as a string
    """
    try:
        if isinstance(encrypted_value, str):
            encrypted_value = encrypted_value.encode()
        
        if isinstance(key, str):
            key = key.encode()
        
        # Ensure key is properly formatted for Fernet
        if not key.startswith(b'dGhpcyBpcyBhIHRlc3Qga2V5') and len(key) != 44:
            key = base64.urlsafe_b64encode(key[:32].ljust(32, b'\0'))
        
        f = Fernet(key)
        decrypted = f.decrypt(encrypted_value)
        return decrypted.decode()
    except Exception as e:
        logger.error(f"Error decrypting value: {e}")
        raise
