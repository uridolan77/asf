"""
Utility functions for encrypting and decrypting sensitive data.
"""

import os
import base64
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

def generate_key() -> bytes:
    """Generate a new encryption key."""
    return Fernet.generate_key()

def derive_key_from_password(password: str, salt: bytes = None) -> tuple:
    """
    Derive an encryption key from a password using PBKDF2.
    
    Args:
        password: The password to derive the key from
        salt: Optional salt for key derivation
        
    Returns:
        Tuple of (key, salt)
    """
    if salt is None:
        salt = os.urandom(16)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key, salt

def encrypt_value(value: str, key: bytes) -> str:
    """
    Encrypt a value using Fernet symmetric encryption.
    
    Args:
        value: The value to encrypt
        key: The encryption key
        
    Returns:
        Encrypted value as a string
    """
    try:
        f = Fernet(key)
        encrypted_data = f.encrypt(value.encode())
        return encrypted_data.decode()
    except Exception as e:
        logger.error(f"Error encrypting value: {e}")
        raise

def decrypt_value(encrypted_value: str, key: bytes) -> str:
    """
    Decrypt a value using Fernet symmetric encryption.
    
    Args:
        encrypted_value: The encrypted value to decrypt
        key: The encryption key
        
    Returns:
        Decrypted value as a string
    """
    try:
        f = Fernet(key)
        decrypted_data = f.decrypt(encrypted_value.encode())
        return decrypted_data.decode()
    except Exception as e:
        logger.error(f"Error decrypting value: {e}")
        raise
