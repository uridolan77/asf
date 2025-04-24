from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Provider schemas

class LLMModelBase(BaseModel):
    model_id: str
    display_name: str
    model_type: Optional[str] = None
    context_window: Optional[int] = None
    max_tokens: Optional[int] = None
    enabled: bool = True

class LLMModelCreate(LLMModelBase):
    pass

class LLMModelResponse(LLMModelBase):
    class Config:
        orm_mode = True

class ConnectionParameterBase(BaseModel):
    param_name: str
    param_value: str
    is_sensitive: bool = False
    environment: str = "development"

class ConnectionParameterCreate(ConnectionParameterBase):
    pass

class ConnectionParameterResponse(ConnectionParameterBase):
    param_id: int
    provider_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class ApiKeyBase(BaseModel):
    key_value: str
    is_encrypted: bool = True
    environment: str = "development"
    expires_at: Optional[datetime] = None

class ApiKeyCreate(ApiKeyBase):
    pass

class ApiKeyResponse(BaseModel):
    key_id: int
    provider_id: str
    is_encrypted: bool
    environment: str
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None

    class Config:
        orm_mode = True

class ProviderBase(BaseModel):
    provider_id: str
    display_name: str
    provider_type: str
    description: Optional[str] = None
    enabled: bool = True

class ProviderCreate(ProviderBase):
    models: Optional[List[LLMModelCreate]] = []
    connection_parameters: Optional[List[ConnectionParameterCreate]] = []
    api_key: Optional[ApiKeyCreate] = None

class ProviderUpdate(BaseModel):
    display_name: Optional[str] = None
    provider_type: Optional[str] = None
    description: Optional[str] = None
    enabled: Optional[bool] = None

class ProviderResponse(ProviderBase):
    created_at: datetime
    updated_at: datetime
    models: List[LLMModelResponse] = []
    connection_parameters: List[Dict[str, Any]] = []

    class Config:
        orm_mode = True
