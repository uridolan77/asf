from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Configuration schemas

class ConfigurationBase(BaseModel):
    config_key: str
    config_value: str
    config_type: str = "string"
    description: Optional[str] = None
    environment: str = "development"

class ConfigurationCreate(ConfigurationBase):
    pass

class ConfigurationUpdate(BaseModel):
    config_value: str
    config_type: Optional[str] = None
    description: Optional[str] = None

class ConfigurationResponse(ConfigurationBase):
    config_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

# User Setting schemas

class UserSettingBase(BaseModel):
    setting_key: str
    setting_value: str

class UserSettingCreate(UserSettingBase):
    user_id: Optional[int] = None

class UserSettingUpdate(BaseModel):
    setting_value: str

class UserSettingResponse(UserSettingBase):
    setting_id: int
    user_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
