#!/usr/bin/env python3
"""
Pydantic models for SpatX Enhanced Platform
"""
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

# User models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    credits: float
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# Token models
class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

# Operation costs
OPERATION_COSTS = {
    "training": 5.0,
    "prediction": 1.0,
    "heatmap": 0.5,
}

def get_operation_cost(operation: str) -> float:
    """Get the cost of an operation in credits"""
    return OPERATION_COSTS.get(operation, 1.0)