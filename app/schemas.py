from pydantic import BaseModel, EmailStr, validator
from datetime import datetime
import re

class ComplaintCreate(BaseModel):
    name: str
    phone_number: str
    email: EmailStr
    complaint_details: str

    @validator('phone_number')
    def validate_phone_number(cls, v):
        """Validate and standardize phone numbers"""
        # Regex for international numbers with optional country code
        pattern = r'^(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$'
        if not re.match(pattern, v):
            raise ValueError(
                'Invalid phone number format. '
                'Accepted formats: 1234567890, 123-456-7890, (123) 456-7890, +91 1234567890'
            )
        # Remove all non-digit characters
        cleaned = re.sub(r'\D', '', v)
        # Keep last 10 digits (standardize format)
        return cleaned[-10:]
    
class ComplaintResponse(BaseModel):
    complaint_id: str
    name: str
    phone_number: str
    email: str
    complaint_details: str
    created_at: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()  # Ensure proper serialization
        }