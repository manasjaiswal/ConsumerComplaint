from sqlalchemy import Column, String, DateTime
from sqlalchemy.sql import func
from app.database import Base

class Complaint(Base):
    __tablename__ = "complaints"
    complaint_id = Column(String, primary_key=True, index=True)
    name = Column(String)
    phone_number = Column(String)
    email = Column(String)
    complaint_details = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())