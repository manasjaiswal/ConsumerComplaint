from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
import uuid
from app.models import Complaint
from app.schemas import ComplaintCreate,ComplaintResponse
from app.database import get_db, engine ,Base

# Create tables (SQLite)
Base.metadata.create_all(bind=engine)

app = FastAPI()

# --- API Endpoints ---
@app.post("/complaints/", response_model=ComplaintResponse)
def create_complaint(
    complaint: ComplaintCreate, 
    db: Session = Depends(get_db)
):
    # Generate complaint ID
    complaint_id = str(uuid.uuid4())
    
    # Create DB record
    db_complaint = Complaint(
        complaint_id=complaint_id,
        **complaint.dict()
    )
    
    # Save to database
    db.add(db_complaint)
    db.commit()
    db.refresh(db_complaint)
    
    return {
        "complaint_id": complaint_id,
        "message": "Complaint created successfully",
        **complaint.dict(),
        "created_at": datetime.now().isoformat()
    }

@app.get("/complaints/{complaint_id}", response_model=ComplaintResponse)
def get_complaint(
    complaint_id: str, 
    db: Session = Depends(get_db)
):
    complaint = db.query(Complaint).filter(
        Complaint.complaint_id == complaint_id
    ).first()
    
    if not complaint:
        raise HTTPException(
            status_code=404, 
            detail="Complaint not found"
        )
    
    # Ensure created_at is properly formatted
    if isinstance(complaint.created_at, str):
        complaint.created_at = datetime.fromisoformat(complaint.created_at)
    
    return complaint

# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "API is running"}