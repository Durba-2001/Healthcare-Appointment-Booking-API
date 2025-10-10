# endpoints/admin_router.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from config import ACCESS_TOKEN
from qdrant import add_pdf_to_qdrant, delete_pdf_from_qdrant

router = APIRouter()

# --------------------------
# Pydantic Models
# --------------------------
class PDFUploadResponse(BaseModel):
    status: str
    doc_id: str

class PDFDeleteResponse(BaseModel):
    status: str

# --------------------------
# Add PDF
# --------------------------
@router.post("/add-pdf", response_model=PDFUploadResponse)
async def add_pdf(file: UploadFile = File(...), token: str = ""):
    if token != ACCESS_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    content = await file.read()
    doc_id = await add_pdf_to_qdrant(content, file.filename)
    return PDFUploadResponse(status="success", doc_id=doc_id)

# --------------------------
# Delete PDF
# --------------------------
@router.delete("/delete-doc/{doc_id}", response_model=PDFDeleteResponse)
async def delete_doc(doc_id: str, token: str = ""):
    if token != ACCESS_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    await delete_pdf_from_qdrant(doc_id)
    return PDFDeleteResponse(status="success")
