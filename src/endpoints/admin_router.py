from fastapi import APIRouter, UploadFile, HTTPException, Depends
from pydantic import BaseModel
from src.utils.config import ACCESS_TOKEN
from src.utils.qdrant import (
    add_pdf_to_qdrant,
    delete_pdf_from_qdrant,
    ensure_collection,
)

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
# Dependency for token auth
# --------------------------
def verify_token(token: str):
    if token != ACCESS_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

# --------------------------
# Add PDF
# --------------------------
@router.post("/add-pdf", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile):
    # Ensure Qdrant collection exists
    ensure_collection()

    # Read and index PDF
    content = await file.read()
    doc_id = await add_pdf_to_qdrant(content, file.filename)

    return PDFUploadResponse(status="success", doc_id=doc_id)

# --------------------------
# Delete PDF
# --------------------------
@router.delete("/delete-doc/{doc_id}", response_model=PDFDeleteResponse)
async def delete_doc(
    doc_id: str,
    authorized: bool = Depends(lambda: verify_token(ACCESS_TOKEN))
):
    await delete_pdf_from_qdrant(doc_id)
    return PDFDeleteResponse(status="success")
