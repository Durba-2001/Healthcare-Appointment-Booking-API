from fastapi import APIRouter, UploadFile
from src.models.admin_schema import PDFUploadResponse, PDFDeleteResponse
from src.utils.qdrant import (
    add_pdf_to_qdrant,
    delete_pdf_from_qdrant,
    ensure_collection,
)

router = APIRouter()

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
async def delete_doc(doc_id: str):
    await delete_pdf_from_qdrant(doc_id)
    return PDFDeleteResponse(status="success")
