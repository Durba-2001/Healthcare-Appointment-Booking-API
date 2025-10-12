from pydantic import BaseModel

# --------------------------
# Pydantic Models
# --------------------------
class PDFUploadResponse(BaseModel):
    status: str
    doc_id: str

class PDFDeleteResponse(BaseModel):
    status: str