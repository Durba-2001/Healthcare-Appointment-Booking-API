# utils/qdrant_utils.py
import uuid
import fitz  # PyMuPDF
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from config import QDRANT_URL, QDRANT_API_KEY, GOOGLE_API_KEY

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

COLLECTION_NAME = "healthcare_docs"

async def add_pdf_to_qdrant(content: bytes, filename: str) -> str:
    doc = fitz.open(stream=content, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    embedding = llm.embed(text)
    doc_id = str(uuid.uuid4())
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(id=doc_id, vector=embedding, payload={"filename": filename})]
    )
    return doc_id

async def delete_pdf_from_qdrant(doc_id: str):
    qdrant.delete(collection_name=COLLECTION_NAME, points_selector={"ids": [doc_id]})
