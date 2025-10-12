import uuid
import fitz  # PyMuPDF
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from src.utils.config import QDRANT_URL, QDRANT_API_KEY, GOOGLE_API_KEY

# ------------------------------
# Initialize embeddings model
# ------------------------------
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",   # ✅ Gemini embedding model (3072-d)
    api_key=GOOGLE_API_KEY
)

# ------------------------------
# Initialize Qdrant client
# ------------------------------
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60.0)
COLLECTION_NAME = "healthcare_docs"
EMBEDDING_DIM = 3072  # ✅ gemini-embedding-001 produces 3072-dimensional vectors

# ------------------------------
# Ensure Qdrant collection exists (auto-fix mismatch)
# ------------------------------
def ensure_collection():
    try:
        info = qdrant.get_collection(collection_name=COLLECTION_NAME)
        existing_dim = info.config.params.vectors.size  # ✅ no .result

        if existing_dim != EMBEDDING_DIM:
            print(f"⚠️ Dimension mismatch: existing={existing_dim}, expected={EMBEDDING_DIM}. Recreating collection...")
            qdrant.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=qmodels.Distance.COSINE
                ),
            )
    except Exception:
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(
                size=EMBEDDING_DIM,
                distance=qmodels.Distance.COSINE
            ),
        )

# Call once at import
ensure_collection()

# ------------------------------
# Add PDF to Qdrant
# ------------------------------
async def add_pdf_to_qdrant(content: bytes, filename: str) -> str:
    # Extract text from PDF
    doc = fitz.open(stream=content, filetype="pdf")
    text = "".join(page.get_text() for page in doc)
    if not text.strip():
        raise ValueError("PDF contains no extractable text.")

    # Create embedding
    embedding = embedding_model.embed_query(text)

    # Unique doc ID
    doc_id = str(uuid.uuid4())

    # Store in Qdrant
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            qmodels.PointStruct(
                id=doc_id,
                vector=embedding,
                payload={
                    "filename": filename,
                    "text_preview": text[:500]
                }
            )
        ],
    )
    return doc_id

# ------------------------------
# Delete PDF from Qdrant
# ------------------------------
async def delete_pdf_from_qdrant(doc_id: str):
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=qmodels.PointIdsList(points=[doc_id])
    )
