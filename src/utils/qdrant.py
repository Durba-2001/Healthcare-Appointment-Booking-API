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
    model="gemini-embedding-001",   # ✅ Use Google Gemini embedding model (3072-dimensional vector)
    api_key=GOOGLE_API_KEY          # Provide the Google API key for authentication
)

# ------------------------------
# Initialize Qdrant client
# ------------------------------
qdrant = QdrantClient(
    url=QDRANT_URL,        # URL of the Qdrant database
    api_key=QDRANT_API_KEY,  # API key for authentication
    timeout=60.0           # Set network timeout to 60 seconds
)
COLLECTION_NAME = "healthcare_docs"  # Name of the Qdrant collection where PDFs will be stored
EMBEDDING_DIM = 3072  # ✅ Embedding dimension produced by gemini-embedding-001

# ------------------------------
# Ensure Qdrant collection exists (auto-fix mismatch)
# ------------------------------
def ensure_collection():
    try:
        # Try to get info about the collection
        info = qdrant.get_collection(collection_name=COLLECTION_NAME)
        existing_dim = info.config.params.vectors.size  # Get existing vector dimension in the collection

        # If the dimension does not match the expected embedding dimension
        if existing_dim != EMBEDDING_DIM:
            print(f"⚠️ Dimension mismatch: existing={existing_dim}, expected={EMBEDDING_DIM}. Recreating collection...")
            # Recreate the collection with correct dimension and cosine distance
            qdrant.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=qmodels.Distance.COSINE
                ),
            )
    except Exception:
        # If collection does not exist or any other error occurs, create it
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(
                size=EMBEDDING_DIM,
                distance=qmodels.Distance.COSINE
            ),
        )

# Call ensure_collection once at import time to make sure collection exists
ensure_collection()

# ------------------------------
# Add PDF to Qdrant
# ------------------------------
async def add_pdf_to_qdrant(content: bytes, filename: str) -> str:
    # Open PDF from bytes
    doc = fitz.open(stream=content, filetype="pdf")
    
    # Extract text from all pages
    text = "".join(page.get_text() for page in doc)
    
    # Raise an error if PDF contains no text
    if not text.strip():
        raise ValueError("PDF contains no extractable text.")

    # Generate embedding vector for the extracted text
    embedding = embedding_model.embed_query(text)

    # Generate a unique identifier for this document
    doc_id = str(uuid.uuid4())

    # Store the embedding and metadata in Qdrant
    qdrant.upsert(
        collection_name=COLLECTION_NAME,  # Which collection to insert into
        points=[
            qmodels.PointStruct(
                id=doc_id,             # Unique ID of the document
                vector=embedding,      # The embedding vector
                payload={
                    "filename": filename,          # Store filename for reference
                    "text_preview": text[:500]     # Store first 500 characters as preview
                }
            )
        ],
    )
    
    # Return the unique document ID
    return doc_id

# ------------------------------
# Delete PDF from Qdrant
# ------------------------------
async def delete_pdf_from_qdrant(doc_id: str):
    # Delete a point/document from Qdrant by its ID
    qdrant.delete(
        collection_name=COLLECTION_NAME,                # Collection name
        points_selector=qmodels.PointIdsList(points=[doc_id])  # Specify which document(s) to delete
    )