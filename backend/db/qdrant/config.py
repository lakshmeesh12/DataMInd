# db/qdrant/config.py
import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "boq_chunks")
QDRANT_COLLECTION_1 = os.getenv("QDRANT_COLLECTION_1", "excel_documents")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# === OpenAI Embedding Dimensions ===
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

# Global client
qdrant_client = QdrantClient(url=QDRANT_URL)

def get_embedding_dim() -> int:
    """Return the correct dimension for the configured model."""
    return EMBEDDING_DIMENSIONS.get(EMBEDDING_MODEL, 1536)

async def ensure_collection(collection_name: str):
    """Create or recreate collection with correct dimension."""
    dim = get_embedding_dim()
    try:
        existing = qdrant_client.get_collection(collection_name)
        if existing.config.params.vectors.size != dim:
            print(f"Dimension mismatch for {collection_name}: {existing.config.params.vectors.size} → {dim}. Recreating...")
            qdrant_client.delete_collection(collection_name)
        else:
            print(f"Collection '{collection_name}' already exists with correct dim={dim}")
            return
    except Exception:
        pass  # Collection doesn't exist → create

    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"Created Qdrant collection: {collection_name} (dim={dim})")

# === Public init functions ===
async def init_qdrant_collection():
    await ensure_collection(QDRANT_COLLECTION)

async def init_qdrant_excel_collection():
    await ensure_collection(QDRANT_COLLECTION_1)