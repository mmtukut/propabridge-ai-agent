"""
PropaBridge — RAG Query Helper
================================
Used by the Gemini Pro agent at inference time.
Retrieves top-k chunks from Vertex AI Vector Search,
then hydrates full content from Firestore.

Usage:
  from query_helper import retrieve_chunks
  chunks = retrieve_chunks("affordable 2 bedroom apartment Abuja", top_k=5)
"""
import os
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from google.cloud import aiplatform, firestore

# 1. Add your digital passport
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ai_pipeline/service_account.json"

# 2. Update to the true Project ID
PROJECT_ID     = "studio-6084005125-75144"
REGION         = "us-central1"
EMBEDDING_MODEL = "text-embedding-004"
ENDPOINT_NAME  = "propabridge-rag-endpoint"
DEPLOYED_ID    = "propabridge_deployed_v1"
COLLECTION     = "propabridge_chunks"

vertexai.init(project=PROJECT_ID, location=REGION)
db = firestore.Client(project=PROJECT_ID)


def embed_query(query_text: str) -> list[float]:
    """Embed a user query using RETRIEVAL_QUERY task type."""
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    result = model.get_embeddings([
        TextEmbeddingInput(text=query_text, task_type="RETRIEVAL_QUERY")
    ])
    return result[0].values


def retrieve_chunks(
    query_text: str,
    top_k: int = 5,
    city_filter: str = None,       # "Abuja" | "Kaduna" | None (both)
    category_filter: str = None,   # "terminology" | "neighborhood" | None
) -> list[dict]:
    """
    Full RAG retrieval:
      1. Embed the query (RETRIEVAL_QUERY task type)
      2. Query Vertex AI Vector Search with optional metadata filters
      3. Fetch full content + metadata from Firestore
      4. Return hydrated chunks for Gemini Pro context window
    """
    query_embedding = embed_query(query_text)

    # ── Build metadata restricts ────────────────────────────────────
    restricts = []
    if city_filter:
        restricts.append({
            "namespace": "city",
            "allow_tokens": [city_filter]
        })
    if category_filter:
        restricts.append({
            "namespace": "category",
            "allow_tokens": [category_filter]
        })

    # ── Query Vector Search ─────────────────────────────────────────
    endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=ENDPOINT_NAME,
        project=PROJECT_ID,
        location=REGION
    )
    response = endpoint.find_neighbors(
        deployed_index_id = DEPLOYED_ID,
        queries           = [query_embedding],
        num_neighbors     = top_k,
        filter            = restricts if restricts else None,
    )
    neighbor_ids = [n.id for n in response[0]]
    print(f"  Vector Search returned {len(neighbor_ids)} neighbors for: '{query_text}'")

    # ── Hydrate from Firestore ──────────────────────────────────────
    hydrated = []
    for chunk_id in neighbor_ids:
        doc = db.collection(COLLECTION).document(chunk_id).get()
        if doc.exists:
            hydrated.append(doc.to_dict())
        else:
            print(f"  ⚠️  Chunk {chunk_id} not found in Firestore")

    return hydrated


def format_context_for_gemini(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for the Gemini Pro prompt."""
    lines = ["=== PropaBridge Knowledge Context ===\n"]
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[Source {i}: {chunk.get('category', 'unknown')} / {chunk.get('source_key', '')}]")
        lines.append(chunk.get("content", ""))
        lines.append("")
    return "\n".join(lines)


# ── Example usage ───────────────────────────────────────────────────────
if __name__ == "__main__":
    query = "What is a Certificate of Occupancy and why does it matter in Nigeria?"
    print(f"Query: {query}\n")

    chunks = retrieve_chunks(query, top_k=3)
    context = format_context_for_gemini(chunks)
    print(context)
