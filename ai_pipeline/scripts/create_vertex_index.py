
import os
import vertexai
from google.cloud import aiplatform

# 1. Add your digital passport
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ai_pipeline/service_account.json"

# 2. Update to the true Project ID
PROJECT_ID     = "studio-6084005125-75144"
REGION         = "us-central1"
GCS_BUCKET     = "propabridge-vector-index"
GCS_PREFIX     = "vertex_index/v1"
INDEX_NAME     = "propabridge-rag-index"
ENDPOINT_NAME  = "propabridge-rag-endpoint"
DEPLOYED_ID    = "propabridge_deployed_v1"
DIMENSIONS     = 768

vertexai.init(project=PROJECT_ID, location=REGION)

# ── Step 1: Create the Vector Search Index ─────────────────────────────
print("Creating Vertex AI Vector Search Index...")
index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name         = INDEX_NAME,
    description          = "PropaBridge real estate RAG chunks — Abuja & Kaduna",
    contents_delta_uri   = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/",
    dimensions           = DIMENSIONS,
    approximate_neighbors_count = 10,
    distance_measure_type = "DOT_PRODUCT_DISTANCE",   # works with cosine-normalised vectors
    index_update_method  = "BATCH_UPDATE",
)
print(f"  ✓ Index created: {index.resource_name}")

# ── Step 2: Create an Index Endpoint ───────────────────────────────────
print("Creating Index Endpoint...")
endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name = ENDPOINT_NAME,
    public_endpoint_enabled = True,
)
print(f"  ✓ Endpoint created: {endpoint.resource_name}")

# ── Step 3: Deploy Index to Endpoint ───────────────────────────────────
print("Deploying index to endpoint (may take 20-40 mins)...")
endpoint.deploy_index(
    index            = index,
    deployed_index_id = DEPLOYED_ID,
)
print(f"  ✓ Deployed index ID: {DEPLOYED_ID}")
print("\n✅ Vertex AI Vector Search is live!")
print(f"   Endpoint: {endpoint.resource_name}")
