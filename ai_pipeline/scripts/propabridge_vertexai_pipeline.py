
import json
import hashlib
import time
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime, timezone


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION  — edit these before running
# ═══════════════════════════════════════════════════════════════

class Config:
    # GCP project settings
    PROJECT_ID          = os.getenv("GCP_PROJECT_ID", "propabridge-prod")
    REGION              = os.getenv("GCP_REGION", "us-central1")

    # Embedding model — text-embedding-004 is the latest stable Gemini embedding
    # Alternatives: "textembedding-gecko@003" (768d), "text-multilingual-embedding-002"
    EMBEDDING_MODEL     = "text-embedding-004"
    EMBEDDING_DIMENSION = 768

    # Task type tells the model how the embedding will be used
    # RETRIEVAL_DOCUMENT = for indexing (document side)
    # RETRIEVAL_QUERY    = for querying (query side at inference time)
    EMBEDDING_TASK_TYPE = "RETRIEVAL_DOCUMENT"

    # Vertex AI Vector Search Index settings
    INDEX_DISPLAY_NAME      = "propabridge-rag-index"
    INDEX_DESCRIPTION       = "PropaBridge real estate RAG chunks — Abuja & Kaduna"
    INDEX_ENDPOINT_NAME     = "propabridge-rag-endpoint"
    DEPLOYED_INDEX_ID       = "propabridge_deployed_v1"
    APPROXIMATE_NEIGHBORS   = 10          # neighbors to return at query time

    # Firestore
    FIRESTORE_COLLECTION    = "propabridge_chunks"
    FIRESTORE_DATABASE      = "(default)"  # or your named DB

    # GCS bucket for Vector Search index data
    GCS_BUCKET              = os.getenv("GCS_BUCKET", "propabridge-vector-index")
    GCS_INDEX_PREFIX        = "vertex_index/v1"

    # Source JSON
    SOURCE_JSON             = "/mnt/user-data/outputs/propabridge_rag_context.json"

    # Output directory for generated JSONL files
    OUTPUT_DIR              = Path("./propabridge_vertex_output")

    # Batch size for embedding API calls (max 250 per request for text-embedding-004)
    EMBEDDING_BATCH_SIZE    = 50

    # Rate limiting — sleep between batches (seconds)
    BATCH_SLEEP_SECONDS     = 0.5


# ═══════════════════════════════════════════════════════════════
# DATA MODEL
# ═══════════════════════════════════════════════════════════════

@dataclass
class PropaBridgeChunk:
    """
    Core chunk model. Holds both the content for embedding
    and the metadata for Firestore + Vector Search restricts.
    """
    chunk_id:       str
    content:        str          # Text to embed
    category:       str          # terminology | neighborhood | tier_summary | agent_instructions
    subcategory:    str          # formal_legal | slang | abuja_premium | kaduna_mid_range ...
    source_key:     str          # Original JSON key (for traceability)
    token_estimate: int = 0

    # Neighborhood-specific (None for terminology chunks)
    city:           Optional[str] = None          # Abuja | Kaduna
    tier:           Optional[str] = None          # Premium | Mid Range | Affordable
    district:       Optional[str] = None
    rent_range_ngn: Optional[str] = None
    latitude:       Optional[float] = None
    longitude:      Optional[float] = None

    # Terminology-specific
    term_display:   Optional[str] = None
    has_legal_refs: bool = False
    has_due_diligence: bool = False
    synonyms:       list = field(default_factory=list)

    created_at:     str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Populated after embedding API call
    embedding:      Optional[list[float]] = None

    def to_firestore_doc(self) -> dict:
        """Serialise to Firestore document (embedding excluded — stored in Vector Search)."""
        doc = asdict(self)
        doc.pop("embedding", None)      # Don't store 768d vector in Firestore
        doc["synonyms"] = ", ".join(self.synonyms) if self.synonyms else ""
        return doc

    def to_vertex_jsonl_record(self) -> dict:
        """
        Serialise to Vertex AI Vector Search JSONL record format.
        Ref: https://cloud.google.com/vertex-ai/docs/vector-search/setup/setup
        """
        if self.embedding is None:
            raise ValueError(f"Chunk {self.chunk_id} has no embedding. Call embed() first.")

        record = {
            "id": self.chunk_id,
            "embedding": self.embedding,
        }

        # ── String restricts (categorical metadata filters) ──────────
        # These enable server-side pre-filtering at query time, e.g.:
        #   query(filter=[Namespace("city", allow=["Abuja"])])
        restricts = []

        restricts.append({
            "namespace": "category",
            "allow": [self.category]
        })
        restricts.append({
            "namespace": "subcategory",
            "allow": [self.subcategory]
        })
        if self.city:
            restricts.append({
                "namespace": "city",
                "allow": [self.city]
            })
        if self.tier:
            restricts.append({
                "namespace": "tier",
                "allow": [self.tier.replace(" ", "_").lower()]
            })
        if self.has_legal_refs:
            restricts.append({
                "namespace": "has_legal_refs",
                "allow": ["true"]
            })
        if self.has_due_diligence:
            restricts.append({
                "namespace": "has_due_diligence",
                "allow": ["true"]
            })

        if restricts:
            record["restricts"] = restricts

        # ── Numeric restricts (range filters) ────────────────────────
        numeric_restricts = []

        numeric_restricts.append({
            "namespace": "token_count",
            "value_int": self.token_estimate
        })
        if self.latitude is not None:
            numeric_restricts.append({
                "namespace": "latitude",
                "value_float": self.latitude
            })
        if self.longitude is not None:
            numeric_restricts.append({
                "namespace": "longitude",
                "value_float": self.longitude
            })

        if numeric_restricts:
            record["numeric_restricts"] = numeric_restricts

        return record


# ═══════════════════════════════════════════════════════════════
# CHUNKING LOGIC  (same semantic strategy as v1, Vertex-native output)
# ═══════════════════════════════════════════════════════════════

def _make_id(scope: str, key: str) -> str:
    raw = f"{scope}::{key}".lower().replace(" ", "_")
    return "pb_" + hashlib.md5(raw.encode()).hexdigest()[:12]

def _tokens(text: str) -> int:
    return len(text) // 4

def _prose(items: list) -> str:
    if not items: return ""
    if len(items) == 1: return str(items[0])
    return ", ".join(str(i) for i in items[:-1]) + f", and {items[-1]}"

def _clean(key: str) -> str:
    return key.replace("_", " ").strip()


def _build_terminology_chunk(term_key: str, data: dict, subcategory: str) -> PropaBridgeChunk:
    display = data.get("term", _clean(term_key))
    parts = [f"[TERM: {display}]"]
    parts.append(f"Definition: {data.get('definition', '')}")

    for label, field_name in [
        ("Nigerian Context",        "nigerian_context"),
        ("Risk Level",              "risk_level"),
        ("Importance",              "importance_score"),
        ("Due Diligence Action",    "due_diligence_action"),
        ("Important Note",          "importance"),
    ]:
        if val := data.get(field_name):
            parts.append(f"{label}: {val}")

    for label, field_name in [
        ("Also known as",           "synonyms"),
        ("Legal References",        "legal_references"),
        ("Key Clauses",             "key_clauses"),
        ("Key Components",          "key_components"),
        ("Common Issues",           "common_issues"),
        ("Types",                   "types"),
        ("Benefits",                "benefits"),
        ("Key Institutions",        "key_institutions"),
        ("Nigerian Examples",       "nigerian_examples"),
        ("Administering Bodies",    "administering_bodies"),
        ("Recommended Professionals", "recommended_professionals"),
        ("Inspection Checklist",    "checklist"),
    ]:
        if vals := data.get(field_name):
            parts.append(f"{label}: {_prose(vals)}.")

    if regulator := data.get("regulator"):
        parts.append(f"Regulator: {regulator}")

    if rates := data.get("rates"):
        rate_text = "; ".join(f"{_clean(k)}: {v}" for k, v in rates.items())
        parts.append(f"Rate Structure: {rate_text}.")

    for label, field_name in [
        ("Typical Annual Rent in Abuja",  "typical_annual_rent_abuja"),
        ("Typical Annual Rent in Kaduna", "typical_annual_rent_kaduna"),
    ]:
        if val := data.get(field_name):
            parts.append(f"{label}: {val}")

    content = "\n".join(parts)

    return PropaBridgeChunk(
        chunk_id       = _make_id("terminology", term_key),
        content        = content,
        category       = "terminology",
        subcategory    = subcategory,
        source_key     = term_key,
        token_estimate = _tokens(content),
        term_display   = display,
        has_legal_refs = bool(data.get("legal_references")),
        has_due_diligence = bool(data.get("due_diligence_action")),
        synonyms       = data.get("synonyms", []),
    )


def _build_district_chunk(district_key: str, district: dict,
                           city: str, tier: str, tier_data: dict) -> PropaBridgeChunk:
    tier_display = _clean(tier).title()
    rent = district.get("annual_rent_2bed_ngn",
           tier_data.get("estimated_2bed_annual_rent", {}).get("range_ngn", "Contact agent"))

    parts = [
        f"[NEIGHBORHOOD: {_clean(district_key).title()} | City: {city} | Tier: {tier_display}]",
        district.get("description", ""),
        f"City: {city}. Market Tier: {tier_display}.",
        f"Estimated Annual Rent for a Standard 2-Bedroom Apartment: {rent}.",
    ]

    if notes := tier_data.get("estimated_2bed_annual_rent", {}).get("notes"):
        parts.append(f"Market Note: {notes}")
    if features := district.get("key_features"):
        parts.append(f"Key Features: {_prose(features)}.")
    if ptypes := district.get("property_types"):
        parts.append(f"Common Property Types: {_prose(ptypes)}.")
    if tier_desc := tier_data.get("description"):
        parts.append(f"Tier Profile: {tier_desc}")
    if coords := district.get("coordinates"):
        parts.append(
            f"Coordinates: Latitude {coords['latitude']}, Longitude {coords['longitude']}."
        )

    content = "\n".join(parts)
    coords  = district.get("coordinates", {})

    return PropaBridgeChunk(
        chunk_id       = _make_id(f"neighborhood_{city.lower()}", district_key),
        content        = content,
        category       = "neighborhood",
        subcategory    = f"{city.lower()}_{tier_display.lower().replace(' ', '_')}",
        source_key     = district_key,
        token_estimate = _tokens(content),
        city           = city,
        tier           = tier_display,
        district       = _clean(district_key).title(),
        rent_range_ngn = rent,
        latitude       = coords.get("latitude"),
        longitude      = coords.get("longitude"),
    )


def _build_tier_summary_chunk(city: str, tier: str, tier_data: dict) -> PropaBridgeChunk:
    tier_display  = _clean(tier).title()
    rent_info     = tier_data.get("estimated_2bed_annual_rent", {})
    district_names = [_clean(k).title() for k in tier_data.get("districts", {}).keys()]

    parts = [
        f"[TIER SUMMARY: {tier_display} Neighborhoods in {city}]",
        tier_data.get("description", ""),
        f"Estimated Annual Rent for a Standard 2-Bedroom Apartment: {rent_info.get('range_ngn', 'Varies')}.",
    ]
    if notes := rent_info.get("notes"):
        parts.append(f"Note: {notes}")
    if district_names:
        parts.append(f"Districts in this tier: {_prose(district_names)}.")

    content = "\n".join(parts)

    return PropaBridgeChunk(
        chunk_id       = _make_id(f"tier_summary_{city.lower()}", tier),
        content        = content,
        category       = "tier_summary",
        subcategory    = f"{city.lower()}_{tier_display.lower().replace(' ', '_')}",
        source_key     = f"{city}_{tier}",
        token_estimate = _tokens(content),
        city           = city,
        tier           = tier_display,
        rent_range_ngn = rent_info.get("range_ngn"),
    )


def _build_agent_instructions_chunk(instructions: dict) -> PropaBridgeChunk:
    parts = [
        "[AGENT INSTRUCTIONS: PropaBridge AI Search Agent]",
        f"Purpose: {instructions.get('purpose', '')}",
        "Usage Guidelines:"
    ]
    for i, g in enumerate(instructions.get("usage_guidelines", []), 1):
        parts.append(f"  {i}. {g}")
    if d := instructions.get("disclaimer"):
        parts.append(f"Disclaimer: {d}")

    content = "\n".join(parts)
    return PropaBridgeChunk(
        chunk_id       = _make_id("system", "agent_instructions"),
        content        = content,
        category       = "agent_instructions",
        subcategory    = "system",
        source_key     = "agent_instructions",
        token_estimate = _tokens(content),
    )


def build_all_chunks(json_path: str) -> list[PropaBridgeChunk]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks: list[PropaBridgeChunk] = []

    print("  → Terminology: formal legal terms")
    for k, v in data["terminology"]["formal_legal_terms"].items():
        chunks.append(_build_terminology_chunk(k, v, "formal_legal"))

    print("  → Terminology: local slang & informal terms")
    for k, v in data["terminology"]["local_slang_and_informal_terms"].items():
        chunks.append(_build_terminology_chunk(k, v, "slang_informal"))

    for city_key, city_label in [("neighborhoods_abuja", "Abuja"),
                                  ("neighborhoods_kaduna", "Kaduna")]:
        print(f"  → Neighborhoods: {city_label}")
        for tier_key, tier_data in data[city_key]["tiers"].items():
            chunks.append(_build_tier_summary_chunk(city_label, tier_key, tier_data))
            for dist_key, dist_data in tier_data.get("districts", {}).items():
                chunks.append(_build_district_chunk(
                    dist_key, dist_data, city_label, tier_key, tier_data
                ))

    print("  → Agent instructions")
    chunks.append(_build_agent_instructions_chunk(data.get("agent_instructions", {})))

    return chunks


# ═══════════════════════════════════════════════════════════════
# VERTEX AI EMBEDDING  (production + dry-run mode)
# ═══════════════════════════════════════════════════════════════

def embed_chunks_vertex(
    chunks: list[PropaBridgeChunk],
    dry_run: bool = False
) -> list[PropaBridgeChunk]:
    """
    Call Vertex AI Embeddings API (text-embedding-004) for each chunk.

    Production path  (dry_run=False):
        Uses google-cloud-aiplatform SDK. Requires:
          pip install google-cloud-aiplatform
          gcloud auth application-default login
          export GCP_PROJECT_ID=your-project-id

    Dry-run path (dry_run=True):
        Generates deterministic mock embeddings so the full pipeline
        can be validated without GCP credentials.
    """

    if dry_run:
        print("\n  ⚠️  DRY-RUN MODE — mock embeddings (cos-normalised random vectors)")
        print("      Set dry_run=False and configure GCP credentials for production.\n")
        return _embed_dry_run(chunks)

    # ── Production: real Vertex AI SDK call ──────────────────────────
    try:
        import vertexai
        from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
    except ImportError:
        raise ImportError(
            "google-cloud-aiplatform not installed.\n"
            "Run: pip install google-cloud-aiplatform\n"
            "Then re-run with dry_run=False."
        )

    vertexai.init(project=Config.PROJECT_ID, location=Config.REGION)
    model = TextEmbeddingModel.from_pretrained(Config.EMBEDDING_MODEL)

    print(f"\n  🔗 Vertex AI Embedding Model: {Config.EMBEDDING_MODEL}")
    print(f"     Dimension: {Config.EMBEDDING_DIMENSION}d")
    print(f"     Task type: {Config.EMBEDDING_TASK_TYPE}")
    print(f"     Batch size: {Config.EMBEDDING_BATCH_SIZE}\n")

    total = len(chunks)
    for batch_start in range(0, total, Config.EMBEDDING_BATCH_SIZE):
        batch = chunks[batch_start: batch_start + Config.EMBEDDING_BATCH_SIZE]
        batch_num = batch_start // Config.EMBEDDING_BATCH_SIZE + 1
        total_batches = (total + Config.EMBEDDING_BATCH_SIZE - 1) // Config.EMBEDDING_BATCH_SIZE

        print(f"  Embedding batch {batch_num}/{total_batches} "
              f"({len(batch)} chunks)...", end=" ", flush=True)

        inputs = [
            TextEmbeddingInput(
                text      = chunk.content,
                task_type = Config.EMBEDDING_TASK_TYPE,
                title     = chunk.source_key        # improves retrieval quality
            )
            for chunk in batch
        ]

        try:
            results = model.get_embeddings(inputs)
            for chunk, result in zip(batch, results):
                chunk.embedding = result.values
            print(f"✓  ({results[0].statistics.token_count} tokens in first chunk)")
        except Exception as e:
            print(f"\n  ✗ Batch {batch_num} failed: {e}")
            raise

        if batch_start + Config.EMBEDDING_BATCH_SIZE < total:
            time.sleep(Config.BATCH_SLEEP_SECONDS)

    return chunks


def _embed_dry_run(chunks: list[PropaBridgeChunk]) -> list[PropaBridgeChunk]:
    """Generate deterministic mock embeddings for pipeline validation."""
    import math

    def _mock_vector(seed_text: str, dim: int = 768) -> list[float]:
        """Deterministic unit-norm vector derived from chunk content hash."""
        seed = int(hashlib.sha256(seed_text.encode()).hexdigest(), 16)
        vec = []
        for i in range(dim):
            # Pseudo-random but fully deterministic float in [-1, 1]
            val = math.sin(seed * (i + 1) * 0.0001) * math.cos(seed * 0.00007 + i)
            vec.append(val)
        # Cosine normalise
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [round(v / norm, 8) for v in vec]

    for chunk in chunks:
        chunk.embedding = _mock_vector(chunk.content)
        print(f"     [DRY-RUN] Embedded: {chunk.source_key:<35} dim={len(chunk.embedding)}")

    return chunks


# ═══════════════════════════════════════════════════════════════
# VERTEX AI VECTOR SEARCH — INDEX JSONL EXPORT
# ═══════════════════════════════════════════════════════════════

def export_vertex_index_jsonl(
    chunks: list[PropaBridgeChunk],
    output_path: str
) -> str:
    """
    Write the Vertex AI Vector Search index JSONL file.

    Each line = one valid JSON record:
      { "id": "pb_xxx", "embedding": [...768 floats...],
        "restricts": [...], "numeric_restricts": [...] }

    Upload this file to GCS, then create/update your Vector Search index.
    GCS path: gs://{GCS_BUCKET}/{GCS_INDEX_PREFIX}/embeddings.json
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            record = chunk.to_vertex_jsonl_record()
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n  ✅ Vertex AI index JSONL → {output_path}")
    print(f"     {written} records written")
    print(f"\n  Next step: Upload to GCS")
    print(f"     gsutil cp {output_path} gs://{Config.GCS_BUCKET}/{Config.GCS_INDEX_PREFIX}/embeddings.json")
    return output_path


# ═══════════════════════════════════════════════════════════════
# FIRESTORE EXPORT
# ═══════════════════════════════════════════════════════════════

def export_firestore_jsonl(
    chunks: list[PropaBridgeChunk],
    output_path: str
) -> str:
    """
    Write a JSONL file of Firestore documents (metadata + full content,
    without the 768-dim embedding vector).

    Production usage — batch write to Firestore:
        from google.cloud import firestore
        db = firestore.Client(project=Config.PROJECT_ID)
        batch = db.batch()
        for doc in docs:
            ref = db.collection(Config.FIRESTORE_COLLECTION).document(doc["chunk_id"])
            batch.set(ref, doc)
        batch.commit()

    The Firestore doc is fetched at query time AFTER Vertex AI Vector Search
    returns the top-k chunk_ids, to hydrate the full chunk content for the
    Gemini Pro prompt context.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            doc = chunk.to_firestore_doc()
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"  ✅ Firestore documents JSONL → {output_path}")
    print(f"     {len(chunks)} documents (no embeddings — those live in Vector Search)")
    return output_path


def export_firestore_batch_script(output_path: str):
    """Generate a ready-to-run Python script for bulk Firestore import."""
    script = '''\
"""
PropaBridge — Firestore Bulk Import Script
Run this after generating firestore_documents.jsonl
  pip install google-cloud-firestore
  python3 firestore_import.py
"""
import json
from google.cloud import firestore

PROJECT_ID  = "propabridge-prod"   # ← change this
COLLECTION  = "propabridge_chunks"
INPUT_FILE  = "propabridge_vertex_output/firestore_documents.jsonl"
BATCH_SIZE  = 500   # Firestore max batch size

db = firestore.Client(project=PROJECT_ID)

with open(INPUT_FILE) as f:
    docs = [json.loads(line) for line in f if line.strip()]

print(f"Importing {len(docs)} documents into {COLLECTION}...")

for i in range(0, len(docs), BATCH_SIZE):
    batch = db.batch()
    for doc in docs[i : i + BATCH_SIZE]:
        ref = db.collection(COLLECTION).document(doc["chunk_id"])
        batch.set(ref, doc)
    batch.commit()
    print(f"  ✓ Committed batch {i // BATCH_SIZE + 1}")

print("\\n✅ Firestore import complete!")
'''
    with open(output_path, "w") as f:
        f.write(script)
    print(f"  ✅ Firestore import script → {output_path}")


# ═══════════════════════════════════════════════════════════════
# VERTEX AI VECTOR SEARCH — INDEX CREATION SCRIPT
# ═══════════════════════════════════════════════════════════════

def export_vertex_index_creation_script(output_path: str):
    """Generate a ready-to-run script to create the Vertex AI Vector Search index."""
    script = f'''\
"""
PropaBridge — Vertex AI Vector Search Index Creation Script
============================================================
Run after uploading the JSONL to GCS:
  gsutil cp propabridge_vertex_output/vertex_index.jsonl \\
    gs://{Config.GCS_BUCKET}/{Config.GCS_INDEX_PREFIX}/embeddings.json

  pip install google-cloud-aiplatform
  python3 create_vertex_index.py
"""
import vertexai
from google.cloud import aiplatform

PROJECT_ID     = "{Config.PROJECT_ID}"
REGION         = "{Config.REGION}"
GCS_BUCKET     = "{Config.GCS_BUCKET}"
GCS_PREFIX     = "{Config.GCS_INDEX_PREFIX}"
INDEX_NAME     = "{Config.INDEX_DISPLAY_NAME}"
ENDPOINT_NAME  = "{Config.INDEX_ENDPOINT_NAME}"
DEPLOYED_ID    = "{Config.DEPLOYED_INDEX_ID}"
DIMENSIONS     = {Config.EMBEDDING_DIMENSION}

vertexai.init(project=PROJECT_ID, location=REGION)

# ── Step 1: Create the Vector Search Index ─────────────────────────────
print("Creating Vertex AI Vector Search Index...")
index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name         = INDEX_NAME,
    description          = "{Config.INDEX_DESCRIPTION}",
    contents_delta_uri   = f"gs://{{GCS_BUCKET}}/{{GCS_PREFIX}}/",
    dimensions           = DIMENSIONS,
    approximate_neighbors_count = {Config.APPROXIMATE_NEIGHBORS},
    distance_measure_type = "DOT_PRODUCT_DISTANCE",   # works with cosine-normalised vectors
    index_update_method  = "BATCH_UPDATE",
)
print(f"  ✓ Index created: {{index.resource_name}}")

# ── Step 2: Create an Index Endpoint ───────────────────────────────────
print("Creating Index Endpoint...")
endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name = ENDPOINT_NAME,
    public_endpoint_enabled = True,
)
print(f"  ✓ Endpoint created: {{endpoint.resource_name}}")

# ── Step 3: Deploy Index to Endpoint ───────────────────────────────────
print("Deploying index to endpoint (may take 20-40 mins)...")
endpoint.deploy_index(
    index            = index,
    deployed_index_id = DEPLOYED_ID,
)
print(f"  ✓ Deployed index ID: {{DEPLOYED_ID}}")
print("\\n✅ Vertex AI Vector Search is live!")
print(f"   Endpoint: {{endpoint.resource_name}}")
'''
    with open(output_path, "w") as f:
        f.write(script)
    print(f"  ✅ Vertex index creation script → {output_path}")


# ═══════════════════════════════════════════════════════════════
# QUERY HELPER  (shows how to query at runtime)
# ═══════════════════════════════════════════════════════════════

def export_query_helper_script(output_path: str):
    """Generate the RAG query helper used by the Gemini Pro agent at runtime."""
    script = f'''\
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
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from google.cloud import aiplatform, firestore

PROJECT_ID     = "{Config.PROJECT_ID}"
REGION         = "{Config.REGION}"
EMBEDDING_MODEL = "{Config.EMBEDDING_MODEL}"
ENDPOINT_NAME  = "{Config.INDEX_ENDPOINT_NAME}"
DEPLOYED_ID    = "{Config.DEPLOYED_INDEX_ID}"
COLLECTION     = "{Config.FIRESTORE_COLLECTION}"

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
        restricts.append({{
            "namespace": "city",
            "allow_tokens": [city_filter]
        }})
    if category_filter:
        restricts.append({{
            "namespace": "category",
            "allow_tokens": [category_filter]
        }})

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
    print(f"  Vector Search returned {{len(neighbor_ids)}} neighbors for: '{{query_text}}'")

    # ── Hydrate from Firestore ──────────────────────────────────────
    hydrated = []
    for chunk_id in neighbor_ids:
        doc = db.collection(COLLECTION).document(chunk_id).get()
        if doc.exists:
            hydrated.append(doc.to_dict())
        else:
            print(f"  ⚠️  Chunk {{chunk_id}} not found in Firestore")

    return hydrated


def format_context_for_gemini(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for the Gemini Pro prompt."""
    lines = ["=== PropaBridge Knowledge Context ===\\n"]
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[Source {{i}}: {{chunk.get('category', 'unknown')}} / {{chunk.get('source_key', '')}}]")
        lines.append(chunk.get("content", ""))
        lines.append("")
    return "\\n".join(lines)


# ── Example usage ───────────────────────────────────────────────────────
if __name__ == "__main__":
    query = "What is a Certificate of Occupancy and why does it matter in Nigeria?"
    print(f"Query: {{query}}\\n")

    chunks = retrieve_chunks(query, top_k=3)
    context = format_context_for_gemini(chunks)
    print(context)
'''
    with open(output_path, "w") as f:
        f.write(script)
    print(f"  ✅ RAG query helper script → {output_path}")


# ═══════════════════════════════════════════════════════════════
# PIPELINE SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════

def print_summary(chunks: list[PropaBridgeChunk]):
    from collections import Counter
    cat_counts = Counter(c.category for c in chunks)
    city_counts = Counter(c.city for c in chunks if c.city)
    total_tokens = sum(c.token_estimate for c in chunks)

    print("\n" + "═" * 62)
    print("  PROPABRIDGE VERTEX AI PIPELINE — SUMMARY")
    print("═" * 62)
    print(f"  Embedding model : {Config.EMBEDDING_MODEL} ({Config.EMBEDDING_DIMENSION}d)")
    print(f"  Total chunks    : {len(chunks)}")
    print(f"  Total tokens    : {total_tokens:,}  (est.)")
    print(f"  Avg tokens      : {total_tokens // len(chunks)}")
    print()
    print("  Chunks by category:")
    for cat, n in sorted(cat_counts.items()):
        print(f"    {cat:<28} {n:>4}")
    print()
    print("  Neighborhood chunks by city:")
    for city, n in sorted(city_counts.items()):
        print(f"    {city:<28} {n:>4}")
    print()
    print("  Vector Search restricts available:")
    print("    String  → category, subcategory, city, tier,")
    print("               has_legal_refs, has_due_diligence")
    print("    Numeric → token_count, latitude, longitude")
    print("═" * 62)


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    OUT = Config.OUTPUT_DIR
    OUT.mkdir(exist_ok=True)

    print("\n🏗️  PropaBridge — Vertex AI Vector Search + Firestore Pipeline")
    print("═" * 62)
    print(f"  Source  : {Config.SOURCE_JSON}")
    print(f"  Model   : {Config.EMBEDDING_MODEL} ({Config.EMBEDDING_DIMENSION}d)")
    print(f"  Project : {Config.PROJECT_ID}")
    print(f"  Region  : {Config.REGION}")
    print("═" * 62 + "\n")

    # ── 1. Build chunks ─────────────────────────────────────────
    print("STEP 1 — Chunking context JSON")
    chunks = build_all_chunks(Config.SOURCE_JSON)
    print(f"         {len(chunks)} chunks built\n")

    # ── 2. Embed (dry_run=True for validation; False in production) ──
    print("STEP 2 — Generating embeddings")
    chunks = embed_chunks_vertex(chunks, dry_run=True)
    print(f"         {len(chunks)} chunks embedded\n")

    # ── 3. Export Vertex AI Vector Search JSONL ──────────────────
    print("STEP 3 — Exporting Vertex AI Vector Search index JSONL")
    export_vertex_index_jsonl(chunks, str(OUT / "vertex_index.jsonl"))

    # ── 4. Export Firestore documents ────────────────────────────
    print("\nSTEP 4 — Exporting Firestore documents JSONL")
    export_firestore_jsonl(chunks, str(OUT / "firestore_documents.jsonl"))

    # ── 5. Export helper scripts ─────────────────────────────────
    print("\nSTEP 5 — Generating deployment scripts")
    export_firestore_batch_script(str(OUT / "firestore_import.py"))
    export_vertex_index_creation_script(str(OUT / "create_vertex_index.py"))
    export_query_helper_script(str(OUT / "query_helper.py"))

    # ── Summary ──────────────────────────────────────────────────
    print_summary(chunks)

    print(f"\n  GCS upload command (run after configuring gcloud):")
    print(f"    gsutil cp {OUT}/vertex_index.jsonl \\")
    print(f"      gs://{Config.GCS_BUCKET}/{Config.GCS_INDEX_PREFIX}/embeddings.json\n")
    print(f"  Then run: python3 {OUT}/create_vertex_index.py\n")
    print("🎉  All files ready for deployment!\n")
