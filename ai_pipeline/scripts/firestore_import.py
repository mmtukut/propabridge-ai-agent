"""
PropaBridge — Firestore Bulk Import Script
"""
import os
import json
from google.cloud import firestore

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ai_pipeline/service_account.json"
PROJECT_ID  = "studio-6084005125-75144"   

COLLECTION  = "propabridge_chunks"

INPUT_FILE  = "ai_pipeline/data/firestore_documents.jsonl"
BATCH_SIZE  = 500   # Firestore max batch size

db = firestore.Client(project=PROJECT_ID)

with open(INPUT_FILE) as f:
    docs = [json.loads(line) for line in f if line.strip()]

print(f"Importing {len(docs)} documents into {COLLECTION}...")

for i in range(0, len(docs), BATCH_SIZE):
    batch = db.batch()
    for doc in docs[i : i + BATCH_SIZE]:
        # doc["chunk_id"] will be the unique document ID in Firestore
        ref = db.collection(COLLECTION).document(doc["chunk_id"])
        batch.set(ref, doc)
    batch.commit()
    print(f"  ✓ Committed batch {i // BATCH_SIZE + 1}")

print("\n✅ Firestore import complete!")