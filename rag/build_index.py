import sys
from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Add repo root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from config import CHUNKS_JSONL, FAISS_INDEX, METADATA_JSON, DEFAULT_EMBED_MODEL, OUTPUTS_DIR
from utils.io import read_jsonl, safe_write_json

def build_index():
    print("Loading chunks...")
    chunks = read_jsonl(CHUNKS_JSONL)
    if not chunks:
        print("No chunks found in chunks.jsonl. Run ingest.py first.")
        return

    print(f"Loading embedding model: {DEFAULT_EMBED_MODEL}")
    model = SentenceTransformer(DEFAULT_EMBED_MODEL)
    
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Normalize for cosine similarity (Inner Product on normalized vectors)
    print("Normalizing embeddings...")
    faiss.normalize_L2(embeddings)
    
    # Build FAISS Index
    d = embeddings.shape[1]
    print(f"Building FAISS IndexFlatIP with dim={d}...")
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    
    # Save Index
    print(f"Saving index to {FAISS_INDEX}...")
    faiss.write_index(index, str(FAISS_INDEX))
    
    # Save Metadata (aligned by index ID)
    # We just save the list of chunks in order. ID 0 = chunks[0]
    print(f"Saving metadata to {METADATA_JSON}...")
    safe_write_json(METADATA_JSON, chunks)
    
    print("Index build complete.")

if __name__ == "__main__":
    build_index()
