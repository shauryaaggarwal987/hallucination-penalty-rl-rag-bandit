import sys
from pathlib import Path
from collections import Counter

# Add repo root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from config import FAISS_INDEX, METADATA_JSON, DEFAULT_EMBED_MODEL
from utils.io import safe_read_json

def inspect_index():
    print("--- FAISS Index Inspector ---")
    
    # Check paths
    print(f"Index Path: {FAISS_INDEX}")
    print(f"Metadata Path: {METADATA_JSON}")
    print(f"Embedding Model Configured: {DEFAULT_EMBED_MODEL}")
    
    if not FAISS_INDEX.exists() or not METADATA_JSON.exists():
        print("\nERROR: Index or metadata not found. Run 'python -m rag.build_index' first.")
        return
        
    metadata = safe_read_json(METADATA_JSON, [])
    print(f"\nTotal Indexed Chunks: {len(metadata)}")
    
    if not metadata:
        print("Metadata is empty.")
        return
        
    source_counts = Counter(chunk.get("doc_name", "unknown") for chunk in metadata)
    
    print(f"\nUnique Source Files in Index: {len(source_counts)}")
    print("\nTop 10 Sources by Chunk Count:")
    for source, count in source_counts.most_common(10):
        print(f"  - {source}: {count} chunks")
        
    print("\nSample Chunk from Index (Index 0):")
    sample = metadata[0]
    print(f"  Doc: {sample.get('doc_name')}")
    print(f"  Text (first 100 chars): {sample.get('text', '')[:100]}...")

if __name__ == "__main__":
    inspect_index()
