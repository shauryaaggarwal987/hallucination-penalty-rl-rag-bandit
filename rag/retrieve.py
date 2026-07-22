import sys
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple

# Add repo root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from config import FAISS_INDEX, METADATA_JSON, DEFAULT_EMBED_MODEL, DEFAULT_TOP_K
from utils.io import safe_read_json

# Global cache to avoid reloading model/index on every call
_INDEX = None
_METADATA = None
_MODEL = None

def load_resources(force_reload: bool = False):
    global _INDEX, _METADATA, _MODEL
    
    if force_reload:
        _INDEX = None
        _METADATA = None
        
    if _INDEX is None:
        import faiss
        if not FAISS_INDEX.exists():
            print(f"Index not found at {FAISS_INDEX}. Run build_index.py.")
            return
        _INDEX = faiss.read_index(str(FAISS_INDEX))
        
    if _METADATA is None:
        _METADATA = safe_read_json(METADATA_JSON, [])
        
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(DEFAULT_EMBED_MODEL)

def retrieve(question: str, top_k: int = DEFAULT_TOP_K, require_nontrivial_corpus: bool = False) -> Tuple[List[str], List[str], List[Dict]]:
    """
    Returns:
        contexts: list of chunk texts
        sources: list of source strings "doc_name#chunk_id"
        hits: list of full hit objects with score
    """
    load_resources()
    if _INDEX is None or _MODEL is None:
        raise RuntimeError("Retrieval resources could not be loaded")
    if require_nontrivial_corpus and _INDEX.ntotal <= top_k:
        raise RuntimeError("Final-mode corpus guard failed: index size must exceed top_k")
    
    # Embed query
    q_vec = _MODEL.encode([question])
    import faiss
    faiss.normalize_L2(q_vec)
    
    # Search
    scores, indices = _INDEX.search(q_vec, top_k)
    
    contexts = []
    sources = []
    hits = []
    
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(_METADATA):
            continue
            
        chunk = _METADATA[idx]
        text = chunk["text"]
        src = chunk.get("source_id") or f"{chunk.get('doc_name', 'unknown')}#{chunk.get('chunk_id','?')}"
        
        contexts.append(text)
        sources.append(src)
        
        hit = chunk.copy()
        hit["score"] = float(score)
        hits.append(hit)
        
    return contexts, sources, hits

if __name__ == "__main__":
    # Test run
    test_q = "What is RAG?"
    print(f"Query: {test_q}")
    c, s, h = retrieve(test_q)
    for i, (ctx, src) in enumerate(zip(c, s)):
        print(f"Rank {i+1} [{src}]: {ctx[:100]}...")
