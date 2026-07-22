import re
from typing import List, Dict

def extract_text_from_pdf(path: str) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(path)
        text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text.append(t)
        return "\n".join(text)
    except Exception as e:
        print(f"Error reading PDF {path}: {e}")
        return ""

def normalize_text(text: str) -> str:
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def sentence_split(text: str) -> List[str]:
    # Simple regex split on [.?!] boundaries
    parts = re.split(r'([.?!])', text)
    sentences = []
    # Reattach punctuation
    for i in range(0, len(parts) - 1, 2):
        s = parts[i] + parts[i+1]
        s = s.strip()
        if s:
            sentences.append(s)
    # Handle last part if no punctuation
    if len(parts) % 2 == 1:
        s = parts[-1].strip()
        if s:
            sentences.append(s)
    return sentences

def chunk_text(text: str, max_words=220, overlap_words=40) -> List[str]:
    words = text.split()
    if not words:
        return []
        
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        
        # Move forward by stride
        # stride = max_words - overlap
        # start += stride
        # if end >= len(words): break
        
        if end >= len(words):
            break
            
        start += (max_words - overlap_words)
        
    return chunks

def format_context(chunks_with_sources: List[Dict]) -> str:
    # Expects list of dicts with 'text', 'doc_name', 'chunk_id'
    out = []
    for c in chunks_with_sources:
        source_id = c.get("source_id") or f"{c.get('doc_name', 'unknown')}#{c.get('chunk_id', '?')}"
        header = f"[source: {source_id}]"
        out.append(f"{header}\n{c.get('text', '')}")
    return "\n\n".join(out)

def simple_token_estimate(text: str) -> int:
    return int(len(text.split()) * 1.3)
