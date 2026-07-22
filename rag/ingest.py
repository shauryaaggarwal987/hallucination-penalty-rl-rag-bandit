import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from config import DATA_DIR, CHUNKS_JSONL
from utils.io import list_documents, read_text_file, write_jsonl
from utils.text import extract_text_from_pdf, normalize_text, chunk_text

SAMPLE_DOCS = {
    "rl_basics.txt": """
Reinforcement Learning (RL) is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. 
Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning. 
The standard model for RL is the Markov Decision Process (MDP).
An MDP consists of a set of states, a set of actions, a transition function, and a reward function.
The goal of the agent is to learn a policy, which is a mapping from states to probabilities of selecting each possible action.
Q-learning is a model-free reinforcement learning algorithm to learn the value of an action in a particular state.
The epsilon-greedy algorithm is a simple method to balance exploration and exploitation by choosing a random action with probability epsilon and the best known action otherwise.
Multi-armed bandits are a simpler form of RL directly modeling the exploration-exploitation trade-off without state transitions.
""",
    "rag_basics.txt": """
Retrieval-Augmented Generation (RAG) is a technique for enhancing the accuracy and reliability of generative AI models with facts fetched from external sources.
In a RAG system, the user's query is first used to retrieve relevant documents or chunks of text from a knowledge base.
These retrieved chunks are then passed to the Large Language Model (LLM) along with the original query as context.
The LLM uses this context to generate a more accurate and grounded response.
This approach reduces hallucinations, where the model invents incorrect information.
Vector databases are commonly used in RAG to store embeddings of documents for efficient similarity search.
Embeddings are vector representations of text where semantically similar texts are close in vector space.
FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors.
""",
    "faiss_basics.txt": """
FAISS (Facebook AI Similarity Search) is a library that allows developers to quickly search for embeddings of multimedia documents that are similar to each other.
It solves the problem of nearest neighbor search in high-dimensional spaces.
IndexFlatIP is a simple index type in FAISS that performs exact search using inner product (dot product) distance.
If vectors are normalized, inner product is equivalent to cosine similarity.
FAISS is written in C++ with complete wrappers for Python.
It supports search on both CPUs and GPUs.
For large datasets, FAISS offers approximate search methods that trade off some accuracy for speed.
"""
}

def create_sample_docs():
    sample_dir = DATA_DIR / "sample_docs"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Only create if dir is empty
    if not any(sample_dir.iterdir()):
        print("Creating sample documents...")
        for name, content in SAMPLE_DOCS.items():
            path = sample_dir / name
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content.strip())
            print(f"Created {path}")

def main():
    # 1. Ensure sample docs
    create_sample_docs()
    
    # 2. List docs
    docs = list_documents(DATA_DIR)
    print(f"Found {len(docs)} documents in {DATA_DIR}")
    
    all_chunks = []
    
    # 3. Process each
    for doc_path in docs:
        print(f"Processing {doc_path.name}...")
        if doc_path.suffix.lower() == ".pdf":
            raw_text = extract_text_from_pdf(str(doc_path))
        else:
            raw_text = read_text_file(doc_path)
            
        text = normalize_text(raw_text)
        if not text:
            print(f"Skipping empty doc {doc_path.name}")
            continue
            
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "doc_name": doc_path.name,
                "doc_path": str(doc_path),
                "chunk_id": i,
                "text": chunk
            })
            
    # 4. Save
    print(f"Writing {len(all_chunks)} chunks to {CHUNKS_JSONL}")
    write_jsonl(CHUNKS_JSONL, all_chunks)
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
