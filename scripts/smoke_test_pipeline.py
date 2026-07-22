import sys
from pathlib import Path

# Add repo root
sys.path.append(str(Path(__file__).parent.parent))

from config import DEFAULT_EPSILON, ACTIONS
from rag.retrieve import retrieve
from llm.generate import LLMClient
from scoring.reward import score_answer
from rl.bandit import EpsilonGreedyBandit

def test_pipeline():
    print("--- End-to-End Pipeline Smoke Test ---")
    
    query = "What is the primary role of a vector database in RAG?"
    print(f"Query: {query}")
    
    print("\n1. Testing Retrieval...")
    contexts, sources, hits = retrieve(query, top_k=2)
    print(f"Retrieved {len(hits)} chunks.")
    if not hits:
        print("WARNING: No chunks retrieved! Is the index built?")
        
    for i, src in enumerate(sources):
        print(f"  Source {i+1}: {src}")
        
    print("\n2. Testing Generation (Action A1)...")
    client = LLMClient()
    print(f"Provider active: {client.provider}")
    
    ans = client.generate("A1", query, hits, temperature=0.1)
    print(f"Answer Preview: {ans[:200]}...")
    
    print("\n3. Testing Scoring...")
    all_context = "\n".join(contexts)
    scores = score_answer(ans, all_context)
    print(f"Groundedness: {scores['groundedness']:.2f}, Penalty: {scores['hallucination_penalty']:.2f}")
    print(f"Final Reward: {scores['reward']:.2f}")
    
    print("\n4. Testing Bandit Update...")
    bandit = EpsilonGreedyBandit(ACTIONS, epsilon=DEFAULT_EPSILON)
    bandit.load()
    old_q = bandit.q_values["A1"]
    bandit.update("A1", scores["reward"])
    new_q = bandit.q_values["A1"]
    print(f"Q-value for A1 shifted from {old_q:.3f} to {new_q:.3f}")
    print("Pipeline check complete!")

if __name__ == "__main__":
    test_pipeline()
