from typing import List, Dict

BASE_RULES = """
- Use only the provided context.
- If the answer is not in the context, say "Not in context".
- Always include citations in the format [source: doc_name#chunk_id].
- Never invent sources.
"""

def format_prompt(action: str, question: str, context_str: str, citation_enforced: bool = False) -> str:
    """
    Returns the user content string for the LLM.
    """
    base = f"Context:\n{context_str}\n\nQuestion: {question}\n"
    if citation_enforced:
        base += "Every substantive factual unit must include at least one valid [source: doc_name#chunk_id] citation.\n"
    
    if action == "A0":
        # Strict
        return f"{BASE_RULES}\nSTRICT MODE: Use ONLY the context. If the answer is not explicitly supported by the context, you MUST say 'Not in context'. Do not use outside knowledge.\n\n{base}"
    
    elif action == "A1":
        # Normal Helpful
        return f"{BASE_RULES}\nProvide a helpful, clear answer based on the context. Include exactly 2 citations if possible.\n\n{base}"
        
    elif action == "A2":
        # Concise
        return f"{BASE_RULES}\nAnswer concisely in max 3 lines. Include citations.\n\n{base}"
        
    elif action == "A3":
        # Step-by-step
        return f"{BASE_RULES}\nThink step-by-step. Break down the answer into numbered steps. Include citations.\n\n{base}"
        
    elif action == "A4":
        # Creative
        return f"{BASE_RULES}\nBe creative and engaging in your wording. You can speculate slightly if grounded in context, but still provide citations. Use a more conversational tone.\n\n{base}"
        
    else:
        # Default fallback
        return f"{BASE_RULES}\nAnswer the question based on context.\n\n{base}"
