import time
import sys
from pathlib import Path
from typing import Dict, Any

# Add repo root
sys.path.append(str(Path(__file__).parent.parent))

from config import DEFAULT_LLM_PROVIDER, DEFAULT_GEMINI_MODEL
from llm.prompts import format_prompt
from utils.text import format_context
from utils.env_loader import get_gemini_key

class GenerationError(RuntimeError):
    """Raised when an LLM generation fails and no valid answer is produced."""
    pass

class LLMClient:
    def __init__(self, provider=None, final_mode: bool = False):
        self.gemini_key = get_gemini_key()
        self.provider = provider or ("gemini" if self.gemini_key else DEFAULT_LLM_PROVIDER)
        self.fallback_reason = None
        
        if self.provider == "gemini":
            if not self.gemini_key:
                self.fallback_reason = "GOOGLE_API_KEY is missing from environment."
                if final_mode:
                    raise GenerationError(self.fallback_reason)
                self.provider = "dryrun"; return
            try:
                from google import genai
                self.client = genai.Client(api_key=self.gemini_key)
            except ImportError:
                self.fallback_reason = "The 'google-genai' package is not installed in the python environment running this app."
                if final_mode:
                    raise GenerationError(self.fallback_reason)
                self.provider = "dryrun"
            except Exception as e:
                self.fallback_reason = f"Error initializing genai Client: {e}"
                if final_mode:
                    raise GenerationError(self.fallback_reason)
                self.provider = "dryrun"

    def generate(self, action: str, question: str, chunks_with_sources: list, temperature: float = 0.5, citation_enforced: bool = False):
        context_str = format_context(chunks_with_sources)
        
        if self.provider == "dryrun":
            return self._dryrun_generate(action, question, chunks_with_sources)
            
        elif self.provider == "gemini":
            return self._gemini_generate(action, question, context_str, temperature, citation_enforced=citation_enforced)
            
        raise GenerationError(f"Unknown LLM provider: {self.provider}")

    def _dryrun_generate(self, action: str, question: str, chunks: list) -> str:
        if not chunks:
            return "Not in context. (DryRun)"
            
        # Deterministic but seemingly intelligent stub
        # 1. Start with action tag
        # 2. Quote some text from top chunk
        # 3. Add fake citation
        top_chunk = chunks[0]
        text_snippet = top_chunk.get("text", "")[:100]
        source_id = top_chunk.get("source_id") or f"{top_chunk.get('doc_name','doc')}#{top_chunk.get('chunk_id','0')}"
        src = f"[source: {source_id}]"
        
        if action == "A0":
            if "not in context" in question.lower():
                return "Not in context."
            return f"{text_snippet}... {src}"
            
        return f"({action}) Based on the documents: {text_snippet}... This is a dry run answer {src}."

    def _gemini_generate(
        self,
        action: str,
        question: str,
        context_str: str,
        temperature: float,
        max_retries: int = 3,
        citation_enforced: bool = False,
    ) -> str:
        prompt = format_prompt(action, question, context_str, citation_enforced=citation_enforced)
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=DEFAULT_GEMINI_MODEL,
                    contents=prompt,
                    config={"temperature": temperature}
                )
                answer = (response.text or "").strip()
                if not answer:
                    raise GenerationError("Gemini returned an empty response.")
                return answer
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        raise GenerationError(f"Gemini generation failed after {max_retries} attempts: {last_error}")


def generate_all_candidates(client: LLMClient, question: str, chunks_with_sources: list) -> Dict[str, str]:
    # Define temps for actions
    temps = {
        "A0": 0.0,
        "A1": 0.2,
        "A2": 0.0,
        "A3": 0.2,
        "A4": 0.9
    }
    candidates = {}
    for action in temps:
        ans = client.generate(action, question, chunks_with_sources, temperature=temps[action])
        candidates[action] = ans
    return candidates
