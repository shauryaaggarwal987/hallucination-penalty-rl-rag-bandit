import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from utils.env_loader import load_environment, get_gemini_key, get_redacted_key
from llm.generate import LLMClient

def main():
    print("--- Gemini Integration Smoke Test ---")
    
    # 1. Load environment
    env_loaded, env_path = load_environment()
    print(f".env loaded: {env_loaded} from {env_path}")
    
    # 2. Check key
    key = get_gemini_key()
    print(f"API Key present: {bool(key)} (Redacted: {get_redacted_key()})")
    
    if not key:
        print("FAIL: No generic GOOGLE_API_KEY found in environment.")
        sys.exit(1)
        
    # 3. Test LLMClient initialization
    try:
        client = LLMClient(provider="gemini")
        print(f"LLMClient initialized successfully. Active provider: {client.provider}")
    except Exception as e:
        print(f"FAIL: LLMClient initialization failed: {e}")
        sys.exit(1)
        
    if client.provider == "dryrun":
        print("FAIL: LLMClient initialized but fell back to 'dryrun'. Check google-genai installation or API key validity.")
        sys.exit(1)
        
    # 4. Generate a simple text
    print("\nAttempting generation call to Gemini...")
    question = "Say hello in one short sentence."
    # We pass an empty list of chunks, action A4 (Creative) so it just says hello
    try:
        ans = client.generate("A4", question, [], temperature=0.7)
        if ans.startswith("Gemini Error:"):
            print(f"FAIL: Generation returned an error internally: {ans}")
            sys.exit(1)
        print(f"\nResponse:\n{ans}\n")
        print("SUCCESS: Smoke test passed!")
    except Exception as e:
        print(f"FAIL: Generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
