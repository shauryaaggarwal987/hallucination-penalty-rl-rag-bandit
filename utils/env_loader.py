import os
from pathlib import Path

def get_repo_root() -> Path:
    """Returns the absolute path to the repository root."""
    # This file is assumed to be at `utils/env_loader.py`
    # The parent of `utils/` is the repo root.
    return Path(__file__).parent.parent.absolute()

def load_environment() -> tuple[bool, str]:
    """
    Finds .env at repo root and conditionally loads it.
    Returns:
        bool: True if .env file exists and was loaded
        str: Absolute path to the checked .env location
    """
    env_path = get_repo_root() / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=env_path)
        except ImportError:
            # Minimal dependency-free fallback; never log values.
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line=line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key,value=line.split("=",1)
                os.environ.setdefault(key.strip(),value.strip().strip("\"'"))
        return True, str(env_path)
    return False, str(env_path)

def get_gemini_key() -> str:
    """
    Returns the Google API key from the environment.
    Exposed centrally to avoid mismatched environment variable reads.
    """
    return os.getenv("GOOGLE_API_KEY", "")

def get_redacted_key() -> str:
    """
    Returns a safe, redacted version of the API key for logging
    without exposing the actual secret value.
    """
    key = get_gemini_key()
    if not key:
        return "Not Set"
    
    if len(key) <= 8:
        return "*" * len(key)
        
    return f"{key[:4]}...{key[-4:]} (Len: {len(key)})"
