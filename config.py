from pathlib import Path
from utils.env_loader import load_environment

# Load environment variables robustly from .env
env_loaded, env_path = load_environment()

# Paths
REPO_ROOT = Path(__file__).parent.absolute()
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
CHUNKS_JSONL = OUTPUTS_DIR / "chunks.jsonl"
FAISS_INDEX = OUTPUTS_DIR / "faiss.index"
METADATA_JSON = OUTPUTS_DIR / "chunk_metadata.json"
BANDIT_STATE_PATH = OUTPUTS_DIR / "bandit_state.json"

# Ensure outputs dir exists
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "sample_docs").mkdir(parents=True, exist_ok=True)

# Settings
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
DEFAULT_SENT_SIM_THRESHOLD = 0.50
SEMANTIC_SCORER_LEGACY_V1 = "semantic_proxy_legacy_v1"
SEMANTIC_SCORER_CLEAN_V2 = "semantic_proxy_clean_v2"
DEFAULT_SEMANTIC_SCORER_VERSION = SEMANTIC_SCORER_CLEAN_V2
NLI_MODEL_NAME = "facebook/bart-large-mnli"
NLI_SUPPORT_THRESHOLD = 0.50
ABSTENTION_TEXT = "Not in context"
ACTIONS = ["A0", "A1", "A2", "A3", "A4"]
DEFAULT_EPSILON = 0.2
EPSILON_DECAY = 0.99
MIN_EPSILON = 0.01
TRAIN_EPISODES = 200
MASTER_SEED = 42
MULTI_SEEDS = [11, 22, 33, 44, 55]
ACTION_TEMPERATURES = {"A0": 0.0, "A1": 0.2, "A2": 0.0, "A3": 0.2, "A4": 0.9}
CONTROLLED_TEMPERATURE = 0.2
CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 40
FINAL_EXPERIMENT_MODE = True

# LLM Defaults
DEFAULT_LLM_PROVIDER = "dryrun"  # final-mode workflows require explicit Gemini validation
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
