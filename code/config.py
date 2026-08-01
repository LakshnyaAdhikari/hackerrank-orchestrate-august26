import os

# Try loading .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Look for .env in current directory or parent directories
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()
except ImportError:
    pass

class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
    
    # API Keys from Environment (never hardcoded)
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
    
    # Processing Options
    USE_LLM_REFINEMENT = os.environ.get('USE_LLM_REFINEMENT', 'false').lower() == 'true'
    EVIDENCE_SIMILARITY_THRESHOLD = 0.40
    
    # Thresholds & Limits
    MIN_CONFIDENCE = 0.60
    MAX_CONFIDENCE = 0.95
    DEFAULT_CONFIDENCE = 0.82

    @classmethod
    def get_dataset_path(cls, filename: str) -> str:
        return os.path.join(cls.DATASET_DIR, filename)

if __name__ == "__main__":
    print("Config loaded successfully. Dataset path:", Config.DATASET_DIR)
    print("Gemini Key set:", bool(Config.GEMINI_API_KEY))
