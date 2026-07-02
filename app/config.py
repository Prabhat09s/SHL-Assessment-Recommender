import os
from dotenv import load_dotenv

load_dotenv()

# --- Required ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# --- Optional, sensible defaults ---
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CATALOG_PATH = os.getenv("CATALOG_PATH", "data/catalog.json")
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "18"))
MAX_RECOMMENDATIONS = int(os.getenv("MAX_RECOMMENDATIONS", "10"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "25"))
