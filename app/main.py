import logging

from fastapi import FastAPI, HTTPException

from .agent import run_agent
from .catalog import Catalog
from .config import CATALOG_PATH
from .models import ChatRequest, ChatResponse, HealthResponse, Recommendation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shl_api")

app = FastAPI(title="SHL Assessment Recommender", version="1.0.0")

# Load once at startup - catalog is read-only and small enough to keep
# fully in memory (a few thousand items at most).
try:
    catalog = Catalog(CATALOG_PATH)
    logger.info("Loaded catalog with %d items from %s", len(catalog), CATALOG_PATH)
except Exception:
    logger.exception("Failed to load catalog from %s", CATALOG_PATH)
    catalog = None


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if catalog is None:
        raise HTTPException(status_code=500, detail="Catalog failed to load on server startup.")
    if not req.messages:
        return ChatResponse(
            reply="Hi! Tell me about the role you're hiring for and I'll help you find the right SHL assessments.",
            recommendations=[],
            end_of_conversation=False,
        )

    result = run_agent(req.messages, catalog)
    return ChatResponse(
        reply=result["reply"],
        recommendations=[Recommendation(**r) for r in result["recommendations"]],
        end_of_conversation=result["end_of_conversation"],
    )
