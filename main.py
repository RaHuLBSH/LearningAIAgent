import logging
import uuid
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from agent.graph import SessionManager
from agent.nodes import compute_progress

# Load environment variables from `.env` (so OPENAI_API_KEY works without manual export)
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agentic-learning")

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Agentic AI Learning Platform - POC")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

session_manager = SessionManager()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    user_id: str | None = None


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Serve the basic chat UI.
    """
    # Starlette's TemplateResponse signature differs across versions.
    # Using the (request, name, context) form works on modern Starlette/FastAPI.
    return templates.TemplateResponse(request, "index.html", {})


@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    """
    Main chat endpoint for the learning agent.
    """
    session_id = payload.session_id or str(uuid.uuid4())
    user_id = payload.user_id or session_id  # simple mapping for POC
    message = payload.message.strip()

    logger.info("Incoming message for session %s: %s", session_id, message)

    result = session_manager.handle_message(session_id=session_id, user_id=user_id, message=message)
    state = result["state"]
    reply = result["reply"]

    progress = compute_progress(state)

    response: Dict[str, Any] = {
        "reply": reply,
        "events": result.get("events", []),
        "topic_progress": result.get("topic_progress", []),
        "session_id": session_id,
        "progress": progress,
        "current_step": result["debug"]["current_step"],
        "mode": result["debug"].get("mode"),
        "card_index": result["debug"].get("card_index"),
        "quiz_index": result["debug"].get("quiz_index"),
    }

    return JSONResponse(response)


if __name__ == "__main__":
    # for local testing: `python main.py`
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

