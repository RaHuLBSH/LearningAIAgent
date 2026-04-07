## Agentic AI Learning Platform (POC)

Simple end-to-end proof of concept using **FastAPI**, **LangGraph**, and a vanilla **HTML/JavaScript** chat UI.

The agent maintains state across turns, loops through `teach → quiz → evaluate → decision`, and adapts its explanations based on your performance.

### Project structure

- `main.py` – FastAPI app (`/` for UI, `/chat` for chat API)
- `agent/state.py` – LangGraph state schema and initial state helper
- `agent/nodes.py` – Intent / planner / teach / quiz / evaluate / decision nodes, plus LLM wrapper
- `agent/graph.py` – LangGraph graph construction and in-memory `SessionManager`
- `templates/index.html` – Minimal but modern chat UI (Jinja2 template)
- `static/script.js` – Frontend chat logic (no frameworks)
- `requirements.txt` – Python dependencies

### Running locally

1. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # on Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set your Groq API key (recommended)**

   ```bash
   export GROQ_API_KEY="your_key_here"  # PowerShell: $env:GROQ_API_KEY="your_key_here"
   ```

   Optional model override:

   ```bash
   export GROQ_MODEL="llama-3.1-8b-instant"
   ```

   If no key is set, the backend falls back to a simple mock so the POC still runs.

   **Alternative**: you can also use OpenAI by setting `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`).

4. **Start the FastAPI server**

   ```bash
   uvicorn main:app --reload
   ```

5. **Open the UI**

   Visit `http://127.0.0.1:8000` in your browser.

### Example usage

Type:

- `Teach me Dynamic Programming`
- Answer the quiz questions.
- Try responding with `Explain simpler` if you are stuck – the agent will adapt explanations and may mark weak topics.

