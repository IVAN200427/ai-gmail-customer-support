# AI pipeline playground

Use [`playground/ai_pipeline.py`](./playground/ai_pipeline.py) to test a customer
question through the AI pipeline without reading, modifying, or sending Gmail
messages.

1. Install the project and its Jupyter kernel dependency with `uv sync --group dev`.
2. In VS Code, open `playground/ai_pipeline.py`, select the project `.venv` as the
   Python/Jupyter kernel, and run the `# %%` cells.
3. Edit `QUESTION` in the second cell and run the final cell.

The output shows the `Classification`, optional `AgentReply` (including its
`Issue`, when present), and final `Decision` Pydantic models. The playground
uses the app settings from `.env` and needs an AI provider API key plus a
configured database with the migrated and seeded knowledge base. The current
settings also require the Google OAuth values at import time, but the
playground does not use Gmail or write ticket/event records.