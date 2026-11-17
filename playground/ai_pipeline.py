# %%
"""Explore the AI pipeline without reading or sending email through Gmail.

The pipeline still calls the configured AI provider and reads the knowledge base
from the configured database.
"""

from pathlib import Path
import sys

import nest_asyncio


_starts = [Path.cwd().resolve()]
_current_file = globals().get("__file__")
if _current_file:
    _starts.insert(0, Path(_current_file).resolve().parent)

PROJECT_ROOT = next(
    (
        candidate
        for start in _starts
        for candidate in (start, *start.parents)
        if (candidate / "pyproject.toml").is_file() and (candidate / "app").is_dir()
    ),
    None,
)
if PROJECT_ROOT is None:
    raise RuntimeError(
        "Could not find the project root. Open this file from the project or set "
        "the Jupyter working directory to the project folder."
    )
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

nest_asyncio.apply()

from app.ai.decide import run_pipeline

# %%
# Replace this with the customer question you want to test.
QUESTION = """I want my money back, this is ridiculous."""

# %%
print("Question:")
print(QUESTION)

trace = run_pipeline(QUESTION.strip())

print("\nClassification:")
print(trace.classification.model_dump_json(indent=2))

print("\nAgentReply (null if an earlier stage escalated):")
print(
    trace.agent_reply.model_dump_json(indent=2)
    if trace.agent_reply is not None
    else "null"
)

print("\nFinal Decision:")
print(trace.decision.model_dump_json(indent=2))
