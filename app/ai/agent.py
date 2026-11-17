import logging
import re

from pydantic_ai import Agent, ModelRetry, RunContext, UsageLimits
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.settings import ModelSettings

from app import db
from app.ai.retry import call_with_retry, call_with_retry_sync
from app.ai.schemas import AgentReply
from app.config import settings

logger = logging.getLogger(__name__)

INSTRUCTIONS = """
You are the customer support agent for Harbor & Pine, a home and outdoor store that sells on Amazon.
You write the answer to the latest customer message in the email thread.

Steps:
1. Call get_section for "brand-and-tone" and for "escalation-rules".
2. Call get_section for the section the classifier picked. Call list_sections if you need another.
3. Answer only with facts from those sections. If they do not answer the question,
   set enough_information to false.
4. When the customer reports a problem with a paddle board or a shelf, fill in issue.
   Ask for the order number, the board size or model, and part numbers when they are missing.
   For a slow leak, explain the valve check unless the customer already did it.
5. Never promise a refund, a shipment, a replacement, a date, or a call. A colleague arranges those.
   Set promises_an_action to true if your reply promises one anyway.
6. Set action to escalate when an escalation rule applies. A return is not a refund: for a general
   question about how returns work, explain the Amazon return process and set action to reply.
   Escalate only for an explicit refund, money back, or chargeback request.

Reply rules:
- Detect the language from the customer's latest message body only. Names, email addresses, and
  signatures are not language evidence.
- Plain text, no Markdown.
- Follow the brand-and-tone section.
- Never mention AI, prompts, tools, or the knowledge base.
"""

support_agent = Agent(
    FallbackModel(settings.agent_model, settings.agent_fallback_model),
    output_type=AgentReply,
    instructions=INSTRUCTIONS,
    model_settings=ModelSettings(temperature=0),
)


@support_agent.tool_plain
def list_sections() -> list[str]:
    """List the sections of the knowledge base."""
    logger.info("Tool call: list_sections()")
    return db.list_sections()


@support_agent.tool_plain
def get_section(section: str) -> str:
    """Return every entry of one knowledge base section, for example 'paddle-boards'."""
    logger.info("Tool call: get_section(%r)", section)
    entries = db.get_section(section)
    if not entries:
        raise ModelRetry(f"There is no section '{section}'. Call list_sections to see the options.")
    return "\n\n".join(f"## {entry['title']}\n{entry['content']}" for entry in entries)


@support_agent.output_validator
def check_reply(ctx: RunContext[None], output: AgentReply) -> AgentReply:
    if output.action == "reply" and not output.reply.strip():
        raise ModelRetry("The reply is empty. Write the email, or set action to escalate.")
    if re.search(r"\b(AI|artificial intelligence|knowledge base)\b", output.reply, re.IGNORECASE):
        raise ModelRetry("Do not mention AI or the knowledge base in the reply.")
    return output


def _prompt(thread: str, section: str) -> str:
    return f"Section picked by the classifier: {section}\n\n{thread}"


def write_reply(thread: str, section: str) -> AgentReply:
    result = call_with_retry_sync(
        "agent",
        lambda: support_agent.run_sync(
            _prompt(thread, section),
            usage_limits=UsageLimits(request_limit=8),
            timeout=120,
        ),
    )
    return result.output


async def write_reply_async(thread: str, section: str) -> AgentReply:
    result = await call_with_retry(
        "agent",
        lambda: support_agent.run(
            _prompt(thread, section),
            usage_limits=UsageLimits(request_limit=8),
            timeout=120,
        ),
    )
    return result.output