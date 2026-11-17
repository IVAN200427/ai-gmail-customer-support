from app.ai.agent import write_reply, write_reply_async
from app.ai.classifier import classify, classify_async
from app.ai.rules import CUSTOMER_STEPS, resolve_issue
from app.ai.schemas import AgentReply, Classification, Decision


def _escalate(
    classification: Classification, reason: str, proposed_action: str | None = None
) -> Decision:
    return Decision(
        action="escalate",
        section=classification.section,
        reply=classification.acknowledgement,
        reason=reason,
        proposed_action=proposed_action,
    )


def _early_escalation(classification: Classification, has_attachments: bool) -> Decision | None:
    if has_attachments:
        return _escalate(classification, "The email has attachments")
    if classification.needs_human:
        return _escalate(classification, classification.reason)
    return None


def _decide_from_answer(classification: Classification, answer: AgentReply) -> Decision:
    if answer.action == "escalate":
        return _escalate(classification, answer.reasoning)
    if not answer.enough_information:
        return _escalate(classification, "The knowledge base does not answer this")
    if answer.issue:
        step = resolve_issue(answer.issue)
        if step not in CUSTOMER_STEPS:
            return _escalate(classification, f"Next step: {step}", proposed_action=step)
    if answer.promises_an_action:
        return _escalate(classification, "The reply promises an action")

    return Decision(
        action="reply",
        section=classification.section,
        reply=answer.reply,
        reason=answer.reasoning,
    )


def decide(thread: str, has_attachments: bool = False) -> Decision:
    """Run the AI steps for one email thread. The models propose, this function decides."""
    classification = classify(thread)
    escalation = _early_escalation(classification, has_attachments)
    if escalation is not None:
        return escalation

    answer = write_reply(thread, classification.section)
    return _decide_from_answer(classification, answer)


async def decide_async(thread: str, has_attachments: bool = False) -> Decision:
    """Async variant used by pydantic_evals so every case runs on the same event loop."""
    classification = await classify_async(thread)
    escalation = _early_escalation(classification, has_attachments)
    if escalation is not None:
        return escalation

    answer = await write_reply_async(thread, classification.section)
    return _decide_from_answer(classification, answer)
