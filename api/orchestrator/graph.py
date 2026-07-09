"""
LangGraph orchestration for multi-agent call analytics.

Graph topology (parallel pipeline):

  START → classifier → quality_agent ──┐
                    ↘ compliance_agent ─┤→ summarizer → END

quality_agent and compliance_agent run in parallel after classifier.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import AsyncOpenAI

from agents.classifier import ClassifierAgent
from agents.compliance_agent import ComplianceAgent
from agents.quality_agent import QualityAgent
from agents.summarizer import SummarizerAgent
from config import get_settings
from utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class AnalysisState(TypedDict):
    # Inputs (set before graph execution)
    transcript_segments: list[dict]
    full_text: str
    # Outputs (filled by agents)
    classification: dict
    quality_score: dict
    compliance: dict
    summary: str
    action_items: list


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    """Build and compile the LangGraph analysis pipeline."""
    settings = get_settings()

    # Without a key the OpenAI client sends an empty "Bearer " header and httpx
    # dies with an opaque LocalProtocolError. Fail fast with an actionable message.
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set — the analytics agents cannot reach the LLM. "
            "Set OPENAI_API_KEY (plus OPENAI_API_BASE_URL and LLM_MODEL) in the API environment."
        )

    # OpenRouter requires Referer + Title headers; harmless for other providers
    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base_url,
        default_headers={
            "HTTP-Referer": "https://github.com/mtbank-ai-engineer",
            "X-Title": "MTBank Speech Analytics",
        },
    )
    model = settings.llm_model

    classifier = ClassifierAgent(client, model)
    quality = QualityAgent(client, model)
    compliance = ComplianceAgent(client, model)
    summarizer = SummarizerAgent(client, model)

    async def run_classifier(state: AnalysisState) -> dict:
        logger.info("Node: classifier")
        return await classifier.run(state)

    async def run_quality(state: AnalysisState) -> dict:
        logger.info("Node: quality_agent")
        return await quality.run(state)

    async def run_compliance(state: AnalysisState) -> dict:
        logger.info("Node: compliance_agent")
        return await compliance.run(state)

    async def run_summarizer(state: AnalysisState) -> dict:
        logger.info("Node: summarizer")
        return await summarizer.run(state)

    builder = StateGraph(AnalysisState)

    builder.add_node("classifier", run_classifier)
    builder.add_node("quality_agent", run_quality)
    builder.add_node("compliance_agent", run_compliance)
    builder.add_node("summarizer", run_summarizer)

    # classifier first, then quality + compliance in parallel, then summarizer
    builder.add_edge(START, "classifier")
    builder.add_edge("classifier", "quality_agent")
    builder.add_edge("classifier", "compliance_agent")
    builder.add_edge("quality_agent", "summarizer")
    builder.add_edge("compliance_agent", "summarizer")
    builder.add_edge("summarizer", END)

    graph = builder.compile()
    logger.info("LangGraph analysis pipeline compiled")
    return graph


_graph: Any = None


def get_graph() -> Any:
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


async def run_analysis(transcript_segments: list[dict], full_text: str) -> dict:
    """Run the full agent pipeline and return the combined state."""
    graph = get_graph()

    initial_state: AnalysisState = {
        "transcript_segments": transcript_segments,
        "full_text": full_text,
        "classification": {},
        "quality_score": {},
        "compliance": {},
        "summary": "",
        "action_items": [],
    }

    logger.info("Starting analysis graph", segments=len(transcript_segments))
    final_state = await graph.ainvoke(initial_state)
    logger.info("Analysis graph completed")
    return final_state
