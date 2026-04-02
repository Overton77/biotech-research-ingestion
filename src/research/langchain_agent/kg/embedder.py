"""
Batch embedding for KG node searchTexts.

Uses OpenAI text-embedding-3-small (1 536 dims) by default.
All texts in a session are batched into a single API call before writing
to Neo4j to minimise round-trips and cost.
"""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from src.research.langchain_agent.kg.constants import DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_DIMENSIONS

def build_embedder(model: str = DEFAULT_EMBEDDING_MODEL) -> OpenAIEmbeddings:
    """Return an OpenAIEmbeddings instance.  Build once; reuse across writes."""
    return OpenAIEmbeddings(model=model)


async def embed_batch(
    texts: list[str],
    embedder: OpenAIEmbeddings | None = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[list[float]]:
    """
    Embed a list of texts in a single async API call.

    Args:
        texts:    Non-empty list of searchText strings.
        embedder: Pre-built OpenAIEmbeddings instance (optional; creates one if absent).
        model:    Model name (only used if embedder is not provided).

    Returns:
        List of embedding vectors in the same order as *texts*.
    """
    if not texts:
        return []

    emb = embedder or build_embedder(model)
    return await emb.aembed_documents(texts)
