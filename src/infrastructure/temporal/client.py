"""Temporal client — connects to the Temporal server.

Reads TEMPORAL_HOST and TEMPORAL_NAMESPACE from environment variables,
falling back to localhost:7233 / default for local development.
"""

from __future__ import annotations

import os

from temporalio.client import Client


async def get_temporal_client() -> Client:
    host = os.getenv("TEMPORAL_HOST", "localhost:7233")
    namespace = os.getenv("TEMPORAL_NAMESPACE", "default")
    return await Client.connect(host, namespace=namespace)
