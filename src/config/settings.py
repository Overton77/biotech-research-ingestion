"""Application settings with fail-fast validation at startup."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated environment configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # MongoDB
    MONGODB_URI: str = Field(..., description="MongoDB connection URI")
    MONGODB_DB: str = Field(default="biotech_research", description="MongoDB database name")

    # PostgreSQL (LangGraph checkpointer / store)
    POSTGRES_URL: str | None = Field(
        default=None,
        alias="POSTGRES_URI",
        description="PostgreSQL URL for LangGraph. Optional if LANGGRAPH_CHECKPOINTER=memory.",
    ) 

    COORDINATOR_POSTGRES_URI: str | None = Field(
        default=None,
        alias="COORDINATOR_POSTGRES_URI",
        description="PostgreSQL URL for coordinator agent. Optional if LANGGRAPH_CHECKPOINTER=memory.",
    )
    DEEP_AGENTS_POSTGRES_URI: str | None = Field(
        default=None,
        alias="DEEP_AGENTS_POSTGRES_URI",
    ) 

    # Redis (Socket.IO adapter, future: interrupt state)
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis URL")

    # AWS S3
    AWS_ACCESS_KEY_ID: str = Field(default="", description="AWS access key")
    AWS_SECRET_ACCESS_KEY: str = Field(default="", description="AWS secret key")
    AWS_S3_BUCKET: str = Field(default="", description="S3 bucket for artifacts")
    AWS_REGION: str = Field(default="us-east-1", description="AWS region")

    # LLM / agents
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key for coordinator agent")
    TAVILY_API_KEY: str = Field(default="", description="Tavily API key for web search")
    LANGSMITH_API_KEY: str = Field(default="", description="LangSmith API key")
    LANGSMITH_TRACING: bool = Field(default=True, description="Enable LangSmith tracing")
    LANGSMITH_PROJECT: str = Field(default="biotech-research", description="LangSmith project name")

    # LangGraph checkpointer: "memory" (dev) or "postgres" (prod)
    LANGGRAPH_CHECKPOINTER: Literal["memory", "postgres"] = Field(
        default="memory",
        description="Checkpointer backend",
    )

    # CORS
    WEB_ORIGIN: str = Field(default="http://localhost:3000", description="Allowed frontend origin for CORS")


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
