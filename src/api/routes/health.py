"""Health check endpoint — MongoDB, Redis, Postgres, S3."""

import logging
from typing import Any

from fastapi import APIRouter, status

from src.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


async def _check_mongodb() -> str:
    from pymongo import AsyncMongoClient

    settings = get_settings()
    client = AsyncMongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=3000)
    try:
        await client.admin.command("ping")
        return "ok"
    except Exception as e:
        logger.warning("MongoDB health check failed: %s", e)
        return f"error: {e!s}"
    finally:
        await client.aclose()


async def _check_redis() -> str:
    import redis.asyncio as redis

    settings = get_settings()
    try:
        r = redis.from_url(settings.REDIS_URL)
        await r.ping()
        await r.aclose()
        return "ok"
    except Exception as e:
        logger.warning("Redis health check failed: %s", e)
        return f"error: {e!s}"


async def _check_postgres() -> str:
    settings = get_settings()
    if not settings.POSTGRES_URL:
        return "skipped"
    try:
        import psycopg
        url = settings.POSTGRES_URL
        if url.startswith("postgresql+psycopg://"):
            url = url.replace("postgresql+psycopg://", "postgresql://", 1)
        async with await psycopg.AsyncConnection.connect(url) as conn:
            await conn.execute("SELECT 1")
        return "ok"
    except Exception as e:
        logger.warning("Postgres health check failed: %s", e)
        return f"error: {e!s}"


async def _check_s3() -> str:
    settings = get_settings()
    if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_S3_BUCKET:
        return "skipped"
    try:
        import boto3
        from botocore.exceptions import ClientError

        client = boto3.client(
            "s3",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
        client.head_bucket(Bucket=settings.AWS_S3_BUCKET)
        return "ok"
    except ClientError as e:
        logger.warning("S3 health check failed: %s", e)
        return f"error: {e!s}"
    except Exception as e:
        logger.warning("S3 health check failed: %s", e)
        return f"error: {e!s}"


@router.get("/health", status_code=status.HTTP_200_OK)
async def health() -> dict[str, Any]:
    """Return status of MongoDB, Redis, Postgres, and S3."""
    checks = {
        "mongodb": await _check_mongodb(),
        "redis": await _check_redis(),
        "postgres": await _check_postgres(),
        "s3": await _check_s3(),
    }
    all_ok = all(c == "ok" or c == "skipped" for c in checks.values())
    return {
        "status": "ok" if all_ok else "degraded",
        "checks": checks,
        "version": "0.1.0",
    }
