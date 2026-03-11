from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import aioboto3
from botocore.config import Config

AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
AWS_PROFILE = os.getenv("AWS_PROFILE")
BIOTECH_RESEARCH_RUNS_BUCKET = os.getenv("BIOTECH_RESEARCH_RUNS_BUCKET")


def get_runs_bucket_name() -> str:
    bucket = BIOTECH_RESEARCH_RUNS_BUCKET
    if not bucket:
        raise ValueError("BIOTECH_RESEARCH_RUNS_BUCKET is not configured")
    return bucket


@asynccontextmanager
async def get_s3_client() -> AsyncIterator[Any]:
    session_kwargs: dict[str, Any] = {"region_name": AWS_REGION}
    if AWS_PROFILE:
        session_kwargs["profile_name"] = AWS_PROFILE

    session = aioboto3.Session(**session_kwargs)

    config = Config(
        retries={"max_attempts": 10, "mode": "standard"},
        connect_timeout=10,
        read_timeout=60,
    )

    async with session.client("s3", config=config) as client:
        yield client


class AsyncS3Client:
    """
    Small reusable async S3 client for JSON, text, bytes, and object listing.
    """

    def __init__(self, bucket: str | None = None) -> None:
        self.bucket = bucket or get_runs_bucket_name()

    async def put_json(
        self,
        key: str,
        data: dict[str, Any],
        *,
        metadata: dict[str, str] | None = None,
        content_type: str = "application/json",
    ) -> str:
        body = json.dumps(data, indent=2, default=str, ensure_ascii=False).encode("utf-8")
        async with get_s3_client() as s3:
            await s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=body,
                ContentType=content_type,
                Metadata=metadata or {},
            )
        return f"s3://{self.bucket}/{key}"

    async def put_text(
        self,
        key: str,
        text: str,
        *,
        metadata: dict[str, str] | None = None,
        content_type: str = "text/plain; charset=utf-8",
    ) -> str:
        async with get_s3_client() as s3:
            await s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=text.encode("utf-8"),
                ContentType=content_type,
                Metadata=metadata or {},
            )
        return f"s3://{self.bucket}/{key}"

    async def put_bytes(
        self,
        key: str,
        body: bytes,
        *,
        metadata: dict[str, str] | None = None,
        content_type: str = "application/octet-stream",
    ) -> str:
        async with get_s3_client() as s3:
            await s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=body,
                ContentType=content_type,
                Metadata=metadata or {},
            )
        return f"s3://{self.bucket}/{key}"

    async def get_json(self, key: str) -> dict[str, Any]:
        async with get_s3_client() as s3:
            response = await s3.get_object(Bucket=self.bucket, Key=key)
            body = await response["Body"].read()
        return json.loads(body.decode("utf-8"))

    async def get_text(self, key: str) -> str:
        async with get_s3_client() as s3:
            response = await s3.get_object(Bucket=self.bucket, Key=key)
            body = await response["Body"].read()
        return body.decode("utf-8")

    async def list_objects(self, prefix: str) -> list[dict[str, Any]]:
        async with get_s3_client() as s3:
            paginator = s3.get_paginator("list_objects_v2")
            results: list[dict[str, Any]] = []

            async for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                contents = page.get("Contents", [])
                results.extend(contents)

        return results

    async def delete_object(self, key: str) -> None:
        async with get_s3_client() as s3:
            await s3.delete_object(Bucket=self.bucket, Key=key)

    async def head_object(self, key: str) -> dict[str, Any]:
        async with get_s3_client() as s3:
            return await s3.head_object(Bucket=self.bucket, Key=key)