from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from dotenv import load_dotenv
from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j.exceptions import Neo4jError


load_dotenv()


@dataclass(frozen=True)
class Neo4jAuraSettings:
    uri: str
    username: str
    password: str
    database: str
    instance_id: str | None = None
    instance_name: str | None = None
    query_api: str | None = None

    @classmethod
    def from_env(cls) -> "Neo4jAuraSettings":
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_AURA_USERNAME")
        password = os.getenv("NEO4J_AURA_PASSWORD")
        database = os.getenv("NEO4J_AURA_DATABASE_NAME", "neo4j")

        missing = [
            name
            for name, value in [
                ("NEO4J_URI", uri),
                ("NEO4J_AURA_USERNAME", username),
                ("NEO4J_AURA_PASSWORD", password),
                ("NEO4J_AURA_DATABASE_NAME", database),
            ]
            if not value
        ]
        if missing:
            raise RuntimeError(f"Missing required Neo4j env vars: {', '.join(missing)}")

        return cls(
            uri=uri,
            username=username,
            password=password,
            database=database,
            instance_id=os.getenv("NEO4J_AURA_INSTANCE_ID"),
            instance_name=os.getenv("NEO4J_AURA_INSTANCE_NAME"),
            query_api=os.getenv("NEO4J_AURA_QUERY_API"),
        )


class Neo4jAuraClient:
    def __init__(self, settings: Neo4jAuraSettings) -> None:
        self.settings = settings
        self.driver: AsyncDriver | None = None

    async def connect(self) -> None:
        if self.driver is not None:
            return

        self.driver = AsyncGraphDatabase.driver(
            self.settings.uri,
            auth=(self.settings.username, self.settings.password),
            # optional tuning
            max_connection_lifetime=60 * 30,
            liveness_check_timeout=30.0,
        )

        await self.driver.verify_connectivity()

    async def close(self) -> None:
        if self.driver is not None:
            await self.driver.close()
            self.driver = None

    async def __aenter__(self) -> "Neo4jAuraClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def _require_driver(self) -> AsyncDriver:
        if self.driver is None:
            raise RuntimeError("Neo4j driver is not connected.")
        return self.driver

    async def execute_query(
        self,
        cypher: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Best for simple one-shot queries.
        Returns rows as plain dicts.
        """
        driver = self._require_driver()
        records, summary, keys = await driver.execute_query(
            cypher,
            parameters_=dict(parameters or {}),
            database_=self.settings.database,
        )
        return [dict(record.items()) for record in records]

    async def execute_write(
        self,
        cypher: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Explicit write-session version if you want tighter control later.
        """
        driver = self._require_driver()
        async with driver.session(database=self.settings.database) as session:
            result = await session.run(cypher, dict(parameters or {}))
            records = await result.data()
            await result.consume()
            return records

    async def execute_read(
        self,
        cypher: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        driver = self._require_driver()
        async with driver.session(database=self.settings.database) as session:
            result = await session.run(cypher, dict(parameters or {}))
            records = await result.data()
            await result.consume()
            return records


async def get_neo4j_aura_client() -> Neo4jAuraClient:
    settings = Neo4jAuraSettings.from_env()
    client = Neo4jAuraClient(settings)
    await client.connect()
    return client 


if __name__ == "__main__":  
    import asyncio 

    async def main() -> None: 
        client = await get_neo4j_aura_client()   

        try: 
            companies = await client.execute_query( 
                """ 
                MATCH (o: Organization) 
                RETURN properties(o) as company 
                """, 
                parameters={} 
            ) 

            print(f"Found {len(companies)} companies") 

        except Exception as e: 
            raise e   

    asyncio.run(main()) 


