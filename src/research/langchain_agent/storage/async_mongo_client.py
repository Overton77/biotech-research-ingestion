from __future__ import annotations

import os 

from pymongo import AsyncMongoClient 

from pymongo.asynchronous.database import AsyncDatabase  # type: ignore[import]
from pymongo.asynchronous.collection import AsyncCollection  # type: ignore[import] 

from dotenv import load_dotenv 

load_dotenv()  


# TODO: Store episodes in Neo4j Instead of MongoDB and only use API to query Episodes 
# TODO: Keeping this for now because episodes are in mongo db collections . All entities will move to neo4j 

MONGO_URI = os.getenv("MONGODB_URI") or os.getenv("MONGO_URI")



def create_mongo_client() -> AsyncMongoClient:
    """
    Create an AsyncMongoClient.

    Keep this as a singleton at app level (e.g. FastAPI startup)
    and reuse it instead of creating a new one per request.
    """
    client: AsyncMongoClient = AsyncMongoClient(MONGO_URI)
    return client


mongo_client: AsyncMongoClient = create_mongo_client() 