# src/temporal/client.py
from temporalio.client import Client

async def get_temporal_client() -> Client:
    return await Client.connect("localhost:7233", namespace="default") 


