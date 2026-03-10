import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=3600,
)