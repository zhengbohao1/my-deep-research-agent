import os

from dotenv import load_dotenv

load_dotenv(override=True)

LLM_API_KEY=os.environ.get("LLM_API_KEY")

LLM_BASE_URL=os.environ.get("LLM_BASE_URL")

EMBED_BASE_URL=os.environ.get("EMBED_BASE_URL")

EMBED_API_KEY=os.environ.get("EMBED_API_KEY")

TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")





