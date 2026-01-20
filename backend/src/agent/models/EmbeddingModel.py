import logging

from langchain_openai import ChatOpenAI

from backend.src.agent.config.env_utils import EMBED_API_KEY, EMBED_BASE_URL


class EmbeddingModel:
    try:
        text_embedding_v3 = ChatOpenAI(
            model="text-embedding-v3",
            api_key=EMBED_API_KEY,
            base_url=EMBED_BASE_URL
        )
    except Exception as e:
        logging.error(f"模型初始化失败: {str(e)}")
        raise