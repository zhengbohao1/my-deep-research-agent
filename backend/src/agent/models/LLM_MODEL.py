import logging

from langchain_openai import ChatOpenAI

from backend.src.agent.config.env_utils import LLM_API_KEY, LLM_BASE_URL, TAVILY_API_KEY
from langchain_tavily import TavilySearch

class ModelInstances:
    try:
        query_generator_model = ChatOpenAI(
            model="qwen-turbo",
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            temperature=0.1
        )

        reflection_model = ChatOpenAI(
            model="qwen-flash",
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL
        )

        answer_model = ChatOpenAI(
            model="qwen-flash",
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL
        )

        leader_llm = ChatOpenAI(
            model="qwen-max",
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL
        )

        tavily_search = TavilySearch(
            max_results=2,
            search_depth="advanced",
            api_key=TAVILY_API_KEY
        )
    except Exception as e:
        logging.error(f"模型初始化失败: {str(e)}")
        raise