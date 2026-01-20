from datetime import datetime

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from backend.src.agent.config.configuration import Configuration
from backend.src.agent.models.LLM_MODEL import ModelInstances
from backend.src.agent.prompts.web_researcher_prompt import web_searcher_instructions
from backend.src.agent.states.overallstate import OverallState
import json

def web_research(state: OverallState, config: RunnableConfig ,context:str) :
    """LangGraph node that performs web research using Tavily Search API.

        Executes a web search using Tavily Search API and then uses DeepSeek to analyze and summarize the results.

        Args:
            state: Current graph state containing the search query and research loop count
            config: Configuration for the runnable, including search API settings

        Returns:
            Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
        """
    # Configure

    # Perform search using Tavily
    # Ensure search_query is a string, not a list
    search_query = state["search_query"]

    if isinstance(search_query, list):
        search_query = search_query[0] if search_query else ""
    search_results = ModelInstances.tavily_search.invoke(search_query)

    # Extract content and URLs from search results
    search_content = ""
    sources_gathered = []

    # Handle different return formats from Tavily
    if isinstance(search_results, list):
        results_to_process = search_results
    elif isinstance(search_results, dict):
        # Tavily typically returns a dict with 'results' key
        results_to_process = search_results.get('results', [])
    elif isinstance(search_results, str):
        # If it's a string, it might be JSON content
        try:
            parsed_results = json.loads(search_results)
            if isinstance(parsed_results, dict) and 'results' in parsed_results:
                results_to_process = parsed_results['results']
            else:
                results_to_process = [{"title": "Search Result", "url": "", "content": search_results}]
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            print(f"Warning: Failed to parse search results as JSON: {e}")
            results_to_process = [{"title": "Search Result", "url": "", "content": search_results}]
    else:
        results_to_process = []

    for i, result in enumerate(results_to_process):
        if isinstance(result, dict):
            title = result.get('title', f'Result {i + 1}')
            url = result.get('url', f'https://search-result-{i + 1}.com')
            content = result.get('content', str(result))
        else:
            title = f'Result {i + 1}'
            url = f'https://search-result-{i + 1}.com'
            content = str(result)

        search_content += f"Source {i + 1}: {title}\nURL: {url}\nContent: {content}\n\n"
        sources_gathered.append({
            "title": title,
            "url": url,
            "content": content[:500] + "..." if len(content) > 500 else content,
            "short_url": f"[{i + 1}]",
            "value": url,
            "label": title  # Add label field for frontend compatibility
        })

    # Format prompt for LLM to analyze the search results
    formatted_prompt = web_searcher_instructions.format(
        current_date=datetime.now().strftime("%Y年%m月%d日"),
        research_topic=search_query,
        context=context
    )

    # Add search results to the prompt
    analysis_prompt = f"{formatted_prompt}\n\n搜索结果：\n{search_content}\n\n请分析这些搜索结果并提供带有引用的综合摘要。请用中文回答。"

    # Use LLM to analyze and summarize the search results
    llm = ModelInstances.answer_model

    response = llm.invoke(analysis_prompt)

    # Insert citation markers
    modified_text = response.content
    for i, source in enumerate(sources_gathered):
        # Replace URL references with short citations
        if source['url'] in modified_text:
            modified_text = modified_text.replace(source['url'], source['short_url'])
        # Also try to match domain names
        domain = source['url'].split('/')[2] if len(source['url'].split('/')) > 2 else source['url']
        if domain in modified_text:
            modified_text = modified_text.replace(domain, source['short_url'])

    print(f"网上搜索url结果如下\n：{sources_gathered}")
    print(f"webreseach结果如下\n：{modified_text}")


    return Command(update={
        "sources_gathered": sources_gathered,
        "web_research_result": [modified_text],
    })