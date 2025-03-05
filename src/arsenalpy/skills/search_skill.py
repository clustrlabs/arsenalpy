import json
import os
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from .skill_utils import skill


# access to serper, brave, tavily APIs complete search solution
class SearchEngine(str, Enum):
    SERPER = "serper"
    BRAVE = "brave"
    TAVILY = "tavily"


class SearchRequest(BaseModel):
    q: str = Field(
        ...,
        description="The search query to execute across supported search engines",
    )
    engine: SearchEngine = Field(
        default=SearchEngine.BRAVE,
        description="The search engine to use (brave, serper, bing, or tavily)",
    )
    llm_rewrite: bool = Field(
        default=False,
        description="Whether to use LLM to rewrite and optimize the search query",
    )
    rerank: bool = Field(
        default=True,
        description="Whether to rerank search results using Cohere's rerank API for better relevance",
    )


class SearchResult(BaseModel):
    """Base model for standardized search results"""

    title: str
    link: str
    snippet: str = ""
    position: int = 0
    relevance_score: float = 0.0


async def _rerank_results(
    query: str, results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Reranks search results using Cohere's rerank API.

    Args:
        query (str): The original search query
        results (List[Dict[str, Any]]): List of search results to rerank

    Returns:
        List[Dict[str, Any]]: Reranked search results with relevance scores
    """
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        print("COHERE_API_KEY not set in environment. Skipping reranking.")
        return results

    # Extract documents from results
    documents = []
    for result in results:
        if "title" in result and "snippet" in result:
            documents.append(f"{result.get('title', '')}. {result.get('snippet', '')}")
        elif "title" in result:
            documents.append(result.get("title", ""))

    if not documents:
        return results

    url = "https://api.cohere.com/v2/rerank"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    data = {
        "model": "rerank-v3.5",
        "query": query,
        "documents": documents,
        "top_n": len(documents),  # Rerank all results
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            rerank_data = response.json()

            # Apply reranking scores to original results
            reranked_results = []
            for item in rerank_data.get("results", []):
                orig_idx = item.get("index")
                if 0 <= orig_idx < len(results):
                    result = results[orig_idx].copy()
                    result["relevance_score"] = item.get("relevance_score", 0.0)
                    reranked_results.append(result)

            # Sort by relevance score
            reranked_results.sort(
                key=lambda x: x.get("relevance_score", 0), reverse=True
            )

            # Update positions
            for idx, result in enumerate(reranked_results):
                result["position"] = idx + 1

            return reranked_results
    except Exception as e:
        print(f"Reranking failed: {str(e)}. Using original results.")
        return results


async def _rewrite_query(query: str) -> str:
    """Use an LLM to rewrite and optimize the search query"""
    from ..agents import Agent, AgentConfig

    def get_api_key() -> Optional[str]:
        provider = get_provider()
        if provider == "openrouter":
            return os.getenv("OPENROUTER_API_KEY")
        elif provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif provider == "x-ai":
            return os.getenv("XAI_API_KEY")
        return None

    def get_model() -> str:
        provider = get_provider()
        if provider == "openrouter":
            return "openai/o3-mini"
        elif provider == "openai":
            return "o3-mini"
        elif provider == "x-ai":
            return "grok-2-1212"
        return "openai/o3-mini"  # default fallback

    def get_provider() -> str:
        if os.getenv("OPENROUTER_API_KEY"):
            return "openrouter"
        elif os.getenv("OPENAI_API_KEY"):
            return "openai"
        elif os.getenv("XAI_API_KEY"):
            return "x-ai"
        return "openrouter"  # default fallback

    print(
        "calling with api key",
        get_api_key(),
        get_model(),
        get_provider(),
        flush=True,
    )

    agent = Agent(
        name="search_optimizer",
        system_prompt="""You are a search query optimization expert that always responds in JSON format.""",
        model=get_model(),
        api_key=get_api_key(),
        provider=get_provider(),
        config=AgentConfig(temperature=0.3, response_format={"type": "json_object"}),
    )

    prompt = """You are a search query optimization expert. Rewrite the following query to make it more effective for search engines.
    Return your response as a JSON object with two fields:
    - optimized_query: the rewritten search query
    - reasoning: brief explanation of why you rewrote it this way

    Original query: {query}
    """

    try:
        response = await agent.do(prompt.format(query=query))
        result = json.loads(response)

        return str(result["optimized_query"])

    except Exception as e:
        print(f"Query rewrite failed: {str(e)}")
        return query  # fallback to original query if rewrite fails


async def _search_tavily(query: str, api_key: str) -> Dict[str, Any]:
    """Execute search using Tavily API"""
    url = "https://api.tavily.com/search"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = {"query": query}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)
        response.raise_for_status()
        result: Dict[str, Any] = response.json()
        return result


async def _search_serper(query: str, api_key: str) -> Dict[str, Any]:
    """Execute search using Serper API"""
    url = "https://google.serper.dev/search"
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": api_key,
    }
    data = {"q": query}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)
        response.raise_for_status()
        result: Dict[str, Any] = response.json()
        return result


async def _search_brave(query: str, api_key: str) -> Dict[str, Any]:
    """Execute search using Brave Search API"""
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key,
    }
    params = {
        "q": query,
        "freshness": "pm",
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        result: Dict[str, Any] = response.json()
        return result


def _standardize_results(
    engine: SearchEngine, raw_results: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Standardizes results from different search engines into a common format.

    Args:
        engine (SearchEngine): The search engine used
        raw_results (Dict[str, Any]): The raw search results

    Returns:
        List[Dict[str, Any]]: Standardized search results
    """
    standardized = []

    if engine == SearchEngine.SERPER:
        for idx, item in enumerate(raw_results.get("organic", [])):
            result = {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "position": idx + 1,
                "relevance_score": 0.0,
            }
            standardized.append(result)

    elif engine == SearchEngine.BRAVE:
        for idx, item in enumerate(raw_results.get("web", {}).get("results", [])):
            result = {
                "title": item.get("title", ""),
                "link": item.get("url", ""),
                "snippet": item.get("description", ""),
                "position": idx + 1,
                "relevance_score": 0.0,
            }
            standardized.append(result)

    elif engine == SearchEngine.TAVILY:
        for idx, item in enumerate(raw_results.get("results", [])):
            result = {
                "title": item.get("title", ""),
                "link": item.get("url", ""),
                "snippet": item.get("content", ""),
                "position": idx + 1,
                "relevance_score": 0.0,
            }
            standardized.append(result)

    return standardized


@skill(SearchRequest)
async def search(
    q: str,
    engine: SearchEngine = SearchEngine.BRAVE,
    llm_rewrite: bool = True,
    rerank: bool = True,
) -> Dict[str, Any]:
    """
    Executes a search query using the specified search engine (Serper, Bing, or Brave).
    Optionally rewrites the query using LLM for better results and reranks results using Cohere's API.

    This skill supports multiple search engines and returns results in a standardized format.
    Required environment variables depend on the chosen engine:
    - SERPER_API_KEY for Serper
    - BRAVE_API_KEY for Brave
    - TAVILY_API_KEY for Tavily
    - COHERE_API_KEY if using rerank=True
    - OPENAI_API_KEY if using llm_rewrite=True

    Args:
        q (str): The search query
        engine (SearchEngine): The search engine to use
        llm_rewrite (bool): Whether to use LLM to optimize the query
        rerank (bool): Whether to rerank results using Cohere's API

    Returns:
        Dict[str, Any]: Search results with the original raw data and standardized results
    """
    print("search tool is being called with q", q, flush=True)
    if llm_rewrite:
        q = await _rewrite_query(q)

    engine_api_keys = {
        SearchEngine.SERPER: "SERPER_API_KEY",
        SearchEngine.BRAVE: "BRAVE_API_KEY",
        SearchEngine.TAVILY: "TAVILY_API_KEY",
    }

    api_key_name = engine_api_keys[engine]
    api_key = os.getenv(api_key_name)

    if not api_key:
        raise ValueError(f"{api_key_name} not set in environment.")

    search_functions = {
        SearchEngine.SERPER: _search_serper,
        SearchEngine.BRAVE: _search_brave,
        SearchEngine.TAVILY: _search_tavily,
    }

    search_function = search_functions[engine]
    raw_results = await search_function(q, api_key)

    # Standardize results into a common format
    standardized_results = _standardize_results(engine, raw_results)

    # Apply reranking if enabled
    if rerank and standardized_results:
        try:
            standardized_results = await _rerank_results(q, standardized_results)
        except Exception as e:
            print(f"Reranking failed: {str(e)}. Using original results.")

    # Include both raw results and standardized results in the response
    return {
        "raw": raw_results,
        "results": standardized_results,
        "query": q,
        "engine": engine,
        "reranked": rerank,
    }
