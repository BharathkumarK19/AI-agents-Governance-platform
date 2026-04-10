import os

import requests
from dotenv import load_dotenv
from tavily import TavilyClient


load_dotenv()


class ResearchToolError(Exception):
    pass


class TavilySearchError(ResearchToolError):
    pass


def _get_client():
    api_key = os.getenv("TAVILY_API_KEY")
    if api_key is not None:
        api_key = api_key.strip()
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not set.")
    return TavilyClient(api_key=api_key)


def search_tavily(query):
    try:
        response = _get_client().search(query=query, max_results=5)
        return response["results"]
    except requests.exceptions.RequestException as exc:
        raise TavilySearchError(
            "Live Tavily research is unavailable right now. "
            "This usually means your internet connection, DNS, firewall, or "
            "Tavily API access is blocked."
        ) from exc
    except Exception as exc:
        raise TavilySearchError(
            "Tavily search failed before the research crew could start."
        ) from exc


def build_source_bundle(results):
    if not results:
        return "No external results were returned from Tavily."

    blocks = []
    for index, item in enumerate(results, start=1):
        title = item.get("title", "Untitled result")
        url = item.get("url", "No URL provided")
        content = item.get("content", "").strip() or "No summary provided."
        blocks.append(
            f"[Source {index}]\nTitle: {title}\nURL: {url}\nSummary: {content}"
        )

    return "\n\n".join(blocks)
