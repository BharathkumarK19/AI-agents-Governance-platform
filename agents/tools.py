import os
import re
from html import unescape

import requests
from dotenv import load_dotenv
from tavily import TavilyClient


load_dotenv()


class ResearchToolError(Exception):
    pass


class TavilySearchError(ResearchToolError):
    pass


class DuckDuckGoSearchError(ResearchToolError):
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


def search_duckduckgo(query, max_results=5):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
        html = response.text
        pattern = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
            r'<a[^>]*class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
            re.DOTALL,
        )
        results = []
        for match in pattern.finditer(html):
            title = _strip_html(match.group("title"))
            url = unescape(match.group("url"))
            snippet = _strip_html(match.group("snippet"))
            if title and snippet:
                results.append(
                    {
                        "title": title,
                        "url": url,
                        "content": snippet,
                        "source": "duckduckgo_search",
                    }
                )
            if len(results) >= max_results:
                break
        return results
    except requests.exceptions.RequestException as exc:
        raise DuckDuckGoSearchError(
            "DuckDuckGo search is unavailable right now."
        ) from exc
    except Exception as exc:
        raise DuckDuckGoSearchError(
            "DuckDuckGo search parsing failed."
        ) from exc


def _strip_html(value):
    text = re.sub(r"<[^>]+>", " ", value or "")
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


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
