from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from .skill_utils import skill


class BrowseMode(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    INTERACTIVE = "interactive"


class BrowseRequest(BaseModel):
    url: str = Field(
        ...,
        description="The URL to browse and extract content from",
    )
    mode: BrowseMode = Field(
        default=BrowseMode.STATIC,
        description="The browsing mode (static, dynamic, or interactive)",
    )
    javascript: Optional[List[str]] = Field(
        default=None,
        description="Custom JavaScript to execute on the page (for dynamic/interactive modes)",
    )
    wait_for_selector: Optional[str] = Field(
        default=None,
        description="CSS selector to wait for before extracting content",
    )
    extract_links: bool = Field(
        default=False,
        description="Whether to extract links from the page",
    )
    content_threshold: float = Field(
        default=0.5,
        description="Threshold for content filtering (0.0 to 1.0)",
    )
    visible_browser: bool = Field(
        default=False,
        description="Whether to show the browser window (not headless)",
    )


class BrowseResult(BaseModel):
    """Base model for standardized browse results"""

    url: str
    content: str
    title: str
    links: Optional[List[Dict[str, str]]] = None
    success: bool
    error: Optional[str] = None


async def _crawl_with_crawl4ai(
    url: str,
    mode: BrowseMode,
    javascript: Optional[List[str]] = None,
    wait_for_selector: Optional[str] = None,
    content_threshold: float = 0.5,
    extract_links: bool = False,
    visible_browser: bool = False,
) -> Dict[str, Any]:
    """
    Crawls a URL using Crawl4AI and extracts content.

    Args:
        url (str): The URL to crawl
        mode (BrowseMode): The browsing mode
        javascript (Optional[List[str]]): Custom JavaScript to execute
        wait_for_selector (Optional[str]): CSS selector to wait for
        content_threshold (float): Threshold for content filtering
        extract_links (bool): Whether to extract links
        visible_browser (bool): Whether to show the browser window

    Returns:
        Dict[str, Any]: The crawl results
    """
    try:
        from crawl4ai import (
            AsyncWebCrawler,
            BrowserConfig,  # type: ignore
            CacheMode,
            CrawlerRunConfig,
        )
        from crawl4ai.content_filter_strategy import (
            PruningContentFilter,  # type: ignore
        )
        from crawl4ai.markdown_generation_strategy import (
            DefaultMarkdownGenerator,  # type: ignore
        )
    except ImportError as e:
        raise ImportError(
            "Crawl4AI is not installed. Please install with: pip install crawl4ai"
        ) from e

    # Configure browser based on mode
    # When visible_browser is True or mode is INTERACTIVE, make browser visible
    headless = not visible_browser and mode != BrowseMode.INTERACTIVE
    browser_config = BrowserConfig(
        headless=headless,
        java_script_enabled=mode != BrowseMode.STATIC,
    )

    # Configure content filter
    md_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(
            threshold=content_threshold, threshold_type="fixed"
        )
    )

    # Configure crawler
    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=md_generator,
    )

    # Add JavaScript if provided
    if javascript and mode != BrowseMode.STATIC:
        crawler_config.js_code = javascript

    # Add selector waiting if provided
    if wait_for_selector and mode != BrowseMode.STATIC:
        crawler_config.wait_for = wait_for_selector

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url, config=crawler_config)

            # Extract and process content properly
            if hasattr(result, "markdown_v2") and hasattr(
                result.markdown_v2, "fit_markdown"
            ):
                # New Crawl4AI API format with markdown_v2
                content = result.markdown_v2.fit_markdown
            elif hasattr(result, "markdown") and isinstance(result.markdown, str):
                # Direct string markdown (older Crawl4AI versions)
                content = result.markdown
            elif hasattr(result, "markdown") and hasattr(
                result.markdown, "fit_markdown"
            ):
                # Object-based markdown with fit_markdown attribute
                content = result.markdown.fit_markdown
            else:
                # Fallback to raw HTML if no markdown found
                content = result.html if hasattr(result, "html") else ""

            # Extract title
            title = result.title if hasattr(result, "title") else ""

            # Get links safely
            links_data = []
            if extract_links:
                # Different ways to extract links based on version
                if hasattr(result, "links") and result.links:
                    if isinstance(result.links, list):
                        # If links is a list of objects with url and text attributes
                        for link in result.links:
                            if hasattr(link, "url") and hasattr(link, "text"):
                                links_data.append({"url": link.url, "text": link.text})
                            elif (
                                isinstance(link, dict)
                                and "url" in link
                                and "text" in link
                            ):
                                links_data.append(
                                    {"url": link["url"], "text": link["text"]}
                                )

                # If no links were found but we have HTML, try to extract from HTML as fallback
                if not links_data and hasattr(result, "html"):
                    # Very simple regex-based extraction
                    import re

                    pattern = r'<a\s+(?:[^>]*?\s+)?href="([^"]*)"[^>]*>(.*?)</a>'
                    for match in re.finditer(pattern, result.html):
                        href, text = match.groups()
                        # Simple cleaning of the link text
                        clean_text = re.sub(r"<[^>]*>", "", text).strip()
                        if href and href.startswith(("http://", "https://")):
                            links_data.append({"url": href, "text": clean_text})

            return {
                "url": url,
                "content": content,
                "title": title,
                "links": links_data if links_data else None,
                "success": True,
                "error": None,
            }
    except Exception as e:
        import traceback

        error_details = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Crawl4AI error: {error_details}")
        return {
            "url": url,
            "content": "",
            "title": "",
            "links": None,
            "success": False,
            "error": str(e),
        }


async def _fetch_with_httpx(url: str) -> Dict[str, Any]:
    """
    Simple HTTP fetcher for static pages (fallback when Crawl4AI is not available)

    Args:
        url (str): The URL to fetch

    Returns:
        Dict[str, Any]: The fetch results
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()

            # Very basic title extraction
            title = ""
            if (
                b"<title>" in response.content.lower()
                and b"</title>" in response.content.lower()
            ):
                start = response.content.lower().find(b"<title>") + 7
                end = response.content.lower().find(b"</title>")
                title = response.content[start:end].decode("utf-8", errors="ignore")

            # Convert HTML to simple text/markdown (very basic)
            content = response.text

            # Very simple link extraction
            links_data = []
            try:
                import re

                pattern = r'<a\s+(?:[^>]*?\s+)?href="([^"]*)"[^>]*>(.*?)</a>'
                for match in re.finditer(pattern, response.text):
                    href, text = match.groups()
                    # Simple cleaning of the link text
                    clean_text = re.sub(r"<[^>]*>", "", text).strip()
                    if href and href.startswith(("http://", "https://")):
                        links_data.append({"url": href, "text": clean_text})
            except Exception:
                pass

            return {
                "url": url,
                "content": content,
                "title": title,
                "links": links_data if links_data else None,
                "success": True,
                "error": None,
            }
    except Exception as e:
        return {
            "url": url,
            "content": "",
            "title": "",
            "links": None,
            "success": False,
            "error": str(e),
        }


@skill(BrowseRequest)
async def browse(
    url: str,
    mode: BrowseMode = BrowseMode.STATIC,
    javascript: Optional[List[str]] = None,
    wait_for_selector: Optional[str] = None,
    extract_links: bool = False,
    content_threshold: float = 0.5,
    visible_browser: bool = False,
) -> Dict[str, Any]:
    """
    Browses a URL and extracts its content as markdown.

    This skill uses Crawl4AI to browse web pages and extract their content.
    It supports different browsing modes for static and dynamic pages.

    Args:
        url (str): The URL to browse
        mode (BrowseMode): The browse mode (static, dynamic, or interactive)
        javascript (Optional[List[str]]): Custom JavaScript to execute on the page
        wait_for_selector (Optional[str]): CSS selector to wait for before extracting
        extract_links (bool): Whether to extract links from the page
        content_threshold (float): Threshold for content filtering (0.0 to 1.0)
        visible_browser (bool): Whether to show the browser window (not headless)

    Returns:
        Dict[str, Any]: Browse results including content and metadata
    """
    print(
        f"browse tool is being called with url: {url}, mode: {mode}, visible: {visible_browser}",
        flush=True,
    )

    try:
        # Try to use Crawl4AI
        result = await _crawl_with_crawl4ai(
            url=url,
            mode=mode,
            javascript=javascript,
            wait_for_selector=wait_for_selector,
            content_threshold=content_threshold,
            extract_links=extract_links,
            visible_browser=visible_browser,
        )
    except ImportError:
        # Fallback to basic httpx if Crawl4AI not available
        print("Crawl4AI not available, falling back to basic HTTP fetcher", flush=True)
        result = await _fetch_with_httpx(url)

    return result
