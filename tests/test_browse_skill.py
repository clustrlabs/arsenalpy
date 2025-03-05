import pytest
from arsenalpy.skills.browse_skill import BrowseMode, browse


@pytest.mark.asyncio
async def test_browse_skill_prints(capsys, monkeypatch):
    # Mock the _crawl_with_crawl4ai function to avoid real web requests
    async def fake_crawl_with_crawl4ai(
        url,
        mode,
        javascript=None,
        wait_for_selector=None,
        content_threshold=0.5,
        extract_links=False,
        visible_browser=False,
    ):
        return {
            "url": url,
            "content": "Fake content from crawl4ai",
            "title": "Fake Title",
            "links": (
                [{"url": "http://example.com", "text": "Example Link"}]
                if extract_links
                else None
            ),
            "success": True,
            "error": None,
        }

    monkeypatch.setattr(
        "arsenalpy.skills.browse_skill._crawl_with_crawl4ai", fake_crawl_with_crawl4ai
    )

    # Call the browse skill with default parameters
    result = await browse(
        url="http://example.com", mode=BrowseMode.STATIC, extract_links=True
    )

    # Capture printed output
    captured = capsys.readouterr().out

    # Verify that print statements were executed
    assert "browse tool is being called with url" in captured

    # Print captured output for debugging (optional)
    print(captured)

    # Verify the result structure
    assert result["url"] == "http://example.com"
    assert result["content"] == "Fake content from crawl4ai"
    assert result["title"] == "Fake Title"
    assert result["links"][0]["url"] == "http://example.com"
    assert result["success"] is True


@pytest.mark.asyncio
async def test_browse_skill_fallback_to_httpx(capsys, monkeypatch):
    # Simulate ImportError for Crawl4AI to trigger fallback
    async def fake_crawl_with_crawl4ai(*args, **kwargs):
        raise ImportError("Crawl4AI is not installed")

    monkeypatch.setattr(
        "arsenalpy.skills.browse_skill._crawl_with_crawl4ai", fake_crawl_with_crawl4ai
    )

    # Mock the _fetch_with_httpx function
    async def fake_fetch_with_httpx(url):
        return {
            "url": url,
            "content": "Fake content from httpx",
            "title": "Fake HTTPX Title",
            "links": None,
            "success": True,
            "error": None,
        }

    monkeypatch.setattr(
        "arsenalpy.skills.browse_skill._fetch_with_httpx", fake_fetch_with_httpx
    )

    # Call the browse skill
    result = await browse(url="http://example.com")

    # Capture printed output
    captured = capsys.readouterr().out

    # Verify fallback message
    assert "Crawl4AI not available, falling back to basic HTTP fetcher" in captured

    # Verify the result structure from httpx fallback
    assert result["url"] == "http://example.com"
    assert result["content"] == "Fake content from httpx"
    assert result["title"] == "Fake HTTPX Title"
    assert result["success"] is True


@pytest.mark.asyncio
async def test_browse_skill_dynamic_mode(capsys, monkeypatch):
    # Mock to track whether dynamic mode parameters are passed correctly
    async def fake_crawl_with_crawl4ai(
        url,
        mode,
        javascript=None,
        wait_for_selector=None,
        content_threshold=0.5,
        extract_links=False,
        visible_browser=False,
    ):
        # Return mode and parameters for validation
        return {
            "url": url,
            "content": f"Content from {mode} mode",
            "title": "Dynamic Page",
            "mode": mode,
            "javascript": javascript,
            "wait_for_selector": wait_for_selector,
            "visible_browser": visible_browser,
            "success": True,
            "error": None,
        }

    monkeypatch.setattr(
        "arsenalpy.skills.browse_skill._crawl_with_crawl4ai", fake_crawl_with_crawl4ai
    )

    # Custom JavaScript and selector for dynamic mode
    custom_js = ["document.querySelector('body').style.backgroundColor = 'red';"]
    custom_selector = "#dynamic-content"

    # Call the browse skill with dynamic mode
    result = await browse(
        url="http://example.com/dynamic",
        mode=BrowseMode.DYNAMIC,
        javascript=custom_js,
        wait_for_selector=custom_selector,
        visible_browser=True,
    )

    # Verify that the parameters were passed correctly
    assert result["mode"] == BrowseMode.DYNAMIC
    assert result["javascript"] == custom_js
    assert result["wait_for_selector"] == custom_selector
    assert result["visible_browser"] is True
    assert result["content"] == f"Content from {BrowseMode.DYNAMIC} mode"


@pytest.mark.asyncio
async def test_browse_skill_error_handling(capsys, monkeypatch):
    # Mock to simulate an error in Crawl4AI
    async def fake_crawl_with_crawl4ai(*args, **kwargs):
        return {
            "url": kwargs.get("url", ""),  # Get url from kwargs instead of args
            "content": "",
            "title": "",
            "links": None,
            "success": False,
            "error": "Simulated crawling error",
        }

    monkeypatch.setattr(
        "arsenalpy.skills.browse_skill._crawl_with_crawl4ai", fake_crawl_with_crawl4ai
    )

    # Call the browse skill
    result = await browse(url="http://example.com/error")

    # Verify the error is captured in the result
    assert result["success"] is False
    assert result["error"] == "Simulated crawling error"
    assert result["content"] == ""


@pytest.mark.asyncio
async def test_browse_skill_extract_links(capsys, monkeypatch):
    # Mock with link extraction
    async def fake_crawl_with_crawl4ai(
        url,
        mode,
        javascript=None,
        wait_for_selector=None,
        content_threshold=0.5,
        extract_links=False,
        visible_browser=False,
    ):
        links = None
        if extract_links:
            links = [
                {"url": "http://example.com/page1", "text": "Page 1"},
                {"url": "http://example.com/page2", "text": "Page 2"},
                {"url": "http://example.com/page3", "text": "Page 3"},
            ]

        return {
            "url": url,
            "content": "Content with links",
            "title": "Page with Links",
            "links": links,
            "success": True,
            "error": None,
        }

    monkeypatch.setattr(
        "arsenalpy.skills.browse_skill._crawl_with_crawl4ai", fake_crawl_with_crawl4ai
    )

    # Call browse with extract_links=True
    result = await browse(url="http://example.com/links", extract_links=True)

    # Verify links were extracted
    assert result["links"] is not None
    assert len(result["links"]) == 3
    assert result["links"][0]["url"] == "http://example.com/page1"
    assert result["links"][1]["text"] == "Page 2"


@pytest.mark.asyncio
async def test_browse_skill_content_threshold(capsys, monkeypatch):
    # Mock to verify content_threshold parameter is passed correctly
    async def fake_crawl_with_crawl4ai(
        url,
        mode,
        javascript=None,
        wait_for_selector=None,
        content_threshold=0.5,
        extract_links=False,
        visible_browser=False,
    ):
        return {
            "url": url,
            "content": f"Content filtered with threshold {content_threshold}",
            "title": "Content Threshold Test",
            "content_threshold": content_threshold,  # Return the parameter for verification
            "success": True,
            "error": None,
        }

    monkeypatch.setattr(
        "arsenalpy.skills.browse_skill._crawl_with_crawl4ai", fake_crawl_with_crawl4ai
    )

    # Call browse with custom content_threshold
    custom_threshold = 0.75
    result = await browse(
        url="http://example.com/content", content_threshold=custom_threshold
    )

    # Verify threshold was passed correctly
    assert result["content_threshold"] == custom_threshold
    assert f"threshold {custom_threshold}" in result["content"]


@pytest.mark.asyncio
async def test_browse_skill_interactive_mode(capsys, monkeypatch):
    # Mock to test interactive mode specifics
    async def fake_crawl_with_crawl4ai(
        url,
        mode,
        javascript=None,
        wait_for_selector=None,
        content_threshold=0.5,
        extract_links=False,
        visible_browser=False,
    ):
        # In interactive mode, browser should always be visible
        browser_visible = mode == BrowseMode.INTERACTIVE or visible_browser

        return {
            "url": url,
            "content": f"Content from {mode} mode",
            "title": "Interactive Test",
            "mode": mode,
            "browser_visible": browser_visible,
            "success": True,
            "error": None,
        }

    monkeypatch.setattr(
        "arsenalpy.skills.browse_skill._crawl_with_crawl4ai", fake_crawl_with_crawl4ai
    )

    # Call browse with interactive mode
    result = await browse(
        url="http://example.com/interactive",
        mode=BrowseMode.INTERACTIVE,
        visible_browser=False,  # This should be overridden in interactive mode
    )

    # Verify interactive mode specifics
    assert result["mode"] == BrowseMode.INTERACTIVE
    # Browser should be visible in interactive mode, regardless of visible_browser parameter
    assert result["browser_visible"] is True


@pytest.mark.asyncio
async def test_browse_skill_httpx_error(capsys, monkeypatch):
    # First, make Crawl4AI unavailable
    async def fake_crawl_with_crawl4ai(*args, **kwargs):
        raise ImportError("Crawl4AI is not installed")

    monkeypatch.setattr(
        "arsenalpy.skills.browse_skill._crawl_with_crawl4ai", fake_crawl_with_crawl4ai
    )

    # Then, simulate an error in the httpx fallback
    async def fake_fetch_with_httpx(url):
        return {
            "url": url,
            "content": "",
            "title": "",
            "links": None,
            "success": False,
            "error": "HTTPX connection error",
        }

    monkeypatch.setattr(
        "arsenalpy.skills.browse_skill._fetch_with_httpx", fake_fetch_with_httpx
    )

    # Call browse
    result = await browse(url="http://example.com/error")

    # Capture output
    captured = capsys.readouterr().out

    # Verify fallback was attempted
    assert "Crawl4AI not available, falling back to basic HTTP fetcher" in captured

    # Verify error from httpx
    assert result["success"] is False
    assert result["error"] == "HTTPX connection error"
