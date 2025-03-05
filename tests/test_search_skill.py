import pytest
from arsenalpy.skills.search_skill import search


@pytest.mark.asyncio
async def test_search_skill_prints(capsys, monkeypatch):
    # Set a dummy API key so that the search function doesn't fail on API key validation.
    monkeypatch.setenv("BRAVE_API_KEY", "test_api_key")

    # Patch the _rewrite_query function to avoid instantiating an Agent.
    async def fake_rewrite_query(query: str) -> str:
        # Return an optimized/re-written version of the query.
        return query + " optimized"

    monkeypatch.setattr(
        "arsenalpy.skills.search_skill._rewrite_query", fake_rewrite_query
    )

    # Patch the _search_brave function to avoid real HTTP calls and return a fake response.
    async def fake_search_brave(query: str, api_key: str):
        return {
            "web": {
                "results": [
                    {
                        "title": "Fake Italian Restaurant",
                        "url": "http://example.com",
                        "description": "A fake restaurant description",
                    }
                ]
            }
        }

    monkeypatch.setattr(
        "arsenalpy.skills.search_skill._search_brave", fake_search_brave
    )

    # Call the search skill with LLM rewriting enabled.
    result = await search(
        q="best italian restaurants in SF", engine="brave", llm_rewrite=True
    )

    # Capture your printed output.
    captured = capsys.readouterr().out
    # Print captured output for debugging (optional)
    print(captured)

    # Verify that your print statements were executed.
    assert "search tool is being called with q" in captured

    # Also, verify that the fake response was returned.
    assert "results" in result
    assert result["results"][0]["title"] == "Fake Italian Restaurant"


@pytest.mark.asyncio
async def test_search_skill_no_llm(monkeypatch):
    # Set dummy API key
    monkeypatch.setenv("BRAVE_API_KEY", "test_api_key")

    # Patch _rewrite_query to ensure it's not called
    async def fail_rewrite(query: str) -> str:
        raise Exception("_rewrite_query should not be called when llm_rewrite is False")

    monkeypatch.setattr("arsenalpy.skills.search_skill._rewrite_query", fail_rewrite)

    # Capture the query passed to _search_brave
    captured_query = None

    async def fake_search_brave(query: str, api_key: str):
        nonlocal captured_query
        captured_query = query
        return {
            "web": {
                "results": [
                    {
                        "title": "Fake Brave Result",
                        "url": "http://example.com",
                        "description": "Description",
                    }
                ]
            }
        }

    monkeypatch.setattr(
        "arsenalpy.skills.search_skill._search_brave", fake_search_brave
    )

    result = await search(q="test query no llm", engine="brave", llm_rewrite=False)

    # Verify that the original query was used and rewrite was skipped
    assert captured_query == "test query no llm"
    assert result["results"][0]["title"] == "Fake Brave Result"


@pytest.mark.asyncio
async def test_search_skill_missing_api_key(monkeypatch):
    # Ensure API key is not set
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)

    with pytest.raises(ValueError) as excinfo:
        await search(q="query", engine="brave", llm_rewrite=False)

    assert "BRAVE_API_KEY not set" in str(excinfo.value)


@pytest.mark.asyncio
async def test_search_skill_serper(monkeypatch):
    # Set dummy API key for SERPER
    monkeypatch.setenv("SERPER_API_KEY", "test_serper_key")

    async def fake_rewrite(query: str) -> str:
        return query + " optimized"

    monkeypatch.setattr("arsenalpy.skills.search_skill._rewrite_query", fake_rewrite)

    async def fake_search_serper(query: str, api_key: str):
        return {
            "organic": [
                {
                    "title": "Fake Serper Result",
                    "link": "http://example.com",
                    "snippet": "Description",
                }
            ]
        }

    monkeypatch.setattr(
        "arsenalpy.skills.search_skill._search_serper", fake_search_serper
    )

    result = await search(q="serper query", engine="serper", llm_rewrite=True)

    # Verify that the rewriting happened and SERPER engine was used
    assert result["results"][0]["title"] == "Fake Serper Result"


@pytest.mark.asyncio
async def test_search_skill_tavily(monkeypatch):
    # Set dummy API key for TAVILY
    monkeypatch.setenv("TAVILY_API_KEY", "test_tavily_key")

    async def fake_rewrite(query: str) -> str:
        return query

    monkeypatch.setattr("arsenalpy.skills.search_skill._rewrite_query", fake_rewrite)

    async def fake_search_tavily(query: str, api_key: str):
        return {
            "results": [
                {
                    "title": "Fake Tavily Result",
                    "url": "http://example.com",
                    "content": "Description",
                }
            ]
        }

    monkeypatch.setattr(
        "arsenalpy.skills.search_skill._search_tavily", fake_search_tavily
    )

    result = await search(q="tavily query", engine="tavily", llm_rewrite=True)

    assert result["results"][0]["title"] == "Fake Tavily Result"


@pytest.mark.asyncio
async def test_search_skill_rewrite_failure(monkeypatch, capsys):
    # Set dummy API key for BRAVE
    monkeypatch.setenv("BRAVE_API_KEY", "test_api_key")

    # Simulate a failing _rewrite_query that prints an error and returns the original query
    async def failing_rewrite(query: str) -> str:
        print("Query rewrite failed: simulated error", flush=True)
        return query

    monkeypatch.setattr("arsenalpy.skills.search_skill._rewrite_query", failing_rewrite)

    captured_query = None

    async def fake_search_brave(query: str, api_key: str):
        nonlocal captured_query
        captured_query = query
        return {
            "web": {
                "results": [
                    {
                        "title": "Fake Brave Result",
                        "url": "http://example.com",
                        "description": "Description",
                    }
                ]
            }
        }

    monkeypatch.setattr(
        "arsenalpy.skills.search_skill._search_brave", fake_search_brave
    )

    result = await search(q="original query", engine="brave", llm_rewrite=True)

    # Capture printed output
    captured = capsys.readouterr().out
    assert "Query rewrite failed: simulated error" in captured

    # Verify that since rewriting failed, the original query was passed to the search function
    assert captured_query == "original query"
    assert result["results"][0]["title"] == "Fake Brave Result"
