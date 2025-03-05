# deep research agent - combines search and browse capabilities for crypto token analysis

import asyncio
import os
from datetime import datetime, timedelta

from arsenalpy.agents.agent import Agent, AgentConfig
from arsenalpy.skills.browse_skill import BrowseMode, browse
from arsenalpy.skills.search_skill import search
from dotenv import load_dotenv

load_dotenv()


async def main():
    # Get current date for time-relevant searches
    current_date = datetime.now()
    last_week = current_date - timedelta(days=7)
    date_str = current_date.strftime("%B %Y")
    week_str = f"{last_week.strftime('%d %B')} to {current_date.strftime('%d %B %Y')}"

    print(f"=== Deep Research: Solana (SOL) Weekly Token Trends ({week_str}) ===")
    # Demonstrate the deep research process

    print("\n> Step 1: Searching for current Solana token trends...")

    # First search for current Solana price and trends
    search_query1 = (
        f"Solana SOL current price analysis trend prediction {date_str} week"
    )
    print(f"\nSearch query 1: '{search_query1}'")
    search_results1 = await search(q=search_query1, rerank=True, engine="brave")

    # Second search for Solana ecosystem tokens and projects
    search_query2 = (
        f"top performing Solana ecosystem tokens projects {date_str} weekly analysis"
    )
    print(f"\nSearch query 2: '{search_query2}'")
    search_results2 = await search(q=search_query2, rerank=True, engine="brave")

    # Third search for technical indicators and on-chain data
    search_query3 = (
        f"Solana SOL technical analysis on-chain data trading volume {week_str}"
    )
    print(f"\nSearch query 3: '{search_query3}'")
    search_results3 = await search(q=search_query3, rerank=True, engine="brave")

    # Print top refined search results
    print(
        f"\nFound {len(search_results1['results'])} results for current price analysis."
    )
    print(f"Found {len(search_results2['results'])} results for ecosystem tokens.")
    print(f"Found {len(search_results3['results'])} results for technical analysis.")

    # Display sample of results
    for i, result in enumerate(search_results1["results"][:3], 1):
        print(f"{i}. {result['title']} | Score: {result['relevance_score']:.4f}")
        print(f"   Link: {result['link']}")
        print(
            f"   Snippet: {result['snippet'][:100]}..."
            if result["snippet"]
            else "   Snippet: [EMPTY]"
        )

    # Combine results from all searches, prioritizing by relevance score
    all_results = (
        search_results1["results"]
        + search_results2["results"]
        + search_results3["results"]
    )

    # Remove duplicates based on URL
    unique_results = []
    seen_urls = set()
    for result in sorted(
        all_results, key=lambda x: x.get("relevance_score", 0), reverse=True
    ):
        if result["link"] not in seen_urls:
            unique_results.append(result)
            seen_urls.add(result["link"])

    # Step 2: Browse top results to get full content
    print("\n> Step 2: Browsing top results to extract full content...")

    browse_results = []
    # Browse top results, focusing on crypto analysis sites, exchanges, and major publications
    for i, result in enumerate(unique_results[:10], 1):
        url = result["link"]
        print(f"\nBrowsing result {i}: {result['title']}")

        try:
            browse_result = await browse(
                url=url,
                mode=BrowseMode.DYNAMIC,  # Use dynamic mode for JS-heavy crypto sites
                extract_links=True,
                content_threshold=0.4,  # Lower threshold to keep more content
                visible_browser=True,  # Show the browser window during browsing
            )

            if browse_result["success"]:
                browse_results.append(browse_result)
                content_preview = (
                    browse_result["content"][:150] + "..."
                    if browse_result["content"]
                    else "[EMPTY]"
                )
                print(f"✓ Successfully browsed {url}")
                print(f"  Title: {browse_result['title']}")
                print(f"  Content length: {len(browse_result['content'])} chars")
                print(f"  Content preview: {content_preview}")
                if browse_result["links"]:
                    print(f"  Found {len(browse_result['links'])} links")
            else:
                print(f"✗ Failed to browse {url}: {browse_result['error']}")
        except Exception as e:
            print(f"✗ Error browsing {url}: {str(e)}")

    # If we didn't get any successful browse results, try again with basic mode
    if not browse_results:
        print("\n> No successful results with dynamic mode. Trying with static mode...")
        for i, result in enumerate(unique_results[:3], 1):
            url = result["link"]
            print(f"\nRetrying with static mode - Result {i}: {result['title']}")

            try:
                browse_result = await browse(
                    url=url,
                    mode=BrowseMode.STATIC,  # Try with static mode
                    extract_links=True,
                    content_threshold=0.3,  # Even lower threshold
                    visible_browser=True,  # Show the browser window
                )

                if browse_result["success"] and browse_result["content"]:
                    browse_results.append(browse_result)
                    content_preview = (
                        browse_result["content"][:150] + "..."
                        if browse_result["content"]
                        else "[EMPTY]"
                    )
                    print(f"✓ Successfully browsed {url} with static mode")
                    print(f"  Title: {browse_result['title']}")
                    print(f"  Content length: {len(browse_result['content'])} chars")
                    print(f"  Content preview: {content_preview}")
                    if browse_result["links"]:
                        print(f"  Found {len(browse_result['links'])} links")
            except Exception as e:
                print(f"✗ Error browsing {url} with static mode: {str(e)}")

    # Step 3: Analyze all gathered information
    print("\n> Step 3: Analyzing all gathered information...")

    # Create a comprehensive prompt with all gathered information
    analysis_prompt = f"""I've collected the latest information about Solana (SOL) token trends for the week of {week_str}. Here's what I found:

# SEARCH RESULTS:
"""

    # Add search results from all queries
    for i, result in enumerate(unique_results[:15], 1):
        analysis_prompt += f"## Result {i}: {result['title']}\n"
        analysis_prompt += f"URL: {result['link']}\n"
        analysis_prompt += f"Snippet: {result['snippet'] if result['snippet'] else '[No snippet available]'}\n\n"

    # Add browsed content
    analysis_prompt += "# DETAILED CONTENT FROM BROWSED PAGES:\n"
    for i, result in enumerate(browse_results, 1):
        analysis_prompt += f"## Page {i}: {result['title']}\n"
        analysis_prompt += f"URL: {result['url']}\n\n"
        analysis_prompt += (
            result["content"][:12000] + "...\n\n"
            if len(result["content"]) > 12000
            else result["content"] + "\n\n"
        )

    # Add the specific analysis request focused on crypto/token trends
    analysis_prompt += f"""
Based on ALL the information above, provide a detailed analysis of current Solana (SOL) token trends for the week of {week_str}, specifically addressing:

1. PRICE ANALYSIS & MARKET MOVEMENTS:
   - Current price and significant price movements during this week
   - Key support and resistance levels with exact price points
   - Trading volume trends and notable changes (provide specific numbers)
   - Market sentiment indicators and how they've shifted this week
   - Comparison to other major cryptocurrencies (BTC, ETH) this week

2. TECHNICAL INDICATORS & CHART PATTERNS:
   - Specific technical indicators (RSI, MACD, etc.) with current values and interpretations
   - Identified chart patterns and what they suggest for short-term price movement
   - Key moving averages (50-day, 200-day) and current position relative to them
   - Volume profile analysis and what it indicates for price action
   - On-chain metrics and what they reveal about network usage and token velocity

3. ECOSYSTEM DEVELOPMENTS:
   - New project launches or updates in the Solana ecosystem this week
   - DeFi, NFT, or other sector-specific developments affecting token prices
   - Top performing tokens in the Solana ecosystem this week (with % gains)
   - Notable funding, partnerships, or integrations announced this week
   - Developer activity metrics and how they're trending

4. ACTIONABLE TRADING INSIGHTS:
   - Short-term price predictions based on technical and fundamental analysis
   - Key price levels to watch for entry/exit positions
   - Risk assessment for current SOL positions
   - Trading strategies appropriate for the current market conditions
   - Recommendations for different trader types (day traders vs. long-term holders)

5. CATALYSTS & UPCOMING EVENTS:
   - Upcoming protocol upgrades, hard forks, or technical implementations
   - Regulatory developments that could impact SOL prices
   - Scheduled events, conferences, or announcements to watch
   - Potential risks or threats to the current price trend
   - Macro factors influencing the broader crypto market that could affect SOL

Format your analysis with clear sections, include specific numbers and data points, and provide information that would be immediately useful for traders and investors. Focus on actionable insights rather than general observations. Be sure to note when information might be time-sensitive or speculative.

IMPORTANT: Clearly separate fact from speculation, and provide confidence levels for any predictions or forecasts you make.
"""

    # If we have no content to analyze, provide a message
    if not browse_results:
        print("\n⚠️ Warning: No content was successfully browsed from any URLs.")
        print("Please check your network connection and try again.")
        print(
            "You may also need to install or update Crawl4AI: pip install -U crawl4ai"
        )
        return

    # Create content analysis agent
    analysis_agent = Agent(
        name="crypto_analyst",
        provider="openrouter",
        system_prompt="""You are an expert cryptocurrency analyst specializing in token trends, price analysis, and trading strategies.""",
        model="anthropic/claude-3.7-sonnet",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        config=AgentConfig(temperature=0.2, max_tokens=4000),
    )

    print(f"\n=== Solana (SOL) Weekly Token Analysis: {week_str} ===\n")
    async for chunk in analysis_agent.do_stream(prompt=analysis_prompt):
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
