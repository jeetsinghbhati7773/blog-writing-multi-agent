from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import List, Dict, Any, Optional

import requests
from langchain_core.tools import tool, BaseTool

logger = logging.getLogger(__name__)

try:
    from langchain_community.tools import DuckDuckGoSearchRun
    search_tool = DuckDuckGoSearchRun(region="us-en")
except Exception as e:
    logger.warning("DuckDuckGo search tool unavailable: %s", e)
    search_tool = None


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage. The API key is read from the ALPHA_VANTAGE_API_KEY
    environment variable (falls back to Alpha Vantage's public "demo" key,
    which only works for a few sample symbols like 'IBM').
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        # Alpha Vantage returns 200 with a "Note"/"Information" message on
        # rate-limit or missing/invalid key rather than an HTTP error.
        if isinstance(data, dict) and not data.get("Global Quote") and (
            data.get("Note") or data.get("Information") or data.get("Error Message")
        ):
            return {
                "error": (
                    "Alpha Vantage did not return a quote (rate limit reached or "
                    "API key missing/invalid). Set ALPHA_VANTAGE_API_KEY in your .env."
                ),
                "raw": data,
            }
        return data
    except Exception as e:
        logger.warning("Stock price lookup failed for %r: %s", symbol, e)
        return {"error": str(e)}


# Dedicated async loop for background MCP tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


# Default MCP server configuration
DEFAULT_MCP_SERVERS = {
    "expense": {
        "transport": "streamable_http",
        "url": os.getenv("FASTMCP_URL", os.getenv("MCP_EXPENSE_URL", "https://splendid-gold-dingo.fastmcp.app/mcp")),
    }
}


def load_mcp_tools(server_config: Optional[Dict[str, Any]] = None) -> List[BaseTool]:
    """
    Loads tools from MCP servers using MultiServerMCPClient.
    Wraps async tools with sync execution support to prevent 'StructuredTool does not support sync invocation'.
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        config = server_config or DEFAULT_MCP_SERVERS
        
        async def _fetch_mcp_tools():
            client = MultiServerMCPClient(config)
            return await client.get_tools()

        raw_tools = run_async(_fetch_mcp_tools())

        wrapped_tools = []
        for t in raw_tools:
            if hasattr(t, "coroutine") and t.coroutine and not getattr(t, "func", None):
                coro = t.coroutine
                t.func = lambda *args, _c=coro, **kwargs: run_async(_c(*args, **kwargs))
            wrapped_tools.append(t)
        return wrapped_tools
    except Exception as e:
        logger.warning("Could not load MCP tools: %s", e)
        return []


def get_all_tools(include_mcp: bool = True) -> List[BaseTool]:
    """
    Returns a combined list of local tools (DuckDuckGo, Stock Price) and MCP tools.
    """
    tools = []
    if search_tool:
        tools.append(search_tool)
    tools.append(get_stock_price)

    if include_mcp:
        mcp_tools = load_mcp_tools()
        tools.extend(mcp_tools)

    return tools


def execute_tool(tool_obj: BaseTool, args: dict) -> Any:
    """
    Safely executes a LangChain tool supporting both sync .invoke() and async .ainvoke().
    Handles StructuredTool async requirements seamlessly.
    """
    try:
        return tool_obj.invoke(args)
    except Exception as e:
        logger.debug("Sync tool invoke failed (%s); retrying via async ainvoke", e)
        return run_async(tool_obj.ainvoke(args))

