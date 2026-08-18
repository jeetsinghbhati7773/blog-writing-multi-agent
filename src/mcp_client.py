from __future__ import annotations

import asyncio
import threading
from typing import List, Dict, Any, Optional

import requests
from langchain_core.tools import tool, BaseTool

try:
    from langchain_community.tools import DuckDuckGoSearchRun
    search_tool = DuckDuckGoSearchRun(region="us-en")
except Exception:
    search_tool = None


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    try:
        r = requests.get(url, timeout=10)
        return r.json()
    except Exception as e:
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
        "url": "https://splendid-gold-dingo.fastmcp.app/mcp",
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
        client = MultiServerMCPClient(config)
        raw_tools = run_async(client.get_tools())

        wrapped_tools = []
        for t in raw_tools:
            if hasattr(t, "coroutine") and t.coroutine and not getattr(t, "func", None):
                coro = t.coroutine
                t.func = lambda *args, _c=coro, **kwargs: run_async(_c(*args, **kwargs))
            wrapped_tools.append(t)
        return wrapped_tools
    except Exception as e:
        print(f"Warning: Could not load MCP tools: {e}")
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
    except Exception:
        return run_async(tool_obj.ainvoke(args))

