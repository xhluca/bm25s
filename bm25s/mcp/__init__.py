try:
    import mcp
except ImportError:
    raise ImportError("MCP is not installed, which is required for the MCP server in BM25S. Please install it with 'pip install bm25s[mcp]' or `pip install mcp`.")

from . import server

__all__ = ["server"]