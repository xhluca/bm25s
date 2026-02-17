
import asyncio
from bm25s.mcp import create_mcp_server

async def test_server():
    index_dir = "bm25s_indices/nq"
    mcp = create_mcp_server(index_dir)
    
    print("Testing get_info tool...")
    info = await mcp.call_tool("get_info", arguments={})
    print(f"Info: {info}")
    
    print("\nTesting retrieve tool...")
    results = await mcp.call_tool("retrieve", arguments={"query": "what is the capital of France", "k": 2})
    print(f"Results: {results}")

if __name__ == "__main__":
    asyncio.run(test_server())
