
import asyncio
from bm25s.mcp import create_mcp_server

async def test_server():
    index_dir = "test_index_mcp"
    mcp = create_mcp_server(index_dir)
    
    print("Testing get_info tool...")
    info = await mcp.call_tool("get_info", arguments={})
    print(f"Info: {info}")
    
    print("\nTesting retrieve tool...")
    results = await mcp.call_tool("retrieve", arguments={"query": "quick fox", "k": 2})
    print(f"Results: {results}")
    
    assert "The quick brown fox" in str(results)
    print("\nVerification successful!")

if __name__ == "__main__":
    asyncio.run(test_server())
