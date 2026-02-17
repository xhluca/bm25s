
import bm25s
from mcp.server.fastmcp import FastMCP

def create_mcp_server(index_dir: str, port: int = 8000):
    # Initialize FastMCP server
    mcp = FastMCP("bm25s")

    # Load the index
    print(f"Loading index from {index_dir}...")
    retriever = bm25s.BM25.load(index_dir, load_corpus=True)
    print("Index loaded successfully.")

    @mcp.tool()
    def retrieve(query: str, k: int = 10) -> str:
        """
        Retrieve documents from the index based on the query.
        
        Args:
            query: The search query string.
            k: The number of documents to retrieve.
        """
        # Tokenize the query
        query_tokens = bm25s.tokenize(query)
        
        # Retrieve documents
        results = retriever.retrieve(query_tokens, k=k)
        
        # Format results
        output = []
        for i, (doc, score) in enumerate(zip(results.documents[0], results.scores[0])):
            output.append(f"Rank {i+1} (Score: {score:.4f}):\n{doc}\n")
            
        return "\n".join(output)

    @mcp.tool()
    def get_info() -> str:
        """
        Get information about the loaded index.
        """
        return f"BM25S Index Info:\n- Vocab Size: {len(retriever.vocab_dict)}\n- Num Docs: {retriever.scores['num_docs']}\n- Backend: {retriever.backend}"

    return mcp

def main(index_dir: str, port: int = 8000):
    mcp = create_mcp_server(index_dir, port)
    mcp.run()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run BM25S MCP Server")
    parser.add_argument("--index-dir", required=True, help="Path to the BM25S index directory")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()
    
    main(args.index_dir, args.port)
