import bm25s.high_level as bm25
import os

def main():
    # We will use the dummy files we created in tests/data
    data_dir = "tests/data"
    
    # 1. Test loading TXT
    # Each line is treated as a document
    print("--- Testing TXT load ---")
    txt_path = os.path.join(data_dir, "dummy.txt")
    # This creates a BM25Search object and indexes the file immediately
    corpus = bm25.load(txt_path)
    retriever = bm25.index(corpus)
    
    query = "test"
    results = retriever.search([query], k=2)
    print(f"Query: '{query}'")
    print(f"Result: {results[0][0]}")
    
    # Verify correct document retrieved
    assert "test" in results[0][0]["document"]

    # 2. Test loading CSV
    # We specify which column contains the document text
    print("\n--- Testing CSV load ---")
    csv_path = os.path.join(data_dir, "dummy.csv")
    corpus = bm25.load(csv_path, document_column="text")
    retriever = bm25.index(corpus)
    
    query = "fast"
    results = retriever.search([query], k=1)
    print(f"Query: '{query}'")
    print(f"Result: {results[0][0]}")
    
    assert "fast" in results[0][0]["document"]

    # 3. Test loading JSONL
    # Each line is a JSON object
    print("\n--- Testing JSONL load ---")
    jsonl_path = os.path.join(data_dir, "dummy.jsonl")
    corpus = bm25.load(jsonl_path, document_column="text")
    retriever = bm25.index(corpus)
    
    query = "world"
    results = retriever.search([query], k=1)
    print(f"Query: '{query}'")
    print(f"Result: {results[0][0]}")
    
    assert "world" in results[0][0]["document"]

    print("\nAll tests passed!")

if __name__ == "__main__":
    main()
