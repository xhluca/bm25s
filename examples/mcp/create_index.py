
import bm25s
import os

def create_test_index():
    corpus = [
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumps over the lazy dog",
        "Hello world",
        "This is a test document for bm25s mcp server"
    ]
    
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus))
    
    save_dir = "test_index_mcp"
    retriever.save(save_dir, corpus=corpus)
    print(f"Index saved to {save_dir}")

if __name__ == "__main__":
    create_test_index()
