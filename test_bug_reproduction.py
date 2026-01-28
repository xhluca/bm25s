#!/usr/bin/env python
"""
Test to reproduce the bug in token_ids indexing path.

Bug: When indexing with raw token IDs (integers that are NOT sequential from 0),
the vocab_dict is created incorrectly, causing an IndexError.

Location: bm25s/__init__.py, line 484
"""
import bm25s


def test_token_ids_bug_with_non_sequential_ids():
    """
    Test that demonstrates the bug when using non-sequential token IDs.
    
    This should work but currently fails with IndexError.
    """
    print("Testing token_ids indexing with non-sequential IDs...")
    
    # Use token IDs that are NOT sequential from 0
    # This is a realistic scenario when using pre-assigned token IDs
    corpus_token_ids = [[100, 200, 300], [200, 300, 400], [100, 400, 500]]
    query_token_ids = [[100, 200]]
    
    retriever = bm25s.BM25()
    
    try:
        # This should work but currently fails
        retriever.index(corpus_token_ids, create_empty_token=False)
        print("✓ Indexing successful")
        
        print(f"vocab_dict: {retriever.vocab_dict}")
        print(f"unique_token_ids_set: {retriever.unique_token_ids_set}")
        
        # Verify vocab_dict structure
        # For token_ids path, it should map {token_id: sequential_index}
        # OR {sequential_index: token_id} depending on design
        # But the values() should represent the actual token IDs used in scoring
        
        # Try to retrieve
        results = retriever.retrieve(query_token_ids, k=2)
        print(f"✓ Retrieval successful")
        print(f"Results: {results.documents}")
        print(f"Scores: {results.scores}")
        
    except Exception as e:
        print(f"✗ FAILED with error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_token_ids_with_sequential_ids():
    """
    Test that works with sequential IDs starting from 0.
    This currently works but may produce incorrect results.
    """
    print("\nTesting token_ids indexing with sequential IDs (0, 1, 2...)...")
    
    # Use token IDs that ARE sequential from 0
    corpus_token_ids = [[0, 1, 2], [1, 2, 3], [0, 3, 4]]
    query_token_ids = [[0, 1]]
    
    retriever = bm25s.BM25()
    
    try:
        retriever.index(corpus_token_ids, create_empty_token=False)
        print("✓ Indexing successful")
        
        print(f"vocab_dict: {retriever.vocab_dict}")
        
        results = retriever.retrieve(query_token_ids, k=2)
        print(f"✓ Retrieval successful")
        print(f"Results: {results.documents}")
        print(f"Scores: {results.scores}")
        
    except Exception as e:
        print(f"✗ FAILED with error: {type(e).__name__}: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("BM25S Token IDs Bug Reproduction Test")
    print("="*60)
    
    # This test should pass after fix
    test1_passed = test_token_ids_bug_with_non_sequential_ids()
    
    # This test currently passes but may have incorrect behavior
    test2_passed = test_token_ids_with_sequential_ids()
    
    print("\n" + "="*60)
    print("Summary:")
    print(f"Test 1 (non-sequential IDs): {'PASS' if test1_passed else 'FAIL'}")
    print(f"Test 2 (sequential IDs): {'PASS' if test2_passed else 'FAIL'}")
    print("="*60)
