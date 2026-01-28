# Bug Report: Critical Issue in token_ids Indexing Path (Version 0.3.0)

## Summary

A critical bug was identified in the `token_ids` indexing path of bm25s version 0.3.0. This bug prevented the use of non-sequential token IDs (e.g., [100, 200, 300]) and caused `IndexError` during indexing or retrieval.

## Root Cause

When indexing with raw token IDs (integers), the code had several interconnected issues:

1. **Incorrect vocab_dict Structure**: The vocab_dict was created as `{token_id: i for i, token_id in enumerate(unique_ids)}` which mapped original token IDs to sequential indices, but this was not used consistently.

2. **Missing Corpus Remapping**: The corpus_token_ids were passed directly to `build_index_from_ids` without remapping the original token IDs to sequential indices. The scoring functions expect sequential indices [0, 1, 2, ...] for array indexing.

3. **Incorrect unique_token_ids_set**: The set was populated with sequential indices instead of the original token IDs, causing query filtering to fail.

4. **Missing Query Remapping**: When queries came in with original token IDs, they were not remapped to sequential indices for scoring.

5. **Save/Load Issues**: The `_indexed_with_token_ids` flag was not persisted, and integer keys were lost during JSON serialization.

## Impact

- **Severity**: High
- **Symptoms**: 
  - IndexError when using non-sequential token IDs
  - Incorrect scoring and document frequencies
  - Failed retrieval after save/load
- **Affected Users**: Anyone using the token_ids indexing path with non-sequential IDs

## Fix

### Code Changes in `bm25s/__init__.py`:

1. **Line 486**: Added sorting of unique_ids for deterministic ordering:
   ```python
   vocab_dict = {token_id: i for i, token_id in enumerate(sorted(unique_ids))}
   ```

2. **Lines 488-497**: Added corpus remapping step:
   ```python
   corpus_token_ids = [
       [vocab_dict[token_id] for token_id in doc_ids]
       for doc_ids in tqdm(corpus_token_ids_original, ...)
   ]
   ```

3. **Lines 520-524**: Set unique_token_ids_set based on indexing path:
   ```python
   if inferred_corpus_obj == "token_ids":
       self.unique_token_ids_set = set(self.vocab_dict.keys())
       self._indexed_with_token_ids = True
   else:
       self.unique_token_ids_set = set(self.vocab_dict.values())
       self._indexed_with_token_ids = False
   ```

4. **Lines 755-757**: Added query remapping during retrieval:
   ```python
   if hasattr(self, '_indexed_with_token_ids') and self._indexed_with_token_ids:
       query_filtered = [self.vocab_dict[token_id] for token_id in query_filtered]
   ```

5. **Save/Load Fixes**:
   - Added `indexed_with_token_ids` to saved params (line 986)
   - Load the flag and convert string keys back to integers (lines 1164-1170)
   - Use the flag to correctly set unique_token_ids_set (lines 1175-1179)

### Test Updates:

1. **tests/core/test_vocab_dict.py**: Changed expected error from `IndexError` to `ValueError` (better error message)

2. **tests/core/test_tokenizer_misc.py**: Updated test_new_ids to accept tied results in any order

## Testing

- Created reproduction test demonstrating the bug with non-sequential IDs
- Verified fix works with both sequential and non-sequential IDs  
- Tested save/load functionality for token_ids path
- All 144 core tests pass (excluding 7 failures due to missing optional orjson dependency)
- CodeQL security scan: 0 alerts

## Recommendations

1. Users who have indexed data using the token_ids path with non-sequential IDs should re-index their data with the fixed version
2. Consider adding more integration tests for the token_ids path with various ID ranges
3. Document the token_ids path behavior more explicitly in the API documentation

## Files Modified

- `bm25s/__init__.py`: Core bug fixes
- `tests/core/test_vocab_dict.py`: Updated test expectations
- `tests/core/test_tokenizer_misc.py`: Updated test for tied results
