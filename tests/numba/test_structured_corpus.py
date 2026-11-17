import unittest

import bm25s
from tests.core import test_structured_corpus


@unittest.skipUnless(bm25s.NUMBA_AVAILABLE, "Numba is not installed")
class TestStructuredCorpusNumba(test_structured_corpus.TestStructuredCorpus):
    backend = "numba"
