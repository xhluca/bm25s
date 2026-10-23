import tempfile
import unittest

import numpy as np

import bm25s
from bm25s import selection as selection_np
from bm25s.tokenization import Tokenized


class TestBM25SEmptyCorpus(unittest.TestCase):
    """Indexing an empty corpus (no documents, or only token-less documents).

    Previously this crashed in ``index`` with ``ValueError: max() iterable
    argument is empty`` and emitted ``RuntimeWarning: Mean of empty slice`` /
    ``invalid value encountered in scalar divide``. It should now build a valid
    empty index instead.
    """

    def assert_empty_index_well_formed(
        self, retriever: "bm25s.BM25", num_docs: int
    ) -> None:
        self.assertEqual(retriever.scores["num_docs"], num_docs)
        # The empty token is seeded at id 0 so token-less queries have a target.
        self.assertEqual(retriever.vocab_dict.get(""), 0)
        # The scored matrix has zero populated entries.
        self.assertEqual(retriever.scores["data"].size, 0)

    def test_index_no_documents_does_not_crash(self) -> None:
        retriever = bm25s.BM25(backend="numpy")
        with np.errstate(all="raise"):  # no NaN/divide warnings allowed
            retriever.index([], show_progress=False)
        self.assert_empty_index_well_formed(retriever, num_docs=0)

    def test_index_only_token_less_documents_does_not_crash(self) -> None:
        retriever = bm25s.BM25(backend="numpy")
        with np.errstate(all="raise"):
            retriever.index([[], [], []], show_progress=False)
        self.assert_empty_index_well_formed(retriever, num_docs=3)

    def test_retrieve_from_empty_corpus_raises_clear_error(self) -> None:
        # k must not exceed the (zero) document count; the existing guard gives
        # a clear, actionable ValueError rather than an internal IndexError.
        retriever = bm25s.BM25(backend="numpy")
        retriever.index([], show_progress=False)
        with self.assertRaises(ValueError):
            retriever.retrieve(
                bm25s.tokenize("foo", show_progress=False), k=1, show_progress=False
            )

    def test_topk_of_empty_scores_returns_empty(self) -> None:
        # An empty score vector must yield aligned empty results, not
        # IndexError: cannot do a non-empty take from an empty axes.
        empty = np.empty(0, dtype=np.float32)
        scores, indices = selection_np.topk(empty, k=3, backend="numpy", sorted=True)
        self.assertEqual(scores.shape, (0,))
        self.assertEqual(indices.shape, (0,))

    def test_token_less_corpus_serves_zero_scores(self) -> None:
        # Regression: a token-less corpus must reserve a real zero-score column
        # for the seeded empty token, so an ordinary (unmatched) query does not
        # trip the "max token ID higher than the number of tokens" guard and
        # instead returns every document with a zero score. Each backend is
        # instantiated natively (selecting numba via backend_selection on a
        # numpy retriever exercises a different path).
        backends = ["numpy"]
        if bm25s.NUMBA_AVAILABLE:
            backends.append("numba")
        for backend in backends:
            retriever = bm25s.BM25(backend=backend)
            retriever.index([[], [], []], show_progress=False)
            # The empty token now owns column 0: indptr has one column (len 2).
            self.assertEqual(len(retriever.scores["indptr"]), 2)
            docs, scores = retriever.retrieve(
                [["hello"]], k=2, backend_selection=backend, show_progress=False
            )
            self.assertEqual(scores.shape, (1, 2))
            np.testing.assert_array_equal(scores, 0.0)
            # An empty-token query behaves the same way.
            docs, scores = retriever.retrieve(
                [[""]], k=2, backend_selection=backend, show_progress=False
            )
            np.testing.assert_array_equal(scores, 0.0)

    def test_old_signature_subclass_non_empty_still_works(self) -> None:
        # B1: index() must not pass create_empty_token to an override that keeps
        # the old (corpus_tokens, show_progress, leave_progress) signature; a
        # non-empty corpus must index and retrieve without TypeError.
        class CustomBM25(bm25s.BM25):
            def build_index_from_tokens(
                self, corpus_tokens, show_progress=True, leave_progress=False
            ):
                return super().build_index_from_tokens(
                    corpus_tokens, show_progress, leave_progress
                )

        retriever = CustomBM25(backend="numpy")
        retriever.index([["hello", "world"]], show_progress=False)
        docs, scores = retriever.retrieve(
            [["hello"]], k=1, backend_selection="numpy", show_progress=False
        )
        self.assertEqual(scores.shape, (1, 1))
        self.assertGreater(scores[0, 0], 0.0)

    def test_old_signature_subclass_empty_token_setting(self) -> None:
        # B1: the old-signature subclass must accept both empty-token settings
        # on a token-less corpus without an unexpected-kwarg TypeError.
        class CustomBM25(bm25s.BM25):
            def build_index_from_tokens(
                self, corpus_tokens, show_progress=True, leave_progress=False
            ):
                return super().build_index_from_tokens(
                    corpus_tokens, show_progress, leave_progress
                )

        for create_empty_token in (True, False):
            retriever = CustomBM25(backend="numpy")
            retriever.index(
                [[], [], []],
                create_empty_token=create_empty_token,
                show_progress=False,
            )

    def test_policy_attribute_is_cleared_after_index(self) -> None:
        # B1 hygiene: the private policy attribute must not leak past index(); after
        # a create_empty_token=True call it is cleared, and an explicit
        # create_empty_token=False index() does not inherit the seeding policy.
        # (index() itself defaults to create_empty_token=True.)
        retriever = bm25s.BM25(backend="numpy")
        retriever.index([[], [], []], create_empty_token=True, show_progress=False)
        self.assertNotIn("_create_empty_token", retriever.__dict__)
        retriever.index(
            [["hello", "world"]], create_empty_token=False, show_progress=False
        )
        self.assertNotIn("", retriever.vocab_dict)
        self.assertNotIn("_create_empty_token", retriever.__dict__)

    def test_policy_attribute_is_cleared_when_build_raises(self) -> None:
        # B1 hygiene: an exception inside build_index_from_tokens must still leave
        # no policy attribute behind (finally restores/cleans state).
        class Boom(bm25s.BM25):
            def build_index_from_tokens(self, corpus_tokens, show_progress=True, leave_progress=False):
                raise RuntimeError("boom")

        retriever = Boom(backend="numpy")
        with self.assertRaises(RuntimeError):
            retriever.index([["hello"]], create_empty_token=True, show_progress=False)
        self.assertNotIn("_create_empty_token", retriever.__dict__)

    def test_index_restores_preexisting_policy_value(self) -> None:
        # B1 reentrancy/restore: when a policy attribute already exists (e.g. an
        # outer caller's active setting), index(create_empty_token=False) must restore
        # that prior value on exit rather than blindly resetting it to False.
        retriever = bm25s.BM25(backend="numpy")
        retriever._create_empty_token = True  # simulate an ambient/outer policy
        retriever.index([["x"]], create_empty_token=False, show_progress=False)
        self.assertIs(retriever._create_empty_token, True)  # prior value restored
        del retriever._create_empty_token  # leave no test residue

    def test_direct_low_level_call_keeps_no_seed_default(self) -> None:
        # The low-level API called directly has no policy attribute, so the
        # historical no-seeding default (getattr False) must be preserved; it
        # returns (scores, vocab_dict) rather than storing vocab_dict on self.
        retriever = bm25s.BM25(backend="numpy")
        _, vocab_dict = retriever.build_index_from_tokens(
            [["hello", "world"]], show_progress=False
        )
        self.assertNotIn("", vocab_dict)

    def test_fresh_tokenized_empty_vocab_builds_real_column(self) -> None:
        # B2: a fresh Tokenized starting from vocab={} must reserve the empty
        # token column BEFORE matrix construction; indptr must have one column
        # instead of the orphaned zero-column state (indptr == [0]).
        retriever = bm25s.BM25(backend="numpy")
        retriever.index(Tokenized(ids=[[], [], []], vocab={}), show_progress=False)
        self.assertEqual(len(retriever.scores["indptr"]), 2)
        self.assertEqual(retriever.vocab_dict.get(""), 0)
        docs, scores = retriever.retrieve(
            [["hello"]], k=2, backend_selection="numpy", show_progress=False
        )
        self.assertEqual(scores.shape, (1, 2))
        np.testing.assert_array_equal(scores, 0.0)
        docs, scores = retriever.retrieve(
            [[""]], k=2, backend_selection="numpy", show_progress=False
        )
        np.testing.assert_array_equal(scores, 0.0)

    def test_tuple_empty_vocab_builds_real_column(self) -> None:
        # B2: the tuple form (corpus_token_ids, vocab_dict) with an empty vocab
        # has the same defect and the same fix.
        retriever = bm25s.BM25(backend="numpy")
        retriever.index(([[], [], []], {}), show_progress=False)
        self.assertEqual(len(retriever.scores["indptr"]), 2)
        docs, scores = retriever.retrieve(
            [["hello"]], k=2, backend_selection="numpy", show_progress=False
        )
        self.assertEqual(scores.shape, (1, 2))
        np.testing.assert_array_equal(scores, 0.0)

    def test_native_numba_tokenized_and_tuple_empty_vocab(self) -> None:
        # B2: the native numba backend must also build a valid column; selecting
        # numba via backend_selection on a numpy retriever tests a different path.
        if not bm25s.NUMBA_AVAILABLE:
            self.skipTest("numba is not installed")
        for corpus in (
            Tokenized(ids=[[], [], []], vocab={}),
            ([[], [], []], {}),
        ):
            retriever = bm25s.BM25(backend="numba")
            retriever.index(corpus, show_progress=False)
            docs, scores = retriever.retrieve(
                [[""]], k=2, backend_selection="numba", show_progress=False
            )
            self.assertEqual(scores.shape, (1, 2))
            np.testing.assert_array_equal(scores, 0.0)

    def test_repeated_indexing_fresh_vocab_is_stable(self) -> None:
        # B2: indexing the same initially-empty Tokenized twice must not widen
        # indptr. Start from vocab={} (a pre-seeded {"": 0} hides this failure).
        tok = Tokenized(ids=[[], [], []], vocab={})
        retriever = bm25s.BM25(backend="numpy")
        retriever.index(tok, show_progress=False)
        indptr_first = retriever.scores["indptr"].copy()
        retriever.index(tok, show_progress=False)
        np.testing.assert_array_equal(retriever.scores["indptr"], indptr_first)
        self.assertEqual(retriever.vocab_dict, {"": 0})

    def test_token_less_corpus_survives_eager_and_mmap_roundtrip(self) -> None:
        retriever = bm25s.BM25(backend="numpy")
        retriever.index([[], [], []], show_progress=False)
        save_dir = tempfile.mkdtemp()
        retriever.save(save_dir)
        for mmap in (False, True):
            loaded = bm25s.BM25.load(save_dir, mmap=mmap)
            docs, scores = loaded.retrieve(
                [["hello"]], k=2, backend_selection="numpy", show_progress=False
            )
            self.assertEqual(scores.shape, (1, 2))
            np.testing.assert_array_equal(scores, 0.0)

    def test_fresh_tokenized_empty_survives_roundtrip(self) -> None:
        # B2: the malformed Tokenized state must not be preserved across
        # eager/mmap save/load; the reserved column survives serialization.
        for mmap in (False, True):
            retriever = bm25s.BM25(backend="numpy")
            retriever.index(Tokenized(ids=[[], [], []], vocab={}), show_progress=False)
            save_dir = tempfile.mkdtemp()
            retriever.save(save_dir)
            loaded = bm25s.BM25.load(save_dir, mmap=mmap)
            docs, scores = loaded.retrieve(
                [["hello"]], k=2, backend_selection="numpy", show_progress=False
            )
            self.assertEqual(scores.shape, (1, 2))
            np.testing.assert_array_equal(scores, 0.0)

    def test_real_corpus_roundtrip_load_corpus(self) -> None:
        # A real corpus saved with the documents and reloaded with
        # load_corpus=True must still rank the matching document (eager and mmap).
        corpus = [
            "hello world foo",
            "bar hello again",
            "world hello today",
        ]
        tokenized = bm25s.tokenize(corpus, stopwords="en", show_progress=False)
        retriever = bm25s.BM25(corpus=corpus, backend="numpy")
        retriever.index(tokenized, show_progress=False)
        save_dir = tempfile.mkdtemp()
        retriever.save(save_dir, corpus=corpus)
        for mmap in (False, True):
            loaded = bm25s.BM25.load(save_dir, mmap=mmap, load_corpus=True)
            docs, scores = loaded.retrieve(
                [["hello"]], k=2, backend_selection="numpy", show_progress=False
            )
            self.assertEqual(scores.shape, (1, 2))
            self.assertGreater(scores[0, 0], 0.0)
            self.assertIn(docs[0, 0]["id"], (1, 2))

    def test_token_less_corpus_nonoccurrence_is_consistent(self) -> None:
        # BM25L / BM25+ build a nonoccurrence array; it must line up with the
        # single reserved column rather than be empty.
        for method in ("bm25l", "bm25+"):
            retriever = bm25s.BM25(method=method, backend="numpy")
            retriever.index([[], [], []], show_progress=False)
            self.assertEqual(len(retriever.nonoccurrence_array), 1)
            docs, scores = retriever.retrieve(
                [["hello"]], k=1, backend_selection="numpy", show_progress=False
            )
            np.testing.assert_array_equal(scores, 0.0)


if __name__ == "__main__":
    unittest.main()
