import unittest
import numpy as np
import Stemmer
import bm25s


class TestBM25SAllowEmpty(unittest.TestCase):
    def test_functional_tokenizer_indexes_blank_documents(self):
        for return_ids in (True, False):
            for stemmer in (None, Stemmer.Stemmer("english")):
                for corpus in (["", " \t\n", "!!!"], ["", "cats", "dogs"]):
                    with self.subTest(
                        return_ids=return_ids, stemmer=stemmer, corpus=corpus
                    ):
                        tokens = bm25s.tokenize(
                            corpus,
                            stemmer=stemmer,
                            return_ids=return_ids,
                            show_progress=False,
                        )
                        retriever = bm25s.BM25(backend="numpy")
                        retriever.index(tokens, show_progress=False)
                        query = bm25s.tokenize(
                            "cats",
                            stemmer=stemmer,
                            return_ids=False,
                            show_progress=False,
                        )[0]
                        scores = retriever.get_scores(query)
                        self.assertEqual(len(scores), len(corpus))
                        self.assertEqual(scores[0], 0)
                        self.assertEqual(scores[2], 0)
                        if corpus[1] == "cats":
                            self.assertGreater(scores[1], 0)
                        else:
                            self.assertEqual(scores[1], 0)

    def test_simple(self):
        all_scores = []

        for allow_empty in [(True, True), (True, False), (False, True), (False, False)]:
            corpus = ['foo', 'dog', 'baz', 'quick god', 'quick fox']
            query = 'quick'
            tokenizer = bm25s.tokenization.Tokenizer(stopwords=["english",],)
            corpus_tokens = tokenizer.tokenize(corpus, show_progress=False, allow_empty=allow_empty[0], return_as="ids")
            retriever = bm25s.BM25(backend="numpy")
            retriever.index(corpus_tokens, show_progress=False)
            query_tokens = tokenizer.tokenize([query], show_progress=False, allow_empty=allow_empty[1], return_as="ids")

            results, scores = retriever.retrieve(query_tokens, k=len(corpus), show_progress=False, n_threads=1, sorted=True)
            all_scores.append(scores)
        
        # Check that the scores are same for both allow_empty=True and allow_empty=False
        # self.assertTrue(np.array_equal(all_scores[0], all_scores[1]), "Scores should be the same for allow_empty=True and allow_empty=False")
        # assert all equals
        for s in all_scores[1:]:
            self.assertTrue(np.array_equal(all_scores[0], s), f"Scores should be the same for allow_empty={allow_empty[0]} and allow_empty={s}")


if __name__ == "__main__":
    unittest.main()
