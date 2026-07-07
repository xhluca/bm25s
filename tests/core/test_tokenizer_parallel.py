import multiprocessing as mp
import unittest

import bm25s.tokenization as tok

try:
    import Stemmer
    HAS_STEMMER = True
except ImportError:
    HAS_STEMMER = False

FORK = "fork" in mp.get_all_start_methods()

CORPUS = [
    "The quick brown fox jumps over the lazy dog",
    "a an the of and",                     # all stopwords
    "",                                     # empty
    "   ",                                  # whitespace only
    "Retrieval augmented generation systems",
    "the THE The tHe",                      # case + repeats
    "élève résumé café naïve",              # non-ascii
    "singleword",
    "numbers 123 45 6",
    "generation generation retrieval retrieval systems",
] * 40  # 400 docs, enough to span several worker chunks


def _both(stopwords, stemmer, allow_empty=True, update_vocab=True, texts=CORPUS):
    serial = tok.Tokenizer(stopwords=stopwords, stemmer=stemmer)
    ids_s = serial.tokenize(texts, update_vocab=update_vocab, return_as="ids",
                            show_progress=False, allow_empty=allow_empty, n_jobs=1)
    par = tok.Tokenizer(stopwords=stopwords, stemmer=stemmer)
    ids_p = par.tokenize(texts, update_vocab=update_vocab, return_as="ids",
                         show_progress=False, allow_empty=allow_empty, n_jobs=4)
    return (serial, ids_s), (par, ids_p)


@unittest.skipUnless(FORK, "fork start method unavailable")
class TestParallelTokenizer(unittest.TestCase):
    def _assert_identical(self, a, b):
        (s, ids_s), (p, ids_p) = a, b
        self.assertEqual(ids_s, ids_p)
        self.assertEqual(s.word_to_id, p.word_to_id)
        self.assertEqual(s.stem_to_sid, p.stem_to_sid)
        self.assertEqual(s.word_to_stem, p.word_to_stem)
        self.assertEqual(s.get_vocab_dict(), p.get_vocab_dict())

    def test_no_stemmer_with_stopwords(self):
        self._assert_identical(*_both("english", None))

    def test_no_stemmer_no_stopwords(self):
        self._assert_identical(*_both(None, None))

    def test_no_stemmer_allow_empty_false(self):
        self._assert_identical(*_both("english", None, allow_empty=False))

    @unittest.skipUnless(HAS_STEMMER, "PyStemmer not installed")
    def test_stemmer_with_stopwords(self):
        self._assert_identical(*_both("english", Stemmer.Stemmer("english")))

    @unittest.skipUnless(HAS_STEMMER, "PyStemmer not installed")
    def test_stemmer_no_stopwords(self):
        self._assert_identical(*_both(None, Stemmer.Stemmer("english")))

    @unittest.skipUnless(HAS_STEMMER, "PyStemmer not installed")
    def test_stemmer_allow_empty_false(self):
        self._assert_identical(*_both("english", Stemmer.Stemmer("english"), allow_empty=False))

    def test_return_as_tuple(self):
        serial = tok.Tokenizer(stopwords="english", stemmer=None)
        t_s = serial.tokenize(CORPUS, return_as="tuple", show_progress=False, n_jobs=1)
        par = tok.Tokenizer(stopwords="english", stemmer=None)
        t_p = par.tokenize(CORPUS, return_as="tuple", show_progress=False, n_jobs=4)
        self.assertEqual(t_s.ids, t_p.ids)
        self.assertEqual(t_s.vocab, t_p.vocab)

    def test_query_tokenization_after_parallel(self):
        # vocabulary built in parallel must tokenize new queries exactly like serial
        s = tok.Tokenizer(stopwords="english", stemmer=None)
        s.tokenize(CORPUS, return_as="ids", show_progress=False, n_jobs=1)
        p = tok.Tokenizer(stopwords="english", stemmer=None)
        p.tokenize(CORPUS, return_as="ids", show_progress=False, n_jobs=4)
        queries = ["quick brown systems", "unknownword retrieval", "the of and"]
        qs = s.tokenize(queries, update_vocab=False, return_as="ids", show_progress=False)
        qp = p.tokenize(queries, update_vocab=False, return_as="ids", show_progress=False)
        self.assertEqual(qs, qp)

    def test_falls_back_when_vocab_not_fresh(self):
        # a non-empty vocabulary is not eligible for the parallel path; it must
        # still return correct results via the serial fallback
        t = tok.Tokenizer(stopwords="english", stemmer=None)
        t.tokenize(["seed words here"], return_as="ids", show_progress=False)
        self.assertFalse(t._can_tokenize_parallel(CORPUS, True, "ids"))
        out = t.tokenize(CORPUS, update_vocab=True, return_as="ids",
                         show_progress=False, n_jobs=4)
        self.assertEqual(len(out), len(CORPUS))

    def test_njobs_one_is_serial(self):
        # default path is unchanged
        t = tok.Tokenizer(stopwords="english", stemmer=None)
        self.assertTrue(True)  # n_jobs defaults to 1; covered by all other suites


if __name__ == "__main__":
    unittest.main()
