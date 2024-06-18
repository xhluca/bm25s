import json
import os
import time
import unittest
from pathlib import Path
import warnings
import logging

import numpy as np
import rank_bm25
from beir.datasets.data_loader import GenericDataLoader
from beir.util import download_and_unzip
import Stemmer
from transformers import AutoTokenizer
import bm25_pt

import bm25s
import bm25s.hf


# Make sure to import or define the functions/classes you're going to use,
# such as bm25s.skl_tokenize and the bm25s.BM25 class, among others.
def save_scores(scores, artifact_dir="tests/artifacts"):
    if os.getenv("ARTIFACTS_DIR"):
        artifacts_dir = Path(os.getenv("BM25_ARTIFACTS_DIR"))
    elif artifact_dir is not None:
        artifacts_dir = Path(artifact_dir)
    else:
        artifacts_dir = Path(__file__).parent / "artifacts"

    if "dataset" not in scores:
        raise ValueError("scores must contain a 'dataset' key.")
    if "model" not in scores:
        raise ValueError("scores must contain a 'model' key.")
    
    artifacts_dir = artifacts_dir / scores["model"]
    artifacts_dir.mkdir(exist_ok=True, parents=True)

    filename = f"{scores['dataset']}-{os.urandom(8).hex()}.json"
    with open(artifacts_dir / filename, "w") as f:
        json.dump(scores, f, indent=2)


class BM25TestCase(unittest.TestCase):
    def compare_with_rank_bm25(
        self,
        dataset,
        artifact_dir="tests/artifacts",
        rel_save_dir="datasets",
        corpus_subsample=None,
        queries_subsample=None,
        method="rank",
    ):
        warnings.filterwarnings("ignore", category=ResourceWarning)

        if method not in ["rank", "bm25+", "bm25l"]:
            raise ValueError("method must be either 'rank' or 'bm25+'.")

        # Download and prepare dataset
        base_url = (
            "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
        )
        url = base_url.format(dataset)
        out_dir = Path(__file__).parent / rel_save_dir
        data_path = download_and_unzip(url, str(out_dir))

        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(
            split="test"
        )

        # Convert corpus and queries to lists
        corpus_lst = [val["title"] + " " + val["text"] for val in corpus.values()]
        queries_lst = list(queries.values())

        if corpus_subsample is not None:
            corpus_lst = corpus_lst[:corpus_subsample]

        if queries_subsample is not None:
            queries_lst = queries_lst[:queries_subsample]

        # Tokenize using sklearn-style tokenizer + PyStemmer
        stemmer = Stemmer.Stemmer("english")

        corpus_token_strs = bm25s.tokenize(
            corpus_lst, stopwords="en", stemmer=stemmer, return_ids=False
        )
        queries_token_strs = bm25s.tokenize(
            queries_lst, stopwords="en", stemmer=stemmer, return_ids=False
        )
        print()
        print(f"Dataset:              {dataset}\n")
        # print corpus and queries size
        print(f"Corpus size:          {len(corpus_lst)}")
        print(f"Queries size:         {len(queries_lst)}")
        print()

        # Initialize and index bm25s with atire + robertson idf (to match rank-bm25)
        if method == "rank":
            bm25_sparse = bm25s.BM25(k1=1.5, b=0.75, method="atire", idf_method="robertson")
        elif method in ["bm25+", "bm25l"]:
            bm25_sparse = bm25s.BM25(k1=1.5, b=0.75, delta=0.5, method=method)
        else:
            raise ValueError("invalid method")
        
        start_time = time.time()
        bm25_sparse.index(corpus_token_strs)
        bm25_sparse_index_time = time.time() - start_time
        print(f"bm25s index time:     {bm25_sparse_index_time:.4f}s")

        # Scoring with bm25-sparse
        start_time = time.time()
        bm25_sparse_scores = [bm25_sparse.get_scores(q) for q in queries_token_strs]
        bm25_sparse_score_time = time.time() - start_time
        print(f"bm25s score time:     {bm25_sparse_score_time:.4f}s")

        # Initialize and index rank-bm25
        start_time = time.time()
        if method == "rank":
            bm25_rank = rank_bm25.BM25Okapi(corpus_token_strs, k1=1.5, b=0.75, epsilon=0.0)
        elif method == "bm25+":
            bm25_rank = rank_bm25.BM25Plus(corpus_token_strs, k1=1.5, b=0.75, delta=0.5)
        elif method == "bm25l":
            bm25_rank = rank_bm25.BM25L(corpus_token_strs, k1=1.5, b=0.75, delta=0.5)
        else:
            raise ValueError("invalid method")
    
        bm25_rank_index_time = time.time() - start_time
        print(f"rank-bm25 index time: {bm25_rank_index_time:.4f}s")

        # Scoring with rank-bm25
        start_time = time.time()
        bm25_rank_scores = [bm25_rank.get_scores(q) for q in queries_token_strs]
        bm25_rank_score_time = time.time() - start_time
        print(f"rank-bm25 score time: {bm25_rank_score_time:.4f}s")

        # print difference in time
        print(
            f"Index Time: BM25S is {bm25_rank_index_time / bm25_sparse_index_time:.2f}x faster than rank-bm25."
        )
        print(
            f"Score Time: BM25S is {bm25_rank_score_time / bm25_sparse_score_time:.2f}x faster than rank-bm25."
        )

        # Check if scores are exactly the same
        sparse_scores = np.array(bm25_sparse_scores)
        rank_scores = np.array(bm25_rank_scores)

        error_msg = f"\nScores between bm25-sparse and rank-bm25 are not exactly the same on dataset {dataset}."
        almost_equal = np.allclose(sparse_scores, rank_scores)
        self.assertTrue(almost_equal, error_msg)

        general_info = {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_jobs": 1,
            "dataset": dataset,
            "corpus_size": len(corpus_lst),
            "queries_size": len(queries_lst),
            "corpus_subsampled": corpus_subsample is not None,
            "queries_subsampled": queries_subsample is not None,
        }
        # Save metrics
        res = {
            "model": "bm25s",
            "index_time": bm25_sparse_index_time,
            "score_time": bm25_sparse_score_time,
        }
        res.update(general_info)
        save_scores(res, artifact_dir=artifact_dir)

        res = {
            "model": "rank-bm25",
            "score_time": bm25_rank_score_time,
            "index_time": bm25_rank_index_time,
        }
        res.update(general_info)
        save_scores(res, artifact_dir=artifact_dir)

    def compare_with_bm25_pt(
        self,
        dataset,
        artifact_dir="tests/artifacts",
        rel_save_dir="datasets",
        corpus_subsample=None,
        queries_subsample=None,
    ):

        warnings.filterwarnings("ignore", category=ResourceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        # Download and prepare dataset
        base_url = (
            "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
        )
        url = base_url.format(dataset)
        out_dir = Path(__file__).parent / rel_save_dir
        data_path = download_and_unzip(url, str(out_dir))

        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(
            split="test"
        )

        # Convert corpus and queries to lists
        corpus_lst = [val["title"] + " " + val["text"] for val in corpus.values()]
        queries_lst = list(queries.values())

        if corpus_subsample is not None:
            corpus_lst = corpus_lst[:corpus_subsample]

        if queries_subsample is not None:
            queries_lst = queries_lst[:queries_subsample]

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        t0 = time.time()
        tokenized_corpus = bm25s.hf.batch_tokenize(tokenizer, corpus_lst)
        time_corpus_tok = time.time() - t0

        t0 = time.time()
        queries_tokenized = bm25s.hf.batch_tokenize(tokenizer, queries_lst)
        time_query_tok = time.time() - t0

        print()
        print(f"Dataset:              {dataset}\n")
        # print corpus and queries size
        print(f"Corpus size:          {len(corpus_lst)}")
        print(f"Queries size:         {len(queries_lst)}")
        print()

        # Initialize and index bm25-sparse
        bm25_sparse = bm25s.BM25(k1=1.5, b=0.75, method="atire", idf_method="lucene")
        start_time = time.time()
        bm25_sparse.index(tokenized_corpus)
        bm25s_index_time = time.time() - start_time
        print(f"bm25s index time:     {bm25s_index_time:.4f}s")

        # Scoring with bm25-sparse
        start_time = time.time()
        bm25_sparse_scores = [bm25_sparse.get_scores(q) for q in queries_tokenized]
        bm25s_score_time = time.time() - start_time
        print(f"bm25s score time:     {bm25s_score_time:.4f}s")

        # Initialize and index rank-bm25
        start_time = time.time()
        model_pt = bm25_pt.BM25(tokenizer=tokenizer, device="cpu", k1=1.5, b=0.75)
        model_pt.index(corpus_lst)
        bm25_pt_index_time = time.time() - start_time
        bm25_pt_index_time -= time_corpus_tok
        print(f"bm25-pt index time:   {bm25_pt_index_time:.4f}s")

        # Scoring with rank-bm25
        start_time = time.time()
        bm25_pt_scores = model_pt.score_batch(queries_lst)
        bm25_pt_scores = bm25_pt_scores.cpu().numpy()
        bm25_pt_score_time = time.time() - start_time
        bm25_pt_score_time -= time_query_tok
        print(f"bm25-pt score time: {bm25_pt_score_time:.4f}s")

        # print difference in time
        print(
            f"Index Time: BM25S is {bm25_pt_index_time / bm25s_index_time:.2f}x faster than bm25-pt."
        )
        print(
            f"Score Time: BM25S is {bm25_pt_score_time / bm25s_score_time:.2f}x faster than bm25-pt."
        )

        # Check if scores are exactly the same
        bm25_sparse_scores = np.array(bm25_sparse_scores)
        bm25_pt_scores = np.array(bm25_pt_scores)

        error_msg = f"\nScores between bm25-sparse and rank-bm25 are not exactly the same on dataset {dataset}."
        almost_equal = np.allclose(bm25_sparse_scores, bm25_pt_scores, atol=1e-4)
        self.assertTrue(almost_equal, error_msg)

        general_info = {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_jobs": 1,
            "dataset": dataset,
            "corpus_size": len(corpus_lst),
            "queries_size": len(queries_lst),
            "corpus_was_subsampled": corpus_subsample is not None,
            "queries_was_subsampled": queries_subsample is not None,
        }
        # Save metrics
        res = {
            "model": "bm25s",
            "index_time": bm25s_index_time,
            "score_time": bm25s_score_time,
        }
        res.update(general_info)
        save_scores(res, artifact_dir=artifact_dir)

        res = {
            "model": "bm25-pt",
            "score_time": bm25_pt_score_time,
            "index_time": bm25_pt_index_time,
        }
        res.update(general_info)
        save_scores(res, artifact_dir=artifact_dir)
