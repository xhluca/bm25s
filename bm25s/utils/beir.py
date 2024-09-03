from pathlib import Path

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(iterable, *args, **kwargs):
        return iterable


BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"


def clean_results_keys(beir_results):
    return {k.split("@")[-1]: v for k, v in beir_results.items()}


def postprocess_results_for_eval(results, scores, query_ids):
    """
    Given the queried results and scores output by BM25S, postprocess them
    to be compatible with BEIR evaluation functions.
    query_ids is a list of query ids in the same order as the results.
    """

    results_record = [
        {"id": qid, "hits": results[i], "scores": list(scores[i])}
        for i, qid in enumerate(query_ids)
    ]

    result_dict_for_eval = {
        res["id"]: {
            docid: float(score) for docid, score in zip(res["hits"], res["scores"])
        }
        for res in results_record
    }

    return result_dict_for_eval


def merge_cqa_dupstack(data_path):
    import json

    data_path = Path(data_path)
    dataset = data_path.name
    assert dataset == "cqadupstack", "Dataset must be CQADupStack"

    # check if corpus.jsonl exists
    corpus_path = data_path / "corpus.jsonl"
    if not corpus_path.exists():
        # combine all the corpus files into one
        # corpus files are located under cqadupstack/<name>/corpus.jsonl
        corpus_files = list(data_path.glob("*/corpus.jsonl"))
        with open(corpus_path, "w") as f:
            for file in tqdm(corpus_files, desc="Merging Corpus", leave=False):
                # get the name of the corpus
                corpus_name = file.parent.name

                with open(file, "r") as f2:
                    for line in tqdm(
                        f2, desc=f"Merging {corpus_name} Corpus", leave=False
                    ):
                        line = json.loads(line)
                        # add the corpus name to _id
                        line["_id"] = f"{corpus_name}_{line['_id']}"
                        # write back to file
                        f.write(json.dumps(line))
                        f.write("\n")

    # now, do the same for queries.jsonl
    queries_path = data_path / "queries.jsonl"
    if not queries_path.exists():
        queries_files = list(data_path.glob("*/queries.jsonl"))
        with open(queries_path, "w") as f:
            for file in tqdm(queries_files, desc="Merging Queries", leave=False):
                # get the name of the corpus
                corpus_name = file.parent.name

                with open(file, "r") as f2:
                    for line in tqdm(
                        f2, desc=f"Merging {corpus_name} Queries", leave=False
                    ):
                        line = json.loads(line)
                        # add the corpus name to _id
                        line["_id"] = f"{corpus_name}_{line['_id']}"
                        # write back to file
                        f.write(json.dumps(line))
                        f.write("\n")

    # now, do the same for qrels/test.tsv
    qrels_path = data_path / "qrels" / "test.tsv"
    qrels_path.parent.mkdir(parents=True, exist_ok=True)

    if not qrels_path.exists():
        qrels_files = list(data_path.glob("*/qrels/test.tsv"))
        with open(qrels_path, "w") as f:
            # First, write the columns: query-id	corpus-id	score
            f.write("query-id\tcorpus-id\tscore\n")
            for file in tqdm(qrels_files, desc="Merging Qrels", leave=False):
                # get the name of the corpus
                corpus_name = file.parent.parent.name
                with open(file, "r") as f2:
                    # skip first line
                    next(f2)

                    for line in tqdm(
                        f2, desc=f"Merging {corpus_name} Qrels", leave=False
                    ):
                        # since it's a tsv, split by tab
                        qid, cid, score = line.strip().split("\t")
                        # add the corpus name to _id
                        qid = f"{corpus_name}_{qid}"
                        cid = f"{corpus_name}_{cid}"
                        # write back to file
                        f.write(f"{qid}\t{cid}\t{score}\n")
