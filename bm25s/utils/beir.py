import logging
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(iterable, *args, **kwargs):
        return iterable


from . import json_functions

BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
GH_URL = "https://github.com/xhluca/bm25s/releases/download/data/{}.zip"


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
                        line = json_functions.loads(line)
                        # add the corpus name to _id
                        line["_id"] = f"{corpus_name}_{line['_id']}"
                        # write back to file
                        f.write(json_functions.dumps(line))
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
                        line = json_functions.loads(line)
                        # add the corpus name to _id
                        line["_id"] = f"{corpus_name}_{line['_id']}"
                        # write back to file
                        f.write(json_functions.dumps(line))
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


def download_dataset(
    dataset,
    base_url=GH_URL,
    save_dir="./datasets",
    unzip=True,
    redownload=False,
    show_progress=True,
):
    import urllib.request
    import zipfile
    from pathlib import Path
    from tqdm.auto import tqdm

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    url = base_url.format(dataset)
    # check if zip file already exist
    save_zip_path = save_dir / "archive" / f"{dataset}.zip"
    save_zip_path.parent.mkdir(parents=True, exist_ok=True)

    if not save_zip_path.exists() or redownload:
        # download the zip file and save it with tqdm progress bar
        pbar = tqdm(
            unit="B",
            unit_scale=True,
            desc=f"Downloading {dataset}",
            leave=False,
            disable=not show_progress,
        )
        with open(save_zip_path, "wb") as f:
            response = urllib.request.urlopen(url)
            total_size = int(response.headers.get("content-length", 0))
            block_size = 8192 * 2
            # set the tqdm total to the total size
            pbar.total = total_size
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                f.write(buffer)
                pbar.update(len(buffer))

        pbar.close()

    # now that we have the zip file, extract it
    if unzip:
        with zipfile.ZipFile(save_zip_path, "r") as zip_ref:
            zip_ref.extractall(save_dir)

        # if it's CQADupStack, merge the corpus, queries, and qrels
        if dataset == "cqadupstack":
            merge_cqa_dupstack(save_dir / dataset)

        return save_dir / dataset
    else:
        return save_zip_path


def load_jsonl(
    dataset,
    fname,
    save_dir="./datasets",
    show_progress=True,
    return_dict=True,
    force_title=False,
    remove=None,
):
    dataset_path = Path(save_dir) / dataset
    corpus_path = dataset_path / fname

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found at {corpus_path}")

    corpus = []
    with open(corpus_path, "r") as f:
        # get the number of bytes in the file
        num_lines = sum(1 for i in open(corpus_path, "rb"))
        pbar = tqdm(
            f,
            desc="[{}] loading {}".format(dataset, fname),
            leave=False,
            disable=not show_progress,
            total=num_lines,
        )
        for line in pbar:
            line = json_functions.loads(line)
            if force_title:
                line["title"] = line.get("title")
            if remove is not None:
                for key in remove:
                    del line[key]
            corpus.append(line)
            # update the progress bar wrt the number of bytes read

    if return_dict:
        corpus = {doc.pop("_id"): doc for doc in corpus}

    return corpus


def load_corpus(dataset, save_dir="./datasets", show_progress=True, return_dict=True):
    return load_jsonl(
        dataset=dataset,
        save_dir=save_dir,
        show_progress=show_progress,
        return_dict=return_dict,
        fname="corpus.jsonl",
        force_title=True,
        remove=["metadata"],
    )


def load_queries(dataset, save_dir="./datasets", show_progress=True, return_dict=True):
    return load_jsonl(
        dataset=dataset,
        save_dir=save_dir,
        show_progress=show_progress,
        return_dict=return_dict,
        fname="queries.jsonl",
        force_title=False,
        remove=["metadata"],
    )


def load_qrels(
    dataset, split="test", save_dir="./datasets", show_progress=True, return_dict=True
):
    """
    This is tsv files
    """
    if split not in ["train", "dev", "test"]:
        raise ValueError("split must be one of ['train', 'dev', 'test']")

    dataset_path = Path(save_dir) / dataset
    qrels_path = dataset_path / "qrels" / f"{split}.tsv"

    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels file not found at {qrels_path}")

    qrels = []
    with open(qrels_path, "r") as f:
        # skip first line
        next(f)
        for line in tqdm(
            f,
            desc="Loading Qrels {}".format(dataset),
            leave=False,
            disable=not show_progress,
        ):
            qid, cid, score = line.strip().split("\t")
            qrels.append((qid, cid, int(score)))

    if return_dict:
        qrels = {qid: {cid: score} for qid, cid, score in qrels}

    return qrels


def evaluate(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
    ignore_identical_ids: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Acknowledgement: This function is adapted from BEIR's EvaluateRetrieval class.
    License for this function: Apache-2.0
    """
    try:
        import pytrec_eval
    
    except ImportError:
        raise ImportError(
            "Please install pytrec_eval to use this function. You can install it via `pip install pytrec_eval`."
        )

    if ignore_identical_ids:
        logging.info(
            "For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this."
        )
        popped = []
        for qid, rels in results.items():
            for pid in list(rels):
                if qid == pid:
                    results[qid].pop(pid)
                    popped.append(pid)

    ndcg = {}
    _map = {}
    recall = {}
    precision = {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string, precision_string}
    )
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)

    for eval in [ndcg, _map, recall, precision]:
        logging.info("\n")
        for k in eval.keys():
            logging.info("{}: {:.4f}".format(k, eval[k]))

    return ndcg, _map, recall, precision
