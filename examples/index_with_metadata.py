"""
Use dictionary corpus entries when retrieval should return metadata instead of
plain text.

bm25s indexes by list position: the tokenized text at position i corresponds to
the corpus entry at position i. Dictionary keys are user-defined. If you save the
corpus with the index, entries must be strings, dictionaries, lists, or tuples
that can be serialized to JSON.
"""
import bm25s

# Create your corpus here

corpus_json = [
    {"text": "a cat is a feline and likes to purr", "metadata": {"source": "internet"}},
    {"text": "a dog is the human's best friend and loves to play", "metadata": {"source": "encyclopedia"}},
    {"text": "a bird is a beautiful animal that can fly", "metadata": {"source": "cnn"}},
    {"text": "a fish is a creature that lives in water and swims", "metadata": {"source": "i made it up"}},
]
corpus_text = [doc["text"] for doc in corpus_json]


# Tokenize the corpus and only keep the ids (faster and saves memory)
corpus_tokens = bm25s.tokenize(corpus_text, stopwords="en")

# Create the BM25 retriever and attach your corpus_json to it
retriever = bm25s.BM25(corpus=corpus_json)
# Now, index the corpus_tokens (the corpus_json is not used yet)
retriever.index(corpus_tokens)

# Query the corpus
query = "does the fish purr like a cat?"
query_tokens = bm25s.tokenize(query)

# Get top-k results as a tuple of (doc, scores). Note that results
# will correspond to the corpus item at the corresponding index
# (you are responsible to make sure each element in corpus_json
# corresponds to each element in your tokenized corpus)
results, scores = retriever.retrieve(query_tokens, k=2)

for i in range(results.shape[1]):
    doc, score = results[0, i], scores[0, i]
    print(f"Rank {i+1} (score: {score:.2f}): {doc}")

# You can save the arrays to a directory...
# Note that this will fail if your corpus passed to `BM25(corpus=...)` is not serializable
retriever.save("animal_index_bm25")

# ...and load them when you need them
import bm25s
reloaded_retriever = bm25s.BM25.load("animal_index_bm25", load_corpus=True)
# set load_corpus=False if you don't need the corpus
