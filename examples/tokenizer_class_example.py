import bm25s
import Stemmer

stemmer = Stemmer.Stemmer('english')

tokenizer = bm25s.tokenization.Tokenizer(stemmer=stemmer)

# Tokenize a string
texts = [
    "This is a test string",
    "Here is another test string!"
]

stream = tokenizer.streaming_tokenize(texts)

print("1", tokenizer.vocab)

print(next(stream))

print("2", tokenizer.vocab)

print(next(stream))

print("3", tokenizer.vocab)

print('-'*20)
tokenizer.reset_vocab()

for token in tokenizer.streaming_tokenize(texts):
    print(token)

print(tokenizer.tokenize(texts, show_progress=True, leave_progress=True))

# todo: test with and without stemming, without and without stopwords, with and without lowercasing
# todo: test inputs to retriever.retrieve() with both list of strings and list of lists of strings and ids and list of of list of ids