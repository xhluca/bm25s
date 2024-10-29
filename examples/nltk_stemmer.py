# This class provides a way to use NLTK stemming functions with bm25s library

from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer

class  NLTKMultiStemmer:
    """
    A class that provides a unified interface for using different stemming algorithms.

    Attributes:
        stemmer_name (str): The name of the stemmer algorithm to use.
        available_stemmers (dict): A dictionary that maps stemmer names to their corresponding stemmer objects.
        stemmer (object): The current stemmer object being used.

    Methods:
        stem(tokens): Applies the current stemmer to a list of tokens and returns the stemmed tokens.
        set_stemmer(stemmer_name): Sets the current stemmer to the specified stemmer name.
    """
    def __init__(self, stemmer_name='porter', language='english'):
        self.stemmer_name = stemmer_name
        self.language = language

        self.available_stemmers = {
            'porter': PorterStemmer(),
            'snowball': SnowballStemmer('english'),
            'lancaster': LancasterStemmer()
        }
        self.stemmer = self.available_stemmers[self.stemmer_name]

    def stem(self, tokens)->list:
        """
        Applies the current stemmer to a list of tokens and returns the stemmed tokens. 
        This is done because bm25s passes a list of strings to the stemmer, 
        and the nltk function expects a single string per call. 

        Args:
            tokens (list): A list of tokens to be stemmed.

        Returns:
            list: A list of stemmed tokens.
        """
        return [self.stemmer.stem(token) for token in tokens]

    def set_stemmer(self, stemmer_name, language='english'):
        """
        Sets the current stemmer to the specified stemmer name

        Args:
            stemmer_name (str): The name of the stemmer to use.
        Raises:
            ValueError: If the specified stemmer name is not available.
        """
        if stemmer_name in self.available_stemmers:
            self.stemmer_name = stemmer_name
            if stemmer_name == 'snowball':
                self.language = language
                self.stemmer = SnowballStemmer(self.language)
            else:
                self.stemmer = self.available_stemmers[stemmer_name]
        else:
            raise ValueError(f"Invalid stemmer name: {stemmer_name}. Available stemmers: {list(self.available_stemmers.keys())}")
# Usage

# Create a NLTKMultiStemmer instance with the default Snowball stemmer
nltk_stemmer = NLTKMultiStemmer()

# Tokenize and stem the corpus using the Snowball stemmer
corpus_tokens = bm25s.tokenize(corpus, stopwords=None, stemmer=nltk_stemmer.stem)

# Change the stemmer to Porter
nltk_stemmer.set_stemmer('porter')

# Change the stemmer to Lancaster
nltk_stemmer.set_stemmer('lancaster')
