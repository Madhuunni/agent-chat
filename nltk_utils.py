import re

import numpy as np
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

# Keep high-signal technical tokens intact before falling back to words/numbers.
# The previous NLTK tokenizer split URLs, CSS selectors, ids, and quoted values into
# many punctuation fragments. For this dataset those details distinguish intents, so
# dropping or fragmenting them creates nearly identical bags of words for different
# labels and the classifier cannot learn reliably.
_TOKEN_PATTERN = re.compile(
    r"https?://[^\s,]+|"  # URLs, including ports/query strings
    r"[A-Za-z_][\w-]*\[[^\]]+\]|"  # CSS selectors such as input[id='emailId']
    r"[A-Za-z_][\w-]*=['\"][^'\"]+['\"]|"  # attribute/value pairs
    r"['\"][^'\"]+['\"]|"  # quoted values
    r"[A-Za-z_][\w-]*|"  # words/identifiers
    r"\d+(?:\.\d+)?"  # numbers and ports
)


def tokenize(sentence):
    """
    Split a sentence into model tokens without requiring external NLTK data.

    The chatbot training data is Selenium-like and contains URLs, selectors,
    attributes, and quoted test values. Preserving those strings as tokens gives
    the bag-of-words model enough signal to separate otherwise similar intents.
    """
    return [token.rstrip(".!?,") for token in _TOKEN_PATTERN.findall(sentence)]


def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    sentence_words = {stem(word) for word in tokenized_sentence}
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1.0

    return bag
