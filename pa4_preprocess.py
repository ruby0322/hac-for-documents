import re
from nltk.stem import PorterStemmer

# Compile regex patterns outside of the function to improve performance
word_pattern = re.compile(r"\b\w+(?:[-]\w+)*\b|\b\d+-\d+\b")
punctuation_pattern = re.compile(r'[.,!?;_]')
contraction_pattern = None  # Placeholder, to be compiled later based on custom_contractions

# Initialize the PorterStemmer once
stemmer = PorterStemmer()

def tokenize(text: str):
    words = word_pattern.findall(text)
    # punctuations = punctuation_pattern.findall(text)
    tokens = words
    return [token for token in tokens if not token.startswith('_') and not token.isdigit() and not any(map(lambda x: x.isdigit(), token.split('-')))]

def contractions_expand(text: str, custom_contractions: dict, contraction_pattern):
    return contraction_pattern.sub(lambda match: custom_contractions[match.group(0)], text)

def stopwords_remove(tokens: list[str], stopwords_set: set[str]) -> list[str]:
    return [token for token in tokens if token not in stopwords_set]

def stem(tokens):
    return [stemmer.stem(token) for token in tokens]

def preprocess(documents, custom_stopwords, custom_contractions, remove_stopwords=True, expand_contractions=True, stemming=True):
    global contraction_pattern

    # Convert stopwords list to set for faster lookup
    stopwords_set = set(custom_stopwords)

    # Compile a single regex pattern for contraction expansion
    if expand_contractions:
        contraction_regex = '|'.join(map(re.escape, custom_contractions.keys()))
        contraction_pattern = re.compile(contraction_regex)

    preprocessed_documents = []
    for text in documents:
        # Expand contractions using custom rules
        if expand_contractions:
            text = contractions_expand(text, custom_contractions, contraction_pattern)

        # Tokenize
        tokens = tokenize(text)

        # Remove stopwords and punctuation if specified
        if remove_stopwords:
            tokens = stopwords_remove(tokens, stopwords_set)

        if stemming:
            tokens = stem(tokens)
        preprocessed_documents.append(tokens)

    return preprocessed_documents