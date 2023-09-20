import re
from collections import Counter
from config import output_chars, spacy_model

import torch
import numpy as np

import spacy
nlp = spacy.load(spacy_model)


def ascii_indexing(image):
    return torch.round(image * (len(output_chars) - 1))


def flatten_captions(captions):
    captions = ' '.join(captions)
    captions = captions.lower()
    captions = re.sub("[^a-z ]+", "", string=captions)
    return captions

def extract_keywords(words):
    words = nlp(words)
    nouns = [w.text for w in words if w.pos_ in ["NOUN", "PROPN"]]
    noun_counts = Counter(nouns)
    keywords = [kw for (kw, count) in noun_counts.most_common(4)]
    return keywords

def keyword_vector(keywords):
    vectors = [nlp(kw).vector for kw in keywords]
    kw_vector = np.mean(vectors, axis=0)
    return kw_vector
