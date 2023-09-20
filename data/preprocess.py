import re
from collections import Counter

import torch

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords


class Ascii:
    def __init__(self, char_count: int):
        self.char_count = char_count
    
    def __call__(self, x):
        return torch.round(x * self.char_count)


def tokenize_captions(captions):
    captions = ' '.join(captions)
    captions = captions.lower()
    captions = re.sub("[^a-z ]+", "", string=captions)
    return captions.split()

def extract_keywords(words):
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w not in stop_words]

    tagged_words = pos_tag(words)
    nouns = [w for (w, tag) in tagged_words if tag.startswith("NN")]

    nouns = Counter(nouns)
    keywords = [kw for (kw, count) in nouns.most_common(4)]

    return keywords

def keyword_vector(keywords):
    return keywords
