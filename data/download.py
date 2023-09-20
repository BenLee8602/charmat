import spacy
from config import spacy_model

spacy.cli.download(spacy_model)
