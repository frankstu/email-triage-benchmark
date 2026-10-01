"""Erkennung von Personennamen mit lokalen spaCy-Modellen (Deutsch und Englisch)."""

from __future__ import annotations

import re
from collections.abc import Sequence

import spacy
from spacy.language import Language

_WORD = re.compile(r"[a-zäöüß]+")
_DE_WORDS = frozenset({"der", "die", "das", "und", "ist", "nicht", "ich", "sie", "wir", "mit",
                       "für", "auf", "ein", "eine", "zu", "von", "den", "dem", "bitte", "ihr",
                       "ihre", "sind", "wird", "auch", "noch", "bei", "aus", "hallo", "grüße"})
_EN_WORDS = frozenset({"the", "and", "is", "not", "you", "we", "with", "for", "on", "a", "to",
                       "of", "your", "please", "this", "that", "are", "will", "be", "from",
                       "have", "our", "hi", "thanks", "regards"})
_PERSON_LABELS = frozenset({"PER", "PERSON"})
# Für die Namenserkennung unnötige Komponenten; tok2vec bleibt, weil ner darauf aufsetzen kann
_UNUSED_PIPES = ("tagger", "morphologizer", "parser", "lemmatizer", "attribute_ruler", "senter")


def detect_language(text: str) -> str:
    words = _WORD.findall(text.lower())
    de = sum(w in _DE_WORDS for w in words)
    en = sum(w in _EN_WORDS for w in words)
    return "en" if en > de else "de"


def _load(name: str) -> Language:
    nlp = spacy.load(name)
    for pipe in _UNUSED_PIPES:
        if pipe in nlp.pipe_names:
            nlp.disable_pipe(pipe)
    return nlp


class SpacyNer:
    def __init__(self, de_model: str = "de_core_news_md", en_model: str = "en_core_web_md") -> None:
        self._models = {"de": _load(de_model), "en": _load(en_model)}

    def __call__(self, texts: Sequence[str]) -> list[list[tuple[int, int]]]:
        results: list[list[tuple[int, int]]] = [[] for _ in texts]
        for lang, nlp in self._models.items():
            indices = [i for i, t in enumerate(texts) if t and detect_language(t) == lang]
            docs = nlp.pipe((texts[i] for i in indices), batch_size=64)
            for i, doc in zip(indices, docs, strict=True):
                results[i] = [(e.start_char, e.end_char) for e in doc.ents
                              if e.label_ in _PERSON_LABELS]
        return results
