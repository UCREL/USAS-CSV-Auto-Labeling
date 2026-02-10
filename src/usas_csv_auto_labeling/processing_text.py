from typing import Callable, Iterable

import spacy


def spacy_sentence_splitter(spacy_pipeline: spacy.Language) -> Callable[[str], Iterable[str]]:
    """
    Returns a function that splits a given text into sentences using the given
    Spacy pipeline.

    Args:
        spacy_pipeline: A Spacy pipeline to use for sentence splitting.
            We assume that the

    Returns:
        A function that takes a string and returns an iterable of strings,
        where each string is a sentence in the input text.
    Raises:
        ValueError: If the given spaCy pipeline does not support sentence splitting.
    """
    def _sentence_splitter(text: str) -> Iterable[str]:
        doc = spacy_pipeline(text)
        for sentence in doc.sents:
            yield sentence.text

    return _sentence_splitter

def tag_text(text: str,
             tagger: spacy.Language,
             sentence_splitter: Callable[[str], Iterable[str]] | None = None
             ) -> Iterable[str]:

    yield ""