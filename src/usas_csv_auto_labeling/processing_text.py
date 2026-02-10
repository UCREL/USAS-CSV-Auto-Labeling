from typing import Callable, Iterable

import spacy
from pydantic import BaseModel, Field, model_validator

from usas_csv_auto_labeling.data_utils import USASTag, USASTagGroup


class TaggedText(BaseModel):
    """
    A class that represents a tagged text.

    Attributes:
        text (str): The text that was tagged.
        tokens (list[str]): The tokens of the text that was tagged.
        lemmas (list[str] | None):: The lemmas of the text that was tagged. Default None.
        pos_tags (list[str] | None): The POS tags of the text that was tagged. Default None.
        usas_tags (list[USASTagGroup]): The USAS tags of the text that was tagged.
    """
    _usas_tags_example: list[list[USASTagGroup]] = [
        [
            USASTagGroup(tags=[USASTag(tag='M6', number_positive_markers=0, number_negative_markers=0, rarity_marker_1=False, rarity_marker_2=False, female=False, male=False, antecedents=False, neuter=False, idiom=False)]),
            USASTagGroup(tags=[USASTag(tag='Z5', number_positive_markers=0, number_negative_markers=0, rarity_marker_1=False, rarity_marker_2=False, female=False, male=False, antecedents=False, neuter=False, idiom=False)]),
            USASTagGroup(tags=[USASTag(tag='Z8', number_positive_markers=0, number_negative_markers=0, rarity_marker_1=False, rarity_marker_2=False, female=False, male=False, antecedents=False, neuter=False, idiom=False)])
        ],
        [
            USASTagGroup(tags=[USASTag(tag='A3', number_positive_markers=1, number_negative_markers=0, rarity_marker_1=False, rarity_marker_2=False, female=False, male=False, antecedents=False, neuter=False, idiom=False)]),
            USASTagGroup(tags=[USASTag(tag='Z5', number_positive_markers=0, number_negative_markers=0, rarity_marker_1=False, rarity_marker_2=False, female=False, male=False, antecedents=False, neuter=False, idiom=False)])
        ],
        [
            USASTagGroup(tags=[USASTag(tag='Z5', number_positive_markers=0, number_negative_markers=0, rarity_marker_1=False, rarity_marker_2=False, female=False, male=False, antecedents=False, neuter=False, idiom=False)])
        ],
        [
            USASTagGroup(tags=[USASTag(tag='Q3', number_positive_markers=0, number_negative_markers=0, rarity_marker_1=False, rarity_marker_2=False, female=False, male=False, antecedents=False, neuter=False, idiom=False)]),
            USASTagGroup(tags=[USASTag(tag='G2.1', number_positive_markers=0, number_negative_markers=0, rarity_marker_1=False, rarity_marker_2=False, female=False, male=False, antecedents=False, neuter=False, idiom=False)])
        ],
        [
            USASTagGroup(tags=[USASTag(tag='PUNCT', number_positive_markers=0, number_negative_markers=0, rarity_marker_1=False, rarity_marker_2=False, female=False, male=False, antecedents=False, neuter=False, idiom=False)])
        ]
    ]
    _mwe_indexes_description = """
    Multi Word Expression (MWE) indexes for each token in the tagged text. If a token
    is a MWE then it is assigned an index starting from 1 if a token is part of more than
    one MWE then it is assigned two MWE indexes, e.g. `set([1,2])` means that the token for
    that index is part of MWE 1 and 2. An empty set represents a token that is not part
    of any MWE.
    """
    text: str = Field(title="Text", description="The text that was tagged", examples=["This is a sentence.", ""])
    tokens: list[str] = Field(title="Tokens", description="The tokens of the text that was tagged", examples=[["This", "is", "a", "sentence", "."], []])
    lemmas: list[str] | None = Field(title="Lemmas", description="The lemmas of the text that was tagged", examples=[["this", "be", "a", "sentence", "."], None], default=None)
    pos_tags: list[str] | None = Field(title="POS Tags", description="The POS tags of the text that was tagged", examples=[["PRON", "AUX", "DET", "NOUN", "PUNCT"], None], default=None)
    usas_tags: list[list[USASTagGroup]] = Field(title="USAS Tags", description="The USAS tags of the text that was tagged", examples=[_usas_tags_example, []])
    mwe_indexes: list[set[int]] = Field(title="MWE indexes", description=_mwe_indexes_description, examples=[[set(), set(), set([1]), set([1]), set()], []])

    @model_validator(mode='after')
    def check_lists_match(self) -> "TaggedText":
        """
        Checks that the length of the tokens, lemmas, POS tags, USAS tags, and MWE indexes
        are all the same if they are not None. If they are not the same, raises a ValueError.

        Returns:
            The TaggedText object
        Raises:
            ValueError: If the length of the tokens, lemmas, POS tags, USAS tags, and MWE indexes are not the same
        """
        number_tokens = len(self.tokens)
        if self.lemmas is not None and number_tokens != len(self.lemmas):
            raise ValueError(f"The number of tokens: {number_tokens} and "
                             f"lemmas must be the same: {len(self.lemmas)}")
        if self.pos_tags is not None and number_tokens != len(self.pos_tags):
            raise ValueError(f"The number of tokens: {number_tokens} "
                             f"and POS tags must be the same: {len(self.pos_tags)}")
        if number_tokens != len(self.usas_tags):
            raise ValueError(f"The number of tokens: {number_tokens} and "
                             f"USAS tags must be the same: {len(self.usas_tags)}")
        if number_tokens != len(self.mwe_indexes):
            raise ValueError(f"The number of tokens: {number_tokens} and "
                             f"MWE indexes must be the same: {len(self.mwe_indexes)}")
        return self

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
             ) -> Iterable[TaggedText]:

    yield ""