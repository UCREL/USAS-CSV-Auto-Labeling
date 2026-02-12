from typing import Callable, Iterable, Type, TypeVar, cast

import spacy
from pydantic import BaseModel, Field, model_validator
from spacy.tokens import Doc

from usas_csv_auto_labeling.data_utils import (
    USASTag,
    USASTagGroup,
    get_all_mwe_token_indexes,
    parse_usas_token_group,
)

ATTRIBUTE_TYPE = TypeVar("ATTRIBUTE_TYPE")

class TaggedText(BaseModel):
    """
    A class that represents a tagged text.

    Attributes:
        text (str): The text that was tagged.
        tokens (list[str]): The tokens of the text that was tagged.
        lemmas (list[str] | None):: The lemmas of the text that was tagged. Default None.
        pos_tags (list[str] | None): The POS tags of the text that was tagged. Default None.
        usas_tags (list[list[USASTagGroup]]): The USAS tags of the text that was tagged.
        mwe_indexes (list[set[int]]): The MWE indexes of the text that was tagged.
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
    of any MWE. TO NOTE: USAS historically does not support overlapping MWEs.
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
        doc: Doc = spacy_pipeline(text)
        for sentence in doc.sents:
            yield sentence.text

    return _sentence_splitter

def tag_text(text: str,
             tagger: spacy.Language,
             sentence_splitter: Callable[[str], Iterable[str]] | None = None,
             lemma_token_extension: str | None = None, # lemma_
             pos_token_extension: str | None = None, # pos_
             token_text_extension: str = "text",
             usas_token_extension: str = "_.pymusas_tags",
             mwe_token_extension: str = "_.pymusas_mwe_indexes"
             ) -> Iterable[TaggedText]:
    """
    Tags the text using the given, `tagger`, spacy pipeline.

    Args:
        text: The text to tag.
        tagger: The spaCy pipeline to use for tagging.
        sentence_splitter: An optional function that splits a given text into sentences.
            If not provided, the text will not be split into sentences before tagging.
        lemma_token_extension: The name of the attribute to get the lemma of a token
            from the spacy Token object. If not provided, the lemma will not be extracted.
        pos_token_extension: The name of the attribute to get the POS tag of a token
            from the spacy Token object. If not provided, the POS tag will not be extracted.
        token_text_extension: The name of the attribute to get the text of a token
            from the spacy Token object.
        usas_token_extension: The name of the custom attribute to get the USAS tags of a token
            from the spacy Token object.
        mwe_token_extension: The name of the custom attribute to get the MWE indexes of a token
            from the spacy Token object.

    Yields:
        An iterable of TaggedText objects, where each TaggedText object corresponds
            to a sentence in the input text. If `sentence_splitter` is not provided,
            the iterable will only contain one TaggedText object.

    Raises:
        ValueError: If an extension for a token is not the expected type.
            For lemma and pos, the expected type is str.
            For usas_tags, the expected type is list[str].
            For mwe_indexes, the expected type is list[tuple[int, int]].
    """
    def get_spacy_attribute(spacy_object: object,
                            attribute_name: str,
                            expected_type: Type[ATTRIBUTE_TYPE]) -> ATTRIBUTE_TYPE:
        """
        Gets an attribute from a spaCy object, whereby the attribute name can be
        prefixed with "_." to get a custom attribute. This function does not support
        nested attributes, e.g. "_._.CUSTOM_ATTRIBUTE".

        Args:
            spacy_object: The spaCy object to get the attribute from.
            attribute_name: The name of the attribute to get. If the attribute name starts with "_.",
                it is assumed to be a custom attribute.
            expected_type: The type that the attribute is expected to be.

        Returns:
            The value of the attribute.

        Raises:
            ValueError: If the attribute is not of the expected type.
        """
        spacy_custom_attribute = False
        if attribute_name[:2] == "_.":
            spacy_custom_attribute = True
            attribute_name = attribute_name[2:]
        
        attribute_value = None
        if spacy_custom_attribute:
            attribute_value = getattr(getattr(spacy_object, "_"), attribute_name)
        else:
            attribute_value = getattr(spacy_object, attribute_name)
        
        if not isinstance(attribute_value, expected_type):
            raise ValueError(f"Expected {attribute_name} to be of type {expected_type}, "
                             f"but got {type(attribute_value)}")
        return attribute_value

    
    def _process_text(text_to_process: str) -> TaggedText:
        tagged_text: Doc = tagger(text_to_process)

        tokens: list[str] = []
        lemmas: list[str] | None = []
        pos_tags: list[str] | None = []
        usas_tag_groups: list[list[USASTagGroup]] = []
        mwe_indexes: list[set[int]] = []
        all_mwe_token_indexes_with_min_value: set[tuple[frozenset[int], int]] = set()

        for token in tagged_text:
            token_text = get_spacy_attribute(token, token_text_extension, str)
            tokens.append(token_text)
            
            if lemma_token_extension is not None:
                lemma = get_spacy_attribute(token, lemma_token_extension, str)
                lemmas.append(lemma)
            
            if pos_token_extension is not None:
                pos_tag = get_spacy_attribute(token, pos_token_extension, str)
                pos_tags.append(pos_tag)

            usas_tags = cast(list[str],
                                        get_spacy_attribute(token, usas_token_extension, list))
            token_usas_tag_groups = parse_usas_token_group(" ".join(usas_tags))
            usas_tag_groups.append(token_usas_tag_groups)

            
            
            token_mwe_indexes_range = cast(list[tuple[int, int]],
                                                                  get_spacy_attribute(token, mwe_token_extension, list))
            
            token_mwe_indexes = get_all_mwe_token_indexes(token_mwe_indexes_range)
            if len(token_mwe_indexes) > 1:
                all_mwe_token_indexes_with_min_value.add((token_mwe_indexes, min(token_mwe_indexes)))
            mwe_indexes.append(set())

        # Computationally mwe indexes in token order has increased the complexity
        # of the code and runtime, but it should make it easier for the annotators
        # to read.
        sorted_all_mwe_token_indexes_with_min_value = sorted(all_mwe_token_indexes_with_min_value, key=lambda x: x[1])
        for mwe_index_value, mwe_index_with_min_value in enumerate(sorted_all_mwe_token_indexes_with_min_value, start=1):
            for mwe_index in mwe_index_with_min_value[0]:
                mwe_indexes[mwe_index].add(mwe_index_value)

        if lemma_token_extension is None:
            lemmas = None
        
        if pos_token_extension is None:
            pos_tags = None
            
        return TaggedText(
            text=text_to_process,
            tokens=tokens,
            lemmas=lemmas,
            pos_tags=pos_tags,
            usas_tags=usas_tag_groups,
            mwe_indexes=mwe_indexes
        )

        return tagged_text

    if sentence_splitter is not None:
        for sentence in sentence_splitter(text):
            yield _process_text(sentence)
    else:
        yield _process_text(text)