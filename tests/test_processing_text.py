from typing import Callable, Iterable

import pytest
import spacy
from spacy import Language
from spacy.tokens import Doc, Token

from usas_csv_auto_labeling.data_utils import USASTag, USASTagGroup
from usas_csv_auto_labeling.processing_text import (
    TaggedText,
    spacy_sentence_splitter,
    tag_text,
)


@pytest.fixture
def spacy_pipeline_with_sentencizer() -> spacy.Language:
    """Fixture that provides a spaCy pipeline with sentencizer component."""
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp

@pytest.fixture
def spacy_pipeline_without_sentencizer() -> spacy.Language:
    """Fixture that provides a spaCy pipeline without sentencizer component."""
    return spacy.blank("en")

def test_spacy_sentence_splitter_empty_text(spacy_pipeline_with_sentencizer: spacy.Language):
    """Test that the sentence splitter handles empty text correctly."""
    splitter = spacy_sentence_splitter(spacy_pipeline_with_sentencizer)

    # Test empty text
    sentences = list(splitter(""))
    assert len(sentences) == 0

def test_spacy_sentence_splitter_single_sentence(spacy_pipeline_with_sentencizer: spacy.Language):
    """Test that the sentence splitter handles single sentences correctly."""
    splitter = spacy_sentence_splitter(spacy_pipeline_with_sentencizer)

    # Test single sentence
    text = "This is a single sentence."
    sentences = list(splitter(text))
    assert len(sentences) == 1
    assert sentences[0] == "This is a single sentence."

def test_spacy_sentence_splitter_multiple_sentences(spacy_pipeline_with_sentencizer: spacy.Language):
    """Test that the sentence splitter handles multiple sentences correctly."""
    splitter = spacy_sentence_splitter(spacy_pipeline_with_sentencizer)

    # Test multiple sentences with various punctuation
    text = "First sentence. Second sentence! Third sentence? Fourth sentence."
    sentences = list(splitter(text))
    assert len(sentences) == 4
    assert sentences[0] == "First sentence."
    assert sentences[1] == "Second sentence!"
    assert sentences[2] == "Third sentence?"
    assert sentences[3] == "Fourth sentence."

def test_spacy_sentence_splitter_without_sentencizer(spacy_pipeline_without_sentencizer: spacy.Language):
    """Test that ValueError is raised when pipeline doesn't support sentence splitting."""
    with pytest.raises(ValueError):
        sentence = "This is a single sentence."
        list(spacy_sentence_splitter(spacy_pipeline_without_sentencizer)(sentence))

def test_spacy_sentence_splitter_returns_callable(spacy_pipeline_with_sentencizer: spacy.Language):
    """Test that the function returns a callable."""
    splitter = spacy_sentence_splitter(spacy_pipeline_with_sentencizer)
    assert callable(splitter)


def test_tagged_text_real_world_sentence():
    """Test TaggedText with a realistic sentence example."""
    # This represents: "The quick brown fox jumps over the lazy dog."
    tokens = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."]
    lemmas = ["the", "quick", "brown", "fox", "jump", "over", "the", "lazy", "dog", "."]
    pos_tags = ["DET", "ADJ", "ADJ", "NOUN", "VERB", "ADP", "DET", "ADJ", "NOUN", "PUNCT"]

    # Create USAS tags - using Z5 as a placeholder for most words
    usas_tags = [
        [USASTagGroup(tags=[USASTag(tag="Z5")])],
        [USASTagGroup(tags=[USASTag(tag="Z5")])],
        [USASTagGroup(tags=[USASTag(tag="Z5")])],
        [USASTagGroup(tags=[USASTag(tag="Z5")])],
        [USASTagGroup(tags=[USASTag(tag="Z5")])],
        [USASTagGroup(tags=[USASTag(tag="Z5")])],
        [USASTagGroup(tags=[USASTag(tag="Z5")])],
        [USASTagGroup(tags=[USASTag(tag="Z5")])],
        [USASTagGroup(tags=[USASTag(tag="Z5")])],
        [USASTagGroup(tags=[USASTag(tag="PUNCT")])]
    ]

    # "quick brown fox" is MWE 1, "lazy dog" is MWE 2
    mwe_indexes = [
        set(), set([1]), set([1]), set([1]), set(),
        set(), set(), set([2]), set([2]), set()
    ]

    tagged_text = TaggedText(
        text="The quick brown fox jumps over the lazy dog.",
        tokens=tokens,
        lemmas=lemmas,
        pos_tags=pos_tags,
        usas_tags=usas_tags,
        mwe_indexes=mwe_indexes
    )

    assert tagged_text.text == "The quick brown fox jumps over the lazy dog."
    assert len(tagged_text.tokens) == 10
    assert tagged_text.mwe_indexes[1] == set([1])  # "quick" is part of MWE 1
    assert tagged_text.mwe_indexes[7] == set([2])  # "lazy" is part of MWE 2

def test_tagged_text_optional_fields_none():
    """Test that TaggedText can be initialized with optional fields as None."""
    usas_tag_groups = [
        [USASTagGroup(tags=[USASTag(tag="Z5")])],
        [USASTagGroup(tags=[USASTag(tag="PUNCT")])]
    ]

    tagged_text = TaggedText(
        text="Test.",
        tokens=["Test", "."],
        lemmas=None,
        pos_tags=None,
        usas_tags=usas_tag_groups,
        mwe_indexes=[set(), set()]
    )

    assert tagged_text.lemmas is None
    assert tagged_text.pos_tags is None

def test_tagged_text_empty_initialization():
    """Test that TaggedText can be initialized with empty values."""
    tagged_text = TaggedText(
        text="",
        tokens=[],
        lemmas=[],
        pos_tags=[],
        usas_tags=[],
        mwe_indexes=[]
    )

    assert tagged_text.text == ""
    assert tagged_text.tokens == []
    assert tagged_text.lemmas == []
    assert tagged_text.pos_tags == []
    assert tagged_text.usas_tags == []
    assert tagged_text.mwe_indexes == []


def test_tagged_text_token_lemma_length_mismatch():
    """Test that ValueError is raised when tokens and lemmas have different lengths."""
    usas_tag_groups = [
        [USASTagGroup(tags=[USASTag(tag="Z5")])],
        [USASTagGroup(tags=[USASTag(tag="PUNCT")])]
    ]

    with pytest.raises(ValueError):
        TaggedText(
            text="Test.",
            tokens=["Test", "."],
            lemmas=["Test"],  # Missing one lemma
            pos_tags=["NOUN", "PUNCT"],
            usas_tags=usas_tag_groups,
            mwe_indexes=[set(), set()]
        )

def test_tagged_text_token_pos_tags_length_mismatch():
    """Test that ValueError is raised when tokens and pos_tags have different lengths."""
    usas_tag_groups = [
        [USASTagGroup(tags=[USASTag(tag="Z5")])],
        [USASTagGroup(tags=[USASTag(tag="PUNCT")])]
    ]

    with pytest.raises(ValueError):
        TaggedText(
            text="Test.",
            tokens=["Test", "."],
            lemmas=["test", "."],
            pos_tags=["NOUN"],  # Missing one POS tag
            usas_tags=usas_tag_groups,
            mwe_indexes=[set(), set()]
        )

def test_tagged_text_token_usas_tags_length_mismatch():
    """Test that ValueError is raised when tokens and usas_tags have different lengths."""
    usas_tag_groups = [
        [USASTagGroup(tags=[USASTag(tag="Z5")])]
        # Missing one USAS tag group
    ]

    with pytest.raises(ValueError):
        TaggedText(
            text="Test.",
            tokens=["Test", "."],
            lemmas=["test", "."],
            pos_tags=["NOUN", "PUNCT"],
            usas_tags=usas_tag_groups,
            mwe_indexes=[set(), set()]
        )

def test_tagged_text_token_mwe_indexes_length_mismatch():
    """Test that ValueError is raised when tokens and mwe_indexes have different lengths."""
    usas_tag_groups = [
        [USASTagGroup(tags=[USASTag(tag="Z5")])],
        [USASTagGroup(tags=[USASTag(tag="PUNCT")])]
    ]

    with pytest.raises(ValueError):
        TaggedText(
            text="Test.",
            tokens=["Test", "."],
            lemmas=["test", "."],
            pos_tags=["NOUN", "PUNCT"],
            usas_tags=usas_tag_groups,
            mwe_indexes=[set()]  # Missing one MWE index
        )


@Language.factory("tagging_test_component")
def create_test_tagger(nlp: Language,
                       name: str,
                       words: list[str],
                       lemma_attribute_name: str | None,
                       lemmas: list[str] | None,
                       pos_tag_attribute_name: str | None,
                       pos_tags: list[str] | None,
                       pymusas_tag_attribute_name: str,
                       pymusas_tags: list[list[str]],
                       pymusas_mwe_index_attribute_name: str,
                       mwe_indexes: list[list[tuple[int, int]]]) -> Callable[[Doc], Doc]:
    def test_tagger(doc: Doc) -> Doc:
        # Do something to the doc
        doc = Doc(nlp.vocab,
                  words=words,
                  spaces=[True] * len(words))
        for token_index, token in enumerate(doc):
            # Mock USAS tags
            setattr(token._, pymusas_tag_attribute_name, pymusas_tags[token_index])
            # Mock MWE indexes
            setattr(token._, pymusas_mwe_index_attribute_name, mwe_indexes[token_index])
            if lemma_attribute_name and lemmas is not None:
                setattr(token, lemma_attribute_name, lemmas[token_index])
            if pos_tag_attribute_name and pos_tags is not None:
                setattr(token, pos_tag_attribute_name, pos_tags[token_index])
        return doc

    return test_tagger

@Language.factory("tagging_test_type_error_component")
def create_test_type_error_tagger(nlp: Language,
                                  name: str,
                       words: list[str],
                       lemma_attribute_name: str | None,
                       lemmas: list[str] | None,
                       pos_tag_attribute_name: str | None,
                       pos_tags: list[str] | None,
                       pymusas_tag_attribute_name: str,
                       pymusas_tags: str,
                       pymusas_mwe_index_attribute_name: str,
                       mwe_indexes: list[list[tuple[int, int]]]) -> Callable[[Doc], Doc]:
    def test_tagger(doc: Doc) -> Doc:
        # Do something to the doc
        doc = Doc(nlp.vocab,
                  words=words,
                  spaces=[True] * len(words))
        for token_index, token in enumerate(doc):
            # Mock USAS tags
            setattr(token._, pymusas_tag_attribute_name, pymusas_tags[token_index])
            # Mock MWE indexes
            setattr(token._, pymusas_mwe_index_attribute_name, mwe_indexes[token_index])
            if lemma_attribute_name and lemmas is not None:
                setattr(token, lemma_attribute_name, lemmas[token_index])
            if pos_tag_attribute_name and pos_tags is not None:
                setattr(token, pos_tag_attribute_name, pos_tags[token_index])
        return doc

    return test_tagger


def simple_sentence_splitter(text: str) -> list[str]:
    return text.split("\n")

def mock_pymusas_tagger(words: list[str],
                        lemma_attribute_name: str | None,
                        lemmas: list[str] | None,
                        pos_tag_attribute_name: str | None,
                        pos_tags: list[str] | None,
                        pymusas_tag_attribute_name: str,
                        pymusas_tags: list[list[str]] | str,
                        pymusas_mwe_index_attribute_name: str,
                        mwe_indexes: list[list[tuple[int, int]]],
                        tagger_factory_name: str) -> spacy.Language:
    nlp = spacy.blank("en")
    nlp.add_pipe(tagger_factory_name, config={
        "words": words,
        "lemma_attribute_name": lemma_attribute_name,
        "lemmas": lemmas,
        "pos_tag_attribute_name": pos_tag_attribute_name,
        "pos_tags": pos_tags,
        "pymusas_tag_attribute_name": pymusas_tag_attribute_name,
        "pymusas_tags": pymusas_tags,
        "pymusas_mwe_index_attribute_name": pymusas_mwe_index_attribute_name,
        "mwe_indexes": mwe_indexes
    })
    return nlp

@pytest.mark.parametrize("lemma_attribute_name", [None, "lemma_"])
@pytest.mark.parametrize("pos_tag_attribute_name", [None, "tag_"])
@pytest.mark.parametrize("pymusas_tag_attribute_name", ["pymusas_tags"])
@pytest.mark.parametrize("pymusas_mwe_index_attribute_name", ["pymusas_mwe_indexes"])
@pytest.mark.parametrize("sentence_splitter", [None, simple_sentence_splitter])
def test_tag_text_simple(lemma_attribute_name: str | None,
                         pos_tag_attribute_name: str | None,
                         pymusas_tag_attribute_name: str,
                         pymusas_mwe_index_attribute_name: str,
                         sentence_splitter: Callable[[str], Iterable[str]] | None
                         ) -> None:
    """
    Tests the tag_text function with a simple example.

    Args:
        lemma_attribute_name: The name of the attribute to get the lemma of a token
            from the spacy Token object.
        pos_tag_attribute_name: The name of the attribute to get the POS tag of a token
            from the spacy Token object.
        pymusas_tag_attribute_name: The name of the custom attribute to get the USAS tags of a token
            from the spacy Token object.
        pymusas_mwe_index_attribute_name: The name of the custom attribute to get the MWE indexes of a token
            from the spacy Token object.
        sentence_splitter: An optional function that splits a given text into sentences.
            If not provided, the text will not be split into sentences before tagging.

    """
    if not Token.has_extension(pymusas_tag_attribute_name):
        Token.set_extension(pymusas_tag_attribute_name, default=None)
    if not Token.has_extension(pymusas_mwe_index_attribute_name):
        Token.set_extension(pymusas_mwe_index_attribute_name, default=None)

    text = "On the river Nile"
    words = ["On", "the", "river", "Nile"]
    lemmas = ["On", "the", "river", "Nile"]
    pos_tags = ["DET", "DET", "NOUN", "NOUN"]
    pymusas_tags = [["Z5"], ["A1.1.1"], ["Z3", "Z2"], ["Z3", "Z2"]]
    mwe_indexes = [[(0, 1), (2, 4)], [(1, 2)], [(0, 1), (2, 4)], [(0, 1), (2, 4)]]

    mock_tagger = mock_pymusas_tagger(
        words=words,
        lemma_attribute_name=lemma_attribute_name,
        lemmas=lemmas,
        pos_tag_attribute_name=pos_tag_attribute_name,
        pos_tags=pos_tags,
        pymusas_tag_attribute_name=pymusas_tag_attribute_name,
        pymusas_tags=pymusas_tags,
        pymusas_mwe_index_attribute_name=pymusas_mwe_index_attribute_name,
        mwe_indexes=mwe_indexes,
        tagger_factory_name="tagging_test_component"
    )

    tagged_text_iter = tag_text(
        text=text,
        tagger=mock_tagger,
        sentence_splitter=sentence_splitter,
        lemma_token_extension=lemma_attribute_name,
        pos_token_extension=pos_tag_attribute_name,
        usas_token_extension="_." + pymusas_tag_attribute_name,
        mwe_token_extension="_." + pymusas_mwe_index_attribute_name
    )
    all_tagged_text = list(tagged_text_iter)

    assert len(all_tagged_text) == 1

    tagged_text = all_tagged_text[0]

    assert isinstance(tagged_text, TaggedText)

    expected_pymusas_tags: list[list[USASTagGroup]] = [
        [USASTagGroup(tags=[USASTag(tag="Z5")])],
        [USASTagGroup(tags=[USASTag(tag="A1.1.1")])],
        [USASTagGroup(tags=[USASTag(tag="Z3")]), USASTagGroup(tags=[USASTag(tag="Z2")])],
        [USASTagGroup(tags=[USASTag(tag="Z3")]), USASTagGroup(tags=[USASTag(tag="Z2")])]
    ]

    expected_mwe_indexes = [set({1}), set(), set({1}), set({1})]  

    assert tagged_text.text == text
    assert tagged_text.tokens == words

    if lemma_attribute_name is None:
        assert tagged_text.lemmas is None
    else:
        assert tagged_text.lemmas == lemmas
    if pos_tag_attribute_name is None:
        assert tagged_text.pos_tags is None
    else:
        assert tagged_text.pos_tags == pos_tags

    assert tagged_text.usas_tags == expected_pymusas_tags
    assert tagged_text.mwe_indexes == expected_mwe_indexes


    with pytest.raises(ValueError):
        mock_tagger = mock_pymusas_tagger(
            words=words,
            lemma_attribute_name=lemma_attribute_name,
            lemmas=lemmas,
            pos_tag_attribute_name=pos_tag_attribute_name,
            pos_tags=pos_tags,
            pymusas_tag_attribute_name=pymusas_tag_attribute_name,
            pymusas_tags="Wrong Type",
            pymusas_mwe_index_attribute_name=pymusas_mwe_index_attribute_name,
            mwe_indexes=mwe_indexes,
            tagger_factory_name="tagging_test_type_error_component"
        )

        list(tag_text(
            text=text,
            tagger=mock_tagger,
            sentence_splitter=sentence_splitter,
            lemma_token_extension=lemma_attribute_name,
            pos_token_extension=pos_tag_attribute_name,
            usas_token_extension="_." + pymusas_tag_attribute_name,
            mwe_token_extension="_." + pymusas_mwe_index_attribute_name
        ))

