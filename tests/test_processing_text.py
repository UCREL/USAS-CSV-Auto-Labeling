import pytest
import spacy

from usas_csv_auto_labeling.data_utils import USASTag, USASTagGroup
from usas_csv_auto_labeling.processing_text import TaggedText, spacy_sentence_splitter


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