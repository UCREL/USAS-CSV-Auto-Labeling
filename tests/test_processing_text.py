import pytest
import spacy

from usas_csv_auto_labeling.processing_text import spacy_sentence_splitter


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