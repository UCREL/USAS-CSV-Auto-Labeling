from typing import Callable, Iterable, cast

import malaya
import spacy
from malaya.torch_model.huggingface import Tagging as MalayaTagging
from malaya.torch_model.rnn import Stem as MalayaStem
from pymusas.lexicon_collection import LexiconCollection, MWELexiconCollection
from pymusas.pos_mapper import UPOS_TO_USAS_CORE, USAS_CORE_TO_UPOS
from pymusas.rankers.lexicon_entry import ContextualRuleBasedRanker
from pymusas.spacy_api.taggers.hybrid import HybridTagger
from pymusas.taggers.rules.mwe import MWERule
from pymusas.taggers.rules.single_word import SingleWordRule
from stanza.pipeline.core import Pipeline as StanzaPipeline

from usas_csv_auto_labeling.processing_text import spacy_sentence_splitter


def get_english_hybrid_tagger() -> spacy.Language:

    hybrid_nlp = spacy.load('en_core_web_trf', exclude=['parser', 'ner'])
    # URLS to the English single and MWE lexicons
    english_single_lexicon_url = ('https://raw.githubusercontent.com/UCREL/Multilingual-USAS/'
                                  '2cc9966a3bdcc84bc204d16bdf4318fc28495016/'
                                  'English/semantic_lexicon_en.tsv')
    english_mwe_lexicon_url = ('https://raw.githubusercontent.com/UCREL/Multilingual-USAS/'
                               '2cc9966a3bdcc84bc204d16bdf4318fc28495016/'
                               'English/mwe-en.tsv')
    lexicon_lookup = LexiconCollection.from_tsv(english_single_lexicon_url, include_pos=True)
    lemma_lexicon_lookup = LexiconCollection.from_tsv(english_single_lexicon_url, include_pos=False)
    mwe_lexicon_lookup = MWELexiconCollection.from_tsv(english_mwe_lexicon_url)
    # The rules that use the lexicons
    single_word_rule = SingleWordRule(lexicon_lookup, lemma_lexicon_lookup)
    mwe_word_rule = MWERule(mwe_lexicon_lookup)
    word_rules = [single_word_rule, mwe_word_rule]
    # The ranker that determines which rule should be used/applied
    ranker_arguments = ContextualRuleBasedRanker.get_construction_arguments(word_rules)
    ranker = ContextualRuleBasedRanker(*ranker_arguments)
    # POS that indicate a Punctuation and Numeric value
    default_punctuation_tags = list(['PUNCT'])
    default_number_tags = list(['NUM'])

    tagger = cast(HybridTagger, hybrid_nlp.add_pipe('pymusas_hybrid_tagger', config={"top_n": 3}))

    tagger.initialize(rules=word_rules,
                      ranker=ranker,
                      default_punctuation_tags=default_punctuation_tags,
                      default_number_tags=default_number_tags,
                      pretrained_model_name_or_path="ucrelnlp/PyMUSAS-Neural-English-Base-BEM")
    return hybrid_nlp


def get_english_sentence_splitter() -> Callable[[str], Iterable[str]]:
    return spacy_sentence_splitter(spacy.load('en_core_web_sm'))


def get_danish_hybrid_tagger() -> spacy.Language:

    hybrid_nlp = spacy.load('da_core_news_lg', exclude=['parser', 'ner'])
    # URLS to the Danish single and MWE lexicons
    danish_single_lexicon_url = ('https://raw.githubusercontent.com/UCREL/Multilingual-USAS/'
                                 '2cc9966a3bdcc84bc204d16bdf4318fc28495016/Danish/'
                                 'semantic_lexicon_dk.tsv')
    danish_mwe_lexicon_url = ('https://raw.githubusercontent.com/UCREL/Multilingual-USAS/'
                              '2cc9966a3bdcc84bc204d16bdf4318fc28495016/Danish/mwe-dk.tsv')
    lexicon_lookup = LexiconCollection.from_tsv(danish_single_lexicon_url, include_pos=True)
    lemma_lexicon_lookup = LexiconCollection.from_tsv(danish_single_lexicon_url, include_pos=False)
    mwe_lexicon_lookup = MWELexiconCollection.from_tsv(danish_mwe_lexicon_url)
    # The rules that use the lexicons
    single_word_rule = SingleWordRule(lexicon_lookup, lemma_lexicon_lookup)
    mwe_word_rule = MWERule(mwe_lexicon_lookup)
    word_rules = [single_word_rule, mwe_word_rule]
    # The ranker that determines which rule should be used/applied
    ranker_arguments = ContextualRuleBasedRanker.get_construction_arguments(word_rules)
    ranker = ContextualRuleBasedRanker(*ranker_arguments)
    # POS that indicate a Punctuation and Numeric value
    default_punctuation_tags = list(['PUNCT'])
    default_number_tags = list(['NUM'])

    tagger = cast(HybridTagger, hybrid_nlp.add_pipe('pymusas_hybrid_tagger', config={"top_n": 3}))

    tagger.initialize(rules=word_rules,
                      ranker=ranker,
                      default_punctuation_tags=default_punctuation_tags,
                      default_number_tags=default_number_tags,
                      pretrained_model_name_or_path="ucrelnlp/PyMUSAS-Neural-Multilingual-Base-BEM")
    return hybrid_nlp

def get_danish_sentence_splitter() -> Callable[[str], Iterable[str]]:
    return spacy_sentence_splitter(spacy.load('da_core_news_lg'))

def get_dutch_hybrid_tagger() -> spacy.Language:

    hybrid_nlp = spacy.load('nl_core_news_lg', exclude=['parser', 'ner'])
    # URL to the Dutch single lexicon
    dutch_single_lexicon_url = ('https://raw.githubusercontent.com/UCREL/Multilingual-USAS/'
                                '2cc9966a3bdcc84bc204d16bdf4318fc28495016/Dutch/'
                                'semantic_lexicon_dut.tsv')
    lexicon_lookup = LexiconCollection.from_tsv(dutch_single_lexicon_url, include_pos=True)
    lemma_lexicon_lookup = LexiconCollection.from_tsv(dutch_single_lexicon_url, include_pos=False)
    # The rules that use the lexicons
    single_word_rule = SingleWordRule(lexicon_lookup, lemma_lexicon_lookup, UPOS_TO_USAS_CORE)
    word_rules = [single_word_rule]
    # The ranker that determines which rule should be used/applied
    ranker_arguments = ContextualRuleBasedRanker.get_construction_arguments(word_rules)
    ranker = ContextualRuleBasedRanker(*ranker_arguments)
    # POS that indicate a Punctuation and Numeric value
    default_punctuation_tags = list(['PUNCT'])
    default_number_tags = list(['NUM'])

    tagger = cast(HybridTagger, hybrid_nlp.add_pipe('pymusas_hybrid_tagger', config={"top_n": 3}))

    tagger.initialize(rules=word_rules,
                      ranker=ranker,
                      default_punctuation_tags=default_punctuation_tags,
                      default_number_tags=default_number_tags,
                      pretrained_model_name_or_path="ucrelnlp/PyMUSAS-Neural-Multilingual-Base-BEM")
    return hybrid_nlp

def get_dutch_sentence_splitter() -> Callable[[str], Iterable[str]]:
    return spacy_sentence_splitter(spacy.load('nl_core_news_md'))

def get_spanish_hybrid_tagger() -> spacy.Language:

    hybrid_nlp = spacy.load('es_dep_news_trf', exclude=['parser', 'ner'])
    # URLS to the Spanish single and MWE lexicons
    spanish_single_lexicon_url = ('https://raw.githubusercontent.com/UCREL/Multilingual-USAS/2cc9966a3bdcc84bc204d16bdf4318fc28495016/Spanish/semantic_lexicon_es.tsv')
    spanish_mwe_lexicon_url = ('https://raw.githubusercontent.com/UCREL/Multilingual-USAS/2cc9966a3bdcc84bc204d16bdf4318fc28495016/Spanish/mwe-es.tsv')
    lexicon_lookup = LexiconCollection.from_tsv(spanish_single_lexicon_url, include_pos=True)
    lemma_lexicon_lookup = LexiconCollection.from_tsv(spanish_single_lexicon_url, include_pos=False)
    mwe_lexicon_lookup = MWELexiconCollection.from_tsv(spanish_mwe_lexicon_url)
    # The rules that use the lexicons
    single_word_rule = SingleWordRule(lexicon_lookup, lemma_lexicon_lookup, UPOS_TO_USAS_CORE)
    mwe_word_rule = MWERule(mwe_lexicon_lookup, USAS_CORE_TO_UPOS)
    word_rules = [single_word_rule, mwe_word_rule]
    # The ranker that determines which rule should be used/applied
    ranker_arguments = ContextualRuleBasedRanker.get_construction_arguments(word_rules)
    ranker = ContextualRuleBasedRanker(*ranker_arguments)
    # POS that indicate a Punctuation and Numeric value
    default_punctuation_tags = list(['PUNCT'])
    default_number_tags = list(['NUM'])

    tagger = cast(HybridTagger, hybrid_nlp.add_pipe('pymusas_hybrid_tagger', config={"top_n": 3}))

    tagger.initialize(rules=word_rules,
                      ranker=ranker,
                      default_punctuation_tags=default_punctuation_tags,
                      default_number_tags=default_number_tags,
                      pretrained_model_name_or_path="ucrelnlp/PyMUSAS-Neural-Multilingual-Base-BEM")
    return hybrid_nlp

def get_spanish_sentence_splitter() -> Callable[[str], Iterable[str]]:
    return spacy_sentence_splitter(spacy.load('es_core_news_sm'))


def get_hindi_neural_tagger() -> spacy.Language:
    nlp = spacy.blank("xx")
    multilingyal_neural_tagger_pipeline = spacy.load("xx_none_none_none_multilingualbasebem",
                                                               config={"components.pymusas_neural_tagger.top_n": 3})
    nlp.add_pipe("pymusas_neural_tagger", source=multilingyal_neural_tagger_pipeline)
    return nlp


def get_hindi_stanza_tagger() -> StanzaPipeline:
    return StanzaPipeline('hi', processors='tokenize,lemma,pos')


def get_igbo_neural_tagger() -> spacy.Language:
    nlp = spacy.blank("xx")
    multilingyal_neural_tagger_pipeline = spacy.load("xx_none_none_none_multilingualbasebem",
                                                               config={"components.pymusas_neural_tagger.top_n": 3})
    nlp.add_pipe("pymusas_neural_tagger", source=multilingyal_neural_tagger_pipeline)
    return nlp

def get_all_malay_models() -> tuple[malaya.tokenizer.Tokenizer, malaya.tokenizer.SentenceTokenizer, MalayaStem, MalayaTagging]:
    tokenizer = malaya.tokenizer.Tokenizer()
    sentence_splitter = malaya.tokenizer.SentenceTokenizer()
    lemmatizer = malaya.stem.huggingface('mesolitica/stem-lstm-512', force_check=True)
    pos_tagger = malaya.pos.huggingface("mesolitica/pos-t5-small-standard-bahasa-cased", force_check=True)
    return (tokenizer, sentence_splitter, lemmatizer, pos_tagger)


def get_malay_hybrid_tagger() -> spacy.Language:
    nlp = spacy.blank("xx")

    malay_single_lexicon_url = ('https://raw.githubusercontent.com/UCREL/Multilingual-USAS/442dc9a975ea1c3b0db20246a5a58565a200d581/Malay/semantic_lexicon_ms.tsv')
    lemma_lexicon_lookup = LexiconCollection.from_tsv(malay_single_lexicon_url, include_pos=False)
    single_word_rule = SingleWordRule({}, lemma_lexicon_lookup)
    word_rules = [single_word_rule]
    ranker_arguments = ContextualRuleBasedRanker.get_construction_arguments(word_rules)
    ranker = ContextualRuleBasedRanker(*ranker_arguments)
    # POS that indicate a Punctuation and Numeric value
    default_punctuation_tags = list(['PUNCT'])
    default_number_tags = list(['NUM'])
    tagger = cast(HybridTagger, nlp.add_pipe('pymusas_hybrid_tagger', config={"top_n": 3}))

    tagger.initialize(rules=word_rules,
                      ranker=ranker,
                      default_punctuation_tags=default_punctuation_tags,
                      default_number_tags=default_number_tags,
                      pretrained_model_name_or_path="ucrelnlp/PyMUSAS-Neural-Multilingual-Base-BEM")
    return nlp