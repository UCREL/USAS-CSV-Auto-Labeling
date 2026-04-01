import logging
import os
import re
import string
from pathlib import Path
from typing import Annotated, Callable, Iterable

import spacy
import typer
import xlsxwriter
from stanza.pipeline.core import Pipeline as StanzaPipeline

import taggers
from usas_csv_auto_labeling.data_utils import load_usas_mapper
from usas_csv_auto_labeling.processing_text import (
    TaggedText,
    tag_igbo_text,
    tag_text,
    tag_text_with_stanza,
)

logger = logging.getLogger(__name__)

def igbo_tagging(pymusas_tagger: spacy.Language) -> Callable[[str], Iterable[TaggedText]]:
    def _tag_text(text: str) -> Iterable[TaggedText]:
        return tag_igbo_text(text, pymusas_tagger)

    return _tag_text

def stanza_tagging(stanza_tagger: StanzaPipeline,
                   pymusas_tagger: spacy.Language
                   ) -> Callable[[str], Iterable[TaggedText]]:
    def _tag_text(text: str) -> Iterable[TaggedText]:
        return tag_text_with_stanza(text, stanza_tagger, pymusas_tagger)

    return _tag_text

def spacy_tagging(tagger: spacy.Language,
                  sentence_splitter: Callable[[str], Iterable[str]] | None = None,
                  lemma_token_extension: str | None = None, # lemma_
                  pos_token_extension: str | None = None, # pos_
                  token_text_extension: str = "text",
                  usas_token_extension: str = "_.pymusas_tags",
                  mwe_token_extension: str = "_.pymusas_mwe_indexes") -> Callable[[str], Iterable[TaggedText]]:
    def _tag_text(text: str) -> Iterable[TaggedText]:
        return tag_text(text, tagger, sentence_splitter, lemma_token_extension, pos_token_extension, token_text_extension, usas_token_extension, mwe_token_extension)

    return _tag_text

def tag_to_excel_sheet(input_data_file_path: Path,
                       output_excel_file_path: Path,
                       tagging_function: Callable[[str], Iterable[TaggedText]],
                       language: str,
                       wikipedia_article_name: str) -> None:
    usas_labels_and_descriptions = load_usas_mapper(usas_tag_descriptions_file=None,
                                                                    tags_to_filter_out=set(["Z99"]))

    headers = ["id",
                                    "sentence id",
                                    "token id",
                                    "token",
                                    "lemma",
                                    "POS",
                                    "predicted USAS",
                                    "predicted MWE",
                                    "corrected USAS",
                                    "corrected MWE"]
                        
    with input_data_file_path.open("r", encoding="utf-8") as f:
        with xlsxwriter.Workbook(str(output_excel_file_path),
                                 {'constant_memory': True}) as workbook:
            worksheet = workbook.add_worksheet()
            bold = workbook.add_format({'bold': 1})
            for header, column_letter in zip(headers, string.ascii_uppercase):
                worksheet.write(f'{column_letter}1', header, bold)


            text = f.read()
            text = re.sub(r"\[\d+\]", "", text)
            attribute_none_error_string = ("{attribute} cannot be found this is "
                                           "likely because the tagger does not "
                                           "have a {attribute} tagger, or the "
                                           "{attribute} attribute given does "
                                           "not match the tagger's, error "
                                           "occurred for file: {input_file_name}")
            
            worksheet_row_index = 2
            no_lemmas = False
            no_pos_tags = False
            for sentence_id, tagged_text in enumerate(tagging_function(text)):
                if tagged_text.lemmas is None and not no_lemmas:
                    no_lemmas = True
                    logger.info(attribute_none_error_string.format(attribute="lemma", input_file_name=input_data_file_path))
                if tagged_text.pos_tags is None and not no_pos_tags:
                    no_pos_tags = True
                    logger.info(attribute_none_error_string.format(attribute="POS tag", input_file_name=input_data_file_path))
                
                for token_id in range(len(tagged_text.tokens)):
                    token = tagged_text.tokens[token_id]
                    
                    lemma = "-"
                    if tagged_text.lemmas is not None:
                        lemma = tagged_text.lemmas[token_id]
                    
                    pos_tag = "-"
                    if tagged_text.pos_tags is not None:
                        pos_tag = tagged_text.pos_tags[token_id]
                    
                    usas_tag_groups = tagged_text.usas_tags[token_id]
                    usas_tag_strings = []
                    for usas_tag_group in usas_tag_groups:
                        usas_tags = []
                        for usas_tag in usas_tag_group.tags:
                            if usas_tag.tag == "Z99":
                                continue
                            elif usas_tag.tag == "PUNCT":
                                pass
                            elif usas_tag.tag not in usas_labels_and_descriptions:
                                continue
                            usas_tags.append(usas_tag.tag)
                        usas_tag_strings.append("/".join(usas_tags))

                    usas_tag_string = "; ".join(usas_tag_strings)
                    
                    
                    mwe_indexes = tagged_text.mwe_indexes[token_id]

                    mwe_index_string = ""
                    if mwe_indexes:
                        mwe_index_string = ";".join(str(index) for index in mwe_indexes)

                    _id = f"{language}|{wikipedia_article_name}|{sentence_id}|{token_id}"
                    worksheet.write_string(f"A{worksheet_row_index}", _id)
                    worksheet.write_number(f"B{worksheet_row_index}", sentence_id)
                    worksheet.write_number(f"C{worksheet_row_index}", token_id)
                    worksheet.write_string(f"D{worksheet_row_index}", token)
                    worksheet.write_string(f"E{worksheet_row_index}", lemma)
                    worksheet.write_string(f"F{worksheet_row_index}", pos_tag)
                    worksheet.write_string(f"G{worksheet_row_index}", usas_tag_string)
                    worksheet.write_string(f"H{worksheet_row_index}", mwe_index_string)
                    worksheet_row_index += 1
                # Skip a row after each sentence
                worksheet_row_index += 1

def traverse_directory(directory: Path) -> Iterable[Path]:
    """
    Yields all files in a directory tree, through recursive search, that ends with '.txt'.

    Args:
        directory: The root directory to traverse
    Yields:
        Iterable[Path]: An iterable of resolved paths to .txt files
    """
    for root, dirs, files in os.walk(str(directory.resolve())):
        if not files:
            continue
        for file in files:
            if not file.endswith('.txt'):
                continue
            yield Path(root, file).resolve()


def get_language_tagging_function(language: str) -> Callable[[str], Iterable[TaggedText]]:

    def get_spacy_tagging_function(tagger: spacy.Language, sentence_splitter: Callable[[str], Iterable[str]] | None = None) -> Callable[[str], Iterable[TaggedText]]:
        return spacy_tagging(tagger, sentence_splitter,
                             lemma_token_extension="lemma_", pos_token_extension="pos_",
                             token_text_extension="text",
                             usas_token_extension="_.pymusas_tags",
                             mwe_token_extension="_.pymusas_mwe_indexes")

    language = language.strip().lower()
    supported_languages = set({
        "english",
        "dutch",
        "spanish",
        "danish",
        "hindi",
        "igbo"
    })

    match language:
        case "english":
            english_tagger = taggers.get_english_hybrid_tagger()
            english_sentence_splitter = taggers.get_english_sentence_splitter()
            tagging_function = get_spacy_tagging_function(english_tagger, english_sentence_splitter)
            return tagging_function
        case "spanish":
            spanish_tagger = taggers.get_spanish_hybrid_tagger()
            spanish_sentence_splitter = taggers.get_spanish_sentence_splitter()
            tagging_function = get_spacy_tagging_function(spanish_tagger, spanish_sentence_splitter)
            return tagging_function
        case "danish":
            danish_tagger = taggers.get_danish_hybrid_tagger()
            danish_sentence_splitter = taggers.get_danish_sentence_splitter()
            tagging_function = get_spacy_tagging_function(danish_tagger, danish_sentence_splitter)
            return tagging_function
        case "dutch":
            dutch_tagger = taggers.get_dutch_hybrid_tagger()
            dutch_sentence_splitter = taggers.get_dutch_sentence_splitter()
            tagging_function = get_spacy_tagging_function(dutch_tagger, dutch_sentence_splitter)
            return tagging_function
        case "hindi":
            hindi_tagger = taggers.get_hindi_neural_tagger()
            hindi_stanza_tagger = taggers.get_hindi_stanza_tagger()
            tagging_function = stanza_tagging(hindi_stanza_tagger, hindi_tagger)
            return tagging_function
        case "igbo":
            igbo_tagger = taggers.get_igbo_neural_tagger()
            return igbo_tagging(igbo_tagger)
        case _:
            raise ValueError(f"Language {language} is not supported, "
                             f"supported languages: {supported_languages}")
    

def main(data_path: Annotated[Path, typer.Argument(help="Path to the data directory", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
         output_path: Annotated[Path, typer.Argument(help="Path to the output directory", exists=False, resolve_path=True)],
         verbose_logging: Annotated[bool, typer.Option(help="Print verbose logging")] = False,
         overwrite: Annotated[bool, typer.Option(help="If the output path exists overwrite all files in it")] = False):
    """
    Tag all of the files in the given data directory (`data_path`) with pre downloaded language taggers
    and write the results to the given output directory (`output_path`), in the same file structure
    as the data directory, in excel format.

    The Excel file has the following columns:

    | id | sentence id | token id | token | lemma | POS | predicted USAS | predicted MWE | corrected USAS | corrected MWE |

    whereby all but the `corrected` columns are filled in by the taggers.
    The `id` is in the following format `{language}|{wikipedia_article_name}|{sentence_id}|{token_id}`

    The data directory file structure should be as follows:
    
    data_path
    |
    |__ language
    |   |
    |   |__ wikipedia_article_name
    |   |   |
    |   |   |__ file_name.txt

    Whereby the `language` is used to determine which tagger to use and both
    the `language` and `wikipedia_article_name` are added to the ID of each token
    tagged and written to the excel output file.

    Languages supported:
    * english
    * dutch
    * spanish
    * danish
    * hindi
    * igbo
    """
    if output_path.exists() and not overwrite:
        raise ValueError(f"Output path {output_path} already exists and overwrite is false "
                         "either delete the output path, "
                         "choose a different output path or set overwrite to true")

    if verbose_logging:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)

    logger.info(f"Tagging all files in {data_path} and writing to {output_path}")

    for data_file in traverse_directory(data_path):

        logger.info(f"Processing file: {data_file}")
        
        language = data_file.parent.name
        wikipedia_article_name = data_file.stem
        output_file_name = f"{wikipedia_article_name}.xlsx"
        
        relative_directory =os.path.relpath(str(data_file.parent), str(data_path))
        output_file = output_path / relative_directory / output_file_name
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.exists() and overwrite:
            output_file.unlink()
        
        tagging_function = get_language_tagging_function(language)
        tag_to_excel_sheet(data_file, output_file, tagging_function, language, wikipedia_article_name)
    
    logger.info(f"Finished processing all files in {data_path}")
        

if __name__ == "__main__":
    typer.run(main)