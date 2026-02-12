import logging
import os
import re
import string
from pathlib import Path
from typing import Annotated, Callable, Iterable

import spacy
import typer
import xlsxwriter

import taggers
from usas_csv_auto_labeling.data_utils import load_usas_mapper
from usas_csv_auto_labeling.processing_text import tag_text

logger = logging.getLogger(__name__)

def tag_to_excel_sheet(input_data_file_path: Path,
                       output_excel_file_path: Path,
                       usas_tagger: spacy.Language,
                       sentence_splitter: Callable[[str], Iterable[str]],
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
    lemma_attribute_name = "lemma_"
    pos_tag_attribute_name = "pos_"
                        
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
            for sentence_id, tagged_text in enumerate(tag_text(text, usas_tagger, sentence_splitter, lemma_attribute_name, pos_tag_attribute_name)):
                if tagged_text.lemmas is None:
                    raise ValueError(attribute_none_error_string.format(attribute="lemma", input_file_name=input_data_file_path))
                if tagged_text.pos_tags is None:
                    raise ValueError(attribute_none_error_string.format(attribute="POS tag", input_file_name=input_data_file_path))
                
                for token_id in range(len(tagged_text.tokens)):
                    token = tagged_text.tokens[token_id]
                    lemma = tagged_text.lemmas[token_id]
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

def main(data_path: Annotated[Path, typer.Argument(help="Path to the data directory", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
         output_path: Annotated[Path, typer.Argument(help="Path to the output directory", exists=False, resolve_path=True)],
         verbose_logging: Annotated[bool, typer.Option(help="Print verbose logging")] = False,
         overwrite: Annotated[bool, typer.Option(help="If the output path exists overwrite all files in it")] = False):
    """
    Tag all of the files in the given data directory (`data_path`) with pre loaded language taggers
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
    """
    if output_path.exists() and not overwrite:
        raise ValueError(f"Output path {output_path} already exists and overwrite is false "
                         "either delete the output path, "
                         "choose a different output path or set overwrite to true")

    if verbose_logging:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)

    logger.info("Loading all of the taggers...")

    english_tagger = taggers.get_english_hybrid_tagger()
    english_sentence_splitter = taggers.get_english_sentence_splitter()
    
    spanish_tagger = taggers.get_spanish_hybrid_tagger()
    spanish_sentence_splitter = taggers.get_spanish_sentence_splitter()

    danish_tagger = taggers.get_danish_hybrid_tagger()
    danish_sentence_splitter = taggers.get_danish_sentence_splitter()
    
    dutch_tagger = taggers.get_dutch_hybrid_tagger()
    dutch_sentence_splitter = taggers.get_dutch_sentence_splitter()

    logger.info("Finnish loading all of the taggers")

    lang_directory_tagger_mapper = {
        "english": english_tagger,
        "dutch": dutch_tagger,
        "spanish": spanish_tagger,
        "danish": danish_tagger
    }

    lang_directory_sentence_splitter_mapper = {
        "english": english_sentence_splitter,
        "dutch": dutch_sentence_splitter,
        "spanish": spanish_sentence_splitter,
        "danish": danish_sentence_splitter
    }

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

        language_tagger = lang_directory_tagger_mapper[language]
        language_sentence_splitter = lang_directory_sentence_splitter_mapper[language]
        
        tag_to_excel_sheet(data_file, output_file, language_tagger, language_sentence_splitter, language, wikipedia_article_name)
    
    logger.info(f"Finished processing all files in {data_path}")
        

if __name__ == "__main__":
    typer.run(main)