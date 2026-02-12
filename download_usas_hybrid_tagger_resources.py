import logging
from typing import Annotated

import typer

import taggers

logger = logging.getLogger(__name__)


def main(verbose_logging: Annotated[bool, typer.Option(help="Print verbose logging")] = False):
    """
    Downloads all of the USAS tagger resources if they have not already been downloaded.
    """
    if verbose_logging:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)

    logger.info("Downloading the USAS tagger resources if they have not already been downloaded")

    logger.info("Downloading the English Tagger")
    _ = taggers.get_english_hybrid_tagger()
    _ = taggers.get_english_sentence_splitter()

    logger.info("Downloading the Spanish Tagger")
    _ = taggers.get_spanish_hybrid_tagger()
    _ = taggers.get_spanish_sentence_splitter()

    logger.info("Downloading the Danish Tagger")
    _ = taggers.get_danish_hybrid_tagger()
    _ = taggers.get_danish_sentence_splitter()


    logger.info("Downloading the Dutch Tagger")
    _ = taggers.get_dutch_hybrid_tagger()
    _ = taggers.get_dutch_sentence_splitter()

    logger.info("Finished downloading all of the USAS tagger resources")
        

if __name__ == "__main__":
    typer.run(main)