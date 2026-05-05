import subprocess
import sys
from enum import Enum
from typing import Annotated, Any, Callable, List

import spacy
import typer
from rich import print as rprint

import taggers

app = typer.Typer()

class Languages(str, Enum):
    en = "English"
    da = "Danish"
    nl = "Dutch"
    es = "Spanish"
    hi = "Hindi"
    ig = "Igbo"
    ms = "Malay"


class SpacyModel(str, Enum):
    en_sm  = "en_core_web_sm"
    en_trf = "en_core_web_trf"
    da_lg = "da_core_news_lg"
    nl_md = "nl_core_news_md"
    nl_lg = "nl_core_news_lg"
    es_sm = "es_core_news_sm"
    es_trf = "es_dep_news_trf"


LANGUAGE_2_SPACY_MODEL: dict[Languages, List[SpacyModel]] = {
    Languages.en: [SpacyModel.en_sm, SpacyModel.en_trf],
    Languages.da: [SpacyModel.da_lg],
    Languages.nl: [SpacyModel.nl_md, SpacyModel.nl_lg],
    Languages.es: [SpacyModel.es_sm, SpacyModel.es_trf]
}

LANGUAGE_2_PREPROCESS_CALLABLE: dict[Languages, Callable[[], Any]] = {
    Languages.en: taggers.get_english_sentence_splitter,
    Languages.da: taggers.get_danish_sentence_splitter,
    Languages.nl: taggers.get_dutch_sentence_splitter,
    Languages.es: taggers.get_spanish_sentence_splitter,
    Languages.hi: taggers.get_hindi_stanza_tagger,
    Languages.ms: taggers.get_all_malay_models
}

LANGUAGE_2_PYMUSAS_CALLABLE: dict[Languages, Callable[[], spacy.Language]] = {
    Languages.en: taggers.get_english_hybrid_tagger,
    Languages.da: taggers.get_danish_hybrid_tagger,
    Languages.nl: taggers.get_dutch_hybrid_tagger,
    Languages.es: taggers.get_spanish_hybrid_tagger,
    Languages.hi: taggers.get_hindi_neural_tagger,
    Languages.ig: taggers.get_igbo_neural_tagger,
    Languages.ms: taggers.get_malay_hybrid_tagger
}

LANGUAGE_2_PYMUSAS_SPACY_MODEL: dict[Languages, str] = {
    Languages.hi: "xx_none_none_none_multilingualbasebem",
    Languages.ig: "xx_none_none_none_multilingualbasebem"
}

PYMUSAS_SPACY_MODEL_2_URL: dict[str, str] = {
    "xx_none_none_none_multilingualbasebem": "https://github.com/UCREL/pymusas-models/releases/download/xx_none_none_none_multilingualbasebem-0.4.0/xx_none_none_none_multilingualbasebem-0.4.0-py3-none-any.whl"
}

    
SPACY_DESCRIPTIONS: dict[SpacyModel, str] = {
    SpacyModel.en_sm:  "English - Small (12MB)",
    SpacyModel.en_trf: "English - Transformer-based (438MB)",
    SpacyModel.da_lg:  "Danish - Large (540MB)",
    SpacyModel.nl_md:  "Dutch - Medium (40MB)",
    SpacyModel.nl_lg:  "Dutch - Large (541MB)",
    SpacyModel.es_sm:  "Spanish - Small (12MB)",
    SpacyModel.es_trf:  "Spanish - Transformer-based (388MB)",
}

PREPROCESS_DESCRIPTIONS: dict[Languages, str] = {
    Languages.hi: "Hindi (HDTB treebank) - (~200MB)",
    Languages.ms: "Malay (Lemmatizer + POS tagger) - (~200MB)"
}

PYMUSAS_SPACY_MODELS_DESCRIPTIONS: dict[Languages, str] = {
    Languages.en: "PyMUSAS English Base Neural Model - (242MB)",
    Languages.da: "PyMUSAS Multilingual Base Neural Model - (1089MB)",
    Languages.nl: "PyMUSAS Multilingual Base Neural Model - (1089MB)",
    Languages.es: "PyMUSAS Multilingual Base Neural Model - (1089MB)",
    Languages.hi: "PyMUSAS Multilingual Base Neural Model - (1089MB)",
    Languages.ig: "PyMUSAS Multilingual Base Neural Model - (1089MB)",
    Languages.ms: "PyMUSAS Multilingual Base Neural Model - (1089MB)"
}


def install_spacy_models(language: Languages):
    rprint(f"Installing {language} specific spaCy models")
    spacy_models = LANGUAGE_2_SPACY_MODEL[language]
    for model in spacy_models:
        if spacy.util.is_package(model):
            rprint(f"[green]✓ {model} is already installed[/green]")
            continue
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "download", model.value],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            rprint(f"[green]✓ {model.value} installed successfully[/green]")
        else:
            rprint(f"[red]✗ Failed to install {model.value}: {result.stderr}[/red]")
    rprint(f"Done installing {language} specific spaCy models")

def install_pymusas_spacy_model(language: Languages):
    rprint(f"Installing {language} specific PyMUSAS spaCy model")
    pymusas_spacy_model = LANGUAGE_2_PYMUSAS_SPACY_MODEL[language]
    if spacy.util.is_package(pymusas_spacy_model):
        rprint(f"[green]✓ {pymusas_spacy_model} is already installed[/green]")
        return
    model_url = PYMUSAS_SPACY_MODEL_2_URL[pymusas_spacy_model]
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", model_url],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        rprint(f"[green]✓ {pymusas_spacy_model} installed successfully[/green]")
    else:
        rprint(f"[red]✗ Failed to install {pymusas_spacy_model}: {result.stderr}[/red]")
    rprint(f"Done installing {language} specific PyMUSAS spaCy model")

@app.command()
def install(
    languages: Annotated[list[Languages] | None, typer.Option("--languages", "-l", help="Install the language specific models for the given languages.")] = None,
    all_languages: Annotated[bool, typer.Option("--all", "-a", help="Install all language specific models.")] = False,
    describe: Annotated[bool, typer.Option("--describe", "-d", help="Describe the models that will be installed and exit.")] = False
    ):
    """
    Install the language specific models. You can either select the languages you want to install or use the --all flag to install all language specific models.

    If you want to describe the models that will be installed use the --describe flag.

    Example:

    To install all language specific models run:
    python models_install.py --all

    To install only the English and Dutch language specific models run:
    python models_install.py -l English -l Dutch

    To describe the models that will be installed run:
    python models_install.py --describe

    To describe English specific models:
    python models_install.py -l English --describe
    """
    selected = languages
    if all_languages:
        selected = list(Languages)
    if selected is None:
        rprint("No languages selected, either use --languages or --all")
        raise typer.Exit(1)

    if describe:
        rprint("Describing the models that will be installed:")
        for language in selected:
            rprint(f"Models that will be installed for this language {language}:")
            if LANGUAGE_2_SPACY_MODEL.get(language):
                rprint("spaCy models that will be installed:")
                for model in LANGUAGE_2_SPACY_MODEL[language]:
                    rprint(f" {SPACY_DESCRIPTIONS[model]}")
            if PREPROCESS_DESCRIPTIONS.get(language):
                rprint("Preprocessing models that will be installed:")
                rprint(f"{PREPROCESS_DESCRIPTIONS[language]}")
            if PYMUSAS_SPACY_MODELS_DESCRIPTIONS.get(language):
                rprint("PyMUSAS spaCy models that will be installed:")
                rprint(f"{PYMUSAS_SPACY_MODELS_DESCRIPTIONS[language]}")
            rprint()
        rprint("Done")
                
        raise typer.Exit(0)

    rprint(f"Installing {len(selected)} language specific models, some languages do use more than one model")
    for language in selected:
        rprint(f"Installing {language} specific models")
        if LANGUAGE_2_SPACY_MODEL.get(language):
            install_spacy_models(language)
        if LANGUAGE_2_PYMUSAS_SPACY_MODEL.get(language):
            install_pymusas_spacy_model(language)
        if LANGUAGE_2_PREPROCESS_CALLABLE.get(language):
            rprint("[green]Installing preprocessing models[/green]")
            LANGUAGE_2_PREPROCESS_CALLABLE[language]()
        rprint("[green]Installing PyMUSAS specific models and lexicons[/green]")
        LANGUAGE_2_PYMUSAS_CALLABLE[language]()
        rprint(f"Done installing {language} specific models")
    rprint(f"Done installing {len(selected)} language specific models")
        
        

if __name__ == "__main__":
    app()