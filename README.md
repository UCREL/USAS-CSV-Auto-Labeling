# USAS-CSV-Auto-Labeling


Tool that annotates data with USAS labels for human verification in Excel format (CSV format will be added at a later point).

Currently the tool is very specific in that it only supports English, Spanish, Danish, and Dutch and it requires the data to be in a specific format.

Before you can run the tool please follow the [setup guide](#setup) and [install the models](#models-to-install). 


## Setup

You can either use the dev container with your favourite editor, e.g. VSCode. Or you can create your setup locally below we demonstrate both.

In both cases they share the same tools, of which these tools are:
* [uv](https://docs.astral.sh/uv/) for Python packaging and development
* [make](https://www.gnu.org/software/make/) (OPTIONAL) for automation of tasks, not strictly required but makes life easier.

### Dev Container

A [dev container](https://containers.dev/) uses a docker container to create the required development environment, the Dockerfile we use for this dev container can be found at [./.devcontainer/Dockerfile](./.devcontainer/Dockerfile). To run it locally it requires docker to be installed, you can also run it in a cloud based code editor, for a list of supported editors/cloud editors see [the following webpage.](https://containers.dev/supporting)

To run for the first time on a local VSCode editor (a slightly more detailed and better guide on the [VSCode website](https://code.visualstudio.com/docs/devcontainers/tutorial)):
1. Ensure docker is running.
2. Ensure the VSCode [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension is installed in your VSCode editor.
3. Open the command pallete `CMD + SHIFT + P` and then select `Dev Containers: Rebuild and Reopen in Container`

You should now have everything you need to develop, `uv`, `make`, for VSCode various extensions like `Pylance`, etc.

If you have any trouble see the [VSCode website.](https://code.visualstudio.com/docs/devcontainers/tutorial).

### Local

To run locally first ensure you have the following tools installted locally:
* [uv](https://docs.astral.sh/uv/getting-started/installation/) for Python packaging and development. (version `0.9.6`)
* [make](https://www.gnu.org/software/make/) (OPTIONAL) for automation of tasks, not strictly required but makes life easier.
  * Ubuntu: `apt-get install make`
  * Mac: [Xcode command line tools](https://mac.install.guide/commandlinetools/4) includes `make` else you can use [brew.](https://formulae.brew.sh/formula/make)
  * Windows: Various solutions proposed in this [blog post](https://earthly.dev/blog/makefiles-on-windows/) on how to install on Windows, inclduing `Cygwin`, and `Windows Subsystem for Linux`.

When developing on the project you will want to install the Python package locally in editable format with all the extra requirements, this can be done like so:

```bash
uv sync
```

### Linting

Linting and formatting with [ruff](https://docs.astral.sh/ruff/) it is a replacement for tools like Flake8, isort, Black etc, and we us [ty](https://github.com/astral-sh/ty) for type checking.

To run the linting:

``` bash
make lint
```

### Tests

To run the tests (uses pytest and coverage) and generate a coverage report:

``` bash
make test
```

### Models to install

The following spaCy models are required to tag and sentence split the data:

``` bash
uv run python -m spacy download en_core_web_sm
uv run python -m spacy download en_core_web_trf
uv run python -m spacy download da_core_news_lg
uv run python -m spacy download nl_core_news_md
uv run python -m spacy download nl_core_news_lg
uv run python -m spacy download es_core_news_sm
uv run python -m spacy download es_dep_news_trf
```

The following will download all of the resources (lexicons and neural models) to run the [Hybrid USAS tagger](https://ucrel.github.io/pymusas/#hybrid) for each language:

``` bash
uv download_usas_hybrid_tagger_resources.py
```

## Tools

### Tag data with USAS labels into Excel format

This tool tags all text files in a given directory (following the format specified in the help shown below) taking into account the language of the text file and outputs an Excel spreadsheet per text file in a given output directory. The Excel spreadsheet will allow annotators to correct the USAS tags and Multi Word Expression (MWE) indexes produced by the USAS tagger, allowing you to create a Gold labelled USAS tagged and MWE indexed dataset that can be used for evaluating and/or training a USAS tagger on the data of your choice.

Below is the help guide for the tool:

``` bash
uv run tag_data_to_excel.py --help
                                                                                                                                                                                                                                                                                                                                                                                                                                         
 Usage: tag_data_to_excel.py [OPTIONS] DATA_PATH OUTPUT_PATH                                                                                                                                                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                                                                                                                                                         
 Tag all of the files in the given data directory (`data_path`) with pre loaded language taggers and write the results to the given output directory (`output_path`), in the same file structure as the data directory, in excel format.                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                                                                                                                                                                         
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
                                                                                                                                                                                                                                                                                                                                                                                                                                         
╭─ Arguments ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    data_path        DIRECTORY  Path to the data directory [required]                                                                                                                                                                                                                                                                                                                                                                │
│ *    output_path      PATH       Path to the output directory [required]                                                                                                                                                                                                                                                                                                                                                              │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --verbose-logging    --no-verbose-logging      Print verbose logging [default: no-verbose-logging]                                                                                                                                                                                                                                                                                                                                    │
│ --overwrite          --no-overwrite            If the output path exists overwrite all files in it [default: no-overwrite]                                                                                                                                                                                                                                                                                                            │
│ --help                                         Show this message and exit.                                                                                                                                                                                                                                                                                                                                                            │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```
