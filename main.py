"""
Main module for the application.
"""

import json
from pathlib import Path
from pprint import pprint
from typing import Optional

import typer

from cran_db_creator import create_db as create_cran_db
from ir_model import DEFAULT_CONFIG, IRModel
from query import Query

app = typer.Typer(add_completion=False)

status = {}

_BUILD_IN_DATABASES = ["cran"]


@app.command("evaluate")
def evaluate_model():
    """
    Evaluates the model for a given database.
    """
    database = status["database"]
    if database not in _BUILD_IN_DATABASES:
        raise typer.Exit(f"Database {database} is not supported for evaluation")
    raise typer.Exit("Not implemented")


@app.command("single")
def single_query(query: str):
    """
    Process a single query
    """
    # Search documents
    model = status["model"]
    results = model.search(query)

    # Display results
    for res in results:
        results.show_result(res)
        opt = input("See next result? [Y/n]:").lower()
        if opt == "n":
            break


@app.command("continuous")
def continuous_queries():
    """
    Continuously read user queries, search documents and display results
    """
    while True:
        query = input("Enter query: ")
        if query in ("", "exit"):
            break
        single_query(query)


@app.command("build")
def build_database_model(
    database: str,
    config_file: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to the configuration file",
    ),
):
    """
    Builds the model for a database

    WARNING: this will overwrite the existing model
    """
    db_folder = Path(f"./database/{database}")
    IRModel(str(db_folder), True, config_file)


@app.command("gen-config")
def generate_config(
    output: Optional[str] = typer.Option(
        "./config.json",
        "--output",
        "-o",
        help="Path to the output file",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing configuration file",
    ),
):
    """
    Generates a configuration file.

    The configuration file is a JSON file containing the default configuration
    for the IR model building the model and querying.


    Available configuration options:

    \b
    -- 'tokenization_method' (str): The tokenization method to use.

    \b
        Possible values:
            'split'   Uses 'split' function from str.
            'nltk'    Uses 'word_tokenizer' from nltk.
    \b
    -- 'include_metadata' (List[str]): List of metadata keys. The metadata
        value of the specified keys will be included in the text used for
        building the model. If key does not exist, it will be ignored. By
        default, metadata is not included (Empty list).
    """
    if Path(output).exists() and not force:
        raise typer.Exit(f"File {output} already exists\n\nUse --force to overwrite")
    with open(output, "w") as config_f:
        json.dump(DEFAULT_CONFIG, config_f, indent=4)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    database: Optional[str] = typer.Option(
        default="cran",
        help="Name of the database to use",
    ),
    rebuild: Optional[bool] = typer.Option(
        default=False,
        help="Rebuild the database (generate 'docs.json' and 'metadata.json')",
    ),
):
    """
    SRI Final Project

    By default the progrma will run the continuous queries command.

    Authors: Jorge Morgado Vega (jorge.morgadov@gmail.com) and Roberto
    García Rodríguez (roberto.garcia@estudiantes.matcom.uh.cu)
    """

    # Load the model according to the database and rebuild parameters
    db_folder = Path(f"./database/{database}")
    if not db_folder.exists() or rebuild:
        if not rebuild:
            raise typer.Exit(
                f"The database '{database}' does not exist. Use the --rebuild\n"
                "option to create it.\n\n"
                "NOTE: The --rebuild option is only available for the\n"
                "cran database."
            )
        if database not in ("cran",):
            raise typer.Exit(
                f"Rebuild not suported for database '{database}'.\n\n"
                "Please build your own database using the 'DatabaseCreator'\n"
                "class available in 'database_creator.py'"
            )
        typer.echo(f"Rebuilding the '{database}' database")
        if database == "cran":
            create_cran_db()
    if ctx.invoked_subcommand != "build":
        status["model"] = IRModel(str(db_folder))
    status["database"] = database
    # Run the continuous queries command by default
    if ctx.invoked_subcommand is None:
        continuous_queries()


if __name__ == "__main__":
    app()
