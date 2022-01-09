"""
Main module for the application.
"""

from pathlib import Path
from typing import List, Optional

import typer

import main_api as api
from ir_model import IRModel
from query import QueryResult

app = typer.Typer(add_completion=False)

status = {}

_BUILD_IN_DATABASES = ["cran", "cisi", "med"]


@app.command("clear-evals")
def clear_evals(database: str):
    """
    Clears the evaluation results.
    """
    if not typer.confirm(
        f"Are you sure you want to clear the evaluation results for {database}?",
        default=True,
    ):
        return
    for deleted_test in api.clear_evals(database):
        typer.echo(f"Removed: {deleted_test}")


@app.command("evaluate")
def evaluate_model(
    database: str,
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-calcualtion of the model tests",
    ),
    compare: bool = typer.Option(
        False,
        "--compare",
        "-cm",
        help="Compare the current results with previous one (if they exist)",
    ),
    configs: Optional[List[str]] = typer.Option(
        None,
        "--configs",
        "-cf",
        help=("Path to the config file(s) to use."),
    ),
):
    """
    Evaluates the model for a given database.

    \b
    Parameters of evaluation:
        - Presision
        - Recall
        - F1
        - Fallout
        - Presition vs. Recall
        - Scores

    Each of these parameters (along with std, min and max values) are estimated
    for diferents top k values (2, 4, 6, ..., 100).

    NOTE: If you specify a config file (o several), each one will be used to
    build a model and evaluate it. The order of the configs will
    determin the orden of evaluation. The last model built (last
    config) will be setted as the current model for that database.

    WARNING: The current model will be overwritten.
    """
    try:
        for msg in api.evaluate_model(
            database,
            force,
            compare,
            configs,
        ):
            typer.echo(msg)
    except Exception as e:
        typer.echo(e)


@app.command("compare")
def compare(
    models: Optional[List[str]] = typer.Option(
        None,
        "--models",
        "-m",
        help="Names of the databases to compare.",
    ),
):
    """
    Compares the results of the models of diferents databases.
    """

    try:
        for msg in api.compare(models):
            typer.echo(msg)
    except Exception as e:
        typer.echo(str(e))


@app.command("single")
def single_query(query: str):
    """
    Process a single query
    """
    # Search documents
    model = status["model"]
    # Display results
    for res in api.single_query(query, model):
        QueryResult.show_result(res)
        if not typer.confirm("See next result?", default=True):
            break


@app.command("continuous")
def continuous_queries():
    """
    Continuously read user queries, search documents and display results
    """
    while True:
        query = input("Enter query: ")
        if query == "":
            break
        single_query(query)


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
    try:
        api.generate_config(output, force)
    except Exception as e:
        typer.echo(e)


@app.command("build-db")
def build_database(
    database: str,
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing configuration file",
    ),
):
    """
    Builds a database (Only supported for: 'cran')

    The only supported database for building is 'cran'. If you want to build a
    database other than 'cran', you can do it manually or by using the
    'DatabaseCreator.create(...)' method.
    """
    try:
        api.build_database(database, force)
    except Exception as e:
        typer.echo(e)


@app.command("build-model")
def build_database_model(
    database: str,
    config_file: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to the configuration file",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing configuration file",
    ),
):
    """
    Builds the model for a database
    """
    try:
        for msg in api.build_database_model(database, config_file, force):
            typer.echo(msg)
    except Exception as e:
        typer.echo(e)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    database: Optional[str] = typer.Option(
        default="cran",
        help="Name of the database to use",
    ),
):
    """
    SRI Final Project

    By default the progrma will run the continuous queries command.

    \b
    Authors:
        - Jorge Morgado Vega (jorge.morgadov@gmail.com)
        - Roberto García Rodríguez (roberto.garcia@estudiantes.matcom.uh.cu)
    """

    db_folder = Path(f"./database/{database}")
    status["database"] = database
    invoked_cmd = ctx.invoked_subcommand

    # Load the model if command is not build
    if invoked_cmd is None or invoked_cmd in ["continuous", "single"]:
        status["model"] = IRModel(str(db_folder))

    # Run the continuous queries command by default
    if ctx.invoked_subcommand is None:
        continuous_queries()


if __name__ == "__main__":
    app()
