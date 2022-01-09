"""
Main module for the application.
"""

import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import typer

from db_cacm import build_cacm_db, cacm_query_tests
from db_cisi import build_cisi_db, cisi_query_tests
from db_cran import build_cran_db, cran_query_tests
from db_lisa import build_lisa_db, lisa_query_tests
from db_med import build_med_db, med_query_tests
from db_npl import build_npl_db, npl_query_tests
from ir_model import DEFAULT_CONFIG, IRModel
from model_tester import ModelTester, QueryTest

app = typer.Typer(add_completion=False)

status = {}

_BUILD_IN_DATABASES = ["cran", "cisi", "med"]

_DB_BUILDERS = {
    "cacm": build_cacm_db,
    "cisi": build_cisi_db,
    "cran": build_cran_db,
    "med": build_med_db,
    "npl": build_npl_db,
    "lisa": build_lisa_db,
}

_DB_QUERY_TESTS = {
    "cacm": cacm_query_tests,
    "cisi": cisi_query_tests,
    "cran": cran_query_tests,
    "med": med_query_tests,
    "npl": npl_query_tests,
    "lisa": lisa_query_tests,
}

_TOPS = list(range(2, 100, 2))

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
    for root, _, files in os.walk(f"database/{database}/model/"):
        for file in files:
            if file.startswith("test_"):
                typer.echo(f"Removing {root}{file}")
                os.remove(os.path.join(root, file))


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
    if database not in _BUILD_IN_DATABASES:
        raise typer.Exit(f"Database {database} is not supported for evaluation")

    typer.echo("Parsing query tests")
    query_tests = _DB_QUERY_TESTS[database]()

    db_folder = Path(f"./database/{database}")
    model = IRModel(str(db_folder))
    if configs is None or not configs:
        _test_model(model, query_tests, force, compare)
        return

    for config in configs:
        typer.echo(f"\nBuilding model using {config}")
        build_database_model(database, config)
        model = IRModel(str(db_folder))
        _test_model(model, query_tests, True, compare, False)


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
    use_all = models is None or not models

    if use_all:
        for _, dirs, _ in os.walk("./database"):
            models = dirs
            break

    ret_vals = []
    for model in models:
        model_path = Path(f"./database/{model}/model")
        if not model_path.exists():
            typer.Exit(f"Model {model} does not exist")

        last_test_path = None
        for root, _, files in os.walk(model_path):
            for file in files:
                if file.startswith("test_"):
                    last_test_path = os.path.join(root, file)
                    ret_vals.append(np.load(last_test_path))
                    break
        if last_test_path is None:
            typer.Exit(f"Model {model} does not have any tests")

    ModelTester.plot_mean_results(ret_vals, models, _TOPS)


def _test_model(
    model: IRModel,
    query_tests: List[QueryTest],
    force: bool,
    compare: bool,
    show: bool = True,
):
    """
    Tests the model for a given database.
    """
    tester = ModelTester(model, force, compare)
    tester.test(query_tests, _TOPS, show)


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
    if Path(output).exists() and not force:
        raise typer.Exit(f"File {output} already exists\n\nUse --force to overwrite")
    with open(output, "w") as config_f:
        json.dump(DEFAULT_CONFIG, config_f, indent=4)


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
    db_folder = Path(f"./database/{database}")
    if not db_folder.exists() or force:
        if database not in _BUILD_IN_DATABASES:
            raise typer.Exit(
                f"Rebuild not suported for database '{database}'.\n\n"
                "Please build your own database using the 'DatabaseCreator'\n"
                "class available in 'database_creator.py'"
            )
        typer.echo(f"Building the '{database}' database")
        _DB_BUILDERS[database]()
    else:
        raise typer.Exit(
            f"Database {database} already exists\n\nUse --force to overwrite"
        )


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
    db_folder = Path(f"./database/{database}")
    model_folder = db_folder / "model"
    if not db_folder.exists():
        raise typer.Exit(f"Database {database} does not exist")
    if not model_folder.exists() or force:
        IRModel(str(db_folder), True, config_file)
    else:
        raise typer.Exit(
            f"Database {database} already has a model folder\n\n"
            "Use --force to overwrite"
        )


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
    if invoked_cmd in ["continuous", "single"]:
        status["model"] = IRModel(str(db_folder))

    # Run the continuous queries command by default
    if ctx.invoked_subcommand is None:
        continuous_queries()


if __name__ == "__main__":
    app()
