"""Main module for the application.
"""

import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from db_cacm import build_cacm_db, cacm_query_tests
from db_cisi import build_cisi_db, cisi_query_tests
from db_cran import build_cran_db, cran_query_tests
from db_lisa import build_lisa_db, lisa_query_tests
from db_med import build_med_db, med_query_tests
from db_npl import build_npl_db, npl_query_tests
from ir_model import DEFAULT_CONFIG, IRModel
from model_tester import ModelTester, QueryTest

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


def get_docs(database: str):
    """
    Gets the documents of a database.
    """
    db_folder = Path(f"./database/{database}")
    if not db_folder.exists():
        raise ValueError(f"Database {database} not found.")
    docs_path = db_folder / "docs.json"
    with open(docs_path, "r") as docs_f:
        return json.load(docs_f)


def get_doc(database: str, doc_idx: int) -> str:
    """
    Returns the content of a document
    """
    db_folder = Path(f"./database/{database}")
    if not db_folder.exists():
        raise ValueError(f"Database {database} does not exist")
    doc_path = db_folder / "docs.json"
    with open(doc_path, "r") as doc_f:
        return json.load(doc_f)[doc_idx]


def clear_evals(database: str):
    """
    Clears the evaluation results.
    """
    for root, _, files in os.walk(f"database/{database}/model/"):
        for file in files:
            if file.startswith("test_"):
                path = os.path.join(root, file)
                yield path
                os.remove(path)


def evaluate_model(
    database: str,
    force: bool = False,
    compare_tests: bool = False,
    configs: Optional[List[str]] = None,
):
    """
    Evaluates the model for a given database.
    """

    yield "Parsing query tests"
    query_tests = _DB_QUERY_TESTS[database]()

    db_folder = Path(f"./database/{database}")
    model = IRModel(str(db_folder))
    if configs is None or not configs:
        _test_model(model, query_tests, force, compare_tests)
        return

    for config in configs:
        yield f"\nBuilding model using {config}"
        build_database_model(database, config)
        model = IRModel(str(db_folder))
        _test_model(model, query_tests, True, compare, False)


def compare(models: Optional[List[str]] = None):
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
            raise FileNotFoundError(f"Model {model} not found.")

        last_test_path = None
        for root, _, files in os.walk(model_path):
            for file in files:
                if file.startswith("test_"):
                    last_test_path = os.path.join(root, file)
                    ret_vals.append(np.load(last_test_path))
                    break
        if last_test_path is None:
            raise ValueError(f"No test file found in {model_path}.")

    ModelTester.plot_mean_results(ret_vals, models, _TOPS)
    yield None


def _test_model(
    model: IRModel,
    query_tests: List[QueryTest],
    force: bool,
    compare_tests: bool,
    show: bool = True,
):
    """
    Tests the model for a given database.
    """
    tester = ModelTester(model, force, compare_tests)
    tester.test(query_tests, _TOPS, show)


def single_query(query: str, model: IRModel):
    """
    Process a single query
    """
    # Search documents
    results = model.search(query)

    # Display results
    for res in results:
        try:
            score = float(res["weight"])
            if score > 0:
                yield res
        except ValueError:
            pass


def generate_config(output: Optional[str] = "./config.json", force: bool = False):
    """
    Generates a configuration file.
    """
    if Path(output).exists() and not force:
        raise ValueError(f"File {output} already exists.")
    with open(output, "w") as config_f:
        json.dump(DEFAULT_CONFIG, config_f, indent=4)


def build_database(database: str, force: bool = False):
    """
    Builds a database
    """
    db_folder = Path(f"./database/{database}")
    if not db_folder.exists() or force:
        if database not in _BUILD_IN_DATABASES:
            raise ValueError(
                f"Rebuild not suported for database '{database}'.\n\n"
                "Please build your own database using the 'DatabaseCreator'\n"
                "class available in 'database_creator.py'"
            )
        _DB_BUILDERS[database]()
    else:
        raise ValueError(f"Database {database} already exists")
    yield None


def build_database_model(
    database: str, config_file: Optional[str] = None, force: bool = False
):
    """
    Builds the model for a database
    """
    db_folder = Path(f"./database/{database}")
    model_folder = db_folder / "model"
    if not db_folder.exists():
        raise ValueError(f"Database {database} does not exist")
    if not model_folder.exists() or force:
        IRModel(str(db_folder), True, config_file)
    else:
        raise ValueError(f"Database {database} already has a model folder")
    yield None
