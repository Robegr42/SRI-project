from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import typer

from database_builder import DatabaseBuilder
from model_tester import QueryTest


def build_npl_db():
    """
    Creates the npl database.
    """
    metadata: List[Dict[str, Any]] = []
    texts: List[str] = []
    npl_file = Path("./test_collections/npl/doc-text")
    if not npl_file.exists():
        raise FileNotFoundError(f"{npl_file} does not exist.")

    with open(npl_file, "r") as npl_f:
        current_id = None
        text = []
        for line in npl_f:
            stripped = line.strip()
            if stripped == "/":
                continue
            if stripped.isnumeric():
                if current_id is not None:
                    metadata.append({"id": current_id})
                    texts.append(" ".join(text))
                    text = []
                current_id = int(stripped)
                continue
            text.append(stripped)
        if text:
            metadata.append({"id": current_id})
            texts.append(" ".join(text))

    DatabaseBuilder.build("npl", metadata, texts)


def npl_query_tests() -> List[QueryTest]:
    """
    Runs the query tests.
    """
    queries_file = Path("./test_collections/npl/query-text")
    relevants_file = Path("./test_collections/npl/rlv-ass")
    if not queries_file.exists():
        raise typer.Exit(f"{queries_file} does not exist.")
    if not relevants_file.exists():
        raise typer.Exit(f"{relevants_file} does not exist.")

    # Parse the queries
    typer.echo("Parsing queries...")
    queries = []
    with open(str(queries_file), "r") as qry_f:
        current_id = None
        text = []
        for line in qry_f:
            stripped = line.strip()
            if stripped == "/":
                continue
            if stripped.isnumeric():
                if current_id is not None:
                    queries.append(" ".join(text).lower())
                    text = []
                current_id = int(stripped)
                continue
            text.append(stripped)
        if text:
            queries.append(" ".join(text).lower())

    # Parse the relevants
    typer.echo("Parsing relevants relevants...")
    with open(str(relevants_file), "r") as rel_f:
        all_text = " ".join(ln.strip() for ln in rel_f.readlines())

    relevants = [
        [int(doc) for doc in entry.split()[1:]] for entry in all_text.split("/")[:-1]
    ]

    assert len(queries) == len(
        relevants
    ), "Query and relevants are not the same length."

    return [QueryTest(queries[i], relevants[i]) for i in range(len(queries))]
