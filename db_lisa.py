from pathlib import Path
from typing import Any, Dict, List

import typer

from database_builder import DatabaseBuilder
from model_tester import QueryTest


def build_lisa_db():
    """
    Creates the lisa database.
    """
    metadata: List[Dict[str, Any]] = []
    texts: List[str] = []
    lisa_file_names = [
        "./test_collections/lisa/LISA0.001",
        "./test_collections/lisa/LISA0.501",
        "./test_collections/lisa/LISA1.001",
        "./test_collections/lisa/LISA1.501",
        "./test_collections/lisa/LISA2.001",
        "./test_collections/lisa/LISA2.501",
        "./test_collections/lisa/LISA3.001",
        "./test_collections/lisa/LISA3.501",
        "./test_collections/lisa/LISA4.001",
        "./test_collections/lisa/LISA4.501",
        "./test_collections/lisa/LISA5.001",
        "./test_collections/lisa/LISA5.501",
        "./test_collections/lisa/LISA5.627",
        "./test_collections/lisa/LISA5.850",
    ]

    for file_path in lisa_file_names:
        lisa_file = Path(file_path)
        if not lisa_file.exists():
            raise FileNotFoundError(f"{lisa_file} does not exist.")

        with open(lisa_file, "r") as lisa_f:
            current_id = None
            in_title = False
            title, text = [], []
            for line in lisa_f:
                stripped = line.strip()
                if stripped.startswith("Document"):
                    current_id = int(stripped.split()[-1])
                    in_title = True
                    continue
                if stripped.startswith("*"):
                    if current_id is not None:
                        metadata.append(
                            {
                                "id": current_id,
                                "title": " ".join(title),
                            }
                        )
                        texts.append(" ".join(text).lower())
                        title, text = [], []
                    continue
                if stripped == "":
                    in_title = False
                    continue
                if in_title:
                    title.append(stripped)
                else:
                    text.append(stripped)
            if text:
                metadata.append(
                    {
                        "id": current_id,
                        "title": " ".join(title),
                    }
                )
                texts.append(" ".join(text).lower())

    DatabaseBuilder.build("lisa", metadata, texts)


def lisa_query_tests() -> List[QueryTest]:
    """
    Runs the query tests.
    """
    queries_file = Path("./test_collections/lisa/LISA.QUE")
    relevants_file = Path("./test_collections/lisa/LISA.REL")
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
            if stripped.isnumeric():
                if current_id is not None:
                    queries.append(" ".join(text).lower()[:-3])
                    text = []
                current_id = int(stripped)
                continue
            text.append(stripped)
        if text:
            queries.append(" ".join(text).lower()[:-3])

    # Parse the relevants
    typer.echo("Parsing relevants relevants...")
    relevants = []
    with open(str(relevants_file), "r") as rel_f:
        rel = []
        for line in rel_f:
            stripped = line.strip()
            if stripped.startswith("Query"):
                relevants.append(rel)
                rel = []
                continue
            if stripped == "" or "Refs" in stripped:
                continue
            numbers = [int(n) for n in stripped.split() if n != "-1"]
            rel.extend(numbers)
        if rel:
            relevants.append(rel)

    assert len(queries) == len(
        relevants
    ), "Query and relevants are not the same length."

    return [QueryTest(queries[i], relevants[i]) for i in range(len(queries))]
