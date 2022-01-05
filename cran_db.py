from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import typer

from database_builder import DatabaseBuilder
from model_tester import QueryTest


class ReadState(Enum):
    """
    Enum for the different states of the file reader.
    """

    NEWFILE = 0
    TITLE = 1
    AUTHORS = 2
    PUB = 3
    TEXT = 4


def build_db():
    """
    Creates the cran database.
    """
    metadata: List[Dict[str, Any]] = []
    texts: List[str] = []
    cran_file = Path("./test_collections/cran/cran.all.1400")
    if not cran_file.exists():
        raise FileNotFoundError(f"{cran_file} does not exist.")

    header_state = {
        ".I": ReadState.NEWFILE,
        ".T": ReadState.TITLE,
        ".A": ReadState.AUTHORS,
        ".B": ReadState.PUB,
        ".W": ReadState.TEXT,
    }

    with open(cran_file, "r") as cran_f:
        state = None
        title, authors, pub, text = [], [], [], []
        doc_id = None
        for line in cran_f:
            in_header = False
            for header, stt in header_state.items():
                if line.startswith(header):
                    state = stt
                    in_header = True
                    break

            if state == ReadState.NEWFILE:
                if text:
                    metadata.append(
                        {
                            "doc_id": doc_id,
                            "title": " ".join(title),
                            "authors": " ".join(authors),
                            "pub": " ".join(pub),
                        }
                    )
                    texts.append(" ".join(text))
                title, authors, pub, text = [], [], [], []
                doc_id = line[3:-1]

            if state is None or in_header:
                continue

            if state == ReadState.TITLE:
                title.append(line.strip())
            elif state == ReadState.AUTHORS:
                authors.append(line.strip())
            elif state == ReadState.PUB:
                pub.append(line.strip())
            elif state == ReadState.TEXT:
                text.append(line.strip())

    DatabaseBuilder.build("cran", metadata, texts)


def query_tests() -> List[QueryTest]:
    """
    Runs the query tests.
    """
    queries_file = Path("./test_collections/cran/cran.qry")
    relevants_file = Path("./test_collections/cran/cranqrel")
    if not queries_file.exists():
        raise typer.Exit(f"{queries_file} does not exist.")
    if not relevants_file.exists():
        raise typer.Exit(f"{relevants_file} does not exist.")

    # Parse the queries
    typer.echo("Parsing queries...")
    queries = []
    with open(str(queries_file), "r") as qry_f:
        query_text = []
        for line in qry_f:
            if line.startswith(".I"):
                if query_text:
                    queries.append(" ".join(query_text))
                    query_text = []
                continue
            if line.startswith(".W"):
                continue
            query_text.append(line.strip())
        if query_text:
            queries.append(" ".join(query_text))

    # Parse the relevants
    typer.echo("Parsing relevants relevants...")
    relevants = [[] for _ in range(len(queries))]

    with open(str(relevants_file), "r") as rel_f:
        for line in rel_f:
            query, doc_id, _ = [int(i) for i in line.split()]
            relevants[query - 1].append(doc_id)

    return [QueryTest(queries[i], relevants[i]) for i in range(len(queries))]
