from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import typer

from database_builder import DatabaseBuilder
from model_tester import QueryTest
from display_tools import pbar


class ReadState(Enum):
    """
    Enum for the different states of the file reader.
    """

    NEWFILE = 0
    TITLE = 1
    AUTHORS = 2
    DATE = 3
    PUB = 4
    TEXT = 5
    XPES = 6


def build_cacm_db():
    """
    Creates the cacm database.
    """
    metadata: List[Dict[str, Any]] = []
    texts: List[str] = []
    cacm_file = Path("./test_collections/cacm/cacm.all")
    if not cacm_file.exists():
        raise FileNotFoundError(f"{cacm_file} does not exist.")

    header_state = {
        ".I": ReadState.NEWFILE,
        ".T": ReadState.TITLE,
        ".A": ReadState.AUTHORS,
        ".N": ReadState.DATE,
        ".B": ReadState.PUB,
        ".W": ReadState.TEXT,
        ".X": ReadState.XPES,
    }

    with open(cacm_file, "r") as cacm_f:
        state = None
        title, authors, date, pub, text = [], [], [], [], []
        doc_id = None
        for line in cacm_f:
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
                            "date": " ".join(date),
                            "pub": " ".join(pub),
                        }
                    )
                    texts.append(" ".join(text))
                title, authors, pub, date, text = [], [], [], [], []
                doc_id = line[3:-1]

            if state is None or in_header or state == ReadState.XPES:
                continue

            if state == ReadState.TITLE:
                title.append(line.strip())
            elif state == ReadState.AUTHORS:
                authors.append(line.strip())
            elif state == ReadState.DATE:
                date.append(line.strip())
            elif state == ReadState.PUB:
                pub.append(line.strip())
            elif state == ReadState.TEXT:
                text.append(line.strip())

    DatabaseBuilder.build("cacm", metadata, texts)


def cacm_query_tests() -> List[QueryTest]:
    """
    Runs the query tests.
    """
    queries_file = Path("./test_collections/cacm/query.text")
    relevants_file = Path("./test_collections/cacm/qrels.text")
    if not queries_file.exists():
        raise typer.Exit(f"{queries_file} does not exist.")
    if not relevants_file.exists():
        raise typer.Exit(f"{relevants_file} does not exist.")

    # Parse the queries
    typer.echo("Parsing queries...")
    queries = []
    til_next_i = False
    with open(str(queries_file), "r") as qry_f:
        query_text = []
        for line in qry_f:
            if not line:
                continue
            if line.startswith(".I"):
                til_next_i = False
                if query_text:
                    queries.append(" ".join(query_text))
                    query_text = []
                continue
            if til_next_i:
                continue
            if line.startswith(".N"):
                til_next_i = True
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
            query, doc_id, _, _ = [int(i) for i in line.split()]
            relevants[query - 1].append(doc_id)

    return [QueryTest(queries[i], relevants[i]) for i in range(len(queries))]
