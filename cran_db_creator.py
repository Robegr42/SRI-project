from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

from database_creator import DatabaseCreator


class ReadState(Enum):
    """
    Enum for the different states of the file reader.
    """

    NEWFILE = 0
    TITLE = 1
    AUTHORS = 2
    PUB = 3
    TEXT = 4


def create_db():
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

    DatabaseCreator.create("cran", metadata, texts)
