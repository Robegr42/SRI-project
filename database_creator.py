"""
Database creator

The class DatabaseCreator is used to structure a documents database for
further processing byt the aplication.

This class will create a folder structure like:

    ./database
    |__ /[database_name]
        |__ docs.json
        |__ metadata.json

For example:

        ./database
        |__ /cran
            |__ docs.json
            |__ metadata.json

The ``docs.json`` file will contain a the text of the documents.

The ``metadata.json`` file will contain the metadata of each document in
a list of dictionaries where each position in the list corresponds to the
document index.
"""

import json
from pathlib import Path
from typing import Any, List

_DEFAULT_DB_PATH = Path("./database")


class DatabaseCreator:
    """
    Abstract class for database creation.

    Parameters
    ----------
    database_name : str
        Name of the database.

    Attributes
    ----------
    database_name : str
        Name of the database.
    database_path : Path
        Path to the database.
    """

    def __init__(self, database_name):
        self.database_name = database_name
        self.database_path = _DEFAULT_DB_PATH / database_name

    @staticmethod
    def create(
        database_name: str,
        metadata: List[dict],
        docs: List[str],
    ):
        """
        Create the database.

        Parameters
        ----------
        database_name : str
            Name of the database.
        metadata : List[dict]
            Dictionary of documents metadata.
        docs : List[str]
            List of documents.
        """
        database = DatabaseCreator(database_name)
        database.create_db(metadata, docs)

    def create_db(self, metadata: List[dict], docs: List[str]):
        """
        Create the database.

        Parameters
        ----------
        metadata : List[dict]
            Dictionary of documents metadata.
        docs : List[str]
            List of documents.
        """
        self.database_path.mkdir(parents=True, exist_ok=True)
        self._create_metadata(metadata)
        self._create_docs(docs)

    def _create_metadata(self, metadata: List[dict]):
        """
        Create the documents metadata.

        Parameters
        ----------
        metadata : dict
            Dictionary of documents metadata.
        """
        metadata_path = self.database_path / "metadata.json"
        with open(str(metadata_path), "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=4)

    def _create_docs(self, docs: List[str]):
        """
        Create the documents.

        Parameters
        ----------
        docs : List[str]
            List of documents.
        """
        docs_path = self.database_path / "docs.json"
        with open(str(docs_path), "w", encoding="utf-8") as file:
            json.dump(docs, file, ensure_ascii=False, indent=4)
