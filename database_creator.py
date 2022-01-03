"""
Database creator

The class DatabaseCreator is used to structure a documents database for
further processing byt the aplication.

This class will create a folder structure like:

    ./database/
    |__ [database_name]/
        |__ texts.txt
        |__ metadata.json

For example:

        ./database/
        |__ cran/
            |__ texts.txt
            |__ metadata.json

The ``texts.txt`` file will contain the text of the documents separated by
``\\n..{i}\\n`` where ``{i}`` is the document index.

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
        metadata: Any,
        texts: List[str],
    ):
        """
        Create the database.

        Parameters
        ----------
        database_name : str
            Name of the database.
        metadata : Any
            Dictionary of documents metadata.
        texts : List[str]
            List of documents texts.
        """
        database = DatabaseCreator(database_name)
        database.create_db(metadata, texts)

    def create_db(self, metadata: Any, texts: List[str]):
        """
        Create the database.

        Parameters
        ----------
        metadata : Any
            Dictionary of documents metadata.
        texts : List[str]
            List of documents texts.
        """
        self.database_path.mkdir(parents=True, exist_ok=True)
        self._create_metadata(metadata)
        self._create_texts(texts)

    def _create_metadata(self, metadata: dict):
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

    def _create_texts(self, texts: List[str]):
        """
        Create the documents texts.

        Parameters
        ----------
        texts : List[str]
            List of documents texts.
        """
        texts_path = self.database_path / "texts.txt"
        with open(str(texts_path), "w", encoding="utf-8") as file:
            for i, text in enumerate(texts):
                file.write(f"\n..{i}\n")
                file.write(text)
