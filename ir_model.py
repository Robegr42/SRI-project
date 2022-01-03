"""
Information Retrieval Model

The IR model is an implementation of the TF-IDF algorithm.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

from query import Query


class IRModel:
    """
    This class is the interface for the IR model.

    Parameters
    ----------
    data_path : str
        The path to the data.
    """

    def __init__(self, database_folder: str, reindex: bool = False):
        self.database_folder = Path(database_folder)
        self.index_folder = self.database_folder / "index"
        self.index_folder.mkdir(exist_ok=True)

        self.words: np.ndarray = None
        self.words_idx: dict = None
        self.freq: np.ndarray = None
        self.norm_freq: np.ndarray = None
        self.idf: np.ndarray = None

        self.docs = None
        if reindex:
            self._build_index()
        else:
            self.words = self._get_index_file("words")
            self.words_idx = {word: i for i, word in enumerate(self.words)}
            self.freq = self._get_index_file("freq")
            self.norm_freq = self._get_index_file("norm_freq")
            self.idf = self._get_index_file("idf")

    def _build_index(self):
        """
        Builds the index.
        """
        # Extract texts
        self.docs = self._get_documents()

        # Tokenize texts by words
        print("Tokenizing texts...")
        docs_words = [doc.split() for doc in self.docs]
        print("Extracting words frequencies...")
        words_frec: List[Dict[str, int]] = [
            Counter(doc_words) for doc_words in docs_words
        ]
        # Extract word set of each document
        print("Extracting words set per document...")
        words_by_doc = np.array([set(doc_words) for doc_words in docs_words])
        # Extract global word set
        print("Extracting global word set...")
        self.words = np.array(
            list(set(word for words in words_by_doc for word in words))
        )
        self.words_idx = {word: i for i, word in enumerate(self.words)}

        # Build frequency matrix (and normalized)
        print("Building frequency matrix (and normilized matrix)...")
        freq = np.zeros((len(self.docs), len(self.words)))
        norm_freq = np.zeros((len(self.docs), len(self.words)))
        percents_logs = np.arange(0, 1, 0.02)
        percent_idx = 0
        for i in range(len(self.docs)):
            if (
                percent_idx < len(percents_logs)
                and i / len(self.docs) > percents_logs[percent_idx]
            ):
                print(f"{percents_logs[percent_idx] * 100:.2f}%")
                percent_idx += 1

            for word in words_by_doc[i]:
                freq[i, self.words_idx[word]] = words_frec[i][word]
            for j in range(len(self.words)):
                norm_freq[i, j] = freq[i, j] / np.max(freq[i])
        self.freq = freq
        self.norm_freq = norm_freq

        # Build inverse document frequency array
        print("Building inverse document frequency array...")
        self.idf = np.log(len(self.docs) / (self.freq > 0).sum(axis=0))

        # Save tables
        print("Saving index...")
        np.save(self.index_folder / "words.npy", self.words)
        np.save(self.index_folder / "freq.npy", self.freq)
        np.save(self.index_folder / "norm_freq.npy", self.norm_freq)
        np.save(self.index_folder / "idf.npy", self.idf)

    def _get_index_file(self, file_name: str) -> np.ndarray:
        """
        Extracts an index file from the index folder.

        Parameters
        ----------
        file_name : str
            The name of the file to extract.

        Returns
        -------
        np.ndarray
            The extracted file.
        """
        file_path = self.index_folder / f"{file_name}.npy"
        if file_path.exists():
            return np.load(str(file_path))
        raise FileNotFoundError(f"'{file_path}' not found")

    def _get_documents(self) -> np.ndarray:
        """
        Extracts all texts from the database.
        """
        docs_file = self.database_folder / "docs.json"
        if docs_file.exists():
            with open(str(docs_file), "r") as d_file:
                return np.array(json.load(d_file))
        raise FileNotFoundError(f"'{docs_file}' not found")

    def search(self, query: Query) -> list:
        """
        Search for relevant documents based on the query.

        Parameters
        ----------
        query : Query
            The query to search for.

        Returns
        -------
        list
            A list of relevant documents.
        """
        raise NotImplementedError
