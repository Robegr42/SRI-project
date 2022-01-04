"""
Information Retrieval Model

The IR model is an implementation of the TF-IDF algorithm.
"""

import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import typer

from query import Query, QueryResult


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
        if not self.index_folder.exists():
            reindex = True
        self.index_folder.mkdir(exist_ok=True)

        self.words: np.ndarray = None
        self.words_idx: dict = None
        self.freq: np.ndarray = None
        self.norm_freq: np.ndarray = None
        self.idf: np.ndarray = None
        self.tf_idf: np.ndarray = None

        self.docs = None
        self.metadata = self._load_metadata_file()
        if reindex:
            self._build_index()
        else:
            self.words = self._get_index_file("words")
            self.words_idx = {word: i for i, word in enumerate(self.words)}
            self.freq = self._get_index_file("freq")
            self.norm_freq = self._get_index_file("norm_freq")
            self.idf = self._get_index_file("idf")
            self.tf_idf = self._get_index_file("tf_idf")

    def _load_metadata_file(self) -> Dict:
        """
        Loads the metadata file.

        Returns
        -------
        Dict
            The loaded metadata.
        """
        file_path = self.database_folder / "metadata.json"
        if file_path.exists():
            with open(str(file_path), "r") as m_file:
                return json.load(m_file)
        raise typer.Exit(f"'{file_path}' not found")

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
        start_time = time.time()
        for i in range(len(self.docs)):
            percent = i / len(self.docs) * 100
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                percent = max(percent, 0.0001)
                time_left = (100 - percent) / percent * (time.time() - start_time)
                formatted_time = time.strftime("%H:%M:%S", time.gmtime(time_left))
                print(
                    f"\r{percent:.2f}% - {formatted_time} left",
                    end="",
                )

            for word in words_by_doc[i]:
                freq[i, self.words_idx[word]] = words_frec[i][word]
            for j in range(len(self.words)):
                norm_freq[i, j] = freq[i, j] / np.max(freq[i])
        self.freq = freq
        self.norm_freq = norm_freq
        end_time = time.time()
        total_formated = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
        print(f"\r100% - Total time: {total_formated}")

        # Build inverse document frequency array
        print("Building inverse document frequency array...")
        self.idf = np.log(len(self.docs) / (self.freq > 0).sum(axis=0))

        # Build TF-IDF matrix
        print("Building TF-IDF matrix...")
        self.tf_idf = self.norm_freq * self.idf

        # Save tables
        print("Saving index...")
        np.save(self.index_folder / "words.npy", self.words)
        np.save(self.index_folder / "freq.npy", self.freq)
        np.save(self.index_folder / "norm_freq.npy", self.norm_freq)
        np.save(self.index_folder / "idf.npy", self.idf)
        np.save(self.index_folder / "tf_idf.npy", self.tf_idf)

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
        raise typer.Exit(f"'{file_path}' not found\n\nTry running with --reindex")

    def _get_documents(self) -> np.ndarray:
        """
        Extracts all texts from the database.
        """
        docs_file = self.database_folder / "docs.json"
        if docs_file.exists():
            with open(str(docs_file), "r") as d_file:
                return np.array(json.load(d_file))
        raise typer.Exit(f"'{docs_file}' not found")

    def _similarty(self, vector_1: np.ndarray, vector_2: np.ndarray) -> float:
        """
        Calculates the similarity between two vectors.

        Parameters
        ----------
        vector_1 : np.ndarray
            The first vector.
        vector_2 : np.ndarray
            The second vector.

        Returns
        -------
        float
            The similarity between the two vectors.
        """
        return np.dot(vector_1, vector_2) / (
            np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
        )

    def search(self, query: Query, smooth_a: Optional[float] = None) -> QueryResult:
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
        results = []

        # Get valid words from query
        q_words = [word for word in query.words if word in self.words_idx]

        # Calculate TF-IDF scores for the query
        q_vector = np.zeros(len(self.words))
        q_words_counter = Counter(q_words)
        for word in q_words:
            q_vector[self.words_idx[word]] = q_words_counter[word]
        q_vector = q_vector / np.max(q_vector)

        print(f"TF-IDF for query: {q_vector}")

        # Calculate TF-IDF scores for each document
        similarty = np.array(
            [self._similarty(q_vector, doc_vector) for doc_vector in self.tf_idf]
        )
        result_list = [(value, i) for i, value in enumerate(similarty)]
        result_list.sort(reverse=True)
        pos = 0
        for value, i in result_list:
            pos += 1
            results.append(
                {
                    "pos": pos,
                    "weight": value,
                    "doc_index": i,
                    "doc_metadata": self.metadata[i],
                }
            )
        return QueryResult(query, results)
