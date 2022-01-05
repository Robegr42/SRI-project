"""
Information Retrieval Model

The IR model is an implementation of the TF-IDF algorithm.
"""

import datetime
import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import typer
from nltk import word_tokenize

from query import Query, QueryResult

DEFAULT_CONFIG = {
    "tokenization_method": "split",
    "include_metadata": [],
}


class IRModel:
    """
    This class is the interface for the IR model.

    Parameters
    ----------
    data_path : str
        The path to the data.
    """

    def __init__(
        self,
        database_folder: str,
        build: bool = False,
        config_file: Optional[str] = None,
    ):
        self.database_folder = Path(database_folder)
        self.model_folder = self.database_folder / "model"
        if not self.model_folder.exists():
            build = True
        self.model_folder.mkdir(exist_ok=True)

        self.config = DEFAULT_CONFIG
        if config_file is not None:
            with open(config_file, "r") as c_file:
                self.config = json.load(c_file)

        self.words: np.ndarray = None
        self.words_idx: dict = None

        # [table, words, table]
        # Tables are:
        #    0 - freq
        #    1 - norm_freq
        #    2 - tf_idf
        self.tf_idf_tables: np.ndarray = None

        self.idf: np.ndarray = None
        self.model_info: dict = None

        self.docs = None
        self.metadata = self._load_metadata_file()
        if build:
            self._build_model()
        else:
            self.words = self._get_model_file("words")
            self.words_idx = {word: i for i, word in enumerate(self.words)}
            self.idf = self._get_model_file("idf")
            self.tf_idf_tables = self._get_model_file("tf_idf_tables")
            self.model_info = self._get_model_file("model_info", ext="json")

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

    def _build_model(self):
        """
        Builds the model.
        """
        start_build_time = time.time()
        # Extract texts
        typer.echo("Extracting texts...")
        docs = self._get_documents()
        for key in self.config["include_metadata"]:
            if key in self.metadata:
                docs += " " + self.metadata[key]
        self.docs = docs

        # Tokenize texts by words
        typer.echo("Tokenizing texts...")
        tokenization_func = self._get_tokenization_func()
        docs_words = [tokenization_func(doc) for doc in self.docs]

        typer.echo("Extracting words frequencies...")
        words_frec: List[Dict[str, int]] = [
            Counter(doc_words) for doc_words in docs_words
        ]

        # Extract word set of each document
        typer.echo("Extracting words set per document...")
        words_by_doc = np.array([set(doc_words) for doc_words in docs_words])

        # Extract global word set
        typer.echo("Extracting global word set...")
        self.words = np.array(
            list(set(word for words in words_by_doc for word in words))
        )
        self.words_idx = {word: i for i, word in enumerate(self.words)}

        # Build frequency matrix (and normalized)
        typer.echo("Building frequency matrix (and normalized matrix)...")
        freq = np.zeros((len(self.docs), len(self.words)))
        norm_freq = np.zeros((len(self.docs), len(self.words)))
        start_time = time.time()
        for i in range(len(self.docs)):
            percent = i / len(self.docs) * 100
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                percent = max(percent, 0.0001)
                time_left = (100 - percent) / percent * elapsed_time
                formated_time = time.strftime("%H:%M:%S", time.gmtime(time_left))
                print(
                    f"\r{percent:.2f}% - {formated_time} left",
                    end="",
                )

            words_frec_i = words_frec[i]
            for word in words_by_doc[i]:
                freq[i, self.words_idx[word]] = words_frec_i[word]
            norm_freq[i, :] = freq[i, :] / np.max(freq[i])
        end_time = time.time()
        total_formated = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
        print(f"\r100% - Total time: {total_formated}")

        # Build inverse document frequency array
        typer.echo("Building inverse document frequency array...")
        self.idf = np.log(len(self.docs) / (freq > 0).sum(axis=0))

        # Build TF-IDF matrix
        typer.echo("Building TF-IDF matrix...")
        tf_idf = norm_freq * self.idf

        # build final tables
        self.tf_idf_tables = np.array([freq, norm_freq, tf_idf])

        typer.echo("Saving models files...")
        np.save(self.model_folder / "words.npy", self.words)
        np.save(self.model_folder / "idf.npy", self.idf)
        np.save(self.model_folder / "tf_idf_tables.npy", self.tf_idf_tables)

        end_build_time = time.time()
        build_time = end_build_time - start_build_time

        # Model info
        typer.echo("Creating model info...")
        self.model_info = {
            "id": time.ctime(time.time()),
            "database_folder": str(self.database_folder),
            "date": datetime.datetime.now().isoformat(),
            "build_time": build_time,
            "build_time_fromated": time.strftime("%H:%M:%S", time.gmtime(build_time)),
            "config": self.config,
        }

        # Save tables
        with open(self.model_folder / "model_info.json", "w") as m_file:
            json.dump(self.model_info, m_file, indent=4)

    def _get_tokenization_func(self) -> Callable:
        """
        Returns the tokenization function.

        Returns
        -------
        Callable
            The tokenization function.
        """
        method = self.config["tokenization_method"]
        if method == "nltk":
            return word_tokenize
        if method == "split":
            return lambda text: text.split()
        raise typer.Exit(f"Unknown tokenization method: {method}")

    def _get_model_file(self, file_name: str, ext: str = "npy") -> np.ndarray:
        """
        Extracts a file from the model folder.

        Parameters
        ----------
        file_name : str
            The name of the file to extract.
        ext : str
            The extension of the file to extract.

        Returns
        -------
        np.ndarray
            The extracted file.
        """
        file_path = self.model_folder / f"{file_name}.{ext}"
        if not file_path.exists():
            raise typer.Exit(
                f"'{file_path}' not found\n\n"
                "Try to rebuild the model using the 'build' command"
            )
        if ext == "json":
            with open(str(file_path), "r") as m_file:
                return json.load(m_file)
        if ext == "npy":
            return np.load(str(file_path))
        raise typer.Exit(f"Invalid file extension: {ext}")

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

    @property
    def freq(self) -> np.ndarray:
        """
        Returns the frequency matrix.

        Returns
        -------
        np.ndarray
            The frequency matrix.
        """
        return self.tf_idf_tables[0]

    @property
    def norm_freq(self) -> np.ndarray:
        """
        Returns the normalized frequency matrix.

        Returns
        -------
        np.ndarray
            The normalized frequency matrix.
        """
        return self.tf_idf_tables[1]

    @property
    def tf_idf(self) -> np.ndarray:
        """
        Returns the TF-IDF matrix.

        Returns
        -------
        np.ndarray
            The TF-IDF matrix.
        """
        return self.tf_idf_tables[2]

    def search(self, raw_query: str, smooth_a: Optional[float] = None) -> QueryResult:
        """
        Search for relevant documents based on the query.

        Parameters
        ----------
        raw_query : str
            The query to search for.

        Returns
        -------
        list
            A list of relevant documents.
        """
        query = Query(raw_query)
        results = []

        # Get valid words from query
        q_words = [word for word in query.words if word in self.words_idx]

        # Calculate TF-IDF scores for the query
        q_vector = np.zeros(len(self.words))
        q_words_counter = Counter(q_words)
        for word in q_words:
            q_vector[self.words_idx[word]] = q_words_counter[word]
        q_vector = q_vector / np.max(q_vector)

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
