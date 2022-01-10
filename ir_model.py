"""
Information Retrieval Model

The IR model is an implementation of the TF-IDF algorithm.
"""

import datetime
import json
import string
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import typer
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer, PorterStemmer

from display_tools import pbar
from query import QueryResult

DEFAULT_CONFIG = {
    "tokenization_method": "split",
    "include_metadata": [],
    "query_alpha_smoothing": 0,
    "remove_stopwords": False,
    "remove_punctuation": False,
    "lemmatization": False,
    "stem": False,
}


_stemmer = PorterStemmer()

def _nltk_pos_tagger(nltk_tag):
    if nltk_tag.startswith("J"):
        return wn.ADJ
    if nltk_tag.startswith("V"):
        return wn.VERB
    if nltk_tag.startswith("N"):
        return wn.NOUN
    if nltk_tag.startswith("R"):
        return wn.ADV
    return None


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
        model_name: str = None,
    ):
        self.model_name = model_name
        self.database_folder = Path(database_folder)
        self.model_folder = self.database_folder / "model"
        if not self.model_folder.exists():
            build = True
        self.model_folder.mkdir(exist_ok=True)

        self.config = DEFAULT_CONFIG
        if config_file is not None:
            with open(config_file, "r") as c_file:
                self.config.update(json.load(c_file))

        self.words: np.ndarray = None
        self.words_idx: dict = None
        self.tf_idf: np.ndarray = None

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
            self.tf_idf = self._get_model_file("tf_idf")
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
        model_id = str(int(time.time() * 1000))
        start_build_time = time.time()
        # Extract texts
        typer.echo("Extracting texts...")
        docs = self._get_documents()
        for key in self.config["include_metadata"]:
            if key in self.metadata:
                docs += " " + self.metadata[key] * 5
        self.docs = docs

        # Tokenize texts by words
        typer.echo("Building tokenization function...")
        tokenization_func = self._get_tokenization_func(self.config)

        typer.echo("Tokenizing texts...")
        docs_words = [tokenization_func(doc) for doc in pbar(self.docs)]

        typer.echo("Extracting words frequencies...")
        words_frec: List[Dict[str, int]] = [
            Counter(doc_words) for doc_words in pbar(docs_words)
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
        norm_freq = np.zeros((len(self.docs), len(self.words)))
        for i in pbar(range(len(self.docs))):
            words_frec_i = words_frec[i]
            for word in words_by_doc[i]:
                norm_freq[i, self.words_idx[word]] = words_frec_i[word]
            norm_freq[i, :] = norm_freq[i, :] / (np.max(norm_freq[i]) + 1)
        end_time = time.time()

        # Build inverse document frequency array
        typer.echo("Building inverse document frequency array...")
        self.idf = np.log(len(self.docs) / ((norm_freq > 0).sum(axis=0) + 1))

        smooth_a = self.config["query_alpha_smoothing"]
        self.idf = (1 - smooth_a) * self.idf + smooth_a

        # Build TF-IDF matrix
        typer.echo("Building TF-IDF matrix...")
        tf_idf = norm_freq * self.idf

        # build final tables
        self.tf_idf = tf_idf

        typer.echo("Saving models files...")
        np.save(self.model_folder / "words.npy", self.words)
        np.save(self.model_folder / "idf.npy", self.idf)
        np.save(self.model_folder / "tf_idf.npy", self.tf_idf)

        end_build_time = time.time()
        build_time = end_build_time - start_build_time

        # Model info
        typer.echo(f"Creating model info {model_id}...")
        self.model_info = {
            "id": model_id,
            "database_folder": str(self.database_folder),
            "date": datetime.datetime.now().isoformat(),
            "build_time": build_time,
            "build_time_fromated": time.strftime("%H:%M:%S", time.gmtime(build_time)),
            "config": self.config,
        }

        # Save tables
        with open(self.model_folder / "model_info.json", "w") as m_file:
            json.dump(self.model_info, m_file, indent=4)

    def _get_tokenization_func(self, config: dict) -> Callable:
        """
        Returns the tokenization function.

        Returns
        -------
        Callable
            The tokenization function.
        """
        tok_func = None
        method = config["tokenization_method"]
        if method == "nltk":
            tok_func = word_tokenize
        elif method == "split":
            tok_func = lambda text: text.split()
        else:
            raise typer.Exit(f"Unknown tokenization method: {method}")

        def tokenization_func(text: str) -> List[str]:
            text = text.replace("-", " ")
            remove_stopwords = config["remove_stopwords"]
            remove_punctuation = config["remove_punctuation"]
            tokens = tok_func(text)
            if config["lemmatization"]:
                tokens = self._lemmatize_tokens(tokens)
            if config["stem"]:
                steammed = []
                for tok in tokens:
                    steammed.append(_stemmer.stem(tok))
                tokens.extend(steammed)

            to_remove = []
            if remove_stopwords:
                to_remove += stopwords.words("english")
            if remove_punctuation:
                to_remove += string.punctuation
            tokens = [token for token in tokens if token not in to_remove]
            return tokens

        return tokenization_func

    def _lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatizes a text.

        Parameters
        ----------
        text : str
            The text to lemmatize.

        Returns
        -------
        str
            The lemmatized text.
        """
        lemmatizer = WordNetLemmatizer()
        tagged_tokens = pos_tag(tokens)
        tokens = [(w, _nltk_pos_tagger(t)) for w, t in tagged_tokens]
        lemm = [w if t is None else lemmatizer.lemmatize(w, t) for w, t in tokens]
        return lemm

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
        norm1 = np.linalg.norm(vector_1)
        norm2 = np.linalg.norm(vector_2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return np.dot(vector_1, vector_2) / (
            np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
        )

    def search(self, raw_query: str) -> QueryResult:
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

        raw_query = raw_query.lower()
        tok_func = self._get_tokenization_func(self.model_info["config"])
        words = set(tok_func(raw_query))
        # print(words)
        smooth_a = self.model_info["config"]["query_alpha_smoothing"]
        results = []

        # Get valid words from query
        q_words = [word for word in words if word in self.words_idx]

        # Calculate TF-IDF scores for the query
        q_vector = np.zeros(len(self.words))
        q_words_counter = Counter(q_words)
        for word in q_words:
            q_vector[self.words_idx[word]] = q_words_counter[word]

        matches = (self.tf_idf > 0) * (q_vector > 0)
        l_term = np.log(len(self.metadata) / (np.sum(matches, axis=0) + 1))
        q_vector = smooth_a + (1 - smooth_a) * (q_vector / (np.max(q_vector) + 1))
        q_vector *= l_term

        # Calculate TF-IDF scores for each document
        docs = self.tf_idf
        similarty = np.array(
            [self._similarty(q_vector, doc_vector) for doc_vector in docs]
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
        return QueryResult(raw_query, results)
