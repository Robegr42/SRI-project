import abc
import json
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import typer

from ir_model import IRModel


class QueryTest:
    """
    Query test.

    Parameters
    ----------
    query : str
        Query string.
    rel_indices : List[int]
        List of relevant document indices the model should return.
    """

    def __init__(self, query: str, rel_indices: List[int]):
        self.query = query
        self.rel_indices = rel_indices


class QueryTestResult:
    """
    A result of a query test.

    Parameters
    ----------
    query : str
        The query string.
    rel_indices : List[int]
        The indices of the relevant documents.
    ret_indices : List[int]
        The indices of the retrieved documents.
    ret_scores : List[float]
        The scores of the retrieved documents.
    total_docs : int
        The total number of documents.
    """

    def __init__(
        self,
        query: str,
        rel_indices: List[int],
        ret_indices: List[int],
        ret_scores: List[float],
        total_docs: int,
    ):
        self.query = query
        self.rel_indices = set(rel_indices)
        self.ret_indices = set(ret_indices)
        self.ret_scores = ret_scores

        self.rel_ret = self.rel_indices.intersection(self.ret_indices)
        self.rel_not_ret = self.rel_indices.difference(self.ret_indices)
        self.not_rel_ret = self.ret_indices.difference(self.rel_indices)
        self.not_rel_count = total_docs - len(self.rel_indices)

    @property
    def empty(self) -> bool:
        """
        Returns True if the result is empty.
        """
        return len(self.ret_indices) == 0

    @property
    def presision(self) -> Union[float, None]:
        """
        Calculate the presision of the model.
        """
        if len(self.ret_indices) == 0:
            return None
        return len(self.rel_ret) / len(self.ret_indices)

    @property
    def recall(self) -> Union[float, None]:
        """
        Calculate the recall of the model.
        """
        if len(self.rel_indices) == 0:
            return None
        return len(self.rel_ret) / len(self.rel_indices)

    @property
    def f1_score(self) -> Union[float, None]:
        """
        Calculate the F1 score of the result.
        """
        pres = self.presision
        recall = self.recall
        if pres is None or recall is None:
            return None
        return 2 * pres * recall / (pres + recall)

    @property
    def fallout(self) -> Union[float, None]:
        """
        Calculate the fallout of the model.
        """
        if self.not_rel_count == 0:
            return None
        return len(self.not_rel_ret) / self.not_rel_count


class ModelTester:
    def __init__(self, model: IRModel):
        self.model = model
        self.results: List[QueryTestResult] = []

    def show_results(self):
        """
        Show the results.
        """
        total = len(self.results)
        not_empty_results = [res for res in self.results if not res.empty]
        empty_count = total - len(not_empty_results)

        typer.echo("Test statistics:")
        typer.echo(f"Total queries: {len(self.results)}")
        typer.echo(f"Empty results: {empty_count}")

        all_pres = np.array([res.presision for res in not_empty_results])
        all_recall = np.array([res.recall for res in not_empty_results])
        all_f1 = np.array([res.f1_score for res in not_empty_results])
        all_fallout = np.array([res.fallout for res in not_empty_results])

        typer.echo("General statistics:")
        pres_mean, pres_std = np.mean(all_pres), np.std(all_pres)
        typer.echo(f"Precision: {pres_mean:.4f} ± {pres_std:.2f}")
        recall_mean, recall_std = np.mean(all_recall), np.std(all_recall)
        typer.echo(f"Recall:    {recall_mean:.4f} ± {recall_std:.2f}")
        f1_mean, f1_std = np.mean(all_f1), np.std(all_f1)
        typer.echo(f"F1 score:  {f1_mean:.4f} ± {f1_std:.2f}")
        fallout_mean, fallout_std = np.mean(all_fallout), np.std(all_fallout)
        typer.echo(f"Fallout:   {fallout_mean:.4f} ± {fallout_std:.2f}")

        see_charts = typer.confirm("Show charts?")

        if not see_charts:
            return

        # Presision vs Recall
        plt.plot(all_recall, all_pres, ".")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision vs Recall")
        plt.show()

        # Average score vs order
        avg_scores = np.array([res.ret_scores for res in self.results])
        avg_scores_mean = np.mean(avg_scores, axis=0)
        avg_scores_std = np.std(avg_scores, axis=0)
        up_bound = avg_scores_mean + avg_scores_std
        low_bound = avg_scores_mean - avg_scores_std
        plt.fill_between(avg_scores_mean, low_bound, up_bound, alpha=0.2)
        plt.xlabel("Order")
        plt.ylabel("Average score")
        plt.title("Average score vs order")
        plt.show()

    def test(self, query_tests: List[QueryTest], top: Optional[int] = 20):
        """
        Test the model.

        Parameters
        ----------
        query_tests: List[QueryTest]
            List of query tests.
        top: Optional[int]
            Number of results to retrieve.

        Returns
        -------
        List[QueryTestResult]
            List of query test results.
        """
        self.results = [self.test_query(query_test, top) for query_test in query_tests]

    def test_query(
        self, query_test: QueryTest, top: Optional[int] = None
    ) -> QueryTestResult:
        """
        Test the model with a query.

        Parameters
        ----------
        query_test: QueryTest
            Query test.
        top: Optional[int]
            Number of results to retrieve.

        Returns
        -------
        QueryTestResult
            Query test result.
        """
        q_result = self.model.search(query_test.query)
        retireved = q_result.results
        if top is not None:
            retireved = retireved[:top]

        rel_indices = query_test.rel_indices
        ret_indices = [res["doc_index"] for res in retireved]
        ret_scores = [res["weight"] for res in retireved]
        total_docs = len(self.model.metadata)
        return QueryTestResult(
            query_test.query,
            rel_indices,
            ret_indices,
            ret_scores,
            total_docs,
        )
