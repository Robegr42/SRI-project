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
        self.rel_indices = rel_indices
        self.ret_indices = ret_indices
        self.ret_scores = ret_scores
        self.total_docs = total_docs

        self.pres_val = None
        self.recall_val = None
        self.fallout_val = None
        self.f1_val = None

    def calculate_metrics(self, top: Optional[int] = None) -> dict:
        """
        Calculate the metrics of the result.
        """
        self.pres_val = self.presision(top)
        self.recall_val = self.recall(top)
        self.fallout_val = self.fallout(top)
        self.f1_val = self.f1_score(top)

    @property
    def empty(self) -> bool:
        """
        Returns True if the result is empty.
        """
        return len(self.ret_indices) == 0

    def presision(self, top: Optional[int] = None) -> Union[float, None]:
        """
        Calculate the presision of the model.
        """
        if top is None:
            top = len(self.rel_indices)
        all_ret = self.ret_indices[:top]
        rel_ret = [i for i in self.rel_indices if i in all_ret]
        total_ret = len(all_ret)
        return len(rel_ret) / total_ret if total_ret > 0 else None

    def recall(self, top: Optional[int] = None) -> Union[float, None]:
        """
        Calculate the recall of the model.
        """
        if top is None:
            top = len(self.rel_indices)
        all_ret = self.ret_indices[:top]
        rel_ret = [i for i in self.rel_indices if i in all_ret]
        total_rel = len(self.rel_indices)
        return len(rel_ret) / total_rel if total_rel > 0 else None

    def f1_score(self, top: Optional[int] = None) -> Union[float, None]:
        """
        Calculate the F1 score of the result.
        """
        pres = self.presision(top)
        recall = self.recall(top)
        if pres is None or recall is None or pres + recall == 0:
            return None
        return 2 * pres * recall / (pres + recall)

    def fallout(self, top: Optional[int] = None) -> Union[float, None]:
        """
        Calculate the fallout of the model.
        """
        if top is None:
            top = len(self.rel_indices)
        all_ret = self.ret_indices[:top]
        not_rel_ret = [i for i in self.rel_indices if i not in all_ret]
        total_not_rel = self.total_docs - len(self.rel_indices)
        return len(not_rel_ret) / total_not_rel if total_not_rel > 0 else None


class ModelTester:
    def __init__(self, model: IRModel, force: bool = False):
        self.model = model
        self.force = force
        self.results: List[QueryTestResult] = []

    def test_top(self, top: Optional[int] = None):
        """
        Show the results.
        """
        results = self.results

        for result in results:
            result.calculate_metrics(top)

        all_pres = np.array([res.pres_val for res in results])
        all_recall = np.array([res.recall_val for res in results])
        all_f1 = np.array([res.f1_val for res in results])
        all_fallout = np.array([res.fallout_val for res in results])

        # Filter out None values
        not_none_inx = all_pres != None
        not_none_inx = np.logical_and(not_none_inx, all_recall != None)
        not_none_inx = np.logical_and(not_none_inx, all_f1 != None)
        not_none_inx = np.logical_and(not_none_inx, all_fallout != None)
        ignored_idx = len(results) - np.sum(not_none_inx)
        all_pres = all_pres[not_none_inx]
        all_recall = all_recall[not_none_inx]
        all_f1 = all_f1[not_none_inx]
        all_fallout = all_fallout[not_none_inx]

        pres_mean, pres_std = np.mean(all_pres), np.std(all_pres)
        pres_min, pres_max = np.min(all_pres), np.max(all_pres)

        recall_mean, recall_std = np.mean(all_recall), np.std(all_recall)
        recall_min, recall_max = np.min(all_recall), np.max(all_recall)

        f1_mean, f1_std = np.mean(all_f1), np.std(all_f1)
        f1_min, f1_max = np.min(all_f1), np.max(all_f1)

        fallout_mean, fallout_std = np.mean(all_fallout), np.std(all_fallout)
        fallout_min, fallout_max = np.min(all_fallout), np.max(all_fallout)

        scores = np.array([res.ret_scores[:top] for res in results])
        scores_mean, scores_std = np.mean(scores), np.std(scores)
        scores_min, scores_max = np.min(scores), np.max(scores)

        ret_val = [
            (pres_mean, pres_std, pres_min, pres_max),
            (recall_mean, recall_std, recall_min, recall_max),
            (f1_mean, f1_std, f1_min, f1_max),
            (fallout_mean, fallout_std, fallout_min, fallout_max),
            (scores_mean, scores_std, scores_min, scores_max),
        ]

        return ret_val

    def test(self, query_tests: List[QueryTest], tops: Optional[List[int]] = None):
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
        typer.echo("Testing model...")
        model_id = self.model.model_info["id"]
        test_result_file = self.model.model_folder / f"tests_{model_id}.npy"
        if not test_result_file.exists() or self.force:
            results = []
            total = len(query_tests)
            for i, q_test in enumerate(query_tests):
                qt_result = self.test_query(q_test)
                progress = (i + 1) / total * 100
                print(f"\r{i}/{total} - {progress:.2f}%", end="")
                results.append(qt_result)
            print("\n")
            self.results = results
            ret_vals = np.array([self.test_top(top) for top in tops])
        else:
            typer.echo("Loading test results...")
            ret_vals = np.load(test_result_file)

        # Save results
        np.save(str(test_result_file), ret_vals)

        all_pres_mean = ret_vals[:, 0, 0]
        all_pres_std = ret_vals[:, 0, 1]
        all_pres_min = ret_vals[:, 0, 2]
        all_pres_max = ret_vals[:, 0, 3]

        all_recall_mean = ret_vals[:, 1, 0]
        all_recall_std = ret_vals[:, 1, 1]
        all_recall_min = ret_vals[:, 1, 2]
        all_recall_max = ret_vals[:, 1, 3]

        all_f1_mean = ret_vals[:, 2, 0]
        all_f1_std = ret_vals[:, 2, 1]
        all_f1_min = ret_vals[:, 2, 2]
        all_f1_max = ret_vals[:, 2, 3]

        all_fallout_mean = ret_vals[:, 3, 0]
        all_fallout_std = ret_vals[:, 3, 1]
        all_fallout_min = ret_vals[:, 3, 2]
        all_fallout_max = ret_vals[:, 3, 3]

        all_scores_mean = ret_vals[:, 4, 0]
        all_scores_std = ret_vals[:, 4, 1]
        all_scores_min = ret_vals[:, 4, 2]
        all_scores_max = ret_vals[:, 4, 3]

        for i, top in enumerate(tops):
            typer.echo(f"\nTop {top}:")
            typer.echo(
                f"Presision:  {all_pres_mean[i]:.3f} ± {all_pres_std[i]:.3f} "
                f"[{all_pres_min[i]:.3f}, {all_pres_max[i]:.3f}]"
            )
            typer.echo(
                f"Recall:     {all_recall_mean[i]:.3f} ± {all_recall_std[i]:.3f} "
                f"[{all_recall_min[i]:.3f}, {all_recall_max[i]:.3f}]"
            )
            typer.echo(
                f"F1 score:   {all_f1_mean[i]:.3f} ± {all_f1_std[i]:.3f} "
                f"[{all_f1_min[i]:.3f}, {all_f1_max[i]:.3f}]"
            )
            typer.echo(
                f"Fallout:    {all_fallout_mean[i]:.3f} ± {all_fallout_std[i]:.3f} "
                f"[{all_fallout_min[i]:.3f}, {all_fallout_max[i]:.3f}]"
            )
            typer.echo(
                f"Score:      {all_scores_mean[i]:.3f} ± {all_scores_std[i]:.3f} "
                f"[{all_scores_min[i]:.3f}, {all_scores_max[i]:.3f}]"
            )

        fig, axs = plt.subplots(2, 3, figsize=(12, 7))
        fig.suptitle("Model performance", fontsize=16)

        # Presision
        plt.sca(axs[0, 0])
        self._plot_metrics(ret_vals, idx=0, title="Precision", color="blue")

        # Recall
        plt.sca(axs[0, 1])
        self._plot_metrics(ret_vals, idx=1, title="Recall", color="orange")

        # F1 score
        plt.sca(axs[1, 0])
        self._plot_metrics(ret_vals, idx=2, title="F1 score", color="green")

        # Fallout
        plt.sca(axs[1, 1])
        self._plot_metrics(ret_vals, idx=3, title="Fallout", color="red")

        # Precision vs Recall
        plt.sca(axs[0, 2])
        plt.plot(all_recall_mean, all_pres_mean, "o-")
        plt.grid()
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision vs Recall")

        # Average score vs order
        plt.sca(axs[1, 2])
        self._plot_metrics(ret_vals, idx=4, title="Score", color="purple")

        plt.tight_layout()
        plt.show()

    def _plot_metrics(self, ret_val, idx, title, color, show: bool = False):
        mean_vals = ret_val[:, idx, 0]
        std_vals = ret_val[:, idx, 1]
        min_vals = ret_val[:, idx, 2]
        max_vals = ret_val[:, idx, 3]

        plt.plot(
            range(1, len(mean_vals) + 1),
            max_vals,
            "--",
            label=f"{title} max",
            alpha=0.5,
            color=color,
        )
        plt.fill_between(
            range(1, len(mean_vals) + 1),
            mean_vals - std_vals,
            mean_vals + std_vals,
            alpha=0.4,
            color=color,
            label=f"{title} std",
        )
        plt.plot(
            range(1, len(mean_vals) + 1), mean_vals, label=f"Avg. {title}", color=color
        )
        plt.plot(
            range(1, len(mean_vals) + 1),
            min_vals,
            "--",
            label=f"{title} min",
            alpha=0.5,
            color=color,
        )
        plt.grid()
        plt.xlabel("Top")
        plt.ylabel(title)
        plt.title(title)
        plt.legend(loc="best")
        if show:
            plt.show()

    def test_query(self, query_test: QueryTest) -> QueryTestResult:
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
