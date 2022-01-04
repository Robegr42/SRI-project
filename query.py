class Query:
    """
    Represents a query to the database.

    Parameters
    ----------
    raw_query: str
        The raw query.
    """
    def __init__(self, raw_query: str):
        self.raw_query = raw_query
        self.words = self.raw_query.split()

    def __str__(self):
        return self.raw_query

class QueryResult:
    """
    Represents a query result.

    Parameters
    ----------
    query: Query
        The query that was executed.
    result: list
        The result of the query.
    """
    def __init__(self, query: Query, results: list):
        self.query = query
        self.results = results

    def __iter__(self):
        return iter(self.results)

    @staticmethod
    def show_result(result: dict):
        """
        Prints the result of a query.

        Parameters
        ----------
        result: dict
            The result to print.
        """
        pos = result['pos']
        weight = result['weight']
        print(f"\n{pos}. ({weight:.4f})")
        print("    index:", result['doc_index'])
        for key, value in result['doc_metadata'].items():
            print(f"    {key}: {value}")
        print()

