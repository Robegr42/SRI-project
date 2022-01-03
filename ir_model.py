from query import Query


class IRModel:
    """
    This class is the interface for the IR model.

    Parameters
    ----------
    data_path : str
        The path to the data.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path

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
