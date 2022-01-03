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
