"""
Main module for the application.
"""

from typing import Optional

import typer

from ir_model import IRModel
from query import Query

app = typer.Typer(add_completion=False)


@app.command("test")
def test_model():
    """
    Test the model
    """
    typer.echo("Not implemented")
    raise typer.Exit()


@app.command("single")
def single_query(query: str):
    """
    Process a single query
    """
    # Process the user query
    # ...

    # Search documents
    # ...

    # Display results
    # ...
    typer.echo("Not implemented")
    raise typer.Exit()


@app.command("continuous")
def continuous_queries():
    """
    Continuously read user queries, search documents and display results
    """
    while True:
        query = input("Enter query: ")
        if query in ("", "exit"):
            break
        single_query(query)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    database: Optional[str] = typer.Option(
        default="cran",
        help="Name of the database to use",
    ),
    rebuild: Optional[bool] = typer.Option(
        default=False,
        help="Rebuild the index",
    ),
):
    """
    SRI Final Project

    By default the progrma will run the continuous queries command.

    Authors: Jorge Morgado Vega (jorge.morgadov@gmail.com) and Roberto
    García Rodríguez (roberto.garcia@estudiantes.matcom.uh.cu)
    """
    # Load the model according to the database and rebuild parameters
    # ...

    # Run the continuous queries command by default
    if ctx.invoked_subcommand is None:
        continuous_queries()


if __name__ == "__main__":
    app()
