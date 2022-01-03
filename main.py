"""
Main module for the application.
"""

import typer

from ir_model import IRModel
from query import Query

app = typer.Typer(add_completion=False)

@app.command("test")
def test_model():
    """
    Test the model
    """
    raise NotImplementedError


@app.command("single")
def single_query(query: str):
    """
    Process a single query
    """
    # Process the user query

    # Search documents

    # Display results
    raise NotImplementedError


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
def main(ctx: typer.Context):
    """
    SRI Final Project

    By default the progrma will run the continuous queries command.

    Authors: Jorge Morgado Vega (jorge.morgadov@gmail.com) and Roberto
    García Rodríguez (roberto.garcia@estudiantes.matcom.uh.cu)
    """
    if ctx.invoked_subcommand is None:
        typer.echo("Initializing database")


if __name__ == "__main__":
    app()
