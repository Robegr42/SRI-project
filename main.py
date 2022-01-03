"""
Main module for the application.
"""

from pathlib import Path
from typing import Optional

import typer

from cran_db_creator import create_db as create_cran_db
from ir_model import IRModel
from query import Query

app = typer.Typer(add_completion=False)

status = {}


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
        help="Rebuild the database (generate 'docs.json' and 'metadata.json')",
    ),
    reindex: Optional[bool] = typer.Option(
        default=False,
        help="Reindex the database (generate IR model)",
    ),
):
    """
    SRI Final Project

    By default the progrma will run the continuous queries command.

    Authors: Jorge Morgado Vega (jorge.morgadov@gmail.com) and Roberto
    García Rodríguez (roberto.garcia@estudiantes.matcom.uh.cu)
    """

    # Load the model according to the database and rebuild parameters
    db_folder = Path(f"./database/{database}")
    if not db_folder.exists() or rebuild:
        if not rebuild:
            raise typer.Exit(
                f"The database '{database}' does not exist. Use the --rebuild\n"
                "option to create it.\n\n"
                "NOTE: The --rebuild option is only available for the\n"
                "cran database."
            )
        if database not in ("cran",):
            raise typer.Exit(
                f"Rebuild not suported for database '{database}'.\n\n"
                "Please build your own database using the 'DatabaseCreator'\n"
                "class available in 'database_creator.py'"
            )
        typer.echo(f"Rebuilding the '{database}' database")
        if database == "cran":
            create_cran_db()
        status["model"] = IRModel(str(db_folder), True)
    else:
        status["model"] = IRModel(str(db_folder), reindex)

    # Run the continuous queries command by default
    if ctx.invoked_subcommand is None:
        continuous_queries()


if __name__ == "__main__":
    app()
