#!/usr/bin/env python
"""
FivcCliche CLI

Command-line interface for FivcCliche - a production-ready, multi-user backend
framework for AI agents built with FastAPI and SQLModel.
"""

import asyncio
import os
import shutil
from pathlib import Path
from typing import cast

import typer
from sqlalchemy import text
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from fivcglue import query_component, IComponentSite
from fivccliche import __version__
from fivccliche.services.implements import service_site
from fivccliche.services.interfaces.modules import IModule, IModuleJob, IModuleSite
from fivccliche.services.interfaces.db import IDatabase

cli = typer.Typer(
    name="FivcCliche",
    help="FivcCliche - Production-ready AI agent backend framework",
    rich_markup_mode="rich",
)

jobs_cli = typer.Typer(
    name="jobs",
    help="List, inspect, and run module jobs",
    rich_markup_mode="rich",
)
cli.add_typer(jobs_cli, name="jobs")

console = Console()

_LEGACY_TTS_PROVIDER_SQL = text(
    "UPDATE user_tts SET model_type = 'dashscope_realtime' WHERE model_type = 'dashscope_tts'"
)

modules: IModuleSite = query_component(cast(IComponentSite, service_site), IModuleSite)

# FastAPI App for ASGI
app = modules.create_application()


def _find_module(module_name: str) -> IModule:
    module = modules.get_module(module_name)
    if module is None:
        console.print(f"[red]❌ Module '{module_name}' not found[/red]")
        raise typer.Exit(1)
    return module


def _find_job(module: IModule, job_name: str) -> IModuleJob:
    job = module.get_job(job_name)
    if job is None:
        console.print(f"[red]❌ Job '{job_name}' not found in module '{module.name}'[/red]")
        raise typer.Exit(1)
    return job


@cli.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind the server to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to run the server on"),
    reload: bool = typer.Option(
        True, "--reload/--no-reload", help="Enable auto-reload on code changes"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be done without executing"
    ),
):
    """
    Start the FivcCliche FastAPI application
    """
    console.print(
        Panel.fit(
            Text("FivcCliche Server", style="bold blue"),
            subtitle="Production-ready AI agent backend",
        )
    )

    if dry_run:
        console.print("[yellow]DRY RUN:[/yellow] Would start FivcCliche server")
        console.print(f"[yellow]Host:[/yellow] {host}")
        console.print(f"[yellow]Port:[/yellow] {port}")
        console.print(f"[yellow]Reload:[/yellow] {reload}")
        return

    try:
        console.print(f"[blue]Starting server at http://{host}:{port}[/blue]")
        console.print("[yellow]Press Ctrl+C to stop the server[/yellow]")

        if verbose:
            console.print("[cyan]Verbose mode enabled[/cyan]")

        modules.run_application(app, host=host, port=port)

    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Error running server: {e}[/red]")
        raise typer.Exit(1) from e


@cli.command()
def info():
    """
    Show information about FivcCliche
    """
    info_text = f"""
    [bold blue]FivcCliche[/bold blue] v{__version__}

    A production-ready, multi-user backend framework for AI agents.
    Built with FastAPI and SQLModel for high-performance, type-safe
    async operations that handle concurrent AI agent requests at scale.

    [bold]Features:[/bold]
    • Production-ready multi-user backend
    • Designed for AI agent interactions
    • Async architecture for concurrent request handling
    • Type-safe with Pydantic 2.0 validation
    • SQLModel ORM for reliable data persistence
    • Built on FastAPI for high performance
    • Modular architecture with component system

    [bold]Usage Examples:[/bold]
    fivccliche migrate                                # Initialize database tables
    fivccliche exec users createsuperuser             # Create admin account
    fivccliche exec users changepassword              # Change a user's password
    fivccliche serve                                  # Start server
    fivccliche serve --port 9000                      # Custom port
    fivccliche serve --host 127.0.0.1 --no-reload    # Production mode
    fivccliche jobs list                              # List module jobs
    fivccliche jobs show MODULE JOB                   # Show job config
    fivccliche exec MODULE JOB                        # Run a job immediately
    fivccliche jobs exec MODULE JOB                   # Same as exec
    fivccliche info                                   # Show this information
    fivccliche clean                                  # Clean temporary files
    """

    console.print(Panel(info_text, title="FivcCliche", border_style="blue"))


@cli.command()
def clean():
    """
    Clean up temporary files and cache generated by FivcCliche
    """
    console.print("[yellow]Cleaning up temporary files...[/yellow]")

    try:
        # Define directories to clean
        cache_dirs = [
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "__pycache__",
            ".coverage",
            "htmlcov",
        ]

        cleaned_count = 0

        for cache_dir in cache_dirs:
            cache_path = Path(cache_dir)
            if cache_path.exists():
                try:
                    if cache_path.is_dir():
                        shutil.rmtree(cache_path)
                    else:
                        cache_path.unlink()
                    console.print(f"[green]✅ Removed: {cache_dir}[/green]")
                    cleaned_count += 1
                except Exception as e:
                    console.print(f"[yellow]⚠️  Could not remove {cache_dir}: {e}[/yellow]")

        # Clean Python cache files
        for root, _dirs, files in os.walk("."):
            for file in files:
                if file.endswith(".pyc"):
                    try:
                        os.remove(os.path.join(root, file))
                        cleaned_count += 1
                    except Exception:
                        pass

        console.print("\n" + "=" * 60)
        console.print("[bold cyan]Cleanup Summary[/bold cyan]")
        console.print("=" * 60)
        console.print(f"[green]Items cleaned: {cleaned_count}[/green]")
        console.print("[green]✅ Cleanup completed successfully![/green]")

    except Exception as e:
        console.print(f"[red]❌ Error during cleanup: {e}[/red]")
        raise typer.Exit(1) from e


@cli.command()
def migrate():
    """
    Initialize and create database tables.

    This command creates all database tables defined in SQLModel models.
    Run this command before using other commands that require database access.
    """
    console.print(
        Panel.fit(
            Text("Database Migration", style="bold blue"),
            subtitle="Initialize database tables",
        )
    )

    try:
        asyncio.run(_migrate_async())
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {e}[/red]")
        raise typer.Exit(1) from e


async def _migrate_async() -> None:
    """
    Async helper function to initialize database tables.
    """
    try:
        console.print("[cyan]Initializing database tables...[/cyan]")

        # Get database service
        db_service = query_component(cast(IComponentSite, service_site), IDatabase)

        # Create all database tables
        async with db_service.get_engine().begin() as conn:
            await conn.run_sync(db_service.get_metadata().create_all)
            await conn.execute(_LEGACY_TTS_PROVIDER_SQL)

        console.print("\n" + "=" * 60)
        console.print("[bold green]✅ Database tables created successfully![/bold green]")
        console.print("=" * 60)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]❌ Database error: {e}[/red]")
        raise typer.Exit(1) from e


@jobs_cli.command("list")
def jobs_list():
    """
    List all jobs exposed by registered modules.
    """
    table = Table(title="Module Jobs")
    table.add_column("Module", style="cyan")
    table.add_column("Job", style="green")
    table.add_column("Config", style="white")

    job_count = 0
    for module in modules.list_modules():
        for job in module.list_jobs():
            table.add_row(module.name, job.name, str(job.config))
            job_count += 1

    if job_count == 0:
        console.print("[yellow]No jobs registered in any module.[/yellow]")
        return

    console.print(table)


@jobs_cli.command("show")
def jobs_show(
    module_name: str = typer.Argument(..., help="Module name"),
    job_name: str = typer.Argument(..., help="Job name"),
):
    """
    Show details for a specific module job.
    """
    module = _find_module(module_name)
    job = _find_job(module, job_name)

    console.print(
        Panel.fit(
            Text(f"{module.name} / {job.name}", style="bold blue"),
            subtitle="Module job",
        )
    )
    console.print(f"[cyan]Name:[/cyan] {job.name}")
    console.print(f"[cyan]Config:[/cyan] {job.config}")


@cli.command("exec")
@jobs_cli.command("exec")
def jobs_exec(
    module_name: str = typer.Argument(..., help="Module name"),
    job_name: str = typer.Argument(..., help="Job name"),
):
    """
    Run a module job immediately (outside the scheduler).
    """
    module = _find_module(module_name)
    job = _find_job(module, job_name)

    console.print(
        Panel.fit(
            Text(f"Run {module.name} / {job.name}", style="bold blue"),
            subtitle="Immediate job execution",
        )
    )

    try:
        console.print(f"[cyan]Running job '{job.name}'...[/cyan]")
        asyncio.run(job.run_async())
        console.print("\n" + "=" * 60)
        console.print("[bold green]✅ Job completed successfully![/bold green]")
        console.print("=" * 60)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]❌ Job failed: {e}[/red]")
        raise typer.Exit(1) from e


def main():
    """
    Main entry point for the CLI
    """
    cli()


if __name__ == "__main__":
    main()
