"""Interactive user administration jobs. Not scheduled."""

import typer
from fivcglue import IComponentSite, query_component
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from fivccliche.services.interfaces.auth import IUserAuthenticator
from fivccliche.services.interfaces.modules import IModuleJob
from fivccliche.utils.deps import get_db_session_context_async

from .utils import get_user_async, update_user_async

console = Console()


class CreateSuperuserJob(IModuleJob):
    """Prompt for an admin account and create it. ``config`` is ``None``."""

    def __init__(self, component_site: IComponentSite) -> None:
        self._component_site = component_site

    @property
    def name(self) -> str:
        return "createsuperuser"

    @property
    def config(self) -> dict | None:
        return None

    async def run_async(self, *args, **kwargs) -> None:
        """Prompt for username, email, and password, then create a superuser."""
        console.print(
            Panel.fit(
                Text("Create Superuser", style="bold blue"),
                subtitle="Create a new admin account",
            )
        )
        try:
            username = typer.prompt("Username")
            if not username or not username.strip():
                console.print("[red]❌ Username cannot be empty[/red]")
                raise typer.Exit(1)

            email = typer.prompt("Email address")
            if not email or not email.strip():
                console.print("[red]❌ Email cannot be empty[/red]")
                raise typer.Exit(1)

            password = typer.prompt("Password", hide_input=True)
            if not password or not password.strip():
                console.print("[red]❌ Password cannot be empty[/red]")
                raise typer.Exit(1)

            password_confirm = typer.prompt("Confirm password", hide_input=True)
            if password != password_confirm:
                console.print("[red]❌ Passwords do not match[/red]")
                raise typer.Exit(1)

            auth = query_component(self._component_site, IUserAuthenticator)
            async with get_db_session_context_async() as session:
                existing_user = await get_user_async(session, username=username)
                if existing_user:
                    console.print(f"[red]❌ User '{username}' already exists[/red]")
                    raise typer.Exit(1)

                existing_email = await get_user_async(session, email=email)
                if existing_email:
                    console.print(f"[red]❌ Email '{email}' is already in use[/red]")
                    raise typer.Exit(1)

            user = await auth.create_user_async(
                username=username,
                email=email,
                password=password,
                is_superuser=True,
            )
            if not user:
                console.print("[red]❌ Failed to create superuser[/red]")
                raise typer.Exit(1)

            console.print("\n" + "=" * 60)
            console.print("[bold green]✅ Superuser created successfully![/bold green]")
            console.print("=" * 60)
            console.print(f"[cyan]Username:[/cyan] {user.username}")
            console.print(f"[cyan]Email:[/cyan] {user.email}")
            console.print("[cyan]Admin:[/cyan] Yes")
            console.print("=" * 60)
        except typer.Abort as e:
            console.print("[yellow]Superuser creation cancelled[/yellow]")
            raise typer.Exit(0) from e
        except typer.Exit:
            raise
        except ValueError as e:
            console.print(f"[red]❌ Validation error: {e}[/red]")
            raise typer.Exit(1) from e
        except Exception as e:
            console.print(f"[red]❌ Database error: {e}[/red]")
            raise typer.Exit(1) from e


class ChangePasswordJob(IModuleJob):
    """Prompt for a username and new password. ``config`` is ``None``."""

    def __init__(self, component_site: IComponentSite) -> None:
        self._component_site = component_site

    @property
    def name(self) -> str:
        return "changepassword"

    @property
    def config(self) -> dict | None:
        return None

    async def run_async(self, *args, **kwargs) -> None:
        """Prompt for a username and new password, then update that user."""
        console.print(
            Panel.fit(
                Text("Change Password", style="bold blue"),
                subtitle="Update user password",
            )
        )
        try:
            username = typer.prompt("Username")
            if not username or not username.strip():
                console.print("[red]❌ Username cannot be empty[/red]")
                raise typer.Exit(1)

            new_password = typer.prompt("New password", hide_input=True)
            if not new_password or not new_password.strip():
                console.print("[red]❌ Password cannot be empty[/red]")
                raise typer.Exit(1)

            password_confirm = typer.prompt("Confirm new password", hide_input=True)
            if new_password != password_confirm:
                console.print("[red]❌ Passwords do not match[/red]")
                raise typer.Exit(1)

            async with get_db_session_context_async() as session:
                user = await get_user_async(session, username=username)
                if not user:
                    console.print(f"[red]❌ User '{username}' not found[/red]")
                    raise typer.Exit(1)
                saved_username = user.username
                await update_user_async(session, user, password=new_password)
                await session.commit()

            console.print("\n" + "=" * 60)
            console.print("[bold green]✅ Password changed successfully![/bold green]")
            console.print("=" * 60)
            console.print(f"[cyan]Username:[/cyan] {saved_username}")
            console.print("=" * 60)
        except typer.Abort as e:
            console.print("[yellow]Password change cancelled[/yellow]")
            raise typer.Exit(0) from e
        except typer.Exit:
            raise
        except Exception as e:
            console.print(f"[red]❌ Database error: {e}[/red]")
            raise typer.Exit(1) from e
