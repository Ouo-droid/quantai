#!/usr/bin/env python
"""
scripts/check_openbb.py
-----------------------
Vérifie que le serveur OpenBB est actif et que les données arrivent.

Usage :
    # 1. Dans un terminal séparé :
    openbb-api

    # 2. Dans un autre terminal :
    uv run python scripts/check_openbb.py
"""

from rich.console import Console
from rich.table import Table
from rich import print as rprint
from data.client import OpenBBClient

console = Console()


def main():
    console.rule("[bold]QuantAI — OpenBB check[/bold]")

    client = OpenBBClient()

    # 1. Health check
    if not client.health():
        rprint("[red]✗ OpenBB API non joignable.[/red]")
        rprint("  Lance d'abord : [bold]openbb-api[/bold]")
        return

    rprint("[green]✓ OpenBB API joignable[/green]")

    # 2. OHLCV
    console.print("\n[dim]→ Test OHLCV (AAPL, 30 jours)...[/dim]")
    try:
        df = client.ohlcv("AAPL", start="2024-01-01")
        rprint(f"[green]✓ OHLCV : {len(df)} barres reçues[/green]")

        table = Table(title="5 dernières barres AAPL")
        for col in ["open", "high", "low", "close", "volume"]:
            table.add_column(col, justify="right")
        for idx, row in df.tail(5).iterrows():
            table.add_row(
                f"{row['open']:.2f}", f"{row['high']:.2f}",
                f"{row['low']:.2f}", f"{row['close']:.2f}",
                f"{row['volume']:,.0f}"
            )
        console.print(table)
    except Exception as e:
        rprint(f"[red]✗ OHLCV failed: {e}[/red]")

    # 3. Macro
    console.print("\n[dim]→ Test macro dashboard...[/dim]")
    try:
        macro = client.macro_dashboard()
        rprint("[green]✓ Macro OK[/green]")
        console.print(macro.to_string())
    except Exception as e:
        rprint(f"[yellow]⚠ Macro partiel: {e}[/yellow]")

    # 4. News
    console.print("\n[dim]→ Test news (AAPL, 3 articles)...[/dim]")
    try:
        news = client.news("AAPL", limit=3)
        rprint(f"[green]✓ News : {len(news)} articles[/green]")
        for n in news:
            rprint(f"  [dim]{n.date.date()}[/dim] {n.title[:80]}")
    except Exception as e:
        rprint(f"[yellow]⚠ News partiel (provider dépendant): {e}[/yellow]")

    console.rule("[bold green]Tout bon — data layer opérationnel[/bold green]")


if __name__ == "__main__":
    main()
