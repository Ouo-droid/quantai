#!/usr/bin/env python
"""
scripts/run_pipeline.py
------------------------
Pipeline QuantAI complet : OpenBB → Signal → Decision → Risk → Alpaca paper

Usage :
    python scripts/run_pipeline.py AAPL
    python scripts/run_pipeline.py AAPL MSFT NVDA --capital 100000
    python scripts/run_pipeline.py AAPL --quantmuse --mirofish
    python scripts/run_pipeline.py --universe  # top 10 S&P 500
    python scripts/run_pipeline.py --status    # positions + P&L
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from data.client import OpenBBClient
from execution.decision_agent import DecisionAgent
from execution.risk import Portfolio, RiskLimits
from execution.router import AlpacaRouter
from signals.aggregator import SignalAggregator

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "JPM", "V", "UNH",
]


def show_status(router: AlpacaRouter) -> None:
    """Affiche le statut du compte et les positions."""
    account = router.get_account()
    if not account:
        print("✗ Impossible de récupérer le compte Alpaca")
        return

    print("\n" + "=" * 60)
    print("  COMPTE ALPACA PAPER")
    print("=" * 60)
    print(f"  Cash           : ${account['cash']:>12,.2f}")
    print(f"  Equity         : ${account['equity']:>12,.2f}")
    print(f"  Buying Power   : ${account['buying_power']:>12,.2f}")
    print(f"  P&L aujourd'hui: ${account['pnl_today']:>+11,.2f}")

    positions = router.get_positions()
    if positions:
        print(f"\n  POSITIONS OUVERTES ({len(positions)})")
        print(f"  {'Symbol':<8} {'Qty':>8} {'Side':>6} {'Entry':>8} {'Value':>10} {'P&L%':>8}")
        print("  " + "-" * 52)
        for p in positions:
            print(
                f"  {p['symbol']:<8} {p['qty']:>8.4f} {p['side']:>6} "
                f"${p['avg_entry']:>7.2f} ${p['market_val']:>9.2f} "
                f"{p['unrealized_pct']:>+7.2f}%"
            )
    else:
        print("\n  Aucune position ouverte")

    orders = router.get_orders(limit=5)
    if orders:
        print("\n  DERNIERS ORDRES")
        for o in orders:
            print(
                f"  {o['side'].upper():<5} {o['symbol']:<8} "
                f"qty={o['filled_qty']:.4f}/{o['qty']:.4f} "
                f"status={o['status']}"
            )


def run_symbol(
    symbol: str,
    client: OpenBBClient,
    agg: SignalAggregator,
    agent: DecisionAgent,
    router: AlpacaRouter,
    capital: float,
    use_quantmuse: bool = False,
    use_mirofish: bool = False,
    dry_run: bool = False,
) -> dict:
    """Exécute le pipeline complet pour un symbole."""
    t0 = time.perf_counter()
    result: dict = {"symbol": symbol, "status": "error"}

    try:
        # Layer 0 — Data
        prices = client.ohlcv(symbol, start="2022-01-01")
        news = client.news(symbol, limit=10) if use_mirofish else []

        # Layer 1 — Signal
        vector = agg.compute(
            prices,
            symbol=symbol,
            use_quantmuse=use_quantmuse,
            use_mirofish=use_mirofish,
            mirofish_news=news if use_mirofish else None,
        )

        # Layer 2a — Compte Alpaca pour sizing réel
        account = router.get_account() if router.is_available() else None
        account_equity = account["equity"] if account else capital

        # Portfolio avec rendements relatifs (pas prix bruts)
        portfolio = Portfolio(cash=account_equity)
        recent_returns = list(prices["close"].pct_change().dropna().values[-252:])
        portfolio.equity_history = [
            account_equity * (1 + r) for r in [0.0] + recent_returns[-99:]
        ]

        order = agent.decide_with_risk(
            vector,
            portfolio,
            recent_returns=recent_returns,
            limits=RiskLimits(
                max_position_pct=0.05,
                daily_var_95=0.02,
                max_drawdown=0.15,
                min_confidence=0.60,
            ),
        )

        elapsed = (time.perf_counter() - t0) * 1000

        print(f"\n{'─' * 55}")
        print(f"  {symbol} — {datetime.now().strftime('%H:%M:%S')}")
        cs = f"{vector.composite_score:+.3f}" if vector.composite_score is not None else "N/A"
        ml = f"{vector.ml_prediction:+.3f}" if vector.ml_prediction is not None else "N/A"
        mf = f"{vector.mirofish_sentiment:+.3f}" if vector.mirofish_sentiment is not None else "N/A"
        print(f"  composite={cs}  ml={ml}  mirofish={mf}")

        if order is None:
            print("  → BLOQUÉ par RiskEngine")
            result["status"] = "blocked"
        else:
            rationale = getattr(order, "rationale", "")[:40]
            print(
                f"  → {order.direction} conf={order.confidence:.2f} "
                f"size={order.size_pct:.1%} | {rationale}"
            )

            if order.direction != "FLAT" and not dry_run and router.is_available():
                fill = router.submit(order, symbol=symbol, account_value=account_equity)
                if fill:
                    print(f"  → ORDRE SOUMIS : {fill}")
                    result["fill"] = str(fill)
                    result["status"] = "submitted"
                else:
                    print("  → Ordre non soumis (indisponible ou erreur)")
                    result["status"] = "failed"
            elif dry_run:
                print("  → DRY RUN — ordre non soumis")
                result["status"] = "dry_run"
            else:
                result["status"] = "flat"

        print(f"  [{elapsed:.0f}ms]")
        result.update({
            "direction":  order.direction if order else "BLOCKED",
            "composite":  vector.composite_score,
            "ml":         vector.ml_prediction,
            "elapsed_ms": elapsed,
        })

    except Exception as e:
        logger.error(f"{symbol} pipeline error: {e}")
        result["error"] = str(e)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="QuantAI Pipeline")
    parser.add_argument("symbols", nargs="*", default=[])
    parser.add_argument("--universe", action="store_true", help="Univers top 10 S&P 500")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--quantmuse", action="store_true", help="Activer QuantMuse ML")
    parser.add_argument("--mirofish",  action="store_true", help="Activer MiroFish")
    parser.add_argument("--dry-run",   action="store_true", help="Simuler sans soumettre")
    parser.add_argument("--status",    action="store_true", help="Afficher compte + positions")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  QUANTAI PIPELINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    router = AlpacaRouter()
    alpaca_ok = router.is_available()
    print(f"\n  Alpaca paper : {'✓ connecté' if alpaca_ok else '⚠ non disponible (dry run)'}")

    if not alpaca_ok:
        args.dry_run = True

    if args.status:
        if alpaca_ok:
            show_status(router)
        else:
            print("  ✗ Alpaca non disponible")
        return

    symbols = args.symbols if args.symbols else (DEFAULT_UNIVERSE if args.universe else [])
    if not symbols:
        print("  Usage : python scripts/run_pipeline.py AAPL [MSFT ...] [--universe]")
        return

    print(f"  Symboles : {', '.join(symbols)}")
    print(f"  Capital  : ${args.capital:,.0f}")
    print(f"  QuantMuse: {'✓' if args.quantmuse else '✗'}")
    print(f"  MiroFish : {'✓' if args.mirofish else '✗'}")
    print(f"  Dry run  : {'✓' if args.dry_run else '✗'}")

    client = OpenBBClient()
    assert client.health(), "OpenBB non joignable — lancer openbb-api"
    agg   = SignalAggregator()
    agent = DecisionAgent()

    t_start = time.perf_counter()
    results = []
    for symbol in symbols:
        result = run_symbol(
            symbol, client, agg, agent, router,
            capital=args.capital,
            use_quantmuse=args.quantmuse,
            use_mirofish=args.mirofish,
            dry_run=args.dry_run,
        )
        results.append(result)

    total = (time.perf_counter() - t_start) * 1000
    print(f"\n{'=' * 60}")
    print(f"  RÉSUMÉ — {len(results)} symboles · {total:.0f}ms total")
    submitted = [r for r in results if r.get("status") == "submitted"]
    blocked   = [r for r in results if r.get("status") == "blocked"]
    flat_dry  = [r for r in results if r.get("status") in ("flat", "dry_run")]
    print(f"  Ordres soumis : {len(submitted)}")
    print(f"  Bloqués Risk  : {len(blocked)}")
    print(f"  FLAT/DryRun   : {len(flat_dry)}")

    if alpaca_ok:
        show_status(router)


if __name__ == "__main__":
    main()
