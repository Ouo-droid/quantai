#!/usr/bin/env python
"""
scripts/run_pipeline.py
------------------------
Pipeline QuantAI complet : OpenBB -> Signal -> Decision -> Risk -> Alpaca paper

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
from execution.decision_logic import DecisionParams, TemporalDecisionEngine
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
        print("X Impossible de recuperer le compte Alpaca")
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
        header = "  " + f"{'Symbol':<8} {'Qty':>8} {'Side':>6} {'Entry':>8} {'Value':>10} {'P&L%':>8}"
        print(header)
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
    """Execute le pipeline complet pour un symbole."""
    t0 = time.perf_counter()
    result: dict = {"symbol": symbol, "status": "error"}

    try:
        # Layer 0 -- Data
        prices = client.ohlcv(symbol, start="2022-01-01")
        news = client.news(symbol, limit=10) if use_mirofish else []

        # Layer 1 -- Signal
        vector = agg.compute(
            prices,
            symbol=symbol,
            use_quantmuse=use_quantmuse,
            use_mirofish=use_mirofish,
            mirofish_news=news if use_mirofish else None,
        )

        # Layer 2a -- Compte Alpaca pour sizing reel
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

        sep = "-" * 55
        print(f"\n{sep}")
        now_str = datetime.now().strftime("%H:%M:%S")
        print(f"  {symbol} -- {now_str}")
        cs = f"{vector.composite_score:+.3f}" if vector.composite_score is not None else "N/A"
        ml = f"{vector.ml_prediction:+.3f}" if vector.ml_prediction is not None else "N/A"
        mf = f"{vector.mirofish_sentiment:+.3f}" if vector.mirofish_sentiment is not None else "N/A"
        print(f"  composite={cs}  ml={ml}  mirofish={mf}")

        if order is None:
            print("  -> BLOQUE par RiskEngine")
            result["status"] = "blocked"
        else:
            rationale = getattr(order, "rationale", "")[:40]
            print(
                f"  -> {order.direction} conf={order.confidence:.2f} "
                f"size={order.size_pct:.1%} | {rationale}"
            )

            if order.direction != "FLAT" and not dry_run and router.is_available():
                fill = router.submit(order, symbol=symbol, account_value=account_equity)
                if fill:
                    print(f"  -> ORDRE SOUMIS : {fill}")
                    result["fill"] = str(fill)
                    result["status"] = "submitted"
                else:
                    print("  -> Ordre non soumis (indisponible ou erreur)")
                    result["status"] = "failed"
            elif dry_run:
                print("  -> DRY RUN -- ordre non soumis")
                result["status"] = "dry_run"
            else:
                result["status"] = "flat"

        print(f"  [{elapsed:.0f}ms]")
        result.update({
            "direction": order.direction if order else "BLOCKED",
            "composite": vector.composite_score,
            "ml": vector.ml_prediction,
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
    parser.add_argument("--mirofish", action="store_true", help="Activer MiroFish")
    parser.add_argument("--dry-run", action="store_true", help="Simuler sans soumettre")
    parser.add_argument("--status", action="store_true", help="Afficher compte + positions")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  QUANTAI PIPELINE")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"  {now_str}")
    print("=" * 60)

    router = AlpacaRouter()
    alpaca_ok = router.is_available()
    alpaca_msg = "OK connecte" if alpaca_ok else "!! non disponible (dry run)"
    print(f"\n  Alpaca paper : {alpaca_msg}")

    if not alpaca_ok:
        args.dry_run = True

    if args.status:
        if alpaca_ok:
            show_status(router)
        else:
            print("  X Alpaca non disponible")
        return

    symbols = args.symbols if args.symbols else (DEFAULT_UNIVERSE if args.universe else [])
    if not symbols:
        print("  Usage : python scripts/run_pipeline.py AAPL [MSFT ...] [--universe]")
        return

    sym_list = ", ".join(symbols)
    qm_flag = "Y" if args.quantmuse else "N"
    mf_flag = "Y" if args.mirofish else "N"
    dr_flag = "Y" if args.dry_run else "N"
    print(f"  Symboles : {sym_list}")
    print(f"  Capital  : ${args.capital:,.0f}")
    print(f"  QuantMuse: {qm_flag}")
    print(f"  MiroFish : {mf_flag}")
    print(f"  Dry run  : {dr_flag}")

    client = OpenBBClient()
    assert client.health(), "OpenBB non joignable -- lancer openbb-api"
    agg = SignalAggregator()
    agent = DecisionAgent()

    # Temporal decision engine
    engine = TemporalDecisionEngine(DecisionParams(
        min_confirmation_days=2,
        min_composite_score=0.30,
        min_confidence=0.70,
        min_signals_aligned=3,
        trading_window_start=10,
        trading_window_end=15,
    ))

    # Afficher le statut actuel
    print("\n=== ETAT DES SIGNAUX ===")
    status_df = engine.status()
    print(status_df.to_string() if not status_df.empty else "  (aucun historique)")

    t_start = time.perf_counter()

    # Collecter tous les signaux
    signals: dict = {}
    results: list[dict] = []
    for symbol in symbols:
        try:
            prices = client.ohlcv(symbol, start="2022-01-01")
            news = client.news(symbol, limit=10) if args.mirofish else []

            vector = agg.compute(
                prices,
                symbol=symbol,
                use_quantmuse=args.quantmuse,
                use_mirofish=args.mirofish,
                mirofish_news=news if args.mirofish else None,
            )

            account = router.get_account() if router.is_available() else None
            account_equity = account["equity"] if account else args.capital

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

            signals[symbol] = (vector, order)

            cs = f"{vector.composite_score:+.3f}" if vector.composite_score is not None else "N/A"
            ml = f"{vector.ml_prediction:+.3f}" if vector.ml_prediction is not None else "N/A"
            direction = order.direction if order else "BLOCKED"
            confidence = f"{order.confidence:.2f}" if order else "N/A"
            print(f"\n  {symbol}: composite={cs} ml={ml} -> {direction} (conf={confidence})")

        except Exception as e:
            logger.error(f"{symbol} pipeline error: {e}")
            results.append({"symbol": symbol, "status": "error", "error": str(e)})

    # Decision temporelle
    actions = engine.process_universe(signals)

    # Executer seulement les actions autorisees
    for action in actions:
        separator = "_" * 55
        print(f"\n{separator}")
        print(f"  {action}")
        if not args.dry_run and router.is_available() and action.trade_order is not None:
            account = router.get_account()
            account_value = account["equity"] if account else args.capital
            fill = router.submit(action.trade_order, action.symbol, account_value)
            if fill:
                print(f"  -> ORDRE SOUMIS : {fill}")
                results.append({"symbol": action.symbol, "status": "submitted", "fill": str(fill)})
            else:
                print("  -> Ordre non soumis (indisponible ou erreur)")
                results.append({"symbol": action.symbol, "status": "failed"})
        else:
            print("  -> DRY RUN")
            results.append({"symbol": action.symbol, "status": "dry_run"})

    # Afficher le statut mis a jour
    print("\n=== ETAT DES SIGNAUX (apres) ===")
    print(engine.status().to_string())

    total = (time.perf_counter() - t_start) * 1000
    eq_sep = "=" * 60
    print(f"\n{eq_sep}")
    n_sym = len(symbols)
    print(f"  RESUME -- {n_sym} symboles -- {total:.0f}ms total")
    submitted = [r for r in results if r.get("status") == "submitted"]
    blocked = [r for r in results if r.get("status") == "blocked"]
    flat_dry = [r for r in results if r.get("status") in ("flat", "dry_run")]
    n_actions = len(actions)
    print(f"  Actions temporelles : {n_actions}")
    print(f"  Ordres soumis : {len(submitted)}")
    print(f"  Bloques Risk  : {len(blocked)}")
    print(f"  FLAT/DryRun   : {len(flat_dry)}")

    if alpaca_ok:
        show_status(router)


if __name__ == "__main__":
    main()
