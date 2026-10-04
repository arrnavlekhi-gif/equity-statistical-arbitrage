"""
walkforward_selection_experiment.py

MASTER NON-COINTEGRATION PAIRS EXPERIMENT

Tests three distinct pairs-trading signal families over the same S&P 500
peer-group universe and the same walk-forward dates:

    1) Gatev-style distance / normalized-price spread
    2) Dynamic Kalman hedge-ratio spread
    3) Factor/residual relative-value spread

Each signal is evaluated under:
    - Equal-weight active pairs
    - Inverse-volatility
    - Constrained minimum-variance

Cointegration is intentionally NOT used in this experiment.

Important research controls
---------------------------
* 252-day formation window by default.
* 21-day trading blocks by default.
* Signals are estimated using formation data or sequential data available
  through the current close only.
* Trades are executed by the project's existing next-day execution engine.
* Transaction and short-borrow costs use the existing execution engine.
* Pair selection uses formation-period signal statistics, never realised
  future P&L or future Sharpe.
* All 9 portfolios use the SAME generated pair-strategy returns, so the
  allocation comparison is apples-to-apples.

Run from the project root:
    python walkforward_selection_experiment.py

Requirements:
    numpy pandas yfinance scipy
and the project's existing:
    analytics/pair_scanner.py
    analytics/portfolio_engine.py
    strategy/execution.py

NOTE:
For "Constrained minimum-variance", analytics/portfolio_engine.py must be the
optimized version that includes that allocation method.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from analytics.pair_scanner import (
    get_industries,
    generate_industry_pairs,
    download_industry,
    prepare_pair_data,
)
from analytics.portfolio_engine import (
    combine_pair_strategies,
    portfolio_statistics,
)
from strategy.execution import run_execution


# ============================================================
# GENERIC HELPERS
# ============================================================

def safe_std(x):
    x = pd.Series(x).dropna()
    return float(x.std(ddof=1)) if len(x) > 1 else np.nan


def half_life_from_series(x):
    """AR(1)-style half-life diagnostic. No cointegration test is used."""
    s = pd.Series(x).dropna()
    if len(s) < 30:
        return np.nan
    lag = s.shift(1)
    delta = s.diff()
    reg = pd.concat([lag.rename("lag"), delta.rename("delta")], axis=1).dropna()
    if len(reg) < 20:
        return np.nan
    X = np.column_stack([np.ones(len(reg)), reg["lag"].values])
    y = reg["delta"].values
    try:
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        lam = float(coef[1])
        if not np.isfinite(lam) or lam >= 0:
            return np.nan
        return float(-np.log(2.0) / lam)
    except Exception:
        return np.nan


def base_model(index, pair):
    out = pd.DataFrame(index=index)
    out["open_a"] = pair["open_a"]
    out["open_b"] = pair["open_b"]
    out["close_a"] = pair["close_a"]
    out["close_b"] = pair["close_b"]
    out["alpha"] = np.nan
    out["beta"] = np.nan
    out["r_squared"] = np.nan
    out["spread"] = np.nan
    out["zscore"] = np.nan
    out["half_life"] = np.nan
    out["orientation"] = "A_on_B"
    out["pair_valid"] = False
    out["block_start"] = pd.NaT
    out["selection_score"] = np.nan
    return out


def rolling_blocks(n, formation, block):
    start = int(formation)
    while start < n:
        end = min(start + int(block), n)
        yield start, end
        start = end


# ============================================================
# MODEL 1: GATEV-STYLE DISTANCE
# ============================================================

def build_distance_model(pair, formation=252, block=21, z_window=60):
    """
    Gatev-style idea:
    normalize log-price paths over the formation window and measure squared
    distance.  The trading spread is the difference between normalized log
    prices, standardized using the end of the formation period.

    selection_score = formation SSD (lower is better).
    """
    px = pair[["close_a", "close_b"]].dropna()
    out = base_model(pair.index, pair)

    for start, end in rolling_blocks(len(px), formation, block):
        f = px.iloc[start-formation:start]
        t = px.iloc[start:end]
        if len(f) < formation or len(t) == 0:
            continue

        la = np.log(f["close_a"])
        lb = np.log(f["close_b"])

        # Gatev normalization: both series start at the same origin.
        na = la - la.iloc[0]
        nb = lb - lb.iloc[0]
        fspread = na - nb

        ssd = float(np.square(fspread).sum())
        mu = float(fspread.tail(z_window).mean())
        sd = safe_std(fspread.tail(z_window))
        hl = half_life_from_series(fspread)

        # Continue the same normalized paths into the trading block.
        ta = np.log(t["close_a"]) - la.iloc[0]
        tb = np.log(t["close_b"]) - lb.iloc[0]
        tspread = ta - tb
        z = (tspread - mu) / sd if np.isfinite(sd) and sd > 0 else pd.Series(np.nan, index=t.index)

        idx = t.index
        out.loc[idx, "alpha"] = 0.0
        out.loc[idx, "beta"] = 1.0
        out.loc[idx, "spread"] = tspread.values
        out.loc[idx, "zscore"] = np.asarray(z)
        out.loc[idx, "half_life"] = hl
        out.loc[idx, "pair_valid"] = np.isfinite(ssd) and np.isfinite(sd) and sd > 0
        out.loc[idx, "block_start"] = idx[0]
        out.loc[idx, "selection_score"] = ssd

    return out


# ============================================================
# MODEL 2: DYNAMIC KALMAN HEDGE RATIO
# ============================================================

def _ols_state(y, x):
    X = np.column_stack([np.ones(len(x)), x])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ coef
    rvar = float(np.var(resid, ddof=2)) if len(resid) > 2 else 1e-4
    return coef.astype(float), max(rvar, 1e-8)


def build_kalman_model(
    pair,
    formation=252,
    block=21,
    q_scale=1e-5,
    min_return_corr=0.20,
):
    """
    State-space relationship:
        log(A_t) = alpha_t + beta_t log(B_t) + epsilon_t

    alpha_t and beta_t follow random walks.  Each trading day's signal uses
    the Kalman innovation after sequentially processing data through that
    close. The trade is still executed next day by run_execution.

    selection_score = formation innovation variance proxy (lower is better)
    after requiring economically related formation returns.
    """
    px = pair[["close_a", "close_b"]].dropna()
    out = base_model(pair.index, pair)

    for start, end in rolling_blocks(len(px), formation, block):
        f = px.iloc[start-formation:start]
        t = px.iloc[start:end]
        if len(f) < formation or len(t) == 0:
            continue

        ya = np.log(f["close_a"]).values
        xb = np.log(f["close_b"]).values
        corr = float(np.corrcoef(np.diff(ya), np.diff(xb))[0, 1]) if len(f) > 2 else np.nan
        if not np.isfinite(corr) or corr < min_return_corr:
            continue

        state, rvar = _ols_state(ya, xb)
        P = np.eye(2) * 0.01
        Q = np.eye(2) * float(q_scale)
        R = max(rvar, 1e-8)

        # Warm the filter through the formation sample.
        innovations = []
        innov_vars = []
        for yv, xv in zip(ya, xb):
            P = P + Q
            H = np.array([1.0, xv])
            pred = float(H @ state)
            e = float(yv - pred)
            S = float(H @ P @ H.T + R)
            K = (P @ H) / S
            state = state + K * e
            P = P - np.outer(K, H) @ P
            innovations.append(e)
            innov_vars.append(S)

        formation_score = float(np.nanmean(np.square(innovations[-60:])))
        hist_innov = list(innovations[-60:])

        block_spreads = []
        block_z = []
        block_alpha = []
        block_beta = []

        for _, row in t.iterrows():
            yv = float(np.log(row["close_a"]))
            xv = float(np.log(row["close_b"]))

            P = P + Q
            H = np.array([1.0, xv])
            pred = float(H @ state)
            e = float(yv - pred)
            S = float(H @ P @ H.T + R)

            # Innovation z-score is available at today's close.
            z = e / np.sqrt(S) if S > 0 else np.nan

            K = (P @ H) / S
            state = state + K * e
            P = P - np.outer(K, H) @ P

            hist_innov.append(e)
            hist_innov = hist_innov[-60:]
            block_spreads.append(e)
            block_z.append(z)
            block_alpha.append(float(state[0]))
            block_beta.append(float(state[1]))

        idx = t.index
        out.loc[idx, "alpha"] = block_alpha
        out.loc[idx, "beta"] = block_beta
        out.loc[idx, "spread"] = block_spreads
        out.loc[idx, "zscore"] = block_z
        out.loc[idx, "half_life"] = half_life_from_series(innovations[-120:])
        out.loc[idx, "pair_valid"] = True
        out.loc[idx, "block_start"] = idx[0]
        out.loc[idx, "selection_score"] = formation_score

    return out


# ============================================================
# MODEL 3: FACTOR / RESIDUAL RELATIVE VALUE
# ============================================================

def _ols_factor(ret, factor):
    df = pd.concat([ret.rename("r"), factor.rename("f")], axis=1).dropna()
    if len(df) < 30:
        return np.nan, np.nan, pd.Series(dtype=float)
    X = np.column_stack([np.ones(len(df)), df["f"].values])
    coef = np.linalg.lstsq(X, df["r"].values, rcond=None)[0]
    resid = pd.Series(df["r"].values - X @ coef, index=df.index)
    return float(coef[0]), float(coef[1]), resid


def build_factor_residual_model(
    pair,
    benchmark_close,
    formation=252,
    block=21,
    z_window=60,
    max_half_life=60,
):
    """
    Remove the peer-group common return factor from each stock separately:

        r_A = a_A + b_A F + eps_A
        r_B = a_B + b_B F + eps_B

    The relative-value spread is cumulative (eps_A - eps_B).

    This asks whether the two stocks diverge beyond movements explained by
    their common peer-group factor.

    selection_score = formation residual-spread half-life (lower is better).
    """
    prices = pd.concat(
        [
            pair["close_a"],
            pair["close_b"],
            benchmark_close.rename("factor_close"),
        ],
        axis=1,
    ).dropna()

    out = base_model(pair.index, pair)

    for start, end in rolling_blocks(len(prices), formation, block):
        f = prices.iloc[start-formation:start]
        t = prices.iloc[start:end]
        if len(f) < formation or len(t) == 0:
            continue

        frets = np.log(f).diff().dropna()
        aA, bA, resA = _ols_factor(frets["close_a"], frets["factor_close"])
        aB, bB, resB = _ols_factor(frets["close_b"], frets["factor_close"])
        if not all(np.isfinite(v) for v in [aA, bA, aB, bB]):
            continue

        common_idx = resA.index.intersection(resB.index)
        diff_res = resA.loc[common_idx] - resB.loc[common_idx]
        fspread = diff_res.cumsum()
        hl = half_life_from_series(fspread)

        # Require actual residual mean reversion, but NOT price cointegration.
        valid = np.isfinite(hl) and 0 < hl <= float(max_half_life)
        if not valid:
            continue

        mu = float(fspread.tail(z_window).mean())
        sd = safe_std(fspread.tail(z_window))
        if not np.isfinite(sd) or sd <= 0:
            continue

        # Apply formation-estimated factor betas to subsequent returns.
        all_block = pd.concat([f.tail(1), t], axis=0)
        tr = np.log(all_block).diff().iloc[1:]
        epsA = tr["close_a"] - (aA + bA * tr["factor_close"])
        epsB = tr["close_b"] - (aB + bB * tr["factor_close"])
        incremental = epsA - epsB

        # Continue residual spread from formation endpoint.
        tspread = fspread.iloc[-1] + incremental.cumsum()
        z = (tspread - mu) / sd

        idx = t.index
        out.loc[idx, "alpha"] = 0.0
        out.loc[idx, "beta"] = 1.0
        out.loc[idx, "spread"] = tspread.values
        out.loc[idx, "zscore"] = z.values
        out.loc[idx, "half_life"] = hl
        out.loc[idx, "pair_valid"] = True
        out.loc[idx, "block_start"] = idx[0]
        out.loc[idx, "selection_score"] = hl

    return out


# ============================================================
# CROSS-SECTIONAL FORMATION-TIME SELECTION
# ============================================================

def apply_top_k_per_block(models, top_k):
    """
    Within one peer group and one signal family, retain the top K pair models
    per trading block according to that model's formation-period score.

    Lower score is better for all three models.
    """
    if not models:
        return models

    rows = []
    for name, model in models.items():
        tested = model.loc[model["block_start"].notna()].copy()
        if tested.empty:
            continue
        snap = (
            tested.groupby("block_start", as_index=False)
            .first()[["block_start", "selection_score", "pair_valid"]]
        )
        snap["pair"] = name
        rows.append(snap)

    if not rows:
        return models

    table = pd.concat(rows, ignore_index=True)
    table = table[
        table["pair_valid"].fillna(False)
        & np.isfinite(table["selection_score"])
    ].copy()

    keep = set()
    for block_start, grp in table.groupby("block_start"):
        chosen = grp.sort_values(
            ["selection_score", "pair"],
            ascending=[True, True],
        ).head(int(top_k))
        for _, row in chosen.iterrows():
            keep.add((str(row["pair"]), pd.Timestamp(block_start)))

    result = {}
    for name, model in models.items():
        m = model.copy()
        selected = pd.Series(False, index=m.index)
        for dt in pd.to_datetime(m["block_start"].dropna().unique()):
            if (name, pd.Timestamp(dt)) in keep:
                selected |= pd.to_datetime(m["block_start"]).eq(pd.Timestamp(dt))
        m["pair_valid"] = m["pair_valid"].fillna(False).astype(bool) & selected
        result[name] = m

    return result


# ============================================================
# EXECUTION
# ============================================================

def execute_model(
    model,
    pair_name,
    industry,
    capital,
    z_entry,
    z_exit,
    z_stop,
    transaction_cost_bps,
    annual_borrow_bps,
):
    daily, ledger = run_execution(
        model,
        capital=float(capital),
        gross_exposure=1.0,
        sizing_mode="Hedge-ratio weighted",
        transaction_cost_bps=float(transaction_cost_bps),
        annual_borrow_bps=float(annual_borrow_bps),
        z_entry=float(z_entry),
        z_exit=float(z_exit),
        z_stop=float(z_stop),
        z_entry_max=None,
        recent_equilibrium_max_days=None,
    )

    strategy = pd.DataFrame(index=daily.index)
    strategy["return"] = daily["net_return"]
    strategy["active"] = (
        (daily["position"] != 0)
        | (daily["net_return"].abs() > 0)
    ).astype(float)
    strategy["gross_exposure"] = daily["gross_exposure"]
    strategy["net_exposure"] = daily["net_exposure"]
    strategy["pair"] = pair_name
    strategy["industry"] = industry

    if ledger is not None and not ledger.empty:
        ledger = ledger.copy()
        ledger["pair"] = pair_name
        ledger["industry"] = industry

    return strategy, ledger


# ============================================================
# REPORTING
# ============================================================

def yearly_stats(r):
    rows = []
    r = r.dropna()
    for year, x in r.groupby(r.index.year):
        st = portfolio_statistics(x)
        rows.append({
            "year": int(year),
            "annual_return": st.get("annual_return"),
            "annual_volatility": st.get("annual_volatility"),
            "sharpe": st.get("sharpe"),
            "sortino": st.get("sortino"),
            "max_drawdown": st.get("max_drawdown"),
        })
    return pd.DataFrame(rows)


def portfolio_summary(model_name, allocation, combined, ledger):
    p = combined["portfolio"]
    st = portfolio_statistics(p["return"])
    return {
        "signal_model": model_name,
        "allocation": allocation,
        "annual_return": st.get("annual_return"),
        "annual_volatility": st.get("annual_volatility"),
        "sharpe": st.get("sharpe"),
        "sortino": st.get("sortino"),
        "max_drawdown": st.get("max_drawdown"),
        "calmar": st.get("calmar"),
        "trades": 0 if ledger is None else len(ledger),
        "unique_pairs_traded": 0 if ledger is None or ledger.empty else ledger["pair"].nunique(),
        "average_active_pairs": float(p["active_pairs"].mean()),
        "maximum_active_pairs": int(p["active_pairs"].max()),
        "average_capital_deployed": float(p["capital_deployed"].mean()),
        "maximum_capital_deployed": float(p["capital_deployed"].max()),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-09-01")
    parser.add_argument("--formation", type=int, default=252)
    parser.add_argument("--block", type=int, default=21)
    parser.add_argument("--z-window", type=int, default=60)
    parser.add_argument("--top-k-per-group", type=int, default=5)
    parser.add_argument("--z-entry", type=float, default=2.0)
    parser.add_argument("--z-exit", type=float, default=0.5)
    parser.add_argument("--z-stop", type=float, default=3.5)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--annual-borrow-bps", type=float, default=50.0)
    parser.add_argument("--capital-per-pair", type=float, default=100000.0)
    parser.add_argument("--max-pair-weight", type=float, default=0.25)
    parser.add_argument("--portfolio-gross", type=float, default=1.0)
    parser.add_argument("--factor-max-half-life", type=float, default=60.0)
    parser.add_argument("--output-dir", default="master_noncointegration_output")
    args = parser.parse_args()

    start_date = pd.Timestamp(args.start)
    end_date = pd.Timestamp(args.end)

    signal_names = ["Distance", "Kalman", "FactorResidual"]
    allocations = [
        "Equal-weight active pairs",
        "Inverse-volatility",
        "Constrained minimum-variance",
    ]

    strategies = {name: {} for name in signal_names}
    ledgers = {name: [] for name in signal_names}
    diagnostics = []
    failures = []

    industries = get_industries()

    print("\n==============================================================")
    print(" MASTER NON-COINTEGRATION PAIRS EXPERIMENT")
    print("==============================================================")
    print(f"Period: {args.start} to {args.end}")
    print(f"Formation: {args.formation} days | Trading block: {args.block} days")
    print(f"Top pairs per peer group/block: {args.top_k_per_group}")
    print("Signals: Distance | Kalman | Factor/Residual")
    print("Allocations: Equal | Inverse-vol | Constrained MinVar")
    print("Cointegration: NOT USED")
    print("==============================================================\n")

    for gi, industry in enumerate(industries, 1):
        print(f"[{gi}/{len(industries)}] {industry}")

        try:
            opens, closes = download_industry(industry, start_date, end_date)
            raw_pairs = generate_industry_pairs(industry)
        except Exception as exc:
            failures.append({"industry": industry, "pair": "", "model": "download", "reason": str(exc)})
            continue

        # Peer-group factor: equal-weight log-return index reconstructed as a
        # synthetic close series. It uses only stocks in this peer group.
        valid_close = closes.dropna(axis=1, how="all")
        peer_ret = np.log(valid_close).diff().mean(axis=1, skipna=True).fillna(0.0)
        peer_factor_close = np.exp(peer_ret.cumsum())
        peer_factor_close.name = "factor_close"

        group_models = {name: {} for name in signal_names}

        for ticker_a, ticker_b in raw_pairs:
            pair_name = f"{ticker_a} / {ticker_b}"
            try:
                pair = prepare_pair_data(opens, closes, ticker_a, ticker_b)
                if len(pair) < args.formation + args.block:
                    continue

                dm = build_distance_model(
                    pair,
                    formation=args.formation,
                    block=args.block,
                    z_window=args.z_window,
                )
                km = build_kalman_model(
                    pair,
                    formation=args.formation,
                    block=args.block,
                )
                fm = build_factor_residual_model(
                    pair,
                    peer_factor_close,
                    formation=args.formation,
                    block=args.block,
                    z_window=args.z_window,
                    max_half_life=args.factor_max_half_life,
                )

                for model_name, model in [
                    ("Distance", dm),
                    ("Kalman", km),
                    ("FactorResidual", fm),
                ]:
                    if model["pair_valid"].fillna(False).any():
                        group_models[model_name][pair_name] = model

            except Exception as exc:
                failures.append({
                    "industry": industry,
                    "pair": pair_name,
                    "model": "build",
                    "reason": str(exc),
                })

        # Formation-time cross-sectional selection is done separately for
        # each model so no model gets to use another model's future outcome.
        for model_name in signal_names:
            selected_models = apply_top_k_per_block(
                group_models[model_name],
                args.top_k_per_group,
            )

            for pair_name, model in selected_models.items():
                if not model["pair_valid"].fillna(False).any():
                    continue
                try:
                    strategy, ledger = execute_model(
                        model=model,
                        pair_name=pair_name,
                        industry=industry,
                        capital=args.capital_per_pair,
                        z_entry=args.z_entry,
                        z_exit=args.z_exit,
                        z_stop=args.z_stop,
                        transaction_cost_bps=args.transaction_cost_bps,
                        annual_borrow_bps=args.annual_borrow_bps,
                    )

                    key = f"{industry} | {pair_name}"
                    strategies[model_name][key] = strategy

                    if ledger is not None and not ledger.empty:
                        ledger = ledger.copy()
                        ledger["signal_model"] = model_name
                        ledgers[model_name].append(ledger)

                    diagnostics.append({
                        "industry": industry,
                        "pair": pair_name,
                        "signal_model": model_name,
                        "valid_days": int(model["pair_valid"].fillna(False).sum()),
                        "mean_selection_score": float(
                            pd.to_numeric(
                                model.loc[model["pair_valid"], "selection_score"],
                                errors="coerce",
                            ).mean()
                        ),
                    })

                except Exception as exc:
                    failures.append({
                        "industry": industry,
                        "pair": pair_name,
                        "model": model_name,
                        "reason": f"execution: {exc}",
                    })

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(diagnostics).to_csv(out / "signal_diagnostics.csv", index=False)
    pd.DataFrame(failures).to_csv(out / "failures.csv", index=False)

    summaries = []
    yearly_frames = []

    for model_name in signal_names:
        if not strategies[model_name]:
            print(f"\nWARNING: {model_name} produced no executable strategies.")
            continue

        model_ledger = (
            pd.concat(ledgers[model_name], ignore_index=True)
            if ledgers[model_name]
            else pd.DataFrame()
        )
        model_ledger.to_csv(
            out / f"{model_name.lower()}_trade_ledger.csv",
            index=False,
        )

        for allocation in allocations:
            print(f"\nCombining {model_name} / {allocation} ...")
            try:
                combined = combine_pair_strategies(
                    strategies[model_name],
                    allocation_method=allocation,
                    max_pair_weight=args.max_pair_weight,
                    portfolio_gross=args.portfolio_gross,
                )
            except ValueError as exc:
                if allocation == "Constrained minimum-variance":
                    raise RuntimeError(
                        "Your analytics/portfolio_engine.py does not appear to "
                        "contain the Constrained minimum-variance allocator. "
                        "Replace it with the optimized portfolio_engine.py from "
                        "the previous step, then rerun."
                    ) from exc
                raise

            tag = (
                model_name.lower()
                + "_"
                + allocation.lower()
                    .replace(" ", "_")
                    .replace("-", "_")
            )

            combined["portfolio"].to_csv(out / f"{tag}_portfolio_daily.csv")
            combined["weights"].to_csv(out / f"{tag}_weights.csv")

            summary = portfolio_summary(
                model_name,
                allocation,
                combined,
                model_ledger,
            )
            summaries.append(summary)

            ys = yearly_stats(combined["portfolio"]["return"])
            ys["signal_model"] = model_name
            ys["allocation"] = allocation
            yearly_frames.append(ys)

    summary_df = pd.DataFrame(summaries)
    yearly_df = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()

    summary_df.to_csv(out / "MASTER_COMPARISON.csv", index=False)
    yearly_df.to_csv(out / "YEAR_BY_YEAR_RESULTS.csv", index=False)

    print("\n\n================ MASTER COMPARISON ================\n")
    if summary_df.empty:
        print("No portfolio results were produced. Inspect failures.csv.")
    else:
        display_cols = [
            "signal_model",
            "allocation",
            "annual_return",
            "annual_volatility",
            "sharpe",
            "sortino",
            "max_drawdown",
            "trades",
            "unique_pairs_traded",
        ]
        print(
            summary_df[display_cols]
            .sort_values("sharpe", ascending=False)
            .to_string(index=False)
        )

    print(f"\nOutputs saved to: {out.resolve()}")
    print("\nPrimary files:")
    print("  MASTER_COMPARISON.csv")
    print("  YEAR_BY_YEAR_RESULTS.csv")
    print("  signal_diagnostics.csv")
    print("  distance_trade_ledger.csv")
    print("  kalman_trade_ledger.csv")
    print("  factorresidual_trade_ledger.csv")
    print("\nDo not tune parameters from this table before reviewing stability by year.")
    print("===================================================\n")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
