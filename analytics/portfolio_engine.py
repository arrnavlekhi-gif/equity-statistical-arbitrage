import numpy as np

import pandas as pd



from analytics.pair_scanner import (

    get_industries,

    generate_industry_pairs,

    download_industry,

    analyse_pair,

    prepare_pair_data,

    prescreen_industry_pairs,

    apply_persistence_filter,

)



from models.walkforward import build_walkforward

from strategy.execution import run_execution





# ============================================================

# CANDIDATE SELECTION

# ============================================================



def select_industry_candidate(

    scanner,

    min_valid_blocks=3,

    min_regimes=2,

    min_trades=3,

):



    if scanner is None or scanner.empty:

        return None



    required = [

        "valid_blocks",

        "valid_fraction",

        "validity_regimes",

        "valid_median_pvalue",

        "trades",

    ]



    for col in required:

        if col not in scanner.columns:

            return None



    eligible = scanner[

        (scanner["valid_blocks"].fillna(0) >= min_valid_blocks)

        &

        (scanner["validity_regimes"].fillna(0) >= min_regimes)

        &

        (scanner["trades"].fillna(0) >= min_trades)

    ].copy()



    if eligible.empty:

        return None



    # --------------------------------------------------------

    # IMPORTANT:

    #

    # Candidate selection does NOT use Sharpe or annual return.

    #

    # Priority:

    # 1. Greater persistence of statistical validity

    # 2. More distinct regimes

    # 3. More valid blocks

    # 4. Lower conditional cointegration p-value

    # --------------------------------------------------------



    eligible = eligible.sort_values(

        by=[

            "valid_fraction",

            "validity_regimes",

            "valid_blocks",

            "valid_median_pvalue",

        ],

        ascending=[

            False,

            False,

            False,

            True,

        ],

        na_position="last",

    )



    return eligible.iloc[0]





# ============================================================

# SCAN ONE INDUSTRY

# ============================================================



def scan_industry_for_portfolio(

    industry, start, end, formation_window=252, trading_window=21, z_window=60,

    coint_threshold=0.05, max_half_life=60, z_entry=2.0, z_exit=0.5, z_stop=3.5,

    transaction_cost_bps=5.0, annual_borrow_bps=50.0, min_valid_blocks=0,

    min_regimes=0, min_trades=0, prescreen_max_pairs=None,

    prescreen_min_return_corr=None, prescreen_min_log_price_corr=None,

    progress_callback=None, industry_index=None, industry_total=None,

    persistence_lookback=6, persistence_min_passes=3,

):

    """

    Test EVERY within-peer pair with the walk-forward model.



    There is deliberately no full-sample correlation or cointegration pre-screen.

    For each 21-day trading block, build_walkforward estimates the relationship

    using only the preceding formation window. A pair is eligible in that block

    only when Engle-Granger p < coint_threshold and 0 < half-life <= max_half_life.

    """

    opens, closes = download_industry(industry, start, end)

    raw_pairs = generate_industry_pairs(industry)



    if progress_callback is not None:

        progress_callback({

            "stage": "rolling_screen", "industry": industry,

            "industry_index": industry_index, "industry_total": industry_total,

            "raw_pair_total": len(raw_pairs),

            "fraction": ((industry_index or 1) - 1) / max(industry_total or 1, 1),

        })



    rows = []

    block_frames = []

    for pair_index, (ticker_a, ticker_b) in enumerate(raw_pairs, start=1):

        if progress_callback is not None:

            base = ((industry_index or 1) - 1) / max(industry_total or 1, 1)

            within = pair_index / max(len(raw_pairs), 1) / max(industry_total or 1, 1)

            progress_callback({

                "stage": "scanning", "industry": industry,

                "industry_index": industry_index, "industry_total": industry_total,

                "pair": f"{ticker_a}/{ticker_b}", "pair_index": pair_index,

                "pair_total": len(raw_pairs), "raw_pair_total": len(raw_pairs),

                "prescreen_survivors": len(raw_pairs), "fraction": base + within,

            })

        try:

            result = analyse_pair(

                ticker_a=ticker_a, ticker_b=ticker_b, industry=industry,

                opens=opens, closes=closes, formation_window=formation_window,

                trading_window=trading_window, z_window=z_window,

                coint_threshold=coint_threshold, max_half_life=max_half_life,

                z_entry=z_entry, z_exit=z_exit, z_stop=z_stop, gross_exposure=1.0,

                transaction_cost_bps=transaction_cost_bps,

                annual_borrow_bps=annual_borrow_bps,

                persistence_lookback=persistence_lookback,

                persistence_min_passes=persistence_min_passes,

            )

            if result is not None:

                result = dict(result)

                block_diag = result.pop("_block_diagnostics", None)

                if isinstance(block_diag, pd.DataFrame) and not block_diag.empty:

                    block_frames.append(block_diag)

                rows.append(result)

        except Exception:

            continue



    scanner = pd.DataFrame(rows)

    diagnostics = {

        "industry": industry,

        "raw_pairs": len(raw_pairs),

        "prescreen_survivors": len(raw_pairs),

        "full_models_tested": len(rows),

    }

    # Kept for API compatibility with the previous version.

    prescreen = pd.DataFrame({

        "pair": [f"{a} / {b}" for a, b in raw_pairs],

        "screen": "rolling cointegration eligibility",

    })

    pair_block_diagnostics = (

        pd.concat(block_frames, ignore_index=True)

        if block_frames else pd.DataFrame()

    )

    return None, scanner, opens, closes, prescreen, diagnostics, pair_block_diagnostics







# ============================================================

# TRADE-PATH DIAGNOSTICS

# ============================================================




# ============================================================
# PRE-ENTRY Z-PATH DIAGNOSTICS
# ============================================================

def _attach_pre_entry_z_features(ledger, model):
    """
    Attach Z-path features known at the entry-signal close.
    Diagnostic only: this does not change any trading decision.
    """
    if ledger is None or ledger.empty:
        return ledger

    out = ledger.copy()
    numeric_cols = [
        "z_at_signal",
        "z_lag_1", "z_lag_5", "z_lag_10",
        "z_change_1", "z_change_5", "z_change_10",
        "abs_z_change_1", "abs_z_change_5", "abs_z_change_10",
        "max_abs_z_last_5", "max_abs_z_last_10",
        "min_abs_z_last_5", "min_abs_z_last_10",
        "days_since_abs_z_le_1", "days_since_abs_z_le_1_5",
        "z_cross_speed_from_1", "z_cross_speed_from_1_5",
    ]
    for col in numeric_cols:
        if col not in out.columns:
            out[col] = np.nan

    if model is None or model.empty or "zscore" not in model.columns:
        return out

    try:
        z = pd.to_numeric(model["zscore"], errors="coerce").copy()
        idx = pd.DatetimeIndex(z.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        z.index = idx
        z = z[~z.index.duplicated(keep="last")].sort_index()

        def _days_since_threshold(pos, threshold):
            if pos <= 0:
                return np.nan
            prior = z.iloc[:pos].abs()
            hits = np.flatnonzero((prior <= threshold).fillna(False).to_numpy())
            if len(hits) == 0:
                return np.nan
            return float(pos - int(hits[-1]))

        for i, trade in out.iterrows():
            signal_raw = trade.get("entry_signal_date", trade.get("entry_date"))
            if pd.isna(signal_raw):
                continue

            signal_date = pd.Timestamp(signal_raw)
            if signal_date.tzinfo is not None:
                signal_date = signal_date.tz_localize(None)

            if signal_date in z.index:
                loc = z.index.get_loc(signal_date)
                pos = int(loc if np.isscalar(loc) else np.flatnonzero(loc)[-1])
            else:
                pos = int(z.index.searchsorted(signal_date, side="right") - 1)
                if pos < 0:
                    continue

            z_t = z.iloc[pos]
            if not np.isfinite(z_t):
                continue
            out.at[i, "z_at_signal"] = float(z_t)

            for lag in (1, 5, 10):
                if pos - lag < 0:
                    continue
                z_lag = z.iloc[pos - lag]
                if not np.isfinite(z_lag):
                    continue
                out.at[i, f"z_lag_{lag}"] = float(z_lag)
                out.at[i, f"z_change_{lag}"] = float(z_t - z_lag)
                out.at[i, f"abs_z_change_{lag}"] = float(abs(z_t) - abs(z_lag))

            for window in (5, 10):
                start = max(0, pos - window + 1)
                hist = z.iloc[start:pos + 1].dropna().abs()
                if len(hist):
                    out.at[i, f"max_abs_z_last_{window}"] = float(hist.max())
                    out.at[i, f"min_abs_z_last_{window}"] = float(hist.min())

            d1 = _days_since_threshold(pos, 1.0)
            d15 = _days_since_threshold(pos, 1.5)
            out.at[i, "days_since_abs_z_le_1"] = d1
            out.at[i, "days_since_abs_z_le_1_5"] = d15
            out.at[i, "z_cross_speed_from_1"] = d1
            out.at[i, "z_cross_speed_from_1_5"] = d15

    except Exception:
        return out

    return out


def _trade_path_diagnostics(ledger, model):

    """Attach ex-post signal-path Z-space MFE/MAE diagnostics.



    Diagnostics are fail-safe: they must never prevent an otherwise valid

    strategy from entering the portfolio.

    """

    if ledger is None or ledger.empty:

        return ledger



    out = ledger.copy()

    for col in [

        "best_abs_z", "worst_abs_z", "best_z", "worst_z",

        "mfe_z", "mae_z", "z_mfe", "z_mae",

    ]:

        if col not in out.columns:

            out[col] = np.nan



    for col in [

        "reached_abs_z_1",

        "ever_moved_toward_equilibrium",

    ]:

        if col not in out.columns:

            out[col] = False



    if "zscore" not in model.columns:

        return out



    try:

        zseries = pd.to_numeric(model["zscore"], errors="coerce").copy()

        zidx = pd.DatetimeIndex(zseries.index)

        if zidx.tz is not None:

            zidx = zidx.tz_localize(None)

        zseries.index = zidx



        for i, trade in out.iterrows():

            start_raw = trade.get("entry_signal_date", trade.get("entry_date"))

            end_raw = trade.get("exit_signal_date", trade.get("exit_date"))



            if pd.isna(start_raw):

                continue



            start_date = pd.Timestamp(start_raw)

            if start_date.tzinfo is not None:

                start_date = start_date.tz_localize(None)



            if pd.isna(end_raw):

                end_date = zseries.index.max()

            else:

                end_date = pd.Timestamp(end_raw)

                if end_date.tzinfo is not None:

                    end_date = end_date.tz_localize(None)



            path = zseries.loc[

                (zseries.index >= start_date) & (zseries.index <= end_date)

            ].dropna()

            if path.empty:

                continue



            entry_z = pd.to_numeric(

                pd.Series([trade.get("entry_z", np.nan)]),

                errors="coerce",

            ).iloc[0]

            if not np.isfinite(entry_z):

                entry_z = float(path.iloc[0])



            sign = np.sign(float(entry_z))

            if sign == 0:

                continue



            progress = sign * (float(entry_z) - path)

            best_idx = progress.idxmax()

            worst_idx = progress.idxmin()



            out.at[i, "best_z"] = float(path.loc[best_idx])

            out.at[i, "worst_z"] = float(path.loc[worst_idx])

            out.at[i, "best_abs_z"] = float(path.abs().min())

            out.at[i, "worst_abs_z"] = float(path.abs().max())

            out.at[i, "mfe_z"] = max(0.0, float(progress.max()))

            out.at[i, "mae_z"] = max(0.0, float(-progress.min()))



            # Backward-compatible aliases used by the existing aggregation/UI.

            out.at[i, "z_mfe"] = out.at[i, "mfe_z"]

            out.at[i, "z_mae"] = out.at[i, "mae_z"]



            out.at[i, "reached_abs_z_1"] = bool((path.abs() <= 1.0).any())

            out.at[i, "ever_moved_toward_equilibrium"] = bool(progress.max() > 0)



    except Exception:

        return out



    if "exit_reason" in out.columns:

        mr = out["exit_reason"].fillna("").astype(str).eq("MEAN_REVERSION")

        out.loc[mr, "reached_abs_z_1"] = True



    return out





# ============================================================

# BUILD DAILY STRATEGY RETURN SERIES FOR SELECTED PAIR

# ============================================================



def build_selected_pair_strategy(

    candidate,

    opens,

    closes,

    formation_window=252,

    trading_window=21,

    z_window=60,

    coint_threshold=0.05,

    max_half_life=60,

    z_entry=2.0,

    z_exit=0.5,

    z_stop=3.5,

    z_entry_max=None,

    recent_equilibrium_max_days=None,

    transaction_cost_bps=5.0,

    annual_borrow_bps=50.0,

    persistence_lookback=6,

    persistence_min_passes=3,

):



    ticker_a = candidate["ticker_a"]

    ticker_b = candidate["ticker_b"]



    pair = prepare_pair_data(

        opens,

        closes,

        ticker_a,

        ticker_b,

    )



    model = build_walkforward(

        pair["close_a"],

        pair["close_b"],

        formation_window=int(formation_window),

        trading_window=int(trading_window),

        z_window=int(z_window),

        coint_threshold=float(coint_threshold),

        max_half_life=float(max_half_life),

    )



    model = apply_persistence_filter(

        model,

        persistence_lookback=int(persistence_lookback),

        persistence_min_passes=int(persistence_min_passes),

    )



    model = model.join(

        pair[

            [

                "open_a",

                "open_b",

                "close_a",

                "close_b",

            ]

        ]

    )



    model = model.dropna(

        subset=[

            "open_a",

            "open_b",

            "close_a",

            "close_b",

        ]

    )



    daily, ledger = run_execution(

        model,

        capital=100000,

        gross_exposure=1.0,

        sizing_mode="Hedge-ratio weighted",

        transaction_cost_bps=float(transaction_cost_bps),

        annual_borrow_bps=float(annual_borrow_bps),

        z_entry=float(z_entry),

        z_exit=float(z_exit),

        z_stop=float(z_stop),

        z_entry_max=(None if z_entry_max is None else float(z_entry_max)),

        recent_equilibrium_max_days=(None if recent_equilibrium_max_days is None else int(recent_equilibrium_max_days)),

    )



    ledger = _attach_pre_entry_z_features(

        ledger,

        model,

    )


    ledger = _trade_path_diagnostics(

        ledger,

        model,

    )



    if ledger is not None and not ledger.empty:

        ledger["pair"] = candidate["pair"]

        ledger["industry"] = candidate["industry"]



    strategy = pd.DataFrame(

        index=daily.index

    )



    strategy["return"] = (

        daily["net_return"]

    )



    strategy["active"] = (

        (daily["position"] != 0)

        | (daily["net_return"].abs() > 0)

    ).astype(float)



    strategy["gross_exposure"] = (

        daily["gross_exposure"]

    )



    strategy["net_exposure"] = (

        daily["net_exposure"]

    )



    strategy["pair"] = candidate["pair"]



    strategy["industry"] = candidate["industry"]



    return strategy, ledger





# ============================================================

# PORTFOLIO PERFORMANCE

# ============================================================



def portfolio_statistics(

    returns,

):



    r = returns.dropna()



    if len(r) == 0:



        return {

            "annual_return": np.nan,

            "annual_volatility": np.nan,

            "sharpe": np.nan,

            "sortino": np.nan,

            "max_drawdown": np.nan,

            "calmar": np.nan,

            "var_95": np.nan,

            "expected_shortfall_95": np.nan,

        }



    wealth = (

        1 + r

    ).cumprod()



    annual_return = (

        wealth.iloc[-1]

        ** (252 / len(r))

        - 1

    )



    annual_volatility = (

        r.std(ddof=1)

        * np.sqrt(252)

    )



    if (

        np.isfinite(annual_volatility)

        and annual_volatility > 0

    ):



        sharpe = (

            r.mean()

            / r.std(ddof=1)

            * np.sqrt(252)

        )



    else:



        sharpe = np.nan



    downside = r[

        r < 0

    ]



    downside_vol = (

        downside.std(ddof=1)

        * np.sqrt(252)

        if len(downside) > 1

        else np.nan

    )



    if (

        np.isfinite(downside_vol)

        and downside_vol > 0

    ):



        sortino = (

            r.mean() * 252

        ) / downside_vol



    else:



        sortino = np.nan



    drawdown = (

        wealth

        / wealth.cummax()

        - 1

    )



    max_drawdown = float(

        drawdown.min()

    )



    if (

        np.isfinite(max_drawdown)

        and max_drawdown < 0

    ):



        calmar = (

            annual_return

            / abs(max_drawdown)

        )



    else:



        calmar = np.nan



    var_95 = float(

        r.quantile(0.05)

    )



    tail = r[

        r <= var_95

    ]



    expected_shortfall_95 = (

        float(tail.mean())

        if len(tail)

        else np.nan

    )



    return {

        "annual_return":

            float(annual_return),



        "annual_volatility":

            float(annual_volatility),



        "sharpe":

            float(sharpe),



        "sortino":

            float(sortino),



        "max_drawdown":

            max_drawdown,



        "calmar":

            float(calmar),



        "var_95":

            var_95,



        "expected_shortfall_95":

            expected_shortfall_95,

    }





# ============================================================

# COMBINE STRATEGIES

# ============================================================



def combine_pair_strategies(

    strategies,

    allocation_method="Equal-weight active pairs",

    max_pair_weight=0.25,

    portfolio_gross=1.0,

):



    if not strategies:

        return None



    # --------------------------------------------------------

    # ALIGN RETURNS

    # --------------------------------------------------------



    returns = pd.concat(

        {

            name: data["return"]

            for name, data in strategies.items()

        },

        axis=1,

    ).fillna(0.0)



    active = pd.concat(

        {

            name: data["active"]

            for name, data in strategies.items()

        },

        axis=1,

    ).fillna(0.0)



    # --------------------------------------------------------

    # WEIGHTS

    # --------------------------------------------------------



    weights = pd.DataFrame(

        0.0,

        index=returns.index,

        columns=returns.columns,

    )



    if allocation_method == "Equal-weight active pairs":



        active_count = active.sum(

            axis=1

        )



        for idx in returns.index:



            n = active_count.loc[idx]



            if n <= 0:

                continue



            raw_weight = min(

                float(portfolio_gross) / n,

                float(max_pair_weight),

            )



            weights.loc[idx] = (

                active.loc[idx]

                * raw_weight

            )



    elif allocation_method == "Inverse-volatility":



        # Shift by one day: today's allocation can use only information
        # available through the previous close.

        rolling_vol = (

            returns

            .rolling(60, min_periods=20)

            .std()

            .shift(1)

            * np.sqrt(252)

        )



        for idx in returns.index:



            active_names = active.columns[

                active.loc[idx] > 0

            ]



            if len(active_names) == 0:

                continue



            vols = rolling_vol.loc[

                idx,

                active_names,

            ]



            vols = vols.replace(

                [np.inf, -np.inf],

                np.nan,

            ).dropna()



            vols = vols[

                vols > 0

            ]



            if len(vols) == 0:



                raw = pd.Series(

                    1.0 / len(active_names),

                    index=active_names,

                )



            else:



                inv = 1.0 / vols



                raw = (

                    inv / inv.sum()

                )



            # Match the feasible gross exposure used by equal-active:
            # with too few active pairs, the pair cap deliberately leaves
            # part of the portfolio in cash.

            target_gross = min(

                float(portfolio_gross),

                len(active_names) * float(max_pair_weight),

            )



            raw = raw * target_gross



            # Iteratively redistribute weight released by the pair cap.

            for _ in range(10):

                capped = raw.clip(upper=float(max_pair_weight))

                shortfall = target_gross - float(capped.sum())

                if shortfall <= 1e-12:

                    raw = capped

                    break

                room = (float(max_pair_weight) - capped).clip(lower=0.0)

                if float(room.sum()) <= 1e-12:

                    raw = capped

                    break

                raw = capped + shortfall * room / room.sum()



            weights.loc[

                idx,

                raw.index,

            ] = raw



    elif allocation_method == "Constrained minimum-variance":



        # Portfolio-level optimizer.  It deliberately does NOT estimate
        # expected pair returns.  It uses only the trailing covariance matrix
        # available before today's allocation, reducing the estimation error
        # that made cross-sectional return ranking unstable.

        try:

            from scipy.optimize import minimize

        except ImportError as exc:

            raise ImportError(

                "Constrained minimum-variance allocation requires scipy."

            ) from exc



        lookback = 60

        min_history = 20

        shrinkage = 0.50



        for pos, idx in enumerate(returns.index):



            active_names = list(

                active.columns[active.loc[idx] > 0]

            )



            n = len(active_names)



            if n == 0:

                continue



            target_gross = min(

                float(portfolio_gross),

                n * float(max_pair_weight),

            )



            # Strictly prior observations only: [pos-lookback, pos).

            hist = returns.iloc[

                max(0, pos - lookback):pos

            ][active_names].copy()



            usable = hist.shape[0] >= min_history



            if usable:

                cov = hist.cov().values.astype(float)

                usable = (

                    cov.shape == (n, n)

                    and np.isfinite(cov).all()

                )



            if usable:

                # Diagonal shrinkage makes the covariance estimate much more
                # stable when many pair strategies overlap or have sparse
                # histories.

                diag = np.diag(np.diag(cov))

                cov = (

                    (1.0 - shrinkage) * cov

                    + shrinkage * diag

                )



                # Small ridge for numerical stability.

                scale = float(np.trace(cov) / max(n, 1))

                ridge = max(scale * 1e-6, 1e-12)

                cov = cov + np.eye(n) * ridge



                x0 = np.repeat(target_gross / n, n)



                bounds = [

                    (0.0, float(max_pair_weight))

                    for _ in range(n)

                ]



                constraints = [

                    {

                        "type": "eq",

                        "fun": lambda w, tg=target_gross:

                            float(np.sum(w) - tg),

                    }

                ]



                result = minimize(

                    lambda w, c=cov:

                        float(w @ c @ w),

                    x0=x0,

                    method="SLSQP",

                    bounds=bounds,

                    constraints=constraints,

                    options={

                        "maxiter": 200,

                        "ftol": 1e-12,

                        "disp": False,

                    },

                )



                if result.success and np.isfinite(result.x).all():

                    raw = pd.Series(

                        result.x,

                        index=active_names,

                    )

                else:

                    usable = False



            if not usable:

                # Warm-up/failure fallback is equal-active, not a future-data
                # estimate.

                raw = pd.Series(

                    target_gross / n,

                    index=active_names,

                )



            weights.loc[

                idx,

                raw.index,

            ] = raw



    else:



        raise ValueError(

            f"Unknown allocation method: "

            f"{allocation_method}"

        )



    # --------------------------------------------------------

    # PORTFOLIO RETURN

    # --------------------------------------------------------



    portfolio_return = (

        returns

        * weights

    ).sum(

        axis=1

    )



    # --------------------------------------------------------

    # PORTFOLIO EQUITY

    # --------------------------------------------------------



    equity = (

        100000

        * (1 + portfolio_return)

        .cumprod()

    )



    # --------------------------------------------------------

    # DRAWDOWN

    # --------------------------------------------------------



    drawdown = (

        equity

        / equity.cummax()

        - 1

    )



    # --------------------------------------------------------

    # ACTIVE PAIRS

    # --------------------------------------------------------



    active_pairs = (

        active.sum(

            axis=1

        )

    )



    # --------------------------------------------------------

    # ACTUAL CAPITAL DEPLOYMENT

    # --------------------------------------------------------



    capital_deployed = (

        weights.sum(

            axis=1

        )

    )



    portfolio = pd.DataFrame(

        {

            "return":

                portfolio_return,



            "equity":

                equity,



            "drawdown":

                drawdown,



            "active_pairs":

                active_pairs,



            "capital_deployed":

                capital_deployed,

        }

    )



    return {

        "portfolio":

            portfolio,



        "pair_returns":

            returns,



        "pair_active":

            active,



        "weights":

            weights,

    }





# ============================================================

# CONTRIBUTION ANALYSIS

# ============================================================



def contribution_analysis(

    pair_returns,

    weights,

    selected,

    ledgers,

):



    contribution_returns = (

        pair_returns

        * weights

    )



    rows = []



    total_contribution = (

        contribution_returns

        .sum()

        .sum()

    )



    for pair in pair_returns.columns:



        pair_contribution = float(

            contribution_returns[

                pair

            ].sum()

        )



        candidate = selected[

            selected["pair"] == pair

        ]



        if len(candidate):



            industry = candidate.iloc[0][

                "industry"

            ]



        else:



            industry = ""



        ledger = ledgers.get(

            pair,

            pd.DataFrame(),

        )



        if len(ledger):



            trades = len(ledger)



            gross_pnl = float(

                ledger[

                    "gross_pnl"

                ].sum()

            )



            transaction_cost = float(

                ledger[

                    "transaction_cost"

                ].sum()

            )



            borrow_cost = float(

                ledger[

                    "borrow_cost"

                ].sum()

            )



            pair_net_pnl = float(

                ledger[

                    "net_pnl"

                ].sum()

            )



        else:



            trades = 0

            gross_pnl = 0.0

            transaction_cost = 0.0

            borrow_cost = 0.0

            pair_net_pnl = 0.0



        if (

            np.isfinite(total_contribution)

            and total_contribution != 0

        ):



            contribution_share = (

                pair_contribution

                / total_contribution

            )



        else:



            contribution_share = np.nan



        rows.append(

            {

                "industry":

                    industry,



                "pair":

                    pair,



                "trades":

                    trades,



                "gross_pair_pnl":

                    gross_pnl,



                "transaction_cost":

                    transaction_cost,



                "borrow_cost":

                    borrow_cost,



                "pair_net_pnl":

                    pair_net_pnl,



                "portfolio_return_contribution":

                    pair_contribution,



                "contribution_share":

                    contribution_share,



                "average_portfolio_weight":

                    float(

                        weights[

                            pair

                        ].mean()

                    ),



                "maximum_portfolio_weight":

                    float(

                        weights[

                            pair

                        ].max()

                    ),

            }

        )



    return pd.DataFrame(

        rows

    )







# ============================================================

# TRADE OUTCOME DIAGNOSTICS

# ============================================================



def _combine_ledgers(ledgers):

    frames = []

    for pair_name, ledger in ledgers.items():

        if ledger is None or ledger.empty:

            continue

        x = ledger.copy()

        if "pair" not in x.columns:

            x["pair"] = pair_name

        frames.append(x)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()





def trade_outcome_diagnostics(ledgers):

    """Aggregate completed-trade outcomes. Ex-post evaluation only."""

    trades = _combine_ledgers(ledgers)



    empty_outcomes = pd.DataFrame(columns=[

        "exit_reason", "trades", "trade_share", "win_rate",

        "avg_net_pnl", "median_net_pnl", "avg_holding_days",

        "avg_z_mfe", "avg_z_mae", "convergence_rate", "total_net_pnl",

    ])

    if trades.empty:

        return empty_outcomes, {}



    numeric_cols = [

        "net_pnl", "gross_pnl", "transaction_cost", "borrow_cost",

        "holding_days", "z_mfe", "z_mae", "best_abs_z", "worst_abs_z",

    ]

    for col in numeric_cols:

        if col not in trades.columns:

            trades[col] = np.nan

        trades[col] = pd.to_numeric(trades[col], errors="coerce")



    for col in ["net_pnl", "gross_pnl", "transaction_cost", "borrow_cost", "holding_days"]:

        trades[col] = trades[col].fillna(0.0)



    if "exit_reason" not in trades.columns:

        trades["exit_reason"] = "UNKNOWN"

    trades["exit_reason"] = trades["exit_reason"].fillna("UNKNOWN").astype(str)



    if "reached_abs_z_1" not in trades.columns:

        trades["reached_abs_z_1"] = False

    trades["reached_abs_z_1"] = trades["reached_abs_z_1"].fillna(False).astype(bool)



    total_trades = len(trades)

    rows = []

    for reason, g in trades.groupby("exit_reason", dropna=False):

        rows.append({

            "exit_reason": reason,

            "trades": int(len(g)),

            "trade_share": float(len(g) / total_trades),

            "win_rate": float((g["net_pnl"] > 0).mean()),

            "avg_net_pnl": float(g["net_pnl"].mean()),

            "median_net_pnl": float(g["net_pnl"].median()),

            "avg_holding_days": float(g["holding_days"].mean()),

            "avg_z_mfe": float(g["z_mfe"].mean()) if g["z_mfe"].notna().any() else np.nan,

            "avg_z_mae": float(g["z_mae"].mean()) if g["z_mae"].notna().any() else np.nan,

            "convergence_rate": float(g["reached_abs_z_1"].mean()),

            "total_net_pnl": float(g["net_pnl"].sum()),

        })



    outcomes = pd.DataFrame(rows).sort_values(

        "trades", ascending=False

    ).reset_index(drop=True)



    winners = trades.loc[trades["net_pnl"] > 0, "net_pnl"]

    losers = trades.loc[trades["net_pnl"] < 0, "net_pnl"]

    gross_profit = float(winners.sum()) if len(winners) else 0.0

    gross_loss_abs = float(abs(losers.sum())) if len(losers) else 0.0

    avg_winner = float(winners.mean()) if len(winners) else np.nan

    avg_loser = float(losers.mean()) if len(losers) else np.nan



    payoff_ratio = (

        avg_winner / abs(avg_loser)

        if np.isfinite(avg_winner) and np.isfinite(avg_loser) and avg_loser != 0

        else np.nan

    )

    profit_factor = gross_profit / gross_loss_abs if gross_loss_abs > 0 else np.nan



    economics = {

        "completed_trades": int(total_trades),

        "win_rate": float((trades["net_pnl"] > 0).mean()),

        "mean_reversion_rate": float((trades["exit_reason"] == "MEAN_REVERSION").mean()),

        "convergence_rate": float(trades["reached_abs_z_1"].mean()),

        "average_winner": avg_winner,

        "average_loser": avg_loser,

        "median_net_pnl": float(trades["net_pnl"].median()),

        "payoff_ratio": float(payoff_ratio) if np.isfinite(payoff_ratio) else np.nan,

        "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else np.nan,

        "gross_pnl": float(trades["gross_pnl"].sum()),

        "transaction_cost": float(trades["transaction_cost"].sum()),

        "borrow_cost": float(trades["borrow_cost"].sum()),

        "net_pnl": float(trades["net_pnl"].sum()),

        "average_holding_days": float(trades["holding_days"].mean()),

        "average_z_mfe": float(trades["z_mfe"].mean()) if trades["z_mfe"].notna().any() else np.nan,

        "average_z_mae": float(trades["z_mae"].mean()) if trades["z_mae"].notna().any() else np.nan,

    }

    return outcomes, economics





# ============================================================

# INDUSTRY MEAN-REVERSION ANALYSIS

# ============================================================



def industry_mean_reversion_analysis(ledgers):

    """Build an ex-post industry diagnostic table."""

    trades = _combine_ledgers(ledgers)

    if trades.empty or "industry" not in trades.columns:

        return pd.DataFrame()



    for col in [

        "net_pnl", "gross_pnl", "transaction_cost", "borrow_cost",

        "z_mfe", "z_mae",

    ]:

        if col not in trades.columns:

            trades[col] = np.nan

        trades[col] = pd.to_numeric(trades[col], errors="coerce")



    if "exit_reason" not in trades.columns:

        trades["exit_reason"] = "UNKNOWN"

    trades["exit_reason"] = trades["exit_reason"].fillna("UNKNOWN").astype(str)



    if "reached_abs_z_1" not in trades.columns:

        trades["reached_abs_z_1"] = False

    trades["reached_abs_z_1"] = trades["reached_abs_z_1"].fillna(False).astype(bool)



    rows = []

    for industry, g in trades.groupby("industry", dropna=False):

        n = len(g)

        mr = g[g["exit_reason"] == "MEAN_REVERSION"]

        hard = g[g["exit_reason"] == "HARD_STOP"]

        time = g[g["exit_reason"] == "TIME_STOP"]

        cbreak = g[g["exit_reason"] == "COINTEGRATION_BREAK"]



        rows.append({

            "industry": industry,

            "trades": int(n),

            "mean_reversion_exits": int(len(mr)),

            "mean_reversion_rate": float(len(mr) / n) if n else np.nan,

            "hard_stop_rate": float(len(hard) / n) if n else np.nan,

            "time_stop_rate": float(len(time) / n) if n else np.nan,

            "cointegration_break_rate": float(len(cbreak) / n) if n else np.nan,

            "overall_win_rate": float((g["net_pnl"] > 0).mean()) if n else np.nan,

            "mean_reversion_win_rate": float((mr["net_pnl"] > 0).mean()) if len(mr) else np.nan,

            "avg_mean_reversion_pnl": float(mr["net_pnl"].mean()) if len(mr) else np.nan,

            "total_mean_reversion_pnl": float(mr["net_pnl"].sum()) if len(mr) else 0.0,

            "convergence_rate": float(g["reached_abs_z_1"].mean()) if n else np.nan,

            "avg_z_mfe": float(g["z_mfe"].mean()) if g["z_mfe"].notna().any() else np.nan,

            "avg_z_mae": float(g["z_mae"].mean()) if g["z_mae"].notna().any() else np.nan,

            "gross_pnl": float(g["gross_pnl"].sum()),

            "transaction_cost": float(g["transaction_cost"].sum()),

            "borrow_cost": float(g["borrow_cost"].sum()),

            "net_pnl": float(g["net_pnl"].sum()),

        })



    return pd.DataFrame(rows).sort_values(

        ["mean_reversion_rate", "convergence_rate", "trades"],

        ascending=[False, False, False],

        na_position="last",

    ).reset_index(drop=True)





def build_industry_subset_portfolio(

    strategies,

    ledgers,

    selected,

    industries,

    allocation_method="Equal-weight active pairs",

    max_pair_weight=0.25,

    portfolio_gross=1.0,

):

    """

    Build an ex-post comparison portfolio from selected industries.



    If the industries were chosen from this same backtest, the result is

    explicitly in-sample research and not untouched OOS evidence.

    """

    industries = list(industries or [])

    if not industries:

        return None



    subset_strategies = {

        name: data

        for name, data in strategies.items()

        if str(data["industry"].iloc[0]) in industries

    }

    if not subset_strategies:

        return None



    subset_ledgers = {

        name: ledger for name, ledger in ledgers.items()

        if name in subset_strategies

    }

    subset_selected = selected[selected["industry"].isin(industries)].copy()



    combined = combine_pair_strategies(

        subset_strategies,

        allocation_method=allocation_method,

        max_pair_weight=max_pair_weight,

        portfolio_gross=portfolio_gross,

    )

    portfolio = combined["portfolio"]

    statistics = portfolio_statistics(portfolio["return"])

    statistics["average_active_pairs"] = float(portfolio["active_pairs"].mean())

    statistics["maximum_active_pairs"] = int(portfolio["active_pairs"].max())

    statistics["average_capital_deployed"] = float(portfolio["capital_deployed"].mean())

    statistics["maximum_capital_deployed"] = float(portfolio["capital_deployed"].max())



    outcomes, economics = trade_outcome_diagnostics(subset_ledgers)

    contribution = contribution_analysis(

        combined["pair_returns"],

        combined["weights"],

        subset_selected,

        subset_ledgers,

    )



    return {

        "industries": industries,

        "selected": subset_selected,

        "portfolio": portfolio,

        "pair_returns": combined["pair_returns"],

        "weights": combined["weights"],

        "statistics": statistics,

        "trade_outcomes": outcomes,

        "trade_economics": economics,

        "contribution": contribution,

        "correlation": combined["pair_returns"].corr(),

        "ledgers": subset_ledgers,

    }







# ============================================================

# WALK-FORWARD INDUSTRY QUALITY DIAGNOSTIC

# ============================================================



def walkforward_industry_quality_analysis(ledgers, strategies, trading_window=21, forward_window=63):

    """Test whether past industry trade-path quality predicts the next 63 trading days.



    Diagnostic only: this function never changes historical eligibility or trades.

    Predictor snapshots are refreshed every ``trading_window`` trading days using

    only trades completed strictly before the snapshot date. Outcomes use trades

    entered during the following ``forward_window`` dates from the actual strategy

    trading calendar.

    """

    trades = _combine_ledgers(ledgers)

    cols = [

        "industry", "block_start", "block_end", "prior_trades",

        "prior_convergence_rate", "prior_hard_stop_rate",

        "prior_avg_z_mfe", "prior_avg_z_mae", "prior_mfe_mae_ratio",

        "prior_win_rate", "next_63d_trades", "next_63d_net_pnl",

        "next_63d_pnl_per_trade", "next_63d_win_rate",

        "next_63d_convergence_rate",

    ]

    if trades.empty or "industry" not in trades.columns or not strategies:

        return pd.DataFrame(columns=cols)



    # Actual daily strategy calendar (not a calendar made from sparse trade dates).

    calendar_parts = []

    for strategy in strategies.values():

        if strategy is not None and not strategy.empty:

            idx = pd.DatetimeIndex(strategy.index)

            if idx.tz is not None:

                idx = idx.tz_localize(None)

            calendar_parts.extend(idx.tolist())

    if not calendar_parts:

        return pd.DataFrame(columns=cols)

    calendar = pd.DatetimeIndex(sorted(set(calendar_parts)))

    if len(calendar) == 0:

        return pd.DataFrame(columns=cols)



    def _date_series(primary, fallback):

        if primary in trades.columns:

            x = pd.to_datetime(trades[primary], errors="coerce")

            if x.notna().any():

                return x.dt.tz_localize(None) if getattr(x.dt, "tz", None) is not None else x

        if fallback in trades.columns:

            x = pd.to_datetime(trades[fallback], errors="coerce")

            return x.dt.tz_localize(None) if getattr(x.dt, "tz", None) is not None else x

        return pd.Series(pd.NaT, index=trades.index, dtype="datetime64[ns]")



    trades = trades.copy()

    trades["_entry_date"] = _date_series("entry_signal_date", "entry_date")

    trades["_exit_date"] = _date_series("exit_signal_date", "exit_date")

    for c in ["net_pnl", "z_mfe", "z_mae"]:

        if c not in trades.columns:

            trades[c] = np.nan

        trades[c] = pd.to_numeric(trades[c], errors="coerce")

    if "reached_abs_z_1" not in trades.columns:

        trades["reached_abs_z_1"] = False

    trades["reached_abs_z_1"] = trades["reached_abs_z_1"].fillna(False).astype(bool)

    if "exit_reason" not in trades.columns:

        trades["exit_reason"] = "UNKNOWN"

    trades["exit_reason"] = trades["exit_reason"].fillna("UNKNOWN").astype(str)



    step = max(int(trading_window), 1)

    rows = []

    for pos in range(0, len(calendar), step):

        block_dates = calendar[pos:pos + step]

        forward_dates = calendar[pos:pos + max(int(forward_window), 1)]

        if len(block_dates) == 0 or len(forward_dates) == 0:

            continue

        b0, b1 = block_dates[0], forward_dates[-1]

        for industry, g in trades.groupby("industry", dropna=False):

            prior = g[g["_exit_date"] < b0]

            future = g[(g["_entry_date"] >= b0) & (g["_entry_date"] <= b1)]

            if prior.empty or future.empty:

                continue

            mfe = prior["z_mfe"].dropna()

            mae = prior["z_mae"].dropna()

            avg_mfe = float(mfe.mean()) if len(mfe) else np.nan

            avg_mae = float(mae.mean()) if len(mae) else np.nan

            ratio = avg_mfe / avg_mae if np.isfinite(avg_mfe) and np.isfinite(avg_mae) and avg_mae > 0 else np.nan

            rows.append({

                "industry": industry, "block_start": b0, "block_end": b1,

                "prior_trades": int(len(prior)),

                "prior_convergence_rate": float(prior["reached_abs_z_1"].mean()),

                "prior_hard_stop_rate": float((prior["exit_reason"] == "HARD_STOP").mean()),

                "prior_avg_z_mfe": avg_mfe, "prior_avg_z_mae": avg_mae,

                "prior_mfe_mae_ratio": ratio,

                "prior_win_rate": float((prior["net_pnl"] > 0).mean()),

                "next_63d_trades": int(len(future)),

                "next_63d_net_pnl": float(future["net_pnl"].sum()),

                "next_63d_pnl_per_trade": float(future["net_pnl"].mean()),

                "next_63d_win_rate": float((future["net_pnl"] > 0).mean()),

                "next_63d_convergence_rate": float(future["reached_abs_z_1"].mean()),

            })

    return pd.DataFrame(rows, columns=cols)



# ============================================================

# RUN FULL MULTI-SECTOR PORTFOLIO

# ============================================================



def run_multi_sector_portfolio(

    start, end, selected_industries=None, formation_window=252, trading_window=21,

    z_window=60, coint_threshold=0.05, max_half_life=60, z_entry=2.0,

    z_exit=0.5, z_stop=3.5, z_entry_max=None, recent_equilibrium_max_days=None, transaction_cost_bps=5.0, annual_borrow_bps=50.0,

    min_valid_blocks=0, min_regimes=0, min_trades=0,

    allocation_method="Equal-weight active pairs", max_pair_weight=0.25,

    portfolio_gross=1.0, prescreen_max_pairs=None, prescreen_min_return_corr=None,

    prescreen_min_log_price_corr=None, progress_callback=None,

    persistence_lookback=6, persistence_min_passes=3,

):

    """

    Fully rolling pair eligibility.



    No pair is selected using full-history Sharpe, return, correlation, valid-block

    fraction, or future cointegration results. Every peer-group pair is tested in

    every walk-forward block using only the preceding formation window. Trading is

    permitted only in blocks where cointegration and half-life conditions pass.

    """

    if selected_industries is None:

        selected_industries = get_industries()



    selected_rows, strategies, ledgers = [], {}, {}

    scanner_results, prescreen_results, failures = {}, {}, []

    pair_block_frames = []

    universe_diagnostics = []

    industry_total = len(selected_industries)



    for industry_index, industry in enumerate(selected_industries, start=1):

        try:

            _, scanner, opens, closes, prescreen, diagnostics, industry_block_diag = scan_industry_for_portfolio(

                industry=industry, start=start, end=end,

                formation_window=formation_window, trading_window=trading_window,

                z_window=z_window, coint_threshold=coint_threshold,

                max_half_life=max_half_life, z_entry=z_entry, z_exit=z_exit,

                z_stop=z_stop, transaction_cost_bps=transaction_cost_bps,

                annual_borrow_bps=annual_borrow_bps,

                progress_callback=progress_callback, industry_index=industry_index,

                industry_total=industry_total,

                persistence_lookback=persistence_lookback,

                persistence_min_passes=persistence_min_passes,

            )

            scanner_results[industry] = scanner

            prescreen_results[industry] = prescreen

            universe_diagnostics.append(diagnostics)

            if isinstance(industry_block_diag, pd.DataFrame) and not industry_block_diag.empty:

                pair_block_frames.append(industry_block_diag)



            if scanner is None or scanner.empty:

                failures.append({"industry": industry, "reason": "No models completed."})

                continue



            # These are NOT ex-post selection filters. Rows with no historically

            # valid block or no trade contribute identically zero, so skipping

            # their second execution pass only saves computation.

            usable = scanner[

                (scanner["valid_blocks"].fillna(0) > 0)

                & (scanner["trades"].fillna(0) > 0)

            ].copy()



            for _, candidate in usable.iterrows():

                try:

                    strategy, ledger = build_selected_pair_strategy(

                        candidate, opens, closes, formation_window=formation_window,

                        trading_window=trading_window, z_window=z_window,

                        coint_threshold=coint_threshold, max_half_life=max_half_life,

                        z_entry=z_entry, z_exit=z_exit, z_stop=z_stop,

                        z_entry_max=z_entry_max,

                        recent_equilibrium_max_days=recent_equilibrium_max_days,

                        transaction_cost_bps=transaction_cost_bps,

                        annual_borrow_bps=annual_borrow_bps,

                        persistence_lookback=persistence_lookback,

                        persistence_min_passes=persistence_min_passes,

                    )

                    pair_name = candidate["pair"]

                    strategies[pair_name] = strategy

                    ledgers[pair_name] = ledger

                    selected_rows.append(candidate.copy())

                except Exception as exc:

                    failures.append({

                        "industry": industry,

                        "pair": candidate.get("pair", ""),

                        "stage": "strategy_rebuild",

                        "reason": str(exc),

                    })

                    continue



        except Exception as exc:

            failures.append({

                "industry": industry,

                "pair": "",

                "stage": "industry_scan",

                "reason": str(exc),

            })



    if not strategies:

        return {

            "selected": pd.DataFrame(), "portfolio": pd.DataFrame(),

            "pair_returns": pd.DataFrame(), "weights": pd.DataFrame(),

            "contribution": pd.DataFrame(), "correlation": pd.DataFrame(),

            "statistics": {}, "ledgers": {},

            "scanner_results": scanner_results,

            "prescreen_results": prescreen_results,

            "universe_diagnostics": pd.DataFrame(universe_diagnostics),

            "failures": pd.DataFrame(failures),

            "research_funnel": pd.DataFrame(),

            "trade_outcomes": pd.DataFrame(),

            "trade_economics": {},

            "industry_mean_reversion": pd.DataFrame(),

            "industry_quality_walkforward": pd.DataFrame(),

            "strategies": {},

            "pair_block_diagnostics": (

                pd.concat(pair_block_frames, ignore_index=True)

                if pair_block_frames else pd.DataFrame()

            ),

        }



    selected = pd.DataFrame(selected_rows).reset_index(drop=True)



    # Aggregate the research funnel across ALL completed pair models, not

    # just pairs that eventually traded. This keeps the diagnostic honest.

    funnel_rows = []

    for industry_name, scan in scanner_results.items():

        if scan is None or scan.empty:

            continue

        funnel_rows.append({

            "industry": industry_name,

            "pair_models": int(len(scan)),

            "tested_blocks": int(scan.get("total_blocks", pd.Series(dtype=float)).fillna(0).sum()),

            "cointegration_pass_blocks": int(scan.get("coint_pass_blocks", pd.Series(dtype=float)).fillna(0).sum()),

            "half_life_pass_blocks": int(scan.get("half_life_pass_blocks", pd.Series(dtype=float)).fillna(0).sum()),

            "persistence_pass_blocks": int(scan.get("persistence_pass_blocks", pd.Series(dtype=float)).fillna(0).sum()),

            "z_entry_blocks": int(scan.get("z_entry_blocks", pd.Series(dtype=float)).fillna(0).sum()),

            "completed_trades": int(scan.get("trades", pd.Series(dtype=float)).fillna(0).sum()),

        })

    research_funnel = pd.DataFrame(funnel_rows)



    if progress_callback is not None:

        progress_callback({"stage": "combining", "fraction": 1.0,

                           "industry_total": industry_total})



    combined = combine_pair_strategies(

        strategies, allocation_method=allocation_method,

        max_pair_weight=max_pair_weight, portfolio_gross=portfolio_gross,

    )

    portfolio = combined["portfolio"]

    pair_returns = combined["pair_returns"]

    weights = combined["weights"]

    statistics = portfolio_statistics(portfolio["return"])

    statistics["average_active_pairs"] = float(portfolio["active_pairs"].mean())

    statistics["maximum_active_pairs"] = int(portfolio["active_pairs"].max())

    statistics["average_capital_deployed"] = float(portfolio["capital_deployed"].mean())

    statistics["maximum_capital_deployed"] = float(portfolio["capital_deployed"].max())



    contribution = contribution_analysis(pair_returns, weights, selected, ledgers)

    correlation = pair_returns.corr()

    trade_outcomes, trade_economics = trade_outcome_diagnostics(ledgers)

    industry_mean_reversion = industry_mean_reversion_analysis(ledgers)

    industry_quality_walkforward = walkforward_industry_quality_analysis(

        ledgers, strategies, trading_window=trading_window

    )



    pair_block_diagnostics = (

        pd.concat(pair_block_frames, ignore_index=True)

        if pair_block_frames else pd.DataFrame()

    )



    if progress_callback is not None:

        progress_callback({"stage": "complete", "fraction": 1.0,

                           "industry_total": industry_total})



    return {

        "selected": selected, "portfolio": portfolio,

        "pair_returns": pair_returns, "weights": weights,

        "contribution": contribution, "correlation": correlation,

        "statistics": statistics, "ledgers": ledgers,

        "scanner_results": scanner_results,

        "prescreen_results": prescreen_results,

        "universe_diagnostics": pd.DataFrame(universe_diagnostics),

        "failures": pd.DataFrame(failures),

        "research_funnel": research_funnel,

        "trade_outcomes": trade_outcomes,

        "trade_economics": trade_economics,

        "industry_mean_reversion": industry_mean_reversion,

        "industry_quality_walkforward": industry_quality_walkforward,

        "strategies": strategies,

        "pair_block_diagnostics": pair_block_diagnostics,

    }


