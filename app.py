from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf

from data.market_data import (
    download_pair,
    normalize_symbol,
)

from models.walkforward import (
    build_walkforward,
)

from models.pair_model import (
    fit_orientation,
    rolling_market_beta,
)

from strategy.execution import (
    run_execution,
)

from analytics.performance import (
    stats,
    pair_quality,
)

from analytics.sensitivity import (
    run_sensitivity,
)

from analytics.pair_scanner import (
    run_pair_scanner,
    get_industries,
    get_industry_tickers,
    generate_industry_pairs,
    get_universe_source,
    get_universe_snapshot_date,
    get_universe_metadata,
    get_universe_security_count,
    get_pair_eligible_security_count,
)

from analytics.portfolio_engine import (
    run_multi_sector_portfolio,
    build_industry_subset_portfolio,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Equity StatArb Lab",
    layout="wide",
)

st.title(
    "Equity Statistical Arbitrage Lab"
)

st.caption(
    "Walk-forward statistical arbitrage research, "
    "cross-sectional pair discovery and multi-pair "
    "portfolio construction."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # RESEARCH MODE
    # --------------------------------------------------------

    st.header(
        "Research Mode"
    )

    research_mode = st.radio(
        "Analysis type",
        [
            "Single Pair Analysis",
            "US Pair Scanner",
            "Multi-Sector Portfolio",
        ],
    )

    # ========================================================
    # SINGLE PAIR
    # ========================================================

    if research_mode == "Single Pair Analysis":

        st.header(
            "Market"
        )

        market = st.selectbox(
            "Equity market",
            [
                "US",
                "India (NSE)",
            ],
        )

        if market == "US":

            default_a = "V"
            default_b = "MA"

            currency = "USD"

        else:

            default_a = "HDFCBANK"
            default_b = "ICICIBANK"

            currency = "INR"

        st.header(
            "Pair"
        )

        a = st.text_input(
            "Equity A (ticker)",
            default_a,
        ).upper().strip()

        b = st.text_input(
            "Equity B (ticker)",
            default_b,
        ).upper().strip()

        if market == "India (NSE)":

            st.caption(
                "Enter NSE symbols without .NS. "
                "The application adds the suffix automatically."
            )

    # ========================================================
    # SECTOR SCANNER
    # ========================================================

    elif research_mode == "US Pair Scanner":

        market = "US"
        currency = "USD"

        st.header(
            "Scanner Universe"
        )

        scanner_industry = st.selectbox(
            "Industry / peer group",
            get_industries(),
        )

        scanner_tickers = (
            get_industry_tickers(
                scanner_industry
            )
        )

        scanner_pairs = (
            generate_industry_pairs(
                scanner_industry
            )
        )

        x1, x2 = st.columns(2)

        x1.metric(
            "Equities",
            len(scanner_tickers),
        )

        x2.metric(
            "Pairs",
            len(scanner_pairs),
        )

        with st.expander(
            "View equities"
        ):

            st.write(
                ", ".join(
                    scanner_tickers
                )
            )

    # ========================================================
    # MULTI-SECTOR PORTFOLIO
    # ========================================================

    else:

        market = "US"
        currency = "USD"

        st.header(
            "Portfolio Universe"
        )

        available_industries = (
            get_industries()
        )

        selected_industries = (
            st.multiselect(
                "Industries",
                available_industries,
                default=available_industries,
            )
        )

        selected_equities = sorted({
            ticker
            for industry in selected_industries
            for ticker in get_industry_tickers(industry)
        })
        raw_pair_count = sum(
            len(generate_industry_pairs(industry))
            for industry in selected_industries
        )

        total_loaded = get_universe_security_count()
        pair_eligible = get_pair_eligible_security_count()

        if total_loaded != 503:
            st.error(
                f"Embedded universe validation failed: expected 503 securities, "
                f"but loaded {total_loaded}. Backtest stopped."
            )
            st.stop()

        u1, u2, u3, u4 = st.columns(4)
        u1.metric("S&P securities loaded", total_loaded)
        u2.metric("Pair-eligible securities", pair_eligible)
        u3.metric("Selected peer groups", len(selected_industries))
        u4.metric("Raw selected pairs", raw_pair_count)

        st.caption(
            f"Universe: {get_universe_source()}. "
            f"Snapshot date: {get_universe_snapshot_date()}."
        )
        st.caption(
            "The complete S&P universe is embedded directly in pair_scanner.py. "
            "Normal peer groups use GICS sub-industries; singleton sub-industries "
            "are pooled only with other singleton sub-industries in the same "
            "GICS sector. Historical tests using this 2026 snapshot retain "
            "survivorship bias."
        )

        with st.expander("Embedded S&P 500 membership and index-addition dates"):
            membership = get_universe_metadata()
            st.dataframe(
                membership[
                    [
                        "Symbol", "Security", "GICS Sector",
                        "GICS Sub-Industry", "Peer Group",
                        "Date added", "Pair Eligible",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    # ========================================================
    # DATA
    # ========================================================

    st.header(
        "Data"
    )

    end = st.date_input(
        "End date",
        date.today(),
    )

    start = st.date_input(
        "Start date",
        end - timedelta(
            days=6 * 365
        ),
    )

    # ========================================================
    # WALK FORWARD
    # ========================================================

    st.header(
        "Walk-Forward Model"
    )

    formation = st.number_input(
        "Formation window (trading days)",
        min_value=126,
        max_value=756,
        value=252,
        step=21,
        help=(
            "Historical observations used to estimate "
            "the pair relationship before each "
            "out-of-sample trading block."
        ),
    )

    block = st.number_input(
        "Trading block (trading days)",
        min_value=5,
        max_value=126,
        value=21,
        step=5,
    )

    zwin = st.number_input(
        "Z-score lookback (trading days)",
        min_value=20,
        max_value=252,
        value=60,
        step=5,
    )

    pth = st.number_input(
        "Cointegration p-value threshold",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
    )

    maxhl = st.number_input(
        "Maximum half-life (trading days)",
        min_value=5,
        max_value=252,
        value=60,
        step=5,
    )

    # ========================================================
    # TRADING RULES
    # ========================================================

    st.header(
        "Trading Rules"
    )

    zentry = st.number_input(
        "Entry threshold |Z| (standard deviations)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.05,
    )

    zexit = st.number_input(
        "Exit threshold |Z| (standard deviations)",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.05,
    )

    z_entry_max = st.number_input(
        "Maximum entry |Z| (upper-entry-cap experiment)",
        min_value=0.5,
        max_value=8.0,
        value=2.5,
        step=0.05,
        help=(
            "New positions may open only when Entry threshold ≤ |Z| < this value. "
            "This does not close an existing position; the Hard stop remains separate."
        ),
    )

    zstop = st.number_input(
        "Hard stop |Z| (standard deviations)",
        min_value=1.0,
        max_value=8.0,
        value=3.5,
        step=0.1,
    )

    run_upper_entry_experiment = st.checkbox(
        "Automatically compare maximum entry |Z| = 2.50 / 2.75 / 3.00 / 3.50",
        value=False,
        help=(
            "Runs the same selected industries and all the same settings four times. "
            "Only the maximum entry |Z| changes. The full normal portfolio output is "
            "left unchanged when this box is off."
        ),
    )

    run_recent_equilibrium_experiment = st.checkbox(
        "Test recent-equilibrium filter: baseline vs ≤ 5 trading days since |Z| ≤ 1.5",
        value=False,
        help=(
            "Runs the predeclared 10-industry research universe twice. Both runs use the "
            "original entry range up to the 3.5 hard stop; only the recent-equilibrium "
            "entry filter changes."
        ),
    )

    # ========================================================
    # EXECUTION COSTS
    # ========================================================

    st.header(
        "Execution"
    )

    tc = st.number_input(
        "Transaction cost (bps per unit turnover)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=0.5,
    )

    bc = st.number_input(
        "Annual short-borrow cost (bps p.a.)",
        min_value=0.0,
        max_value=2000.0,
        value=50.0,
        step=10.0,
    )

    # ========================================================
    # SINGLE PAIR EXECUTION
    # ========================================================

    if research_mode == "Single Pair Analysis":

        gross = st.slider(
            "Gross exposure (%)",
            min_value=25,
            max_value=200,
            value=100,
            step=25,
        )

        sizing = st.selectbox(
            "Position sizing",
            [
                "Hedge-ratio weighted",
                "Dollar neutral",
                "Beta neutral",
            ],
        )

    # ========================================================
    # PORTFOLIO SETTINGS
    # ========================================================

    if research_mode == "Multi-Sector Portfolio":

        st.header("Rolling Eligibility Rule")

        st.caption(
            "No correlation pre-screen and no full-history pair selection. "
            "Every within-peer pair is tested at each rebalance using only the "
            "preceding formation window. A pair is eligible for the next trading "
            "block only when Engle-Granger p is below the cointegration threshold "
            "and its estimated half-life is positive and within the maximum. "
            "A past-only persistence rule can additionally require the relationship "
            "to have passed in prior formation windows before capital is deployed."
        )

        persistence_lookback = st.number_input(
            "Persistence lookback (prior formation windows)",
            min_value=1,
            max_value=24,
            value=6,
            step=1,
            help=(
                "Number of PRIOR walk-forward formation windows used to judge "
                "whether cointegration has persisted. The current window is not "
                "included, so the rule remains strictly past-only."
            ),
        )

        persistence_min_passes = st.number_input(
            "Minimum prior cointegration passes",
            min_value=1,
            max_value=24,
            value=3,
            step=1,
            help=(
                "A currently valid pair may trade only if at least this many of "
                "the previous persistence-lookback windows also passed the "
                "cointegration and half-life rule."
            ),
        )

        if persistence_min_passes > persistence_lookback:
            st.warning(
                "Minimum prior passes cannot exceed the persistence lookback. "
                "Increase the lookback or reduce the required passes."
            )

        # Legacy arguments are retained internally for API compatibility only.
        # They no longer filter or rank pairs.
        prescreen_max_pairs = 0
        prescreen_min_return_corr = 0.0
        prescreen_min_log_price_corr = 0.0
        min_valid_blocks = 0
        min_regimes = 0
        min_trades = 0

        st.header(
            "Portfolio Construction"
        )

        allocation_method = (
            st.selectbox(
                "Allocation method",
                [
                    "Equal-weight active pairs",
                    "Inverse-volatility",
                ],
            )
        )

        portfolio_gross_pct = (
            st.slider(
                "Maximum portfolio allocation (%)",
                min_value=25,
                max_value=200,
                value=100,
                step=25,
            )
        )

        max_pair_weight_pct = (
            st.slider(
                "Maximum allocation per pair (%)",
                min_value=5,
                max_value=100,
                value=25,
                step=5,
            )
        )

    # ========================================================
    # RUN BUTTON
    # ========================================================

    if research_mode == "Single Pair Analysis":

        button_text = (
            "Calibrate Pair"
        )

    elif research_mode == "US Pair Scanner":

        button_text = (
            "Run Sector Scanner"
        )

    else:

        button_text = (
            "Build Multi-Sector Portfolio"
        )

    run = st.button(
        button_text,
        type="primary",
        use_container_width=True,
    )


# ============================================================
# DATE VALIDATION
# ============================================================

if start >= end:

    st.error(
        "Start date must be earlier than end date."
    )

    st.stop()

if z_entry_max <= zentry:
    st.error("Maximum entry |Z| must be greater than the entry threshold |Z|.")
    st.stop()

if z_entry_max > zstop:
    st.error("Maximum entry |Z| cannot exceed the hard-stop |Z|.")
    st.stop()


# ============================================================
# MULTI-SECTOR PORTFOLIO
# ============================================================

if research_mode == "Multi-Sector Portfolio":

    st.header(
        "Multi-Sector Statistical Arbitrage Portfolio"
    )

    st.caption(
        "Every within-peer pair is evaluated walk-forward. A pair can trade only "
        "when its current formation window passes cointegration/half-life AND its "
        "past-only persistence rule. Sharpe and return never determine eligibility."
    )

    if len(selected_industries) == 0:

        st.warning(
            "Select at least one industry."
        )

        st.stop()

    # --------------------------------------------------------
    # UNIVERSE SIZE
    # --------------------------------------------------------

    total_equities = sum(
        len(
            get_industry_tickers(
                industry
            )
        )
        for industry
        in selected_industries
    )

    total_pairs = sum(
        len(
            generate_industry_pairs(
                industry
            )
        )
        for industry
        in selected_industries
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Industries",
        len(selected_industries),
    )

    m2.metric(
        "Pairable equities",
        total_equities,
    )

    m3.metric(
        "Raw within-peer pairs",
        total_pairs,
    )

    m4.metric(
        "Rolling pair models",
        total_pairs,
    )

    if not run:

        st.info(
            "Click **Build Multi-Sector Portfolio** "
            "to scan the selected industries and construct "
            "the portfolio."
        )

        st.stop()

    # --------------------------------------------------------
    # RUN PORTFOLIO ENGINE
    # --------------------------------------------------------

    st.subheader("Construction Progress")
    progress_bar = st.progress(0.0)
    progress_status = st.empty()
    progress_detail = st.empty()

    def update_portfolio_progress(info):
        fraction = float(info.get("fraction", 0.0) or 0.0)
        fraction = min(max(fraction, 0.0), 1.0)
        progress_bar.progress(fraction)

        stage = info.get("stage")
        industry_name = info.get("industry", "")
        industry_index = info.get("industry_index", "")
        industry_total = info.get("industry_total", len(selected_industries))

        if stage == "rolling_screen":
            raw_total = info.get("raw_pair_total", 0)
            progress_status.markdown(
                f"**Peer group {industry_index} / {industry_total}: {industry_name} — rolling eligibility**"
            )
            progress_detail.caption(
                f"Testing {raw_total} within-peer pairs using formation-window information only."
            )
        elif stage == "scanning":
            pair_name = info.get("pair", "")
            pair_index = info.get("pair_index", 0)
            pair_total = info.get("pair_total", 0)
            raw_total = info.get("raw_pair_total", 0)
            progress_status.markdown(
                f"**Peer group {industry_index} / {industry_total}: {industry_name} — full walk-forward**"
            )
            progress_detail.caption(
                f"Model {pair_index} / {pair_total}: {pair_name}  •  "
                f"rolling model {pair_index} / {pair_total}; no correlation pre-screen."
            )
        elif stage == "combining":
            progress_status.markdown("**Pair scanning complete — combining selected strategies...**")
            progress_detail.caption("Constructing portfolio returns, weights, risk and contribution diagnostics.")
        elif stage == "complete":
            progress_bar.progress(1.0)
            progress_status.success("Portfolio construction complete.")
            progress_detail.caption("All selected peer groups processed (100%).")

    # --------------------------------------------------------
    # PREDECLARED RECENT-EQUILIBRIUM EXPERIMENT
    # --------------------------------------------------------

    if run_recent_equilibrium_experiment:
        recent_eq_industries = [
            "Asset Management & Custody Banks",
            "Transaction & Payment Processing Services",
            "Diversified Banks",
            "Oil & Gas Exploration & Production",
            "Biotechnology",
            "Health Care Equipment",
            "Semiconductors & Semiconductor Equipment",
            "Aerospace & Defense",
            "Electric Utilities",
            "Packaged Foods & Meats",
        ]
        experiment_specs = [
            ("Baseline (no recent-equilibrium filter)", None),
            ("Recent equilibrium ≤ 5 trading days", 5),
        ]
        experiment_rows = []

        st.subheader("Recent-Equilibrium Entry Experiment")
        st.caption(
            "Predeclared test on the same 10-industry research universe. Both runs use "
            "Entry |Z| = 2.0 and allow entries up to the 3.5 hard stop. The only changed "
            "rule is whether the spread must have visited |Z| ≤ 1.5 within the prior "
            "5 trading observations. No future Z values are used."
        )
        exp_progress = st.progress(0.0)
        exp_status = st.empty()

        for exp_index, (label, max_days) in enumerate(experiment_specs, start=1):
            exp_status.markdown(f"**Run {exp_index} / 2 — {label}**")
            exp_result = run_multi_sector_portfolio(
                start=start,
                end=end,
                selected_industries=recent_eq_industries,
                formation_window=int(formation),
                trading_window=int(block),
                z_window=int(zwin),
                coint_threshold=float(pth),
                max_half_life=float(maxhl),
                z_entry=float(zentry),
                z_exit=float(zexit),
                z_stop=float(zstop),
                z_entry_max=float(zstop),
                recent_equilibrium_max_days=max_days,
                transaction_cost_bps=float(tc),
                annual_borrow_bps=float(bc),
                min_valid_blocks=int(min_valid_blocks),
                min_regimes=int(min_regimes),
                min_trades=int(min_trades),
                allocation_method=allocation_method,
                max_pair_weight=max_pair_weight_pct / 100,
                portfolio_gross=portfolio_gross_pct / 100,
                prescreen_max_pairs=int(prescreen_max_pairs),
                prescreen_min_return_corr=float(prescreen_min_return_corr),
                prescreen_min_log_price_corr=float(prescreen_min_log_price_corr),
                persistence_lookback=int(persistence_lookback),
                persistence_min_passes=int(persistence_min_passes),
                progress_callback=None,
            )

            exp_stats = exp_result.get("statistics", {})
            exp_ledgers = exp_result.get("ledgers", {})
            exp_frames = [df for df in exp_ledgers.values() if df is not None and not df.empty]
            exp_ledger = pd.concat(exp_frames, ignore_index=True) if exp_frames else pd.DataFrame()
            n_trades = len(exp_ledger)
            if n_trades and "exit_reason" in exp_ledger.columns:
                reasons = exp_ledger["exit_reason"].astype(str).str.upper()
                hard_stop_rate = reasons.str.contains("HARD").mean()
                mean_reversion_rate = reasons.str.contains("MEAN|REVERSION").mean()
            else:
                hard_stop_rate = np.nan
                mean_reversion_rate = np.nan

            experiment_rows.append({
                "Rule": label,
                "Annual return": exp_stats.get("annual_return", np.nan),
                "Annual volatility": exp_stats.get("annual_volatility", np.nan),
                "Sharpe": exp_stats.get("sharpe", np.nan),
                "Sortino": exp_stats.get("sortino", np.nan),
                "Maximum drawdown": exp_stats.get("max_drawdown", np.nan),
                "Calmar": exp_stats.get("calmar", np.nan),
                "Trades": n_trades,
                "Hard-stop rate": hard_stop_rate,
                "Mean-reversion rate": mean_reversion_rate,
                "Average active pairs": exp_stats.get("average_active_pairs", np.nan),
                "Average capital deployed": exp_stats.get("average_capital_deployed", np.nan),
            })
            exp_progress.progress(exp_index / 2)

        exp_status.success("Recent-equilibrium experiment complete.")
        recent_eq_df = pd.DataFrame(experiment_rows)
        st.dataframe(
            recent_eq_df.style.format({
                "Annual return": "{:.2%}", "Annual volatility": "{:.2%}",
                "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
                "Maximum drawdown": "{:.2%}", "Calmar": "{:.2f}",
                "Hard-stop rate": "{:.1%}", "Mean-reversion rate": "{:.1%}",
                "Average active pairs": "{:.2f}", "Average capital deployed": "{:.1%}",
            }, na_rep="—"),
            use_container_width=True,
        )
        st.download_button(
            "Download recent-equilibrium experiment CSV",
            data=recent_eq_df.to_csv(index=False).encode("utf-8"),
            file_name="recent_equilibrium_experiment.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.info(
            "Experiment mode stops here after the two portfolio runs. Untick the box to "
            "return to the normal full portfolio and all ledger/diagnostic tabs."
        )
        st.stop()

    # --------------------------------------------------------
    # AUTOMATIC UPPER-ENTRY-CAP EXPERIMENT
    # --------------------------------------------------------

    if run_upper_entry_experiment:
        experiment_caps = [2.50, 2.75, 3.00, 3.50]
        experiment_rows = []

        st.subheader("Upper Entry |Z| Experiment")
        st.caption(
            "The selected industries and every other strategy setting are held fixed. "
            "Only the maximum entry |Z| changes across the four runs."
        )
        exp_progress = st.progress(0.0)
        exp_status = st.empty()

        for cap_index, cap in enumerate(experiment_caps, start=1):
            exp_status.markdown(
                f"**Run {cap_index} / {len(experiment_caps)} — Maximum entry |Z| = {cap:.2f}**"
            )

            exp_result = run_multi_sector_portfolio(
                start=start,
                end=end,
                selected_industries=selected_industries,
                formation_window=int(formation),
                trading_window=int(block),
                z_window=int(zwin),
                coint_threshold=float(pth),
                max_half_life=float(maxhl),
                z_entry=float(zentry),
                z_exit=float(zexit),
                z_stop=float(zstop),
                z_entry_max=float(cap),
                transaction_cost_bps=float(tc),
                annual_borrow_bps=float(bc),
                min_valid_blocks=int(min_valid_blocks),
                min_regimes=int(min_regimes),
                min_trades=int(min_trades),
                allocation_method=allocation_method,
                max_pair_weight=max_pair_weight_pct / 100,
                portfolio_gross=portfolio_gross_pct / 100,
                prescreen_max_pairs=int(prescreen_max_pairs),
                prescreen_min_return_corr=float(prescreen_min_return_corr),
                prescreen_min_log_price_corr=float(prescreen_min_log_price_corr),
                persistence_lookback=int(persistence_lookback),
                persistence_min_passes=int(persistence_min_passes),
                progress_callback=None,
            )

            exp_stats = exp_result.get("statistics", {})
            exp_ledgers = exp_result.get("ledgers", {})
            exp_frames = [
                df for df in exp_ledgers.values()
                if df is not None and not df.empty
            ]
            exp_ledger = (
                pd.concat(exp_frames, ignore_index=True)
                if exp_frames else pd.DataFrame()
            )
            n_trades = len(exp_ledger)
            if n_trades and "exit_reason" in exp_ledger.columns:
                reasons = exp_ledger["exit_reason"].astype(str).str.upper()
                hard_stop_rate = reasons.str.contains("HARD").mean()
                mean_reversion_rate = reasons.str.contains("MEAN|REVERSION").mean()
            else:
                hard_stop_rate = np.nan
                mean_reversion_rate = np.nan

            experiment_rows.append({
                "Maximum entry |Z|": cap,
                "Annual return": exp_stats.get("annual_return", np.nan),
                "Annual volatility": exp_stats.get("annual_volatility", np.nan),
                "Sharpe": exp_stats.get("sharpe", np.nan),
                "Sortino": exp_stats.get("sortino", np.nan),
                "Maximum drawdown": exp_stats.get("max_drawdown", np.nan),
                "Calmar": exp_stats.get("calmar", np.nan),
                "Trades": n_trades,
                "Hard-stop rate": hard_stop_rate,
                "Mean-reversion rate": mean_reversion_rate,
                "Average active pairs": exp_stats.get("average_active_pairs", np.nan),
                "Average capital deployed": exp_stats.get("average_capital_deployed", np.nan),
            })
            exp_progress.progress(cap_index / len(experiment_caps))

        exp_status.success("Upper-entry-cap experiment complete.")
        experiment_df = pd.DataFrame(experiment_rows)

        st.dataframe(
            experiment_df.style.format({
                "Maximum entry |Z|": "{:.2f}",
                "Annual return": "{:.2%}",
                "Annual volatility": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Sortino": "{:.2f}",
                "Maximum drawdown": "{:.2%}",
                "Calmar": "{:.2f}",
                "Hard-stop rate": "{:.1%}",
                "Mean-reversion rate": "{:.1%}",
                "Average active pairs": "{:.2f}",
                "Average capital deployed": "{:.1%}",
            }, na_rep="—"),
            use_container_width=True,
        )

        st.download_button(
            "Download upper-entry experiment CSV",
            data=experiment_df.to_csv(index=False).encode("utf-8"),
            file_name="upper_entry_z_experiment.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.info(
            "Experiment mode stops here so the app does not run a fifth portfolio. "
            "Untick the experiment box to run the normal full portfolio and view all ledgers and diagnostics."
        )
        st.stop()

    try:

        with st.spinner(
            f"Scanning {len(selected_industries)} industries, "
            f"rolling-testing {total_pairs} within-peer pairs and constructing "
            f"the portfolio..."
        ):

            result = (
                run_multi_sector_portfolio(
                    start=start,
                    end=end,

                    selected_industries=(
                        selected_industries
                    ),

                    formation_window=int(
                        formation
                    ),

                    trading_window=int(
                        block
                    ),

                    z_window=int(
                        zwin
                    ),

                    coint_threshold=float(
                        pth
                    ),

                    max_half_life=float(
                        maxhl
                    ),

                    z_entry=float(
                        zentry
                    ),

                    z_exit=float(
                        zexit
                    ),

                    z_stop=float(
                        zstop
                    ),

                    z_entry_max=float(
                        z_entry_max
                    ),

                    recent_equilibrium_max_days=None,

                    transaction_cost_bps=float(
                        tc
                    ),

                    annual_borrow_bps=float(
                        bc
                    ),

                    min_valid_blocks=int(
                        min_valid_blocks
                    ),

                    min_regimes=int(
                        min_regimes
                    ),

                    min_trades=int(
                        min_trades
                    ),

                    allocation_method=(
                        allocation_method
                    ),

                    max_pair_weight=(
                        max_pair_weight_pct
                        / 100
                    ),

                    portfolio_gross=(
                        portfolio_gross_pct
                        / 100
                    ),

                    prescreen_max_pairs=int(
                        prescreen_max_pairs
                    ),

                    prescreen_min_return_corr=float(
                        prescreen_min_return_corr
                    ),

                    prescreen_min_log_price_corr=float(
                        prescreen_min_log_price_corr
                    ),

                    persistence_lookback=int(
                        persistence_lookback
                    ),

                    persistence_min_passes=int(
                        persistence_min_passes
                    ),

                    progress_callback=(
                        update_portfolio_progress
                    ),
                )
            )
            st.session_state["multi_sector_result"] = result

    except Exception as exc:

        st.error(
            f"Portfolio construction failed: {exc}"
        )

        st.stop()

    selected = result[
        "selected"
    ]

    portfolio = result[
        "portfolio"
    ]

    contribution = result[
        "contribution"
    ]

    correlation = result[
        "correlation"
    ]

    weights = result[
        "weights"
    ]

    statistics = result[
        "statistics"
    ]

    failures = result[
        "failures"
    ]

    research_funnel = result.get(
        "research_funnel",
        pd.DataFrame(),
    )

    trade_outcomes = result.get(
        "trade_outcomes",
        pd.DataFrame(),
    )

    trade_economics = result.get(
        "trade_economics",
        {},
    )

    industry_mean_reversion = result.get(
        "industry_mean_reversion",
        pd.DataFrame(),
    )

    industry_quality_walkforward = result.get(
        "industry_quality_walkforward",
        pd.DataFrame(),
    )

    pair_block_diagnostics = result.get(
        "pair_block_diagnostics",
        pd.DataFrame(),
    )

    ledgers = result.get("ledgers", {})
    trade_ledger_frames = []
    for pair_name, ledger_df in ledgers.items():
        if ledger_df is None or ledger_df.empty:
            continue
        x = ledger_df.copy()
        if "pair" not in x.columns:
            x["pair"] = pair_name
        trade_ledger_frames.append(x)
    full_trade_ledger = (
        pd.concat(trade_ledger_frames, ignore_index=True)
        if trade_ledger_frames else pd.DataFrame()
    )

    strategies = result.get(
        "strategies",
        {},
    )

    scanner_results = result.get(
        "scanner_results",
        {},
    )

    universe_diagnostics = result.get(
        "universe_diagnostics",
        pd.DataFrame(),
    )

    if selected.empty:

        st.warning(
            "No industry produced a pair satisfying "
            "the current candidate requirements."
        )

        if not failures.empty:

            st.dataframe(
                failures,
                use_container_width=True,
            )

        st.stop()

    if not universe_diagnostics.empty:
        st.subheader("Rolling Eligibility Diagnostics")
        d1, d2, d3 = st.columns(3)
        d1.metric("Within-peer pairs", int(universe_diagnostics["raw_pairs"].sum()))
        d2.metric("Pairs sent to rolling test", int(universe_diagnostics["prescreen_survivors"].sum()))
        d3.metric("Rolling models completed", int(universe_diagnostics["full_models_tested"].sum()))
        with st.expander("Peer-group screening diagnostics"):
            st.dataframe(universe_diagnostics, use_container_width=True)

    # ========================================================
    # PORTFOLIO SUMMARY
    # ========================================================

    st.subheader(
        "Portfolio Performance"
    )

    p1, p2, p3, p4, p5 = (
        st.columns(5)
    )

    p1.metric(
        "Annual return (% p.a.)",
        f"{statistics['annual_return']:.2%}",
    )

    p2.metric(
        "Annualized volatility (% p.a.)",
        f"{statistics['annual_volatility']:.2%}",
    )

    p3.metric(
        "Sharpe ratio",
        f"{statistics['sharpe']:.2f}",
    )

    p4.metric(
        "Sortino ratio",
        f"{statistics['sortino']:.2f}",
    )

    p5.metric(
        "Maximum drawdown",
        f"{statistics['max_drawdown']:.2%}",
    )

    q1, q2, q3, q4 = (
        st.columns(4)
    )

    q1.metric(
        "Calmar ratio",
        f"{statistics['calmar']:.2f}",
    )

    q2.metric(
        "Average active pairs",
        f"{statistics['average_active_pairs']:.2f}",
    )

    q3.metric(
        "Maximum active pairs",
        f"{statistics['maximum_active_pairs']}",
    )

    q4.metric(
        "Average capital deployed",
        f"{statistics['average_capital_deployed']:.1%}",
    )

    # ========================================================
    # PORTFOLIO TABS
    # ========================================================

    portfolio_tabs = st.tabs(
        [
            "Selected Pairs",
            "Portfolio Equity",
            "Drawdown",
            "Contribution",
            "Correlation",
            "Weights",
            "Risk",
            "Research Diagnostics",
            "Industry Mean Reversion",
            "Industry Portfolio",
            "Walk-Forward Industry Quality",
            "Pair-Block Diagnostics",
            "Full Trade Ledger",
        ]
    )

    # ========================================================
    # SELECTED PAIRS
    # ========================================================

    with portfolio_tabs[0]:

        st.subheader(
            "Pairs With At Least One Out-of-Sample Trade"
        )

        st.caption(
            "This is an evaluation table, not a selection ranking. Eligibility is decided separately in each block using only the preceding formation window. Sharpe, return and future valid blocks are never used to decide eligibility."
        )

        selected_display = selected[
            [
                "industry",
                "pair",
                "valid_blocks",
                "valid_fraction",
                "validity_regimes",
                "valid_median_pvalue",
                "valid_median_half_life",
                "valid_median_r_squared",
                "trades",
                "sharpe",
            ]
        ].copy()

        selected_display[
            "valid_fraction"
        ] *= 100

        selected_display = (
            selected_display.rename(
                columns={
                    "industry":
                        "Industry",

                    "pair":
                        "Selected pair",

                    "valid_blocks":
                        "Valid blocks",

                    "valid_fraction":
                        "Valid blocks (%)",

                    "validity_regimes":
                        "Distinct regimes",

                    "valid_median_pvalue":
                        "Median p-value when valid",

                    "valid_median_half_life":
                        "Median half-life when valid",

                    "valid_median_r_squared":
                        "Median R² when valid",

                    "trades":
                        "Completed trades",

                    "sharpe":
                        "Pair Sharpe (evaluation only)",
                }
            )
        )

        st.dataframe(
            selected_display.style.format(
                {
                    "Valid blocks (%)":
                        "{:.1f}%",

                    "Median p-value when valid":
                        "{:.3f}",

                    "Median half-life when valid":
                        "{:.1f}",

                    "Median R² when valid":
                        "{:.3f}",

                    "Pair Sharpe (evaluation only)":
                        "{:.2f}",
                },
                na_rep="—",
            ),

            use_container_width=True,
        )

        if not failures.empty:

            with st.expander(
                "Industries without eligible candidates"
            ):

                st.dataframe(
                    failures,
                    use_container_width=True,
                )

    # ========================================================
    # EQUITY
    # ========================================================

    with portfolio_tabs[1]:

        st.subheader(
            "Combined Portfolio Equity"
        )

        equity_fig = go.Figure()

        equity_fig.add_trace(
            go.Scatter(
                x=portfolio.index,
                y=portfolio[
                    "equity"
                ],
                name="StatArb portfolio",
            )
        )

        equity_fig.update_layout(
            xaxis_title="Date",

            yaxis_title=(
                "Portfolio value (USD)"
            ),

            hovermode="x unified",

            height=520,
        )

        st.plotly_chart(
            equity_fig,
            use_container_width=True,
        )

        st.subheader(
            "Capital Deployment"
        )

        deployment_fig = go.Figure()

        deployment_fig.add_trace(
            go.Scatter(
                x=portfolio.index,

                y=(
                    portfolio[
                        "capital_deployed"
                    ]
                    * 100
                ),

                name="Capital deployed",
            )
        )

        deployment_fig.update_layout(
            xaxis_title="Date",

            yaxis_title=(
                "Allocated capital (%)"
            ),

            hovermode="x unified",

            height=400,
        )

        st.plotly_chart(
            deployment_fig,
            use_container_width=True,
        )

    # ========================================================
    # DRAWDOWN
    # ========================================================

    with portfolio_tabs[2]:

        st.subheader(
            "Portfolio Drawdown"
        )

        dd_fig = go.Figure()

        dd_fig.add_trace(
            go.Scatter(
                x=portfolio.index,

                y=(
                    portfolio[
                        "drawdown"
                    ]
                    * 100
                ),

                name="Drawdown",
                fill="tozeroy",
            )
        )

        dd_fig.update_layout(
            xaxis_title="Date",

            yaxis_title="Drawdown (%)",

            hovermode="x unified",

            height=500,
        )

        st.plotly_chart(
            dd_fig,
            use_container_width=True,
        )

    # ========================================================
    # CONTRIBUTION
    # ========================================================

    with portfolio_tabs[3]:

        st.subheader(
            "Pair Contribution Analysis"
        )

        contribution_display = (
            contribution.copy()
        )

        contribution_display[
            "contribution_share"
        ] *= 100

        contribution_display[
            "average_portfolio_weight"
        ] *= 100

        contribution_display[
            "maximum_portfolio_weight"
        ] *= 100

        contribution_display = (
            contribution_display.rename(
                columns={
                    "industry":
                        "Industry",

                    "pair":
                        "Pair",

                    "trades":
                        "Trades",

                    "gross_pair_pnl":
                        "Pair gross P&L (USD)",

                    "transaction_cost":
                        "Transaction costs (USD)",

                    "borrow_cost":
                        "Borrow costs (USD)",

                    "pair_net_pnl":
                        "Pair net P&L (USD)",

                    "portfolio_return_contribution":
                        "Portfolio return contribution",

                    "contribution_share":
                        "Contribution share (%)",

                    "average_portfolio_weight":
                        "Average portfolio weight (%)",

                    "maximum_portfolio_weight":
                        "Maximum portfolio weight (%)",
                }
            )
        )

        st.dataframe(
            contribution_display.style.format(
                {
                    "Pair gross P&L (USD)":
                        "${:,.2f}",

                    "Transaction costs (USD)":
                        "${:,.2f}",

                    "Borrow costs (USD)":
                        "${:,.2f}",

                    "Pair net P&L (USD)":
                        "${:,.2f}",

                    "Portfolio return contribution":
                        "{:.4f}",

                    "Contribution share (%)":
                        "{:.1f}%",

                    "Average portfolio weight (%)":
                        "{:.1f}%",

                    "Maximum portfolio weight (%)":
                        "{:.1f}%",
                },
                na_rep="—",
            ),

            use_container_width=True,
        )

    # ========================================================
    # CORRELATION
    # ========================================================

    with portfolio_tabs[4]:

        st.subheader(
            "Pair Strategy Return Correlation"
        )

        if correlation.empty:

            st.info(
                "Not enough selected pair strategies "
                "for correlation analysis."
            )

        else:

            corr_fig = (
                px.imshow(
                    correlation,

                    text_auto=".2f",

                    aspect="auto",

                    title=(
                        "Correlation of Daily "
                        "Pair Strategy Returns"
                    ),
                )
            )

            corr_fig.update_layout(
                height=600,
            )

            st.plotly_chart(
                corr_fig,
                use_container_width=True,
            )

    # ========================================================
    # WEIGHTS
    # ========================================================

    with portfolio_tabs[5]:

        st.subheader(
            "Portfolio Pair Weights"
        )

        weight_fig = go.Figure()

        for pair in weights.columns:

            weight_fig.add_trace(
                go.Scatter(
                    x=weights.index,

                    y=(
                        weights[
                            pair
                        ]
                        * 100
                    ),

                    name=pair,
                )
            )

        weight_fig.update_layout(
            xaxis_title="Date",

            yaxis_title=(
                "Portfolio allocation (%)"
            ),

            hovermode="x unified",

            height=550,
        )

        st.plotly_chart(
            weight_fig,
            use_container_width=True,
        )

    # ========================================================
    # RISK
    # ========================================================

    with portfolio_tabs[6]:

        st.subheader(
            "Portfolio Risk Statistics"
        )

        r1, r2, r3, r4 = (
            st.columns(4)
        )

        r1.metric(
            "95% daily VaR",
            f"{statistics['var_95']:.2%}",
        )

        r2.metric(
            "95% expected shortfall",
            f"{statistics['expected_shortfall_95']:.2%}",
        )

        r3.metric(
            "Maximum capital deployed",
            f"{statistics['maximum_capital_deployed']:.1%}",
        )

        r4.metric(
            "Selected pair strategies",
            len(selected),
        )

        st.caption(
            "VaR is the empirical 5th percentile of daily "
            "portfolio returns. Expected shortfall is the "
            "average return on observations at or below "
            "that threshold."
        )

    # ========================================================
    # RESEARCH DIAGNOSTICS
    # ========================================================

    with portfolio_tabs[7]:

        st.subheader("Eligibility & Opportunity Funnel")
        st.caption(
            "Diagnostic only. Historical eligibility is still determined block by "
            "block using information available before the next trading block."
        )

        if research_funnel.empty:
            st.info("No funnel diagnostics were produced for this run.")
        else:
            funnel_cols = [
                "tested_blocks",
                "cointegration_pass_blocks",
                "half_life_pass_blocks",
                "persistence_pass_blocks",
                "z_entry_blocks",
                "completed_trades",
            ]
            existing_cols = [c for c in funnel_cols if c in research_funnel.columns]
            totals = research_funnel[existing_cols].sum()

            tested_n = int(totals.get("tested_blocks", 0))
            coint_n = int(totals.get("cointegration_pass_blocks", 0))
            combined_n = int(totals.get("half_life_pass_blocks", 0))
            persist_n = int(totals.get("persistence_pass_blocks", 0))
            z_n = int(totals.get("z_entry_blocks", 0))
            trades_n = int(totals.get("completed_trades", 0))

            # The scanner now reports half-life by itself as a separate diagnostic.
            hl_only_n = 0
            raw_valid_n = 0
            for scan in scanner_results.values():
                if scan is None or scan.empty:
                    continue
                if "half_life_only_pass_blocks" in scan.columns:
                    hl_only_n += int(scan["half_life_only_pass_blocks"].fillna(0).sum())
                if "raw_valid_blocks" in scan.columns:
                    raw_valid_n += int(scan["raw_valid_blocks"].fillna(0).sum())

            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Pair-blocks tested", f"{tested_n:,}")
            f2.metric(
                "Cointegration passes",
                f"{coint_n:,}",
                help=f"{coint_n / tested_n:.1%} of tested blocks" if tested_n else None,
            )
            f3.metric(
                "Half-life-only passes",
                f"{hl_only_n:,}",
                help=(
                    f"Blocks satisfying 0 < HL ≤ {float(maxhl):g} days, "
                    "irrespective of cointegration."
                ),
            )
            f4.metric(
                "Cointegration + half-life",
                f"{combined_n:,}",
                help=(
                    f"{combined_n / coint_n:.1%} of cointegration passes also "
                    "satisfied the half-life rule."
                    if coint_n else None
                ),
            )

            g1, g2, g3 = st.columns(3)
            g1.metric(
                "Persistence passes",
                f"{persist_n:,}",
                help=(
                    f"{persist_n / combined_n:.1%} of current cointegration + "
                    "half-life passes also satisfied the past-only persistence rule."
                    if combined_n else None
                ),
            )
            g2.metric(
                "Eligible blocks reaching entry Z",
                f"{z_n:,}",
                help=f"{z_n / persist_n:.1%} of persistence-eligible blocks" if persist_n else None,
            )
            g3.metric("Completed trades", f"{trades_n:,}")

            if coint_n == combined_n and coint_n > 0:
                st.info(
                    "Cointegration and cointegration + half-life counts are identical "
                    "in this run. The counts are now calculated separately, so this "
                    "means every p-value-qualified block also satisfied the half-life rule."
                )

            st.markdown("#### Funnel by peer group")
            display_funnel = research_funnel.copy()

            display_funnel["Cointegration pass %"] = np.where(
                display_funnel["tested_blocks"] > 0,
                display_funnel["cointegration_pass_blocks"] / display_funnel["tested_blocks"],
                np.nan,
            )

            display_funnel["Coint + HL / coint %"] = np.where(
                display_funnel["cointegration_pass_blocks"] > 0,
                display_funnel["half_life_pass_blocks"] / display_funnel["cointegration_pass_blocks"],
                np.nan,
            )

            display_funnel["Persistence pass %"] = np.where(
                display_funnel["half_life_pass_blocks"] > 0,
                display_funnel["persistence_pass_blocks"] / display_funnel["half_life_pass_blocks"],
                np.nan,
            )

            display_funnel["Entry-opportunity %"] = np.where(
                display_funnel["persistence_pass_blocks"] > 0,
                display_funnel["z_entry_blocks"] / display_funnel["persistence_pass_blocks"],
                np.nan,
            )

            display_funnel = display_funnel.rename(
                columns={
                    "industry": "Peer group",
                    "pair_models": "Pair models",
                    "tested_blocks": "Tested blocks",
                    "cointegration_pass_blocks": "Cointegration passes",
                    "half_life_pass_blocks": "Cointegration + half-life",
                    "persistence_pass_blocks": "Persistence passes",
                    "z_entry_blocks": "Entry-Z blocks",
                    "completed_trades": "Completed trades",
                }
            )

            st.dataframe(
                display_funnel.style.format(
                    {
                        "Cointegration pass %": "{:.1%}",
                        "Coint + HL / coint %": "{:.1%}",
                        "Persistence pass %": "{:.1%}",
                        "Entry-opportunity %": "{:.1%}",
                    },
                    na_rep="—",
                ),
                use_container_width=True,
            )

        st.divider()
        st.subheader("Trade Outcome Diagnostics")

        if not isinstance(trade_economics, dict):
            # Backward compatibility with older engine output.
            if (
                isinstance(trade_economics, pd.DataFrame)
                and {"metric", "value"}.issubset(trade_economics.columns)
            ):
                label_map = {
                    "Completed trades": "completed_trades",
                    "Overall win rate": "win_rate",
                    "Mean-reversion exit rate": "mean_reversion_rate",
                    "Average winner": "average_winner",
                    "Average loser": "average_loser",
                    "Payoff ratio": "payoff_ratio",
                    "Profit factor": "profit_factor",
                    "Gross P&L": "gross_pnl",
                    "Transaction costs": "transaction_cost",
                    "Borrow costs": "borrow_cost",
                    "Net P&L": "net_pnl",
                    "Average holding days": "average_holding_days",
                }
                trade_economics = {
                    label_map.get(str(row["metric"]), str(row["metric"])): row["value"]
                    for _, row in trade_economics.iterrows()
                }
            else:
                trade_economics = {}

        def _metric_value(key, pct=False, money=False, decimals=2):
            value = trade_economics.get(key, np.nan)
            try:
                value = float(value)
            except Exception:
                return "—"
            if not np.isfinite(value):
                return "—"
            if pct:
                return f"{value:.1%}"
            if money:
                return f"${value:,.2f}"
            if key == "completed_trades":
                return f"{int(value):,}"
            return f"{value:.{decimals}f}"

        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Completed trades", _metric_value("completed_trades"))
        t2.metric("Overall win rate", _metric_value("win_rate", pct=True))
        t3.metric("Mean-reversion exit rate", _metric_value("mean_reversion_rate", pct=True))
        t4.metric("Reached |Z| ≤ 1", _metric_value("convergence_rate", pct=True))

        u1, u2, u3, u4 = st.columns(4)
        u1.metric("Average winner", _metric_value("average_winner", money=True))
        u2.metric("Average loser", _metric_value("average_loser", money=True))
        u3.metric("Payoff ratio", _metric_value("payoff_ratio"))
        u4.metric("Profit factor", _metric_value("profit_factor"))

        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Gross P&L", _metric_value("gross_pnl", money=True))
        v2.metric("Transaction costs", _metric_value("transaction_cost", money=True))
        v3.metric("Borrow costs", _metric_value("borrow_cost", money=True))
        v4.metric("Net P&L", _metric_value("net_pnl", money=True))

        w1, w2, w3 = st.columns(3)
        w1.metric("Average holding days", _metric_value("average_holding_days"))
        w2.metric("Average Z-space MFE", _metric_value("average_z_mfe"))
        w3.metric("Average Z-space MAE", _metric_value("average_z_mae"))

        if trade_outcomes.empty:
            st.info("No completed-trade outcome diagnostics are available.")
        else:
            outcome_display = trade_outcomes.copy()

            rename_map = {
                "exit_reason": "Exit reason",
                "trades": "Trades",
                "trade_share": "% of completed trades",
                "win_rate": "Win rate",
                "avg_net_pnl": "Average net P&L",
                "median_net_pnl": "Median net P&L",
                "avg_holding_days": "Average holding days",
                "avg_z_mfe": "Average Z MFE",
                "avg_z_mae": "Average Z MAE",
                "convergence_rate": "Reached |Z| ≤ 1",
                "total_net_pnl": "Total net P&L",
            }
            outcome_display = outcome_display.rename(columns=rename_map)

            formatters = {}
            for c in ["% of completed trades", "Win rate", "Reached |Z| ≤ 1"]:
                if c in outcome_display.columns:
                    formatters[c] = "{:.1%}"
            for c in ["Average net P&L", "Median net P&L", "Total net P&L"]:
                if c in outcome_display.columns:
                    formatters[c] = "${:,.2f}"
            for c in ["Average holding days", "Average Z MFE", "Average Z MAE"]:
                if c in outcome_display.columns:
                    formatters[c] = "{:.2f}"

            st.dataframe(
                outcome_display.style.format(formatters, na_rep="—"),
                use_container_width=True,
            )

            st.caption(
                "Z-space MFE measures the largest movement toward equilibrium after "
                "entry; Z-space MAE measures the largest movement farther away. "
                "These are ex-post diagnostics and do not affect historical trading."
            )

    # ========================================================
    # INDUSTRY MEAN-REVERSION ANALYSIS
    # ========================================================

    with portfolio_tabs[8]:

        st.subheader("Industry Mean-Reversion Analysis")
        st.caption(
            "This is an ex-post diagnostic of the current backtest. It is designed "
            "to identify which peer groups actually produced convergence after entry, "
            "not to alter the original 17-industry historical decisions."
        )

        if industry_mean_reversion.empty:
            st.info("No industry-level trade diagnostics were produced.")
        else:
            industry_display = industry_mean_reversion.copy()

            rename_map = {
                "industry": "Industry",
                "trades": "Trades",
                "mean_reversion_exits": "MR exits",
                "mean_reversion_rate": "MR exit rate",
                "hard_stop_rate": "Hard-stop rate",
                "time_stop_rate": "Time-stop rate",
                "cointegration_break_rate": "Cointegration-break rate",
                "overall_win_rate": "Overall win rate",
                "mean_reversion_win_rate": "MR win rate",
                "avg_mean_reversion_pnl": "Average MR P&L",
                "total_mean_reversion_pnl": "Total MR P&L",
                "convergence_rate": "Reached |Z| ≤ 1",
                "avg_z_mfe": "Average Z MFE",
                "avg_z_mae": "Average Z MAE",
                "gross_pnl": "Gross P&L",
                "transaction_cost": "Transaction costs",
                "borrow_cost": "Borrow costs",
                "net_pnl": "Net P&L",
            }

            industry_display = industry_display.rename(columns=rename_map)

            pct_cols = [
                "MR exit rate",
                "Hard-stop rate",
                "Time-stop rate",
                "Cointegration-break rate",
                "Overall win rate",
                "MR win rate",
                "Reached |Z| ≤ 1",
            ]
            money_cols = [
                "Average MR P&L",
                "Total MR P&L",
                "Gross P&L",
                "Transaction costs",
                "Borrow costs",
                "Net P&L",
            ]

            formatters = {
                c: "{:.1%}" for c in pct_cols if c in industry_display.columns
            }
            formatters.update({
                c: "${:,.2f}" for c in money_cols if c in industry_display.columns
            })
            for c in ["Average Z MFE", "Average Z MAE"]:
                if c in industry_display.columns:
                    formatters[c] = "{:.2f}"

            st.dataframe(
                industry_display.style.format(formatters, na_rep="—"),
                use_container_width=True,
            )

            st.caption(
                "MR exit rate uses the strategy's strict |Z| exit threshold. "
                "Reached |Z| ≤ 1 is the broader convergence diagnostic and can reveal "
                "trades that substantially reverted without reaching |Z| ≤ 0.25."
            )

    # ========================================================
    # EX-POST INDUSTRY SUBSET PORTFOLIO
    # ========================================================

    with portfolio_tabs[9]:

        st.subheader("Industry-Selected Research Portfolio")

        st.warning(
            "Industries chosen from the table above are selected using this same "
            "backtest period. Results here are therefore in-sample research results, "
            "not untouched out-of-sample evidence."
        )

        if industry_mean_reversion.empty:
            st.info("Run must contain industry-level trade diagnostics first.")
        else:
            industry_options = industry_mean_reversion["industry"].dropna().astype(str).tolist()

            default_n = min(5, len(industry_options))
            default_subset = industry_options[:default_n]

            chosen_industries = st.multiselect(
                "Industries included in research subset",
                industry_options,
                default=default_subset,
                key="mr_industry_subset",
                help=(
                    "The table is displayed by realized mean-reversion behavior. "
                    "You remain in control of which industries enter this research portfolio."
                ),
            )

            subset_allocation = st.selectbox(
                "Subset allocation method",
                ["Equal-weight active pairs", "Inverse-volatility"],
                index=0,
                key="mr_subset_allocation",
            )

            if chosen_industries:
                subset_result = build_industry_subset_portfolio(
                    strategies=strategies,
                    ledgers=result.get("ledgers", {}),
                    selected=selected,
                    industries=chosen_industries,
                    allocation_method=subset_allocation,
                    max_pair_weight=max_pair_weight_pct / 100,
                    portfolio_gross=portfolio_gross_pct / 100,
                )

                if subset_result is None:
                    st.info("The selected industries did not contain an executable pair strategy.")
                else:
                    subset_stats = subset_result["statistics"]

                    st.markdown("#### Base portfolio vs selected-industry portfolio")

                    comparison = pd.DataFrame(
                        {
                            "All selected industries": {
                                "Annual return": statistics["annual_return"],
                                "Annual volatility": statistics["annual_volatility"],
                                "Sharpe": statistics["sharpe"],
                                "Sortino": statistics["sortino"],
                                "Maximum drawdown": statistics["max_drawdown"],
                                "Calmar": statistics["calmar"],
                                "Average active pairs": statistics["average_active_pairs"],
                                "Average capital deployed": statistics["average_capital_deployed"],
                            },
                            "Industry-selected research portfolio": {
                                "Annual return": subset_stats["annual_return"],
                                "Annual volatility": subset_stats["annual_volatility"],
                                "Sharpe": subset_stats["sharpe"],
                                "Sortino": subset_stats["sortino"],
                                "Maximum drawdown": subset_stats["max_drawdown"],
                                "Calmar": subset_stats["calmar"],
                                "Average active pairs": subset_stats["average_active_pairs"],
                                "Average capital deployed": subset_stats["average_capital_deployed"],
                            },
                        }
                    )

                    st.dataframe(
                        comparison.style.format(
                            {
                                "All selected industries": "{:.4f}",
                                "Industry-selected research portfolio": "{:.4f}",
                            },
                            na_rep="—",
                        ),
                        use_container_width=True,
                    )

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Subset annual return", f"{subset_stats['annual_return']:.2%}")
                    c2.metric("Subset volatility", f"{subset_stats['annual_volatility']:.2%}")
                    c3.metric("Subset Sharpe", f"{subset_stats['sharpe']:.2f}")
                    c4.metric("Subset max drawdown", f"{subset_stats['max_drawdown']:.2%}")

                    subset_portfolio = subset_result["portfolio"]

                    compare_fig = go.Figure()
                    compare_fig.add_trace(
                        go.Scatter(
                            x=portfolio.index,
                            y=portfolio["equity"],
                            name="Base portfolio",
                        )
                    )
                    compare_fig.add_trace(
                        go.Scatter(
                            x=subset_portfolio.index,
                            y=subset_portfolio["equity"],
                            name="Industry-selected portfolio",
                        )
                    )
                    compare_fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Portfolio value (USD)",
                        hovermode="x unified",
                        height=520,
                    )
                    st.plotly_chart(compare_fig, use_container_width=True)

                    st.caption(
                        "This comparison holds the trading rules fixed. Any difference "
                        "comes from the selected industry universe and allocation method, "
                        "not from changing the entry, exit, cointegration, half-life or "
                        "persistence thresholds."
                    )
            else:
                st.info("Select at least one industry for the research subset.")

    # ========================================================
    # WALK-FORWARD INDUSTRY QUALITY
    # ========================================================

    with portfolio_tabs[10]:
        st.subheader("Walk-Forward Industry Quality")
        st.caption(
            "Diagnostic only. Predictor snapshots are refreshed every 21 trading days, "
            "using only industry trades completed before each snapshot. Outcomes are "
            "measured over the following 63 trading days. This does not alter pair "
            "eligibility or portfolio returns."
        )

        wf = industry_quality_walkforward.copy()
        if wf.empty:
            st.info("No walk-forward industry-quality observations were produced for this run.")
        else:
            min_prior = st.number_input(
                "Minimum prior completed trades shown", min_value=1, max_value=100,
                value=5, step=1, key="wf_quality_min_prior"
            )
            view = wf[wf["prior_trades"] >= int(min_prior)].copy()
            if view.empty:
                st.info("No observations meet that prior-trade minimum. Try a smaller value.")
            else:
                target = "next_63d_pnl_per_trade"
                def _corr(col):
                    z = view[[col, target]].replace([np.inf, -np.inf], np.nan).dropna()
                    return float(z[col].corr(z[target])) if len(z) >= 3 else np.nan

                c1, c2, c3 = st.columns(3)
                c1.metric("Prior MFE/MAE → next 63d P&L/trade", f"{_corr('prior_mfe_mae_ratio'):.3f}")
                c2.metric("Prior convergence → next 63d P&L/trade", f"{_corr('prior_convergence_rate'):.3f}")
                c3.metric("Prior hard-stop → next 63d P&L/trade", f"{_corr('prior_hard_stop_rate'):.3f}")

                st.caption(
                    "Positive MFE/MAE or convergence correlation is directionally supportive; "
                    "a negative hard-stop correlation is directionally supportive. Treat these "
                    "as research diagnostics, not evidence of a profitable filter by themselves."
                )
                display_cols = [
                    "industry", "block_start", "block_end", "prior_trades",
                    "prior_mfe_mae_ratio", "prior_convergence_rate",
                    "prior_hard_stop_rate", "next_63d_trades",
                    "next_63d_pnl_per_trade", "next_63d_win_rate",
                    "next_63d_convergence_rate",
                ]
                st.dataframe(
                    view[display_cols].sort_values(["block_start", "industry"]),
                    use_container_width=True,
                )


    # ========================================================
    # PAIR-BLOCK DIAGNOSTICS
    # ========================================================
    with portfolio_tabs[11]:
        st.subheader("Pair-Block Walk-Forward Diagnostics")
        st.caption(
            "One row per pair per tested trading block. Model statistics are estimated "
            "from the preceding formation window. This table is for chronological, "
            "no-look-ahead research and does not change portfolio eligibility."
        )
        if pair_block_diagnostics.empty:
            st.info("No pair-block diagnostics were produced for this run.")
        else:
            st.metric("Pair-block observations", f"{len(pair_block_diagnostics):,}")
            st.dataframe(pair_block_diagnostics, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Pair-Block Diagnostics CSV",
                data=pair_block_diagnostics.to_csv(index=False).encode("utf-8"),
                file_name="pair_block_diagnostics.csv",
                mime="text/csv",
                key="download_pair_block_diagnostics",
            )

    # ========================================================
    # FULL TRADE LEDGER
    # ========================================================
    with portfolio_tabs[12]:
        st.subheader("Full Individual Trade Ledger")
        st.caption(
            "Every completed pair trade retained by the portfolio engine, including "
            "entry/exit information, costs, net P&L and available MFE/MAE diagnostics."
        )
        if full_trade_ledger.empty:
            st.info("No completed trades were recorded for this run.")
        else:
            st.metric("Completed trades", f"{len(full_trade_ledger):,}")
            st.dataframe(full_trade_ledger, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Full Trade Ledger CSV",
                data=full_trade_ledger.to_csv(index=False).encode("utf-8"),
                file_name="full_trade_ledger.csv",
                mime="text/csv",
                key="download_full_trade_ledger",
            )

    st.stop()



# ============================================================
# US PAIR SCANNER
# ============================================================

if research_mode == "US Pair Scanner":

    st.header(
        f"{scanner_industry} Pair Scanner"
    )

    st.caption(
        f"Evaluating {len(scanner_pairs)} within-industry "
        f"pair combinations among {len(scanner_tickers)} equities."
    )

    if not run:

        st.info(
            "Click **Run Sector Scanner**."
        )

        st.stop()

    try:

        with st.spinner(
            f"Evaluating {len(scanner_pairs)} pairs..."
        ):

            scanner = run_pair_scanner(
                industry=scanner_industry,
                start=start,
                end=end,
                formation_window=int(formation),
                trading_window=int(block),
                z_window=int(zwin),
                coint_threshold=float(pth),
                max_half_life=float(maxhl),
                z_entry=float(zentry),
                z_exit=float(zexit),
                z_stop=float(zstop),
                gross_exposure=1.0,
                transaction_cost_bps=float(tc),
                annual_borrow_bps=float(bc),
            )

    except Exception as exc:

        st.error(
            f"Scanner failed: {exc}"
        )

        st.stop()

    if scanner.empty:

        st.warning(
            "No scanner results."
        )

        st.stop()

    # --------------------------------------------------------
    # STATISTICAL SCREEN
    # --------------------------------------------------------

    st.subheader(
        f"{scanner_industry} Statistical Pair Screen"
    )

    statistical_display = scanner[
        [
            "pair",
            "total_blocks",
            "valid_blocks",
            "valid_fraction",
            "validity_regimes",
            "valid_median_pvalue",
            "valid_median_half_life",
            "valid_median_r_squared",
            "trades",
        ]
    ].copy()

    statistical_display[
        "valid_fraction"
    ] *= 100

    statistical_display = (
        statistical_display.rename(
            columns={
                "pair":
                    "Pair",

                "total_blocks":
                    "Walk-forward blocks",

                "valid_blocks":
                    "Valid blocks",

                "valid_fraction":
                    "Valid blocks (%)",

                "validity_regimes":
                    "Distinct valid regimes",

                "valid_median_pvalue":
                    "Median p-value when valid",

                "valid_median_half_life":
                    "Median half-life when valid (trading days)",

                "valid_median_r_squared":
                    "Median R² when valid",

                "trades":
                    "Completed trades",
            }
        )
    )

    st.dataframe(
        statistical_display.style.format(
            {
                "Valid blocks (%)":
                    "{:.1f}%",

                "Median p-value when valid":
                    "{:.3f}",

                "Median half-life when valid (trading days)":
                    "{:.1f}",

                "Median R² when valid":
                    "{:.3f}",
            },

            na_rep="—",
        ),

        use_container_width=True,
    )

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    st.subheader(
        "Out-of-Sample Trading Performance"
    )

    performance_display = scanner[
        [
            "pair",
            "trades",
            "win_rate",
            "avg_holding_days",
            "annual_return",
            "annual_volatility",
            "sharpe",
            "max_drawdown",
            "time_in_market",
            "net_pnl",
        ]
    ].copy()

    for col in [
        "win_rate",
        "annual_return",
        "annual_volatility",
        "max_drawdown",
        "time_in_market",
    ]:

        performance_display[
            col
        ] *= 100

    performance_display = (
        performance_display.rename(
            columns={
                "pair":
                    "Pair",

                "trades":
                    "Completed trades",

                "win_rate":
                    "Win rate (%)",

                "avg_holding_days":
                    "Average holding period (trading days)",

                "annual_return":
                    "Annual return (%)",

                "annual_volatility":
                    "Annualized volatility (%)",

                "sharpe":
                    "Sharpe ratio",

                "max_drawdown":
                    "Maximum drawdown (%)",

                "time_in_market":
                    "Time in market (%)",

                "net_pnl":
                    "Net P&L (USD)",
            }
        )
    )

    st.dataframe(
        performance_display.style.format(
            {
                "Win rate (%)":
                    "{:.1f}%",

                "Average holding period (trading days)":
                    "{:.1f}",

                "Annual return (%)":
                    "{:.2f}%",

                "Annualized volatility (%)":
                    "{:.2f}%",

                "Sharpe ratio":
                    "{:.2f}",

                "Maximum drawdown (%)":
                    "{:.2f}%",

                "Time in market (%)":
                    "{:.1f}%",

                "Net P&L (USD)":
                    "${:,.2f}",
            },

            na_rep="—",
        ),

        use_container_width=True,
    )

    st.stop()


    # ========================================================
    # RESEARCH DIAGNOSTICS
    # ========================================================

    with portfolio_tabs[7]:

        st.subheader("Eligibility & Opportunity Funnel")
        st.caption(
            "Diagnostic only — none of these aggregate results are used to select "
            "historical pairs. Each block decision still uses only information "
            "available before that out-of-sample trading block."
        )

        if research_funnel.empty:
            st.info("No funnel diagnostics were produced for this run.")
        else:
            funnel_cols = [
                "tested_blocks",
                "cointegration_pass_blocks",
                "half_life_pass_blocks",
                "persistence_pass_blocks",
                "z_entry_blocks",
                "completed_trades",
            ]
            totals = research_funnel[funnel_cols].sum()
            tested_n = int(totals["tested_blocks"])
            coint_n = int(totals["cointegration_pass_blocks"])
            hl_n = int(totals["half_life_pass_blocks"])
            persist_n = int(totals["persistence_pass_blocks"])
            z_n = int(totals["z_entry_blocks"])
            trades_n = int(totals["completed_trades"])

            f1, f2, f3 = st.columns(3)
            f1.metric("Pair-blocks tested", f"{tested_n:,}")
            f2.metric(
                "Cointegration passes",
                f"{coint_n:,}",
                help=f"{(coint_n / tested_n):.1%} of tested blocks" if tested_n else None,
            )
            f3.metric(
                "Half-life passes",
                f"{hl_n:,}",
                help=f"{(hl_n / coint_n):.1%} of cointegration passes" if coint_n else None,
            )

            g1, g2, g3 = st.columns(3)
            g1.metric(
                "Persistence passes",
                f"{persist_n:,}",
                help=f"{(persist_n / hl_n):.1%} of cointegration + half-life passes" if hl_n else None,
            )
            g2.metric(
                "Eligible blocks reaching entry Z",
                f"{z_n:,}",
                help=f"{(z_n / persist_n):.1%} of persistence-eligible blocks" if persist_n else None,
            )
            g3.metric("Completed trades", f"{trades_n:,}")

            st.markdown("#### Funnel by peer group")
            display_funnel = research_funnel.copy()
            display_funnel["Cointegration pass %"] = np.where(
                display_funnel["tested_blocks"] > 0,
                display_funnel["cointegration_pass_blocks"] / display_funnel["tested_blocks"],
                np.nan,
            )
            display_funnel["Persistence pass %"] = np.where(
                display_funnel["half_life_pass_blocks"] > 0,
                display_funnel["persistence_pass_blocks"] / display_funnel["half_life_pass_blocks"],
                np.nan,
            )
            display_funnel["Entry-opportunity %"] = np.where(
                display_funnel["persistence_pass_blocks"] > 0,
                display_funnel["z_entry_blocks"] / display_funnel["persistence_pass_blocks"],
                np.nan,
            )
            display_funnel = display_funnel.rename(columns={
                "industry": "Peer group",
                "pair_models": "Pair models",
                "tested_blocks": "Tested blocks",
                "cointegration_pass_blocks": "Cointegration passes",
                "half_life_pass_blocks": "Half-life passes",
                "persistence_pass_blocks": "Persistence passes",
                "z_entry_blocks": "Entry-Z blocks",
                "completed_trades": "Completed trades",
            })
            st.dataframe(
                display_funnel.style.format({
                    "Cointegration pass %": "{:.1%}",
                    "Persistence pass %": "{:.1%}",
                    "Entry-opportunity %": "{:.1%}",
                }, na_rep="—"),
                use_container_width=True,
            )

            st.caption(
                "Interpret the funnel sequentially: cointegration asks whether a "
                "formation-period equilibrium relationship exists; half-life removes "
                "relationships that revert too slowly; persistence asks whether that "
                "relationship was repeatedly present in prior windows; Entry-Z blocks "
                "show whether an eligible spread actually produced a tradable excursion."
            )


# ============================================================
# SINGLE PAIR ANALYSIS
# ============================================================

if not run:

    st.info(
        "Choose two equities and click **Calibrate Pair**."
    )

    st.stop()


if a == b:

    st.error(
        "Equity A and Equity B must be different."
    )

    st.stop()


# ============================================================
# DATA
# ============================================================

with st.spinner(
    "Downloading market data..."
):

    ohlc = download_pair(
        a,
        b,
        start,
        end,
        market,
    )


close_a = ohlc[
    "close_a"
]

close_b = ohlc[
    "close_b"
]


# ============================================================
# WALK FORWARD
# ============================================================

with st.spinner(
    "Running walk-forward model..."
):

    model = build_walkforward(
        close_a,
        close_b,
        int(formation),
        int(block),
        int(zwin),
        float(pth),
        float(maxhl),
    )


model = model.join(
    ohlc[
        [
            "open_a",
            "open_b",
            "close_a",
            "close_b",
            "volume_a",
            "volume_b",
        ]
    ]
)


# ============================================================
# MARKET BETA
# ============================================================

benchmark = (
    "^GSPC"
    if market == "US"
    else "^NSEI"
)


mkt = yf.download(
    benchmark,
    start=start,
    end=end + timedelta(days=1),
    auto_adjust=True,
    progress=False,
)


if isinstance(
    mkt.columns,
    pd.MultiIndex,
):

    market_close = (
        mkt["Close"].iloc[:, 0]
    )

else:

    market_close = (
        mkt["Close"]
    )


market_returns = (
    market_close
    .reindex(model.index)
    .pct_change()
)


beta_a = rolling_market_beta(
    close_a
    .pct_change()
    .reindex(model.index),

    market_returns,

    126,
)


beta_b = rolling_market_beta(
    close_b
    .pct_change()
    .reindex(model.index),

    market_returns,

    126,
)


# ============================================================
# EXECUTION
# ============================================================

daily, ledger = run_execution(
    model.dropna(
        subset=[
            "open_a",
            "open_b",
            "close_a",
            "close_b",
        ]
    ),

    gross_exposure=gross / 100,

    sizing_mode=sizing,

    transaction_cost_bps=tc,

    annual_borrow_bps=bc,

    z_entry=zentry,

    z_exit=zexit,

    z_stop=zstop,

    z_entry_max=z_entry_max,

    beta_a=beta_a,

    beta_b=beta_b,
)


perf = stats(
    daily["net_return"],
    daily["position"],
)


# ============================================================
# CURRENT ORIENTATION
# ============================================================

formation_df = pd.concat(
    [
        close_a,
        close_b,
    ],
    axis=1,
).dropna().tail(
    int(formation)
)


ab = fit_orientation(
    formation_df.iloc[:, 0],
    formation_df.iloc[:, 1],
)


ba2 = fit_orientation(
    formation_df.iloc[:, 1],
    formation_df.iloc[:, 0],
)


valid = model[
    model["pair_valid"]
    &
    model["zscore"].notna()
]


latest = (
    valid.iloc[-1]
    if len(valid)
    else None
)


if latest is not None:

    pscore = pair_quality(
        latest[
            "cointegration_pvalue"
        ],

        latest[
            "half_life"
        ],

        latest[
            "r_squared"
        ],
    )

else:

    pscore = 0


# ============================================================
# SINGLE PAIR SUMMARY
# ============================================================

st.subheader(
    "Strategy Summary"
)


s1, s2, s3, s4, s5 = (
    st.columns(5)
)


s1.metric(
    "Annual return (% p.a.)",
    f"{perf['annual_return']:.2%}",
)

s2.metric(
    "Annualized volatility (% p.a.)",
    f"{perf['annual_volatility']:.2%}",
)

s3.metric(
    "Sharpe ratio (annualized)",
    f"{perf['sharpe']:.2f}",
)

s4.metric(
    "Maximum drawdown (%)",
    f"{perf['max_drawdown']:.2%}",
)

s5.metric(
    "Time in market (%)",
    f"{perf['time_in_market']:.1%}",
)


st.metric(
    "Pair quality score (/100)",
    f"{pscore:.0f}/100",
)


# ============================================================
# SINGLE PAIR TABS
# ============================================================

tabs = st.tabs(
    [
        "Orientation",
        "Signal",
        "Backtest",
        "Trade Ledger",
        "Sensitivity",
        "Stability",
    ]
)


# ============================================================
# ORIENTATION
# ============================================================

with tabs[0]:

    comp = pd.DataFrame(
        {
            "A on B": {
                "Cointegration p-value":
                    ab[
                        "cointegration_pvalue"
                    ],

                "Half-life (trading days)":
                    ab[
                        "half_life"
                    ],

                "Hedge ratio β":
                    ab[
                        "beta"
                    ],

                "R²":
                    ab[
                        "r_squared"
                    ],
            },

            "B on A": {
                "Cointegration p-value":
                    ba2[
                        "cointegration_pvalue"
                    ],

                "Half-life (trading days)":
                    ba2[
                        "half_life"
                    ],

                "Hedge ratio β":
                    ba2[
                        "beta"
                    ],

                "R²":
                    ba2[
                        "r_squared"
                    ],
            },
        }
    )

    st.dataframe(
        comp,
        use_container_width=True,
    )


# ============================================================
# SIGNAL
# ============================================================

with tabs[1]:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=model.index,
            y=model["zscore"],
            name="Spread Z-score",
        )
    )

    fig.add_hline(
        y=zentry,
        line_dash="dash",
    )

    fig.add_hline(
        y=-zentry,
        line_dash="dash",
    )

    fig.add_hline(
        y=zstop,
        line_dash="dot",
    )

    fig.add_hline(
        y=-zstop,
        line_dash="dot",
    )

    fig.add_hline(
        y=zexit,
        line_dash="dot",
    )

    fig.add_hline(
        y=-zexit,
        line_dash="dot",
    )

    fig.update_layout(
        title="Walk-Forward Spread Signal",

        xaxis_title="Date",

        yaxis_title=(
            "Spread Z-score "
            "(standard deviations)"
        ),

        hovermode="x unified",

        height=520,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# BACKTEST
# ============================================================

with tabs[2]:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=daily["equity"],
            name="Portfolio equity",
        )
    )

    fig.update_layout(
        xaxis_title="Date",

        yaxis_title=(
            f"Portfolio value ({currency})"
        ),

        hovermode="x unified",

        height=500,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# LEDGER
# ============================================================

with tabs[3]:

    if ledger.empty:

        st.info(
            "No completed trades."
        )

    else:

        st.dataframe(
            ledger,
            use_container_width=True,
        )

        l1, l2, l3, l4 = (
            st.columns(4)
        )

        l1.metric(
            f"Gross P&L ({currency})",
            f"{ledger['gross_pnl'].sum():,.2f}",
        )

        l2.metric(
            f"Transaction costs ({currency})",
            f"{ledger['transaction_cost'].sum():,.2f}",
        )

        l3.metric(
            f"Short-borrow costs ({currency})",
            f"{ledger['borrow_cost'].sum():,.2f}",
        )

        l4.metric(
            f"Net P&L ({currency})",
            f"{ledger['net_pnl'].sum():,.2f}",
        )


# ============================================================
# SENSITIVITY
# ============================================================

with tabs[4]:

    sens = run_sensitivity(
        model.dropna(
            subset=[
                "open_a",
                "open_b",
                "close_a",
                "close_b",
            ]
        ),

        dict(
            gross_exposure=gross / 100,

            sizing_mode=sizing,

            transaction_cost_bps=tc,

            annual_borrow_bps=bc,

            z_stop=zstop,

            beta_a=beta_a,

            beta_b=beta_b,
        ),
    )

    pivot = sens.pivot(
        index="entry_z",
        columns="exit_z",
        values="sharpe",
    )

    pivot.index.name = (
        "Entry threshold |Z|"
    )

    pivot.columns.name = (
        "Exit threshold |Z|"
    )

    st.dataframe(
        pivot.style.format(
            "{:.2f}"
        ),

        use_container_width=True,
    )


# ============================================================
# STABILITY
# ============================================================

with tabs[5]:

    pfig = go.Figure()

    pfig.add_trace(
        go.Scatter(
            x=model.index,

            y=model[
                "cointegration_pvalue"
            ],

            name="Cointegration p-value",
        )
    )

    pfig.add_hline(
        y=pth,
        line_dash="dash",
    )

    pfig.update_layout(
        title=(
            "Engle-Granger "
            "Cointegration Stability"
        ),

        xaxis_title="Date",

        yaxis_title=(
            "Cointegration p-value "
            "(dimensionless)"
        ),

        height=420,
    )

    st.plotly_chart(
        pfig,
        use_container_width=True,
    )


    hfig = go.Figure()

    hfig.add_trace(
        go.Scatter(
            x=model.index,

            y=model[
                "half_life"
            ],

            name="Half-life",
        )
    )

    hfig.add_hline(
        y=maxhl,
        line_dash="dash",
    )

    hfig.update_layout(
        title=(
            "Mean-Reversion "
            "Half-Life Through Time"
        ),

        xaxis_title="Date",

        yaxis_title=(
            "Estimated half-life "
            "(trading days)"
        ),

        height=420,
    )

    st.plotly_chart(
        hfig,
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Research tool only. Historical statistical relationships "
    "can break, execution costs and short availability can change, "
    "and backtested performance does not guarantee future results."
)