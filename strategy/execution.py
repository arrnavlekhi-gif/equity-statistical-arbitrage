from dataclasses import dataclass, asdict







import numpy as np



import pandas as pd







from .sizing import dollar_neutral, hedge_ratio, beta_neutral











@dataclass



class Trade:



    entry_signal_date: object



    entry_date: object



    exit_signal_date: object



    exit_date: object







    direction: str



    orientation: str







    entry_z: float



    exit_z: float







    entry_price_a: float



    entry_price_b: float



    exit_price_a: float



    exit_price_b: float







    shares_a: float



    shares_b: float







    gross_pnl: float = 0.0



    transaction_cost: float = 0.0



    borrow_cost: float = 0.0



    net_pnl: float = 0.0







    holding_days: int = 0



    exit_reason: str | None = None











def run_execution(



    df,



    capital=100000,



    gross_exposure=1.0,



    sizing_mode="Hedge-ratio weighted",



    transaction_cost_bps=5,



    annual_borrow_bps=50,



    z_entry=2.0,



    z_exit=0.25,



    z_stop=3.5,




    z_entry_max=None,
    recent_equilibrium_max_days=None,
    beta_a=1.0,



    beta_b=1.0,



):







    # ---------------------------------------------------------



    # INITIAL STATE



    # ---------------------------------------------------------







    state = 0



    current = None



    ledger = []



    pending = None







    equity = float(capital)







    # Previous mark prices used for mark-to-market P&L.



    prev_mark_a = None



    prev_mark_b = None







    # Number of trading observations the current position



    # has been held for.



    trading_days_held = 0

    # Most recent row where |Z| <= 1.5; used only when the filter is enabled.
    last_abs_z_le_1_5_pos = None







    daily = pd.DataFrame(index=df.index)







    for c in [



        "position",



        "weight_a",



        "weight_b",



        "gross_exposure",



        "net_exposure",



        "gross_pnl",



        "transaction_cost",



        "borrow_cost",



        "net_pnl",



        "equity",



    ]:



        daily[c] = 0.0







    # ---------------------------------------------------------



    # HELPER FOR ROLLING BETA SERIES



    # ---------------------------------------------------------







    def bval(x, idx):







        if isinstance(x, pd.Series):







            v = x.reindex(df.index).loc[idx]







            return 1.0 if not np.isfinite(v) else float(v)







        return float(x)







    # ---------------------------------------------------------



    # MAIN BACKTEST LOOP



    # ---------------------------------------------------------







    for i, (idx, row) in enumerate(df.iterrows()):







        day_gross_pnl = 0.0



        day_transaction_cost = 0.0



        day_borrow_cost = 0.0







        # =====================================================



        # 1. EXECUTE PENDING ORDER AT TODAY'S OPEN



        # =====================================================







        if pending is not None:







            # -------------------------------------------------



            # ENTER POSITION



            # -------------------------------------------------







            if pending["type"] == "ENTER" and state == 0:







                direction = pending["direction"]







                if sizing_mode == "Dollar neutral":







                    wa, wb = dollar_neutral(



                        direction,



                        gross_exposure,



                    )







                elif sizing_mode == "Beta neutral":







                    wa, wb = beta_neutral(



                        direction,



                        bval(beta_a, idx),



                        bval(beta_b, idx),



                        gross_exposure,



                    )







                else:







                    wa, wb = hedge_ratio(

                        direction,

                        pending["beta"],

                        gross_exposure,

                        pending["orientation"],

                    )







                # Dollar allocation at entry.



                notional_a = equity * wa



                notional_b = equity * wb







                # Shares established at today's OPEN.



                shares_a = notional_a / float(row.open_a)



                shares_b = notional_b / float(row.open_b)







                entry_transaction_cost = (



                    abs(notional_a) + abs(notional_b)



                ) * transaction_cost_bps / 10000







                current = Trade(



                    entry_signal_date=pending["signal_date"],



                    entry_date=idx,



                    exit_signal_date=None,



                    exit_date=None,



                    direction=(



                        "LONG SPREAD"



                        if direction == 1



                        else "SHORT SPREAD"



                    ),



                    orientation=pending["orientation"],



                    entry_z=pending["z"],



                    exit_z=np.nan,



                    entry_price_a=float(row.open_a),



                    entry_price_b=float(row.open_b),



                    exit_price_a=np.nan,



                    exit_price_b=np.nan,



                    shares_a=shares_a,



                    shares_b=shares_b,



                    transaction_cost=entry_transaction_cost,



                )







                equity -= entry_transaction_cost







                day_transaction_cost += entry_transaction_cost







                state = direction







                # The position begins at today's OPEN.



                prev_mark_a = float(row.open_a)



                prev_mark_b = float(row.open_b)







                trading_days_held = 0







            # -------------------------------------------------



            # EXIT POSITION



            # -------------------------------------------------







            elif (



                pending["type"] == "EXIT"



                and state != 0



                and current is not None



            ):







                exit_a = float(row.open_a)



                exit_b = float(row.open_b)







                # IMPORTANT:



                # Only mark from the previous close to today's



                # exit open here.



                #



                # We DO NOT add full entry-to-exit P&L again.



                exit_mark_pnl = (



                    (exit_a - prev_mark_a) * current.shares_a



                    + (exit_b - prev_mark_b) * current.shares_b



                )







                equity += exit_mark_pnl



                day_gross_pnl += exit_mark_pnl







                exit_transaction_cost = (



                    abs(current.shares_a * exit_a)



                    + abs(current.shares_b * exit_b)



                ) * transaction_cost_bps / 10000







                equity -= exit_transaction_cost



                day_transaction_cost += exit_transaction_cost







                current.exit_signal_date = pending["signal_date"]



                current.exit_date = idx







                current.exit_z = pending["z"]







                current.exit_price_a = exit_a



                current.exit_price_b = exit_b







                # Full trade P&L is calculated ONCE for the



                # ledger, but is NOT added to equity again.



                current.gross_pnl = (



                    (exit_a - current.entry_price_a)



                    * current.shares_a



                    + (exit_b - current.entry_price_b)



                    * current.shares_b



                )







                current.transaction_cost += exit_transaction_cost







                current.holding_days = trading_days_held







                current.exit_reason = pending["reason"]







                current.net_pnl = (



                    current.gross_pnl



                    - current.transaction_cost



                    - current.borrow_cost



                )







                ledger.append(current)







                current = None



                state = 0







                prev_mark_a = None



                prev_mark_b = None







                trading_days_held = 0







            pending = None







        # =====================================================



        # 2. MARK OPEN POSITION TO TODAY'S CLOSE



        # =====================================================







        if state != 0 and current is not None:







            close_a = float(row.close_a)



            close_b = float(row.close_b)







            # P&L from the most recent mark to today's close.



            #



            # Entry day:



            #     today's open -> today's close



            #



            # Normal holding day:



            #     yesterday's close -> today's close



            #



            # Therefore overnight movement is included.



            mtm_pnl = (



                (close_a - prev_mark_a) * current.shares_a



                + (close_b - prev_mark_b) * current.shares_b



            )







            equity += mtm_pnl



            day_gross_pnl += mtm_pnl







            # -------------------------------------------------



            # SHORT BORROW COST



            # -------------------------------------------------







            short_notional = 0.0







            if current.shares_a < 0:







                short_notional += abs(



                    current.shares_a * close_a



                )







            if current.shares_b < 0:







                short_notional += abs(



                    current.shares_b * close_b



                )







            borrow_cost = (



                short_notional



                * (annual_borrow_bps / 10000)



                / 252



            )







            current.borrow_cost += borrow_cost







            equity -= borrow_cost







            day_borrow_cost += borrow_cost







            # Update latest mark.



            prev_mark_a = close_a



            prev_mark_b = close_b







            trading_days_held += 1







            # -------------------------------------------------



            # EXPOSURE



            # -------------------------------------------------







            notional_a = current.shares_a * close_a



            notional_b = current.shares_b * close_b







            if equity != 0:







                daily.loc[



                    idx,



                    [



                        "position",



                        "weight_a",



                        "weight_b",



                        "gross_exposure",



                        "net_exposure",



                    ],



                ] = [



                    state,



                    notional_a / equity,



                    notional_b / equity,



                    (



                        abs(notional_a)



                        + abs(notional_b)



                    )



                    / equity,



                    (



                        notional_a



                        + notional_b



                    )



                    / equity,



                ]







        # =====================================================



        # 3. STORE DAILY ACCOUNTING



        # =====================================================







        daily.loc[idx, "gross_pnl"] = day_gross_pnl







        daily.loc[



            idx,



            "transaction_cost",



        ] = day_transaction_cost







        daily.loc[



            idx,



            "borrow_cost",



        ] = day_borrow_cost







        daily.loc[



            idx,



            "net_pnl",



        ] = (



            day_gross_pnl



            - day_transaction_cost



            - day_borrow_cost



        )







        daily.loc[idx, "equity"] = equity







        # =====================================================



        # 4. GENERATE SIGNAL AT TODAY'S CLOSE



        # =====================================================







        z = row.zscore







        if not np.isfinite(z):



            continue







        valid = bool(row.pair_valid)

        # Strictly backward-looking: the current row is not counted as a prior hit.
        days_since_abs_z_le_1_5 = (
            None if last_abs_z_le_1_5_pos is None
            else int(i - last_abs_z_le_1_5_pos)
        )
        recent_equilibrium_ok = (
            True if recent_equilibrium_max_days is None
            else (days_since_abs_z_le_1_5 is not None
                  and days_since_abs_z_le_1_5 <= int(recent_equilibrium_max_days))
        )







        hl = row.half_life







        max_hold = (



            max(



                1,



                int(round(2.5 * hl)),



            )



            if np.isfinite(hl) and hl > 0



            else 60



        )







        # -----------------------------------------------------

        # ENTRY RANGE

        # -----------------------------------------------------

        # Upper entry cap is independent of the hard stop.
        # If omitted, preserve the original behaviour.

        entry_cap = (
            float(z_stop)
            if z_entry_max is None
            else min(float(z_entry_max), float(z_stop))
        )


        # -----------------------------------------------------



        # ENTRY SIGNAL



        # -----------------------------------------------------







        if state == 0:







            if (



                valid



                and recent_equilibrium_ok

                and z_entry <= z < entry_cap



            ):







                pending = {



                    "type": "ENTER",



                    "direction": -1,



                    "signal_date": idx,



                    "z": float(z),



                    "beta": float(row.beta),



                    "orientation": row.orientation,



                }







            elif (



                valid



                and recent_equilibrium_ok

                and -entry_cap < z <= -z_entry



            ):







                pending = {



                    "type": "ENTER",



                    "direction": 1,



                    "signal_date": idx,



                    "z": float(z),



                    "beta": float(row.beta),



                    "orientation": row.orientation,



                }







        # -----------------------------------------------------



        # EXIT SIGNAL



        # -----------------------------------------------------







        elif current is not None:







            reason = None







            if abs(z) >= z_stop:







                reason = "HARD_STOP"







            elif abs(z) <= z_exit:







                reason = "MEAN_REVERSION"







            elif trading_days_held >= max_hold:







                reason = "TIME_STOP"







            elif not valid:







                reason = "COINTEGRATION_BREAK"







            if reason is not None:







                pending = {



                    "type": "EXIT",



                    "signal_date": idx,



                    "z": float(z),



                    "reason": reason,



                }







    # =========================================================




        # Update only after today's entry decision.
        if abs(float(z)) <= 1.5:
            last_abs_z_le_1_5_pos = i

    # 5. HANDLE OPEN POSITION AT END OF BACKTEST



    # =========================================================







    if current is not None:







        last_idx = df.index[-1]



        last_row = df.iloc[-1]







        exit_a = float(last_row.close_a)



        exit_b = float(last_row.close_b)







        current.exit_signal_date = last_idx



        current.exit_date = last_idx







        current.exit_z = (



            float(last_row.zscore)



            if np.isfinite(last_row.zscore)



            else np.nan



        )







        current.exit_price_a = exit_a



        current.exit_price_b = exit_b







        current.gross_pnl = (



            (exit_a - current.entry_price_a)



            * current.shares_a



            + (exit_b - current.entry_price_b)



            * current.shares_b



        )







        # Charge an exit transaction cost at the final close.



        exit_transaction_cost = (



            abs(current.shares_a * exit_a)



            + abs(current.shares_b * exit_b)



        ) * transaction_cost_bps / 10000







        current.transaction_cost += exit_transaction_cost







        equity -= exit_transaction_cost







        # Add final transaction cost to the final day's



        # accounting.



        daily.loc[



            last_idx,



            "transaction_cost",



        ] += exit_transaction_cost







        daily.loc[



            last_idx,



            "net_pnl",



        ] -= exit_transaction_cost







        daily.loc[last_idx, "equity"] = equity







        current.holding_days = trading_days_held







        current.exit_reason = "END_OF_BACKTEST"







        current.net_pnl = (



            current.gross_pnl



            - current.transaction_cost



            - current.borrow_cost



        )







        ledger.append(current)







    # =========================================================



    # 6. FINAL DAILY RETURNS



    # =========================================================







    daily["equity"] = (



        daily["equity"]



        .replace(0, np.nan)



        .ffill()



        .fillna(capital)



    )







    daily["net_return"] = (



        daily["equity"]



        .pct_change()



        .fillna(0.0)



    )







    ledger_df = pd.DataFrame(



        [asdict(t) for t in ledger]



    )







    return daily, ledger_df