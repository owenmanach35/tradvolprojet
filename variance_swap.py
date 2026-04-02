"""
variance_swap.py — Utilities for variance swap analysis.

Project-specific helpers for pricing, replication, and skew analysis
of variance swaps. None of these duplicate existing investment_lab functions:
- The zero-mean realized variance convention differs from metrics/volatility.py (std-based).
- ATM IV extraction, MFIV, skew slope, and Derman approximation are not available elsewhere.
"""

import numpy as np
import pandas as pd

from investment_lab.backtest import StrategyBacktester
from investment_lab.constants import TRADING_DAYS_PER_YEAR
from investment_lab.dataclass import VarianceSwapLegSpec
from investment_lab.option_selection import select_closest_maturity
from investment_lab.option_trade import VarianceSwap
from investment_lab.util import ffill_options_data


def _ensure_ticker(df_options: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Add a synthetic ticker when the source contains a single underlier only."""
    df = df_options.copy()
    ticker_was_missing = "ticker" not in df.columns
    if ticker_was_missing:
        df["ticker"] = "SPY"
    return df, ticker_was_missing


def _select_target_maturity_slice(
    df_options: pd.DataFrame,
    dte_lo: int,
    dte_hi: int,
) -> tuple[pd.DataFrame, bool]:
    """
    Keep one expiry slice per date/ticker, chosen as the closest maturity to the
    midpoint of the requested DTE window.
    """
    df, ticker_was_missing = _ensure_ticker(df_options)
    dte_target = int(round((dte_lo + dte_hi) / 2))
    df_filtered = df[df["day_to_expiration"].between(dte_lo, dte_hi)].copy()
    if df_filtered.empty:
        return df_filtered, ticker_was_missing
    return select_closest_maturity(df_filtered, day_to_expiry_target=dte_target), ticker_was_missing


def _finalize_grouped_output(df_result: pd.DataFrame, ticker_was_missing: bool) -> pd.DataFrame:
    if ticker_was_missing and "ticker" in df_result.columns:
        return df_result.drop(columns=["ticker"])
    return df_result


def _finalize_grouped_series(df_result: pd.DataFrame, value_col: str, ticker_was_missing: bool) -> pd.Series:
    df_result = _finalize_grouped_output(df_result, ticker_was_missing)
    index_cols = ["date"] if "ticker" not in df_result.columns else ["date", "ticker"]
    return df_result.set_index(index_cols)[value_col]


def _compute_strike_widths(strikes: np.ndarray) -> np.ndarray:
    """
    Half-interval strike widths used in the static replication discretization.
    """
    strikes = np.asarray(strikes, dtype=float)
    if len(strikes) == 0:
        return np.array([], dtype=float)
    if len(strikes) == 1:
        return np.array([0.0], dtype=float)

    widths = np.empty(len(strikes), dtype=float)
    widths[0] = strikes[1] - strikes[0]
    widths[-1] = strikes[-1] - strikes[-2]
    if len(strikes) > 2:
        widths[1:-1] = 0.5 * (strikes[2:] - strikes[:-2])
    return widths


def _infer_forward_from_parity(grp: pd.DataFrame, r: float, ttm: float) -> float:
    """
    Infer the forward from call-put parity around the ATM strike when possible.
    Fallback to spot growth when no common strikes are available.
    """
    cp = (
        grp.pivot_table(index="strike", columns="call_put", values="mid", aggfunc="first")
        .dropna(subset=["C", "P"], how="any")
        .reset_index()
    )
    spot = float(grp["spot"].iloc[0])
    if cp.empty:
        return spot * np.exp(r * ttm)

    cp["parity_gap"] = (cp["C"] - cp["P"]).abs()
    parity_row = cp.sort_values(["parity_gap", "strike"]).iloc[0]
    return float(parity_row["strike"] + np.exp(r * ttm) * (parity_row["C"] - parity_row["P"]))


def compute_realized_variance_series(
    returns: pd.Series,
    window: int = 21,
    ann_factor: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """
    Rolling realized variance using the zero-mean variance swap convention:

        σ²(t) = (ann_factor / window) × Σ r²_i

    Differs from metrics.volatility.rolling_realized_volatility which uses std
    (mean-subtracted). For variance swaps, drift is excluded by market convention.

    Args:
        returns: Daily log-return series.
        window: Rolling window in trading days.
        ann_factor: Annualization factor (default 252).

    Returns:
        Rolling annualized realized variance series.
    """
    return returns.rolling(window).apply(
        lambda x: np.sum(x**2) * ann_factor / len(x), raw=True
    )


def get_atm_iv_series(
    df_options: pd.DataFrame,
    dte_lo: int = 21,
    dte_hi: int = 35,
) -> pd.DataFrame:
    """
    Extract ATM implied volatility per date from a raw options DataFrame.

    Selects the call option with moneyness closest to 1.0 within the DTE range.

    Args:
        df_options: Options DataFrame with columns: date, day_to_expiration,
                    implied_volatility, moneyness, call_put.
        dte_lo: Lower bound on days to expiration (inclusive).
        dte_hi: Upper bound on days to expiration (inclusive).

    Returns:
        DataFrame with columns: date, atm_iv, atm_var.
    """
    df_slice, ticker_was_missing = _select_target_maturity_slice(df_options, dte_lo=dte_lo, dte_hi=dte_hi)
    df_filtered = df_slice[
        df_slice["implied_volatility"].notna()
        & df_slice["implied_volatility"].between(0.02, 3.0)
    ].copy()

    def _pick_atm(grp: pd.DataFrame) -> pd.Series:
        calls = grp[grp["call_put"] == "C"]
        if calls.empty:
            return pd.Series({"atm_iv": np.nan})
        idx = (calls["moneyness"] - 1.0).abs().idxmin()
        return pd.Series({"atm_iv": calls.loc[idx, "implied_volatility"]})

    group_cols = ["date", "ticker"]
    df_atm = df_filtered.groupby(group_cols).apply(_pick_atm, include_groups=False).reset_index()
    df_atm["atm_var"] = df_atm["atm_iv"] ** 2
    return _finalize_grouped_output(df_atm, ticker_was_missing)


def get_iv_at_moneyness(
    df_options: pd.DataFrame,
    moneyness_target: float,
    dte_lo: int = 21,
    dte_hi: int = 35,
    tol: float = 0.03,
    call_or_put: str = "P",
) -> pd.Series:
    """
    Extract implied volatility at a target moneyness level per date.

    Args:
        df_options: Options DataFrame.
        moneyness_target: Target moneyness (e.g. 0.90 for 90% put).
        dte_lo: Lower bound on days to expiration (inclusive).
        dte_hi: Upper bound on days to expiration (inclusive).
        tol: Tolerance band around moneyness_target.
        call_or_put: "P" for puts, "C" for calls.

    Returns:
        Series indexed by date with IV at the target moneyness.
    """
    df_slice, ticker_was_missing = _select_target_maturity_slice(df_options, dte_lo=dte_lo, dte_hi=dte_hi)
    df_filtered = df_slice[
        df_slice["implied_volatility"].notna()
        & df_slice["implied_volatility"].between(0.02, 3.0)
    ].copy()

    def _pick_iv(grp: pd.DataFrame) -> float:
        sub = grp[
            (grp["call_put"] == call_or_put)
            & grp["moneyness"].between(moneyness_target - tol, moneyness_target + tol)
        ]
        if sub.empty:
            return np.nan
        return float(
            sub.loc[(sub["moneyness"] - moneyness_target).abs().idxmin(), "implied_volatility"]
        )

    group_cols = ["date", "ticker"]
    df_iv = df_filtered.groupby(group_cols).apply(_pick_iv, include_groups=False).reset_index(name="iv_target")
    return _finalize_grouped_series(df_iv, "iv_target", ticker_was_missing)


def compute_mfiv_series(
    df_options: pd.DataFrame,
    df_rates: pd.DataFrame,
    dte_lo: int = 21,
    dte_hi: int = 35,
    rate_col: str = "1 Mo",
    min_mid: float = 0.01,
) -> pd.DataFrame:
    """
    Compute Model-Free Implied Variance (MFIV) per date.

    Implements the Carr-Madan static replication formula:

        σ²_MFIV = (2 × e^(rT) / T) × [∫_0^F P(K)/K² dK + ∫_F^∞ C(K)/K² dK]

    Discretized with weights ΔK_i / K_i² via np.gradient.

    Args:
        df_options: Raw options DataFrame (unfiltered).
        df_rates: Rates DataFrame with columns: date, <rate_col>.
        dte_lo / dte_hi: DTE filter bounds.
        rate_col: Risk-free rate column in df_rates.
        min_mid: Minimum mid-price to exclude illiquid options.

    Returns:
        DataFrame with columns: date, mfiv, mfiv_vol.
    """
    df_slice, ticker_was_missing = _select_target_maturity_slice(df_options, dte_lo=dte_lo, dte_hi=dte_hi)
    df_filtered = df_slice[
        df_slice["implied_volatility"].notna()
        & df_slice["implied_volatility"].between(0.02, 2.5)
        & (df_slice["mid"] > min_mid)
    ].copy()
    df_filtered = df_filtered.merge(df_rates[["date", rate_col]], on="date", how="left")
    df_filtered[rate_col] = df_filtered[rate_col].fillna(0.02)

    def _mfiv_one_date(grp: pd.DataFrame) -> float:
        # day_to_expiration is in calendar days; use 365 to match the annualization
        # convention used in the implied_volatility column (Black-Scholes with T=cal/365)
        T = grp["day_to_expiration"].iloc[0] / 365
        r = grp[rate_col].iloc[0]
        F = _infer_forward_from_parity(grp, r=r, ttm=T)

        chain = (
            grp.pivot_table(index="strike", columns="call_put", values="mid", aggfunc="first")
            .sort_index()
        )
        if chain.empty or not {"C", "P"}.intersection(chain.columns):
            return np.nan

        strikes = chain.index.to_numpy(dtype=float)
        dK = _compute_strike_widths(strikes)
        call_mid = chain["C"] if "C" in chain.columns else pd.Series(np.nan, index=chain.index)
        put_mid = chain["P"] if "P" in chain.columns else pd.Series(np.nan, index=chain.index)

        otm_mid = np.where(
            strikes < F,
            put_mid.to_numpy(dtype=float),
            np.where(
                strikes > F,
                call_mid.to_numpy(dtype=float),
                0.5 * np.nan_to_num(call_mid.to_numpy(dtype=float), nan=0.0)
                + 0.5 * np.nan_to_num(put_mid.to_numpy(dtype=float), nan=0.0),
            ),
        )
        valid = np.isfinite(otm_mid) & (otm_mid > 0)
        if valid.sum() < 3:
            return np.nan

        strike_term = np.sum(dK[valid] * otm_mid[valid] / strikes[valid] ** 2)
        return max(2.0 * np.exp(r * T) / T * strike_term, 0.0)

    group_cols = ["date", "ticker"]
    mfiv = (
        df_filtered.groupby(group_cols)
        .apply(_mfiv_one_date, include_groups=False)
        .reset_index(name="mfiv")
    )
    mfiv["mfiv_vol"] = np.sqrt(mfiv["mfiv"].clip(lower=0))
    mfiv = mfiv.dropna(subset=["mfiv_vol"])
    mfiv = mfiv[mfiv["mfiv_vol"].between(0.02, 2.0)]
    return _finalize_grouped_output(mfiv.reset_index(drop=True), ticker_was_missing)


def derman_kvar(
    atm_iv: "float | pd.Series",
    skew_slope: "float | pd.Series",
    T: float,
) -> "float | pd.Series":
    """
    Derman approximation for the variance swap fair strike (1st order in skew²).

    For a vol surface affine in log-moneyness σ(K) ≈ σ_ATM + skew × ln(K/F):

        K_var ≈ σ_ATM × √(1 + 3T × skew²)

    Args:
        atm_iv: ATM implied volatility in decimal (e.g. 0.20). Scalar or Series.
        skew_slope: Slope dσ/d(ln K/F), typically negative for equity indices.
        T: Time to maturity in years.

    Returns:
        Derman fair strike in the same units as atm_iv.
    """
    return atm_iv * np.sqrt(1 + 3 * T * skew_slope**2)


def compute_forward_variance_vol(
    var_short: float,
    T_short: float,
    var_long: float,
    T_long: float,
) -> float:
    """
    Compute forward realized volatility between T_short and T_long.

    From variance additivity:
        F²(T_short → T_long) = (T_long × var_long − T_short × var_short) / (T_long − T_short)

    Args:
        var_short: Annualized variance at T_short.
        T_short: Short maturity (months or years, consistent with T_long).
        var_long: Annualized variance at T_long.
        T_long: Long maturity (months or years, consistent with T_short).

    Returns:
        Forward volatility (√forward variance), floored at 0.
    """
    fwd_var = (T_long * var_long - T_short * var_short) / (T_long - T_short)
    return float(np.sqrt(max(0.0, fwd_var)))


def build_variance_swap_positions(
    df_spy: pd.DataFrame,
    weight: float = -1.0,
    day_to_expiry_target: int = 21,
    rebal_week_day: "list[int] | None" = None,
) -> pd.DataFrame:
    """
    Build rolling variance swap positions from pre-loaded SPY options data.

    Uses VarianceSwap._select_options() to construct the 1/K² replication portfolio,
    then expands to daily positions via _convert_trades_to_timeseries().

    Args:
        df_spy: Pre-loaded SPY options DataFrame (from SPYOptionLoader).
        weight: Position weight — negative for short variance (e.g. -1.0).
        day_to_expiry_target: Target DTE for each swap (default 21 trading days).
        rebal_week_day: Rebalancing days as day-of-week integers (0=Mon … 4=Fri).
                        Defaults to [1] (Tuesday).

    Returns:
        DataFrame with columns: date, option_id, entry_date, leg_name, weight, ticker.
        Ready to pass directly to PreloadedBacktester.
    """
    if rebal_week_day is None:
        rebal_week_day = [1]

    df = df_spy.copy()
    if "ticker" not in df.columns:
        df["ticker"] = "SPY"

    leg: VarianceSwapLegSpec = {
        "day_to_expiry_target": day_to_expiry_target,
        "strike_spacing": 5.0,
        "weight": weight,
        "rebal_week_day": rebal_week_day,
    }

    trades = VarianceSwap._select_options(df, legs=[leg])
    return VarianceSwap._convert_trades_to_timeseries(trades)


class PreloadedBacktester(StrategyBacktester):
    """
    StrategyBacktester variant that uses pre-loaded option data
    instead of reloading from the parquet database.

    Required for SPYOptionLoader data (spy_2020_2022.parquet), which lacks
    the 'ticker' column expected by StrategyBacktester._preprocess_positions().

    Usage:
        positions = build_variance_swap_positions(df_spy)
        bt = PreloadedBacktester(positions, df_spy)
        bt.compute_backtest()
        bt.nav   # NAV series
        bt.pnl   # P&L decomposition (delta, gamma, theta, vega)
    """

    def __init__(self, df_positions: pd.DataFrame, df_options: pd.DataFrame) -> None:
        super().__init__(df_positions)
        self._df_options = df_options.copy()
        if "ticker" not in self._df_options.columns:
            self._df_options["ticker"] = "SPY"

    def _preprocess_positions(self, df_positions: pd.DataFrame) -> pd.DataFrame:
        """Override: merge with pre-loaded data instead of reloading from file."""
        df_options = self._df_options

        # Spot rows — mirrors StrategyBacktester._preprocess_positions logic
        df_spot = (
            df_options.groupby(["date", "ticker"], as_index=False)
            .agg(spot=("spot", "first"))
            .assign(
                option_id=lambda x: x["ticker"],
                bid=lambda x: x["spot"],
                ask=lambda x: x["spot"],
                mid=lambda x: x["spot"],
                delta=1.0,
            )
        )
        df_options_with_spot = pd.concat([df_options, df_spot], ignore_index=True)

        df_extended = df_positions.merge(
            df_options_with_spot, how="left", on=["ticker", "option_id", "date"]
        )
        df_extended = df_extended[
            (df_extended["date"] <= df_extended["expiration"])
            | df_extended["expiration"].isna()
        ]
        return ffill_options_data(df_extended)


def compute_gamma_swap_series(
    df_options: pd.DataFrame,
    df_rates: pd.DataFrame,
    dte_lo: int = 21,
    dte_hi: int = 35,
    rate_col: str = "1 Mo",
    min_mid: float = 0.01,
) -> pd.DataFrame:
    """
    Compute the gamma swap implied strike (annualized) per date.

    The gamma swap kernel replaces 1/K² with 1/K, reducing the weight on
    deep OTM puts compared to the variance swap.

        K²_gamma = (2 × e^(rT) / (T × S_0)) × [∫ P(K)/K dK + ∫ C(K)/K dK]

    Args:
        df_options: Raw SPY options DataFrame.
        df_rates: Rates DataFrame with columns: date, <rate_col>.
        dte_lo: Lower DTE bound (inclusive).
        dte_hi: Upper DTE bound (inclusive).
        rate_col: Risk-free rate column in df_rates.
        min_mid: Minimum mid-price filter.

    Returns:
        DataFrame with columns: date, gamma_var, gamma_vol.
    """
    dte_target = int(round((dte_lo + dte_hi) / 2))

    df_slice = df_options[
        df_options["day_to_expiration"].between(dte_lo, dte_hi)
        & df_options["implied_volatility"].notna()
        & df_options["implied_volatility"].between(0.02, 2.5)
        & df_options["mid"].gt(min_mid)
    ].copy()

    if "ticker" not in df_slice.columns:
        df_slice["ticker"] = "SPY"

    df_slice = select_closest_maturity(df_slice, day_to_expiry_target=dte_target)
    df_slice = df_slice.merge(df_rates[["date", rate_col]], on="date", how="left")
    df_slice[rate_col] = df_slice[rate_col].fillna(0.02)

    def _gamma_one_group(grp: pd.DataFrame) -> float:
        T = grp["day_to_expiration"].iloc[0] / 365
        r = grp[rate_col].iloc[0]
        s0 = grp["spot"].iloc[0]
        fwd = _infer_forward_from_parity(grp, r=r, ttm=T)

        chain = (
            grp.pivot_table(index="strike", columns="call_put", values="mid", aggfunc="first")
            .sort_index()
        )
        if chain.empty:
            return np.nan

        strikes = chain.index.to_numpy(dtype=float)
        dK = _compute_strike_widths(strikes)
        calls = chain["C"] if "C" in chain.columns else pd.Series(np.nan, index=chain.index)
        puts = chain["P"] if "P" in chain.columns else pd.Series(np.nan, index=chain.index)

        otm_mid = np.where(
            strikes < fwd,
            puts.to_numpy(dtype=float),
            np.where(
                strikes > fwd,
                calls.to_numpy(dtype=float),
                0.5 * np.nan_to_num(calls.to_numpy(dtype=float), nan=0.0)
                + 0.5 * np.nan_to_num(puts.to_numpy(dtype=float), nan=0.0),
            ),
        )

        valid = np.isfinite(otm_mid) & (otm_mid > 0)
        if valid.sum() < 3:
            return np.nan

        integral = np.sum(dK[valid] * otm_mid[valid] / strikes[valid])
        return max(2.0 * np.exp(r * T) / (T * s0) * integral, 0.0)

    gamma_swap = (
        df_slice.groupby(["date", "ticker"])
        .apply(_gamma_one_group, include_groups=False)
        .reset_index(name="gamma_var")
    )
    gamma_swap["gamma_vol"] = np.sqrt(gamma_swap["gamma_var"].clip(lower=0))
    gamma_swap = gamma_swap[gamma_swap["gamma_vol"].between(0.02, 2.0)].reset_index(drop=True)
    return _finalize_grouped_output(gamma_swap, "ticker" not in df_options.columns)


def build_gamma_swap_positions(
    df_options: pd.DataFrame,
    weight: float = -1.0,
    day_to_expiry_target: int = 21,
    rebal_week_day: "list[int] | None" = None,
    strike_spacing: float = 5.0,
) -> pd.DataFrame:
    """
    Build rolling gamma swap positions using the 1/K kernel.

    Similar to build_variance_swap_positions but with dK/K weights instead of dK/K².

    Args:
        df_options: Pre-loaded SPY options DataFrame.
        weight: Position weight (negative = short).
        day_to_expiry_target: Target DTE.
        rebal_week_day: Rebalancing days (0=Mon … 4=Fri). Defaults to [1] (Tuesday).
        strike_spacing: Minimum spacing between retained strikes.

    Returns:
        DataFrame with columns: date, option_id, entry_date, leg_name, weight, ticker.
    """
    if rebal_week_day is None:
        rebal_week_day = [1]

    df = df_options.copy()
    if "ticker" not in df.columns:
        df["ticker"] = "SPY"

    selected = select_closest_maturity(df, day_to_expiry_target=day_to_expiry_target)
    selected = selected.loc[
        ((selected["call_put"] == "P") & (selected["moneyness"] <= 1.0))
        | ((selected["call_put"] == "C") & (selected["moneyness"] >= 1.0))
    ].copy()
    selected["leg_name"] = "GAMMA SWAP"
    selected = selected.loc[selected["date"].dt.day_of_week.isin(rebal_week_day)]

    def _compute_gamma_weights(grp: pd.DataFrame) -> pd.DataFrame:
        grp = VarianceSwap._thin_strikes(grp.copy(), strike_spacing=strike_spacing)
        weighted_groups = []
        for call_put in ("P", "C"):
            side_df = grp[grp["call_put"] == call_put].copy().sort_values("strike")
            if side_df.empty:
                continue
            widths = VarianceSwap._compute_strike_widths(
                side_df["strike"].to_numpy(dtype=float)
            )
            side_df["weight"] = widths / side_df["strike"].to_numpy(dtype=float)
            weighted_groups.append(side_df)
        if not weighted_groups:
            grp["weight"] = 0.0
            return grp
        return pd.concat(weighted_groups, ignore_index=True)

    weighted = pd.concat(
        [
            _compute_gamma_weights(grp.copy())
            for _, grp in selected.groupby(["date", "expiration"], sort=False)
        ],
        ignore_index=True,
    )
    normalized = pd.concat(
        [
            VarianceSwap._normalize_strike_weights(grp.copy(), target_weight=weight)
            for _, grp in weighted.groupby("date", sort=False)
        ],
        ignore_index=True,
    )
    trades = normalized.rename(columns={"date": "entry_date"})[
        ["entry_date", "option_id", "expiration", "leg_name", "weight", "ticker"]
    ]
    return VarianceSwap._convert_trades_to_timeseries(trades)


def compute_bkm_X_series(
    df_option_quotes: pd.DataFrame,
    df_rates: "pd.DataFrame | None" = None,
    day_to_expiry_target: int = 21,
    rate_col: str = "1 Mo",
    min_mid: float = 0.01,
) -> pd.DataFrame:
    """
    Compute BKM (Bakshi-Kapadia-Madan 2003) moment contracts V, W, X per date.

    V = ∫ 2 × e^(rT) × Q(K)/K² × dK       (variance contract)
    W = ∫ e^(rT) × Q(K) × (6ln(K/F) - 3ln²(K/F)) / K² × dK   (cubic)
    X = ∫ e^(rT) × Q(K) × (12ln²(K/F) - 4ln³(K/F)) / K² × dK  (quartic)

    The ratio X/V² is a model-free proxy for risk-neutral excess kurtosis.

    Args:
        df_option_quotes: Raw SPY options DataFrame.
        df_rates: Rates DataFrame (optional). Uses 2% default if None.
        day_to_expiry_target: Target DTE for maturity selection.
        rate_col: Risk-free rate column in df_rates.
        min_mid: Minimum mid-price filter.

    Returns:
        DataFrame with columns: date, spot, dte, r_t, V, W, X.
    """
    if df_rates is not None and rate_col in df_rates.columns:
        rate_map = df_rates.set_index("date")[rate_col].to_dict()
    else:
        rate_map = {}

    rows = []

    for date, df_date in df_option_quotes.groupby("date"):
        df_date = df_date.copy()
        if "ticker" not in df_date.columns:
            df_date["ticker"] = "SPY"
        if "implied_volatility" in df_date.columns:
            df_date = df_date[df_date["implied_volatility"].between(0.02, 2.5)]
        selected = select_closest_maturity(df_date, day_to_expiry_target=day_to_expiry_target)
        if selected.empty:
            continue

        selected = selected.copy()
        dte = selected["day_to_expiration"].iloc[0]
        t = dte / 365.0
        if t <= 0:
            continue

        r_t = float(rate_map.get(date, 0.02))
        ert = np.exp(r_t * t)

        selected["forward"] = _infer_forward_from_parity(selected, r_t, t)
        forward = selected["forward"].iloc[0]

        chain = (
            selected[selected["mid"].ge(min_mid)]
            .pivot_table(index="strike", columns="call_put", values="mid", aggfunc="first")
            .sort_index()
        )
        if chain.empty or not {"C", "P"}.issubset(chain.columns):
            continue

        max_log_m = 0.7
        log_m_chain = np.log(chain.index.to_numpy(dtype=float) / forward)
        chain = chain[np.abs(log_m_chain) <= max_log_m]
        if chain.empty:
            continue

        all_K = chain.index.to_numpy(dtype=float)
        dK_full = _compute_strike_widths(all_K)

        call_mid = chain["C"].to_numpy(dtype=float)
        put_mid = chain["P"].to_numpy(dtype=float)

        otm_price = np.where(
            all_K < forward,
            put_mid,
            np.where(
                all_K > forward,
                call_mid,
                0.5 * np.nan_to_num(call_mid, nan=0.0)
                + 0.5 * np.nan_to_num(put_mid, nan=0.0),
            ),
        )

        valid = np.isfinite(otm_price) & (otm_price > 0)
        n_p = (valid & (all_K < forward)).sum()
        n_c = (valid & (all_K > forward)).sum()
        if n_p < 5 or n_c < 5:
            continue

        K = all_K[valid]
        dK = dK_full[valid]
        price = otm_price[valid]
        log_m = np.log(K / forward)

        V = ert * (2.0 * price * dK / K**2).sum()
        W = ert * (price * dK * (6.0 * log_m - 3.0 * log_m**2) / K**2).sum()
        X = ert * (price * dK * (12.0 * log_m**2 - 4.0 * log_m**3) / K**2).sum()

        rows.append({
            "date": date,
            "spot": selected["spot"].iloc[0],
            "dte": dte,
            "r_t": r_t,
            "V": V,
            "W": W,
            "X": X,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Carr-Madan kurtosis swap — internally consistent M2 / M4 family
# ---------------------------------------------------------------------------

def compute_kurtosis_cm_series(
    df_options: pd.DataFrame,
    df_rates: pd.DataFrame,
    dte_lo: int = 21,
    dte_hi: int = 35,
    rate_col: str = "1 Mo",
    min_mid: float = 0.01,
) -> pd.DataFrame:
    """
    Compute Carr-Madan moment contracts M2_CM, M4_CM and the consistent
    kurtosis ratio κ_CM = M4_CM / M2_CM² per date.

    Both moments are derived from the spanning formula applied to simple
    returns R = S_T/F − 1, ensuring internal consistency:

        f(S_T) = (S_T/F − 1)^n  →  f''(K) = n(n−1)(K/F − 1)^(n−2) / F²

    Concretely:

        M2_CM = e^(rT) × (2/F²) × ∫ Q(K) dK                 [kernel: 2/F²]
        M4_CM = e^(rT) × ∫ 12(K−F)²/F⁴ × Q(K) dK           [kernel: 12(K−F)²/F⁴]
        κ_CM  = M4_CM / M2_CM²

    Unlike the BKM proxy X/V², κ_CM is fully self-consistent: both moments
    measure the same object (the simple-return distribution), and κ_CM = 3
    for any Gaussian distribution regardless of T and F.

    Args:
        df_options: Raw SPY options DataFrame.
        df_rates: Rates DataFrame with columns: date, <rate_col>.
        dte_lo: Lower DTE bound (inclusive).
        dte_hi: Upper DTE bound (inclusive).
        rate_col: Risk-free rate column in df_rates.
        min_mid: Minimum mid-price filter.

    Returns:
        DataFrame with columns: date, M2_CM, M4_CM, kappa_CM.
    """
    dte_target = int(round((dte_lo + dte_hi) / 2))

    df_slice = df_options[
        df_options["day_to_expiration"].between(dte_lo, dte_hi)
        & df_options["implied_volatility"].notna()
        & df_options["implied_volatility"].between(0.02, 2.5)
        & df_options["mid"].gt(min_mid)
    ].copy()

    if "ticker" not in df_slice.columns:
        df_slice["ticker"] = "SPY"

    df_slice = select_closest_maturity(df_slice, day_to_expiry_target=dte_target)
    df_slice = df_slice.merge(df_rates[["date", rate_col]], on="date", how="left")
    df_slice[rate_col] = df_slice[rate_col].fillna(0.02)

    def _cm_one_group(grp: pd.DataFrame) -> pd.Series:
        T = grp["day_to_expiration"].iloc[0] / 365.0
        r = grp[rate_col].iloc[0]
        fwd = _infer_forward_from_parity(grp, r=r, ttm=T)
        ert = np.exp(r * T)

        chain = (
            grp.pivot_table(index="strike", columns="call_put", values="mid", aggfunc="first")
            .sort_index()
        )
        if chain.empty:
            return pd.Series({"M2_CM": np.nan, "M4_CM": np.nan, "kappa_CM": np.nan})

        strikes = chain.index.to_numpy(dtype=float)
        dK = _compute_strike_widths(strikes)
        calls = chain["C"] if "C" in chain.columns else pd.Series(np.nan, index=chain.index)
        puts = chain["P"] if "P" in chain.columns else pd.Series(np.nan, index=chain.index)

        otm_mid = np.where(
            strikes < fwd,
            puts.to_numpy(dtype=float),
            np.where(
                strikes > fwd,
                calls.to_numpy(dtype=float),
                0.5 * np.nan_to_num(calls.to_numpy(dtype=float), nan=0.0)
                + 0.5 * np.nan_to_num(puts.to_numpy(dtype=float), nan=0.0),
            ),
        )
        valid = np.isfinite(otm_mid) & (otm_mid > 0)
        if valid.sum() < 3:
            return pd.Series({"M2_CM": np.nan, "M4_CM": np.nan, "kappa_CM": np.nan})

        K = strikes[valid]
        w = dK[valid]
        Q = otm_mid[valid]

        # M2_CM = e^(rT) × (2/F²) × ∫ Q(K) dK
        M2_CM = ert * (2.0 / fwd ** 2) * float(np.sum(w * Q))

        # M4_CM = e^(rT) × ∫ [12(K−F)²/F⁴] × Q(K) dK
        M4_CM = ert * float(np.sum(12.0 * (K - fwd) ** 2 / fwd ** 4 * w * Q))

        # κ_CM = M4_CM / M2_CM²  (= 3 for Gaussian)
        kappa_CM = float(M4_CM / M2_CM ** 2) if M2_CM > 1e-12 else np.nan

        return pd.Series({"M2_CM": M2_CM, "M4_CM": M4_CM, "kappa_CM": kappa_CM})

    result = (
        df_slice.groupby(["date", "ticker"])
        .apply(_cm_one_group, include_groups=False)
        .reset_index()
    )
    result = result.dropna(subset=["kappa_CM"]).reset_index(drop=True)
    return _finalize_grouped_output(result, "ticker" not in df_options.columns)


def build_kurtosis_swap_positions(
    df_options: pd.DataFrame,
    weight: float = -1.0,
    day_to_expiry_target: int = 21,
    rebal_week_day: "list[int] | None" = None,
    strike_spacing: float = 5.0,
) -> pd.DataFrame:
    """
    Build rolling kurtosis swap positions using the Carr-Madan quartic kernel.

    Applies the spanning formula to f(S_T) = (S_T/F − 1)⁴, giving:

        f''(K) = 12(K − F)² / F⁴

    Properties of this kernel versus variance/gamma kernels:
    - Always non-negative (no short-option complications near ATM).
    - Zero at K = F: ATM options carry zero weight.
    - Symmetric around F: puts and calls at equal distance from F get
      equal weight, unlike 1/K² which overweights deep OTM puts.
    - Parabolic growth: tail options matter, but weight grows smoothly
      rather than exploding like the BKM-X kernel.

    The forward F is inferred from call-put parity at each rebalancing date,
    so the kernel is recomputed on each roll (pseudo-static replication).

    Args:
        df_options: Pre-loaded SPY options DataFrame.
        weight: Position weight (negative = short).
        day_to_expiry_target: Target DTE.
        rebal_week_day: Rebalancing days (0=Mon … 4=Fri). Defaults to [1] (Tuesday).
        strike_spacing: Minimum spacing between retained strikes.

    Returns:
        DataFrame with columns: date, option_id, entry_date, leg_name, weight, ticker.
        Ready to pass directly to PreloadedBacktester.
    """
    if rebal_week_day is None:
        rebal_week_day = [1]

    df = df_options.copy()
    if "ticker" not in df.columns:
        df["ticker"] = "SPY"

    selected = select_closest_maturity(df, day_to_expiry_target=day_to_expiry_target)
    selected = selected.loc[
        ((selected["call_put"] == "P") & (selected["moneyness"] <= 1.0))
        | ((selected["call_put"] == "C") & (selected["moneyness"] >= 1.0))
    ].copy()
    selected["leg_name"] = "KURTOSIS SWAP"
    selected = selected.loc[selected["date"].dt.day_of_week.isin(rebal_week_day)]

    def _compute_kurtosis_weights(grp: pd.DataFrame) -> pd.DataFrame:
        # Infer forward from call-put parity (rate=0 approximation valid for 21-DTE, low-rate env.)
        T = grp["day_to_expiration"].iloc[0] / 365.0
        fwd = _infer_forward_from_parity(grp, r=0.0, ttm=T)

        grp = VarianceSwap._thin_strikes(grp.copy(), strike_spacing=strike_spacing)
        weighted_groups = []
        for call_put in ("P", "C"):
            side_df = grp[grp["call_put"] == call_put].copy().sort_values("strike")
            if side_df.empty:
                continue
            strikes = side_df["strike"].to_numpy(dtype=float)
            widths = VarianceSwap._compute_strike_widths(strikes)
            # Carr-Madan quartic kernel: f''(K) = 12(K-F)²/F⁴
            kernel = 12.0 * (strikes - fwd) ** 2 / fwd ** 4
            side_df["weight"] = kernel * widths
            # Drop strikes with negligible weight (very near ATM where kernel ≈ 0)
            side_df = side_df[side_df["weight"] > 1e-14]
            weighted_groups.append(side_df)

        if not weighted_groups:
            grp["weight"] = 0.0
            return grp
        return pd.concat(weighted_groups, ignore_index=True)

    weighted = pd.concat(
        [
            _compute_kurtosis_weights(grp.copy())
            for _, grp in selected.groupby(["date", "expiration"], sort=False)
        ],
        ignore_index=True,
    )
    normalized = pd.concat(
        [
            VarianceSwap._normalize_strike_weights(grp.copy(), target_weight=weight)
            for _, grp in weighted.groupby("date", sort=False)
        ],
        ignore_index=True,
    )
    trades = normalized.rename(columns={"date": "entry_date"})[
        ["entry_date", "option_id", "expiration", "leg_name", "weight", "ticker"]
    ]
    return VarianceSwap._convert_trades_to_timeseries(trades)
