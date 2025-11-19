"""
Small wrapper around your make_predictions() function for Docker/Make.

IMPORTANT:
- Copy all the imports + helper functions + the make_predictions() function
  you already tested in your notebook into this file.
- This file is what Docker will call.
"""

# ---- paste your imports here ----
# e.g.
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
import pytz
from joblib import load
import warnings

# suppress warnings globally (including FutureWarning)
warnings.filterwarnings("ignore")

# ---- paste ALL helper functions + TASK1_MODELS / TASK2_MODELS / TASK3_MODELS
#      loading + make_predictions() itself, exactly as in your notebook ----

# make_predictions() should be defined above this line


# Entry point when run as a script
if __name__ == "__main__":
    make_predictions()
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
import pytz
from joblib import load
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 0. CONFIG: where models live & load-area metadata
# ============================================================

# Path where you saved your trained models (you decide the exact location)
MODEL_DIR = Path("models")

# These must exist and contain dictionaries keyed by load_area:
#   task1_models.pkl : {area: {"global_pipe", "local_pipe", "alpha",
#                              "g_num", "g_cat", "l_num", "l_cat"}}
#   task2_models.pkl : {area: {"mnl_pipe",
#                              "feature_cols_numeric",
#                              "feature_cols_categoric"}}
#   task3_models.pkl : {area: {"model", "method", "feature_cols"}}
TASK1_MODELS = load(MODEL_DIR / "task1_models.pkl")
TASK2_MODELS = load(MODEL_DIR / "task2_models.pkl")
TASK3_MODELS = load(MODEL_DIR / "task3_models.pkl")

# Load-area order for L1_*, PH_*, PD_* (this must match grading order)
LOAD_AREA_ORDER = [
    "AECO","AEPAPT","AEPIMP","AEPKPT","AEPOPT","AP","BC","CE","DAY","DEOK",
    "DOM","DPLCO","DUQ","EASTON","EKPC","JC","ME","OE","OVEC","PAPWR",
    "PE","PEPCO","PLCO","PN","PS","RECO","SMECO","UGI","VMEU"
]

LOAD_AREA_COORDS = {
    "AECO":  (39.45, -74.50),
    "AEPAPT":(37.25, -81.30),
    "AEPIMP":(38.45, -81.60),
    "AEPKPT":(38.20, -83.10),
    "AEPOPT":(39.90, -82.90),
    "AP":    (37.30, -80.90),
    "BC":    (40.80, -79.95),
    "CE":    (41.85, -86.10),
    "DAY":   (39.75, -84.20),
    "DEOK":  (39.10, -84.50),
    "DOM":   (37.55, -77.45),
    "DPLCO": (38.90, -75.50),
    "DUQ":   (40.45, -79.90),
    "EASTON":(39.55, -75.10),
    "EKPC":  (37.75, -84.30),
    "JC":    (40.35, -74.65),
    "ME":    (40.20, -76.00),
    "OE":    (41.10, -81.25),
    "OVEC":  (38.85, -82.85),
    "PAPWR": (40.70, -77.80),
    "PE":    (40.00, -75.20),
    "PEPCO": (38.90, -76.95),
    "PLCO":  (40.95, -77.40),
    "PN":    (41.15, -77.80),
    "PS":    (40.75, -74.15),
    "RECO":  (41.00, -74.10),
    "SMECO": (38.40, -76.70),
    "UGI":   (40.25, -75.65),
    "VMEU":  (37.30, -76.00),
}


OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
]


# ============================================================
# 1. Calendar helpers (must match training logic)
# ============================================================

def meteorological_season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    elif month in (3, 4, 5):
        return "spring"
    elif month in (6, 7, 8):
        return "summer"
    else:
        return "autumn"


def compute_thanksgiving_dates(years):
    """Return dict year -> Thanksgiving date (4th Thursday of Nov)."""
    from pandas.tseries.offsets import Week
    tg = {}
    for y in years:
        base = pd.Timestamp(y, 11, 1)
        first_thu = base + Week(weekday=3)
        tg[y] = (first_thu + Week(3)).normalize()
    return tg


def add_calendar_features(df):
    """
    df has 'ts_ept' (datetime in America/New_York) and 'load_area'.
    Adds date_ept, hour_ept, dow_ept, doy, month, season, holiday flags, etc.
    """
    df["date_ept"] = df["ts_ept"].dt.normalize()
    df["hour_ept"] = df["ts_ept"].dt.hour
    df["dow_ept"]  = df["ts_ept"].dt.weekday  # 0=Mon
    df["doy"]      = df["ts_ept"].dt.dayofyear
    df["month"]    = df["ts_ept"].dt.month

    df["season_str"] = df["month"].apply(meteorological_season)

    # Thanksgiving + week
    years = df["ts_ept"].dt.year.unique()
    tg_dates = compute_thanksgiving_dates(years)
    df["is_thanksgiving"] = df["ts_ept"].dt.normalize().map(
        lambda d: int(d == tg_dates.get(d.year, pd.NaT))
    )
    df["is_thanksgiving_week"] = df["ts_ept"].dt.normalize().map(
        lambda d: int(
            tg_dates.get(d.year, pd.NaT) - pd.Timedelta(days=3)
            <= d
            <= tg_dates.get(d.year, pd.NaT) + pd.Timedelta(days=2)
        )
    )
    df["is_weekend"] = df["dow_ept"].isin([5, 6]).astype(int)

    # Season as numeric for Task 3 peak-day model
    season_map = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}
    df["season_num"] = df["season_str"].map(season_map).astype("Int64")

    return df


def add_harmonics_and_temp_features(df):
    """Add sin/cos harmonics + temp_cool/heat etc., matching Task 1 code."""
    h = df["hour_ept"].values
    for k in [1, 2]:
        df[f"sin_h{k}"] = np.sin(2 * np.pi * k * h / 24.0)
        df[f"cos_h{k}"] = np.cos(2 * np.pi * k * h / 24.0)

    dow_num = df["dow_ept"].astype(int).values
    df["sin_dow"] = np.sin(2 * np.pi * dow_num / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * dow_num / 7.0)

    df["sin_doy"] = np.sin(2 * np.pi * df["doy"] / 365.0)
    df["cos_doy"] = np.cos(2 * np.pi * df["doy"] / 365.0)

    T = df["temperature_2m"]
    df["temp2"]      = T**2
    df["temp_cool"]  = np.maximum(T - 65.0, 0.0)
    df["temp_heat"]  = np.maximum(45.0 - T, 0.0)

    df["temp_cool_sin_h1"] = df["temp_cool"] * df["sin_h1"]
    df["temp_cool_cos_h1"] = df["temp_cool"] * df["cos_h1"]
    df["temp_heat_sin_h1"] = df["temp_heat"] * df["sin_h1"]
    df["temp_heat_cos_h1"] = df["temp_heat"] * df["cos_h1"]

    # Cast categoricals (as in training)
    df["season"]  = df["season_str"].astype("category")
    df["dow_ept"] = df["dow_ept"].astype("category")
    df["month_cat"] = df["month"].astype("category")
    return df


# ============================================================
# 2. Hybrid curve utilities (shared with Task 1 & 2 & 3)
# ============================================================

def extract_curve_features(L):
    """Given 24-vector L, return the same hybrid curve features as training."""
    L = np.asarray(L)
    hmax = int(L.argmax())
    hmin = int(L.argmin())

    morning_slope  = L[9]  - L[5]
    evening_slope  = L[18] - L[14]
    morning_level  = L[8]
    evening_level  = L[19]
    midday_level   = L[13]
    midday_vs_even = midday_level - evening_level

    return {
        "pred_peak_hour_pred": hmax,
        "pred_max_load": float(L[hmax]),
        "pred_min_load": float(L[hmin]),
        "pred_avg_load": float(L.mean()),
        "morning_slope": float(morning_slope),
        "evening_slope": float(evening_slope),
        "morning_level": float(morning_level),
        "evening_level": float(evening_level),
        "midday_level": float(midday_level),
        "midday_vs_evening": float(midday_vs_even),
    }


def get_hybrid_curve_for_day(df_day, model_cfg):
    """
    Use saved Task1 models to compute 24h hybrid load for a single day
    for one area. df_day: 24 hourly rows with all Task1 features.
    model_cfg: dict from TASK1_MODELS[area].
    """
    global_pipe = model_cfg["global_pipe"]
    local_pipe  = model_cfg["local_pipe"]
    alpha       = model_cfg["alpha"]
    g_num       = model_cfg["g_num"]
    g_cat       = model_cfg["g_cat"]
    l_num       = model_cfg["l_num"]
    l_cat       = model_cfg["l_cat"]

    Xg = df_day[g_num + g_cat]
    Xl = df_day[l_num + l_cat]

    Lg = global_pipe.predict(Xg)
    Ll = local_pipe.predict(Xl)

    return alpha * Lg + (1.0 - alpha) * Ll


# ============================================================
# 3. Open-Meteo forecast fetcher
# ============================================================

def fetch_forecast_for_area(area, lat, lon, start_date, end_date):
    """
    Fetch hourly forecast for [start_date, end_date] (inclusive) in
    America/New_York timezone for one load_area.
    Returns DataFrame with columns time, ts_ept, weather vars, load_area.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "America/New_York",  # directly in EPT
    }
    r = requests.get(OPENMETEO_URL, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()
    hourly = j.get("hourly", {})
    if "time" not in hourly:
        raise RuntimeError(f"No hourly data returned for {area}")

    df = pd.DataFrame(hourly)
    df["ts_ept"] = pd.to_datetime(df["time"])
    df["load_area"] = area
    return df


def fetch_forecast_all_areas(start_date, end_date):
    """Fetch hourly forecast for all load areas and add features."""
    dfs = []
    for area in LOAD_AREA_ORDER:
        lat, lon = LOAD_AREA_COORDS[area]
        df_area = fetch_forecast_for_area(area, lat, lon, start_date, end_date)
        dfs.append(df_area)
    df_fc = pd.concat(dfs, ignore_index=True)

    df_fc = add_calendar_features(df_fc)
    df_fc = add_harmonics_and_temp_features(df_fc)

    # Rename month_cat back to month to match training expectations
    df_fc["month"] = df_fc["month_cat"]
    return df_fc


# ============================================================
# 4. DAILY AGGREGATION FOR TASK 2 & TASK 3
# ============================================================

def build_daily_features_for_area(df_area, model_cfg_task1):
    """
    For a single area forecast df (10 days of hourly data),
    build a daily table with:
      - calendar & weather summaries
      - hybrid OLS curve features
      - approximate daily_peak_load = pred_max_load (for Task3 consistency)
    """
    rows = []
    for d, df_day in df_area.groupby(df_area["date_ept"]):
        df_day = df_day.sort_values("ts_ept")
        if df_day.shape[0] != 24:
            # require full day
            continue

        # daily weather summaries
        temp_mean = float(df_day["temperature_2m"].mean())
        temp_min  = float(df_day["temperature_2m"].min())
        temp_max  = float(df_day["temperature_2m"].max())
        rh_mean   = float(df_day["relative_humidity_2m"].mean())
        wind_mean = float(df_day["wind_speed_10m"].mean())
        precip_sum = float(df_day["precipitation"].sum())

        # calendar
        dow  = int(df_day["dow_ept"].iloc[0])
        month = int(df_day["month"].iloc[0])
        doy   = int(df_day["doy"].iloc[0])
        season_str = df_day["season_str"].iloc[0]
        season_map = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}
        season_num = int(season_map.get(season_str, -1))
        is_thanksgiving       = int(df_day["is_thanksgiving"].iloc[0])
        is_thanksgiving_week  = int(df_day["is_thanksgiving_week"].iloc[0])
        is_weekend            = int(df_day["is_weekend"].iloc[0])

        # hybrid curve (24h)
        L_pred = get_hybrid_curve_for_day(df_day, model_cfg_task1)
        curve_feats = extract_curve_features(L_pred)

        row = {
            "date": d,
            "dow": dow,
            "month": month,
            "doy": doy,
            "season": season_num,
            "is_weekend": is_weekend,
            "is_thanksgiving": is_thanksgiving,
            "is_thanksgiving_week": is_thanksgiving_week,
            "temp_mean": temp_mean,
            "temp_min": temp_min,
            "temp_max": temp_max,
            "rh_mean": rh_mean,
            "wind_mean": wind_mean,
            "precip_sum": precip_sum,
            # approximate daily peak load for Task3 (not a feature)
            "daily_peak_load": curve_feats["pred_max_load"],
        }
        row.update(curve_feats)
        rows.append(row)

    daily_df = pd.DataFrame(rows)
    daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.normalize()
    return daily_df.sort_values("date")


# ============================================================
# 5. MAIN PREDICTION FUNCTION
# ============================================================

def make_predictions():
    """
    Main entry point.
    - Figures out 'today' in America/New_York.
    - Fetches 10-day forecast (tomorrow .. +9).
    - Uses pre-trained models to output:
         "YYYY-MM-DD", L1_00, ..., L29_23, PH_1, ..., PH_29, PD_1, ..., PD_29
    printed to stdout.
    """
    # ----- Figure out dates -----
    tz_ept = pytz.timezone("America/New_York")
    now_ept = datetime.now(tz_ept)
    today_date = now_ept.date()
    target_date = today_date + timedelta(days=1)        # day we predict for
    window_end_date = today_date + timedelta(days=10)   # inclusive (10 days)

    # ----- Fetch forecast & build hourly feature table -----
    df_fc = fetch_forecast_all_areas(start_date=target_date,
                                     end_date=window_end_date)

    # We'll need both hourly (for Task1) and daily aggregated (for Tasks2/3)
    # so keep df_fc as is and aggregate per area when needed.

    # Containers for outputs in load-area order
    task1_hourly_preds = {}   # area -> length-24 array
    task2_peak_hour    = {}   # area -> int 0-23
    task3_peak_day_ind = {}   # area -> 0/1

    # ----- Loop over load areas -----
    for area in LOAD_AREA_ORDER:
        df_area = df_fc[df_fc["load_area"] == area].copy()
        if df_area.empty:
            raise RuntimeError(f"No forecast data for area {area}")

        # ---------- Task 1: 24h hybrid load for TARGET DATE ----------
        model_cfg1 = TASK1_MODELS[area]

        df_day_target = df_area[df_area["date_ept"] == pd.Timestamp(target_date)].copy()
        df_day_target = df_day_target.sort_values("ts_ept")
        if df_day_target.shape[0] != 24:
            raise RuntimeError(f"Area {area}: target-date hours != 24")

        L_pred = get_hybrid_curve_for_day(df_day_target, model_cfg1)
        task1_hourly_preds[area] = np.round(L_pred).astype(int)

        # ---------- Build daily table for this area (10 days) ----------
        daily_df = build_daily_features_for_area(df_area, model_cfg1)

        # Ensure we have 10 consecutive days from target_date
        # (may drop if forecast missing)
        mask_window = (daily_df["date"] >= pd.Timestamp(target_date)) & \
                      (daily_df["date"] <= pd.Timestamp(window_end_date))
        daily_window = daily_df[mask_window].copy()
        if daily_window.shape[0] < 10:
            raise RuntimeError(f"Area {area}: daily window has <10 days")

        # Reindex to exactly 10 days (in order) using first 10 rows
        daily_window = daily_window.sort_values("date").iloc[:10]

        # ---------- Task 2: peak hour for TARGET DATE ----------
        model_cfg2 = TASK2_MODELS[area]
        mnl_pipe = model_cfg2["mnl_pipe"]
        num_cols = model_cfg2["feature_cols_numeric"]
        cat_cols = model_cfg2["feature_cols_categoric"]

        # build feature row for target_date from daily_window
        row_t2 = daily_window[daily_window["date"] == pd.Timestamp(target_date)]
        row_t2 = row_t2.copy()
        
        # 1) Ensure we have the expected day-of-week column name
        if "dow" in row_t2.columns and "dow_ept" not in row_t2.columns:
            row_t2.loc[:, "dow_ept"] = row_t2["dow"].astype(int)
        
        # 2) Ensure 'season' is in the same format as during training
        #    (strings: "winter", "spring", "summer", "autumn")
        season_reverse = {0: "winter", 1: "spring", 2: "summer", 3: "autumn"}
        
        if "season" in row_t2.columns:
            # If season is numeric (0–3), map back to strings
            if np.issubdtype(row_t2["season"].dtype, np.number):
                row_t2.loc[:, "season"] = row_t2["season"].map(season_reverse)
            # Ensure dtype is object/string
            row_t2.loc[:, "season"] = row_t2["season"].astype(object)
        
        # 3) Small safety check: make sure all expected columns are present
        missing = [c for c in (num_cols + cat_cols) if c not in row_t2.columns]
        if missing:
            raise RuntimeError(
                f"Task 2: missing columns {missing} in prediction row for area {area}"
        )
        X_t2 = row_t2[num_cols + cat_cols]
        ph_pred = int(mnl_pipe.predict(X_t2)[0])
        task2_peak_hour[area] = ph_pred

        # ---------- Task 3: peak-day indicator (is target-date in top-2?) ----------
        model_cfg3 = TASK3_MODELS[area]
        model3  = model_cfg3["model"]
        method3 = model_cfg3["method"]
        feat3   = model_cfg3["feature_cols"]

        X_window = daily_window[feat3].copy().to_numpy()

        # positions 0..9 correspond to days target_date .. target_date+9
        if "1. RF Regression" in method3:
            # Regression: model predicts (approx) daily peak load, then rank
            y_hat_load = model3.predict(X_window)
            top_idx = np.argsort(y_hat_load)[-2:]
        else:
            # Classification: use predicted probabilities for class 1
            if hasattr(model3, "predict_proba"):
                probs = model3.predict_proba(X_window)[:, 1]
            else:
                scores = model3.decision_function(X_window)
                probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
            top_idx = np.argsort(probs)[-2:]

        # target date is index 0 in the window
        task3_peak_day_ind[area] = int(0 in top_idx)

    # =======================================================
    # 6. Assemble output line in required format
    # =======================================================
    pieces = []

    # First item: current date (today), not target date
    pieces.append(f"\"{target_date.isoformat()}\"")

    # Lz_hh in load-area order, each 24 hours
    for idx, area in enumerate(LOAD_AREA_ORDER, start=1):
        loads = task1_hourly_preds[area]
        for h in range(24):
            label = f"L{idx}_{h:02d}"
            val   = int(loads[h])
            pieces.append(str(val))

    # PH_z (peak hour 0-23 for each area)
    for idx, area in enumerate(LOAD_AREA_ORDER, start=1):
        label = f"PH_{idx}"
        pieces.append(str(int(task2_peak_hour[area])))

    # PD_z (peak day indicator 0/1 for each area)
    for idx, area in enumerate(LOAD_AREA_ORDER, start=1):
        label = f"PD_{idx}"
        pieces.append(str(int(task3_peak_day_ind[area])))

    # Print single CSV-style line
    line = ", ".join(pieces)
    print(line)

# Entry point when run as a script
if __name__ == "__main__":
    make_predictions()
