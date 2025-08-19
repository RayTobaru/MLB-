#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
precompute_everything.py

Nightly job that:
  • Caches park factors
  • Builds RE24
  • Trains K/9 ensemble + intervals
  • Fits starter IP regression
  • Negative-binomial dispersions
  • Batter career rates
  • Per-PA isotonic calibrators for H/TB/HR (90d)
  • HR per-game calibrators by hitter archetype (90d/180d)

Exports:
  predict_k9, predict_k9_interval, sample_starter_ip, sample_pa_outcome, shrink_vs_pitcher,
  kt_montecarlo, simulate_full_game, H_theta/H_p/TB_theta/TB_p, NB_K_theta/NB_K_p,
  iso_hit_calibrators, iso_pa_calibrators, hr_game_cals, HR_LAMBDA_SCALE, infer_hr_archetype,
  expected_matchup_xiso, features_base, league_feature_means, LEAGUE_BAT, IP_REG, etc.
"""

import io, pickle, cloudpickle, logging, functools
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from sklearn.linear_model import Ridge, Lasso, PoissonRegressor, QuantileRegressor, LogisticRegression, Ridge as RidgeReg
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgb
import math

from fetch import (
    ID2ABBR, YEAR,
    disk_cache_h2h, fetch_retrosheet_pbp,
    get_recent_bb_stats, get_statcast_batter_features, get_statcast_pitcher_features,
    pitching_stats, batting_stats, clean_name, name_to_mlbam_id, fetch_statcast_raw,
    get_statcast_pitch_data, get_statcast_batter_data, get_batted_ball_profile,
    fetch_yearly_park_factors, fetch_monthly_park_factors,
    get_recent_pitcher_k9, get_recent_pitcher_era, select_high_leverage_reliever,
    empirical_bayes_shrink, empirical_bayes_shrink_era, get_game_weather,
    fetch_boxscore_officials, load_umpire_network_stats, get_catcher_framing_leaderboard,
    framing_runs_for, get_pitch_type_profile, pitcher_mix_last_starts, batter_xiso_by_pitch,
    team_bullpen_hrpa, tail_wind_pct, full_pitching_staff
)

# --- Robust imports so Pylance never flags "undefined" ---
try:
    from multiprocessing.pool import ThreadPool as _ThreadPool
    ThreadPool = _ThreadPool
except Exception:
    ThreadPool = None  # fallback to sequential map below

try:
    from sklearn.feature_selection import RFE as _RFE
    RFE = _RFE
except Exception:
    RFE = None  # fallback: keep lasso-selected features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
CACHE_DIR = Path("./cache"); CACHE_DIR.mkdir(exist_ok=True)
print("CACHE_DIR is:", CACHE_DIR.resolve())

SIMS_K = 5000

# ------------------------ cache helpers ------------------------
def disk_cache(filename):
    def deco(fn):
        @functools.wraps(fn)
        def wrapped(*a, **k):
            p = CACHE_DIR / filename
            if p.exists():
                try:
                    return cloudpickle.loads(p.read_bytes())
                except Exception:
                    pass
            res = fn(*a, **k)
            try:
                p.write_bytes(cloudpickle.dumps(res))
            except Exception:
                pass
            return res
        return wrapped
    return deco

def parquet_cache_df(key_fmt: str):
    def deco(fn):
        @functools.wraps(fn)
        def wrapped(*a, **k):
            fname = key_fmt.format(*a, **k) if "{" in key_fmt else key_fmt
            pq = CACHE_DIR / fname
            pkl = CACHE_DIR / (fname.replace(".parquet", "") + ".pkl")
            if pq.exists():
                try:
                    return pd.read_parquet(pq)
                except Exception:
                    try: pq.unlink()
                    except Exception: pass
            if pkl.exists():
                try:
                    return pickle.loads(pkl.read_bytes())
                except Exception:
                    try: pkl.unlink()
                    except Exception: pass
            df = fn(*a, **k)
            try:
                if isinstance(df, pd.DataFrame):
                    try:
                        df.to_parquet(pq, index=False)
                    except Exception:
                        pkl.write_bytes(pickle.dumps(df))
                else:
                    pkl.write_bytes(pickle.dumps(df))
            except Exception:
                pass
            return df
        return wrapped
    return deco

def _nz_df(df):
    """Return df if it's a non-empty DataFrame, else an empty DataFrame."""
    return df if (isinstance(df, pd.DataFrame) and not df.empty) else pd.DataFrame()

def expected_sp_ip_simple(starter_pid: int) -> float:
    """
    Very stable SP IP expectation using season IP/GS with a small pull to league.
    Falls back to 5.4 IP if we can't see the starter in season stats.
    """
    try:
        ps = pitching_stats(YEAR, qual=0).copy()
        ps["Name_norm"] = ps.Name.apply(clean_name)
        ps["pid"] = ps["Name_norm"].apply(name_to_mlbam_id)
        rec = ps[ps["pid"] == starter_pid]
        if rec.empty:
            return 5.4
        ip = float(rec["IP"].iloc[0] or 0.0)
        gs = float(rec["GS"].iloc[0] or 1.0)
        ip_per_start = ip / max(gs, 1.0)
        # mild shrink to a leaguey 5.4 IP for robustness
        return float(0.8 * ip_per_start + 0.2 * 5.4)
    except Exception:
        return 5.4

def sp_pa_share(starter_pid: int) -> float:
    """
    Approx fraction of a batting team's PAs that occur vs the starter.
    Total team PA per game ~ 38; ~4.2 PA/inning → share ≈ IP/9.
    Clamp to reasonable bounds (15%..95%).
    """
    m = expected_sp_ip_simple(starter_pid)
    return float(np.clip(m / 9.0, 0.15, 0.95))

@disk_cache("recent_evt_feats.pkl")
def build_recent_event_features(days_short:int=1, days_med:int=7):
    """
    Returns {batter_pid: {hr_1d, pa_1d, hr_7d, pa_7d, bbe95_7d, pulled_fly_7d, xwobacon_7d}}
    pulled_fly_7d ≈ pulled + airborne (using bb_type + coarse spray proxy).
    """
    end = datetime.today()
    s1  = (end - timedelta(days=days_short)).strftime("%Y-%m-%d")
    s7  = (end - timedelta(days=days_med)).strftime("%Y-%m-%d")
    e   = end.strftime("%Y-%m-%d")

    # yesterday (or last 1d window)
    d1 = _nz_df(fetch_statcast_raw(s1, e))
    d7 = _nz_df(fetch_statcast_raw(s7, e))

    out = {}
    # --- 1d ---
    if not d1.empty:
        d1["is_pa"] = d1["events"].notna().astype(int)
        g1 = d1.groupby("batter")
        hr1 = g1["events"].apply(lambda s: (s=="home_run").sum()).rename("hr_1d")
        pa1 = g1["is_pa"].sum().rename("pa_1d")
        df1 = pd.concat([hr1, pa1], axis=1).reset_index()
    else:
        df1 = pd.DataFrame(columns=["batter","hr_1d","pa_1d"])

    # --- 7d ---
    if not d7.empty:
        d7 = d7.copy()
        d7["is_pa"] = d7["events"].notna().astype(int)
        d7 = d7.replace([np.inf,-np.inf], np.nan)
        # ensure numeric
        for c in ("launch_speed","launch_angle"):
            if c in d7.columns:
                d7[c] = pd.to_numeric(d7[c], errors="coerce")

        g7   = d7.groupby("batter")
        hr7  = g7["events"].apply(lambda s: (s=="home_run").sum()).rename("hr_7d")
        pa7  = g7["is_pa"].sum().rename("pa_7d")

        # quality contact
        bbe95 = g7.apply(lambda g: (pd.to_numeric(g.launch_speed, errors="coerce")>=95).sum()).rename("bbe95_7d")

        # pulled airborne proxy (coarse but robust to missing spray): pull≈ launch_angle within [-20,20] + fly_ball
        def _pulled_fly(g):
            la_ok = pd.to_numeric(g.launch_angle, errors="coerce").between(-20,20)
            fb_ok = (g.bb_type=="fly_ball")
            return int((la_ok & fb_ok).sum())
        pfly = g7.apply(_pulled_fly).rename("pulled_fly_7d")

        # xwOBA on contact
        xw = g7.apply(lambda g: pd.to_numeric(g.get("estimated_woba_using_speedangle", pd.Series(dtype=float)), errors="coerce").mean()) \
               .rename("xwobacon_7d")

        df7 = pd.concat([hr7, pa7, bbe95, pfly, xw], axis=1).reset_index()
    else:
        df7 = pd.DataFrame(columns=["batter","hr_7d","pa_7d","bbe95_7d","pulled_fly_7d","xwobacon_7d"])

    # merge
    df = pd.merge(df1, df7, how="outer", on="batter").fillna(0)
    for _, r in df.iterrows():
        out[int(r["batter"])] = {
            "hr_1d":        int(r.get("hr_1d", 0)),
            "pa_1d":        int(r.get("pa_1d", 0)),
            "hr_7d":        int(r.get("hr_7d", 0)),
            "pa_7d":        int(r.get("pa_7d", 0)),
            "bbe95_7d":     int(r.get("bbe95_7d", 0)),
            "pulled_fly_7d":int(r.get("pulled_fly_7d", 0)),
            "xwobacon_7d":  float(r.get("xwobacon_7d", 0.0)) if pd.notna(r.get("xwobacon_7d", np.nan)) else 0.0
        }
    return out

RECENT_EVT = build_recent_event_features()

def get_recent_evt_feats(batter_pid) -> dict:
    default = {"hr_1d":0,"pa_1d":0,"hr_7d":0,"pa_7d":0,"bbe95_7d":0,"pulled_fly_7d":0,"xwobacon_7d":0.0}
    try:
        pid = int(batter_pid)
    except Exception:
        return default
    return RECENT_EVT.get(pid, default)

@disk_cache("recent_pitcher_hr_allow.pkl")
def build_pitcher_recent_hr_allow(days:int=30):
    """
    Returns {pitcher_id: {pa_30d, hr_30d, hrpa_30d, bbe95_allowed, pulled_fly_allowed}}
    """
    end = datetime.today()
    s   = (end - timedelta(days=days)).strftime("%Y-%m-%d")
    e   = end.strftime("%Y-%m-%d")
    df  = _nz_df(fetch_statcast_raw(s, e))
    if df.empty:
        return {}

    df = df.copy()
    df["is_pa"] = df["events"].notna().astype(int)
    for c in ("launch_speed","launch_angle"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    g   = df.groupby("pitcher")
    pa  = g["is_pa"].sum().rename("pa")
    hr  = g["events"].apply(lambda s: (s=="home_run").sum()).rename("hr")
    b95 = g.apply(lambda g2: (pd.to_numeric(g2.launch_speed, errors="coerce")>=95).sum()).rename("bbe95")

    # pulled airborne proxy per pitcher
    def _pulled_fly(g2: pd.DataFrame) -> int:
        la = pd.to_numeric(g2.get("launch_angle"), errors="coerce")
        bt = g2.get("bb_type")
        if la is None or bt is None:
            return 0
        # pulled ≈ launch_angle in [-20,20]; airborne = fly_ball
        return int(((bt == "fly_ball") & la.between(-20, 20)).sum())

    pf  = g.apply(lambda g2: _pulled_fly(g2)).rename("pulled_fly")
    dd  = pd.concat([pa,hr,b95,pf], axis=1).reset_index()
    dd["hrpa"] = dd["hr"]/dd["pa"].clip(lower=1)
    return {int(r.pitcher): {"pa_30d": int(r.pa), "hr_30d": int(r.hr),
                             "hrpa_30d": float(r.hrpa), "bbe95_allowed": int(r.bbe95),
                             "pulled_fly_allowed": int(r.pulled_fly)} for _, r in dd.iterrows()}

PIT_RECENT = build_pitcher_recent_hr_allow()

def get_pitcher_recent(pid) -> dict:
    default = {"pa_30d":0,"hr_30d":0,"hrpa_30d":0.0,"bbe95_allowed":0,"pulled_fly_allowed":0}
    try:
        pid = int(pid)
    except Exception:
        return default
    return PIT_RECENT.get(pid, default)

def batter_recent_multiplier(evt: dict) -> float:
    """
    Conservative 'hot-contact' bump:
      • HR yesterday: +3%
      • Hard-hit per PA (≥95 mph) last 7d: up to ±10%
      • Pulled airborne contact per PA last 7d: up to ±12%
      • xwOBAcon last 7d: up to ±12%
    """
    m = 1.0
    if evt.get("hr_1d", 0) > 0:
        m *= 1.03
    pa7 = max(float(evt.get("pa_7d", 0)), 1.0)
    hh_rate   = float(evt.get("bbe95_7d", 0)) / pa7
    pfly_rate = float(evt.get("pulled_fly_7d", 0)) / pa7
    xw = float(evt.get("xwobacon_7d", 0.0))

    # league-ish anchors: hh/PA≈0.10, pulled_fly/PA≈0.035, xwOBAcon≈0.360
    m *= float(np.clip(1.0 + 0.60*(hh_rate - 0.10), 0.90, 1.10))
    m *= float(np.clip(1.0 + 0.70*(pfly_rate - 0.035), 0.90, 1.12))
    if math.isfinite(xw) and xw > 0:
        m *= float(np.clip(1.0 + 0.80*(xw - 0.360), 0.90, 1.12))
    return float(np.clip(m, 0.85, 1.20))

def pitcher_recent_multiplier(rec: dict, league_hrpa: float) -> float:
    """
    If a pitcher has been allowing more HR/PA over last 30d than league, nudge up to ±20%.
    """
    hrpa = float(rec.get("hrpa_30d", 0.0))
    if hrpa <= 0 or league_hrpa <= 0:
        return 1.0
    rel = (hrpa/league_hrpa) - 1.0
    return float(np.clip(1.0 + 0.50*rel, 0.80, 1.20))


# ------------------------ league context ------------------------
_LEAGUE_RE24 = {
    (0,0):0.485,(0,1):0.856,(0,2):1.089,(0,3):1.391,(0,4):1.397,(0,5):1.761,(0,6):2.004,(0,7):2.390,
    (1,0):0.243,(1,1):0.533,(1,2):0.781,(1,3):1.109,(1,4):1.129,(1,5):1.485,(1,6):1.744,(1,7):2.071,
    (2,0):0.109,(2,1):0.294,(2,2):0.453,(2,3):0.709,(2,4):0.719,(2,5):1.075,(2,6):1.315,(2,7):1.676
}

@disk_cache("league_bat_rates.pkl")
def compute_league_bat_rates(year: int) -> dict:
    df = batting_stats(year, qual=0).copy()
    df["1B_only"] = df.H - df["2B"] - df["3B"] - df.HR
    totPA = max(float(df.PA.sum()), 1.0)
    rates = {
        "1B": df["1B_only"].sum()/totPA,
        "2B": df["2B"].sum()/totPA,
        "3B": df["3B"].sum()/totPA,
        "HR": df.HR.sum()/totPA,
        "RBI": df.RBI.sum()/totPA,
        "R": df.R.sum()/totPA
    }
    rates["TB"] = rates["1B"] + 2*rates["2B"] + 3*rates["3B"] + 4*rates["HR"]
    return rates

LEAGUE_BAT = compute_league_bat_rates(YEAR)

def _fit_nb_dispersion(arr: np.ndarray):
    mu, var = np.mean(arr), np.var(arr)
    if var > mu:
        theta = max(mu**2/(var - mu), 0.1)
        p = theta/(theta + mu)
    else:
        theta, p = np.inf, 0.0
    return theta, p

@disk_cache("nb_dispersion_bat.pkl")
def compute_batter_dispersions(year: int):
    df = batting_stats(year, qual=0).copy()
    df['1B'] = df.H - df['2B'] - df['3B'] - df.HR
    df['TB'] = df['1B'] + 2*df['2B'] + 3*df['3B'] + 4*df.HR
    H_theta, H_p = _fit_nb_dispersion(df.H.values)
    TB_theta, TB_p = _fit_nb_dispersion(df.TB.values)
    return H_theta, H_p, TB_theta, TB_p

H_theta, H_p, TB_theta, TB_p = compute_batter_dispersions(YEAR)

@disk_cache("nb_dispersion_k.pkl")
def compute_strikeout_dispersion(year: int):
    df = pitching_stats(year, qual=0)
    so = df.SO.replace([np.inf, np.nan], 0).values
    return _fit_nb_dispersion(so)

NB_K_theta, NB_K_p = compute_strikeout_dispersion(YEAR)
NB_K_theta = NB_K_theta * 0.5  # mild inflation

# ------------------------ RE24 ------------------------
def compute_re24_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    df = pbp.copy()
    df["PA_ID"] = df.groupby("game_pk").cumcount()
    def runs_after(g):
        s = 0
        for _, r in g.iterrows():
            s += r.runs_scored
            if r.outs_before >= 3:
                break
        return s
    df["inning_id"] = df.game_pk.astype(str) + "_" + df.inning.astype(str)
    rows = []
    for (_, pa), g in df.groupby(["inning_id", "PA_ID"]):
        rows.append((int(g.outs_before.iloc[0]), int(g.bases_before.iloc[0]), runs_after(g)))
    re = pd.DataFrame(rows, columns=["outs", "bases", "runs24"])
    return re.groupby(["outs", "bases"]).runs24.mean().reset_index().rename(columns={"runs24": "RE24"})

@disk_cache("run_exp_matrix.pkl")
def build_run_expectancy_matrix():
    try:
        pbp = fetch_retrosheet_pbp(YEAR)
    except Exception as e:
        logging.warning(f"RE24 fetch error: {e}")
        pbp = pd.DataFrame()
    if not pbp.empty:
        re = compute_re24_from_pbp(pbp)
        M = np.zeros((25, 25))
        for _, r in re.iterrows():
            M[int(r.outs)*8 + int(r.bases), :] = r.RE24
        return M
    M = np.zeros((25, 25))
    for (o, b), v in _LEAGUE_RE24.items():
        M[o*8 + b, :] = v
    return M

RUN_EXP_MATRIX = build_run_expectancy_matrix()

# ------------------------ K/9 ensemble ------------------------
@disk_cache("k9_ensemble_v2.pkl")
def train_ensemble_k9_model():
    dfs = []
    for y in range(YEAR-4, YEAR+1):
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):  # quiet
                df = pitching_stats(y, qual=0)
        except Exception:
            continue
        df = df.copy()
        df["Season"] = y
        df["IP_per_start"] = (df.IP / df.GS).fillna(0)
        df["Name_norm"] = df.Name.apply(clean_name)
        df["pid"] = df.Name_norm.apply(name_to_mlbam_id)
        dfs.append(df)
    if not dfs:
        return (lambda r: 8.0), (lambda r: (8.0, 8.0)), [], {}
    LP_full = pd.concat(dfs, ignore_index=True)
    seasons = sorted(LP_full.Season.unique())
    if len(seasons) < 2:
        means = {f: LP_full[f].mean() for f in LP_full.columns if f not in ("Season","Name_norm","pid")}
        return (lambda row: means.get("K%", 8.0)), (lambda row: (8.0,8.0)), [], means

    pids = [int(x) for x in LP_full.pid.dropna().unique()]
    if ThreadPool:
        with ThreadPool(8) as pool:
            sc_feats = pool.map(get_statcast_pitcher_features, pids)
            pt_feats = pool.map(get_pitch_type_profile,      pids)
    else:
        sc_feats = [get_statcast_pitcher_features(pid) for pid in pids]
        pt_feats = [get_pitch_type_profile(pid) for pid in pids]

    df_sc = pd.DataFrame(sc_feats, index=pids).reset_index().rename(columns={"index": "pid"})
    df_pt = pd.DataFrame(pt_feats, index=pids).reset_index().rename(columns={"index": "pid"})
    bb7   = [get_recent_bb_stats(pid,  7) or {} for pid in pids]
    bb14  = [get_recent_bb_stats(pid, 14) or {} for pid in pids]
    df_bb7  = pd.DataFrame(bb7,  index=pids).reset_index().rename(columns={"index": "pid"})
    df_bb14 = pd.DataFrame(bb14, index=pids).reset_index().rename(columns={"index": "pid"})

    LP_full = (LP_full.merge(df_sc,  on="pid", how="left")
                      .merge(df_pt,  on="pid", how="left")
                      .merge(df_bb7, on="pid", how="left")
                      .merge(df_bb14,on="pid", how="left"))

    LP_full["Shrunk_K9"]  = empirical_bayes_shrink(    LP_full["K/9"], LP_full["IP"])
    LP_full["Shrunk_ERA"] = empirical_bayes_shrink_era(LP_full["ERA"], LP_full["IP"])

    for c in ["ev_mean_7d","ev_std_7d","la_mean_7d","barrel_pct_7d",
          "ev_mean_14d","ev_std_14d","la_mean_14d","barrel_pct_14d"]:
        if c in LP_full.columns:
            LP_full[c] = LP_full[c].fillna(0.0)
        else:
            LP_full[c] = 0.0

    mpf = fetch_monthly_park_factors(YEAR); m = datetime.today().month
    LP_full["wind_tail_pct"]  = 0.0
    LP_full["park_hr_factor"] = LP_full["Team"].map(lambda t: mpf.get(t,{}).get(m,{}).get("HR",100)/100.0)

    base_feats = ["SwStr%","K%","FIP","BB%","WHIP","IP_per_start","Shrunk_K9","Shrunk_ERA",
                  "FB_pct","OS_pct","SwStr_FB","SwStr_OS",
                  "ev_mean_7d","ev_std_7d","la_mean_7d","barrel_pct_7d",
                  "ev_mean_14d","ev_std_14d","la_mean_14d","barrel_pct_14d","wind_tail_pct","park_hr_factor"]
    for c in ("FB_pct","OS_pct","SwStr_FB","SwStr_OS"):
        if c in LP_full.columns:
            LP_full[c] = LP_full[c].fillna(0.0)
        else:
            LP_full[c] = 0.0

    X = LP_full[base_feats].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = LP_full["K/9"].fillna(0)

    lasso = Lasso(alpha=0.01, max_iter=5000).fit(X, y)
    sel   = [f for f,coef in zip(base_feats, lasso.coef_) if abs(coef) > 1e-4] or base_feats

    if RFE:
        rfe = RFE(RandomForestRegressor(n_estimators=100, random_state=0),
                  n_features_to_select=min(30, len(sel))).fit(X[sel], y)
        final_feats = [f for f,s in zip(sel, rfe.support_) if s]
    else:
        final_feats = sel

    yrs = sorted(LP_full.Season.unique()); splits = []
    for i in range(len(yrs)-1):
        tr = LP_full[LP_full.Season.isin(yrs[:i+1])].index
        va = LP_full[LP_full.Season==yrs[i+1]].index
        splits.append((tr,va))

    def rs(mdl,params):
        return RandomizedSearchCV(mdl, params, n_iter=10, cv=splits,
                                  scoring="neg_mean_squared_error", random_state=0).fit(X[final_feats], y)

    learners = [
        GradientBoostingRegressor(random_state=0),
        xgb.XGBRegressor(tree_method="hist", objective="reg:squarederror", random_state=0, verbosity=0),
        RandomForestRegressor(random_state=0),
        PoissonRegressor(alpha=1e-3, max_iter=1000),
        lgb.LGBMRegressor(random_state=0, verbosity=-1)
    ]
    params = [
        {"n_estimators":[100,200],"max_depth":[3,5],"learning_rate":[0.01,0.1]},
        {"n_estimators":[100,200],"max_depth":[3,5],"learning_rate":[0.01,0.1]},
        {"n_estimators":[100,200],"max_depth":[None,5]},
        {"alpha":[1e-3,1e-2,1e-1]},
        {"n_estimators":[100,200],"max_depth":[3,5],"learning_rate":[0.01,0.1]}
    ]

    base = []
    for mdl, p in zip(learners, params):
        best = rs(mdl, p).best_estimator_
        best.fit(X[final_feats], y)
        base.append(best)

    oof = np.zeros((len(X), len(base)))
    for tr, va in splits:
        clones = [type(m)(**m.get_params()) for m in base]
        for i, mdl in enumerate(clones):
            mdl.fit(X.iloc[tr][final_feats], y.iloc[tr])
            oof[va, i] = mdl.predict(X.iloc[va][final_feats])

    meta_mean = Ridge(alpha=1.0).fit(oof, y)
    meta_q10  = QuantileRegressor(quantile=0.10, alpha=0).fit(oof, y)
    meta_q90  = QuantileRegressor(quantile=0.90, alpha=0).fit(oof, y)
    iso_k9    = IsotonicRegression(out_of_bounds="clip").fit(meta_mean.predict(oof), y)

    league_feature_means = {f: X[f].mean() for f in final_feats}
    logging.info("▶ K/9 ensemble ready.")

    def predict_k9(row: pd.Series) -> float:
        v = row[final_feats].fillna(0).values.reshape(1, -1)
        p = np.array([m.predict(v)[0] for m in base]).reshape(1, -1)
        return float(iso_k9.predict([float(meta_mean.predict(p)[0])])[0])

    def predict_k9_interval(row: pd.Series) -> tuple:
        v = row[final_feats].fillna(0).values.reshape(1, -1)
        p = np.array([m.predict(v)[0] for m in base]).reshape(1, -1)
        return float(meta_q10.predict(p)[0]), float(meta_q90.predict(p)[0])

    return predict_k9, predict_k9_interval, final_feats, league_feature_means

# ------------------------ Hits/TB ensemble ------------------------
@disk_cache("hit_ensemble_v2.pkl")
def train_ensemble_hit_model():
    dfs = []
    for y in range(YEAR-4, YEAR+1):
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                df = batting_stats(y, qual=0)
        except Exception:
            continue
        df = df.copy()
        df["Season"]    = y
        df["Name_norm"] = df.Name.apply(clean_name)
        df["pid"]       = df.Name_norm.apply(name_to_mlbam_id)
        dfs.append(df)
    if not dfs:
        return (lambda r: 0.0), (lambda r: (0.0,0.0)), [], {}

    HB = pd.concat(dfs, ignore_index=True)
    HB['1B'] = HB.H - HB['2B'] - HB['3B'] - HB.HR
    HB['TB'] = HB['1B'] + 2*HB['2B'] + 3*HB['3B'] + 4*HB.HR
    HB[['1B','2B','3B','HR','TB']] = HB[['1B','2B','3B','HR','TB']].fillna(0)

    pids = [int(x) for x in HB.pid.dropna().unique()]
    if ThreadPool:
        with ThreadPool(8) as pool:
            feats = pool.map(get_statcast_batter_features, pids)
    else:
        feats = [get_statcast_batter_features(pid) for pid in pids]
    df_sc = pd.DataFrame(feats, index=pids).reset_index().rename(columns={"index":"pid"})
    HB = HB.merge(df_sc, on="pid", how="left")

    base_feats = ["PA","BB","HBP","SB","ISO"] + [c for c in df_sc.columns if c!="pid"]
    Xh = HB[base_feats].fillna(0)
    yh = HB["H"]

    lasso = Lasso(alpha=0.01, max_iter=5000).fit(Xh, yh)
    sel   = [f for f,coef in zip(base_feats, lasso.coef_) if abs(coef) > 1e-4] or base_feats

    if RFE:
        rfe = RFE(RandomForestRegressor(n_estimators=100, random_state=0),
                  n_features_to_select=min(30, len(sel))).fit(Xh[sel], yh)
        feats_fin = [f for f,s in zip(sel, rfe.support_) if s]
    else:
        feats_fin = sel

    yrs = sorted(HB.Season.unique())
    splits = []
    for i in range(len(yrs)-1):
        tr = HB[HB.Season.isin(yrs[:i+1])].index
        va = HB[HB.Season==yrs[i+1]].index
        splits.append((tr,va))

    def rs(mdl, params):
        return RandomizedSearchCV(mdl, params, n_iter=10, cv=splits,
                                  scoring="neg_mean_squared_error", random_state=0).fit(Xh[feats_fin], yh)

    learners = [
        GradientBoostingRegressor(random_state=0),
        xgb.XGBRegressor(tree_method="hist", objective="reg:squarederror", random_state=0, verbosity=0),
        RandomForestRegressor(random_state=0),
        PoissonRegressor(alpha=1e-3, max_iter=1000),
        lgb.LGBMRegressor(random_state=0, verbosity=-1)
    ]
    params = [
        {"n_estimators":[100,200],"max_depth":[3,5],"learning_rate":[0.01,0.1]},
        {"n_estimators":[100,200],"max_depth":[3,5],"learning_rate":[0.01,0.1]},
        {"n_estimators":[100,200],"max_depth":[None,5]},
        {"alpha":[1e-3,1e-2,1e-1]},
        {"n_estimators":[100,200],"max_depth":[3,5],"learning_rate":[0.01,0.1]}
    ]

    base = []
    for mdl, p in zip(learners, params):
        best = rs(mdl, p).best_estimator_
        best.fit(Xh[feats_fin], yh)
        base.append(best)

    oofh = np.zeros((len(Xh), len(base)))
    for tr, va in splits:
        clones = [type(m)(**m.get_params()) for m in base]
        for i, mdl in enumerate(clones):
            mdl.fit(Xh.iloc[tr][feats_fin], yh.iloc[tr])
            oofh[va, i] = mdl.predict(Xh.iloc[va][feats_fin])

    mr_mean = Ridge(alpha=1.0).fit(oofh, yh)
    mr_q10  = QuantileRegressor(quantile=0.10, alpha=0).fit(oofh, yh)
    mr_q90  = QuantileRegressor(quantile=0.90, alpha=0).fit(oofh, yh)

    iso_cal = {}
    raw = mr_mean.predict(oofh)
    thresholds = {
        "P(Hits≥1)":("H",1),"P(Hits≥2)":("H",2),"P(Hits≥3)":("H",3),"P(Hits≥4)":("H",4),
        "P(1B≥1)":("1B",1),"P(2B≥1)":("2B",1),"P(3B≥1)":("3B",1),"P(HR≥1)":("HR",1),
        "P(TB≥1)":("TB",1),"P(TB≥2)":("TB",2),"P(TB≥3)":("TB",3),"P(TB≥4)":("TB",4),
        "P(RBI≥1)":("RBI",1),"P(Run≥1)":("R",1)
    }
    for ev, (col, cut) in thresholds.items():
        ybin = (HB[col] >= cut).astype(int).values
        iso_cal[ev] = IsotonicRegression(out_of_bounds="clip").fit(raw, ybin)

    league_hit_means = {f: Xh[f].mean() for f in feats_fin}
    logging.info("▶ Hit/TB ensemble ready.")

    def predict_hits(row: pd.Series) -> float:
        v = row[feats_fin].fillna(0).values.reshape(1, -1)
        p = np.array([m.predict(v)[0] for m in base]).reshape(1, -1)
        return float(mr_mean.predict(p)[0])

    def predict_hits_interval(row: pd.Series) -> tuple:
        v = row[feats_fin].fillna(0).values.reshape(1, -1)
        p = np.array([m.predict(v)[0] for m in base]).reshape(1, -1)
        return float(mr_q10.predict(p)[0]), float(mr_q90.predict(p)[0])

    return predict_hits, predict_hits_interval, feats_fin, league_hit_means, iso_cal

# ------------------------ HR models (stack + micros) ------------------------
@disk_cache("hr_feat_matrix.pkl")
def _build_hr_matrix():
    Xs, ys = [], []
    for y in range(YEAR-3, YEAR+1):
        sb = fetch_statcast_raw(f"{y}-03-01", f"{y}-11-01")
        if sb is None or sb.empty:
            continue

        sb = sb.rename(columns={"launch_speed":"exit_velocity"})
        # robust against weird NaNs / infs (keep bb_type & events non-null)
        sb = sb.replace([np.inf, -np.inf], np.nan)
        sb[["exit_velocity","launch_angle"]] = sb[["exit_velocity","launch_angle"]].fillna(0.0).astype(float)
        sb = sb.dropna(subset=["bb_type","events"])

        sb["HR"] = (sb.events=="home_run").astype(int)
        sb["barrel_pct"] = sb.get("barrel", 0.0)

        hr_sum  = sb.groupby("game_pk")["HR"].transform("sum")
        fly_sum = sb.groupby("game_pk").apply(lambda g: (g.bb_type=="fly_ball").sum()).reindex(sb.index).fillna(0)
        sb["HR_FB_rate"] = (hr_sum / fly_sum.replace(0, np.nan)).fillna(0.0)

        sb["pull_pct"]   = sb.launch_angle.between(-20,20).astype(int)
        sb["park_hr_factor"] = 1.0

        feats = ["exit_velocity","launch_angle","barrel_pct","HR_FB_rate","pull_pct","park_hr_factor"]
        Xs.append(sb[feats].fillna(0.0))
        ys.append(sb["HR"])
    return pd.concat(Xs, ignore_index=True), pd.concat(ys, ignore_index=True)

@disk_cache("hr_base_models.pkl")
def _train_hr_bases():
    X, y = _build_hr_matrix()
    learners = [
        GradientBoostingClassifier(n_estimators=120, random_state=0),
        RandomForestClassifier(n_estimators=300, random_state=0),
        LogisticRegression(max_iter=2500),
        lgb.LGBMClassifier(n_estimators=250, random_state=0),
        xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=250, random_state=0)
    ]
    return [m.fit(X, y) for m in learners]

@disk_cache("hr_feat_cols.pkl")
def _get_hr_feat_cols():
    X, _ = _build_hr_matrix()
    return tuple(X.columns)

@disk_cache("hr_stack_meta.pkl")
def _train_hr_stacker():
    X, y = _build_hr_matrix()
    base = _train_hr_bases()
    tscv = TimeSeriesSplit(n_splits=4)
    M = np.zeros((len(X), len(base)))
    for tr, va in tscv.split(X):
        for i, mdl in enumerate(base):
            m2 = clone(mdl)
            m2.fit(X.iloc[tr], y.iloc[tr])
            M[va, i] = m2.predict_proba(X.iloc[va])[:, 1]
    meta = RidgeReg(alpha=1.0).fit(M, y)
    raw  = meta.predict(M)
    iso  = IsotonicRegression(out_of_bounds="clip").fit(raw, y)
    return base, meta, iso

def predict_hr_proba(feat_row: pd.Series) -> float:
    base, m, iso = _train_hr_stacker()
    cols = _get_hr_feat_cols()
    xi = feat_row.reindex(cols, fill_value=0).values.reshape(1, -1)
    preds = np.array([mdl.predict_proba(xi)[:, 1][0] for mdl in base]).reshape(1, -1)
    return float(iso.predict([m.predict(preds)[0]])[0])

@disk_cache("hr_count_pt_model.pkl")
def train_hr_count_pt_model():
    sb = fetch_statcast_raw(f"{YEAR-1}-03-01", f"{YEAR-1}-11-01").dropna(subset=["balls","strikes","pitch_type","events"])
    sb["HR"] = (sb.events=="home_run").astype(int)
    sb["count"] = sb.balls.astype(str) + "-" + sb.strikes.astype(str)
    X = pd.get_dummies(sb[["count","pitch_type"]], drop_first=True)
    y = sb.HR
    return LogisticRegression(max_iter=1500).fit(X, y), X.columns.tolist()

_hr_count_pt_model, _hr_count_pt_cols = train_hr_count_pt_model()
def predict_hr_count_pt(feat_row: pd.Series) -> float:
    X = feat_row.reindex(_hr_count_pt_cols, fill_value=0).values.reshape(1, -1)
    return float(_hr_count_pt_model.predict_proba(X)[:,1][0])

@disk_cache("hr_hardhit_model.pkl")
def train_hr_hardhit_model():
    df = fetch_statcast_raw(f"{YEAR-1}-03-01", f"{YEAR-1}-11-01").dropna(subset=["launch_speed","events"])
    df["HR"] = (df.events=="home_run").astype(int)
    return LogisticRegression(max_iter=600).fit(df[["launch_speed"]], df.HR)
_hr_hardhit_model = train_hr_hardhit_model()
def predict_hr_hardhit(feat_row: pd.Series) -> float:
    return float(_hr_hardhit_model.predict_proba([[feat_row.get("exit_velocity", 0.0)]])[:,1][0])

@disk_cache("hr_2swhiff_model.pkl")
def train_hr_2swhiff_model():
    df = fetch_statcast_raw(f"{YEAR-1}-03-01", f"{YEAR-1}-11-01")
    df = df[(df.strikes==2) & df.description.notnull()]
    df["whiff"] = df.description.str.contains("swinging_strike").astype(int)
    df["HR"]    = (df.events=="home_run").astype(int)
    return LogisticRegression(max_iter=600).fit(df[["whiff"]], df.HR)
_hr_2swhiff_model = train_hr_2swhiff_model()
def predict_hr_2swhiff(feat_row: pd.Series) -> float:
    return float(_hr_2swhiff_model.predict_proba([[feat_row.get("hr_rate_2S", 0.0)]])[:,1][0])

@disk_cache("hr_pullangle_model.pkl")
def train_hr_pullangle_model():
    df = fetch_statcast_raw(f"{YEAR-1}-03-01", f"{YEAR-1}-11-01").dropna(subset=["launch_angle","events"])
    df["pull"] = df.launch_angle.between(-20,20).astype(int)
    df["HR"]   = (df.events=="home_run").astype(int)
    return LogisticRegression(max_iter=600).fit(df[["pull"]], df.HR)
_hr_pullangle_model = train_hr_pullangle_model()
def predict_hr_pullangle(feat_row: pd.Series) -> float:
    return float(_hr_pullangle_model.predict_proba([[feat_row.get("pull_pct", 0.0)]])[:,1][0])

# ------------------------ materialize main models ------------------------
predict_k9,  predict_k9_interval,  k9_model_feats,  league_feature_means  = train_ensemble_k9_model()
predict_hits,predict_hits_interval,hit_model_feats, league_hit_means, iso_hit_calibrators = train_ensemble_hit_model()

# ------------------------ extra helpers ------------------------
features_base     = k9_model_feats  + ["Recent_K9_7d","Recent_K9_14d","Recent_ERA_7d","Recent_ERA_14d","Days_Rest","Travel_Days","IP_7d","IP_14d","Apps_7d","Apps_14d"]
features_hit_base = hit_model_feats + ["IP_7d","Apps_7d","IP_14d","Apps_14d"]

@disk_cache("h2h_lr_splits_filtered.pkl")
def compute_h2h_splits(): return {}

def expected_matchup_xiso(pit_pid:int, bat_pid:int) -> float:
    mix = pitcher_mix_last_starts(pit_pid) or {}
    bx  = batter_xiso_by_pitch(bat_pid)    or {}
    if not mix or not bx:
        return 0.0
    return float(sum(w * bx.get(pt, 0.0) for pt, w in mix.items()))

def batter_game_hr_prob(
    batter_pid: int,
    batter_stand: str | None,
    batting_team: str,         # e.g., "SEA"
    opp_team: str,             # the fielding team (their SP/bullpen)
    opp_starter_pid: int,
    exp_pa: float,
    game_pk: int | None = None
) -> float:
    """
    Estimate P(HR≥1) for a batter in a game by blending vs-starter and vs-bullpen
    HR/PA, then applying park+wind and matchup xISO, and finally calibrating
    with the per-PA isotonic (HR,1).
    """
    # --- baseline batter HR/PA over 180d (shrunk to league)
    end = datetime.today()
    start = (end - timedelta(days=180)).strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")
    sb = fetch_statcast_raw(start, end_s)
    league_hr = max(LEAGUE_BAT.get("HR", 0.025), 1e-6)
    if sb is None or sb.empty:
        base_hrpa = league_hr
    else:
        b = sb[sb["batter"] == batter_pid]
        pa_b = int(b["events"].notna().sum())
        hr_b = int((b["events"] == "home_run").sum())
        prior_pa = 120
        base_hrpa = ((hr_b + league_hr * prior_pa) / (pa_b + prior_pa)) if (pa_b + prior_pa) > 0 else league_hr

    # --- opponent side: SP vs-side HR allowance & bullpen HR/PA
    sp_feats = get_statcast_pitcher_features(opp_starter_pid) or {}
    side = (batter_stand or "R").upper()
    if side.startswith("L"):
        sp_hr = float(sp_feats.get("vs_L_HR_rate", np.nan))
    else:
        sp_hr = float(sp_feats.get("vs_R_HR_rate", np.nan))
    f_sp = (sp_hr / league_hr) if (np.isfinite(sp_hr) and sp_hr > 0) else 1.0

    bp_hr, _ = team_bullpen_hrpa(opp_team, exclude_pid=opp_starter_pid)
    f_bp = (bp_hr / league_hr) if bp_hr > 0 else 1.0

    share_sp = sp_pa_share(opp_starter_pid)
    p_pitch = base_hrpa * (share_sp * f_sp + (1.0 - share_sp) * f_bp)

    # --- matchup xISO bump (pitch-mix × batter-by-pitch)
    try:
        xiso = expected_matchup_xiso(opp_starter_pid, batter_pid)
        # elasticity ~ ±30% for roughly ±0.08 ISO around ~.160
        bump = 1.0 + 0.5 * ((xiso - 0.160) / 0.160)
        p_pitch *= float(np.clip(bump, 0.7, 1.3))
    except Exception:
        pass

    # --- park & wind
    mon = datetime.today().month
    pf = fetch_monthly_park_factors(YEAR) or {}
    pf_hr = (pf.get(opp_team, {}).get(mon, {}).get("HR", 100) / 100.0)
    wind_mul = 1.0 + float(tail_wind_pct(game_pk or 0))  # 0..~0.3
    p_adj = p_pitch * pf_hr * wind_mul

    # --- per-game P(HR≥1)
    lam = float(exp_pa) * float(p_adj) * float(HR_LAMBDA_SCALE)
    lam = max(0.0, min(lam, 5.0))
    if ("HR", 1) in iso_pa_calibrators:
        p_game = float(iso_pa_calibrators[("HR", 1)].predict([lam])[0])
    else:
        p_game = float(1.0 - np.exp(-lam))
    return float(np.clip(p_game, 0.0, 0.999))

# ------------------------ IP regression & sampler ------------------------
@disk_cache("ip_regression.pkl")
def fit_ip_regression(year:int):
    df = pitching_stats(year, qual=0).dropna(subset=["IP","K/9"])
    df["Recent_K9_5"]  = df["K/9"].rolling(5,  min_periods=1).mean()
    df["Recent_K9_10"] = df["K/9"].rolling(10, min_periods=1).mean()
    return Ridge(alpha=1.0).fit(df[["IP","Recent_K9_5","Recent_K9_10"]], df.IP)

IP_REG = fit_ip_regression(YEAR)

def sample_starter_ip(sip: float, rec_k9: float) -> float:
    mu = IP_REG.predict([[sip, rec_k9, rec_k9]])[0]
    return float(np.clip(np.random.normal(mu, 0.8), 0.1, sip))

# ------------------------ batter career + recent splits ------------------------
@disk_cache("batter_career_rates.pkl")
def build_batter_career_rates(years:list, weights:list=None) -> pd.DataFrame:
    if weights is None:
        weights = np.linspace(0.1, 0.4, len(years))
    frames = []
    for y, w in zip(years, weights):
        try:
            bat = batting_stats(y, qual=0)
        except Exception:
            continue
        bat = bat.copy()
        bat["Name_norm"] = bat.Name.apply(clean_name)
        bat["1B"] = bat.H - bat["2B"] - bat["3B"] - bat.HR
        bat["TB"] = bat["1B"] + 2*bat["2B"] + 3*bat["3B"] + 4*bat.HR
        bat["weight"] = w
        frames.append(bat[["Name_norm","PA","H","1B","2B","3B","HR","TB","BB","HBP","RBI","R","weight"]])
    df = pd.concat(frames, ignore_index=True)
    df["wPA"] = df.PA * df.weight
    agg = df.groupby("Name_norm").apply(lambda g: pd.Series({
        "PA":       g.wPA.sum(),
        "H_rate":   (g.H   * g.weight).sum()/g.wPA.sum() if g.wPA.sum()>0 else 0,
        "1B_rate":  (g["1B"]* g.weight).sum()/g.wPA.sum() if g.wPA.sum()>0 else 0,
        "2B_rate":  (g["2B"]* g.weight).sum()/g.wPA.sum() if g.wPA.sum()>0 else 0,
        "3B_rate":  (g["3B"]* g.weight).sum()/g.wPA.sum() if g.wPA.sum()>0 else 0,
        "HR_rate":  (g.HR  * g.weight).sum()/g.wPA.sum() if g.wPA.sum()>0 else 0,
        "TB_rate":  (g.TB  * g.weight).sum()/g.wPA.sum() if g.wPA.sum()>0 else 0,
        "RBI_rate": (g.RBI * g.weight).sum()/g.wPA.sum() if g.wPA.sum()>0 else 0,
        "R_rate":   (g.R   * g.weight).sum()/g.wPA.sum() if g.wPA.sum()>0 else 0,
    })).reset_index()
    return agg

BATTER_CAREER = build_batter_career_rates([YEAR-3, YEAR-2, YEAR-1, YEAR])

@disk_cache("recent_splits.pkl")
def build_recent_splits(days:int=30) -> dict:
    return {}

RECENT_SPLITS = build_recent_splits()

def shrink_vs_pitcher(obs_rate, obs_pa, fallback_rate, prior_pa:int=2) -> float:
    pa = max(float(obs_pa), 0.0)
    prior = max(float(prior_pa), 0.0)
    if pa <= 0:
        return float(fallback_rate)
    return float((float(obs_rate)*pa + float(fallback_rate)*prior) / (pa + prior))

# ------------------------ K Monte Carlo helpers ------------------------
def sample_pa_outcome(rate_dict: dict) -> str:
    allowed = ("1B","2B","3B","HR","BB","HBP")
    probs = {k: float(rate_dict.get(k, 0.0)) for k in allowed}
    s = float(sum(v for v in probs.values() if np.isfinite(v)))
    s = float(np.clip(s, 0.0, 0.999))  # ceiling to avoid negative OUT mass
    p_out = 1.0 - s

    events  = list(probs.keys()) + ["OUT"]
    weights = np.array(list(probs.values()) + [p_out], dtype=float)
    weights /= weights.sum() if weights.sum() > 0 else 1.0
    return np.random.choice(events, p=weights)

def kt_montecarlo(r:dict, lineup:list) -> dict:
    base   = r["Pred_K9"]
    recent = get_recent_pitcher_k9(r["Pitcher_ID"]) or base
    pred   = 0.5*base + 0.5*recent
    if r.get("FF_vel", 0) >= 95:
        pred *= 1.03
    cntL = sum("(L)" in h for h in lineup)
    cntR = sum("(R)" in h or "(S)" in h for h in lineup)
    if cntL + cntR:
        pw = ((r.get("whiff_L",0)*cntL) + (r.get("whiff_R",0)*cntR)) / (cntL + cntR)
        if pw > 0.30:
            pred *= 1.02
    mu = pred * r.get("IP_per_start", 5.0) / 9.0
    pf = fetch_monthly_park_factors(YEAR).get(r["Team"], {})
    mu *= pf.get(datetime.today().month, {}).get("SO", 100)/100.0
    if np.isfinite(NB_K_theta) and NB_K_p > 0:
        p_i = NB_K_theta/(NB_K_theta + mu) if (NB_K_theta + mu) > 0 else 1.0
        samp = nbinom(n=NB_K_theta, p=p_i).rvs(size=SIMS_K)
    else:
        samp = np.random.poisson(mu, size=SIMS_K)
    mean_k = samp.mean()
    med    = int(np.percentile(samp, 50))
    probs  = {f"P(K≥{k})": float((samp>=k).mean()) for k in range(2,11)}
    lo, hi = predict_k9_interval(pd.Series(r))
    return {"Mean_K_start": mean_k, "Median_K_start": med, **probs, "Pred_K9": round(pred,3), "90% CI": f"{int(lo)}–{int(hi)}"}

# ------------------------ HR per-game calibrators by archetype ------------------------
def infer_hr_archetype(name_norm:str|None, stand:str|None, iso:float|None, pull_pct:float|None) -> str:
    s = (stand or "R").upper()
    side = "L" if s.startswith("L") else "R"
    power = False
    try:
        if iso is not None and np.isfinite(iso) and float(iso) >= 0.200:
            power = True
    except Exception:
        pass
    try:
        if not power and pull_pct is not None and np.isfinite(pull_pct) and float(pull_pct) >= 0.45:
            power = True
    except Exception:
        pass
    return f"{side}-Power" if power else f"{side}-Contact"

@disk_cache("hr_game_cals.pkl")
def build_hr_game_calibrators(days_recent:int=90, days_rate:int=180):
    end = datetime.today()
    sr  = (end - timedelta(days=days_recent)).strftime("%Y-%m-%d")
    sr2 = (end - timedelta(days=days_rate)).strftime("%Y-%m-%d")
    e   = end.strftime("%Y-%m-%d")

    df_recent = fetch_statcast_raw(sr, e)
    if df_recent is None or df_recent.empty:
        return {}

    df_recent["is_pa"] = df_recent["events"].notna().astype(int)
    gb = df_recent.groupby(["batter","game_pk"])
    PAg = gb["is_pa"].sum().rename("PA_game")
    HRg = gb["events"].apply(lambda s: (s=="home_run").sum()).rename("C_HR")
    recent = pd.concat([PAg, HRg], axis=1).reset_index()

    df_rate = fetch_statcast_raw(sr2, e)
    if df_rate is None or df_rate.empty:
        return {}

    df_rate["is_pa"] = df_rate["events"].notna().astype(int)
    gb2 = df_rate.groupby("batter")
    PA  = gb2["is_pa"].sum().rename("PA")
    HR  = gb2["events"].apply(lambda s: (s=="home_run").sum()).rename("HR")
    rate = pd.concat([PA, HR], axis=1).reset_index()
    rate["R_HR_rate"] = rate["HR"] / rate["PA"].clip(lower=1)

    pids = [int(x) for x in recent["batter"].unique().tolist()]
    if not pids:
        return {}

    if ThreadPool:
        with ThreadPool(8) as pool:
            feats = pool.map(get_statcast_batter_features, pids)
        df_feat = pd.DataFrame(feats, index=pids).reset_index().rename(columns={"index":"batter"})
    else:
        feats = [get_statcast_batter_features(pid) for pid in pids]
        df_feat = pd.DataFrame(feats, index=pids).reset_index().rename(columns={"index":"batter"})

    # --- NEW: guarantee expected columns before the merge ---
    if "stand" not in df_feat.columns:
        df_feat["stand"] = "R"
    if "pull_pct" not in df_feat.columns:
        df_feat["pull_pct"] = 0.40
    # normalize types/sanity
    df_feat["stand"] = df_feat["stand"].astype(str).str.upper().str[0].fillna("R")
    df_feat["pull_pct"] = pd.to_numeric(df_feat["pull_pct"], errors="coerce").clip(0, 1).fillna(0.40)

    # ISO for the season (ok if this fails; we backfill NaN)
    try:
        bat = batting_stats(YEAR, qual=0).copy()
        bat["Name_norm"] = bat.Name.apply(clean_name)
        bat["pid"] = bat["Name_norm"].apply(name_to_mlbam_id)
        iso_map = bat.set_index("pid")["ISO"].to_dict()
        df_feat["ISO_season"] = df_feat["batter"].map(iso_map)
    except Exception:
        df_feat["ISO_season"] = np.nan

    data = (
        recent.merge(rate[["batter","R_HR_rate"]], on="batter", how="left")
              .merge(df_feat[["batter","stand","pull_pct","ISO_season"]], on="batter", how="left")
    )

    data["R_HR_rate"] = data["R_HR_rate"].fillna(0.0)
    data["PA_game"]   = data["PA_game"].fillna(0)
    data["lam_hat"]   = data["PA_game"] * data["R_HR_rate"]
    data["y"]         = (data["C_HR"] >= 1).astype(int)

    cals = {}
    # guard: need some variation to fit isotonic
    if data["lam_hat"].nunique() >= 2 and data["y"].nunique() >= 2:
        try:
            iso_all = IsotonicRegression(out_of_bounds="clip").fit(
                data["lam_hat"].to_numpy(dtype=float),
                data["y"].to_numpy(dtype=int)
            )
            cals[("HR", 1, "ALL")] = iso_all
            cals[("HR", 1)]        = iso_all
        except Exception:
            pass

    def _arch(row):
        return infer_hr_archetype(
            None,
            row.get("stand","R"),
            row.get("ISO_season", np.nan),
            row.get("pull_pct", np.nan)
        )

    try:
        data["arch"] = data.apply(_arch, axis=1)
        for arch, g in data.groupby("arch"):
            if len(g) < 800 or g["lam_hat"].nunique() < 2 or g["y"].nunique() < 2:
                continue
            cals[("HR", 1, arch)] = IsotonicRegression(out_of_bounds="clip").fit(
                g["lam_hat"].to_numpy(dtype=float),
                g["y"].to_numpy(dtype=int)
            )
    except Exception:
        pass

    return cals

hr_game_cals = build_hr_game_calibrators()

# Global λ scale (tunable; rough market tilt)
@disk_cache("hr_lambda_scale.pkl")
def _estimate_hr_lambda_scale() -> float:
    end = datetime.today()
    sr  = (end - timedelta(days=90)).strftime("%Y-%m-%d")
    e   = end.strftime("%Y-%m-%d")
    df  = fetch_statcast_raw(sr, e)
    if df is None or df.empty:
        return 1.15
    df["is_pa"] = df["events"].notna().astype(int)
    g   = df.groupby(["batter","game_pk"])
    pag = g["is_pa"].sum().rename("PA_game")
    y   = (g["events"].apply(lambda s: (s=="home_run").sum()) >= 1).astype(int).rename("Y")
    recent = pd.concat([pag, y], axis=1).reset_index()

    sr2 = (end - timedelta(days=180)).strftime("%Y-%m-%d")
    r   = fetch_statcast_raw(sr2, e)
    if r is None or r.empty:
        return 1.15
    r["is_pa"] = r["events"].notna().astype(int)
    b  = r.groupby("batter")
    pa = b["is_pa"].sum()
    hr = b["events"].apply(lambda s: (s=="home_run").sum())
    rate = (hr / pa.clip(lower=1)).rename("pHR")
    base = pd.concat([pa.rename("PA"), rate], axis=1).reset_index()

    dfm = recent.merge(base[["batter","pHR"]], on="batter", how="left").fillna({"pHR": 0.0})
    lam = dfm["PA_game"] * dfm["pHR"]
    y   = dfm["Y"].values.astype(int)
    grid = np.linspace(0.9, 1.35, 10)
    best_s, best_ll = 1.15, -1e18
    for s in grid:
        p = 1.0 - np.exp(-s * lam.values)
        p = np.clip(p, 1e-6, 1-1e-6)
        ll = np.sum(y*np.log(p) + (1-y)*np.log(1-p))
        if ll > best_ll:
            best_ll, best_s = ll, s
    return float(best_s)

HR_LAMBDA_SCALE = _estimate_hr_lambda_scale()

# ------------------------ Per-PA isotonic (H/TB/HR) ------------------------
@disk_cache("iso_pa_calibrators.pkl")
def build_pa_isotonic_calibrators(days_recent:int=90, days_rate:int=180):
    end = datetime.today()
    sr  = (end - timedelta(days=days_recent)).strftime("%Y-%m-%d")
    sr2 = (end - timedelta(days=days_rate)).strftime("%Y-%m-%d")
    e   = end.strftime("%Y-%m-%d")

    df_recent = fetch_statcast_raw(sr, e)
    if df_recent is None or df_recent.empty:
        return {}
    df_recent["is_pa"] = df_recent["events"].notna().astype(int)
    gb = df_recent.groupby(["batter","game_pk"])
    pa = gb["is_pa"].sum().rename("PA_game")
    h1 = gb["events"].apply(lambda s: (s=="single").sum()).rename("C_1B")
    h2 = gb["events"].apply(lambda s: (s=="double").sum()).rename("C_2B")
    h3 = gb["events"].apply(lambda s: (s=="triple").sum()).rename("C_3B")
    hr = gb["events"].apply(lambda s: (s=="home_run").sum()).rename("C_HR")
    tb = (h1 + 2*h2 + 3*h3 + 4*hr).rename("C_TB")
    recent = pd.concat([pa,h1,h2,h3,hr,tb], axis=1).reset_index()

    df_rate = fetch_statcast_raw(sr2, e)
    if df_rate is None or df_rate.empty:
        return {}
    df_rate["is_pa"] = df_rate["events"].notna().astype(int)
    gb2 = df_rate.groupby("batter")
    PA  = gb2["is_pa"].sum().rename("PA")
    R1  = gb2["events"].apply(lambda s: (s=="single").sum()).rename("R_1B")
    R2  = gb2["events"].apply(lambda s: (s=="double").sum()).rename("R_2B")
    R3  = gb2["events"].apply(lambda s: (s=="triple").sum()).rename("R_3B")
    RR  = gb2["events"].apply(lambda s: (s=="home_run").sum()).rename("R_HR")
    RTB = (R1 + 2*R2 + 3*R3 + 4*RR).rename("R_TB")
    rate = pd.concat([PA,R1,R2,R3,RR,RTB], axis=1).reset_index()
    for c in ["R_1B","R_2B","R_3B","R_HR","R_TB"]:
        rate[c+"_rate"] = rate[c] / rate["PA"].clip(lower=1)

    data = (recent.merge(rate[["batter","R_1B_rate","R_2B_rate","R_3B_rate","R_HR_rate","R_TB_rate"]],
                         on="batter", how="left")).fillna(0.0)
    data["lam_H"]  = data["PA_game"] * (data["R_1B_rate"] + data["R_2B_rate"] + data["R_3B_rate"] + data["R_HR_rate"])
    data["lam_TB"] = data["PA_game"] * data["R_TB_rate"]
    data["lam_HR"] = data["PA_game"] * data["R_HR_rate"]
    data["H_count"] = data["C_1B"] + data["C_2B"] + data["C_3B"] + data["C_HR"]

    labs = {
        ("H",1):(data["H_count"]>=1).astype(int),
        ("H",2):(data["H_count"]>=2).astype(int),
        ("H",3):(data["H_count"]>=3).astype(int),
        ("H",4):(data["H_count"]>=4).astype(int),
        ("TB",1):(data["C_TB"]>=1).astype(int),
        ("TB",2):(data["C_TB"]>=2).astype(int),
        ("TB",3):(data["C_TB"]>=3).astype(int),
        ("TB",4):(data["C_TB"]>=4).astype(int),
        ("HR",1):(data["C_HR"]>=1).astype(int),
    }
    xs = {("H",k): data["lam_H"]  for k in (1,2,3,4)} \
       | {("TB",k): data["lam_TB"] for k in (1,2,3,4)} \
       | {("HR",1): data["lam_HR"]}

    cals = {}
    for key in labs:
        cals[key] = IsotonicRegression(out_of_bounds="clip").fit(xs[key].values.astype(float),
                                                                 labs[key].values.astype(int))
    return cals

iso_pa_calibrators = build_pa_isotonic_calibrators()

def _sanity_check_pa_calibrators(cals: dict) -> None:
    try:
        keys = list(cals.keys())
        logging.info(f"iso_pa_calibrators: {len(keys)} keys; sample {keys[:6]}")
        test = np.array([0.0, 0.25, 0.5, 1.0], dtype=float)
        for k in [("HR",1), ("H",1), ("TB",2)]:
            if k in cals:
                y = np.asarray(cals[k].predict(test))
                if np.any(np.diff(y) < -1e-8):
                    logging.warning(f"Non-monotone calibrator {k} over test points.")
    except Exception as e:
        logging.warning(f"Calibrator sanity check skipped: {e}")

_sanity_check_pa_calibrators(iso_pa_calibrators)

# ------------------------ Simulation helpers (added back) ------------------------
def simulate_markov_inning(pitcher_profile: dict,
                           batter_rates: list[dict],
                           transition_matrix=None) -> int:
    """
    Very simple Markov inning using event rates in batter_rates.
    State = outs*8 + bases_bitmask(1B,2B,3B).
    """
    state, runs, outs = 0, 0, 0
    idx = 0
    while outs < 3:
        rates = batter_rates[idx % len(batter_rates)]
        evt = sample_pa_outcome(rates)

        if evt == 'HR':
            b = state % 8
            runs += (b & 1) + ((b & 2) >> 1) + ((b & 4) >> 2) + 1
            state = outs*8 + 0

        elif evt == '3B':
            b = state % 8
            runs += (b & 1) + ((b & 2) >> 1) + ((b & 4) >> 2)
            state = outs*8 + 4

        elif evt == '2B':
            b = state % 8
            runs += ((b & 2) >> 1) + ((b & 4) >> 2)
            state = outs*8 + (2 | ((b & 1) << 1))

        elif evt == '1B':
            b = state % 8
            runs += ((b & 4) >> 2)
            new_b = ((b << 1) & 7) | 1
            state = outs*8 + new_b

        elif evt in ('BB','HBP'):
            b = state % 8
            if b == 7:
                runs += 1
            state = outs*8 + ((b | 1) & 7)

        else:  # OUT
            outs += 1
            state = outs*8 + (state % 8)

        idx += 1

    return runs

def simulate_reliever(rel: dict,
                      batter_rates: list[dict],
                      bullpen_profile: dict,
                      base_out_state: int) -> dict:
    if not rel:
        ip = np.random.uniform(0.1, 1.0)
        return {"IP": ip, "Runs_Allowed": poisson.rvs(1.5), "Pitcher_ID": None}

    season_ip = rel.get("Season_IP", 10.0)
    rec_k9    = rel.get("Recent_K9_30d", rel.get("Season_K9", 8.5))
    ip        = sample_starter_ip(season_ip, rec_k9)
    ip        = float(np.clip(ip, 0.1, 2.0))

    runs = 0
    outs_int = int(np.floor(ip))
    for _ in range(outs_int):
        runs += simulate_markov_inning(rel, batter_rates, RUN_EXP_MATRIX)

    frac = ip - outs_int
    if frac > 0:
        runs += int(round(frac * simulate_markov_inning(rel, batter_rates, RUN_EXP_MATRIX)))

    return {"IP": ip, "Runs_Allowed": runs, "Pitcher_ID": rel.get("Pitcher_ID")}

def simulate_full_game(team_abbr: str,
                       lineup: list,
                       starter_stats: dict,
                       df_rel: pd.DataFrame,
                       bullpen_profile: dict,
                       batter_rates: list[dict]) -> dict:
    ip_allowed = min(
        starter_stats.get("IP_per_start", 5.0),
        sample_starter_ip(starter_stats.get("IP_per_start", 5.0), starter_stats.get("Pred_K9", 8.0))
    )

    runs = 0
    outs_int = int(np.floor(ip_allowed))
    for _ in range(outs_int):
        runs += simulate_markov_inning(starter_stats, batter_rates, RUN_EXP_MATRIX)

    frac = ip_allowed - outs_int
    if frac > 0:
        runs += int(round(frac * simulate_markov_inning(starter_stats, batter_rates, RUN_EXP_MATRIX)))

    usage, used, inning = [], set(), outs_int + 1
    while inning <= 9:
        rel = select_high_leverage_reliever(df_rel, used, inning)
        if not rel:
            break
        used.add(rel["Pitcher_ID"])
        out = simulate_reliever(rel, batter_rates, bullpen_profile, 0)
        runs += out["Runs_Allowed"]
        usage.append((rel["Name"], out))
        inning += int(np.ceil(out["IP"]))

    return {"Runs": runs, "Usage": usage}

# ------------------------ exports & cache_all ------------------------
__all__ = [
    'predict_k9','predict_k9_interval','features_base','league_feature_means',
    'kt_montecarlo','sample_starter_ip','simulate_full_game','shrink_vs_pitcher','sample_pa_outcome',
    'H_theta','H_p','TB_theta','TB_p','NB_K_theta','NB_K_p','LEAGUE_BAT',
    'predict_hits','predict_hits_interval','features_hit_base','league_hit_means','iso_hit_calibrators',
    'predict_hr_proba','predict_hr_count_pt','predict_hr_hardhit','predict_hr_2swhiff','predict_hr_pullangle',
    'iso_pa_calibrators','expected_matchup_xiso','hr_game_cals','HR_LAMBDA_SCALE','infer_hr_archetype','IP_REG','batter_game_hr_prob',
    'get_recent_evt_feats','batter_recent_multiplier','get_pitcher_recent','pitcher_recent_multiplier'

]

def cache_all():
    logging.info("1) park factors"); fetch_yearly_park_factors(YEAR); fetch_monthly_park_factors(YEAR)
    logging.info("2) run-exp"); build_run_expectancy_matrix()
    logging.info("3) train K9"); train_ensemble_k9_model()
    logging.info("3b) train Hits"); train_ensemble_hit_model()
    logging.info("4) NB dispersions"); compute_batter_dispersions(YEAR); compute_strikeout_dispersion(YEAR)
    logging.info("5) IP regression"); fit_ip_regression(YEAR)
    logging.info("6) batter career"); build_batter_career_rates([YEAR-3,YEAR-2,YEAR-1,YEAR])
    logging.info("7) per-PA calibrators"); build_pa_isotonic_calibrators()
    logging.info("8) HR archetype calibrators"); build_hr_game_calibrators()
    logging.info("9) HR λ-scale"); _estimate_hr_lambda_scale()

if __name__=="__main__":
    cache_all()
    predict_k9,  predict_k9_interval,  _, league_feature_means  = train_ensemble_k9_model()
    predict_hits,predict_hits_interval, _, league_hit_means, iso_hit_calibrators = train_ensemble_hit_model()
    print("precompute_everything complete!")
