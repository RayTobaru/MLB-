#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, json, sys, argparse, logging, warnings
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta, date as _date

import numpy as np
import pandas as pd
from scipy.stats import poisson
from tqdm import trange
from sklearn.isotonic import IsotonicRegression  # for market alignment

# silence pybaseball banner
import builtins
_orig_print = builtins.print
def _quiet_print(*args, **kwargs):
    if args and isinstance(args[0], str) and args[0].startswith("Gathering Player Data"):
        return
    _orig_print(*args, **kwargs)
builtins.print = _quiet_print

import fetch
from precompute_everything import (
    LEAGUE_BAT, simulate_full_game, kt_montecarlo, predict_k9, predict_k9_interval, shrink_vs_pitcher,
    features_base, league_feature_means, predict_hr_proba, IP_REG, sample_starter_ip,
    predict_hr_count_pt,  # keep
    iso_pa_calibrators, expected_matchup_xiso,
    hr_game_cals, HR_LAMBDA_SCALE, infer_hr_archetype, sp_pa_share,
    get_recent_evt_feats, batter_recent_multiplier,
    get_pitcher_recent, pitcher_recent_multiplier,
    get_season_iso,
    load_b14_feats, load_season_hr_splits, load_pitcher_vs_side_hr,
    build_count_pitchtype_tables, build_hr_count_pt_onehot,build_heart_rate_table
  # <-- ADD THESE
)

pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.3f}".format)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("pybaseball").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.INFO)

YEAR = datetime.today().year

from fetch import (
    ID2ABBR, fetch_monthly_park_factors, get_game_weather,
    fetch_lineup_and_starters, fetch_unofficial_lineup, clean_name, name_to_mlbam_id,
    build_player_pa_maps, build_bullpen_dataframe, profile_bullpen, safe_get, strip_pos,
    pitching_stats, statcast_bat_cache, h2h_cache, batting_stats, fetch_statcast_raw,
    tail_wind_pct
)

# -------- League baselines --------
_all_p = pitching_stats(YEAR, qual=0)
_starters = _all_p[_all_p.GS > 0]
LEAGUE_AVG_IP_PER_START = float(_starters.IP.sum() / _starters.GS.sum()) if len(_starters) else 5.2
BF_PER_9 = 38.0
LEAGUE_HR9 = float((_all_p.HR.sum() * 9.0) / max(_all_p.IP.sum(), 1e-6)) if len(_all_p)>0 else 1.1
LEAGUE_HR_PER_PA = LEAGUE_HR9 / BF_PER_9

# --- lineup-spot PA fallbacks ---
SPOT_PA_HOME = [4.72, 4.59, 4.48, 4.37, 4.26, 4.15, 4.05, 3.95, 3.86]
SPOT_PA_AWAY = [x + 0.10 for x in SPOT_PA_HOME]
def _iso_or_poisson(cal_key, lam, k=1):
    iso = iso_pa_calibrators.get(cal_key)
    if iso is not None:
        try:
            return float(iso.predict([lam])[0])
        except Exception:
            pass
    return float(1 - poisson.cdf(k-1, lam))

_bat = batting_stats(YEAR, qual=0).copy()
_bat["Name_norm"] = _bat.Name.apply(clean_name)
_bat_idx = _bat.set_index("Name_norm")

def prob_to_american(p: float) -> str:
    try: p = float(p)
    except Exception: return "—"
    p = min(max(p, 1e-6), 0.9999)
    return f"{-int(round(100 * p / (1 - p)))}" if p >= 0.5 else f"+{int(round(100 * (1 - p) / p))}"

def american_to_prob(odds):
    o = float(odds)
    return (100.0/(o+100.0)) if o>0 else (-o/(-o+100.0))

def _dec_from_prob(pb: float) -> float:
    pb = float(np.clip(pb, 1e-6, 0.999999))
    return 1.0 / pb

def _ev_pct(p_model: float, dec_price: float) -> float:
    return float(p_model * dec_price - 1.0) * 100.0

def _safe_float(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default

def _z(x, mu, sd):
    x = _safe_float(x, np.nan)
    return 0.0 if (not np.isfinite(x) or sd <= 0) else (x - mu) / sd

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def compute_batter_14d_feats(batter_id: int, sc180: pd.DataFrame) -> dict:
    """
    Robust last-14d barrel%, hard-hit%, EV from raw Statcast, with fallbacks.
    """
    out = {"barrel_14d_pct": np.nan, "hardhit_14d_pct": np.nan, "ev_mean_14d": np.nan}
    if sc180 is None or sc180.empty or "batter" not in sc180.columns:
        return out

    try:
        cutoff = (datetime.today() - timedelta(days=14)).strftime("%Y-%m-%d")
        g = sc180[(sc180.batter == batter_id) &
                  (sc180.get("game_date", pd.Series(dtype=str)) >= cutoff)].copy()
        if g.empty:
            return out

        # numeric EV
        if "launch_speed" in g.columns:
            g["ls"] = pd.to_numeric(g["launch_speed"], errors="coerce")
        else:
            g["ls"] = np.nan

        # barrel: prefer 'barrel' flag; fallback to a simple proxy
        if "barrel" in g.columns:
            g["is_barrel"] = pd.to_numeric(g["barrel"], errors="coerce").fillna(0) == 1
        else:
            ang = pd.to_numeric(g.get("launch_angle"), errors="coerce")
            g["is_barrel"] = ang.between(26, 30) & (g["ls"] >= 98)

        out["barrel_14d_pct"] = float(np.nanmean(g["is_barrel"])) if len(g) else np.nan
        out["hardhit_14d_pct"] = float(np.nanmean(g["ls"] >= 95)) if len(g) else np.nan
        out["ev_mean_14d"]     = float(np.nanmean(g["ls"])) if len(g) else np.nan
        return out
    except Exception:
        return out


def _pct_or_ratio(row, pct_keys, num_key):
    for k in pct_keys:
        if k in row and pd.notna(row[k]):
            try:
                return float(row[k]) / 100.0
            except Exception:
                pass
    if num_key in row and "PA" in row and row["PA"] > 0:
        return float(row[num_key]) / float(row["PA"])
    return np.nan

def bip_rate_for(name_norm: str) -> float:
    if name_norm not in _bat_idx.index: return 0.67
    row = _bat_idx.loc[name_norm]
    k = _pct_or_ratio(row, ["K%","SO%"], "SO")
    bb = _pct_or_ratio(row, ["BB%"], "BB")
    hbp = float(row.get("HBP",0))/max(float(row.get("PA",1)),1.0)
    if not (np.isfinite(k) and np.isfinite(bb) and np.isfinite(hbp)): return 0.67
    return float(np.clip(1.0 - k - bb - hbp, 0.45, 0.82))

def build_starter_dict(team, name, pid, venue_team, ump_feats=None, framing_feats=None, weather=None):
    r = {f: league_feature_means.get(f, 0.0) for f in features_base}
    df = pitching_stats(YEAR, qual=0)
    id_col = next((c for c in df.columns if c.lower() in ("pid","playerid","player_id","mlbam_id")), None)
    if id_col:
        try:
            df[id_col] = df[id_col].astype(int)
            rec = df[df[id_col] == int(pid)]
        except Exception:
            rec = df[df[id_col] == pid]
    else:
        rec = pd.DataFrame()
    if rec.empty:
        df["Name_norm"] = df.Name.apply(clean_name)
        rec = df[df.Name_norm == clean_name(name)]
    if not rec.empty and rec.iloc[0].GS >= 3:
        rec = rec.iloc[0]
        ipps = float(rec.IP / rec.GS)
        for c in ("FIP","WHIP","K%","BB%","SwStr%"):
            if c in rec: r[c] = float(rec[c])
        r["_HR9"] = float(9.0 * rec.HR / rec.IP) if rec.IP > 0 else LEAGUE_HR9
    else:
        ipps = LEAGUE_AVG_IP_PER_START
        r["_HR9"] = LEAGUE_HR9
    if pid:
        sc = fetch.get_statcast_pitcher_features(pid) or {}
        for k, v in sc.items():
            if k in r and pd.notna(v):
                r[k] = float(v)
    r["IP_per_start"] = ipps
        # --- venue K park factor for K/9 model ---
    try:
        mpf = fetch_monthly_park_factors(YEAR)
        m   = datetime.today().month
        r["park_k_factor"] = float(mpf.get(venue_team, {}).get(m, {}).get("SO", 100)) / 100.0
    except Exception:
        r["park_k_factor"] = league_feature_means.get("park_k_factor", 1.0)

    r["Pred_K9"] = predict_k9(pd.Series(r))
    pred_mean_ip = float(IP_REG.predict([[ipps, r["Pred_K9"], r["Pred_K9"]]])[0])
    r["Proj_IP"] = min(sample_starter_ip(pred_mean_ip, r["Pred_K9"]), 7.0)
    ws = (weather or {}).get("wind_mph")
    if isinstance(ws,(int,float)) and ws > 15: r["Pred_K9"] *= (1 + min(ws/100, 0.05))
    if ump_feats and "ump_k9" in ump_feats: r["Pred_K9"] *= float(ump_feats["ump_k9"])
    frame_pct = (framing_feats or {}).get(f"{team.lower()}_frame", 0.0)
    r["Pred_K9"] *= (1 + frame_pct/100.0)
    # NOTE: set Team to the VENUE so kt_montecarlo uses the correct park for strikeouts
    r.update({"Team": venue_team, "Pitcher_ID": pid, "Name": name, "Pitcher_Org": team})

    return r

def _temp_mult(tF):
    if not isinstance(tF,(int,float)): return 1.0
    return float(np.clip(1.0 + 0.012*((float(tF)-70.0)/10.0), 0.85, 1.18))

def _pitcher_hr_mult(starter_row):
    hr9 = float(starter_row.get("_HR9", LEAGUE_HR9))
    hr_pa = hr9 / BF_PER_9
    rel = (hr_pa - LEAGUE_HR_PER_PA) / max(LEAGUE_HR_PER_PA, 1e-9)
    return float(np.clip(1.0 + 0.60*rel, 0.75, 1.35))

def _bip_adjustment(ump):
    kf = float(ump.get("ump_k9",1.0)); bf = float(ump.get("ump_bb9",1.0))
    return 1.0 / (1.0 + 0.4*(kf-1.0) + 0.2*(bf-1.0))

def calibrated_hr_prob(exp_pa: float, p_hr_pa: float, arch: str, hr_scale_override: float|None=None):
    lam = float(exp_pa) * float(p_hr_pa)
    s = float(hr_scale_override) if hr_scale_override else float(HR_LAMBDA_SCALE)
    iso = hr_game_cals.get(("HR",1,arch)) or hr_game_cals.get(("HR",1,"ALL")) or hr_game_cals.get(("HR",1))
    if iso is not None:
        p = float(iso.predict([s*lam])[0]); src = f"iso[{'arch' if ('HR',1,arch) in hr_game_cals else 'ALL'}]"
    else:
        p = 1.0 - np.exp(-lam); src = "poisson"
    p = float(np.clip(p, 0.01, 0.40))
    return p, {"lam_raw": lam, "lam_scaled": s*lam, "src": src, "s_used": s}

def bullpen_hrpa(team_abbr: str) -> float:
    df = pitching_stats(YEAR, qual=0)
    if df is None or df.empty: return LEAGUE_HR_PER_PA
    rel = df[(df.Team == team_abbr) & (df.G > df.GS) & (df.IP > 0)]
    if rel.empty: return LEAGUE_HR_PER_PA
    hr9 = float(9.0 * rel.HR.sum() / max(rel.IP.sum(), 1e-6))
    return float(np.clip(hr9 / BF_PER_9, 0.002, 0.05))

def _poisson_game_hr(exp_pa: float, p_hr_pa: float, scale: float|None) -> float:
    s = float(scale) if scale is not None else float(HR_LAMBDA_SCALE)
    lam = float(exp_pa) * float(p_hr_pa) * s
    return float(np.clip(1.0 - np.exp(-lam), 1e-4, 0.95))
# ---------- HR matchup score (0..100) ----------
def hr_matchup_score(
    barrel14, hh14, ev14,
    season_hr_total, season_hr_vs_side,
    sp_hr9, sp_hr_rate_vs_side,
    park_hr_idx, wind_tail, mxiso
) -> float:
    zsum = 0.0
    zsum += 1.10 * _z(barrel14, 0.08, 0.05)
    zsum += 0.80 * _z(hh14,     0.40, 0.10)
    zsum += 0.50 * _z(ev14,     90.5, 2.5)
    zsum += 0.35 * _z(season_hr_total, 12, 8)
    zsum += 0.35 * _z(season_hr_vs_side, 8, 6)
    zsum += 0.80 * _z(sp_hr9,   1.10, 0.40)
    zsum += 0.80 * _z(sp_hr_rate_vs_side, 0.025, 0.010)
    zsum += 0.60 * _z(park_hr_idx, 1.00, 0.15)
    zsum += 0.50 * _z(wind_tail, 0.00, 0.10)
    zsum += 0.70 * _z(mxiso, 0.165, 0.030)
    return float(round(100.0 * _sigmoid(zsum), 1))

# ---------- Optional: calibrate score -> observed HR% via logistic ----------
_SCORE_HISTORY_CSV = os.path.join("cache", "hr_score_history.csv")
def load_score_calibrator():
    try:
        df = pd.read_csv(_SCORE_HISTORY_CSV)[["score","label"]].dropna()
        if len(df) < 200: return None
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(df[["score"]].values, df["label"].astype(int).values)
        return lambda s: float(lr.predict_proba(np.array([[float(s)]]))[0,1])
    except Exception:
        return None

def fallback_matchup_xiso(pitcher_id: int, batter_id: int, days: int = 365) -> float:
    """
    Proxy xISO: batter (xSLG-xBA) by pitch type blended by pitcher's mix.
    Falls back to realized ISO if estimated metrics missing.
    """
    try:
        end = datetime.today()
        start = (end - timedelta(days=days)).strftime("%Y-%m-%d")
        sc = fetch_statcast_raw(start, end.strftime("%Y-%m-%d"))
        if sc is None or sc.empty:
            return np.nan

        sc = sc.copy()

        # Pitcher mix
        gp = sc[sc["pitcher"] == pitcher_id]
        if gp.empty or "pitch_type" not in gp.columns:
            return np.nan
        mix = gp["pitch_type"].value_counts(normalize=True)

        # Batter by pitch
        gb = sc[sc["batter"] == batter_id]
        if gb.empty or "pitch_type" not in gb.columns:
            return np.nan

        has_est = {"estimated_ba_using_speedangle","estimated_slg_using_speedangle"}.issubset(gb.columns)
        if has_est:
            tmp = gb.groupby("pitch_type").agg(
                xba=("estimated_ba_using_speedangle","mean"),
                xslg=("estimated_slg_using_speedangle","mean"),
            )
            tmp["xiso"] = (tmp["xslg"] - tmp["xba"]).clip(lower=0)
        else:
            gb["1B"] = (gb["events"] == "single").astype(int)
            gb["2B"] = (gb["events"] == "double").astype(int)
            gb["3B"] = (gb["events"] == "triple").astype(int)
            gb["HR"] = (gb["events"] == "home_run").astype(int)
            gb["AB_proxy"] = ((gb["events"].notna()) &
                              ~gb["events"].isin(["walk","hit_by_pitch","intent_walk"])).astype(int)
            tmp = gb.groupby("pitch_type").agg(
                TB=lambda s: (gb.loc[s.index, "1B"] + 2*gb.loc[s.index, "2B"] +
                              3*gb.loc[s.index, "3B"] + 4*gb.loc[s.index, "HR"]).sum(),
                H =lambda s: (gb.loc[s.index, "1B"] + gb.loc[s.index, "2B"] +
                              gb.loc[s.index, "3B"] + gb.loc[s.index, "HR"]).sum(),
                AB=lambda s: gb.loc[s.index, "AB_proxy"].sum()
            )
            tmp["AVG"]  = tmp["H"]  / tmp["AB"].clip(lower=1)
            tmp["SLG"]  = tmp["TB"] / tmp["AB"].clip(lower=1)
            tmp["xiso"] = (tmp["SLG"] - tmp["AVG"]).clip(lower=0)

        common = set(tmp.index) & set(mix.index)
        if not common:
            return np.nan
        x = sum(float(tmp.loc[pt, "xiso"]) * float(mix.loc[pt]) for pt in common)
        return float(np.clip(x, 0.05, 0.40))
    except Exception:
        return np.nan

# ---------- shortlist helper ----------
def _minmax_norm(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    lo, hi = x.min(), x.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(x)), index=s.index)
    return (x - lo) / (hi - lo)

def pick_top_hr_targets(
    df: pd.DataFrame,
    limit: int = 8,
    min_prob: float = 0.06,
    weight_prob: float = 0.80,   # main driver
    weight_score: float = 0.20   # tie-breaker / context
) -> pd.DataFrame:
    """
    Shortlist for MOST LIKELY HR hitters.
    Ignores EV/Edge completely; ranks by model P(HR≥1) + HR_Score (optional).
    """
    if df.empty or "P(HR≥1)" not in df.columns:
        return pd.DataFrame()

    d = df.copy()
    d = d[pd.to_numeric(d["P(HR≥1)"], errors="coerce").notna()]
    d = d[d["P(HR≥1)"] >= float(min_prob)]

    pN = _minmax_norm(d["P(HR≥1)"])
    sN = _minmax_norm(d["HR_Score"]) if "HR_Score" in d.columns else pd.Series(0.0, index=d.index)

    wP, wS = float(weight_prob), float(weight_score)
    d["PickScore"] = wP * pN + wS * sN
    return d.sort_values(["PickScore", "P(HR≥1)","HR_Score"], ascending=[False, False, False]).head(limit)

# ---------- Market CSV / FD vertical ----------
def load_market_probs(path: str) -> dict[str, float]:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("name") or cols.get("player") or cols.get("batter")
    if not name_col: raise ValueError("market CSV must have Name/Player/Batter column")
    if "Prob" in df.columns: p = pd.to_numeric(df["Prob"], errors="coerce").astype(float)
    elif "prob" in df.columns: p = pd.to_numeric(df["prob"], errors="coerce").astype(float)
    elif "Odds" in df.columns: p = df["Odds"].apply(american_to_prob)
    elif "odds" in df.columns: p = df["odds"].apply(american_to_prob)
    else: raise ValueError("market CSV must include Prob or Odds")
    p = np.where(p>1, p/100.0, p)
    out={}
    for nm, ip in zip(df[name_col], p):
        if pd.notna(ip): out[clean_name(strip_pos(str(nm)))] = float(np.clip(ip, 1e-4, 0.95))
    return out

def _load_fd_vertical(path: str) -> dict[str, float]:
    s = pd.read_csv(path, header=None, dtype=str, keep_default_na=False).iloc[:, 0].astype(str).tolist()
    out = {}; i = 0
    while i + 1 < len(s):
        nm_raw  = s[i].strip(); odds_raw = str(s[i+1]).strip()
        odds_txt = (odds_raw.replace("+","").replace("\u2212","-").replace("–","-").replace("—","-").replace(",",""))
        try: odds = float(odds_txt)
        except Exception: i += 1; continue
        nm = clean_name(strip_pos(nm_raw))
        out[nm] = float(np.clip(american_to_prob(odds), 1e-4, 0.95))
        i += 2
    return out

def _try_load_fdodds_default() -> dict[str, float]:
    for fn in ("FDodds.csv","FDodds.CSV","FDodds","fdodds.csv","fdodds"):
        if os.path.exists(fn): return _load_fd_vertical(fn)
    return {}

# ---------- Market alignment ----------
def fit_market_calibrator(p_model, p_book):
    x = np.asarray(p_model, float); y = np.asarray(p_book, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x>0) & (x<1) & (y>0) & (y<1)
    x, y = x[ok], y[ok]
    if len(x) < 12: return None
    try:
        lo = x <= 0.20; hi = ~lo
        f_lo = IsotonicRegression(out_of_bounds="clip").fit(x[lo], y[lo]) if lo.sum() >= 8 else None
        f_hi = IsotonicRegression(out_of_bounds="clip").fit(x[hi], y[hi]) if hi.sum() >= 8 else None
        if f_lo or f_hi:
            def piecewise(p):
                if p <= 0.20 and f_lo: return float(np.clip(f_lo.predict([p])[0], 1e-4, 0.95))
                if f_hi:              return float(np.clip(f_hi.predict([p])[0], 1e-4, 0.95))
                return float(np.clip(p, 1e-4, 0.95))
            return piecewise
        iso = IsotonicRegression(out_of_bounds="clip").fit(x, y)
        return lambda p: float(np.clip(iso.predict([p])[0], 1e-4, 0.95))
    except Exception:
        from sklearn.linear_model import LogisticRegression
        X = np.log(x/(1-x)).reshape(-1,1)
        lr = LogisticRegression(solver="lbfgs").fit(X, y)
        return lambda p: float(np.clip(lr.predict_proba([[np.log(p/(1-p))]])[0,1], 1e-4, 0.95))

# ---------- Blend alpha memory ----------
_ALPHA_PATH = os.path.join("cache", "hr_blend_alpha_fd.json")
def _load_alpha_memory(default=0.70):
    try:
        with open(_ALPHA_PATH, "r") as f:
            return float(json.load(f).get("alpha", default))
    except Exception:
        return float(default)
def _save_alpha_memory(alpha):
    try:
        os.makedirs(os.path.dirname(_ALPHA_PATH), exist_ok=True)
        with open(_ALPHA_PATH, "w") as f:
            json.dump({"alpha": float(alpha)}, f)
    except Exception: pass

# ---------- Market history memory ----------
_MARKET_LOG = os.path.join("cache", "market_history_fd.csv")
def _append_market_log(pairs, date_str):
    if not pairs: return
    df = pd.DataFrame(pairs, columns=["model_p", "book_p"]); df["date"] = date_str
    try:
        os.makedirs(os.path.dirname(_MARKET_LOG), exist_ok=True)
        if os.path.exists(_MARKET_LOG):
            df_old = pd.read_csv(_MARKET_LOG); df = pd.concat([df_old, df], ignore_index=True)
        df.tail(3000).to_csv(_MARKET_LOG, index=False)
    except Exception: pass
def _load_market_history():
    try:
        df = pd.read_csv(_MARKET_LOG)
        return df[["model_p","book_p"]].dropna().values.tolist()
    except Exception:
        return []

# ---------- schedule helpers ----------
def _resolve_date(args) -> str:
    base = _date.today()
    if getattr(args, "tomorrow", False): base = base + timedelta(days=1)
    elif getattr(args, "yesterday", False): base = base - timedelta(days=1)
    elif getattr(args, "date", None): return args.date
    return base.strftime("%Y-%m-%d")

def fetch_matchups_for_date(date_str: str):
    resp = safe_get("https://statsapi.mlb.com/api/v1/schedule", {"sportId": 1, "date": date_str})
    js = resp.json() if resp else {}
    games = []
    for d in js.get("dates", []):
        for g in d.get("games", []):
            try:
                away_id = g["teams"]["away"]["team"]["id"]
                home_id = g["teams"]["home"]["team"]["id"]
                games.append(f"{ID2ABBR[away_id]} @ {ID2ABBR[home_id]}")
            except Exception:
                continue
    return games
def main():
    p = argparse.ArgumentParser()
    p.add_argument("-g","--game", type=int)
    p.add_argument("-m","--matchup")
    p.add_argument("--sims", type=int, default=0, help="Number of game sims to run (0 = skip)")
    p.add_argument("--dbg-cal", action="store_true", help="Print HR calibrator debug lines")
    p.add_argument("--hr-scale", type=float, default=None, help="Override HR lambda scale")
    p.add_argument("--show-odds", dest="show_odds", action="store_true", help="Add American odds column for P(HR≥1)")
    p.add_argument("--market", type=str, default=None, help="CSV with Name + Odds or Prob for HR")
    p.add_argument("--hr-blend", type=str, default="0.7", help="Blend toward calibrated HR model (0..1) or 'auto'")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible sims")
    p.add_argument("--csv", type=str, default=None, help="Write batter props to CSV")
    p.add_argument("--date", type=str, default=None, help="ISO date (YYYY-MM-DD). Defaults to today/tomorrow flags.")
    p.add_argument("--today", action="store_true"); p.add_argument("--tomorrow", action="store_true"); p.add_argument("--yesterday", action="store_true")
    p.add_argument("--dbg-fetch", action="store_true", help="Print Statcast pull size and key columns")

    args = p.parse_args()

    if args.seed is not None: np.random.seed(args.seed)

    use_date = _resolve_date(args)
    games = fetch_matchups_for_date(use_date)
    if not games: sys.exit("No MLB games on that date.")
    for i, gm in enumerate(games, 1): print(f"{i}. {gm}")

        # Pre-fetch data
    sc180 = fetch_statcast_raw(
        (datetime.today()-timedelta(days=180)).strftime("%Y-%m-%d"),
        datetime.today().strftime("%Y-%m-%d")
    )
    if args.dbg_fetch:
        cols = [] if sc180 is None else list(sc180.columns)
        have = [c for c in ["game_date","batter","pitcher","events","launch_speed","launch_angle",
                            "barrel","p_throws","stand","pitch_type",
                            "estimated_ba_using_speedangle","estimated_slg_using_speedangle"] if c in cols]
        print(f"[DBG] sc180 rows={0 if sc180 is None else len(sc180)}; has: {', '.join(have)}")

    # === NEW: load precomputed caches (fast lookups) ===
    B14              = load_b14_feats()               # DataFrame indexed by batter id
    SEASON_HR_SPLITS = load_season_hr_splits()        # dict {batter:{HR_vs_R, HR_vs_L}}
    PITCH_SIDE       = load_pitcher_vs_side_hr()      # dict {pitcher:{vs_L_HR_rate, vs_R_HR_rate}}
    COUNT_PT         = build_count_pitchtype_tables()   # pitcher mix by count/side + batter end-at-count + priors
    HEART            = build_heart_rate_table()  # {"league": float, "pitcher": {pid: float}}

    if args.game and 1<=args.game<=len(games): sel = str(args.game)
    elif args.matchup: sel = args.matchup
    else: sel = input("Pick # or 'AWY @ HOM': ").strip()

    away, home = (games[int(sel)-1].split(" @ ") if sel.isdigit() else tuple(map(str.strip, sel.split("@"))))

    # PA maps
    _buf = io.StringIO()
    with redirect_stdout(_buf), redirect_stderr(_buf):
        SEASON_PA_G, TEAM_PA_GAME = build_player_pa_maps(YEAR)
    SEASON_PA_G, TEAM_PA_GAME = dict(SEASON_PA_G), dict(TEAM_PA_GAME)

    # lineups, weather
    try:
        (a_name,h_name,a_line,h_line,a_pid,h_pid, away_cid,home_cid, ump_feats,framing_feats) = fetch_lineup_and_starters(away,home)
    except RuntimeError:
        logging.warning(f"Official lineup not ready for {away}@{home}, using unofficial.")
        a_line = fetch_unofficial_lineup(away); h_line = fetch_unofficial_lineup(home)
        a_name = h_name = None; a_pid = h_pid = None
        ump_feats = {"ump_k9":1.0,"ump_bb9":1.0}; framing_feats = {"away_frame":0.0,"home_frame":0.0}

    # gamePk, weather
    sched = safe_get("https://statsapi.mlb.com/api/v1/schedule", {"sportId":1,"date":use_date,"hydrate":"probablePitchers"})
    js = sched.json() if sched else {}
    game = next((g for d in js.get("dates",[]) for g in d["games"]
                 if (ID2ABBR[g["teams"]["away"]["team"]["id"]], ID2ABBR[g["teams"]["home"]["team"]["id"]])==(away,home)), {})
    gamePk = game.get("gamePk")
    weather = get_game_weather(gamePk) if gamePk else {}
    wind_tail = tail_wind_pct(gamePk) if gamePk else 0.0
    temp_f = None
    for k in ("temp_f","temperature","tempF","temp"):
        v = weather.get(k)
        if isinstance(v,(int,float)): temp_f = float(v); break

    away_r = build_starter_dict(
        away, a_name, a_pid, venue_team=home,
        ump_feats=ump_feats,
        framing_feats={"away_frame": framing_feats.get("away_frame", 0.0)},
        weather=weather
    )
    home_r = build_starter_dict(
        home, h_name, h_pid, venue_team=home,
        ump_feats=ump_feats,
        framing_feats={"home_frame": framing_feats.get("home_frame", 0.0)},
        weather=weather
    )

    # Today's park = home park for both teams
    mpf = fetch_monthly_park_factors(YEAR)
    month = datetime.today().month

    away_rates, home_rates = [], []
    batter_rows = []

    if args.dbg_cal:
        sample_keys = list(hr_game_cals.keys())[:4]
        print(f"[DBG] HR_LAMBDA_SCALE={HR_LAMBDA_SCALE:.3f} | hr_game_cals={len(hr_game_cals)} sample={sample_keys}")

    score_cal = load_score_calibrator()

    # market input
    mk, mk_src = {}, None
    if args.market: mk = load_market_probs(args.market); mk_src = args.market
    else:
        mk = _try_load_fdodds_default(); mk_src = next((x for x in ("FDodds.csv","FDodds.CSV","FDodds","fdodds.csv","fdodds") if os.path.exists(x)), None)
    if mk: print(f"[market] loaded {len(mk)} players from {mk_src}; sample: {list(mk.keys())[:6]}")
    else:  print("[market] no FD odds found; Book_* and EV columns will be empty.")

    # ---- per-batter loop ----
    for team, lineup, st, rates in [(away,a_line,away_r,away_rates), (home,h_line,home_r,home_rates)]:
        opp_st        = home_r if team == away else away_r
        opp_team_abbr = home   if team == away else away
        bp_hrpa       = bullpen_hrpa(opp_team_abbr)
        bp_mult       = float(np.clip((bp_hrpa / LEAGUE_HR_PER_PA) if bp_hrpa > 0 else 1.0, 0.60, 1.60))

        for spot, nm_full in enumerate(lineup, start=1):
            nm = clean_name(strip_pos(nm_full)); bat_id = name_to_mlbam_id(nm)
            base = LEAGUE_BAT.copy()

            # H2H shrink vs SP
            h2h = h2h_cache.get((bat_id, opp_st["Pitcher_ID"]), {})
            if h2h.get("PA",0) >= 3:
                pa = float(h2h["PA"]); prior_mass = 4.0 if pa >= 10 else 6.0
                for e in ("1B","2B","3B","HR","TB","RBI","R"):
                    r_obs = float(h2h.get(f"{e}_rate", base[e]))
                    base[e] = shrink_vs_pitcher(r_obs, pa, base[e], prior_pa=prior_mass)

            scb = statcast_bat_cache.get(bat_id, {})
            barrel  = scb.get("barrel_pct", np.nan)
            ev_mean = scb.get("ev_mean",    np.nan)
            la_mean = scb.get("la_mean",    np.nan)

            # Prefer precomputed 14d features; fall back to statcast cache, then local compute if still NaN
            b14_row = (B14.loc[bat_id] if (hasattr(B14, "index") and bat_id in B14.index) else None)
            bar14 = float(b14_row["barrel_14d_pct"]) if b14_row is not None else scb.get("barrel_14d_pct", np.nan)
            hh14  = float(b14_row["hardhit_14d_pct"]) if b14_row is not None else scb.get("hardhit_14d_pct", np.nan)
            ev14  = float(b14_row["ev_mean_14d"])     if b14_row is not None else scb.get("ev_mean_14d",     np.nan)

            if not np.isfinite(bar14) or not np.isfinite(hh14) or not np.isfinite(ev14):
                # last-resort fallback (uses today’s raw pull if present)
                f14 = compute_batter_14d_feats(bat_id, sc180)
                if not np.isfinite(bar14): bar14 = f14["barrel_14d_pct"]
                if not np.isfinite(hh14):  hh14  = f14["hardhit_14d_pct"]
                if not np.isfinite(ev14):  ev14  = f14["ev_mean_14d"]

            # 180d slice for sweet-spot
            bb = sc180[sc180.batter == bat_id] if (sc180 is not None and "batter" in sc180.columns) else pd.DataFrame()

            hr_feat = {
                "exit_velocity": ev_mean if np.isfinite(ev_mean) else 0.0,
                "launch_angle":  la_mean if np.isfinite(la_mean) else 0.0,
                "barrel_pct":    barrel if np.isfinite(barrel) else 0.0,
                "HR_FB_rate":    scb.get("HR_FB_rate", 0.0),
                "pull_pct":      scb.get("pull_pct", 0.0),
                "park_hr_factor": 1.0,
            }
            if not bb.empty and {"launch_speed","launch_angle"}.issubset(bb.columns):
                mask = (bb.launch_speed > 105) & bb.launch_angle.between(25, 35)
                hr_feat["sweet_spot_frac"] = float(mask.mean())
            else:
                hr_feat["sweet_spot_frac"] = 0.0

            feat = pd.Series(hr_feat)
            p_stack = float(predict_hr_proba(feat))

            # --- Count × pitch-type integration using cached tables ---
            b_side = str(scb.get("stand", "R")).upper()[:1]
            b_side = "L" if b_side == "L" else "R"

            # Batter: P(AB ends with contact at count)
            end_w = COUNT_PT["BAT"].get(bat_id, None)
            if not end_w:
                end_w = COUNT_PT["PRIORS"]["end_at_count"] or {"0-0": 1.0}

            # Pitcher: P(pitch_type | count, side)
            mix_map = COUNT_PT["PITCH"].get(opp_st.get("Pitcher_ID") or 0, {}).get(b_side, None)
            if not mix_map:
                mix_map = COUNT_PT["PRIORS"]["mix"].get(b_side, {})

            # integrate logistic p(HR | count, pt) over (count, pt)
            p_ct, wsum = 0.0, 0.0
            for c, w_c in end_w.items():
                if w_c <= 0: 
                    continue
                pt_dist = (mix_map.get(c) or mix_map.get("ALL") or {})
                if not pt_dist:
                    continue
                for pt, m in pt_dist.items():
                    if m <= 0:
                        continue
                    onehot = build_hr_count_pt_onehot(c, pt)
                    p_ct += float(w_c) * float(m) * float(predict_hr_count_pt(onehot))
                    wsum += float(w_c) * float(m)

            if wsum > 0:
                p_ct /= wsum
            else:
                # ultra-safe fallback if everything is missing
                p_ct = float(predict_hr_count_pt(build_hr_count_pt_onehot("0-0", "FF")))

            # Blend main stacker with count×pt term (tunable)
            p_hr = 0.60*p_stack + 0.40*p_ct


            # recent-event multipliers
            evt = get_recent_evt_feats(bat_id)
            p_hr *= batter_recent_multiplier(evt)

            # recent barrels + ISO bump
            if np.isfinite(bar14):
                p_hr *= float(np.clip(1.0 + 0.6*float(bar14), 0.90, 1.25))

            
            try:
                b_iso = float(_bat_idx.loc[nm].get("ISO", np.nan)); lg_iso = float(_bat["ISO"].mean())
                if np.isfinite(b_iso) and np.isfinite(lg_iso) and lg_iso > 0:
                    p_hr *= float(np.clip(1.0 + 1.10*(b_iso-lg_iso)/lg_iso, 0.75, 1.45))
            except Exception: pass

            # matchup xISO
            mxiso = np.nan
            if bat_id and opp_st.get("Pitcher_ID"):
                try: mxiso = float(expected_matchup_xiso(opp_st["Pitcher_ID"], bat_id))
                except Exception: mxiso = np.nan
            if not np.isfinite(mxiso):
                mxiso = fallback_matchup_xiso(opp_st.get("Pitcher_ID") or 0, bat_id)
    
            if np.isfinite(mxiso): p_hr *= float(np.clip(1.0 + 1.10*(mxiso - 0.165), 0.75, 1.35))

            # Environment
            env_mult = 1.0
            if temp_f is not None: env_mult *= _temp_mult(temp_f)
            pf_team = mpf.get(home, {}).get(month, {})
            stand_key = "HR_L" if (scb.get("stand","R")).upper().startswith("L") else "HR_R"
            pf_side = pf_team.get(stand_key, pf_team.get("HR", 100)) / 100.0
            env_mult *= pf_side
            pullish = float(np.clip(scb.get("pull_pct", 0.4), 0.2, 0.7))
            if wind_tail > 0: env_mult *= (1.0 + 0.6 * wind_tail * (0.6 + 0.8*pullish))

            # SP vs BP blend
            sp_share   = sp_pa_share(opp_st.get("Pitcher_ID"))
            pit_rec    = get_pitcher_recent(opp_st.get("Pitcher_ID") or 0)
            pit_recent_mult = pitcher_recent_multiplier(pit_rec, LEAGUE_HR_PER_PA)
            p_pa_vs_sp = p_hr * _pitcher_hr_mult(opp_st) * pit_recent_mult * env_mult
            # --- Middle% (Heart%) tiny multiplier: higher heart% → small HR bump
            h_league = HEART.get("league", 0.15); h_pitch = HEART.get("pitcher", {}).get(opp_st.get("Pitcher_ID"), h_league)
            p_pa_vs_sp *= float(np.clip(1.0 + 0.35 * ((h_pitch / max(h_league, 1e-6)) - 1.0), 0.90, 1.12))

            p_pa_vs_bp = p_hr * bp_mult * env_mult
            p_pa_final = float(np.clip(sp_share * p_pa_vs_sp + (1.0 - sp_share) * p_pa_vs_bp, 0.003, 0.40))

            # exp-PA fallback by spot
            fallback_pa = (SPOT_PA_AWAY if team == away else SPOT_PA_HOME)[min(spot-1, 8)]
            exp_pa = SEASON_PA_G.get(nm, fallback_pa)

            # archetype + two experts
            stand = scb.get("stand", None)
            b_iso_cur = get_season_iso(bat_id)
            arch = infer_hr_archetype(nm, stand, b_iso_cur, scb.get("pull_pct", None))

            p_iso, dbg = calibrated_hr_prob(exp_pa, p_pa_final, arch, args.hr_scale)
            p_poi = _poisson_game_hr(exp_pa, p_pa_final, args.hr_scale)

            # blend alpha
            if isinstance(args.hr_blend, str) and args.hr_blend.lower()=="auto":
                alpha = np.nan
            else:
                try: alpha = float(args.hr_blend)
                except Exception: alpha = _load_alpha_memory(default=0.70)
                alpha = float(np.clip(alpha, 0.0, 1.0))
            if np.isnan(alpha):
                p_game_hr = p_iso; _p_iso_tmp, _p_poi_tmp = p_iso, p_poi
            else:
                p_game_hr = float(np.clip(alpha * p_iso + (1.0 - alpha) * p_poi, 1e-4, 0.95))
                _p_iso_tmp = _p_poi_tmp = None

            # research inputs for score & narrative
            bar14 = scb.get("barrel_14d_pct", np.nan)
            hh14  = scb.get("hardhit_14d_pct", np.nan)
            ev14  = scb.get("ev_mean_14d", np.nan)
            if (not np.isfinite(hh14) or not np.isfinite(ev14)) and sc180 is not None and "game_date" in sc180.columns:
                try:
                    cutoff = (datetime.today()-timedelta(days=14)).strftime("%Y-%m-%d")
                    rec14 = sc180[(sc180.batter==bat_id) & (sc180.game_date >= cutoff)].copy()
                    if not rec14.empty and "launch_speed" in rec14.columns:
                        rec14["ls"] = pd.to_numeric(rec14["launch_speed"], errors="coerce")
                        if not np.isfinite(hh14): hh14 = _safe_float((rec14["ls"] >= 95).mean(), np.nan)
                        if not np.isfinite(ev14): ev14 = _safe_float(rec14["ls"].mean(), np.nan)
                except Exception: pass

            season_hr_total = _safe_float(_bat_idx.loc[nm].get("HR", np.nan)) if nm in _bat_idx.index else np.nan
            sp_feats = fetch.get_statcast_pitcher_features(opp_st.get("Pitcher_ID")) or {}
            p_throws_val = str(sp_feats.get("throws", sp_feats.get("p_throws", "R"))).upper()[:1]
            if p_throws_val not in ("R","L"): p_throws_val = "R"

            # Prefer precomputed pitcher vs-side HR/PA rates (fast, robust)
            vs_rates_cached = PITCH_SIDE.get(opp_st.get("Pitcher_ID") or 0, {})
            if vs_rates_cached:
                sp_feats.setdefault("vs_L_HR_rate", vs_rates_cached.get("vs_L_HR_rate", np.nan))
                sp_feats.setdefault("vs_R_HR_rate", vs_rates_cached.get("vs_R_HR_rate", np.nan))



            spl = SEASON_HR_SPLITS.get(bat_id, {})
            season_hr_vs_side = _safe_float(spl.get("HR_vs_R" if p_throws_val=="R" else "HR_vs_L", np.nan))
            sp_hr9 = _safe_float(opp_st.get("_HR9", np.nan))
            if (scb.get("stand","R")).upper().startswith("L"):
                sp_hr_rate_vs_side = _safe_float(sp_feats.get("vs_L_HR_rate", np.nan))
            else:
                sp_hr_rate_vs_side = _safe_float(sp_feats.get("vs_R_HR_rate", np.nan))
            park_hr_idx  = _safe_float(pf_side, 1.0)
            wind_tail_pct_val = _safe_float(wind_tail, 0.0)
            mxiso_use = _safe_float(mxiso, np.nan)
            bp_hr9 = float(bp_hrpa * BF_PER_9) if np.isfinite(bp_hrpa) else np.nan

            score_raw = hr_matchup_score(bar14, hh14, ev14, season_hr_total, season_hr_vs_side,
                                         sp_hr9, sp_hr_rate_vs_side, park_hr_idx, wind_tail_pct_val, mxiso_use)
            score_calib = score_cal(score_raw) if score_cal else np.nan

            # narrative (kept off main table; used in shortlist)
            bbar  = f"{bar14*100:.1f}%" if np.isfinite(bar14) else "—"
            bhh   = f"{hh14*100:.1f}%" if np.isfinite(hh14) else "—"
            bev   = f"{ev14:.1f}"       if np.isfinite(ev14) else "—"
            shr   = f"{int(season_hr_total)}" if np.isfinite(season_hr_total) else "—"
            shrvs = f"{int(season_hr_vs_side)}" if np.isfinite(season_hr_vs_side) else "—"
            sp_name = opp_st.get("Name") or "Opp SP"
            sps_hr9 = f"{_safe_float(sp_hr9,0):.2f}" if np.isfinite(_safe_float(sp_hr9,np.nan)) else "—"
            sps_side= f"{_safe_float(sp_hr_rate_vs_side,np.nan):.3f}" if np.isfinite(_safe_float(sp_hr_rate_vs_side,np.nan)) else "—"
            pidx    = f"{park_hr_idx*100:.0f}%"; wtxt = f"{wind_tail_pct_val*100:.0f}%"
            bp9     = f"{_safe_float(bp_hr9,np.nan):.2f}" if np.isfinite(_safe_float(bp_hr9,np.nan)) else "—"
            mxs     = f"{mxiso_use:.3f}" if np.isfinite(mxiso_use) else "—"
            _explain = (f"last14: barrel {bbar}, hard-hit {bhh}, EV {bev}. Season HR {shr} ({shrvs} vs {p_throws_val}). "
                        f"{sp_name}: HR/9 {sps_hr9}, HR/PA vs side {sps_side}. Park {pidx}, wind tail {wtxt}, BP HR/9 {bp9}, xISO {mxs}.")

            if args.dbg_cal and spot <= 3:
                print(f"  [DBG] {team} #{spot} {nm}: arch={arch}, src={dbg['src']}, s={dbg['s_used']:.3f}, "
                      f"λ={dbg['lam_raw']:.4f}->{dbg['lam_scaled']:.4f}, P_iso={p_iso:.3f}, P_poi={p_poi:.3f}, P_blend={p_game_hr:.3f}")

            # non-HR events with ump BIP tweak
            bip_adj = _bip_adjustment(ump_feats)
            final = {"HR": p_pa_final}
            for e in ("1B","2B","3B","TB","R","RBI"):
                val = base[e]
                if e != "HR" and np.isfinite(barrel): val *= (1 + (barrel/100.0)*0.22)
                if e in ("1B","2B","3B","TB"): val *= bip_adj
                final[e] = shrink_vs_pitcher(val, exp_pa, base[e], prior_pa=2)

            # xwOBA nudge into TB
            if not bb.empty and "estimated_woba_using_speedangle" in bb.columns:
                try:
                    xw = float(bb["estimated_woba_using_speedangle"].mean())
                    final["TB"] = 0.5*final["TB"] + 0.5*(xw/1.25)
                except Exception: pass

            rates.append(final)

            # per-game lambdas
            lam1 = exp_pa * final["1B"]; lam2 = exp_pa * final["2B"]; lam3 = exp_pa * final["3B"]
            lam4 = -np.log(max(1.0 - p_game_hr, 1e-8))
            lamH = lam1 + lam2 + lam3 + lam4
            lamTB = exp_pa * final["TB"]; lamRBI = exp_pa * final["RBI"]; lamR = exp_pa * final["R"]

            batter_rows.append({
                "Team": team, "Spot": spot, "Name": nm_full, "exp_PA": round(exp_pa, 3),
                "P(Hits≥1)": _iso_or_poisson(("H",1), lamH, 1),
                "P(Hits≥2)": _iso_or_poisson(("H",2), lamH, 2),
                "P(Hits≥3)": _iso_or_poisson(("H",3), lamH, 3),
                "P(Hits≥4)": _iso_or_poisson(("H",4), lamH, 4),
                "P(1B≥1)": float(1 - poisson.cdf(0, lam1)),
                "P(2B≥1)": float(1 - poisson.cdf(0, lam2)),
                "P(3B≥1)": float(1 - poisson.cdf(0, lam3)),
                "P(HR≥1)": float(p_game_hr),
                "HR_iso": float(_p_iso_tmp if _p_iso_tmp is not None else p_iso),
                "HR_poi": float(_p_poi_tmp if _p_poi_tmp is not None else p_poi),
                "P(TB≥1)": _iso_or_poisson(("TB",1), lamTB, 1),
                "P(TB≥2)": _iso_or_poisson(("TB",2), lamTB, 2),
                "P(TB≥3)": _iso_or_poisson(("TB",3), lamTB, 3),
                "P(TB≥4)": _iso_or_poisson(("TB",4), lamTB, 4),
                "P(RBI≥1)": float(1 - poisson.cdf(0, lamRBI)),
                "P(Run≥1)": float(1 - poisson.cdf(0, lamR)),
                "HR_Score": float(score_raw),
                "Score_Cal": float(score_calib) if np.isfinite(_safe_float(score_calib, np.nan)) else np.nan,
                "_Explain": _explain,
            })

    # ---- auto-fit blend alpha from book (if requested) ----
    if isinstance(args.hr_blend, str) and args.hr_blend.lower()=="auto" and mk:
        pairs = []
        for r in batter_rows:
            nm_norm = clean_name(strip_pos(r["Name"]))
            if nm_norm in mk: pairs.append((r["HR_iso"], r["HR_poi"], mk[nm_norm]))
        if len(pairs) >= 8:
            xs_iso = np.array([a for a,_,_ in pairs], float)
            xs_poi = np.array([b for _,b,_ in pairs], float)
            y_book = np.array([c for _,_,c in pairs], float)
            alphas = np.linspace(0.0, 1.0, 51)
            mse = [np.mean((np.clip(a*xs_iso + (1-a)*xs_poi, 1e-4, 0.95) - y_book)**2) for a in alphas]
            alpha_auto = float(alphas[int(np.argmin(mse))])
            prev = _load_alpha_memory(default=0.70); alpha_auto = 0.8*prev + 0.2*alpha_auto; _save_alpha_memory(alpha_auto)
            for r in batter_rows:
                p = np.clip(alpha_auto*r["HR_iso"] + (1-alpha_auto)*r["HR_poi"], 1e-4, 0.95)
                r["P(HR≥1)"] = float(p)
            if args.dbg_cal: print(f"[auto α] learned={alpha_auto:.2f} (smoothed)")

    # ---- optional market remap (with history) ----
    if mk:
        pairs_today = []; x_model, y_book = [], []
        for r in batter_rows:
            nm_norm = clean_name(strip_pos(r["Name"]))
            if nm_norm in mk:
                model_p = r["P(HR≥1)"]; book_p  = mk[nm_norm]
                x_model.append(model_p); y_book.append(book_p); pairs_today.append((model_p, book_p))
        hist = _load_market_history()
        if hist: hx, hy = zip(*hist); x_model = list(x_model)+list(hx); y_book = list(y_book)+list(hy)
        f = fit_market_calibrator(x_model, y_book)
        if f:
            for r in batter_rows:
                pm = f(r["P(HR≥1)"])
                r["P(HR≥1)"] = float(np.clip(pm, max(1e-4, r["P(HR≥1)"]*(1-0.20)), min(0.95, r["P(HR≥1)"]*(1+0.20))))
        _append_market_log(pairs_today, use_date)

    # fill rare gaps for team rates
    def league_rate_dict():
        return {e: LEAGUE_BAT[e] for e in ("1B","2B","3B","HR","TB","RBI","R")}
    if len(away_rates) != 9: away_rates = [league_rate_dict()] * 9
    if len(home_rates) != 9: home_rates = [league_rate_dict()] * 9

    # Starter K/O table
    starter_rows = []
    for team, st, lineup in [(away,away_r,a_line),(home,home_r,h_line)]:
        sr = kt_montecarlo(st, lineup)
        sr.update({"Team": team, "Name": st["Name"], "Pitcher_ID": st["Pitcher_ID"], "Proj_IP": st["Proj_IP"],
                   "IP_per_start": st["IP_per_start"], "Pred_K9": st["Pred_K9"]})
        starter_rows.append(sr)
    df_ko = pd.DataFrame(starter_rows)
    ko_cols = ["Team","Name","Pitcher_ID","Proj_IP","IP_per_start","Pred_K9","Mean_K_start","Median_K_start"] + \
              [f"P(K≥{k})" for k in range(2,11)] + ["90% CI"]
    print("\n=== Starter KO Probabilities ===")
    print(df_ko[ko_cols].to_string(index=False))

    # Batter table
    df_bat = pd.DataFrame(batter_rows)
    if mk and not df_bat.empty:
        names_norm = [clean_name(strip_pos(n)) for n in df_bat["Name"]]
        missing = [n for n in names_norm if n not in mk]
        if missing:
            msg = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
            print(f"[market] no odds for {len(missing)} lineup players: {msg}")

    # Decorate with book columns + odds + EV (no Kelly)
    if not df_bat.empty:
        if getattr(args, "show_odds", False) or mk:
            df_bat["HR_Odds"] = df_bat["P(HR≥1)"].apply(prob_to_american)
        if mk:
            df_bat["Book_Prob"] = [ mk.get(clean_name(strip_pos(n)), np.nan) for n in df_bat["Name"] ]
            df_bat["Book_Odds"] = df_bat["Book_Prob"].apply(lambda p: prob_to_american(p) if pd.notna(p) else "—")
            df_bat["Edge"]      = df_bat["P(HR≥1)"] - df_bat["Book_Prob"]
            dprice = df_bat["Book_Prob"].apply(_dec_from_prob)
            df_bat["EV_%"]     = (df_bat["P(HR≥1)"] * dprice - 1.0) * 100.0

        # Score percentile for today’s slate
        if "HR_Score" in df_bat.columns:
            df_bat["Score_Pctl"] = (df_bat["HR_Score"].rank(pct=True) * 100.0).round(1)

    # print table
    if not df_bat.empty:
        base_cols = ["Team","Spot","Name","exp_PA","P(HR≥1)"]
        if "HR_Odds" in df_bat.columns: base_cols += ["HR_Odds"]
        if "Book_Prob" in df_bat.columns: base_cols += ["Book_Prob","Book_Odds","Edge","EV_%"]
        if "HR_Score" in df_bat.columns: base_cols += ["HR_Score","Score_Pctl"]
        more_cols = [c for c in df_bat.columns if c.startswith("P(") and c not in {"P(HR≥1)"}]
        cols = base_cols + sorted(more_cols)
        print("\n=== Batter Matchup Props ===")
        print(df_bat[cols].to_string(index=False))
    else:
        logging.warning("No batter matchup props; skipping.")

    # Top value targets (if market)
    if not df_bat.empty and "EV_%" in df_bat.columns:
        rec = df_bat.copy()
        rec = rec[(~rec["EV_%"].isna()) & (rec["P(HR≥1)"] >= 0.06)]
        if not rec.empty:
            rec = rec.sort_values(["EV_%","HR_Score","P(HR≥1)"], ascending=[False, False, False]).head(10)
            print("\n[Top value targets by EV then Score]")
            print(rec[["Team","Name","P(HR≥1)","HR_Odds","Book_Odds","EV_%","HR_Score"]].to_string(index=False))

    # -------- Shortlist (clean + neat) --------
    if not df_bat.empty:
        picks = pick_top_hr_targets(df_bat, limit=8, min_prob=0.06)
        if not picks.empty:
            print("\n[Shortlist] Best HR targets (model P + score{}):")
            lines = []
            for _, r in picks.iterrows():
                fair = prob_to_american(r["P(HR≥1)"])
                score= r.get("HR_Score", np.nan)
                pctl = r.get("Score_Pctl", np.nan)
                expl = r.get("_Explain", "")
                lines.append(
                    f"- {r['Team']} {r['Name']}: P(HR≥1) {r['P(HR≥1)']:.3f} (fair {fair}) | "
                    f"Score {score:.1f} (pct {pctl:.0f})\n  · {expl}"
                )
            print("\n".join(lines))

    # CSV export
    if not df_bat.empty and args.csv:
        df_bat.to_csv(args.csv, index=False)
        print(f"\n[wrote] {args.csv}")

    # Team quick totals
    def _team_quick_totals(team, lineup, rates):
        lamH = lamHR = lamTB = 0.0
        for spot, nm_full in enumerate(lineup, start=1):
            nm = clean_name(strip_pos(nm_full))
            fallback_pa = (SPOT_PA_AWAY if team == away else SPOT_PA_HOME)[min(spot-1, 8)]
            exp_pa = SEASON_PA_G.get(nm, fallback_pa)
            r = rates[spot-1]
            stand = statcast_bat_cache.get(name_to_mlbam_id(nm), {}).get("stand", None)
            b_iso_cur = get_season_iso(name_to_mlbam_id(nm))
            arch = infer_hr_archetype(nm, stand, b_iso_cur, statcast_bat_cache.get(name_to_mlbam_id(nm), {}).get("pull_pct", None))

            p_game_hr, _ = calibrated_hr_prob(exp_pa, r["HR"], arch, args.hr_scale)
            lam4 = -np.log(max(1.0 - p_game_hr, 1e-8))
            lamH  += exp_pa*(r["1B"] + r["2B"] + r["3B"]) + lam4
            lamHR += lam4
            lamTB += exp_pa*r["TB"]
        return lamH, lamHR, lamTB

    print("\n=== Team Quick Totals (expected per game) ===")
    for team, lineup, rates in [(away,a_line,away_rates), (home,h_line,home_rates)]:
        lamH, lamHR, lamTB = _team_quick_totals(team, lineup, rates)
        print(f"{team}:  Exp Hits {lamH:.2f}  |  Exp HR {lamHR:.2f}  |  Exp TB {lamTB:.2f}")

    # Optional Monte Carlo
    if args.sims and args.sims > 0:
        print(f"\n=== Simulating {args.sims} Games ===")
        aw_w=hm_w=tie=0; aw_runs=[]; hm_runs=[]
        dfA, profA = build_bullpen_dataframe(away), profile_bullpen(build_bullpen_dataframe(away))
        dfH, profH = build_bullpen_dataframe(home), profile_bullpen(build_bullpen_dataframe(home))
        for _ in trange(args.sims, desc="MC"):
            sa = simulate_full_game(away, a_line, away_r, dfA, profA, away_rates)
            sh = simulate_full_game(home, h_line, home_r, dfH, profH, home_rates)
            ra, rb = sa["Runs"], sh["Runs"]; aw_runs.append(ra); hm_runs.append(rb)
            if ra > rb: aw_w += 1
            elif rb > ra: hm_w += 1
            else: tie += 1
        print(f"{away} avg: {np.mean(aw_runs):.1f}, {home} avg: {np.mean(hm_runs):.1f}")
        print(f"Win% — {away}: {aw_w/args.sims:.3f}, {home}: {hm_w/args.sims:.3f}, Tie: {tie/args.sims:.3f}")
if __name__ == "__main__":
    main()
