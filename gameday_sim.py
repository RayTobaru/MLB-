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
    predict_hr_count_pt, predict_hr_hardhit, predict_hr_2swhiff, predict_hr_pullangle,
    iso_pa_calibrators, expected_matchup_xiso,
    hr_game_cals, HR_LAMBDA_SCALE, infer_hr_archetype, sp_pa_share
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
    """Convert probability (0..1) to American odds string."""
    try:
        p = float(p)
    except Exception:
        return "—"
    p = min(max(p, 1e-6), 0.9999)
    if p >= 0.5:
        return f"{-int(round(100 * p / (1 - p)))}"
    else:
        return f"+{int(round(100 * (1 - p) / p))}"

def american_to_prob(odds):
    o = float(odds)
    return (100.0/(o+100.0)) if o>0 else (-o/(-o+100.0))

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
    if name_norm not in _bat_idx.index:
        return 0.67
    row = _bat_idx.loc[name_norm]
    k = _pct_or_ratio(row, ["K%","SO%"], "SO")
    bb = _pct_or_ratio(row, ["BB%"], "BB")
    hbp = float(row.get("HBP",0))/max(float(row.get("PA",1)),1.0)
    if not (np.isfinite(k) and np.isfinite(bb) and np.isfinite(hbp)):
        return 0.67
    return float(np.clip(1.0 - k - bb - hbp, 0.45, 0.82))

def build_starter_dict(team, name, pid, ump_feats=None, framing_feats=None, weather=None):
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
            if c in rec:
                r[c] = float(rec[c])
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
    r["Pred_K9"] = predict_k9(pd.Series(r))
    pred_mean_ip = float(IP_REG.predict([[ipps, r["Pred_K9"], r["Pred_K9"]]])[0])
    r["Proj_IP"] = min(sample_starter_ip(pred_mean_ip, r["Pred_K9"]), 7.0)
    ws = (weather or {}).get("wind_mph")
    if isinstance(ws,(int,float)) and ws > 15:
        r["Pred_K9"] *= (1 + min(ws/100, 0.05))
    if ump_feats and "ump_k9" in ump_feats:
        r["Pred_K9"] *= float(ump_feats["ump_k9"])
    frame_pct = (framing_feats or {}).get(f"{team.lower()}_frame", 0.0)
    r["Pred_K9"] *= (1 + frame_pct/100.0)
    r.update({"Team": team, "Name": name, "Pitcher_ID": pid})
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
    kf = float(ump.get("ump_k9",1.0))
    bf = float(ump.get("ump_bb9",1.0))
    return 1.0 / (1.0 + 0.4*(kf-1.0) + 0.2*(bf-1.0))

def calibrated_hr_prob(exp_pa: float, p_hr_pa: float, arch: str, hr_scale_override: float|None=None):
    lam = float(exp_pa) * float(p_hr_pa)
    s = float(hr_scale_override) if hr_scale_override else float(HR_LAMBDA_SCALE)
    iso = hr_game_cals.get(("HR",1,arch)) or hr_game_cals.get(("HR",1,"ALL")) or hr_game_cals.get(("HR",1))
    if iso is not None:
        p = float(iso.predict([s*lam])[0]); src = f"iso[{ 'arch' if ('HR',1,arch) in hr_game_cals else 'ALL' }]"
    else:
        p = 1.0 - np.exp(-lam)
        src = "poisson"
    p = float(np.clip(p, 0.01, 0.40))
    return p, {"lam_raw": lam, "lam_scaled": s*lam, "src": src, "s_used": s}

def bullpen_hrpa(team_abbr: str) -> float:
    df = pitching_stats(YEAR, qual=0)
    if df is None or df.empty:
        return LEAGUE_HR_PER_PA
    rel = df[(df.Team == team_abbr) & (df.G > df.GS) & (df.IP > 0)]
    if rel.empty:
        return LEAGUE_HR_PER_PA
    hr9 = float(9.0 * rel.HR.sum() / max(rel.IP.sum(), 1e-6))
    return float(np.clip(hr9 / BF_PER_9, 0.002, 0.05))

# ---------- Optional sportsbook alignment helpers ----------
def load_market_probs(path: str) -> dict[str, float]:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("name") or cols.get("player") or cols.get("batter")
    if not name_col:
        raise ValueError("market CSV must have Name/Player/Batter column")
    if "Prob" in df.columns:
        p = pd.to_numeric(df["Prob"], errors="coerce").astype(float)
    elif "prob" in df.columns:
        p = pd.to_numeric(df["prob"], errors="coerce").astype(float)
    elif "Odds" in df.columns:
        p = df["Odds"].apply(american_to_prob)
    elif "odds" in df.columns:
        p = df["odds"].apply(american_to_prob)
    else:
        raise ValueError("market CSV must include Prob or Odds")
    p = np.where(p>1, p/100.0, p)
    out={}
    for nm, ip in zip(df[name_col], p):
        if pd.notna(ip):
            out[clean_name(strip_pos(str(nm)))] = float(np.clip(ip, 1e-4, 0.95))
    return out

def fit_market_calibrator(p_model, p_book):
    x = np.asarray(p_model, float); y = np.asarray(p_book, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x>0) & (x<1) & (y>0) & (y<1)
    x, y = x[ok], y[ok]
    if len(x) < 12:
        return None
    try:
        # split around 0.20 (≈ +400)
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
        # last ditch: Platt scaling on logits
        from sklearn.linear_model import LogisticRegression
        X = np.log(x/(1-x)).reshape(-1,1)
        lr = LogisticRegression(solver="lbfgs").fit(X, y)
        return lambda p: float(np.clip(lr.predict_proba([[np.log(p/(1-p))]])[0,1], 1e-4, 0.95))

# ---------- Dual-expert helpers ----------
def _poisson_game_hr(exp_pa: float, p_hr_pa: float, scale: float|None) -> float:
    s = float(scale) if scale is not None else float(HR_LAMBDA_SCALE)
    lam = float(exp_pa) * float(p_hr_pa) * s
    return float(np.clip(1.0 - np.exp(-lam), 1e-4, 0.95))

# ---------- Date + slate ----------
def _resolve_date(args) -> str:
    base = _date.today()
    if getattr(args, "tomorrow", False):
        base = base + timedelta(days=1)
    elif getattr(args, "yesterday", False):
        base = base - timedelta(days=1)
    elif getattr(args, "date", None):
        return args.date
    return base.strftime("%Y-%m-%d")

def fetch_matchups_for_date(date_str: str):
    resp = safe_get("https://statsapi.mlb.com/api/v1/schedule",
                    {"sportId": 1, "date": date_str})
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

# ---------- FDodds vertical loader ----------
def _load_fd_vertical(path: str) -> dict[str, float]:
    """Parse file like: Name on row i, +330 on row i+1."""
    s = pd.read_csv(path, header=None).iloc[:, 0].astype(str).tolist()
    out = {}
    i = 0
    while i + 1 < len(s):
        nm = clean_name(strip_pos(s[i]))
        raw = str(s[i+1]).strip().replace("+", "")
        try:
            odds = float(raw)
        except Exception:
            i += 1
            continue
        out[nm] = float(np.clip(american_to_prob(odds), 1e-4, 0.95))
        i += 2
    return out

def _try_load_fdodds_default() -> dict[str, float]:
    for fn in ("FDodds.csv", "FDodds.CSV"):
        if os.path.exists(fn):
            return _load_fd_vertical(fn)
    return {}

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
    except Exception:
        pass

# ---------- Market history memory ----------
_MARKET_LOG = os.path.join("cache", "market_history_fd.csv")

def _append_market_log(pairs, date_str):
    # pairs: list of (model_p, book_p)
    if not pairs: return
    df = pd.DataFrame(pairs, columns=["model_p", "book_p"])
    df["date"] = date_str
    try:
        os.makedirs(os.path.dirname(_MARKET_LOG), exist_ok=True)
        if os.path.exists(_MARKET_LOG):
            df_old = pd.read_csv(_MARKET_LOG)
            df = pd.concat([df_old, df], ignore_index=True)
        df.tail(3000).to_csv(_MARKET_LOG, index=False)
    except Exception:
        pass

def _load_market_history():
    try:
        df = pd.read_csv(_MARKET_LOG)
        return df[["model_p","book_p"]].dropna().values.tolist()
    except Exception:
        return []

# ---------- Safety cap for remap ----------
def _cap_move(p_model, p_mapped, rel_cap=0.20):
    lo = p_model*(1-rel_cap); hi = p_model*(1+rel_cap)
    return float(np.clip(p_mapped, max(1e-4, lo), min(0.95, hi)))

# -----------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-g","--game", type=int)
    p.add_argument("-m","--matchup")
    p.add_argument("--sims", type=int, default=0, help="Number of game sims to run (0 = skip)")
    p.add_argument("--dbg-cal", action="store_true", help="Print HR calibrator debug lines")
    p.add_argument("--hr-scale", type=float, default=None, help="Override HR lambda scale")
    p.add_argument("--show-odds", dest="show_odds", action="store_true",
                   help="Add American odds column for P(HR≥1)")
    p.add_argument("--market", type=str, default=None, help="CSV with Name + Odds or Prob for HR")
    p.add_argument("--hr-blend", type=str, default="0.7",
                   help="Blend toward calibrated HR model (0..1) or 'auto' to learn from book")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible sims")
    p.add_argument("--csv", type=str, default=None, help="Write batter props to CSV")
    p.add_argument("--date", type=str, default=None, help="ISO date (YYYY-MM-DD). Defaults to today.")
    p.add_argument("--today", action="store_true", help="Force today (local).")
    p.add_argument("--tomorrow", action="store_true", help="Force tomorrow (local).")
    p.add_argument("--yesterday", action="store_true", help="Force yesterday (local).")
    args = p.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    use_date = _resolve_date(args)
    games = fetch_matchups_for_date(use_date)
    if not games:
        sys.exit("No MLB games on that date.")
    for i, gm in enumerate(games, 1):
        print(f"{i}. {gm}")

    if args.game and 1<=args.game<=len(games):
        sel = str(args.game)
    elif args.matchup:
        sel = args.matchup
    else:
        sel = input("Pick # or 'AWY @ HOM': ").strip()

    away, home = (games[int(sel)-1].split(" @ ") if sel.isdigit()
                  else tuple(map(str.strip, sel.split("@"))))

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
        a_line = fetch_unofficial_lineup(away)
        h_line = fetch_unofficial_lineup(home)
        a_name = h_name = None
        a_pid = h_pid = None
        ump_feats = {"ump_k9":1.0,"ump_bb9":1.0}
        framing_feats = {"away_frame":0.0,"home_frame":0.0}

    # gamePk, weather
    sched = safe_get("https://statsapi.mlb.com/api/v1/schedule",
                     {"sportId":1,"date":use_date,"hydrate":"probablePitchers"})
    js = sched.json() if sched else {}
    game = next((g for d in js.get("dates",[]) for g in d["games"]
                 if (ID2ABBR[g["teams"]["away"]["team"]["id"]],
                     ID2ABBR[g["teams"]["home"]["team"]["id"]])==(away,home)), {})
    gamePk = game.get("gamePk")
    weather = get_game_weather(gamePk) if gamePk else {}
    wind_tail = tail_wind_pct(gamePk) if gamePk else 0.0
    temp_f = None
    for k in ("temp_f","temperature","tempF","temp"):
        v = weather.get(k)
        if isinstance(v,(int,float)):
            temp_f = float(v); break

    away_r = build_starter_dict(away, a_name, a_pid, ump_feats=ump_feats,
                                framing_feats={"away_frame":framing_feats.get("away_frame",0.0)}, weather=weather)
    home_r = build_starter_dict(home, h_name, h_pid, ump_feats=ump_feats,
                                framing_feats={"home_frame":framing_feats.get("home_frame",0.0)}, weather=weather)

    # Today's park = home park for both teams
    mpf = fetch_monthly_park_factors(YEAR)
    month = datetime.today().month

    away_rates, home_rates = [], []
    batter_rows = []

    if args.dbg_cal:
        sample_keys = list(hr_game_cals.keys())[:4]
        print(f"[DBG] HR_LAMBDA_SCALE={HR_LAMBDA_SCALE:.3f} | hr_game_cals={len(hr_game_cals)} sample={sample_keys}")

    # Pre-fetch 180d Statcast once
    sc180 = fetch_statcast_raw(
        (datetime.today()-timedelta(days=180)).strftime("%Y-%m-%d"),
        datetime.today().strftime("%Y-%m-%d")
    )

    # market input (FDodds auto or --market)
    mk = load_market_probs(args.market) if args.market else _try_load_fdodds_default()

    # ---- build per-batter rows ----
    for team, lineup, st, rates in [(away,a_line,away_r,away_rates), (home,h_line,home_r,home_rates)]:
        opp_st        = home_r if team == away else away_r
        opp_team_abbr = home   if team == away else away
        bp_hrpa       = bullpen_hrpa(opp_team_abbr)  # per-PA for bullpen
        bp_mult       = float(np.clip((bp_hrpa / LEAGUE_HR_PER_PA) if bp_hrpa > 0 else 1.0, 0.60, 1.60))

        for spot, nm_full in enumerate(lineup, start=1):
            nm = clean_name(strip_pos(nm_full))
            bat_id = name_to_mlbam_id(nm)
            base = LEAGUE_BAT.copy()

            # H2H shrink vs OPP STARTER
            h2h = h2h_cache.get((bat_id, opp_st["Pitcher_ID"]), {})
            if h2h.get("PA",0) >= 3:
                pa = float(h2h["PA"])
                prior_mass = 4.0 if pa >= 10 else 6.0
                for e in ("1B","2B","3B","HR","TB","RBI","R"):
                    r_obs = float(h2h.get(f"{e}_rate", base[e]))
                    base[e] = shrink_vs_pitcher(r_obs, pa, base[e], prior_pa=prior_mass)

            scb = statcast_bat_cache.get(bat_id, {})
            barrel = scb.get("barrel_pct", np.nan)
            ev_mean = scb.get("ev_mean", np.nan)
            la_mean = scb.get("la_mean", np.nan)

            # 180d slice for sweet-spot
            if sc180 is not None and "batter" in sc180.columns:
                bb = sc180[sc180.batter == bat_id]
            else:
                bb = pd.DataFrame()

            hr_feat = {
                "exit_velocity": ev_mean if np.isfinite(ev_mean) else 0.0,
                "launch_angle":  la_mean if np.isfinite(la_mean) else 0.0,
                "barrel_pct":    barrel if np.isfinite(barrel) else 0.0,
                "HR_FB_rate":    scb.get("HR_FB_rate", 0.0),
                "pull_pct":      scb.get("pull_pct", 0.0),
                "park_hr_factor": 1.0,  # per-batter park comes below
            }
            if not bb.empty and {"launch_speed","launch_angle"}.issubset(bb.columns):
                mask = (bb.launch_speed > 105) & bb.launch_angle.between(25, 35)
                hr_feat["sweet_spot_frac"] = float(mask.mean())
            else:
                hr_feat["sweet_spot_frac"] = 0.0

            feat = pd.Series(hr_feat)
            # per-PA HR (stacked model + micros)
            p_stack = float(predict_hr_proba(feat))
            p_sub = np.mean([
                predict_hr_count_pt(feat),
                predict_hr_hardhit(feat),
                predict_hr_2swhiff(feat),
                predict_hr_pullangle(feat)
            ])
            p_hr = 0.75*p_stack + 0.25*p_sub

            # recent barrels + ISO bump
            b14 = scb.get("barrel_14d_pct", np.nan)
            if np.isfinite(b14):
                p_hr *= (1.0 + 0.6*float(b14))
            try:
                b_iso = float(_bat_idx.loc[nm].get("ISO", np.nan))
                lg_iso = float(_bat["ISO"].mean())
                if np.isfinite(b_iso) and np.isfinite(lg_iso) and lg_iso > 0:
                    p_hr *= float(np.clip(1.0 + 1.10*(b_iso-lg_iso)/lg_iso, 0.75, 1.45))
            except Exception:
                pass

            # matchup xISO (pitch mix × batter xISO by pitch) vs OPP STARTER
            if bat_id and opp_st.get("Pitcher_ID"):
                try:
                    mxiso = expected_matchup_xiso(opp_st["Pitcher_ID"], bat_id)
                except Exception:
                    mxiso = 0.165
                p_hr *= float(np.clip(1.0 + 1.10*(mxiso-0.165), 0.75, 1.35))

            # --- Environment with handed park & wind sensitivity ---
            env_mult = 1.0
            if temp_f is not None:
                env_mult *= _temp_mult(temp_f)

            pf_team = mpf.get(home, {}).get(month, {})
            stand_key = "HR_L" if (scb.get("stand","R")).upper().startswith("L") else "HR_R"
            pf_side = pf_team.get(stand_key, pf_team.get("HR", 100)) / 100.0
            env_mult *= pf_side

            pullish = float(np.clip(scb.get("pull_pct", 0.4), 0.2, 0.7))
            if wind_tail > 0:
                env_mult *= (1.0 + 0.6 * wind_tail * (0.6 + 0.8*pullish))

            # --- SP vs BP blend (preserve batter skill)
            sp_share   = sp_pa_share(opp_st.get("Pitcher_ID"))
            p_pa_vs_sp = p_hr * _pitcher_hr_mult(opp_st) * env_mult
            p_pa_vs_bp = p_hr * bp_mult * env_mult
            p_pa_final = sp_share * p_pa_vs_sp + (1.0 - sp_share) * p_pa_vs_bp
            p_pa_final = float(np.clip(p_pa_final, 0.003, 0.40))

            # --- exp-PA fallback by spot
            fallback_pa = (SPOT_PA_AWAY if team == away else SPOT_PA_HOME)[min(spot-1, 8)]
            exp_pa = SEASON_PA_G.get(nm, fallback_pa)

            # --- archetype + two experts
            stand = scb.get("stand", None)
            try:
                b_iso_cur = float(_bat_idx.loc[nm].get("ISO", np.nan))
            except Exception:
                b_iso_cur = np.nan
            arch = infer_hr_archetype(nm, stand, b_iso_cur, scb.get("pull_pct", None))

            p_iso, dbg = calibrated_hr_prob(exp_pa, p_pa_final, arch, args.hr_scale)
            p_poi = _poisson_game_hr(exp_pa, p_pa_final, args.hr_scale)

            # Determine alpha (blend)
            if isinstance(args.hr_blend, str) and args.hr_blend.lower()=="auto":
                alpha = np.nan  # set later (post-loop) from book
            else:
                try:
                    alpha = float(args.hr_blend)
                except Exception:
                    alpha = _load_alpha_memory(default=0.70)
                alpha = float(np.clip(alpha, 0.0, 1.0))

            if np.isnan(alpha):
                p_game_hr = p_iso  # placeholder
                _p_iso_tmp, _p_poi_tmp = p_iso, p_poi
            else:
                p_game_hr = float(np.clip(alpha * p_iso + (1.0 - alpha) * p_poi, 1e-4, 0.95))
                _p_iso_tmp = _p_poi_tmp = None

            if args.dbg_cal and spot <= 3:
                print(f"  [DBG] {team} #{spot} {nm}: arch={arch}, src={dbg['src']}, s={dbg['s_used']:.3f}, "
                      f"λ={dbg['lam_raw']:.4f}->{dbg['lam_scaled']:.4f}, "
                      f"P_iso={p_iso:.3f}, P_poi={p_poi:.3f}, P_blend={p_game_hr:.3f}")

            # Build other per-PA rates from baseline
            bip_adj = _bip_adjustment(ump_feats)
            final = {"HR": p_pa_final}
            for e in ("1B","2B","3B","TB","R","RBI"):
                val = base[e]
                if e != "HR" and np.isfinite(barrel):
                    val *= (1 + (barrel/100.0)*0.22)
                if e in ("1B","2B","3B","TB"):
                    val *= bip_adj
                final[e] = shrink_vs_pitcher(val, exp_pa, base[e], prior_pa=2)

            # xwOBA nudge into TB
            if not bb.empty and "estimated_woba_using_speedangle" in bb.columns:
                try:
                    xw = float(bb["estimated_woba_using_speedangle"].mean())
                    final["TB"] = 0.5*final["TB"] + 0.5*(xw/1.25)
                except Exception:
                    pass

            rates.append(final)

            # Per-game λ back-outs
            lam1 = exp_pa * final["1B"]
            lam2 = exp_pa * final["2B"]
            lam3 = exp_pa * final["3B"]
            lam4 = -np.log(max(1.0 - p_game_hr, 1e-8))
            lamH = lam1 + lam2 + lam3 + lam4
            lamTB = exp_pa * final["TB"]
            lamRBI = exp_pa * final["RBI"]
            lamR = exp_pa * final["R"]

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
            })

    # ---- auto-fit blend alpha from book (if requested) ----
    alpha_auto = None
    if isinstance(args.hr_blend, str) and args.hr_blend.lower()=="auto" and mk:
        pairs = []
        for r in batter_rows:
            nm_norm = clean_name(strip_pos(r["Name"]))
            if nm_norm in mk:
                pairs.append((r["HR_iso"], r["HR_poi"], mk[nm_norm]))
        if len(pairs) >= 8:
            xs_iso = np.array([a for a,_,_ in pairs], float)
            xs_poi = np.array([b for _,b,_ in pairs], float)
            y_book = np.array([c for _,_,c in pairs], float)
            alphas = np.linspace(0.0, 1.0, 51)
            mse = [np.mean((np.clip(a*xs_iso + (1-a)*xs_poi, 1e-4, 0.95) - y_book)**2) for a in alphas]
            alpha_auto = float(alphas[int(np.argmin(mse))])
            # Smooth with memory and persist
            prev = _load_alpha_memory(default=0.70)
            alpha_auto = 0.8*prev + 0.2*alpha_auto
            _save_alpha_memory(alpha_auto)
            # Apply to all rows
            for r in batter_rows:
                p = np.clip(alpha_auto*r["HR_iso"] + (1-alpha_auto)*r["HR_poi"], 1e-4, 0.95)
                r["P(HR≥1)"] = float(p)
            if args.dbg_cal:
                print(f"[auto α] learned={alpha_auto:.2f} (smoothed)")

    # ---- optional market remap (with history) ----
    if mk:
        x_model, y_book = [], []
        for r in batter_rows:
            nm_norm = clean_name(strip_pos(r["Name"]))
            if nm_norm in mk:
                x_model.append(r["P(HR≥1)"]); y_book.append(mk[nm_norm])
        # add memory
        hist = _load_market_history()
        if hist:
            hx, hy = zip(*hist)
            x_model = list(x_model) + list(hx)
            y_book  = list(y_book)  + list(hy)
        f = fit_market_calibrator(x_model, y_book)
        if f:
            for r in batter_rows:
                pm = f(r["P(HR≥1)"])
                r["P(HR≥1)"] = _cap_move(r["P(HR≥1)"], pm, rel_cap=0.20)
        # log today (without history)
        _append_market_log(list(zip(x_model[:len(y_book)], y_book[:len(x_model)])), use_date)

    # fill rare gaps
    def league_rate_dict():
        return {e: LEAGUE_BAT[e] for e in ("1B","2B","3B","HR","TB","RBI","R")}
    if len(away_rates) != 9:
        away_rates = [league_rate_dict()] * 9
    if len(home_rates) != 9:
        home_rates = [league_rate_dict()] * 9

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

    # Batter props table
    df_bat = pd.DataFrame(batter_rows)

    # Decorate with book columns + HR_Odds before any jitter
    if not df_bat.empty:
        if mk:
            df_bat["Book_Prob"] = [ mk.get(clean_name(strip_pos(n)), np.nan) for n in df_bat["Name"] ]
            df_bat["Book_Odds"] = df_bat["Book_Prob"].apply(lambda p: prob_to_american(p) if pd.notna(p) else "—")
            df_bat["Edge"] = df_bat["P(HR≥1)"] - df_bat["Book_Prob"]
        if getattr(args, "show_odds", False) or mk:
            df_bat["HR_Odds"] = df_bat["P(HR≥1)"].apply(prob_to_american)

    # light, deterministic display jitter only when NO market
    if not df_bat.empty and not mk:
        today_seed = int(datetime.today().strftime("%Y%m%d"))
        def _jit(name, p):
            rng = np.random.default_rng(abs(hash((today_seed, clean_name(strip_pos(name))))) % (2**32))
            j = (rng.random() - 0.5) * 0.003  # ±0.15%
            return float(np.clip(float(p) + j, 5e-4, 0.9995))
        for c in [c for c in df_bat.columns if c.startswith("P(")]:
            df_bat[c] = [ _jit(n, v) for n, v in zip(df_bat["Name"], df_bat[c]) ]

    if not df_bat.empty:
        base_cols = ["Team","Spot","Name","exp_PA","P(HR≥1)"]
        if "HR_Odds" in df_bat.columns:
            base_cols += ["HR_Odds"]
        if "Book_Prob" in df_bat.columns:
            base_cols += ["Book_Prob","Book_Odds","Edge"]
        more_cols = [c for c in df_bat.columns if c.startswith("P(") and c not in {"P(HR≥1)"}]
        cols = base_cols + sorted(more_cols)
        print("\n=== Batter Matchup Props ===")
        print(df_bat[cols].to_string(index=False))
    else:
        logging.warning("No batter matchup props; skipping.")

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
            try:
                b_iso_cur = float(_bat_idx.loc[nm].get("ISO", np.nan))
            except Exception:
                b_iso_cur = np.nan
            arch = infer_hr_archetype(nm, stand, b_iso_cur,
                                      statcast_bat_cache.get(name_to_mlbam_id(nm), {}).get("pull_pct", None))
            p_game_hr, _ = calibrated_hr_prob(exp_pa, r["HR"], arch, args.hr_scale)
            lam4 = -np.log(max(1.0 - p_game_hr, 1e-8))
            lamH += exp_pa*(r["1B"] + r["2B"] + r["3B"]) + lam4
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
