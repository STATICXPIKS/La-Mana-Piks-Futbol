import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List

_MODEL_BASE_CACHE: Dict[str, Any] = {}


def _american_to_decimal(odds_val: Any) -> float:
    if odds_val is None:
        return 0.0
    try:
        val = float(odds_val)
    except (ValueError, TypeError):
        return 0.0
    if val == 0:
        return 0.0
    if 1.01 <= val <= 50.0:
        return round(val, 3)
    if val >= 100:
        return round((val / 100.0) + 1.0, 3)
    if val <= -100:
        return round((100.0 / abs(val)) + 1.0, 3)
    return 0.0


def _learn_league_parameters(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "home_adv": 1.21,
            "rho": -0.075,
            "shot_conversion": 0.31,
            "avg_home_goals": 1.52,
            "avg_away_goals": 1.24
        }

    avg_hg = float(df["FTHG"].mean())
    avg_ag = float(df["FTAG"].mean())
    home_adv = round(max(1.05, min(1.38, avg_hg / max(avg_ag, 0.85))), 4)

    total_games = len(df)
    p_00 = len(df[(df["FTHG"] == 0) & (df["FTAG"] == 0)]) / max(total_games, 1)
    p_11 = len(df[(df["FTHG"] == 1) & (df["FTAG"] == 1)]) / max(total_games, 1)
    expected_00 = math.exp(-(avg_hg + avg_ag))
    excess_low = (p_00 + p_11) - (expected_00 + expected_00 * avg_hg * avg_ag)
    rho = round(max(-0.14, min(-0.02, -0.065 - excess_low * 0.45)), 4)

    sot_df = df[(df["HST"] + df["AST"]) > 0]
    if not sot_df.empty:
        conv = float((sot_df["FTHG"] + sot_df["FTAG"]).sum() / (sot_df["HST"] + sot_df["AST"]).sum())
    else:
        conv = 0.31

    return {
        "home_adv": home_adv,
        "rho": rho,
        "shot_conversion": round(conv, 4),
        "avg_home_goals": round(avg_hg, 3),
        "avg_away_goals": round(avg_ag, 3)
    }


def _compute_elo_ratings(df: pd.DataFrame) -> Dict[str, float]:
    elo: Dict[str, float] = {}
    if df.empty:
        return elo

    df_sorted = df.sort_values("Date")
    k_base = 22.0
    home_bonus = 55.0

    for h, a, hg, ag in zip(
        df_sorted["HomeTeam"].tolist(),
        df_sorted["AwayTeam"].tolist(),
        df_sorted["FTHG"].astype(int).tolist(),
        df_sorted["FTAG"].astype(int).tolist()
    ):
        eh = elo.get(h, 1500.0)
        ea = elo.get(a, 1500.0)

        exp_h = 1.0 / (1.0 + 10.0 ** ((ea - (eh + home_bonus)) / 400.0))
        res_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)

        goal_diff = abs(hg - ag)
        mult = 1.0 if goal_diff <= 1 else (1.28 if goal_diff == 2 else 1.45)

        delta = k_base * mult * (res_h - exp_h)
        elo[h] = eh + delta
        elo[a] = ea - delta

    return {k: round(v, 1) for k, v in elo.items()}


def _compute_attack_defense_strengths(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if df.empty:
        return {}

    df_sorted = df.sort_values("Date").copy()
    max_date = df_sorted["Date"].max()
    days_ago = (max_date - df_sorted["Date"]).dt.days.clip(lower=0)
    df_sorted["weight"] = np.exp(-days_ago / 145.0)

    w_sum = df_sorted["weight"].sum()
    league_hg = float((df_sorted["FTHG"] * df_sorted["weight"]).sum() / max(w_sum, 1e-6))
    league_ag = float((df_sorted["FTAG"] * df_sorted["weight"]).sum() / max(w_sum, 1e-6))

    teams = set(df_sorted["HomeTeam"]).union(set(df_sorted["AwayTeam"]))
    strengths: Dict[str, Dict[str, float]] = {}

    for t in teams:
        h_df = df_sorted[df_sorted["HomeTeam"] == t]
        a_df = df_sorted[df_sorted["AwayTeam"] == t]

        if not h_df.empty:
            hw = h_df["weight"].sum()
            att_h = float((h_df["FTHG"] * h_df["weight"]).sum() / hw) / max(league_hg, 0.5)
            def_h = float((h_df["FTAG"] * h_df["weight"]).sum() / hw) / max(league_ag, 0.5)
        else:
            att_h, def_h = 1.0, 1.0

        if not a_df.empty:
            aw = a_df["weight"].sum()
            att_a = float((a_df["FTAG"] * a_df["weight"]).sum() / aw) / max(league_ag, 0.5)
            def_a = float((a_df["FTHG"] * a_df["weight"]).sum() / aw) / max(league_hg, 0.5)
        else:
            att_a, def_a = 1.0, 1.0

        strengths[t] = {
            "att_home": round(max(0.55, min(1.85, att_h)), 3),
            "def_home": round(max(0.55, min(1.85, def_h)), 3),
            "att_away": round(max(0.55, min(1.85, att_a)), 3),
            "def_away": round(max(0.55, min(1.85, def_a)), 3),
        }

    return strengths


def _get_cached_league_model(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula parámetros de liga, Elo histórico y fuerzas de ataque/defensa
    UNA SOLA VEZ por actualización del dataset en lugar de 20 veces.
    """
    sig = f"{len(df)}_{df['Date'].max() if not df.empty else 'empty'}"
    if _MODEL_BASE_CACHE.get("sig") == sig:
        return _MODEL_BASE_CACHE["data"]

    data = {
        "params": _learn_league_parameters(df),
        "elo_map": _compute_elo_ratings(df),
        "strengths": _compute_attack_defense_strengths(df),
    }
    _MODEL_BASE_CACHE["sig"] = sig
    _MODEL_BASE_CACHE["data"] = data
    return data


def _run_vectorized_dixon_coles_mc(
    xg_home: float,
    xg_away: float,
    rho: float,
    n_sims: int = 25000
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed=42)
    h_goals = rng.poisson(lam=xg_home, size=n_sims)
    a_goals = rng.poisson(lam=xg_away, size=n_sims)

    u = rng.random(size=n_sims)
    swap_mask = (h_goals == 0) & (a_goals == 0) & (u < abs(rho) * 0.65)
    h_goals = np.where(swap_mask & (xg_home >= xg_away), 1, h_goals)
    a_goals = np.where(swap_mask & (xg_home < xg_away), 1, a_goals)

    home_wins = int(np.sum(h_goals > a_goals))
    draws = int(np.sum(h_goals == a_goals))
    away_wins = int(np.sum(h_goals < a_goals))

    tot_goals = h_goals + a_goals
    over_1_5 = int(np.sum(tot_goals > 1.5))
    over_2_5 = int(np.sum(tot_goals > 2.5))
    under_2_5 = n_sims - over_2_5
    over_3_5 = int(np.sum(tot_goals > 3.5))

    btts_yes = int(np.sum((h_goals > 0) & (a_goals > 0)))
    btts_no = n_sims - btts_yes

    goleada_local = int(np.sum((h_goals - a_goals) >= 3))
    goleada_visita = int(np.sum((a_goals - h_goals) >= 3))

    pairs = h_goals * 100 + a_goals
    unique_codes, counts = np.unique(pairs, return_counts=True)
    order = np.argsort(-counts)[:8]
    top_scores = []
    for idx in order:
        code = int(unique_codes[idx])
        hg_i = code // 100
        ag_i = code % 100
        pct = round((int(counts[idx]) / n_sims) * 100.0, 2)
        top_scores.append({
            "marcador": f"{hg_i}-{ag_i}",
            "probabilidad_pct": pct
        })

    return {
        "iteraciones_monte_carlo": n_sims,
        "probabilidades_1X2_pct": {
            "local_1": round((home_wins / n_sims) * 100.0, 2),
            "empate_X": round((draws / n_sims) * 100.0, 2),
            "visita_2": round((away_wins / n_sims) * 100.0, 2),
        },
        "mercados_goles_pct": {
            "over_1_5": round((over_1_5 / n_sims) * 100.0, 2),
            "over_2_5": round((over_2_5 / n_sims) * 100.0, 2),
            "under_2_5": round((under_2_5 / n_sims) * 100.0, 2),
            "over_3_5": round((over_3_5 / n_sims) * 100.0, 2),
            "ambos_anotan_SI": round((btts_yes / n_sims) * 100.0, 2),
            "ambos_anotan_NO": round((btts_no / n_sims) * 100.0, 2),
            "goleada_local_3plus_pct": round((goleada_local / n_sims) * 100.0, 2),
            "goleada_visita_3plus_pct": round((goleada_visita / n_sims) * 100.0, 2),
        },
        "marcadores_exactos_mas_probables": top_scores
    }


def analyze_market_only(
    odds_raw: Dict[str, Any],
    historical_df: pd.DataFrame,
    home_team: str,
    away_team: str
) -> Dict[str, Any]:
    dec_1 = _american_to_decimal(odds_raw.get("home_moneyline"))
    dec_x = _american_to_decimal(odds_raw.get("draw_moneyline"))
    dec_2 = _american_to_decimal(odds_raw.get("away_moneyline"))

    h2h_closing: List[Dict[str, Any]] = []
    if not historical_df.empty:
        h2h = historical_df[
            ((historical_df["HomeTeam"] == home_team) & (historical_df["AwayTeam"] == away_team)) |
            ((historical_df["HomeTeam"] == away_team) & (historical_df["AwayTeam"] == home_team))
        ].sort_values("Date").tail(4)

        for _, r in h2h.iterrows():
            c1 = float(r.get("PSH", 0) or r.get("B365H", 0) or 0)
            cx = float(r.get("PSD", 0) or r.get("B365D", 0) or 0)
            c2 = float(r.get("PSA", 0) or r.get("B365A", 0) or 0)
            h2h_closing.append({
                "fecha": r["Date"].strftime("%Y-%m-%d"),
                "local": r["HomeTeam"],
                "visitante": r["AwayTeam"],
                "cierre_pinnacle_1X2": f"{c1} / {cx} / {c2}" if c1 > 0 else "N/D",
                "resultado_real": f"{int(r['FTHG'])}-{int(r['FTAG'])}"
            })

    if dec_1 > 1.0 and dec_x > 1.0 and dec_2 > 1.0:
        inv_sum = (1.0 / dec_1) + (1.0 / dec_x) + (1.0 / dec_2)
        overround = round((inv_sum - 1.0) * 100.0, 2)
        imp_1 = round(((1.0 / dec_1) / inv_sum) * 100.0, 2)
        imp_x = round(((1.0 / dec_x) / inv_sum) * 100.0, 2)
        imp_2 = round(((1.0 / dec_2) / inv_sum) * 100.0, 2)
        return {
            "proveedor": odds_raw.get("proveedor", "Mercado"),
            "linea_goles_mercado": odds_raw.get("over_under_linea", 2.5),
            "cuotas_decimales": {"local_1": dec_1, "empate_X": dec_x, "visita_2": dec_2},
            "probabilidades_implicitas_sin_vig_pct": {"local_1": imp_1, "empate_X": imp_x, "visita_2": imp_2},
            "margen_casa_overround_pct": overround,
            "estado_cuotas": "Cuotas 1X2 completas",
            "historico_cuotas_cierre_h2h": h2h_closing
        }

    return {
        "proveedor": odds_raw.get("proveedor", "Mercado"),
        "linea_goles_mercado": odds_raw.get("over_under_linea", 2.5),
        "cuotas_decimales": {
            "local_1": dec_1 if dec_1 > 0 else None,
            "empate_X": dec_x if dec_x > 0 else None,
            "visita_2": dec_2 if dec_2 > 0 else None
        },
        "estado_cuotas": "Cuotas 1X2 aún no publicadas por el proveedor para este partido",
        "historico_cuotas_cierre_h2h": h2h_closing
    }


def analyze_sports_and_context(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    windows: Dict[str, Any],
    injuries: Dict[str, List[str]],
    former_players: List[str]
) -> Dict[str, Any]:
    base = _get_cached_league_model(df)
    params = base["params"]
    elo_map = base["elo_map"]
    strengths = base["strengths"]

    elo_h = elo_map.get(home_team, 1500.0)
    elo_a = elo_map.get(away_team, 1500.0)
    elo_diff = round(elo_h - elo_a, 1)

    sh = strengths.get(home_team, {"att_home": 1.0, "def_home": 1.0, "att_away": 1.0, "def_away": 1.0})
    sa = strengths.get(away_team, {"att_home": 1.0, "def_home": 1.0, "att_away": 1.0, "def_away": 1.0})

    base_xg_h = params["avg_home_goals"] * sh["att_home"] * sa["def_away"]
    base_xg_a = params["avg_away_goals"] * sa["att_away"] * sh["def_home"]

    elo_factor = max(-0.22, min(0.22, (elo_diff + 45.0) / 1450.0))
    xg_h = base_xg_h * (1.0 + elo_factor)
    xg_a = base_xg_a * (1.0 - elo_factor * 0.85)

    lg10 = windows.get("local_general_10", {})
    vg10 = windows.get("visita_general_10", {})
    if lg10.get("partidos_analizados", 0) >= 4:
        form_h = (lg10.get("promedio_goles_anotados_10", 1.3) * 0.65 +
                  lg10.get("promedio_tiros_puerta_favor_10", 4.2) * params["shot_conversion"] * 0.35)
        xg_h = xg_h * 0.72 + form_h * 0.28

    if vg10.get("partidos_analizados", 0) >= 4:
        form_a = (vg10.get("promedio_goles_anotados_10", 1.1) * 0.65 +
                  vg10.get("promedio_tiros_puerta_favor_10", 3.8) * params["shot_conversion"] * 0.35)
        xg_a = xg_a * 0.72 + form_a * 0.28

    h2h = windows.get("h2h_ultimos_10", {})
    if h2h.get("partidos_encontrados", 0) >= 2:
        h2h_h = float(h2h.get("promedio_goles_local_en_h2h", xg_h))
        h2h_a = float(h2h.get("promedio_goles_visita_en_h2h", xg_a))
        xg_h = xg_h * 0.88 + h2h_h * 0.12
        xg_a = xg_a * 0.88 + h2h_a * 0.12

    inj_h_count = len(injuries.get("local", []))
    inj_a_count = len(injuries.get("visitante", []))
    xg_h *= max(0.84, 1.0 - inj_h_count * 0.025)
    xg_a *= max(0.84, 1.0 - inj_a_count * 0.025)

    xg_h = round(max(0.45, min(3.40, xg_h)), 3)
    xg_a = round(max(0.35, min(3.10, xg_a)), 3)

    mc_res = _run_vectorized_dixon_coles_mc(xg_h, xg_a, params["rho"], n_sims=25000)

    return {
        "parametros_aprendidos_liga": {
            "factor_localia_empirico": params["home_adv"],
            "correlacion_dixon_coles_rho": params["rho"],
            "tasa_conversion_tiros_puerta_premier": params["shot_conversion"]
        },
        "fuerzas_y_elo_dinamico": {
            "elo_local": elo_h,
            "elo_visitante": elo_a,
            "diferencial_elo": elo_diff,
            "fuerza_ataque_local": sh["att_home"],
            "fuerza_defensa_local": sh["def_home"],
            "fuerza_ataque_visitante": sa["att_away"],
            "fuerza_defensa_visitante": sa["def_away"],
            "xg_proyectado_local": xg_h,
            "xg_proyectado_visitante": xg_a
        },
        "simulacion_monte_carlo": mc_res,
        "estadisticas_historicas_y_forma": windows,
        "factores_extra_cancha": {
            "bajas_y_lesiones_reportadas": injuries,
            "jugadores_pasado_en_ambos_equipos": former_players
        }
    }
