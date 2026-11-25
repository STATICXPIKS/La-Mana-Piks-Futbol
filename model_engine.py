import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# =====================================================================
# BLOQUE 1: MOTOR EXCLUSIVO DE MERCADO Y CUOTAS (SIN SESGO DEPORTIVO)
# =====================================================================

def american_to_decimal(odds: Any) -> float:
    if odds is None:
        return 0.0
    try:
        val = float(odds)
        if val == 0:
            return 0.0
        if 1.01 <= val <= 50.0:
            return val
        if val > 0:
            return (val / 100.0) + 1.0
        else:
            return (100.0 / abs(val)) + 1.0
    except Exception:
        return 0.0

def analyze_market_only(odds_raw: Dict[str, Any], historical_df: pd.DataFrame, home_team: str, away_team: str) -> Dict[str, Any]:
    dec_home = american_to_decimal(odds_raw.get("home_moneyline"))
    dec_draw = american_to_decimal(odds_raw.get("draw_moneyline"))
    dec_away = american_to_decimal(odds_raw.get("away_moneyline"))

    market_block = {
        "proveedor": odds_raw.get("proveedor", "No disponible"),
        "linea_goles_mercado": odds_raw.get("over_under_linea"),
        "cuotas_decimales": {
            "local_1": round(dec_home, 3) if dec_home > 0 else None,
            "empate_X": round(dec_draw, 3) if dec_draw > 0 else None,
            "visita_2": round(dec_away, 3) if dec_away > 0 else None
        }
    }

    if dec_home > 1.0 and dec_draw > 1.0 and dec_away > 1.0:
        imp_h = 1.0 / dec_home
        imp_d = 1.0 / dec_draw
        imp_a = 1.0 / dec_away
        total_implied = imp_h + imp_d + imp_a
        vig = total_implied - 1.0

        market_block["margen_casa_vig_pct"] = round(vig * 100.0, 2)
        market_block["probabilidad_implicita_justa_pct"] = {
            "local_1": round((imp_h / total_implied) * 100.0, 2),
            "empate_X": round((imp_d / total_implied) * 100.0, 2),
            "visita_2": round((imp_a / total_implied) * 100.0, 2)
        }
    else:
        market_block["estado_cuotas"] = "Cuotas 1X2 aún no publicadas por el proveedor para este partido"

    h2h = historical_df[
        ((historical_df["HomeTeam"] == home_team) & (historical_df["AwayTeam"] == away_team)) |
        ((historical_df["HomeTeam"] == away_team) & (historical_df["AwayTeam"] == home_team))
    ].tail(5)

    past_odds = []
    for _, r in h2h.iterrows():
        if pd.notna(r.get("PSH")) and pd.notna(r.get("PSD")) and pd.notna(r.get("PSA")):
            past_odds.append({
                "fecha": r["DateParsed"].strftime("%Y-%m-%d"),
                "local": r["HomeTeam"],
                "visita": r["AwayTeam"],
                "cierre_pinnacle_1X2": f"{r['PSH']} / {r['PSD']} / {r['PSA']}",
                "resultado_real": f"{int(r['FTHG'])}-{int(r['FTAG'])}"
            })
    market_block["historico_cuotas_cierre_h2h"] = past_odds

    return market_block


# =====================================================================
# BLOQUE 2: MOTOR DEPORTIVO Y EXTRA-CANCHA (100% DATOS REALES)
# =====================================================================

def compute_dynamic_elo(df: pd.DataFrame) -> Dict[str, float]:
    teams = sorted(list(set(df["HomeTeam"]).union(set(df["AwayTeam"]))))
    elo = {t: 1500.0 for t in teams}

    home_win_rate = float((df["FTHG"] > df["FTAG"]).mean())
    away_win_rate = float((df["FTHG"] < df["FTAG"]).mean())
    home_adv_elo = 400.0 * np.log10(max(home_win_rate, 0.01) / max(away_win_rate, 0.01))

    for _, row in df.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        hg, ag = row["FTHG"], row["FTAG"]

        exp_h = 1.0 / (1.0 + 10.0 ** ((elo[a] - (elo[h] + home_adv_elo)) / 400.0))
        actual_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)

        goal_diff = abs(hg - ag)
        mov_mult = np.log(goal_diff + 1.0) + 1.0
        k_dynamic = 20.0 * mov_mult

        delta = k_dynamic * (actual_h - exp_h)
        elo[h] += delta
        elo[a] -= delta

    return elo

def fit_dixon_coles_real(df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float], float, float]:
    """
    Estima fuerzas iterativas de ataque y defensa ponderadas por decaimiento temporal
    (algoritmo iterativo de máxima verosimilitud Poisson-xG) + parámetro rho empírico
    de correlación en marcadores bajos (0-0, 1-0, 0-1, 1-1).
    """
    total_goals = float(df["FTHG"].sum() + df["FTAG"].sum())
    total_sot = float(df["HST"].fillna(0).sum() + df["AST"].fillna(0).sum())
    league_conversion = (total_goals / total_sot) if total_sot > 0 else 0.30

    df = df.copy()
    df["Home_xG"] = np.where(df["HST"].notna(), 0.5 * df["FTHG"] + 0.5 * (df["HST"] * league_conversion), df["FTHG"])
    df["Away_xG"] = np.where(df["AST"].notna(), 0.5 * df["FTAG"] + 0.5 * (df["AST"] * league_conversion), df["FTAG"])

    max_date = df["DateParsed"].max()
    days_ago = (max_date - df["DateParsed"]).dt.days.values
    half_life = float(np.median(days_ago)) if len(days_ago) > 0 and np.median(days_ago) > 0 else 180.0
    df["w"] = np.exp(-np.log(2) * days_ago / half_life)

    teams = sorted(list(set(df["HomeTeam"]).union(set(df["AwayTeam"]))))
    w_sum = df["w"].sum()
    avg_home_xg = float((df["Home_xG"] * df["w"]).sum() / w_sum)
    avg_away_xg = float((df["Away_xG"] * df["w"]).sum() / w_sum)
    home_advantage = avg_home_xg / max(avg_away_xg, 0.1)

    att = {t: 1.0 for t in teams}
    defn = {t: 1.0 for t in teams}

    # Ajuste iterativo MLE exacto de ataque y defensa cruzados (30 iteraciones rápidas)
    for _ in range(30):
        new_att = {}
        new_defn = {}
        for t in teams:
            h_m = df[df["HomeTeam"] == t]
            a_m = df[df["AwayTeam"] == t]

            scored_w = (h_m["Home_xG"] * h_m["w"]).sum() + (a_m["Away_xG"] * a_m["w"]).sum()
            exp_scored_w = (
                (h_m["w"] * h_m["AwayTeam"].map(defn) * avg_home_xg).sum() +
                (a_m["w"] * a_m["HomeTeam"].map(defn) * avg_away_xg).sum()
            )
            new_att[t] = float(scored_w / max(exp_scored_w, 0.1))

            conceded_w = (h_m["Away_xG"] * h_m["w"]).sum() + (a_m["Home_xG"] * a_m["w"]).sum()
            exp_conceded_w = (
                (h_m["w"] * h_m["AwayTeam"].map(att) * avg_away_xg).sum() +
                (a_m["w"] * a_m["HomeTeam"].map(att) * avg_home_xg).sum()
            )
            new_defn[t] = float(conceded_w / max(exp_conceded_w, 0.1))

        mean_att = np.mean(list(new_att.values()))
        att = {t: new_att[t] / mean_att for t in teams}
        defn = new_defn

    # Calibración empírica de rho (Dixon-Coles) comparando tasa real de 0-0 vs Poisson independiente
    obs_00 = float(((df["FTHG"] == 0) & (df["FTAG"] == 0)).mean())
    exp_00 = float(np.mean(np.exp(-(avg_home_xg + avg_away_xg))))
    rho_dc = float(np.clip((exp_00 - obs_00) / max(avg_home_xg * avg_away_xg * exp_00, 0.05), -0.25, 0.15))

    return att, defn, home_advantage, rho_dc

def poisson_pmf_array(k_max: int, lam: float) -> np.ndarray:
    k = np.arange(k_max)
    factorials = np.array([math.factorial(int(i)) for i in k], dtype=float)
    return (np.exp(-lam) * (lam ** k)) / factorials

def run_monte_carlo_dixon_coles(lambda_home: float, lambda_away: float, rho: float, n_sims: int = 25000) -> Dict[str, Any]:
    max_goals = 10
    p_h = poisson_pmf_array(max_goals, lambda_home)
    p_a = poisson_pmf_array(max_goals, lambda_away)
    probs_matrix = np.outer(p_h, p_a)

    probs_matrix[0, 0] *= max(1.0 - (lambda_home * lambda_away * rho), 0.01)
    probs_matrix[0, 1] *= max(1.0 + (lambda_home * rho), 0.01)
    probs_matrix[1, 0] *= max(1.0 + (lambda_away * rho), 0.01)
    probs_matrix[1, 1] *= max(1.0 - rho, 0.01)
    probs_matrix /= probs_matrix.sum()

    rng = np.random.default_rng()
    sampled_indices = rng.choice(max_goals * max_goals, size=n_sims, p=probs_matrix.ravel())
    sim_home_goals = sampled_indices // max_goals
    sim_away_goals = sampled_indices % max_goals

    home_wins = float(np.mean(sim_home_goals > sim_away_goals))
    draws = float(np.mean(sim_home_goals == sim_away_goals))
    away_wins = float(np.mean(sim_home_goals < sim_away_goals))

    over_1_5 = float(np.mean((sim_home_goals + sim_away_goals) > 1.5))
    over_2_5 = float(np.mean((sim_home_goals + sim_away_goals) > 2.5))
    over_3_5 = float(np.mean((sim_home_goals + sim_away_goals) > 3.5))
    btts_yes = float(np.mean((sim_home_goals > 0) & (sim_away_goals > 0)))

    scores, counts = np.unique(list(zip(sim_home_goals, sim_away_goals)), axis=0, return_counts=True)
    top_idx = np.argsort(-counts)[:5]
    top_scores = [
        {"marcador": f"{int(scores[i][0])}-{int(scores[i][1])}", "probabilidad_pct": round(float(counts[i] / n_sims) * 100.0, 2)}
        for i in top_idx
    ]

    return {
        "iteraciones_monte_carlo": n_sims,
        "probabilidades_1X2_pct": {
            "local_1": round(home_wins * 100.0, 2),
            "empate_X": round(draws * 100.0, 2),
            "visita_2": round(away_wins * 100.0, 2)
        },
        "mercados_goles_pct": {
            "over_1_5": round(over_1_5 * 100.0, 2),
            "over_2_5": round(over_2_5 * 100.0, 2),
            "under_2_5": round((1.0 - over_2_5) * 100.0, 2),
            "over_3_5": round(over_3_5 * 100.0, 2),
            "ambos_anotan_SI": round(btts_yes * 100.0, 2),
            "ambos_anotan_NO": round((1.0 - btts_yes) * 100.0, 2)
        },
        "marcadores_exactos_mas_probables": top_scores
    }

def analyze_sports_and_context(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    windows: Dict[str, Any],
    injuries: Dict[str, Any],
    former_players: list
) -> Dict[str, Any]:
    elo_ratings = compute_dynamic_elo(df)
    attack_str, defense_str, home_adv_factor, rho_dc = fit_dixon_coles_real(df)

    league_conv = float((df["FTHG"].sum() + df["FTAG"].sum()) / max(df["HST"].fillna(0).sum() + df["AST"].fillna(0).sum(), 1.0))

    att_h = attack_str.get(home_team, 1.0)
    def_h = defense_str.get(home_team, 1.0)
    att_a = attack_str.get(away_team, 1.0)
    def_a = defense_str.get(away_team, 1.0)

    base_xg_home = att_h * def_a * home_adv_factor
    base_xg_away = att_a * def_h

    h_venue = windows.get("local_en_casa_10", {})
    a_venue = windows.get("visita_fuera_10", {})
    if h_venue and a_venue:
        recent_sot_xg_h = h_venue.get("promedio_tiros_puerta_favor", 0.0) * league_conv
        recent_sot_xg_a = a_venue.get("promedio_tiros_puerta_favor", 0.0) * league_conv
        if recent_sot_xg_h > 0:
            base_xg_home = 0.70 * base_xg_home + 0.30 * recent_sot_xg_h
        if recent_sot_xg_a > 0:
            base_xg_away = 0.70 * base_xg_away + 0.30 * recent_sot_xg_a

    rivalry = windows.get("rivalidad_y_clasico", {})
    if rivalry.get("es_clasico_o_derbi"):
        emp = rivalry.get("impacto_empirico_derbis_liga", {})
        g_derby = emp.get("promedio_goles_derbis", 0.0)
        g_norm = emp.get("promedio_goles_partidos_normales", 0.0)
        if g_derby > 0 and g_norm > 0:
            derby_ratio = g_derby / g_norm
            base_xg_home *= derby_ratio
            base_xg_away *= derby_ratio

    sim_results = run_monte_carlo_dixon_coles(base_xg_home, base_xg_away, rho_dc, n_sims=25000)

    return {
        "parametros_aprendidos_liga": {
            "factor_localia_empirico": round(home_adv_factor, 4),
            "correlacion_dixon_coles_rho": round(rho_dc, 4),
            "tasa_conversion_tiros_puerta_premier": round(league_conv, 4)
        },
        "fuerzas_y_elo_dinamico": {
            "elo_local": round(elo_ratings.get(home_team, 1500.0), 1),
            "elo_visitante": round(elo_ratings.get(away_team, 1500.0), 1),
            "diferencial_elo": round(elo_ratings.get(home_team, 1500.0) - elo_ratings.get(away_team, 1500.0), 1),
            "fuerza_ataque_local": round(att_h, 3),
            "fuerza_defensa_local": round(def_h, 3),
            "fuerza_ataque_visitante": round(att_a, 3),
            "fuerza_defensa_visitante": round(def_a, 3),
            "xg_proyectado_local": round(base_xg_home, 3),
            "xg_proyectado_visitante": round(base_xg_away, 3)
        },
        "simulacion_monte_carlo": sim_results,
        "estadisticas_historicas_y_forma": windows,
        "factores_extra_cancha": {
            "rivalidad": rivalry,
            "bajas_y_lesiones_reportadas": injuries,
            "jugadores_pasado_en_ambos_equipos": former_players
        }
    }
