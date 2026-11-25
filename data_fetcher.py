import pandas as pd
import numpy as np
import requests
from difflib import get_close_matches
from datetime import datetime
from typing import Dict, List, Any

# Mapeo oficial de los 20 equipos reales de la temporada actual 2026-2027 y 2025-2026
TEAM_NAME_MAP = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton & Hove Albion": "Brighton",
    "Brighton": "Brighton",
    "Chelsea": "Chelsea",
    "Coventry City": "Coventry",
    "Coventry City FC": "Coventry",
    "Coventry": "Coventry",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Hull City": "Hull",
    "Hull City AFC": "Hull",
    "Hull": "Hull",
    "Ipswich Town": "Ipswich",
    "Ipswich": "Ipswich",
    "Leeds United": "Leeds",
    "Leeds": "Leeds",
    "Liverpool": "Liverpool",
    "Manchester City": "Man City",
    "M. City": "Man City",
    "Man City": "Man City",
    "Manchester United": "Man United",
    "M. United": "Man United",
    "Man United": "Man United",
    "Newcastle United": "Newcastle",
    "Newcastle": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Nott'm Forest": "Nott'm Forest",
    "Sunderland": "Sunderland",
    "Sunderland AFC": "Sunderland",
    "Tottenham Hotspur": "Tottenham",
    "Tottenham": "Tottenham",
    "West Ham United": "West Ham",
    "West Ham": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Wolves": "Wolves",
    "Leicester City": "Leicester",
    "Leicester": "Leicester",
    "Burnley": "Burnley",
    "Southampton": "Southampton"
}

# Ciudades reales de los 20 clubes de la temporada actual
TEAM_CITY = {
    "Arsenal": "Londres", "Chelsea": "Londres", "Tottenham": "Londres",
    "Crystal Palace": "Londres", "Fulham": "Londres", "Brentford": "Londres",
    "West Ham": "Londres", "Man City": "Manchester", "Man United": "Manchester",
    "Liverpool": "Liverpool", "Everton": "Liverpool", "Aston Villa": "Birmingham",
    "Coventry": "Coventry (Midlands)", "Newcastle": "Tyne and Wear",
    "Sunderland": "Tyne and Wear", "Nott'm Forest": "Nottingham",
    "Brighton": "Brighton", "Bournemouth": "Bournemouth",
    "Ipswich": "Ipswich", "Leeds": "Yorkshire", "Hull": "Yorkshire"
}

# Clásicos y rivalidades históricas reales (incluyendo equipos ascendidos 2026-2027)
HISTORIC_RIVALRIES = {
    frozenset(["Arsenal", "Tottenham"]): "North London Derby (Clásico del Norte de Londres)",
    frozenset(["Arsenal", "Chelsea"]): "London Derby (Clásico de Londres)",
    frozenset(["Liverpool", "Man United"]): "North West Derby (Clásico de Inglaterra)",
    frozenset(["Liverpool", "Everton"]): "Merseyside Derby (Clásico de Liverpool)",
    frozenset(["Man City", "Man United"]): "Manchester Derby (Clásico de Manchester)",
    frozenset(["Newcastle", "Sunderland"]): "Tyne-Wear Derby (Clásico del Noreste)",
    frozenset(["Leeds", "Hull"]): "Yorkshire Derby",
    frozenset(["Leeds", "Man United"]): "Roses Rivalry (Clásico de las Rosas)",
    frozenset(["Aston Villa", "Coventry"]): "West Midlands Rivalry",
    frozenset(["Crystal Palace", "Brighton"]): "M23 Derby",
    frozenset(["Chelsea", "Fulham"]): "West London Derby",
    frozenset(["Brentford", "Fulham"]): "West London Derby"
}

def normalize_team(name: str, known_teams: List[str] = None) -> str:
    clean = name.strip()
    if clean in TEAM_NAME_MAP:
        mapped = TEAM_NAME_MAP[clean]
    else:
        mapped = clean

    # Si se pasan los equipos reales del CSV, verifica coincidencia automática
    if known_teams and mapped not in known_teams:
        matches = get_close_matches(mapped, known_teams, n=1, cutoff=0.6)
        if matches:
            return matches[0]
    return mapped

def classify_rivalry(home: str, away: str) -> Dict[str, Any]:
    pair = frozenset([home, away])
    same_city = (TEAM_CITY.get(home) is not None) and (TEAM_CITY.get(home) == TEAM_CITY.get(away))
    if pair in HISTORIC_RIVALRIES:
        return {
            "es_clasico_o_derbi": True,
            "nombre_rivalidad": HISTORIC_RIVALRIES[pair],
            "misma_ciudad": same_city,
            "ciudad": TEAM_CITY.get(home, "N/A")
        }
    if same_city:
        return {
            "es_clasico_o_derbi": True,
            "nombre_rivalidad": f"Derbi de {TEAM_CITY[home]}",
            "misma_ciudad": True,
            "ciudad": TEAM_CITY[home]
        }
    return {
        "es_clasico_o_derbi": False,
        "nombre_rivalidad": "Partido regular de liga",
        "misma_ciudad": False,
        "ciudad": f"{TEAM_CITY.get(home, 'Local')} vs {TEAM_CITY.get(away, 'Visita')}"
    }

def load_historical_data() -> pd.DataFrame:
    """
    Descarga los datos reales de Premier League 2025-2026 y 2026-2027,
    más Championship 2025-2026 (E1) para tener el historial reciente real
    de los equipos recién ascendidos (Hull City, Coventry City, Ipswich, etc.).
    """
    urls = [
        ("2025-2026-PL", "https://www.football-data.co.uk/mmz4281/2526/E0.csv"),
        ("2025-2026-CH", "https://www.football-data.co.uk/mmz4281/2526/E1.csv"),
        ("2026-2027-PL", "https://www.football-data.co.uk/mmz4281/2627/E0.csv"),
    ]
    frames = []
    for season_label, url in urls:
        try:
            df = pd.read_csv(url)
            df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
            df["Season"] = season_label
            df["DateParsed"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            df = df.dropna(subset=["DateParsed"])
            frames.append(df)
        except Exception as e:
            print(f"Aviso cargando {season_label}: {e}")

    if not frames:
        raise RuntimeError("No se pudieron descargar los datos históricos.")

    full_df = pd.concat(frames, ignore_index=True).sort_values("DateParsed").reset_index(drop=True)

    # Filtrar para conservar únicamente los equipos que participan en la Premier League 2026-2027
    current_pl = full_df[full_df["Season"] == "2026-2027-PL"]
    if not current_pl.empty:
        active_teams = set(current_pl["HomeTeam"]).union(set(current_pl["AwayTeam"]))
        full_df = full_df[
            full_df["HomeTeam"].isin(active_teams) | full_df["AwayTeam"].isin(active_teams)
        ].reset_index(drop=True)

    numeric_cols = ["FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC", "B365H", "B365D", "B365A", "PSH", "PSD", "PSA"]
    for col in numeric_cols:
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(full_df[col], errors="coerce")

    full_df["IsDerby"] = full_df.apply(
        lambda r: classify_rivalry(r["HomeTeam"], r["AwayTeam"])["es_clasico_o_derbi"], axis=1
    )
    return full_df

def extract_team_windows(df: pd.DataFrame, home_team: str, away_team: str, reference_date: datetime) -> Dict[str, Any]:
    past_df = df[df["DateParsed"] < pd.to_datetime(reference_date.date())].copy()
    season_df = past_df[past_df["Season"] == "2026-2027-PL"]

    def get_full_season_venue_avg(team: str) -> Dict[str, Any]:
        h_games = season_df[season_df["HomeTeam"] == team]
        a_games = season_df[season_df["AwayTeam"] == team]
        return {
            "temporada": "2026-2027",
            "partidos_local_temporada": len(h_games),
            "promedio_goles_anotados_local_temporada": round(float(h_games["FTHG"].mean()), 3) if not h_games.empty else 0.0,
            "promedio_goles_recibidos_local_temporada": round(float(h_games["FTAG"].mean()), 3) if not h_games.empty else 0.0,
            "partidos_visita_temporada": len(a_games),
            "promedio_goles_anotados_visita_temporada": round(float(a_games["FTAG"].mean()), 3) if not a_games.empty else 0.0,
            "promedio_goles_recibidos_visita_temporada": round(float(a_games["FTHG"].mean()), 3) if not a_games.empty else 0.0,
        }

    def get_general_last_10(team: str) -> Dict[str, Any]:
        team_games = past_df[(past_df["HomeTeam"] == team) | (past_df["AwayTeam"] == team)].tail(10)
        if team_games.empty:
            return {}
        gf, ga, sot_for, sot_ag, pts, seq = [], [], [], [], [], []
        for _, row in team_games.iterrows():
            is_home = row["HomeTeam"] == team
            g_s = row["FTHG"] if is_home else row["FTAG"]
            g_c = row["FTAG"] if is_home else row["FTHG"]
            s_f = row["HST"] if is_home else row["AST"]
            s_a = row["AST"] if is_home else row["HST"]
            gf.append(g_s); ga.append(g_c)
            if pd.notna(s_f): sot_for.append(s_f)
            if pd.notna(s_a): sot_ag.append(s_a)
            if g_s > g_c: pts.append(3); seq.append("W")
            elif g_s == g_c: pts.append(1); seq.append("D")
            else: pts.append(0); seq.append("L")

        rest_days = int((pd.to_datetime(reference_date.date()) - team_games["DateParsed"].max()).days)
        return {
            "partidos_analizados": len(team_games),
            "racha_ultimos_10": "".join(seq),
            "puntos_ultimos_10": int(sum(pts)),
            "promedio_goles_anotados_10": round(float(np.mean(gf)), 3),
            "promedio_goles_recibidos_10": round(float(np.mean(ga)), 3),
            "promedio_tiros_puerta_favor_10": round(float(np.mean(sot_for)), 3) if sot_for else 0.0,
            "promedio_tiros_puerta_contra_10": round(float(np.mean(sot_ag)), 3) if sot_ag else 0.0,
            "dias_descanso_reales": rest_days
        }

    def get_venue_last_10(team: str, venue: str) -> Dict[str, Any]:
        if venue == "home":
            games = past_df[past_df["HomeTeam"] == team].tail(10)
            gf, ga, sot_f, sot_a = games["FTHG"].dropna(), games["FTAG"].dropna(), games["HST"].dropna(), games["AST"].dropna()
        else:
            games = past_df[past_df["AwayTeam"] == team].tail(10)
            gf, ga, sot_f, sot_a = games["FTAG"].dropna(), games["FTHG"].dropna(), games["AST"].dropna(), games["HST"].dropna()
        if games.empty:
            return {}
        return {
            "condicion": "Local" if venue == "home" else "Visitante",
            "partidos_analizados": len(games),
            "record_ultimos_10": f"{int((gf > ga).sum())}G-{int((gf == ga).sum())}E-{int((gf < ga).sum())}P",
            "promedio_goles_anotados": round(float(gf.mean()), 3),
            "promedio_goles_recibidos": round(float(ga.mean()), 3),
            "promedio_tiros_puerta_favor": round(float(sot_f.mean()), 3) if not sot_f.empty else 0.0,
            "promedio_tiros_puerta_contra": round(float(sot_a.mean()), 3) if not sot_a.empty else 0.0
        }

    def get_h2h_last_10(home: str, away: str) -> Dict[str, Any]:
        h2h = past_df[
            ((past_df["HomeTeam"] == home) & (past_df["AwayTeam"] == away)) |
            ((past_df["HomeTeam"] == away) & (past_df["AwayTeam"] == home))
        ].tail(10)
        if h2h.empty:
            return {"partidos_encontrados": 0}
        hw, aw, dr, hg_list, ag_list, duels = 0, 0, 0, [], [], []
        for _, r in h2h.iterrows():
            hg, ag = (int(r["FTHG"]), int(r["FTAG"])) if r["HomeTeam"] == home else (int(r["FTAG"]), int(r["FTHG"]))
            hg_list.append(hg); ag_list.append(ag)
            if hg > ag: hw += 1
            elif hg < ag: aw += 1
            else: dr += 1
            duels.append({
                "fecha": r["DateParsed"].strftime("%Y-%m-%d"),
                "local": r["HomeTeam"],
                "visitante": r["AwayTeam"],
                "marcador": f"{int(r['FTHG'])}-{int(r['FTAG'])}"
            })
        return {
            "partidos_encontrados": len(h2h),
            "victorias_local_actual": hw,
            "empates": dr,
            "victorias_visita_actual": aw,
            "promedio_goles_local_en_h2h": round(float(np.mean(hg_list)), 3),
            "promedio_goles_visita_en_h2h": round(float(np.mean(ag_list)), 3),
            "ultimos_duelos": duels
        }

    pl_only = past_df[past_df["Season"].str.contains("PL")]
    derby_games = pl_only[pl_only["IsDerby"] == True]
    regular_games = pl_only[pl_only["IsDerby"] == False]
    rivalry_info = classify_rivalry(home_team, away_team)
    rivalry_info["impacto_empirico_derbis_liga"] = {
        "promedio_goles_derbis": round(float((derby_games["FTHG"] + derby_games["FTAG"]).mean()), 3) if not derby_games.empty else 0.0,
        "promedio_goles_partidos_normales": round(float((regular_games["FTHG"] + regular_games["FTAG"]).mean()), 3) if not regular_games.empty else 0.0,
        "tasa_empates_derbis": round(float((derby_games["FTHG"] == derby_games["FTAG"]).mean()), 3) if not derby_games.empty else 0.0,
        "tasa_empates_normales": round(float((regular_games["FTHG"] == regular_games["FTAG"]).mean()), 3) if not regular_games.empty else 0.0
    }

    return {
        "promedios_goles_temporada_completa": {
            "local": get_full_season_venue_avg(home_team),
            "visitante": get_full_season_venue_avg(away_team)
        },
        "local_general_10": get_general_last_10(home_team),
        "visita_general_10": get_general_last_10(away_team),
        "local_en_casa_10": get_venue_last_10(home_team, "home"),
        "visita_fuera_10": get_venue_last_10(away_team, "away"),
        "h2h_ultimos_10": get_h2h_last_10(home_team, away_team),
        "rivalidad_y_clasico": rivalry_info
    }

def detect_former_players(home_team_id: str, away_team_id: str, home_name: str, away_name: str, known_teams: List[str]) -> List[Dict[str, str]]:
    former_players = []
    teams_to_check = [
        (home_team_id, home_name, away_name),
        (away_team_id, away_name, home_name)
    ]
    for tid, current_club, rival_club in teams_to_check:
        try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/teams/{tid}/roster"
            resp = requests.get(url, timeout=8)
            if resp.status_code != 200:
                continue
            for ath in resp.json().get("athletes", []):
                for prev in ath.get("previousTeams", []):
                    prev_norm = normalize_team(prev.get("displayName", ""), known_teams)
                    if prev_norm == rival_club:
                        former_players.append({
                            "jugador": ath.get("displayName", "Desconocido"),
                            "equipo_actual": current_club,
                            "ex_equipo": rival_club,
                            "posicion": ath.get("position", {}).get("abbreviation", "N/A")
                        })
        except Exception:
            continue
    return former_players

def fetch_live_espn_slate(known_teams: List[str] = None) -> List[Dict[str, Any]]:
    url = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    matches = []
    for event in data.get("events", []):
        comp = event["competitions"][0]
        competitors = comp["competitors"]
        home_comp = next(c for c in competitors if c["homeAway"] == "home")
        away_comp = next(c for c in competitors if c["homeAway"] == "away")

        home_raw = home_comp["team"]["displayName"]
        away_raw = away_comp["team"]["displayName"]
        home_norm = normalize_team(home_raw, known_teams)
        away_norm = normalize_team(away_raw, known_teams)

        odds_data = {}
        if comp.get("odds"):
            p_odds = comp["odds"][0]
            odds_data = {
                "proveedor": p_odds.get("provider", {}).get("name", "Mercado"),
                "over_under_linea": p_odds.get("overUnder"),
                "detalles": p_odds.get("details"),
                "home_moneyline": p_odds.get("homeTeamOdds", {}).get("moneyLine"),
                "away_moneyline": p_odds.get("awayTeamOdds", {}).get("moneyLine"),
                "draw_moneyline": p_odds.get("drawOdds", {}).get("moneyLine")
            }

        injuries = {"local": [], "visitante": []}
        try:
            s_url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/summary?event={event['id']}"
            s_resp = requests.get(s_url, timeout=8)
            if s_resp.status_code == 200:
                for team_inj in s_resp.json().get("injuries", []):
                    t_name = normalize_team(team_inj.get("team", {}).get("displayName", ""), known_teams)
                    players = [
                        f"{a.get('athlete', {}).get('displayName')} ({a.get('status')})"
                        for a in team_inj.get("injuries", [])
                    ]
                    if t_name == home_norm:
                        injuries["local"] = players
                    elif t_name == away_norm:
                        injuries["visitante"] = players
        except Exception:
            pass

        ex_players = detect_former_players(
            home_comp["team"]["id"], away_comp["team"]["id"], home_norm, away_norm, known_teams or []
        )

        matches.append({
            "event_id": event["id"],
            "fecha_utc": event["date"],
            "estado": event["status"]["type"]["description"],
            "local_espn": home_raw,
            "visitante_espn": away_raw,
            "local_modelo": home_norm,
            "visitante_modelo": away_norm,
            "record_local_temporada": home_comp.get("records", [{}])[0].get("summary", "N/A"),
            "record_visita_temporada": away_comp.get("records", [{}])[0].get("summary", "N/A"),
            "cuotas_mercado_raw": odds_data,
            "bajas_reportadas": injuries,
            "jugadores_en_ambos_equipos": ex_players
        })

    return matches
