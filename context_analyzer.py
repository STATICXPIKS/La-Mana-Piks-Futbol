import requests
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from typing import Dict, Any, List
from data_fetcher import normalize_team

def fetch_live_standings_pressure(home_team: str, away_team: str, known_teams: List[str]) -> Dict[str, Any]:
    """
    Descarga la tabla real de posiciones de la Premier League desde ESPN
    y calcula la presión competitiva real (distancia a Champions, Europa y Descenso).
    """
    url = "https://site.api.espn.com/apis/v2/sports/soccer/eng.1/standings"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        entries = data["children"][0]["standings"]["entries"]

        table = []
        for idx, entry in enumerate(entries, start=1):
            t_name = normalize_team(entry["team"]["displayName"], known_teams)
            stats = {}
            for s in entry.get("stats", []):
                val = s.get("value") if s.get("value") is not None else s.get("displayValue", 0)
                try:
                    stats[s["name"]] = int(float(val))
                except Exception:
                    stats[s["name"]] = 0

            table.append({
                "equipo": t_name,
                "posicion": stats.get("rank") or idx,
                "puntos": stats.get("points", 0),
                "jugados": stats.get("gamesPlayed", 0),
                "dif_goles": stats.get("pointDifferential", 0)
            })

        table.sort(key=lambda x: (x["posicion"], -x["puntos"], -x["dif_goles"]))
        for i, row in enumerate(table, start=1):
            row["posicion"] = i

        if len(table) < 18:
            return {"estado": "Tabla incompleta"}

        pts_1st = table[0]["puntos"]
        pts_4th = table[3]["puntos"]
        pts_6th = table[5]["puntos"]
        pts_17th = table[16]["puntos"]
        pts_18th = table[17]["puntos"]

        def evaluate_team_pressure(team: str) -> Dict[str, Any]:
            row = next((r for r in table if r["equipo"] == team), None)
            if not row:
                return {"posicion_actual": "-", "puntos": 0, "contexto_clasificacion": "Sin datos en tabla"}
            pos = row["posicion"]
            pts = row["puntos"]

            if pos == 1:
                objetivo = f"Líder (+{pts - table[1]['puntos']} pts sobre el 2°)"
            elif pos <= 4:
                objetivo = f"Zona Champions (a {pts_1st - pts} pts del líder, +{pts - table[4]['puntos']} pts sobre el 5°)"
            elif pos <= 7:
                objetivo = f"Pelea Europa (a {pts_4th - pts} pts de Champions)"
            elif pos >= 18:
                objetivo = f"Zona de DESCENSO (a {pts_17th - pts} pts de salvación)"
            elif pos >= 15:
                objetivo = f"Alerta descenso (solo +{pts - pts_18th} pts sobre el puesto 18)"
            else:
                objetivo = f"Mitad de tabla (a {pts_6th - pts} pts de Europa / +{pts - pts_18th} pts sobre descenso)"

            return {
                "posicion_actual": pos,
                "puntos": pts,
                "diferencia_goles": row["dif_goles"],
                "contexto_clasificacion": objetivo
            }

        return {
            "local": evaluate_team_pressure(home_team),
            "visitante": evaluate_team_pressure(away_team)
        }
    except Exception as e:
        return {"error": f"No se pudo consultar tabla en vivo: {e}"}

def analyze_tactical_styles_and_neutralization(df: pd.DataFrame, home_team: str, away_team: str) -> Dict[str, Any]:
    pl_df = df[df["Season"].str.contains("PL")].copy()
    if pl_df.empty:
        pl_df = df.copy()

    avg_shots = float((pl_df["HS"].mean() + pl_df["AS"].mean()) / 2.0)
    avg_corners = float((pl_df["HC"].mean() + pl_df["AC"].mean()) / 2.0)
    avg_goals = float((pl_df["FTHG"].mean() + pl_df["FTAG"].mean()) / 2.0)

    def get_team_tactical_profile(team: str) -> Dict[str, Any]:
        h_g = pl_df[pl_df["HomeTeam"] == team]
        a_g = pl_df[pl_df["AwayTeam"] == team]

        shots_for = pd.concat([h_g["HS"], a_g["AS"]]).dropna().mean()
        shots_ag = pd.concat([h_g["AS"], a_g["HS"]]).dropna().mean()
        corners_for = pd.concat([h_g["HC"], a_g["AC"]]).dropna().mean()
        goals_for = pd.concat([h_g["FTHG"], a_g["FTAG"]]).dropna().mean()

        territorial_dominance = ((shots_for / avg_shots) + (corners_for / avg_corners)) / 2.0
        defensive_containment = avg_shots / max(shots_ag, 1.0)
        clinical_counter_efficiency = (goals_for / max(shots_for, 1.0)) / (avg_goals / max(avg_shots, 1.0))

        if territorial_dominance >= 1.10 and defensive_containment >= 1.05:
            estilo = "Posesión dominante y presión alta (alto volumen ofensivo y asfixia al rival)"
        elif territorial_dominance >= 1.05 and defensive_containment < 1.00:
            estilo = "Ofensivo vertical / Ida y vuelta (genera mucho pero concede espacios atrás)"
        elif territorial_dominance < 0.95 and clinical_counter_efficiency >= 1.05:
            estilo = "Bloque bajo y contragolpe letal (cede iniciativa pero castiga con alta eficacia)"
        elif defensive_containment >= 1.05 and territorial_dominance < 1.00:
            estilo = "Bloque medio/bajo reactivo y ordenado (prioriza neutralizar y cerrar espacios)"
        else:
            estilo = "Equilibrado / Transiciones mixtas"

        return {
            "estilo_detectado": estilo,
            "indice_dominio_territorial": round(float(territorial_dominance), 3),
            "indice_contencion_defensiva": round(float(defensive_containment), 3),
            "indice_eficacia_contragolpe": round(float(clinical_counter_efficiency), 3),
            "tiros_totales_favor_prom": round(float(shots_for), 2),
            "tiros_totales_contra_prom": round(float(shots_ag), 2),
            "corners_favor_prom": round(float(corners_for), 2)
        }

    prof_h = get_team_tactical_profile(home_team)
    prof_a = get_team_tactical_profile(away_team)

    h_attack_vs_a_block = prof_h["indice_dominio_territorial"] / max(prof_a["indice_contencion_defensiva"], 0.5)
    a_attack_vs_h_block = prof_a["indice_dominio_territorial"] / max(prof_h["indice_contencion_defensiva"], 0.5)
    neutralization_index = 1.0 - abs(h_attack_vs_a_block - a_attack_vs_h_block)

    if prof_h["indice_dominio_territorial"] > 1.08 and prof_a["indice_contencion_defensiva"] > 1.08:
        dinamica = f"Choque de estilos opuestos: {home_team} monopolizará el balón frente al bloque cerrado de {away_team}. Alto riesgo de partido trabado si no cae gol temprano."
    elif prof_h["indice_dominio_territorial"] > 1.05 and prof_a["indice_eficacia_contragolpe"] > 1.08:
        dinamica = f"Alerta de contragolpe: {home_team} adelantará líneas y {away_team} tiene alta eficacia para castigar espacios a la espalda."
    elif prof_h["indice_contencion_defensiva"] > 1.05 and prof_a["indice_contencion_defensiva"] > 1.05:
        dinamica = "Neutralización mutua alta: ambos equipos priorizan el orden defensivo y reducen el volumen de tiros del rival."
    elif h_attack_vs_a_block > 1.05 and a_attack_vs_h_block > 1.05:
        dinamica = "Baja neutralización (partido abierto): ambos ataques superan la capacidad de contención rival; favorece escenarios de intercambio de golpes."
    else:
        dinamica = "Equilibrio táctico estándar según sus promedios de volumen y contención."

    return {
        "perfil_tactico_local": prof_h,
        "perfil_tactico_visitante": prof_a,
        "indice_neutralizacion_mutua": round(float(np.clip(neutralization_index, 0.0, 1.0)), 3),
        "lectura_choque_estilos": dinamica
    }

def fetch_extra_pitch_news(home_espn: str, away_espn: str, home_short: str, away_short: str) -> Dict[str, Any]:
    rss_urls = [
        ("BBC Sport", "https://feeds.bbci.co.uk/sport/football/rss.xml"),
        ("Sky Sports", "https://www.skysports.com/rss/11095")
    ]

    keywords_home = {home_espn.lower(), home_short.lower()}
    keywords_away = {away_espn.lower(), away_short.lower()}
    if "man united" in keywords_away or "manchester united" in keywords_away:
        keywords_away.update(["man utd", "pochettino"])
    if "man united" in keywords_home or "manchester united" in keywords_home:
        keywords_home.update(["man utd", "pochettino"])
    if "man city" in keywords_away or "manchester city" in keywords_away:
        keywords_away.update(["man city", "maresca"])
    if "man city" in keywords_home or "manchester city" in keywords_home:
        keywords_home.update(["man city", "maresca"])

    alert_terms = [
        "injury", "injured", "fitness", "doubt", "ruled out", "hamstring", "illness",
        "sacked", "manager", "boss", "pressure", "return", "training", "ban", "suspended",
        "blow", "boost", "contract", "dispute", "rumour", "transfer"
    ]

    news_home, news_away = [], []

    for source_name, url in rss_urls:
        try:
            resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                continue
            root = ET.fromstring(resp.content)
            for item in root.findall(".//item")[:40]:
                title = (item.findtext("title") or "").strip()
                desc = (item.findtext("description") or "").strip()
                pub_date = (item.findtext("pubDate") or "").strip()
                full_text = f"{title} {desc}".lower()

                is_alert = any(term in full_text for term in alert_terms)
                entry = {
                    "fuente": source_name,
                    "titular": title,
                    "fecha": pub_date,
                    "alerta_impacto_extracancha": is_alert
                }

                if any(k in full_text for k in keywords_home) and len(news_home) < 5:
                    news_home.append(entry)
                if any(k in full_text for k in keywords_away) and len(news_away) < 5:
                    news_away.append(entry)
        except Exception:
            continue

    return {
        "noticias_y_alertas_local": news_home,
        "noticias_y_alertas_visitante": news_away
    }
