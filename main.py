from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone, timedelta
from typing import Dict, Any
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from data_fetcher import load_historical_data, extract_team_windows, fetch_live_espn_slate, normalize_team
from model_engine import analyze_market_only, analyze_sports_and_context
from context_analyzer import (
    fetch_live_standings_pressure,
    analyze_tactical_styles_and_neutralization,
    fetch_extra_pitch_news
)

app = FastAPI(
    title="La Maña Picks - Motor Cuantitativo Fútbol",
    description="Análisis independiente de Mercado (Cuotas) y Rendimiento Deportivo/Extra-Cancha basado 100% en datos reales.",
    version="2.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {
        "status": "online",
        "proyecto": "La-Mana-Piks-Futbol",
        "liga": "Premier League (2026-2027)",
        "arquitectura": "2 Motores Independientes + Auditoría Pre-Juego"
    }

def fetch_multi_day_slate(known_teams: list) -> list:
    """
    Consulta la cartelera de ESPN de los próximos 10 días para capturar
    todos los días de la jornada (ej. Viernes, Sábado, Domingo, Lunes).
    """
    now = datetime.now(timezone.utc)
    start_str = now.strftime("%Y%m%d")
    end_str = (now + timedelta(days=10)).strftime("%Y%m%d")
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard?dates={start_str}-{end_str}"

    try:
        resp = requests.get(url, timeout=12)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return fetch_live_espn_slate(known_teams)

    events = data.get("events", [])
    if not events:
        return fetch_live_espn_slate(known_teams)

    slate = []
    for ev in events:
        comp = ev["competitions"][0]
        status_state = comp["status"]["type"]["state"]
        if status_state == "post":
            continue  # Solo partidos futuros o en curso

        competitors = comp["competitors"]
        home_c = next(c for c in competitors if c["homeAway"] == "home")
        away_c = next(c for c in competitors if c["homeAway"] == "away")

        home_espn = home_c["team"]["displayName"]
        away_espn = away_c["team"]["displayName"]
        home_norm = normalize_team(home_espn, known_teams)
        away_norm = normalize_team(away_espn, known_teams)

        odds_data = {}
        if comp.get("odds"):
            primary_odds = comp["odds"][0]
            odds_data = {
                "proveedor": primary_odds.get("provider", {}).get("name", "Mercado"),
                "over_under_linea": primary_odds.get("overUnder"),
                "home_moneyline": primary_odds.get("homeTeamOdds", {}).get("moneyLine"),
                "away_moneyline": primary_odds.get("awayTeamOdds", {}).get("moneyLine"),
                "draw_moneyline": primary_odds.get("drawOdds", {}).get("moneyLine"),
            }

        injuries_home, injuries_away = [], []
        for inj_group in comp.get("injuries", []):
            t_name = normalize_team(inj_group.get("team", {}).get("displayName", ""), known_teams)
            for item in inj_group.get("injuries", []):
                ath = item.get("athlete", {}).get("displayName", "")
                st = item.get("status", "")
                if ath:
                    label = f"{ath} ({st})" if st else ath
                    if t_name == home_norm:
                        injuries_home.append(label)
                    elif t_name == away_norm:
                        injuries_away.append(label)

        slate.append({
            "event_id": ev["id"],
            "fecha_utc": ev["date"],
            "estado": comp["status"]["type"]["description"],
            "local_espn": home_espn,
            "visitante_espn": away_espn,
            "local_modelo": home_norm,
            "visitante_modelo": away_norm,
            "record_local_temporada": home_c.get("records", [{}])[0].get("summary", ""),
            "record_visita_temporada": away_c.get("records", [{}])[0].get("summary", ""),
            "cuotas_mercado_raw": odds_data,
            "bajas_reportadas": {"local": injuries_home, "visitante": injuries_away},
            "jugadores_en_ambos_equipos": []
        })

    return slate if slate else fetch_live_espn_slate(known_teams)

@app.get("/api/premier/jornada")
def get_premier_league_analysis() -> Dict[str, Any]:
    try:
        hist_df = load_historical_data()
        known_teams = sorted(list(set(hist_df["HomeTeam"]).union(set(hist_df["AwayTeam"]))))
        slate = fetch_multi_day_slate(known_teams)

        resultados = []
        for match in slate:
            ref_date = datetime.fromisoformat(match["fecha_utc"].replace("Z", "+00:00"))
            home = match["local_modelo"]
            away = match["visitante_modelo"]

            bloque_mercado = analyze_market_only(
                odds_raw=match["cuotas_mercado_raw"],
                historical_df=hist_df,
                home_team=home,
                away_team=away
            )

            windows = extract_team_windows(hist_df, home, away, ref_date)
            bloque_deportivo = analyze_sports_and_context(
                df=hist_df,
                home_team=home,
                away_team=away,
                windows=windows,
                injuries=match["bajas_reportadas"],
                former_players=match["jugadores_en_ambos_equipos"]
            )

            bloque_deportivo["estilos_de_juego_y_neutralizacion"] = analyze_tactical_styles_and_neutralization(
                hist_df, home, away
            )
            bloque_deportivo["factores_extra_cancha"]["presion_tabla_y_clasificacion"] = fetch_live_standings_pressure(
                home, away, known_teams
            )
            bloque_deportivo["factores_extra_cancha"]["prensa_lesiones_y_rumores_rss"] = fetch_extra_pitch_news(
                match["local_espn"], match["visitante_espn"], home, away
            )

            resultados.append({
                "partido_info": {
                    "event_id": match["event_id"],
                    "fecha_utc": match["fecha_utc"],
                    "estado": match["estado"],
                    "local": match["local_espn"],
                    "visitante": match["visitante_espn"],
                    "local_modelo": home,
                    "visitante_modelo": away,
                    "record_local": match["record_local_temporada"],
                    "record_visita": match["record_visita_temporada"]
                },
                "bloque_1_mercado_cuotas": bloque_mercado,
                "bloque_2_deportivo_y_extracancha": bloque_deportivo
            })

        return {
            "actualizado_utc": datetime.now(timezone.utc).isoformat(),
            "total_partidos_jornada": len(resultados),
            "partidos": resultados
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/premier/auditoria")
def get_premier_backtest_audit() -> Dict[str, Any]:
    """
    Auditoría retrospectiva estricta: toma la última fecha de partidos jugados
    y simula qué habría pronosticado el modelo usando ÚNICAMENTE datos previos
    a esos juegos (sin mirar los resultados futuros).
    """
    try:
        hist_df = load_historical_data()
        pl_df = hist_df[hist_df["Season"].str.contains("PL")].copy()
        if pl_df.empty:
            pl_df = hist_df.copy()

        # Tomar los últimos 10 partidos jugados de Premier League (la última jornada completa)
        last_10 = pl_df.sort_values("DateParsed").tail(10)
        if last_10.empty:
            return {"fecha_auditoria": "N/A", "partidos": []}

        cutoff_date = last_10["DateParsed"].min()
        prior_df = hist_df[hist_df["DateParsed"] < cutoff_date].copy()
        if len(prior_df) < 30:
            prior_df = hist_df.iloc[:-len(last_10)].copy()

        audit_matches = []
        aciertos_pick = 0
        aciertos_ml = 0

        for _, row in last_10.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            hg = int(row["FTHG"])
            ag = int(row["FTAG"])
            match_dt = row["DateParsed"].to_pydatetime().replace(tzinfo=timezone.utc)

            windows = extract_team_windows(prior_df, home, away, match_dt)
            dep = analyze_sports_and_context(
                df=prior_df,
                home_team=home,
                away_team=away,
                windows=windows,
                injuries={"local": [], "visitante": []},
                former_players=[]
            )
            dep["estilos_de_juego_y_neutralizacion"] = analyze_tactical_styles_and_neutralization(prior_df, home, away)

            p1x2 = dep["simulacion_monte_carlo"]["probabilidades_1X2_pct"]
            pgol = dep["simulacion_monte_carlo"]["mercados_goles_pct"]

            # Evaluar mejor pick pre-juego
            prob_1x = round(p1x2["local_1"] + p1x2["empate_X"], 1)
            prob_x2 = round(p1x2["visita_2"] + p1x2["empate_X"], 1)

            opciones = [
                {"tipo": "1X", "label": f"{home} o Empate (1X)", "prob": prob_1x, "gano": hg >= ag},
                {"tipo": "X2", "label": f"{away} o Empate (X2)", "prob": prob_x2, "gano": ag >= hg},
                {"tipo": "1", "label": f"{home} gana (1)", "prob": p1x2["local_1"], "gano": hg > ag},
                {"tipo": "2", "label": f"{away} gana (2)", "prob": p1x2["visita_2"], "gano": ag > hg},
                {"tipo": "O25", "label": "Over 2.5 Goles", "prob": pgol["over_2_5"], "gano": (hg + ag) > 2.5},
                {"tipo": "U25", "label": "Under 2.5 Goles", "prob": pgol["under_2_5"], "gano": (hg + ag) < 2.5}
            ]
            opciones.sort(key=lambda x: x["prob"], reverse=True)
            pick_elegido = opciones[0]

            # Favorito directo 1X2
            ml_fav_home = p1x2["local_1"] >= p1x2["visita_2"]
            ml_acerto = (hg > ag) if ml_fav_home else (ag > hg)

            if pick_elegido["gano"]:
                aciertos_pick += 1
            if ml_acerto:
                aciertos_ml += 1

            mercado_hist = analyze_market_only(
                {"proveedor": "Pinnacle Cierre", "home_moneyline": row.get("PSH"), "draw_moneyline": row.get("PSD"), "away_moneyline": row.get("PSA")},
                prior_df, home, away
            )

            audit_matches.append({
                "fecha": row["DateParsed"].strftime("%Y-%m-%d"),
                "local": home,
                "visitante": away,
                "marcador_real": f"{hg}–{ag}",
                "pick_pre_juego": pick_elegido["label"],
                "probabilidad_pick": pick_elegido["prob"],
                "acerto": bool(pick_elegido["gano"]),
                "bloque_1_mercado_cuotas": mercado_hist,
                "bloque_2_deportivo_y_extracancha": dep
            })

        fecha_label = last_10["DateParsed"].max().strftime("%d %b").upper()
        return {
            "fecha_etiqueta": fecha_label,
            "total": len(audit_matches),
            "aciertos_pick": aciertos_pick,
            "aciertos_ml": aciertos_ml,
            "partidos": audit_matches
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/premier/simular/{local}/{visitante}")
def simulate_custom_match(local: str, visitante: str) -> Dict[str, Any]:
    try:
        hist_df = load_historical_data()
        known_teams = sorted(list(set(hist_df["HomeTeam"]).union(set(hist_df["AwayTeam"]))))
        home = normalize_team(local, known_teams)
        away = normalize_team(visitante, known_teams)

        if home not in known_teams or away not in known_teams:
            raise HTTPException(
                status_code=400,
                detail=f"Equipo no encontrado. Equipos disponibles: {known_teams}"
            )

        now = datetime.now(timezone.utc)
        windows = extract_team_windows(hist_df, home, away, now)
        bloque_deportivo = analyze_sports_and_context(
            df=hist_df,
            home_team=home,
            away_team=away,
            windows=windows,
            injuries={"local": [], "visitante": []},
            former_players=[]
        )
        bloque_deportivo["estilos_de_juego_y_neutralizacion"] = analyze_tactical_styles_and_neutralization(
            hist_df, home, away
        )
        bloque_deportivo["factores_extra_cancha"]["presion_tabla_y_clasificacion"] = fetch_live_standings_pressure(
            home, away, known_teams
        )
        bloque_deportivo["factores_extra_cancha"]["prensa_lesiones_y_rumores_rss"] = fetch_extra_pitch_news(
            home, away, home, away
        )
        bloque_mercado = analyze_market_only({}, hist_df, home, away)

        return {
            "cruce": f"{home} vs {away}",
            "local_modelo": home,
            "visitante_modelo": away,
            "bloque_1_mercado_cuotas": bloque_mercado,
            "bloque_2_deportivo_y_extracancha": bloque_deportivo
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/premier/liga-info")
def get_league_standings_and_news() -> Dict[str, Any]:
    try:
        hist_df = load_historical_data()
        known_teams = sorted(list(set(hist_df["HomeTeam"]).union(set(hist_df["AwayTeam"]))))

        standings_list = []
        try:
            url = "https://site.api.espn.com/apis/v2/sports/soccer/eng.1/standings"
            resp = requests.get(url, timeout=10)
            data = resp.json()
            entries = data["children"][0]["standings"]["entries"]
            for idx, entry in enumerate(entries, start=1):
                team_obj = entry["team"]
                t_norm = normalize_team(team_obj["displayName"], known_teams)
                logos = team_obj.get("logos", [])
                logo_url = logos[0]["href"] if logos else ""
                stats = {}
                for s in entry.get("stats", []):
                    val = s.get("value") if s.get("value") is not None else s.get("displayValue", 0)
                    try:
                        stats[s["name"]] = float(val)
                    except Exception:
                        stats[s["name"]] = 0

                jj = int(stats.get("gamesPlayed", 0))
                gf = int(stats.get("pointsFor", 0))
                gc = int(stats.get("pointsAgainst", 0))
                pts = int(stats.get("points", 0))
                dif = int(stats.get("pointDifferential", gf - gc))
                gfj = round(gf / max(jj, 1), 2)
                gcj = round(gc / max(jj, 1), 2)
                pyth = round(((gf ** 1.35) / max((gf ** 1.35) + (gc ** 1.35), 1)) * 100.0, 1) if (gf + gc) > 0 else 50.0

                standings_list.append({
                    "pos": int(stats.get("rank") or idx),
                    "equipo_espn": team_obj["displayName"],
                    "equipo_modelo": t_norm,
                    "abreviatura": team_obj.get("abbreviation", t_norm[:3].upper()),
                    "escudo": logo_url,
                    "jj": jj,
                    "g": int(stats.get("wins", 0)),
                    "e": int(stats.get("ties", 0)),
                    "p": int(stats.get("losses", 0)),
                    "gf": gf,
                    "gc": gc,
                    "dif": dif,
                    "pts": pts,
                    "gfj": gfj,
                    "gcj": gcj,
                    "pyth": pyth
                })
            standings_list.sort(key=lambda x: (x["pos"], -x["pts"], -x["dif"]))
        except Exception:
            pass

        rss_urls = [
            ("BBC Sport", "https://feeds.bbci.co.uk/sport/football/rss.xml"),
            ("Sky Sports", "https://www.skysports.com/rss/11095")
        ]
        alert_terms = [
            "injury", "injured", "fitness", "doubt", "ruled out", "hamstring", "illness",
            "sacked", "manager", "boss", "pressure", "return", "training", "ban", "suspended",
            "blow", "boost", "contract", "dispute", "rumour", "weather", "postponed"
        ]
        noticias = []
        for source_name, url in rss_urls:
            try:
                r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code != 200:
                    continue
                root = ET.fromstring(r.content)
                for item in root.findall(".//item")[:25]:
                    title = (item.findtext("title") or "").strip()
                    desc = (item.findtext("description") or "").strip()
                    pub = (item.findtext("pubDate") or "").strip()
                    txt = f"{title} {desc}".lower()
                    is_alert = any(t in txt for t in alert_terms)
                    noticias.append({
                        "fuente": source_name,
                        "titular": title,
                        "resumen": desc[:180] + ("..." if len(desc) > 180 else ""),
                        "fecha": pub,
                        "tipo": "ALERTA / BAJA / DT" if is_alert else "NOTICIA LIGA",
                        "critica": is_alert
                    })
            except Exception:
                continue

        return {
            "tabla_posiciones": standings_list,
            "noticias_liga": noticias[:30]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
