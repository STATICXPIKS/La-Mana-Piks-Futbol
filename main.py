from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
from typing import Dict, Any
import requests
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
    version="2.0.0"
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
        "arquitectura": "2 Motores Independientes (Mercado vs Deportivo/Táctico/Extra-Cancha)"
    }

@app.get("/api/premier/jornada")
def get_premier_league_analysis() -> Dict[str, Any]:
    try:
        hist_df = load_historical_data()
        known_teams = sorted(list(set(hist_df["HomeTeam"]).union(set(hist_df["AwayTeam"]))))
        slate = fetch_live_espn_slate(known_teams)

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
    """
    Entrega la tabla real completa de los 20 equipos de Premier League (con escudos)
    y el feed general de noticias, lesiones y alertas de última hora.
    """
    try:
        hist_df = load_historical_data()
        known_teams = sorted(list(set(hist_df["HomeTeam"]).union(set(hist_df["AwayTeam"]))))

        # 1. Tabla real en vivo desde ESPN
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

        # 2. Noticias generales de Premier League (BBC Sport + Sky Sports RSS)
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
