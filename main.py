from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
from typing import Dict, Any
from data_fetcher import load_historical_data, extract_team_windows, fetch_live_espn_slate, normalize_team
from model_engine import analyze_market_only, analyze_sports_and_context

app = FastAPI(
    title="La Maña Piks - Motor Cuantitativo Premier League",
    description="Análisis independiente de Mercado (Cuotas) y Rendimiento Deportivo/Extra-Cancha basado 100% en datos reales.",
    version="1.0.0"
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
        "arquitectura": "2 Motores Independientes (Mercado vs Deportivo/Extra-Cancha)"
    }

@app.get("/api/premier/jornada")
def get_premier_league_analysis() -> Dict[str, Any]:
    """
    Descarga los datos reales actuales y entrega para cada partido de la jornada
    dos bloques totalmente independientes:
    1) analisis_mercado_cuotas
    2) analisis_deportivo_y_extracancha
    """
    try:
        hist_df = load_historical_data()
        known_teams = sorted(list(set(hist_df["HomeTeam"]).union(set(hist_df["AwayTeam"]))))
        slate = fetch_live_espn_slate(known_teams)

        resultados = []
        for match in slate:
            ref_date = datetime.fromisoformat(match["fecha_utc"].replace("Z", "+00:00"))
            home = match["local_modelo"]
            away = match["visitante_modelo"]

            # 1. Análisis exclusivo de cuotas del mercado
            bloque_mercado = analyze_market_only(
                odds_raw=match["cuotas_mercado_raw"],
                historical_df=hist_df,
                home_team=home,
                away_team=away
            )

            # 2. Extracción de ventanas históricas reales y análisis deportivo/extra-cancha
            windows = extract_team_windows(hist_df, home, away, ref_date)
            bloque_deportivo = analyze_sports_and_context(
                df=hist_df,
                home_team=home,
                away_team=away,
                windows=windows,
                injuries=match["bajas_reportadas"],
                former_players=match["jugadores_en_ambos_equipos"]
            )

            resultados.append({
                "partido_info": {
                    "event_id": match["event_id"],
                    "fecha_utc": match["fecha_utc"],
                    "estado": match["estado"],
                    "local": match["local_espn"],
                    "visitante": match["visitante_espn"],
                    "record_local": match["record_local_temporada"],
                    "record_visita": match["record_visita_temporada"]
                },
                "bloque_1_mercado_cuotas": bloque_mercado,
                "bloque_2_deportivo_y_extracancha": bloque_deportivo
            })

        return {
            "actualizado_utc": datetime.now(timezone.utc).isoformat(),
            "total_partidos_jornada": len(resultados),
            "equipos_activos_detectados": len(known_teams),
            "partidos": resultados
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/premier/simular/{local}/{visitante}")
def simulate_custom_match(local: str, visitante: str) -> Dict[str, Any]:
    """
    Permite analizar cualquier cruce específico entre 2 equipos de los 20 reales
    de la Premier League 2026-2027 (ej. /api/premier/simular/Arsenal/M.%20City).
    """
    try:
        hist_df = load_historical_data()
        known_teams = sorted(list(set(hist_df["HomeTeam"]).union(set(hist_df["AwayTeam"]))))
        home = normalize_team(local, known_teams)
        away = normalize_team(visitante, known_teams)

        if home not in known_teams or away not in known_teams:
            raise HTTPException(
                status_code=400,
                detail=f"Equipo no encontrado. Equipos reales disponibles: {known_teams}"
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
        bloque_mercado = analyze_market_only({}, hist_df, home, away)

        return {
            "cruce": f"{home} vs {away}",
            "bloque_1_mercado_cuotas": bloque_mercado,
            "bloque_2_deportivo_y_extracancha": bloque_deportivo
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
