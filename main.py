from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
import threading
import os
import json
import time
import requests
import xml.etree.ElementTree as ET
from data_fetcher import load_historical_data, extract_team_windows, fetch_live_espn_slate, normalize_team
from model_engine import analyze_market_only, analyze_sports_and_context
from context_analyzer import (
    analyze_tactical_styles_and_neutralization,
)

app = FastAPI(
    title="La Maña Picks - Motor Cuantitativo Fútbol",
    description="Análisis independiente de Mercado (Cuotas) y Rendimiento Deportivo/Extra-Cancha basado 100% en datos reales.",
    version="2.7.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_CACHE: Dict[str, Any] = {}
CACHE_TTL_SECONDS = 900
AUDIT_LOG_FILE = "auditoria_real_picks.json"
MODEL_START_DATE = "2026-11-25"
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json"
}

def get_cached(key: str):
    item = _CACHE.get(key)
    if item and (time.time() - item["ts"]) < CACHE_TTL_SECONDS:
        return item["data"]
    return None

def set_cached(key: str, data: Any):
    _CACHE[key] = {"ts": time.time(), "data": data}

def get_historical_cached():
    df = get_cached("hist_df")
    if df is None:
        df = load_historical_data()
        set_cached("hist_df", df)
    return df

def get_rss_feed_cached() -> List[Dict[str, Any]]:
    cached = get_cached("rss_news")
    if cached is not None:
        return cached

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
            r = requests.get(url, timeout=5, headers=HTTP_HEADERS)
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

    if noticias:
        set_cached("rss_news", noticias)
    return noticias

def filter_team_news(all_news: List[Dict[str, Any]], home_espn: str, away_espn: str, home_short: str, away_short: str) -> Dict[str, Any]:
    kh = {home_espn.lower(), home_short.lower()}
    ka = {away_espn.lower(), away_short.lower()}
    if "man united" in ka or "manchester united" in ka:
        ka.update(["man utd", "pochettino"])
    if "man united" in kh or "manchester united" in kh:
        kh.update(["man utd", "pochettino"])
    if "man city" in ka or "manchester city" in ka:
        ka.update(["man city", "maresca"])
    if "man city" in kh or "manchester city" in kh:
        kh.update(["man city", "maresca"])

    nh, na = [], []
    for n in all_news:
        txt = f"{n['titular']} {n['resumen']}".lower()
        entry = {
            "fuente": n["fuente"],
            "titular": n["titular"],
            "fecha": n["fecha"],
            "alerta_impacto_extracancha": n["critica"]
        }
        if any(k in txt for k in kh) and len(nh) < 5:
            nh.append(entry)
        if any(k in txt for k in ka) and len(na) < 5:
            na.append(entry)
    return {"noticias_y_alertas_local": nh, "noticias_y_alertas_visitante": na}

def get_standings_cached(known_teams: list) -> List[Dict[str, Any]]:
    cached = get_cached("standings_raw")
    if cached is not None:
        return cached

    standings_list = []
    try:
        url = "https://site.api.espn.com/apis/v2/sports/soccer/eng.1/standings"
        resp = requests.get(url, timeout=6, headers=HTTP_HEADERS)
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
        for i, r in enumerate(standings_list, start=1):
            r["pos"] = i
    except Exception:
        pass

    if standings_list:
        set_cached("standings_raw", standings_list)
    return standings_list

def build_table_pressure_from_cached(table: List[Dict[str, Any]], home_team: str, away_team: str) -> Dict[str, Any]:
    if len(table) < 18:
        return {"estado": "Tabla no disponible"}
    pts_1st = table[0]["pts"]
    pts_4th = table[3]["pts"]
    pts_6th = table[5]["pts"]
    pts_17th = table[16]["pts"]
    pts_18th = table[17]["pts"]

    def eval_t(team: str) -> Dict[str, Any]:
        row = next((r for r in table if r["equipo_modelo"] == team), None)
        if not row:
            return {"posicion_actual": "-", "puntos": 0, "contexto_clasificacion": "Sin datos"}
        pos = row["pos"]
        pts = row["pts"]
        if pos == 1:
            obj = f"Líder (+{pts - table[1]['pts']} pts sobre el 2°)"
        elif pos <= 4:
            obj = f"Zona Champions (a {pts_1st - pts} pts del líder, +{pts - table[4]['pts']} pts sobre el 5°)"
        elif pos <= 7:
            obj = f"Pelea Europa (a {pts_4th - pts} pts de Champions)"
        elif pos >= 18:
            obj = f"Zona DESCENSO (a {pts_17th - pts} pts de salvación)"
        elif pos >= 15:
            obj = f"Alerta descenso (solo +{pts - pts_18th} pts sobre el 18°)"
        else:
            obj = f"Mitad de tabla (a {pts_6th - pts} pts de Europa / +{pts - pts_18th} pts sobre descenso)"
        return {"posicion_actual": pos, "puntos": pts, "diferencia_goles": row["dif"], "contexto_clasificacion": obj}

    return {"local": eval_t(home_team), "visitante": eval_t(away_team)}

def parse_espn_events_to_slate(events: list, known_teams: list, seen_ids: set) -> list:
    parsed = []
    for ev in events:
        ev_id = str(ev.get("id", ""))
        if not ev_id or ev_id in seen_ids:
            continue
        comps = ev.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]
        if comp.get("status", {}).get("type", {}).get("state") == "post":
            continue
        seen_ids.add(ev_id)

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

        parsed.append({
            "event_id": ev_id,
            "fecha_utc": ev.get("date", datetime.now(timezone.utc).isoformat()),
            "estado": comp.get("status", {}).get("type", {}).get("description", "Scheduled"),
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
    return parsed

def fetch_full_round_espn_slate(known_teams: list) -> list:
    now = datetime.now(timezone.utc)
    start_str = now.strftime("%Y%m%d")
    end_str = (now + timedelta(days=9)).strftime("%Y%m%d")

    urls = [
        f"https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard?dates={start_str}-{end_str}",
        "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"
    ]

    seen_ids = set()
    slate = []

    for url in urls:
        try:
            resp = requests.get(url, timeout=8, headers=HTTP_HEADERS)
            if resp.status_code == 200:
                events = resp.json().get("events", [])
                slate.extend(parse_espn_events_to_slate(events, known_teams, seen_ids))
        except Exception:
            continue

    if not slate:
        slate = fetch_live_espn_slate(known_teams)

    slate.sort(key=lambda x: x["fecha_utc"])
    return slate

def load_audit_store() -> Dict[str, Any]:
    if os.path.exists(AUDIT_LOG_FILE):
        try:
            with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"modelo_iniciado": MODEL_START_DATE, "registros": {}}

def save_audit_store(store: Dict[str, Any]):
    try:
        with open(AUDIT_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def registrar_pronosticos_prejuego(resultados: List[Dict[str, Any]]):
    store = load_audit_store()
    registros = store.setdefault("registros", {})
    changed = False

    for item in resultados:
        info = item["partido_info"]
        ev_id = str(info["event_id"])
        if ev_id in registros:
            continue

        dep = item["bloque_2_deportivo_y_extracancha"]
        p1x2 = dep["simulacion_monte_carlo"]["probabilidades_1X2_pct"]
        pgol = dep["simulacion_monte_carlo"]["mercados_goles_pct"]
        home = info["local_modelo"]
        away = info["visitante_modelo"]

        prob_1x = round(p1x2["local_1"] + p1x2["empate_X"], 1)
        prob_x2 = round(p1x2["visita_2"] + p1x2["empate_X"], 1)

        opciones = [
            {"tipo": "1X", "label": f"{home} o Empate (1X)", "prob": prob_1x},
            {"tipo": "X2", "label": f"{away} o Empate (X2)", "prob": prob_x2},
            {"tipo": "1", "label": f"{home} gana (1)", "prob": p1x2["local_1"]},
            {"tipo": "2", "label": f"{away} gana (2)", "prob": p1x2["visita_2"]},
            {"tipo": "O25", "label": "Over 2.5 Goles", "prob": pgol["over_2_5"]},
            {"tipo": "U25", "label": "Under 2.5 Goles", "prob": pgol["under_2_5"]}
        ]
        opciones.sort(key=lambda x: x["prob"], reverse=True)
        pick = opciones[0]

        registros[ev_id] = {
            "event_id": ev_id,
            "fecha_utc": info["fecha_utc"],
            "emitido_utc": datetime.now(timezone.utc).isoformat(),
            "local": home,
            "visitante": away,
            "local_espn": info["local"],
            "visitante_espn": info["visitante"],
            "pick_tipo": pick["tipo"],
            "pick_label": pick["label"],
            "prob_pick": pick["prob"],
            "favorito_1x2": "1" if p1x2["local_1"] >= p1x2["visita_2"] else "2",
            "xg_local": dep["fuerzas_y_elo_dinamico"]["xg_proyectado_local"],
            "xg_visita": dep["fuerzas_y_elo_dinamico"]["xg_proyectado_visitante"],
            "estado": "PENDIENTE",
            "marcador_real": None,
            "acerto_pick": None,
            "acerto_1x2": None
        }
        changed = True

    if changed:
        save_audit_store(store)

def compute_jornada_payload() -> Dict[str, Any]:
    hist_df = get_historical_cached()
    known_teams = sorted(list(set(hist_df["HomeTeam"]).union(set(hist_df["AwayTeam"]))))
    slate = fetch_full_round_espn_slate(known_teams)
    standings_table = get_standings_cached(known_teams)
    all_rss = get_rss_feed_cached()

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
        bloque_deportivo["factores_extra_cancha"]["presion_tabla_y_clasificacion"] = build_table_pressure_from_cached(
            standings_table, home, away
        )
        bloque_deportivo["factores_extra_cancha"]["prensa_lesiones_y_rumores_rss"] = filter_team_news(
            all_rss, match["local_espn"], match["visitante_espn"], home, away
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

    if resultados:
        registrar_pronosticos_prejuego(resultados)

    payload = {
        "actualizado_utc": datetime.now(timezone.utc).isoformat(),
        "total_partidos_jornada": len(resultados),
        "partidos": resultados
    }
    if len(resultados) > 0:
        set_cached("jornada_full", payload)
    return payload

@app.on_event("startup")
def warmup_cache_on_boot():
    def _bg():
        try:
            compute_jornada_payload()
        except Exception:
            pass
    threading.Thread(target=_bg, daemon=True).start()

@app.get("/")
def health_check():
    return {
        "status": "online",
        "proyecto": "La-Mana-Piks-Futbol",
        "liga": "Premier League (2026-2027)",
        "modelo_iniciado": MODEL_START_DATE
    }

@app.get("/api/premier/jornada")
def get_premier_league_analysis() -> Dict[str, Any]:
    cached_resp = get_cached("jornada_full")
    if cached_resp is not None and len(cached_resp.get("partidos", [])) > 0:
        return cached_resp

    try:
        return compute_jornada_payload()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/premier/auditoria")
def get_premier_real_audit() -> Dict[str, Any]:
    try:
        store = load_audit_store()
        registros = store.get("registros", {})
        changed = False

        for ev_id, reg in registros.items():
            if reg.get("estado") == "FINALIZADO":
                continue
            try:
                day_str = reg["fecha_utc"][:10].replace("-", "")
                url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard?dates={day_str}"
                r = requests.get(url, timeout=5, headers=HTTP_HEADERS)
                if r.status_code == 200:
                    for ev in r.json().get("events", []):
                        if str(ev.get("id")) == ev_id:
                            comp = ev["competitions"][0]
                            if comp["status"]["type"]["state"] == "post":
                                comps = comp["competitors"]
                                hc = next(c for c in comps if c["homeAway"] == "home")
                                ac = next(c for c in comps if c["homeAway"] == "away")
                                hg = int(hc.get("score", 0))
                                ag = int(ac.get("score", 0))

                                p_tipo = reg["pick_tipo"]
                                gano_pick = (
                                    (p_tipo == "1X" and hg >= ag) or
                                    (p_tipo == "X2" and ag >= hg) or
                                    (p_tipo == "1" and hg > ag) or
                                    (p_tipo == "2" and ag > hg) or
                                    (p_tipo == "O25" and (hg + ag) > 2.5) or
                                    (p_tipo == "U25" and (hg + ag) < 2.5)
                                )
                                gano_1x2 = (hg > ag) if reg["favorito_1x2"] == "1" else (ag > hg)

                                reg["estado"] = "FINALIZADO"
                                reg["marcador_real"] = f"{hg}–{ag}"
                                reg["acerto_pick"] = bool(gano_pick)
                                reg["acerto_1x2"] = bool(gano_1x2)
                                changed = True
            except Exception:
                continue

        if changed:
            save_audit_store(store)

        lista = sorted(registros.values(), key=lambda x: x["fecha_utc"])
        finalizados = [x for x in lista if x["estado"] == "FINALIZADO"]
        aciertos_pick = sum(1 for x in finalizados if x["acerto_pick"])
        aciertos_1x2 = sum(1 for x in finalizados if x["acerto_1x2"])

        return {
            "modelo_iniciado": MODEL_START_DATE,
            "total_registrados": len(lista),
            "total_finalizados": len(finalizados),
            "aciertos_pick": aciertos_pick,
            "aciertos_ml": aciertos_1x2,
            "partidos": lista
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/premier/simular/{local}/{visitante}")
def simulate_custom_match(local: str, visitante: str) -> Dict[str, Any]:
    try:
        hist_df = get_historical_cached()
        known_teams = sorted(list(set(hist_df["HomeTeam"]).union(set(hist_df["AwayTeam"]))))
        home = normalize_team(local, known_teams)
        away = normalize_team(visitante, known_teams)

        if home not in known_teams or away not in known_teams:
            raise HTTPException(status_code=400, detail=f"Equipo no encontrado. Disponibles: {known_teams}")

        standings_table = get_standings_cached(known_teams)
        all_rss = get_rss_feed_cached()

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
        bloque_deportivo["factores_extra_cancha"]["presion_tabla_y_clasificacion"] = build_table_pressure_from_cached(
            standings_table, home, away
        )
        bloque_deportivo["factores_extra_cancha"]["prensa_lesiones_y_rumores_rss"] = filter_team_news(
            all_rss, home, away, home, away
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
        hist_df = get_historical_cached()
        known_teams = sorted(list(set(hist_df["HomeTeam"]).union(set(hist_df["AwayTeam"]))))
        standings_list = get_standings_cached(known_teams)
        noticias = get_rss_feed_cached()
        return {
            "tabla_posiciones": standings_list,
            "noticias_liga": noticias[:30]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
