"""
Football Value Bet Dashboard — Streamlit app
Runs analyzer.py logic and displays results visually
"""

import os
import json
import datetime
import streamlit as st
import pandas as pd
from analyzer import (
    FootballDataClient, OddsClient, PoissonModel,
    detect_value_bets, teams_match,
    COMPETITIONS, SPORT_KEYS, VALUE_THRESHOLD, MIN_MATCHES_FOR_MODEL,
    DEMO_MATCHES, DEMO_HISTORY, DEMO_ODDS,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚽ Value Bet Analyzer",
    page_icon="⚽",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  h1, h2, h3 { font-family: 'Syne', sans-serif; }
  .match-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
  }
  .value-pill {
    background: #16a34a22;
    border: 1px solid #16a34a;
    color: #4ade80;
    border-radius: 999px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    display: inline-block;
  }
  .no-value-pill {
    background: #ffffff0a;
    border: 1px solid #334155;
    color: #64748b;
    border-radius: 999px;
    padding: 2px 12px;
    font-size: 0.78rem;
    display: inline-block;
  }
  .edge-bar-bg {
    background: #1e293b;
    border-radius: 999px;
    height: 6px;
    width: 100%;
  }
  .metric-val { font-size: 1.6rem; font-weight: 700; font-family: 'Syne', sans-serif; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ⚽ Football Value Bet Analyzer")
st.markdown(f"**{datetime.datetime.now().strftime('%A, %d %B %Y')}** — Partidos de hoy con edge de mercado")

st.divider()

# ── API keys ─────────────────────────────────────────────────────────────────
FD_KEY   = os.getenv("FOOTBALL_DATA_API_KEY", "YOUR_FOOTBALL_DATA_KEY")
ODDS_KEY = os.getenv("ODDS_API_KEY", "YOUR_ODDS_API_KEY")
DEMO     = FD_KEY == "YOUR_FOOTBALL_DATA_KEY" or ODDS_KEY == "YOUR_ODDS_API_KEY"

if DEMO:
    st.info("ℹ️ **Modo demo** — configura las variables de entorno `FOOTBALL_DATA_API_KEY` y `ODDS_API_KEY` para datos reales.", icon="ℹ️")

# ── Fetch & analyze ───────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)   # Cache 1 hour
def run_analysis():
    if DEMO:
        today_matches = DEMO_MATCHES
        all_odds      = DEMO_ODDS
        fd_client     = None
        odds_client   = None
    else:
        fd_client   = FootballDataClient(FD_KEY)
        odds_client = OddsClient(ODDS_KEY)
        today_matches = fd_client.get_today_matches()
        all_odds = []
        for sk in SPORT_KEYS:
            all_odds.extend(odds_client.get_odds(sk))

    model   = PoissonModel()
    results = []

    for match in today_matches:
        home_team = match["homeTeam"]["name"]
        away_team = match["awayTeam"]["name"]
        home_id   = match["homeTeam"]["id"]
        away_id   = match["awayTeam"]["id"]
        comp      = match["_competition"]
        kickoff   = match.get("utcDate", "")[:16].replace("T", " ") + " UTC"

        if DEMO:
            home_hist = DEMO_HISTORY.get(home_id, [])
            away_hist = DEMO_HISTORY.get(away_id, [])
        else:
            home_hist = fd_client.get_team_last_matches(home_id)
            away_hist = fd_client.get_team_last_matches(away_id)

        if len(home_hist) < MIN_MATCHES_FOR_MODEL or len(away_hist) < MIN_MATCHES_FOR_MODEL:
            continue

        lam_h, lam_a = model.expected_goals(home_hist, away_hist, home_id, away_id)
        probs        = model.match_probabilities(lam_h, lam_a)

        matched_event = next(
            (ev for ev in all_odds
             if teams_match(home_team, away_team,
                            ev.get("home_team",""), ev.get("away_team",""))),
            None
        )

        if matched_event:
            if DEMO:
                best = {"home": 0.0, "draw": 0.0, "away": 0.0}
                for bm in matched_event.get("bookmakers", []):
                    for mkt in bm.get("markets", []):
                        for o in mkt.get("outcomes", []):
                            n, p = o.get("name",""), o.get("price", 0)
                            if n == home_team:   best["home"] = max(best["home"], p)
                            elif n == away_team: best["away"] = max(best["away"], p)
                            elif n == "Draw":    best["draw"] = max(best["draw"], p)
            else:
                best = odds_client.best_odds(matched_event)
            bets     = detect_value_bets(probs, best)
            has_odds = True
        else:
            best, bets, has_odds = {}, [], False

        results.append({
            "home_team":   home_team,
            "away_team":   away_team,
            "competition": comp,
            "kickoff":     kickoff,
            "lambda_home": lam_h,
            "lambda_away": lam_a,
            "model_probs": probs,
            "best_odds":   best,
            "value_bets":  bets,
            "has_odds":    has_odds,
        })

    return results

# ── Run ───────────────────────────────────────────────────────────────────────
with st.spinner("Analizando partidos y buscando value bets…"):
    results = run_analysis()

if not results:
    st.warning("No hay partidos programados hoy o no hay suficientes datos históricos.")
    st.stop()

# ── KPI row ──────────────────────────────────────────────────────────────────
total_bets    = sum(len(r["value_bets"]) for r in results)
value_matches = sum(1 for r in results if r["value_bets"])
best_edge     = max((b["edge"] for r in results for b in r["value_bets"]), default=0)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Partidos analizados", len(results))
k2.metric("Con value bets", value_matches)
k3.metric("Total value bets", total_bets)
k4.metric("Mejor edge", f"+{best_edge:.1%}" if best_edge else "—")

st.divider()

# ── Filter sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Filtros")
    only_value = st.toggle("Solo partidos con value bets", value=False)
    min_edge   = st.slider("Edge mínimo (%)", 0, 50, 5) / 100
    st.markdown("---")
    st.markdown("### ℹ️ Sobre el modelo")
    st.markdown(
        "Usa la distribución de **Poisson** para estimar goles esperados "
        "según el historial reciente de cada equipo. Un *value bet* aparece "
        "cuando la probabilidad del modelo supera la implícita en las cuotas."
    )
    if st.button("🔄 Actualizar datos"):
        st.cache_data.clear()
        st.rerun()

# ── Top value bets table ──────────────────────────────────────────────────────
all_bets_rows = []
for r in results:
    for b in r["value_bets"]:
        if b["edge"] >= min_edge:
            all_bets_rows.append({
                "Partido":     f"{r['home_team']} vs {r['away_team']}",
                "Liga":        r["competition"],
                "Hora":        r["kickoff"],
                "Apuesta":     b["outcome"],
                "Cuota":       f"{b['best_odds']:.2f}",
                "Modelo":      f"{b['model_prob']:.1%}",
                "Mercado":     f"{b['implied_prob']:.1%}",
                "Edge":        f"+{b['edge']:.1%}",
                "_edge_raw":   b["edge"],
            })

if all_bets_rows:
    st.markdown("### 🎯 Ranking de Value Bets")
    df = pd.DataFrame(all_bets_rows).sort_values("_edge_raw", ascending=False).drop(columns=["_edge_raw"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.divider()

# ── Match cards ───────────────────────────────────────────────────────────────
st.markdown("### 📋 Todos los partidos")

filtered = [r for r in results if not only_value or r["value_bets"]]

for r in filtered:
    has_bets   = bool(r["value_bets"])
    badge      = '<span class="value-pill">🎯 VALUE BET</span>' if has_bets else '<span class="no-value-pill">Sin edge</span>'
    probs      = r["model_probs"]
    lam_h      = r["lambda_home"]
    lam_a      = r["lambda_away"]

    with st.expander(f"**{r['home_team']}** vs **{r['away_team']}**  —  {r['competition']}  |  {r['kickoff']}", expanded=has_bets):
        col_info, col_probs = st.columns([2, 3])

        with col_info:
            st.markdown(badge, unsafe_allow_html=True)
            st.markdown(f"**Goles esperados**")
            st.markdown(f"🏠 {r['home_team']}: **{lam_h}**")
            st.markdown(f"✈️ {r['away_team']}: **{lam_a}**")

        with col_probs:
            st.markdown("**Probabilidades del modelo**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Local gana", f"{probs['home_win']:.1%}")
            c2.metric("Empate",     f"{probs['draw']:.1%}")
            c3.metric("Visitante",  f"{probs['away_win']:.1%}")

        if r["value_bets"]:
            st.markdown("**Value bets detectadas:**")
            for b in r["value_bets"]:
                if b["edge"] < min_edge:
                    continue
                edge_pct = min(b["edge"] * 100, 100)
                bcol1, bcol2, bcol3, bcol4 = st.columns([2, 1, 1, 3])
                bcol1.markdown(f"🎯 **{b['outcome']}**")
                bcol2.markdown(f"Cuota: **{b['best_odds']:.2f}**")
                bcol3.markdown(f"Edge: **+{b['edge']:.1%}**")
                bcol4.progress(int(edge_pct))
        elif not r["has_odds"]:
            st.caption("⚠️ Sin cuotas disponibles para este partido.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("⚠️ Solo con fines educativos. Las apuestas conllevan riesgo de pérdida. El modelo de Poisson es una simplificación estadística.")
