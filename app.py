"""
Football Value Bet Dashboard — Streamlit app
Runs analyzer.py logic and displays results organised by league.
"""

import os
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
  .match-card.has-value { border-color: #16a34a; }

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
  .league-header {
    font-size: 1.1rem;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
    padding: 0.4rem 0;
  }
  .value-bet-row {
    background: #16a34a18;
    border: 1px solid #16a34a55;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin-top: 0.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .edge-badge {
    background: #16a34a;
    color: white;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.8rem;
    font-weight: 700;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ⚽ Football Value Bet Analyzer")
st.markdown(f"**{datetime.datetime.now().strftime('%A, %d %B %Y')}** — Today's matches with market edge")
st.divider()

# ── API keys ──────────────────────────────────────────────────────────────────
FD_KEY   = os.getenv("FOOTBALL_DATA_API_KEY", "YOUR_FOOTBALL_DATA_KEY")
ODDS_KEY = os.getenv("ODDS_API_KEY", "YOUR_ODDS_API_KEY")
DEMO     = FD_KEY == "YOUR_FOOTBALL_DATA_KEY" or ODDS_KEY == "YOUR_ODDS_API_KEY"

if DEMO:
    st.info("ℹ️ **Demo mode** — set env vars `FOOTBALL_DATA_API_KEY` and `ODDS_API_KEY` for live data.", icon="ℹ️")

# ── Fetch & analyze ───────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
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
    results_by_league: dict[str, list] = {}

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

        insufficient_data = (
            len(home_hist) < MIN_MATCHES_FOR_MODEL or
            len(away_hist) < MIN_MATCHES_FOR_MODEL
        )

        if insufficient_data:
            lam_h, lam_a, probs = None, None, None
            best, bets, has_odds = {}, [], False
        else:
            lam_h, lam_a = model.expected_goals(home_hist, away_hist, home_id, away_id)
            probs        = model.match_probabilities(lam_h, lam_a)

            matched_event = next(
                (ev for ev in all_odds
                 if teams_match(home_team, away_team,
                                ev.get("home_team", ""), ev.get("away_team", ""))),
                None
            )

            if matched_event:
                if DEMO:
                    best = {"home": 0.0, "draw": 0.0, "away": 0.0}
                    for bm in matched_event.get("bookmakers", []):
                        for mkt in bm.get("markets", []):
                            for o in mkt.get("outcomes", []):
                                n, p = o.get("name", ""), o.get("price", 0)
                                if n == home_team:   best["home"] = max(best["home"], p)
                                elif n == away_team: best["away"] = max(best["away"], p)
                                elif n == "Draw":    best["draw"] = max(best["draw"], p)
                else:
                    best = odds_client.best_odds(matched_event)
                bets     = detect_value_bets(probs, best)
                has_odds = True
            else:
                best, bets, has_odds = {}, [], False

        results_by_league.setdefault(comp, []).append({
            "home_team":        home_team,
            "away_team":        away_team,
            "competition":      comp,
            "kickoff":          kickoff,
            "lambda_home":      lam_h,
            "lambda_away":      lam_a,
            "model_probs":      probs,
            "best_odds":        best,
            "value_bets":       bets,
            "has_odds":         has_odds,
            "insufficient_data": insufficient_data,
        })

    return results_by_league

# ── Run ───────────────────────────────────────────────────────────────────────
with st.spinner("Analysing matches and searching for value bets…"):
    results_by_league = run_analysis()

if not results_by_league:
    st.warning("No matches scheduled today or insufficient historical data.")
    st.stop()

all_matches = [m for matches in results_by_league.values() for m in matches]

# ── KPI row ───────────────────────────────────────────────────────────────────
modelled      = [r for r in all_matches if not r["insufficient_data"]]
total_bets    = sum(len(r["value_bets"]) for r in modelled)
value_matches = sum(1 for r in modelled if r["value_bets"])
best_edge     = max((b["edge"] for r in modelled for b in r["value_bets"]), default=0)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Matches today", len(all_matches))
k2.metric("Fully analysed", len(modelled))
k3.metric("Value bets found", total_bets)
k4.metric("Best edge", f"+{best_edge:.1%}" if best_edge else "—")

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ℹ️ About the model")
    st.markdown(
        "Uses a **Poisson distribution** to estimate expected goals from each "
        "team's recent form. A *value bet* is flagged when the model probability "
        "exceeds the bookmaker's implied probability by at least 5%."
    )
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

# ── Top 3 value bets ─────────────────────────────────────────────────────────
all_bets_rows = []
for r in all_matches:
    for b in r["value_bets"]:
        all_bets_rows.append({
            "Match":     f"{r['home_team']} vs {r['away_team']}",
            "League":    r["competition"],
            "Time":      r["kickoff"],
            "Bet":       b["outcome"],
            "Odds":      f"{b['best_odds']:.2f}",
            "Model":     f"{b['model_prob']:.1%}",
            "Market":    f"{b['implied_prob']:.1%}",
            "Edge":      f"+{b['edge']:.1%}",
            "_edge_raw": b["edge"],
            "_home":     r["home_team"],
            "_away":     r["away_team"],
            "_outcome":  b["outcome"],
            "_odds":     b["best_odds"],
            "_model":    b["model_prob"],
            "_market":   b["implied_prob"],
            "_edge":     b["edge"],
        })

all_bets_rows.sort(key=lambda x: x["_edge_raw"], reverse=True)
top3 = all_bets_rows[:3]

if top3:
    st.markdown("### 🏆 Top 3 Value Bets")
    medals = ["🥇", "🥈", "🥉"]
    cols = st.columns(3)
    for i, (col, bet) in enumerate(zip(cols, top3)):
        with col:
            st.markdown(f"""
<div style="background:#0f172a;border:1px solid #16a34a;border-radius:12px;padding:1.1rem 1.2rem;">
  <div style="font-size:1.5rem;margin-bottom:4px;">{medals[i]}</div>
  <div style="font-size:0.75rem;color:#64748b;margin-bottom:6px;">{bet['League']} · {bet['Time']}</div>
  <div style="font-weight:700;font-size:0.95rem;margin-bottom:2px;">{bet['_home']} vs {bet['_away']}</div>
  <div style="color:#4ade80;font-weight:600;margin-bottom:10px;">🎯 {bet['_outcome']}</div>
  <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#94a3b8;">
    <span>Odds <strong style="color:#e2e8f0;">{bet['_odds']:.2f}</strong></span>
    <span>Model <strong style="color:#e2e8f0;">{bet['_model']:.1%}</strong></span>
    <span>Market <strong style="color:#e2e8f0;">{bet['_market']:.1%}</strong></span>
  </div>
  <div style="margin-top:10px;background:#16a34a;color:white;border-radius:6px;padding:5px 0;text-align:center;font-weight:700;font-size:1rem;">
    +{bet['_edge']:.1%} edge
  </div>
</div>
""", unsafe_allow_html=True)
    st.divider()

# ── Leagues ───────────────────────────────────────────────────────────────────
LEAGUE_FLAG = {
    "Premier League":   "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "La Liga":          "🇪🇸",
    "Serie A":          "🇮🇹",
    "Bundesliga":       "🇩🇪",
    "Ligue 1":          "🇫🇷",
    "Champions League": "🏆",
    "Europa League":    "🌍",
}

st.markdown("### 📋 Matches by League")

# Sort: leagues with value bets first
sorted_leagues = sorted(
    results_by_league.items(),
    key=lambda kv: -sum(len(m["value_bets"]) for m in kv[1])
)

for league_name, matches in sorted_leagues:
    flag = LEAGUE_FLAG.get(league_name, "⚽")
    league_value_count = sum(len(m["value_bets"]) for m in matches)
    badge = f"🎯 {league_value_count} value bet{'s' if league_value_count != 1 else ''}" if league_value_count else "No value bets"

    with st.expander(f"{flag} **{league_name}** — {len(matches)} match{'es' if len(matches) != 1 else ''}  ·  {badge}", expanded=league_value_count > 0):
        for r in matches:
            has_bets = bool(r["value_bets"])
            probs    = r["model_probs"]

            with st.container():
                has_bets = bool(r["value_bets"])
                probs    = r["model_probs"]

                # Match header
                col_title, col_badge = st.columns([5, 1])
                with col_title:
                    st.markdown(f"#### {r['home_team']} vs {r['away_team']}")
                    st.caption(f"🕐 {r['kickoff']}")
                with col_badge:
                    if r["insufficient_data"]:
                        st.markdown('<span class="no-value-pill">No data</span>', unsafe_allow_html=True)
                    elif has_bets:
                        st.markdown('<span class="value-pill">🎯 VALUE</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="no-value-pill">No edge</span>', unsafe_allow_html=True)

                if r["insufficient_data"]:
                    st.caption("⚠️ Not enough historical data to run the Poisson model for this match.")
                else:
                    # xG + probabilities
                    col_xg, col_probs = st.columns([2, 3])
                    with col_xg:
                        st.markdown("**Expected Goals**")
                        st.markdown(f"🏠 {r['home_team']}: **{r['lambda_home']}**")
                        st.markdown(f"✈️ {r['away_team']}: **{r['lambda_away']}**")
                    with col_probs:
                        st.markdown("**Model Probabilities**")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Home Win", f"{probs['home_win']:.1%}")
                        c2.metric("Draw",     f"{probs['draw']:.1%}")
                        c3.metric("Away Win", f"{probs['away_win']:.1%}")

                    # Odds row
                    if r["has_odds"] and r["best_odds"]:
                        st.markdown("**Best Available Odds**")
                        o1, o2, o3 = st.columns(3)
                        o1.metric("Home", f"{r['best_odds'].get('home', 0):.2f}")
                        o2.metric("Draw", f"{r['best_odds'].get('draw', 0):.2f}")
                        o3.metric("Away", f"{r['best_odds'].get('away', 0):.2f}")

                    # Value bets
                    if has_bets:
                        st.markdown("**Value Bets Detected**")
                        for b in r["value_bets"]:
                            b1, b2, b3, b4, b5 = st.columns([2, 1, 1, 1, 2])
                            b1.markdown(f"🎯 **{b['outcome']}**")
                            b2.markdown(f"Odds: **{b['best_odds']:.2f}**")
                            b3.markdown(f"Model: **{b['model_prob']:.1%}**")
                            b4.markdown(f"Market: **{b['implied_prob']:.1%}**")
                            b5.markdown(f"Edge: **+{b['edge']:.1%}**")
                            st.progress(min(int(b["edge"] * 100), 100))
                    elif not r["has_odds"]:
                        st.caption("⚠️ No odds available for this match.")

                st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────
st.caption("⚠️ For educational purposes only. Betting involves risk of loss. The Poisson model is a statistical simplification.")
